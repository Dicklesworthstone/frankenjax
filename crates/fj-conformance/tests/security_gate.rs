#![forbid(unsafe_code)]

use fj_conformance::e2e_log::{
    E2ECompatibilityMode, E2EForensicLogV1, E2ELogStatus, validate_e2e_log_path, write_e2e_log,
};
use fj_conformance::security_adversarial::{
    SECURITY_ADVERSARIAL_BEAD_ID, SECURITY_ADVERSARIAL_REPORT_SCHEMA_VERSION,
    SECURITY_THREAT_MODEL_SCHEMA_VERSION, SecurityAdversarialReport,
    build_security_adversarial_report, build_security_threat_model, security_adversarial_markdown,
    security_adversarial_summary_json, validate_security_adversarial_report,
    write_security_adversarial_outputs,
};
use std::collections::BTreeSet;
use std::path::Path;

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn security_gate_has_required_threat_categories() {
    let root = repo_root();
    let report = build_security_adversarial_report(&root);
    assert_eq!(
        report.schema_version,
        SECURITY_ADVERSARIAL_REPORT_SCHEMA_VERSION
    );
    assert_eq!(report.bead_id, SECURITY_ADVERSARIAL_BEAD_ID);
    assert_eq!(report.status, "pass", "report: {report:#?}");
    assert!(report.issues.is_empty(), "issues: {:#?}", report.issues);

    let category_ids = report
        .threat_categories
        .iter()
        .map(|category| category.category_id.as_str())
        .collect::<BTreeSet<_>>();
    for required in [
        "tc_cache_confusion",
        "tc_transform_ordering",
        "tc_ir_validation",
        "tc_shape_dtype_signatures",
        "tc_subjaxpr_control_flow",
        "tc_serialized_fixtures_logs",
        "tc_ffi_boundaries",
        "tc_durability_corruption",
        "tc_evidence_ledger_recovery",
    ] {
        assert!(category_ids.contains(required), "missing {required}");
    }
}

#[test]
fn security_gate_fuzz_families_are_complete_and_replayable() {
    let root = repo_root();
    let report = build_security_adversarial_report(&root);
    assert_eq!(
        report.coverage.fuzz_family_count,
        report.fuzz_families.len()
    );
    assert_eq!(
        report.coverage.fuzz_complete_count,
        report.fuzz_families.len()
    );
    assert_eq!(
        report.coverage.crash_free_family_count,
        report.fuzz_families.len()
    );
    assert_eq!(
        report.coverage.timeout_free_family_count,
        report.fuzz_families.len()
    );

    for family in &report.fuzz_families {
        assert!(
            family.observed_seed_count >= family.seed_floor,
            "family {family:#?}"
        );
        assert_eq!(
            family.observed_seed_count,
            family.deterministic_replay_count
        );
        assert_eq!(family.expected_error_class, family.actual_error_class);
        assert_eq!(family.panic_status, "no_panic");
        assert_eq!(family.crash_status, "no_crash");
        assert_eq!(family.timeout_status, "no_timeout");
        assert!(!family.replay_command.trim().is_empty());
        assert!(
            root.join(&family.target_source).exists(),
            "missing target {}",
            family.target_source
        );
        assert!(
            root.join(&family.corpus_path).exists(),
            "missing corpus {}",
            family.corpus_path
        );
    }
}

#[test]
fn security_gate_rows_are_typed_panic_free_and_evidence_linked() {
    let root = repo_root();
    let report = build_security_adversarial_report(&root);
    assert_eq!(
        report.coverage.typed_error_row_count,
        report.adversarial_rows.len()
    );
    assert_eq!(
        report.coverage.panic_free_row_count,
        report.adversarial_rows.len()
    );

    for row in &report.adversarial_rows {
        assert_eq!(row.expected_error_class, row.actual_error_class);
        assert_ne!(row.actual_error_class, "none");
        assert_eq!(row.panic_status, "no_panic");
        assert_eq!(row.crash_status, "no_crash");
        assert_eq!(row.timeout_status, "no_timeout");
        assert!(!row.evidence_refs.is_empty());
        assert!(!row.replay_command.trim().is_empty());
    }
}

#[test]
fn security_threat_model_is_complete_schema_shape() {
    let root = repo_root();
    let report = build_security_adversarial_report(&root);
    let model = build_security_threat_model(&report);
    assert_eq!(model.schema_version, SECURITY_THREAT_MODEL_SCHEMA_VERSION);
    assert_eq!(model.bead_id, SECURITY_ADVERSARIAL_BEAD_ID);
    assert_eq!(model.status, "complete");
    assert_eq!(
        model.summary.total_categories,
        model.threat_categories.len()
    );
    assert_eq!(
        model.summary.evidence_green,
        model.threat_categories.len(),
        "all threat model categories should be green"
    );
    assert_eq!(
        model.summary.fuzz_complete,
        model.threat_categories.len(),
        "all threat model categories should have complete fuzz coverage"
    );
    assert!(model.next_steps.is_empty());
}

#[test]
fn security_validation_rejects_incomplete_fuzz_and_missing_evidence() {
    let root = repo_root();
    let mut report = build_security_adversarial_report(&root);
    report.fuzz_families[0].observed_seed_count = 0;
    report.fuzz_families[0].deterministic_replay_count = 0;
    report.fuzz_families[0].actual_error_class = "missing_seed_corpus".to_owned();
    report.threat_categories[0].evidence_refs.clear();
    report.adversarial_rows[0].evidence_refs.clear();

    let issue_codes = validate_security_adversarial_report(&root, &report)
        .into_iter()
        .map(|issue| issue.code)
        .collect::<BTreeSet<_>>();
    assert!(issue_codes.contains("insufficient_seed_corpus"));
    assert!(issue_codes.contains("bad_fuzz_error_class"));
    assert!(issue_codes.contains("missing_category_evidence"));
    assert!(issue_codes.contains("missing_row_evidence"));
}

#[test]
fn security_validation_rejects_duplicate_ids_and_unbound_hashes() {
    let root = repo_root();
    let mut report = build_security_adversarial_report(&root);
    assert!(
        report.fuzz_families.len() > 3,
        "report needs multiple fuzz families for integrity regression"
    );
    assert!(
        !report.adversarial_rows.is_empty(),
        "report needs adversarial rows for replay regression"
    );

    report
        .threat_categories
        .push(report.threat_categories[0].clone());
    report.fuzz_families.push(report.fuzz_families[0].clone());
    report
        .adversarial_rows
        .push(report.adversarial_rows[0].clone());
    report.fuzz_families[0].artifact_hashes.clear();
    report.fuzz_families[1].artifact_hashes.insert(
        "crates/fj-conformance/fuzz/corpus/seed/cache_key_builder/seed_strict_unknown_feature.json"
            .to_owned(),
        "0".repeat(64),
    );
    report.fuzz_families[2].artifact_hashes.insert(
        "crates/fj-conformance/fuzz/corpus/seed/cache_key_builder/seed_strict_unknown_feature.json"
            .to_owned(),
        "not-a-sha256".to_owned(),
    );
    report.fuzz_families[3].artifact_hashes.insert(
        "crates/fj-conformance/fuzz/corpus/missing-security-seed".to_owned(),
        "0".repeat(64),
    );
    report.adversarial_rows[0].replay_command.clear();

    let issue_codes = validate_security_adversarial_report(&root, &report)
        .into_iter()
        .map(|issue| issue.code)
        .collect::<BTreeSet<_>>();
    assert!(issue_codes.contains("duplicate_category_id"));
    assert!(issue_codes.contains("duplicate_fuzz_family_id"));
    assert!(issue_codes.contains("duplicate_adversarial_row_id"));
    assert!(issue_codes.contains("missing_fuzz_artifact_hashes"));
    assert!(issue_codes.contains("stale_fuzz_artifact_hash"));
    assert!(issue_codes.contains("bad_fuzz_artifact_hash"));
    assert!(issue_codes.contains("missing_hashed_fuzz_artifact"));
    assert!(issue_codes.contains("missing_row_replay"));
}

#[test]
fn security_validation_rejects_report_contract_and_coverage_drift() {
    let root = repo_root();
    let mut report = build_security_adversarial_report(&root);
    report.status = "green".to_owned();
    report.matrix_policy.clear();
    report.threat_categories[0].category_id = "tc_untracked_boundary".to_owned();
    report.coverage.observed_category_count += 1;
    report.coverage.fuzz_family_count += 1;
    report.coverage.adversarial_row_count += 1;
    report.coverage.crash_free_family_count = 0;
    report.coverage.timeout_free_family_count = 0;
    report.coverage.evidence_ref_count += 1;

    let issue_codes = validate_security_adversarial_report(&root, &report)
        .into_iter()
        .map(|issue| issue.code)
        .collect::<BTreeSet<_>>();
    assert!(issue_codes.contains("bad_report_status"));
    assert!(issue_codes.contains("empty_matrix_policy"));
    assert!(issue_codes.contains("unknown_threat_category"));
    assert!(issue_codes.contains("missing_required_category"));
    assert!(issue_codes.contains("bad_observed_category_count"));
    assert!(issue_codes.contains("bad_fuzz_family_count"));
    assert!(issue_codes.contains("bad_adversarial_row_count"));
    assert!(issue_codes.contains("bad_crash_free_family_count"));
    assert!(issue_codes.contains("bad_timeout_free_family_count"));
    assert!(issue_codes.contains("bad_evidence_ref_count"));
}

#[test]
fn security_outputs_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    let root = repo_root();
    let temp = tempfile::tempdir()?;
    let threat_model_path = temp.path().join("security_threat_model.v1.json");
    let report_path = temp.path().join("security_adversarial_gate.v1.json");
    let markdown_path = temp.path().join("security_adversarial_gate.v1.md");
    let report = write_security_adversarial_outputs(
        &root,
        &threat_model_path,
        &report_path,
        &markdown_path,
    )?;

    let raw = std::fs::read_to_string(&report_path)?;
    let parsed: SecurityAdversarialReport = serde_json::from_str(&raw)?;
    assert_eq!(parsed.status, report.status);
    assert_eq!(parsed.fuzz_families.len(), report.fuzz_families.len());
    assert!(validate_security_adversarial_report(&root, &parsed).is_empty());

    let markdown = std::fs::read_to_string(&markdown_path)?;
    assert!(markdown.contains("Security Adversarial Gate"));
    assert!(markdown.contains("No security adversarial gate issues found."));
    Ok(())
}

#[test]
fn security_summary_is_dashboard_ready() {
    let root = repo_root();
    let report = build_security_adversarial_report(&root);
    let summary = security_adversarial_summary_json(&report);
    assert_eq!(summary["status"], "pass");
    assert_eq!(summary["category_count"], report.threat_categories.len());
    assert_eq!(summary["fuzz_family_count"], report.fuzz_families.len());
    assert_eq!(summary["issue_count"], 0);

    let markdown = security_adversarial_markdown(&report);
    assert!(markdown.contains("tc_cache_confusion"));
    assert!(markdown.contains("ff_transform_composition_verifier"));
}

#[test]
fn security_e2e_log_schema_accepts_gate_shape() -> Result<(), Box<dyn std::error::Error>> {
    let temp = tempfile::tempdir()?;
    let log_path = temp.path().join("security.e2e.json");
    let mut log = E2EForensicLogV1::new(
        SECURITY_ADVERSARIAL_BEAD_ID,
        "e2e_security_gate",
        "security_gate_test",
        vec!["security_gate_test".to_owned()],
        ".",
        E2ECompatibilityMode::Strict,
        E2ELogStatus::Pass,
    );
    log.fixture_ids = vec!["adv_cache_unknown_feature".to_owned()];
    log.oracle_ids = vec!["local:security-adversarial-gate.v1".to_owned()];
    log.transform_stack = vec!["cache".to_owned(), "jit".to_owned(), "ffi".to_owned()];
    log.replay_command = "./scripts/run_security_gate.sh --enforce".to_owned();
    write_e2e_log(&log_path, &log)?;
    let validation = validate_e2e_log_path(&log_path, Path::new("."));
    assert!(
        validation.is_ok(),
        "e2e log should validate: {validation:#?}"
    );
    Ok(())
}
