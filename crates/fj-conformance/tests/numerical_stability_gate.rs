#![forbid(unsafe_code)]

use fj_conformance::e2e_log::{E2ELogStatus, validate_e2e_log_path};
use fj_conformance::numerical_stability::{
    NUMERICAL_STABILITY_BEAD_ID, NUMERICAL_STABILITY_REPORT_SCHEMA_VERSION,
    build_numerical_stability_report, classify_non_finite_f64, f64_ulp_distance,
    finite_difference_step, literal_f64_bits_roundtrip, numerical_stability_markdown,
    validate_numerical_stability_report, write_numerical_stability_outputs,
};
use std::collections::BTreeSet;

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn numerical_stability_report_covers_required_families() {
    let root = repo_root();
    let report = build_numerical_stability_report(&root);
    assert_eq!(
        report.schema_version,
        NUMERICAL_STABILITY_REPORT_SCHEMA_VERSION
    );
    assert_eq!(report.bead_id, NUMERICAL_STABILITY_BEAD_ID);
    assert_eq!(report.status, "pass", "issues: {:#?}", report.issues);
    assert!(report.issues.is_empty());
    assert_eq!(report.summary.required_family_count, 11);
    assert_eq!(report.summary.observed_family_count, 11);
    assert_eq!(report.summary.stale_count, 0);
    assert_eq!(report.summary.regression_count, 0);
    assert_eq!(report.summary.rows_with_platform, report.summary.row_count);

    let families = report
        .rows
        .iter()
        .map(|row| row.family.as_str())
        .collect::<BTreeSet<_>>();
    for required in [
        "special_math_tails",
        "linalg_near_singular",
        "ad_gradient_check",
        "fft_scaling",
        "dtype_promotion",
        "complex_branch_values",
        "rng_determinism",
        "literal_bit_roundtrip",
        "non_finite_division",
        "finite_diff_policy",
        "platform_metadata",
    ] {
        assert!(families.contains(required), "missing {required}");
    }
}

#[test]
fn numerical_stability_validation_rejects_bad_rows() {
    let root = repo_root();
    let mut report = build_numerical_stability_report(&root);
    assert!(
        report.rows.len() > 1,
        "report must contain at least two rows"
    );
    let second_id = report.rows[1].case_id.clone();
    let first = &mut report.rows[0];
    first.case_id = second_id;
    first.tolerance_policy_id = "missing_policy".to_owned();
    first.platform_fingerprint_id = "missing_platform".to_owned();
    first.non_finite_classification = "unclassified".to_owned();
    first.deterministic_replay_count = 1;

    let issue_codes = validate_numerical_stability_report(&root, &report)
        .into_iter()
        .map(|issue| issue.code)
        .collect::<BTreeSet<_>>();
    assert!(issue_codes.contains("duplicate_case_id"));
    assert!(issue_codes.contains("missing_tolerance_policy"));
    assert!(issue_codes.contains("missing_platform_fingerprint"));
    assert!(issue_codes.contains("missing_non_finite_classification"));
    assert!(issue_codes.contains("insufficient_replay_count"));
}

#[test]
fn numerical_stability_validation_rejects_threshold_and_redaction_drift() {
    let root = repo_root();
    let mut report = build_numerical_stability_report(&root);
    assert!(report.rows.len() > 1, "report must contain linalg row");
    let row = &mut report.rows[1];
    row.max_abs_error = 9.0;
    row.reference_source = "contains SECRET material".to_owned();
    row.status = "fail".to_owned();
    row.stale = true;
    row.regression = true;

    let issue_codes = validate_numerical_stability_report(&root, &report)
        .into_iter()
        .map(|issue| issue.code)
        .collect::<BTreeSet<_>>();
    assert!(issue_codes.contains("tolerance_exceeded"));
    assert!(issue_codes.contains("redaction_policy_violation"));
    assert!(issue_codes.contains("non_pass_row"));
    assert!(issue_codes.contains("stale_row"));
    assert!(issue_codes.contains("regression_row"));
}

#[test]
fn numerical_stability_helpers_cover_non_finite_bits_and_fd_policy() {
    let nan_bits = 0x7ff8_0000_0000_1234;
    let nan = f64::from_bits(nan_bits);
    assert_eq!(
        classify_non_finite_f64(nan),
        "nan_payload_preserved:0008000000001234"
    );
    assert_eq!(
        classify_non_finite_f64(f64::INFINITY),
        "positive_infinity_preserved"
    );
    assert_eq!(
        classify_non_finite_f64(f64::NEG_INFINITY),
        "negative_infinity_preserved"
    );
    assert_eq!(
        f64_ulp_distance(1.0, f64::from_bits(1.0f64.to_bits() + 1)),
        Some(1)
    );
    assert_eq!(f64_ulp_distance(nan, 1.0), None);
    assert!(literal_f64_bits_roundtrip(nan_bits));

    let unit_step = finite_difference_step(1.0);
    let large_step = finite_difference_step(1.0e9);
    assert!((1e-8..=1e-3).contains(&unit_step));
    assert_eq!(large_step, 1e-3);
}

#[test]
fn numerical_stability_outputs_write_and_validate_e2e_log() -> Result<(), String> {
    let root = repo_root();
    let temp = tempfile::tempdir().map_err(|err| err.to_string())?;
    let report_path = temp.path().join("numerical_stability_matrix.v1.json");
    let markdown_path = temp.path().join("numerical_stability_matrix.v1.md");
    let e2e_path = temp.path().join("e2e_numerical_stability_gate.e2e.json");
    let report = write_numerical_stability_outputs(&root, &report_path, &markdown_path)
        .map_err(|err| err.to_string())?;
    assert_eq!(report.status, "pass");
    assert!(report_path.exists());
    assert!(markdown_path.exists());

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_fj_numerical_stability_gate"))
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
    assert_eq!(parsed.bead_id, NUMERICAL_STABILITY_BEAD_ID);
    Ok(())
}

#[test]
fn numerical_stability_markdown_is_dashboard_ready() {
    let root = repo_root();
    let report = build_numerical_stability_report(&root);
    let markdown = numerical_stability_markdown(&report);
    assert!(markdown.contains("Numerical Stability Matrix"));
    assert!(markdown.contains("special_math_tails"));
    assert!(markdown.contains("No numerical-stability issues found."));
}
