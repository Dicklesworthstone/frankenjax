#![forbid(unsafe_code)]

use fj_conformance::oracle_recapture::{
    EXPECTED_ORACLE_CASE_TOTAL, LegacyAnchor, ORACLE_DRIFT_REPORT_SCHEMA_VERSION,
    ORACLE_RECAPTURE_MATRIX_SCHEMA_VERSION, OracleFixtureSpec, build_oracle_recapture_matrix,
    build_oracle_recapture_matrix_from_specs, oracle_drift_report, validate_oracle_drift_report,
    validate_oracle_recapture_matrix,
};
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn spec_for(rel_path: &str, expected_count: usize) -> OracleFixtureSpec {
    OracleFixtureSpec {
        family_id: "unit_family".to_owned(),
        display_name: "Unit family".to_owned(),
        fixture_path: rel_path.to_owned(),
        expected_schema_version: "frankenjax.unit-fixtures.v1".to_owned(),
        expected_case_count: expected_count,
        expected_oracle_version_prefix: "0.9.2".to_owned(),
        expected_x64_enabled: Some(true),
        legacy_anchors: vec![LegacyAnchor {
            path: "tests/unit_test.py".to_owned(),
            symbol: "UnitTest".to_owned(),
            required: true,
        }],
        recapture_command: vec!["python3".to_owned(), "capture.py".to_owned()],
    }
}

fn write_fixture(root: &Path, rel_path: &str, value: serde_json::Value) {
    let path = root.join(rel_path);
    fs::create_dir_all(path.parent().expect("fixture parent should exist"))
        .expect("fixture parent should be creatable");
    fs::write(
        path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&value).expect("fixture should serialize")
        ),
    )
    .expect("fixture should be writable");
}

#[test]
fn committed_matrix_counts_all_848_oracle_cases() {
    let root = repo_root();
    let matrix = build_oracle_recapture_matrix(&root);

    assert_eq!(
        matrix.schema_version,
        ORACLE_RECAPTURE_MATRIX_SCHEMA_VERSION
    );
    assert_eq!(matrix.expected_total_cases, EXPECTED_ORACLE_CASE_TOTAL);
    assert_eq!(matrix.actual_total_cases, EXPECTED_ORACLE_CASE_TOTAL);
    assert_eq!(matrix.rows.len(), 5);
    assert_eq!(
        matrix
            .rows
            .iter()
            .find(|row| row.family_id == "transforms")
            .expect("transforms row")
            .actual_case_count,
        613
    );
}

#[test]
fn committed_matrix_flags_stale_oracle_versions() {
    let root = repo_root();
    let matrix = build_oracle_recapture_matrix(&root);
    let codes: Vec<(&str, &str)> = matrix
        .issues
        .iter()
        .map(|issue| {
            (
                issue.family_id.as_deref().unwrap_or("-"),
                issue.code.as_str(),
            )
        })
        .collect();

    assert!(codes.contains(&("composition", "stale_oracle_version")));
    assert!(codes.contains(&("dtype_promotion", "stale_oracle_version")));
    assert!(
        !codes.contains(&("composition", "missing_recapture_command")),
        "composition should have recapture command"
    );
    assert!(
        !codes.contains(&("dtype_promotion", "missing_recapture_command")),
        "dtype_promotion should have recapture command"
    );
}

#[test]
fn missing_required_family_is_rejected() {
    let root = repo_root();
    let mut matrix = build_oracle_recapture_matrix(&root);
    matrix.rows.retain(|row| row.family_id != "rng");

    let codes: Vec<String> = validate_oracle_recapture_matrix(&matrix)
        .into_iter()
        .map(|issue| issue.code)
        .collect();
    assert!(codes.contains(&"missing_family".to_owned()));
}

#[test]
fn duplicate_case_ids_are_rejected() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let rel = "fixtures/unit.json";
    write_fixture(
        tmp.path(),
        rel,
        json!({
            "schema_version": "frankenjax.unit-fixtures.v1",
            "metadata": {"jax_version": "0.9.2", "x64_enabled": true},
            "generated_by": "unit",
            "cases": [
                {"case_id": "dupe"},
                {"case_id": "dupe"}
            ]
        }),
    );

    let matrix = build_oracle_recapture_matrix_from_specs(tmp.path(), &[spec_for(rel, 2)]);
    let row = matrix.rows.first().expect("one row");
    assert_eq!(row.duplicate_case_ids, vec!["dupe"]);
    assert!(row.issue_codes.contains(&"duplicate_case_ids".to_owned()));
}

#[test]
fn drift_report_detects_changed_fixture_hashes() {
    let root = repo_root();
    let current = build_oracle_recapture_matrix(&root);
    let mut baseline = current.clone();
    baseline
        .rows
        .iter_mut()
        .find(|row| row.family_id == "transforms")
        .expect("transforms baseline row")
        .fixture_sha256 =
        "0000000000000000000000000000000000000000000000000000000000000000".to_owned();

    let report = oracle_drift_report(
        &current,
        Some(&baseline),
        Some("baseline.json".to_owned()),
        true,
    );

    assert!(report.changed.contains(&"transforms".to_owned()));
    assert_eq!(report.gate_status, "fail");
}

#[test]
fn drift_report_schema_round_trips_and_validates() {
    let root = repo_root();
    let current = build_oracle_recapture_matrix(&root);
    let report = oracle_drift_report(&current, None, None, false);
    let encoded = serde_json::to_string(&report).expect("report should serialize");
    let decoded: fj_conformance::oracle_recapture::OracleDriftReport =
        serde_json::from_str(&encoded).expect("report should deserialize");

    assert_eq!(decoded.schema_version, ORACLE_DRIFT_REPORT_SCHEMA_VERSION);
    assert!(validate_oracle_drift_report(&decoded).len() >= decoded.issues.len());
}
