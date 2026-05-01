#![forbid(unsafe_code)]

use fj_conformance::e2e_log::{
    E2E_FORENSIC_LOG_SCHEMA_VERSION, E2ECompatibilityMode, E2EForensicLogV1, E2ELogStatus,
    artifact_sha256_hex, classify_process_status, validate_e2e_log_path, validate_e2e_log_value,
    validation_report_for_paths, write_e2e_log,
};
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn write_artifact(root: &Path, rel_path: &str, content: &str) -> String {
    let path = root.join(rel_path);
    fs::create_dir_all(path.parent().expect("artifact parent should exist"))
        .expect("artifact parent should be creatable");
    fs::write(&path, content).expect("artifact should be writable");
    artifact_sha256_hex(&path).expect("artifact hash should compute")
}

fn valid_log_value(root: &Path) -> Value {
    let artifact_path = "artifacts/e2e/contract-test.stdout.log";
    let artifact_hash = write_artifact(root, artifact_path, "contract test stdout\n");
    json!({
        "schema_version": E2E_FORENSIC_LOG_SCHEMA_VERSION,
        "bead_id": "frankenjax-cstq.16",
        "scenario_id": "e2e_forensic_log_contract_unit",
        "test_id": "e2e_forensic_log_contract::valid_log_value",
        "packet_id": "FJ-CSTQ-016",
        "command": ["cargo", "test", "-p", "fj-conformance"],
        "working_dir": root.display().to_string(),
        "environment": {
            "os": "linux",
            "arch": "x86_64",
            "rust_version": "rustc 1.0.0-test",
            "cargo_version": "cargo 1.0.0-test",
            "cargo_target_dir": "/tmp/fj-target",
            "env_vars": {
                "RUSTUP_TOOLCHAIN": "nightly",
                "API_TOKEN": "[REDACTED]"
            },
            "timestamp_unix_ms": 1
        },
        "feature_flags": ["default"],
        "fixture_ids": ["fixture:contract-unit"],
        "oracle_ids": ["oracle:none"],
        "transform_stack": ["jit", "grad"],
        "mode": "strict",
        "inputs": {"x": 3.0},
        "expected": {"value": 9.0},
        "actual": {"value": 9.0},
        "tolerance": {
            "policy_id": "f64-default",
            "atol": 1e-12,
            "rtol": 1e-12,
            "ulp": null,
            "notes": "exact for this fixture"
        },
        "error": {
            "expected": null,
            "actual": null,
            "taxonomy_class": "none"
        },
        "timings": {
            "setup_ms": 1,
            "trace_ms": 2,
            "dispatch_ms": 3,
            "eval_ms": 4,
            "verify_ms": 5,
            "total_ms": 15
        },
        "allocations": {
            "allocation_count": 10,
            "allocated_bytes": 2048,
            "peak_rss_bytes": null,
            "measurement_backend": "test-double-free"
        },
        "artifacts": [
            {
                "kind": "stdout_log",
                "path": artifact_path,
                "sha256": artifact_hash,
                "required": true
            }
        ],
        "replay_command": "cargo test -p fj-conformance --test e2e_forensic_log_contract -- valid_log_accepts_unknown_fields --exact --nocapture",
        "status": "pass",
        "failure_summary": null,
        "redactions": [
            {"path": "$.environment.env_vars.API_TOKEN", "reason": "secret-like env var"}
        ],
        "metadata": {
            "dashboard_lane": "e2e_logging"
        }
    })
}

fn issue_codes(
    result: Result<E2EForensicLogV1, Vec<fj_conformance::e2e_log::E2ELogValidationIssue>>,
) -> Vec<String> {
    result
        .expect_err("validation should fail")
        .into_iter()
        .map(|issue| issue.code)
        .collect()
}

#[test]
fn valid_log_accepts_unknown_fields() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut value = valid_log_value(tmp.path());
    value.as_object_mut().expect("object").insert(
        "future_dashboard_field".to_owned(),
        json!({"accepted": true}),
    );

    let parsed = validate_e2e_log_value(&value, tmp.path()).expect("valid log");
    assert_eq!(parsed.schema_version, E2E_FORENSIC_LOG_SCHEMA_VERSION);
    assert_eq!(parsed.status, E2ELogStatus::Pass);
}

#[test]
fn missing_required_field_is_rejected() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut value = valid_log_value(tmp.path());
    value
        .as_object_mut()
        .expect("object")
        .remove("replay_command");

    let codes = issue_codes(validate_e2e_log_value(&value, tmp.path()));
    assert!(codes.contains(&"missing_required_field".to_owned()));
}

#[test]
fn malformed_status_is_rejected() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut value = valid_log_value(tmp.path());
    value["status"] = json!("maybe");

    let codes = issue_codes(validate_e2e_log_value(&value, tmp.path()));
    assert!(codes.contains(&"malformed_log".to_owned()));
}

#[test]
fn stale_artifact_hash_is_rejected() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut value = valid_log_value(tmp.path());
    value["artifacts"][0]["sha256"] =
        json!("0000000000000000000000000000000000000000000000000000000000000000");

    let codes = issue_codes(validate_e2e_log_value(&value, tmp.path()));
    assert!(codes.contains(&"stale_artifact_hash".to_owned()));
}

#[test]
fn unredacted_sensitive_environment_value_is_rejected() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut value = valid_log_value(tmp.path());
    value["environment"]["env_vars"]["API_TOKEN"] = json!("live-token-value");

    let codes = issue_codes(validate_e2e_log_value(&value, tmp.path()));
    assert!(codes.contains(&"unredacted_sensitive_value".to_owned()));
}

#[test]
fn redacted_replay_command_is_rejected() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut value = valid_log_value(tmp.path());
    value["replay_command"] = json!("[REDACTED]");

    let codes = issue_codes(validate_e2e_log_value(&value, tmp.path()));
    assert!(codes.contains(&"redacted_replay_command".to_owned()));
}

#[test]
fn failing_status_requires_failure_summary() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut value = valid_log_value(tmp.path());
    value["status"] = json!("fail");
    value["failure_summary"] = Value::Null;

    let codes = issue_codes(validate_e2e_log_value(&value, tmp.path()));
    assert!(codes.contains(&"missing_failure_summary".to_owned()));
}

#[test]
fn process_status_classifier_is_stable() {
    assert_eq!(
        classify_process_status(Some(0), false, false),
        E2ELogStatus::Pass
    );
    assert_eq!(
        classify_process_status(Some(101), false, false),
        E2ELogStatus::Fail
    );
    assert_eq!(
        classify_process_status(None, true, false),
        E2ELogStatus::Timeout
    );
    assert_eq!(
        classify_process_status(None, false, true),
        E2ELogStatus::Crash
    );
    assert_eq!(
        classify_process_status(None, false, false),
        E2ELogStatus::Error
    );
}

#[test]
fn report_rejects_nonconforming_logs_with_reasons() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let good_path = tmp.path().join("good.e2e.json");
    let bad_path = tmp.path().join("bad.e2e.json");

    let good: E2EForensicLogV1 =
        serde_json::from_value(valid_log_value(tmp.path())).expect("valid fixture");
    write_e2e_log(&good_path, &good).expect("write good log");
    fs::write(&bad_path, "{\"schema_version\":\"wrong\"}\n").expect("write bad log");

    let report = validation_report_for_paths(&[good_path, bad_path], tmp.path());
    assert_eq!(report.checked, 2);
    assert_eq!(report.passed, 1);
    assert_eq!(report.failed, 1);
    assert!(
        report
            .logs
            .iter()
            .flat_map(|log| log.issues.iter())
            .any(|issue| issue.code == "missing_required_field")
    );
}

#[test]
fn report_rejects_empty_log_sets() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let report = validation_report_for_paths(&[], tmp.path());

    assert_eq!(report.status, "fail");
    assert_eq!(report.checked, 0);
    assert_eq!(report.failed, 1);
    assert!(
        report
            .logs
            .iter()
            .flat_map(|log| log.issues.iter())
            .any(|issue| issue.code == "no_logs_found")
    );
}

#[test]
fn builder_outputs_serializable_contract_logs() {
    let mut log = E2EForensicLogV1::new(
        "frankenjax-cstq.16",
        "builder_contract",
        "e2e_forensic_log_contract::builder_outputs_serializable_contract_logs",
        vec!["cargo".to_owned(), "test".to_owned()],
        "/tmp/frankenjax",
        E2ECompatibilityMode::Hardened,
        E2ELogStatus::Skip,
    );
    log.replay_command = "cargo test -p fj-conformance builder_contract".to_owned();
    log.metadata
        .insert("reason".to_owned(), json!("construction smoke"));

    let encoded = serde_json::to_string(&log).expect("serialize");
    assert!(encoded.contains(E2E_FORENSIC_LOG_SCHEMA_VERSION));
}

#[test]
fn bootstrap_e2e_forensic_log_sample_validates() {
    let root = repo_root();
    let sample = root.join("artifacts/e2e/e2e_forensic_log_contract_bootstrap.e2e.json");
    let parsed = validate_e2e_log_path(&sample, &root).expect("bootstrap sample log validates");
    assert_eq!(parsed.bead_id, "frankenjax-cstq.16");
    assert_eq!(parsed.scenario_id, "e2e_forensic_log_contract_bootstrap");
}
