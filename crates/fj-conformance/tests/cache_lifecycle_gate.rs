#![forbid(unsafe_code)]

use fj_cache::legacy_parity::{
    CacheParityStatus, CacheParitySurface, cache_legacy_parity_ledger,
    validate_cache_legacy_parity_ledger,
};
use fj_conformance::cache_lifecycle::{
    CACHE_LIFECYCLE_REPORT_SCHEMA_VERSION, build_cache_lifecycle_report,
    cache_lifecycle_summary_json, write_cache_lifecycle_outputs,
};
use fj_conformance::e2e_log::{
    E2ECompatibilityMode, E2EForensicLogV1, E2ELogStatus, validate_e2e_log_path, write_e2e_log,
};
use std::path::Path;

#[test]
fn cache_lifecycle_report_has_required_scenarios() {
    let report = build_cache_lifecycle_report(Path::new("."));
    assert_eq!(report.schema_version, CACHE_LIFECYCLE_REPORT_SCHEMA_VERSION);
    assert_eq!(report.status, "pass", "report: {report:#?}");
    for required in [
        "compile_option_ordering",
        "transform_order_differentiation",
        "strict_unknown_metadata_rejection",
        "hardened_unknown_metadata_inclusion",
        "corrupt_read_bypass",
        "failed_write_stays_miss",
    ] {
        assert!(
            report
                .scenarios
                .iter()
                .any(|scenario| scenario.scenario_id == required),
            "missing scenario {required}: {report:#?}"
        );
    }
}

#[test]
fn cache_legacy_parity_ledger_validates_and_names_exclusions() {
    let ledger = cache_legacy_parity_ledger();
    let issues = validate_cache_legacy_parity_ledger(&ledger);
    assert!(issues.is_empty(), "ledger issues: {issues:?}");
    assert!(ledger.rows.iter().any(|row| {
        row.surface == CacheParitySurface::CompilationCacheMetadata
            && row.status == CacheParityStatus::ExplicitExclusion
            && row
                .exclusion_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("GCS"))
    }));
}

#[test]
fn cache_lifecycle_summary_is_dashboard_ready() {
    let ledger = cache_legacy_parity_ledger();
    let report = build_cache_lifecycle_report(Path::new("."));
    let summary = cache_lifecycle_summary_json(&ledger, &report);
    assert_eq!(summary["gate_status"], "pass");
    assert_eq!(summary["ledger_rows"], ledger.rows.len());
    assert_eq!(summary["scenario_count"], report.scenarios.len());
}

#[test]
fn cache_lifecycle_outputs_round_trip() {
    let temp = tempfile::tempdir().expect("tempdir");
    let ledger_path = temp.path().join("ledger.json");
    let report_path = temp.path().join("report.json");
    let markdown_path = temp.path().join("ledger.md");
    let (ledger, report) =
        write_cache_lifecycle_outputs(Path::new("."), &ledger_path, &report_path, &markdown_path)
            .expect("write outputs");

    assert_eq!(ledger.rows.len(), cache_legacy_parity_ledger().rows.len());
    assert_eq!(report.status, "pass");
    assert!(ledger_path.exists());
    assert!(report_path.exists());
    assert!(markdown_path.exists());
}

#[test]
fn cache_lifecycle_e2e_log_schema_accepts_gate_shape() {
    let temp = tempfile::tempdir().expect("tempdir");
    let log_path = temp.path().join("cache.e2e.json");
    let mut log = E2EForensicLogV1::new(
        "frankenjax-cstq.6",
        "e2e_cache_lifecycle_gate",
        "cache_lifecycle_gate_test",
        vec!["cache_lifecycle_gate_test".to_owned()],
        ".",
        E2ECompatibilityMode::Strict,
        E2ELogStatus::Pass,
    );
    log.fixture_ids = vec!["compile_option_ordering".to_owned()];
    log.oracle_ids = vec!["P2C005-A01".to_owned()];
    log.replay_command = "./scripts/run_cache_lifecycle_gate.sh --enforce".to_owned();
    write_e2e_log(&log_path, &log).expect("write e2e log");
    validate_e2e_log_path(&log_path, Path::new(".")).expect("e2e log should validate");
}
