#![forbid(unsafe_code)]

use fj_conformance::e2e_log::{
    E2ECompatibilityMode, E2EForensicLogV1, E2ELogStatus, validate_e2e_log_path, write_e2e_log,
};
use fj_conformance::memory_performance::{
    DEFAULT_PEAK_RSS_BUDGET_BYTES, MEMORY_PERFORMANCE_REPORT_SCHEMA_VERSION,
    build_memory_performance_report, memory_performance_markdown, memory_performance_summary_json,
    validate_memory_performance_report, write_memory_performance_outputs,
};
use std::path::Path;

#[test]
fn memory_performance_report_has_required_workloads() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report = build_memory_performance_report(temp.path());
    assert_eq!(
        report.schema_version,
        MEMORY_PERFORMANCE_REPORT_SCHEMA_VERSION
    );
    assert_eq!(report.bead_id, "frankenjax-cstq.4");
    assert_eq!(report.status, "pass", "report: {report:#?}");
    assert!(report.issues.is_empty(), "issues: {:#?}", report.issues);
    for required in [
        "trace_canonical_fingerprint",
        "dispatch_jit_scalar",
        "ad_grad_scalar",
        "vmap_vector_add_one",
        "fft_complex_vector",
        "linalg_cholesky_matrix",
        "cache_hit_miss",
        "durability_sidecar_round_trip",
    ] {
        assert!(
            report
                .workloads
                .iter()
                .any(|workload| workload.workload_id == required),
            "missing workload {required}: {report:#?}"
        );
    }
}

#[test]
fn memory_performance_rows_are_measured_and_budgeted() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report = build_memory_performance_report(temp.path());
    for workload in &report.workloads {
        assert_eq!(workload.status, "pass", "workload: {workload:#?}");
        assert_ne!(workload.measurement_backend, "not_measured");
        assert_ne!(workload.measurement_backend, "unavailable");
        assert!(
            workload.peak_rss_bytes.is_some_and(|value| value > 0),
            "workload must record peak RSS: {workload:#?}"
        );
        assert_eq!(
            workload.peak_rss_budget_bytes,
            DEFAULT_PEAK_RSS_BUDGET_BYTES
        );
        assert!(
            workload.logical_output_units > 0,
            "workload must include a behavior witness: {workload:#?}"
        );
        assert!(
            !workload.evidence_refs.is_empty(),
            "workload must include evidence refs: {workload:#?}"
        );
    }
    assert!(validate_memory_performance_report(&report).is_empty());
}

#[test]
fn memory_performance_outputs_round_trip() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report_path = temp.path().join("memory_performance_gate.v1.json");
    let markdown_path = temp.path().join("memory_performance_gate.v1.md");
    let report = write_memory_performance_outputs(temp.path(), &report_path, &markdown_path)
        .expect("outputs should write");

    let raw = std::fs::read_to_string(&report_path).expect("report should be readable");
    let parsed: fj_conformance::memory_performance::MemoryPerformanceReport =
        serde_json::from_str(&raw).expect("report JSON should parse");
    assert_eq!(parsed.status, report.status);
    assert_eq!(parsed.workloads.len(), report.workloads.len());

    let markdown = std::fs::read_to_string(&markdown_path).expect("markdown should be readable");
    assert!(markdown.contains("Memory Performance Gate"));
    assert!(markdown.contains("No memory performance issues found."));
}

#[test]
fn memory_performance_summary_is_dashboard_ready() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report = build_memory_performance_report(temp.path());
    let summary = memory_performance_summary_json(&report);
    assert_eq!(summary["status"], "pass");
    assert_eq!(summary["workload_count"], report.workloads.len());
    assert_eq!(summary["issue_count"], 0);
    assert!(
        summary["max_peak_rss_bytes"]
            .as_u64()
            .is_some_and(|value| value > 0)
    );
    let markdown = memory_performance_markdown(&report);
    assert!(markdown.contains("trace_canonical_fingerprint"));
    assert!(markdown.contains("durability_sidecar_round_trip"));
}

#[test]
fn memory_performance_e2e_log_schema_accepts_gate_shape() {
    let temp = tempfile::tempdir().expect("tempdir");
    let log_path = temp.path().join("memory.e2e.json");
    let mut log = E2EForensicLogV1::new(
        "frankenjax-cstq.4",
        "e2e_memory_performance_gate",
        "memory_performance_gate_test",
        vec!["memory_performance_gate_test".to_owned()],
        ".",
        E2ECompatibilityMode::Strict,
        E2ELogStatus::Pass,
    );
    log.fixture_ids = vec!["trace_canonical_fingerprint".to_owned()];
    log.oracle_ids = vec!["linux:/proc/self/status".to_owned()];
    log.replay_command = "./scripts/run_memory_performance_gate.sh --enforce".to_owned();
    write_e2e_log(&log_path, &log).expect("write e2e log");
    validate_e2e_log_path(&log_path, Path::new(".")).expect("e2e log should validate");
}
