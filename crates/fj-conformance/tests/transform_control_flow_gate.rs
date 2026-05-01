#![forbid(unsafe_code)]

use fj_conformance::e2e_log::{
    E2ECompatibilityMode, E2EForensicLogV1, E2ELogStatus, validate_e2e_log_path, write_e2e_log,
};
use fj_conformance::transform_control_flow::{
    TRANSFORM_CONTROL_FLOW_BEAD_ID, TRANSFORM_CONTROL_FLOW_REPORT_SCHEMA_VERSION,
    TransformControlFlowReport, build_transform_control_flow_report,
    transform_control_flow_markdown, transform_control_flow_summary_json,
    validate_transform_control_flow_report, write_transform_control_flow_outputs,
};
use std::path::Path;

#[test]
fn transform_control_flow_matrix_has_required_rows() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report = build_transform_control_flow_report(temp.path());
    assert_eq!(
        report.schema_version,
        TRANSFORM_CONTROL_FLOW_REPORT_SCHEMA_VERSION
    );
    assert_eq!(report.bead_id, TRANSFORM_CONTROL_FLOW_BEAD_ID);
    assert_eq!(report.status, "pass", "report: {report:#?}");
    assert!(report.issues.is_empty(), "issues: {:#?}", report.issues);

    for required in [
        "jit_grad_cond_false",
        "vmap_grad_cond_mixed_predicates",
        "jit_vmap_grad_cond_false",
        "vmap_grad_scan_mul",
        "jit_vmap_grad_scan_mul",
        "vmap_grad_while_mul",
        "vmap_switch_batched_indices",
        "vmap_switch_scalar_index_batched_operand",
        "grad_grad_square",
        "vmap_grad_grad_square",
        "jit_vmap_grad_grad_square",
        "vmap_jit_grad_square",
        "value_and_grad_multi_output",
        "jacobian_quadratic",
        "hessian_quadratic",
        "scan_multi_carry_state",
        "vmap_multi_output_return",
        "jit_mixed_dtype_add",
        "grad_vmap_vector_output_fail_closed",
        "vmap_empty_batch_fail_closed",
        "vmap_out_axes_none_nonconstant_fail_closed",
    ] {
        assert!(
            report.cases.iter().any(|case| case.case_id == required),
            "missing required case {required}: {report:#?}"
        );
    }
}

#[test]
fn transform_control_flow_fail_closed_rows_are_typed() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report = build_transform_control_flow_report(temp.path());
    let fail_closed = report
        .cases
        .iter()
        .filter(|case| case.support_status == "fail_closed")
        .collect::<Vec<_>>();
    assert_eq!(fail_closed.len(), report.fail_closed_rows);
    assert!(
        fail_closed.len() >= 3,
        "expected multiple fail-closed rows: {report:#?}"
    );
    for case in fail_closed {
        assert_eq!(case.status, "pass", "case should pass: {case:#?}");
        assert!(
            case.error_class
                .as_deref()
                .is_some_and(|class| class.starts_with("transform_execution.")),
            "fail-closed case must record transform error class: {case:#?}"
        );
        assert!(
            !case.strict_hardened_rationale.trim().is_empty(),
            "fail-closed case must explain strict/hardened rationale: {case:#?}"
        );
    }
}

#[test]
fn transform_control_flow_performance_sentinels_record_latency_and_rss() -> Result<(), String> {
    let temp = tempfile::tempdir().expect("tempdir");
    let report = build_transform_control_flow_report(temp.path());
    for workload in [
        "perf_vmap_scan_loop_stack",
        "perf_vmap_while_loop_stack",
        "perf_jit_vmap_grad_cond",
        "perf_batched_switch",
    ] {
        let sentinel = report
            .performance_sentinels
            .iter()
            .find(|sentinel| sentinel.workload_id == workload)
            .ok_or_else(|| format!("missing performance sentinel {workload}"))?;
        assert_eq!(sentinel.status, "pass", "sentinel: {sentinel:#?}");
        assert!(sentinel.iterations > 0);
        assert!(sentinel.p50_ns > 0);
        assert!(sentinel.p95_ns >= sentinel.p50_ns);
        assert!(sentinel.p99_ns >= sentinel.p95_ns);
        assert_ne!(sentinel.measurement_backend, "unavailable");
        assert!(
            sentinel.peak_rss_bytes.is_some_and(|bytes| bytes > 0),
            "sentinel must record peak RSS: {sentinel:#?}"
        );
    }
    Ok(())
}

#[test]
fn transform_control_flow_outputs_round_trip() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report_path = temp.path().join("transform_control_flow_matrix.v1.json");
    let markdown_path = temp.path().join("transform_control_flow_matrix.v1.md");
    let report = write_transform_control_flow_outputs(temp.path(), &report_path, &markdown_path)
        .expect("outputs should write");

    let raw = std::fs::read_to_string(&report_path).expect("report should be readable");
    let parsed: TransformControlFlowReport =
        serde_json::from_str(&raw).expect("report JSON should parse");
    assert_eq!(parsed.status, report.status);
    assert_eq!(parsed.cases.len(), report.cases.len());
    assert!(validate_transform_control_flow_report(&parsed).is_empty());

    let markdown = std::fs::read_to_string(&markdown_path).expect("markdown should be readable");
    assert!(markdown.contains("Transform Control-Flow Matrix Gate"));
    assert!(markdown.contains("No transform control-flow matrix issues found."));
}

#[test]
fn transform_control_flow_summary_is_dashboard_ready() {
    let temp = tempfile::tempdir().expect("tempdir");
    let report = build_transform_control_flow_report(temp.path());
    let summary = transform_control_flow_summary_json(&report);
    assert_eq!(summary["status"], "pass");
    assert_eq!(summary["case_count"], report.cases.len());
    assert_eq!(summary["issue_count"], 0);
    assert_eq!(
        summary["performance_sentinel_count"],
        report.performance_sentinels.len()
    );

    let markdown = transform_control_flow_markdown(&report);
    assert!(markdown.contains("jit_vmap_grad_cond_false"));
    assert!(markdown.contains("grad_vmap_vector_output_fail_closed"));
}

#[test]
fn transform_control_flow_e2e_log_schema_accepts_gate_shape() {
    let temp = tempfile::tempdir().expect("tempdir");
    let log_path = temp.path().join("transform_control_flow.e2e.json");
    let mut log = E2EForensicLogV1::new(
        TRANSFORM_CONTROL_FLOW_BEAD_ID,
        "e2e_transform_control_flow_gate",
        "transform_control_flow_gate_test",
        vec!["transform_control_flow_gate_test".to_owned()],
        ".",
        E2ECompatibilityMode::Strict,
        E2ELogStatus::Pass,
    );
    log.fixture_ids = vec!["jit_grad_cond_false".to_owned()];
    log.oracle_ids = vec!["analytic:jax.grad(cond)".to_owned()];
    log.transform_stack = vec!["jit".to_owned(), "grad".to_owned(), "cond".to_owned()];
    log.replay_command = "./scripts/run_transform_control_flow_gate.sh --enforce".to_owned();
    write_e2e_log(&log_path, &log).expect("write e2e log");
    validate_e2e_log_path(&log_path, Path::new(".")).expect("e2e log should validate");
}
