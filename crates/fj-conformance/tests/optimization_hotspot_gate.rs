#![forbid(unsafe_code)]

use fj_conformance::e2e_log::{E2EForensicLogV1, validate_e2e_log_path};
use fj_conformance::optimization_hotspots::{
    HOTSPOT_FOLLOW_UP_THRESHOLD, OPTIMIZATION_HOTSPOT_BEAD_ID, OPTIMIZATION_HOTSPOT_SCHEMA_VERSION,
    build_optimization_hotspot_report, optimization_hotspot_markdown,
    optimization_hotspot_summary_json, validate_optimization_hotspot_report,
    write_optimization_hotspot_outputs,
};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn issue_codes(
    report: &fj_conformance::optimization_hotspots::OptimizationHotspotReport,
) -> BTreeSet<String> {
    validate_optimization_hotspot_report(&repo_root(), report)
        .into_iter()
        .map(|issue| issue.code)
        .collect()
}

#[test]
fn optimization_hotspot_report_covers_required_families() {
    let root = repo_root();
    let report = build_optimization_hotspot_report(&root);
    assert_eq!(report.schema_version, OPTIMIZATION_HOTSPOT_SCHEMA_VERSION);
    assert_eq!(report.bead_id, OPTIMIZATION_HOTSPOT_BEAD_ID);
    assert_eq!(report.status, "pass", "report: {report:#?}");
    assert!(report.issues.is_empty(), "issues: {:#?}", report.issues);
    assert_eq!(report.follow_up_threshold, HOTSPOT_FOLLOW_UP_THRESHOLD);

    let families = report
        .rows
        .iter()
        .map(|row| row.family.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        families,
        BTreeSet::from([
            "ad_tape_backward_map",
            "cache_key_hashing",
            "durability_encode_decode",
            "egraph_saturation",
            "fft_linalg_reduction_mixes",
            "shape_kernels",
            "tensor_materialization",
            "vmap_multiplier",
        ])
    );
    assert_eq!(report.summary.row_count, 8);
    assert_eq!(report.summary.follow_up_required_count, 2);
    assert_eq!(report.summary.follow_up_created_count, 2);
}

#[test]
fn optimization_hotspot_rows_have_measurements_and_rank_order() {
    let root = repo_root();
    let report = build_optimization_hotspot_report(&root);
    let mut previous_score = f64::INFINITY;
    for (idx, row) in report.rows.iter().enumerate() {
        assert_eq!(row.rank, (idx + 1) as u32);
        assert_eq!(row.status, "measured");
        assert!(row.sample_count >= 16, "row: {row:#?}");
        assert!(row.p50_ns > 0, "row: {row:#?}");
        assert!(row.p95_ns >= row.p50_ns, "row: {row:#?}");
        assert!(row.p99_ns >= row.p95_ns, "row: {row:#?}");
        assert!(
            row.peak_rss_bytes.is_some_and(|value| value > 0),
            "row: {row:#?}"
        );
        assert_ne!(row.measurement_backend, "unavailable");
        assert!(row.behavior_witness > 0, "row: {row:#?}");
        assert!(!row.profile_case_ids.is_empty(), "row: {row:#?}");
        assert!((0.0..=1.0).contains(&row.confidence));
        assert!(row.priority_score <= previous_score);
        previous_score = row.priority_score;
    }
}

#[test]
fn optimization_hotspot_vmap_row_profiles_scaling_matrix() {
    let root = repo_root();
    let report = build_optimization_hotspot_report(&root);
    let vmap_row = report
        .rows
        .iter()
        .find(|row| row.hotspot_id == "hotspot-vmap-multiplier-001")
        .expect("vmap hotspot row should exist");
    let cases = vmap_row
        .profile_case_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    assert!(cases.contains("batch_size_8_axis0"));
    assert!(cases.contains("batch_size_64_axis0"));
    assert!(cases.contains("rank2_axis0_batchtrace"));
    assert!(cases.contains("rank2_axis1_loop_fallback"));
    assert!(cases.contains("vmap_grad_vector"));
    assert!(cases.contains("scan_scalar_carry_batched_xs"));
    assert!(
        vmap_row
            .one_lever_candidate
            .contains("BatchTrace vectorization"),
        "row: {vmap_row:#?}"
    );
}

#[test]
fn optimization_hotspot_threshold_controls_follow_up_beads() {
    let root = repo_root();
    let report = build_optimization_hotspot_report(&root);
    let planned_follow_ups = report
        .follow_up_beads
        .iter()
        .map(|bead| bead.bead_id.as_str())
        .collect::<BTreeSet<_>>();
    for row in &report.rows {
        if row.priority_score >= HOTSPOT_FOLLOW_UP_THRESHOLD {
            assert!(
                row.follow_up_bead_id.is_some(),
                "threshold row needs br follow-up: {row:#?}"
            );
        } else if let Some(follow_up) = row.follow_up_bead_id.as_deref() {
            assert!(
                planned_follow_ups.contains(follow_up),
                "below-threshold rows may only retain predeclared br follow-ups: {row:#?}"
            );
        }
    }
    let follow_up_hotspots = report
        .follow_up_beads
        .iter()
        .map(|bead| bead.hotspot_id.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        follow_up_hotspots,
        BTreeSet::from([
            "hotspot-egraph-saturation-001",
            "hotspot-vmap-multiplier-001"
        ])
    );
}

#[test]
fn optimization_hotspot_validation_rejects_contract_drift() {
    let root = repo_root();
    let mut report = build_optimization_hotspot_report(&root);
    report.schema_version = "wrong".to_owned();
    report.bead_id = "wrong".to_owned();
    report.follow_up_threshold = 1.0;
    report.required_hotspot_families.pop();
    report.status = "unknown".to_owned();

    let codes = issue_codes(&report);
    assert!(codes.contains("bad_schema_version"));
    assert!(codes.contains("bad_bead_id"));
    assert!(codes.contains("bad_follow_up_threshold"));
    assert!(codes.contains("missing_required_family"));
    assert!(codes.contains("bad_report_status"));
}

#[test]
fn optimization_hotspot_validation_rejects_forged_rows() {
    let root = repo_root();
    let mut report = build_optimization_hotspot_report(&root);
    report.rows[0].p95_ns = 0;
    report.rows[0].peak_rss_bytes = None;
    report.rows[0].measurement_backend = "unavailable".to_owned();
    report.rows[0].behavior_witness = 0;
    report.rows[0].profile_case_ids.clear();
    report.rows[0].confidence = 1.5;
    report.rows[0].behavior_proof_template_ref = "artifacts/missing/template.md".to_owned();
    report.rows[0].profile_artifact_refs.clear();
    report.rows[0].sample_count = 1;
    report.rows[1].priority_score = HOTSPOT_FOLLOW_UP_THRESHOLD;
    report.rows[1].follow_up_bead_id = None;
    report.rows[2].follow_up_bead_id = Some("frankenjax-unneeded".to_owned());

    let codes = issue_codes(&report);
    assert!(codes.contains("missing_latency_quantile"));
    assert!(codes.contains("missing_peak_rss"));
    assert!(codes.contains("memory_not_measured"));
    assert!(codes.contains("missing_behavior_witness"));
    assert!(codes.contains("missing_profile_cases"));
    assert!(codes.contains("bad_confidence"));
    assert!(codes.contains("missing_artifact_ref"));
    assert!(codes.contains("missing_artifact_refs"));
    assert!(codes.contains("too_few_samples"));
    assert!(codes.contains("missing_threshold_follow_up"));
    assert!(codes.contains("unnecessary_follow_up"));
}

#[test]
fn optimization_hotspot_outputs_write_and_validate_e2e_log() -> Result<(), String> {
    let root = repo_root();
    let temp = tempfile::tempdir().map_err(|err| err.to_string())?;
    let report_path = temp.path().join("optimization_hotspot_scoreboard.v1.json");
    let markdown_path = temp.path().join("optimization_hotspot_scoreboard.v1.md");
    let e2e_path = temp.path().join("e2e_optimization_hotspot_gate.e2e.json");
    let report = write_optimization_hotspot_outputs(&root, &report_path, &markdown_path)
        .map_err(|err| err.to_string())?;
    assert_eq!(report.status, "pass");
    assert!(report_path.exists());
    assert!(markdown_path.exists());
    let markdown = std::fs::read_to_string(&markdown_path).map_err(|err| err.to_string())?;
    assert!(markdown.contains("Optimization Hotspot Scoreboard"));
    assert!(markdown.contains("hotspot-vmap-multiplier-001"));

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_fj_optimization_hotspot_gate"))
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
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    validate_e2e_log_path(&e2e_path, Path::new(".")).map_err(|err| format!("{err:#?}"))?;
    let raw = std::fs::read_to_string(&e2e_path).map_err(|err| err.to_string())?;
    let log: E2EForensicLogV1 = serde_json::from_str(&raw).map_err(|err| err.to_string())?;
    assert_eq!(log.bead_id, OPTIMIZATION_HOTSPOT_BEAD_ID);
    assert_eq!(log.status, fj_conformance::e2e_log::E2ELogStatus::Pass);
    Ok(())
}

#[test]
fn optimization_hotspot_summary_and_markdown_are_dashboard_ready() {
    let root = repo_root();
    let report = build_optimization_hotspot_report(&root);
    let summary = optimization_hotspot_summary_json(&report);
    assert_eq!(summary["status"], "pass");
    assert_eq!(summary["summary"]["row_count"], 8);
    assert_eq!(summary["summary"]["follow_up_required_count"], 2);
    assert_eq!(summary["summary"]["follow_up_created_count"], 2);

    let markdown = optimization_hotspot_markdown(&report);
    assert!(markdown.contains("One-Lever Queue"));
    assert!(markdown.contains("frankenjax-cstq.11.1"));
    assert!(markdown.contains("frankenjax-cstq.11.2"));
}
