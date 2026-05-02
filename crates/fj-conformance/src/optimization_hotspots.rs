#![forbid(unsafe_code)]

use crate::durability::{
    SidecarConfig, encode_artifact_to_sidecar, generate_decode_proof, scrub_sidecar,
};
use crate::memory_performance::sample_process_memory;
use fj_cache::{CacheKeyInput, build_cache_key};
use fj_core::{
    CompatibilityMode, DType, Literal, Primitive, ProgramSpec, Shape, TensorValue,
    TraceTransformLedger, Transform, Value, build_program,
};
use fj_dispatch::{DispatchRequest, dispatch};
use fj_egraph::optimize_jaxpr;
use fj_lax::eval_primitive;
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub const OPTIMIZATION_HOTSPOT_SCHEMA_VERSION: &str =
    "frankenjax.optimization-hotspot-scoreboard.v1";
pub const OPTIMIZATION_HOTSPOT_BEAD_ID: &str = "frankenjax-cstq.11";
pub const HOTSPOT_FOLLOW_UP_THRESHOLD: f64 = 2.0;

const REQUIRED_HOTSPOT_FAMILIES: &[&str] = &[
    "vmap_multiplier",
    "ad_tape_backward_map",
    "tensor_materialization",
    "shape_kernels",
    "cache_key_hashing",
    "egraph_saturation",
    "fft_linalg_reduction_mixes",
    "durability_encode_decode",
];
static DURABILITY_PROBE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationHotspotReport {
    pub schema_version: String,
    pub bead_id: String,
    pub generated_at_unix_ms: u128,
    pub status: String,
    pub policy: String,
    pub follow_up_threshold: f64,
    pub required_hotspot_families: Vec<String>,
    pub summary: OptimizationHotspotSummary,
    pub rows: Vec<OptimizationHotspotRow>,
    pub follow_up_beads: Vec<OptimizationFollowUpBead>,
    pub issues: Vec<OptimizationHotspotIssue>,
    pub artifact_refs: Vec<String>,
    pub replay_command: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationHotspotSummary {
    pub required_family_count: usize,
    pub observed_family_count: usize,
    pub row_count: usize,
    pub rows_with_p95: usize,
    pub rows_with_p99: usize,
    pub rows_with_memory: usize,
    pub rows_with_profile_artifacts: usize,
    pub rows_with_behavior_proof_templates: usize,
    pub rows_with_replay: usize,
    pub follow_up_threshold_milli: u32,
    pub follow_up_required_count: usize,
    pub follow_up_created_count: usize,
    pub issue_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizationHotspotRow {
    pub rank: u32,
    pub hotspot_id: String,
    pub family: String,
    pub owner_crate: String,
    pub status: String,
    pub p50_ns: u128,
    pub p95_ns: u128,
    pub p99_ns: u128,
    pub peak_rss_bytes: Option<u64>,
    pub measurement_backend: String,
    pub sample_count: usize,
    pub confidence: f64,
    pub priority_score: f64,
    pub score_components: HotspotScoreComponents,
    pub one_lever_candidate: String,
    pub behavior_proof_template_ref: String,
    pub profile_artifact_refs: Vec<String>,
    pub source_evidence_refs: Vec<String>,
    pub follow_up_bead_id: Option<String>,
    pub replay_command: String,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HotspotScoreComponents {
    pub latency_weight: f64,
    pub p99_tail_weight: f64,
    pub memory_weight: f64,
    pub confidence_weight: f64,
    pub complexity_weight: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationFollowUpBead {
    pub bead_id: String,
    pub hotspot_id: String,
    pub title: String,
    pub priority: u8,
    pub one_lever_scope: String,
    pub required_evidence: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizationHotspotIssue {
    pub code: String,
    pub path: String,
    pub message: String,
}

impl OptimizationHotspotIssue {
    #[must_use]
    pub fn new(
        code: impl Into<String>,
        path: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code: code.into(),
            path: path.into(),
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizationHotspotOutputPaths {
    pub report: PathBuf,
    pub markdown: PathBuf,
    pub e2e: PathBuf,
}

impl OptimizationHotspotOutputPaths {
    #[must_use]
    pub fn for_root(root: &Path) -> Self {
        Self {
            report: root.join("artifacts/performance/optimization_hotspot_scoreboard.v1.json"),
            markdown: root.join("artifacts/performance/optimization_hotspot_scoreboard.v1.md"),
            e2e: root.join("artifacts/e2e/e2e_optimization_hotspot_gate.e2e.json"),
        }
    }
}

#[derive(Debug, Clone)]
struct HotspotDefinition {
    hotspot_id: &'static str,
    family: &'static str,
    owner_crate: &'static str,
    confidence: f64,
    score_components: HotspotScoreComponents,
    one_lever_candidate: &'static str,
    behavior_proof_template_ref: &'static str,
    profile_artifact_refs: Vec<&'static str>,
    source_evidence_refs: Vec<&'static str>,
    follow_up_bead_id: Option<&'static str>,
    replay_command: &'static str,
    notes: Vec<&'static str>,
}

#[derive(Debug, Clone)]
struct MeasuredHotspot {
    p50_ns: u128,
    p95_ns: u128,
    p99_ns: u128,
    peak_rss_bytes: Option<u64>,
    measurement_backend: String,
    sample_count: usize,
}

#[must_use]
pub fn build_optimization_hotspot_report(root: &Path) -> OptimizationHotspotReport {
    let mut rows = hotspot_definitions()
        .into_iter()
        .map(|definition| build_row(root, definition))
        .collect::<Vec<_>>();
    apply_measured_priority_scores(&mut rows);
    rows.sort_by(|left, right| {
        right
            .priority_score
            .total_cmp(&left.priority_score)
            .then_with(|| left.hotspot_id.cmp(&right.hotspot_id))
    });
    for (idx, row) in rows.iter_mut().enumerate() {
        row.rank = (idx + 1) as u32;
    }

    let follow_up_beads = follow_up_beads();
    let mut report = OptimizationHotspotReport {
        schema_version: OPTIMIZATION_HOTSPOT_SCHEMA_VERSION.to_owned(),
        bead_id: OPTIMIZATION_HOTSPOT_BEAD_ID.to_owned(),
        generated_at_unix_ms: now_unix_ms(),
        status: "pass".to_owned(),
        policy:
            "Optimization work must be profile-first, ranked by measured p95/p99/RSS and confidence, scoped to one lever, and backed by behavior-proof templates before code changes. Rows scoring below 2.0 stay monitored; rows scoring at or above 2.0 must have explicit br follow-up beads."
                .to_owned(),
        follow_up_threshold: HOTSPOT_FOLLOW_UP_THRESHOLD,
        required_hotspot_families: REQUIRED_HOTSPOT_FAMILIES
            .iter()
            .map(|family| (*family).to_owned())
            .collect(),
        summary: summarize_rows(&rows, &follow_up_beads, 0),
        rows,
        follow_up_beads,
        issues: Vec::new(),
        artifact_refs: vec![
            "artifacts/performance/global_performance_gate.v1.json".to_owned(),
            "artifacts/performance/memory_performance_gate.v1.json".to_owned(),
            "artifacts/performance/profiling_workflow.md".to_owned(),
            "artifacts/performance/isomorphism_proof_template.v1.md".to_owned(),
        ],
        replay_command: "./scripts/run_optimization_hotspot_gate.sh --enforce".to_owned(),
    };
    report.issues = validate_optimization_hotspot_report(root, &report);
    if !report.issues.is_empty() {
        report.status = "fail".to_owned();
    }
    report.summary = summarize_rows(&report.rows, &report.follow_up_beads, report.issues.len());
    report
}

pub fn write_optimization_hotspot_outputs(
    root: &Path,
    report_path: &Path,
    markdown_path: &Path,
) -> Result<OptimizationHotspotReport, std::io::Error> {
    let report = build_optimization_hotspot_report(root);
    write_json(report_path, &report)?;
    if let Some(parent) = markdown_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(markdown_path, optimization_hotspot_markdown(&report))?;
    Ok(report)
}

#[must_use]
pub fn optimization_hotspot_summary_json(report: &OptimizationHotspotReport) -> JsonValue {
    json!({
        "schema_version": report.schema_version,
        "bead_id": report.bead_id,
        "status": report.status,
        "follow_up_threshold": report.follow_up_threshold,
        "summary": report.summary,
        "issue_count": report.issues.len(),
        "issues": report.issues,
        "rows": report.rows.iter().map(|row| {
            json!({
                "rank": row.rank,
                "hotspot_id": row.hotspot_id,
                "family": row.family,
                "owner_crate": row.owner_crate,
                "p50_ns": row.p50_ns,
                "p95_ns": row.p95_ns,
                "p99_ns": row.p99_ns,
                "peak_rss_bytes": row.peak_rss_bytes,
                "confidence": row.confidence,
                "priority_score": row.priority_score,
                "follow_up_bead_id": row.follow_up_bead_id,
            })
        }).collect::<Vec<_>>(),
        "follow_up_beads": report.follow_up_beads,
    })
}

#[must_use]
pub fn optimization_hotspot_markdown(report: &OptimizationHotspotReport) -> String {
    let mut out = String::new();
    out.push_str("# Optimization Hotspot Scoreboard\n\n");
    out.push_str(&format!(
        "- Schema: `{}`\n- Bead: `{}`\n- Status: `{}`\n- Follow-up threshold: `{:.1}`\n- Rows: `{}`\n\n",
        report.schema_version,
        report.bead_id,
        report.status,
        report.follow_up_threshold,
        report.summary.row_count,
    ));
    out.push_str("| Rank | Family | Hotspot | p95 ns | p99 ns | Peak RSS | Score | Follow-up |\n");
    out.push_str("|---:|---|---|---:|---:|---:|---:|---|\n");
    for row in &report.rows {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{:.2}` | `{}` |\n",
            row.rank,
            row.family,
            row.hotspot_id,
            row.p95_ns,
            row.p99_ns,
            row.peak_rss_bytes
                .map_or_else(|| "n/a".to_owned(), |value| value.to_string()),
            row.priority_score,
            row.follow_up_bead_id.as_deref().unwrap_or("monitor"),
        ));
    }
    out.push_str("\n## One-Lever Queue\n\n");
    for row in &report.rows {
        out.push_str(&format!(
            "- `{}`: {} Proof template `{}`. Replay `{}`.\n",
            row.hotspot_id,
            row.one_lever_candidate,
            row.behavior_proof_template_ref,
            row.replay_command,
        ));
    }
    if report.issues.is_empty() {
        out.push_str("\nNo optimization hotspot issues found.\n");
    } else {
        out.push_str("\n## Issues\n\n");
        for issue in &report.issues {
            out.push_str(&format!(
                "- `{}` at `{}`: {}\n",
                issue.code, issue.path, issue.message
            ));
        }
    }
    out
}

#[must_use]
pub fn validate_optimization_hotspot_report(
    root: &Path,
    report: &OptimizationHotspotReport,
) -> Vec<OptimizationHotspotIssue> {
    let mut issues = Vec::new();
    if report.schema_version != OPTIMIZATION_HOTSPOT_SCHEMA_VERSION {
        issues.push(OptimizationHotspotIssue::new(
            "bad_schema_version",
            "$.schema_version",
            "optimization hotspot schema marker changed",
        ));
    }
    if report.bead_id != OPTIMIZATION_HOTSPOT_BEAD_ID {
        issues.push(OptimizationHotspotIssue::new(
            "bad_bead_id",
            "$.bead_id",
            "optimization hotspot report must stay bound to frankenjax-cstq.11",
        ));
    }
    if !matches!(report.status.as_str(), "pass" | "fail") {
        issues.push(OptimizationHotspotIssue::new(
            "bad_report_status",
            "$.status",
            format!("report status `{}` must be pass or fail", report.status),
        ));
    }
    if report.policy.trim().is_empty() {
        issues.push(OptimizationHotspotIssue::new(
            "empty_policy",
            "$.policy",
            "optimization report needs a user-facing policy",
        ));
    }
    if (report.follow_up_threshold - HOTSPOT_FOLLOW_UP_THRESHOLD).abs() > f64::EPSILON {
        issues.push(OptimizationHotspotIssue::new(
            "bad_follow_up_threshold",
            "$.follow_up_threshold",
            "follow-up threshold must remain 2.0",
        ));
    }
    if report.replay_command.trim().is_empty() {
        issues.push(OptimizationHotspotIssue::new(
            "missing_report_replay",
            "$.replay_command",
            "report needs a stable replay command",
        ));
    }

    validate_declared_families(report, &mut issues);
    validate_rows(root, report, &mut issues);
    validate_follow_ups(report, &mut issues);
    validate_artifact_refs(root, "$.artifact_refs", &report.artifact_refs, &mut issues);

    let expected_summary = summarize_rows(&report.rows, &report.follow_up_beads, issues.len());
    if report.summary != expected_summary {
        issues.push(OptimizationHotspotIssue::new(
            "bad_summary",
            "$.summary",
            "summary counts must match rows, follow-up beads, and issues",
        ));
    }

    issues
}

fn validate_declared_families(
    report: &OptimizationHotspotReport,
    issues: &mut Vec<OptimizationHotspotIssue>,
) {
    let required = REQUIRED_HOTSPOT_FAMILIES
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut declared = BTreeSet::new();
    for family in &report.required_hotspot_families {
        if !declared.insert(family.as_str()) {
            issues.push(OptimizationHotspotIssue::new(
                "duplicate_required_family",
                "$.required_hotspot_families",
                format!("duplicate required family `{family}`"),
            ));
        }
        if !required.contains(family.as_str()) {
            issues.push(OptimizationHotspotIssue::new(
                "unknown_required_family",
                "$.required_hotspot_families",
                format!("unknown required family `{family}`"),
            ));
        }
    }
    for required_family in required.difference(&declared) {
        issues.push(OptimizationHotspotIssue::new(
            "missing_required_family",
            "$.required_hotspot_families",
            format!("required hotspot family `{required_family}` is not declared"),
        ));
    }
}

fn validate_rows(
    root: &Path,
    report: &OptimizationHotspotReport,
    issues: &mut Vec<OptimizationHotspotIssue>,
) {
    let required = REQUIRED_HOTSPOT_FAMILIES
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut observed = BTreeSet::new();
    let mut row_ids = BTreeSet::new();
    let mut ranks = BTreeSet::new();
    let follow_up_ids = report
        .follow_up_beads
        .iter()
        .map(|bead| bead.bead_id.as_str())
        .collect::<BTreeSet<_>>();

    for (idx, row) in report.rows.iter().enumerate() {
        let path = format!("$.rows[{idx}]");
        observed.insert(row.family.as_str());
        if !row_ids.insert(row.hotspot_id.as_str()) {
            issues.push(OptimizationHotspotIssue::new(
                "duplicate_hotspot_id",
                path.clone(),
                format!("duplicate hotspot id `{}`", row.hotspot_id),
            ));
        }
        if !ranks.insert(row.rank) {
            issues.push(OptimizationHotspotIssue::new(
                "duplicate_rank",
                format!("{path}.rank"),
                format!("duplicate rank `{}`", row.rank),
            ));
        }
        if !required.contains(row.family.as_str()) {
            issues.push(OptimizationHotspotIssue::new(
                "unknown_hotspot_family",
                format!("{path}.family"),
                format!("unknown hotspot family `{}`", row.family),
            ));
        }
        if row.owner_crate.trim().is_empty()
            || row.status != "measured"
            || row.one_lever_candidate.trim().is_empty()
            || row.behavior_proof_template_ref.trim().is_empty()
            || row.replay_command.trim().is_empty()
        {
            issues.push(OptimizationHotspotIssue::new(
                "missing_row_contract_field",
                path.clone(),
                "row must declare owner, measured status, one-lever candidate, proof template, and replay command",
            ));
        }
        if row.p50_ns == 0 || row.p95_ns == 0 || row.p99_ns == 0 {
            issues.push(OptimizationHotspotIssue::new(
                "missing_latency_quantile",
                path.clone(),
                "row must record non-zero p50, p95, and p99 measurements",
            ));
        }
        if row.p50_ns > row.p95_ns || row.p95_ns > row.p99_ns {
            issues.push(OptimizationHotspotIssue::new(
                "bad_latency_order",
                path.clone(),
                "latency quantiles must be ordered p50 <= p95 <= p99",
            ));
        }
        if row.sample_count < 16 {
            issues.push(OptimizationHotspotIssue::new(
                "too_few_samples",
                format!("{path}.sample_count"),
                "hotspot measurements need at least 16 samples",
            ));
        }
        match row.peak_rss_bytes {
            Some(value) if value > 0 => {}
            _ => issues.push(OptimizationHotspotIssue::new(
                "missing_peak_rss",
                format!("{path}.peak_rss_bytes"),
                "row must record non-zero peak RSS bytes",
            )),
        }
        if row.measurement_backend == "unavailable" || row.measurement_backend.trim().is_empty() {
            issues.push(OptimizationHotspotIssue::new(
                "memory_not_measured",
                format!("{path}.measurement_backend"),
                "row must record a concrete memory measurement backend",
            ));
        }
        if !(0.0..=1.0).contains(&row.confidence) || !row.confidence.is_finite() {
            issues.push(OptimizationHotspotIssue::new(
                "bad_confidence",
                format!("{path}.confidence"),
                "confidence must be a finite value between 0 and 1",
            ));
        }
        if row.priority_score < 0.0 || !row.priority_score.is_finite() {
            issues.push(OptimizationHotspotIssue::new(
                "bad_priority_score",
                format!("{path}.priority_score"),
                "priority score must be finite and non-negative",
            ));
        }
        validate_score_components(row, &path, issues);
        validate_artifact_refs(
            root,
            &format!("{path}.profile_artifact_refs"),
            &row.profile_artifact_refs,
            issues,
        );
        validate_artifact_refs(
            root,
            &format!("{path}.source_evidence_refs"),
            &row.source_evidence_refs,
            issues,
        );
        validate_artifact_refs(
            root,
            &format!("{path}.behavior_proof_template_ref"),
            std::slice::from_ref(&row.behavior_proof_template_ref),
            issues,
        );
        if row.priority_score >= HOTSPOT_FOLLOW_UP_THRESHOLD {
            match row.follow_up_bead_id.as_deref() {
                Some(id) if follow_up_ids.contains(id) => {}
                Some(id) => issues.push(OptimizationHotspotIssue::new(
                    "missing_follow_up_record",
                    format!("{path}.follow_up_bead_id"),
                    format!("follow-up bead `{id}` is not listed in follow_up_beads"),
                )),
                None => issues.push(OptimizationHotspotIssue::new(
                    "missing_threshold_follow_up",
                    format!("{path}.follow_up_bead_id"),
                    format!(
                        "score {:.2} crosses threshold {:.1} and needs a br follow-up bead",
                        row.priority_score, HOTSPOT_FOLLOW_UP_THRESHOLD
                    ),
                )),
            }
        } else if row.follow_up_bead_id.is_some() {
            issues.push(OptimizationHotspotIssue::new(
                "unnecessary_follow_up",
                format!("{path}.follow_up_bead_id"),
                format!(
                    "score {:.2} is below threshold {:.1}; do not create backlog noise",
                    row.priority_score, HOTSPOT_FOLLOW_UP_THRESHOLD
                ),
            ));
        }
    }

    for required_family in required.difference(&observed) {
        issues.push(OptimizationHotspotIssue::new(
            "missing_hotspot_family",
            "$.rows",
            format!("required hotspot family `{required_family}` is not covered"),
        ));
    }
    for expected_rank in 1..=report.rows.len() as u32 {
        if !ranks.contains(&expected_rank) {
            issues.push(OptimizationHotspotIssue::new(
                "missing_rank",
                "$.rows",
                format!("rank `{expected_rank}` is absent"),
            ));
        }
    }
    for window in report.rows.windows(2) {
        if window[0].priority_score < window[1].priority_score {
            issues.push(OptimizationHotspotIssue::new(
                "rows_not_ranked_by_score",
                "$.rows",
                "rows must be sorted by descending priority score",
            ));
            break;
        }
    }
}

fn validate_score_components(
    row: &OptimizationHotspotRow,
    path: &str,
    issues: &mut Vec<OptimizationHotspotIssue>,
) {
    for (field, value) in [
        ("latency_weight", row.score_components.latency_weight),
        ("p99_tail_weight", row.score_components.p99_tail_weight),
        ("memory_weight", row.score_components.memory_weight),
        ("confidence_weight", row.score_components.confidence_weight),
        ("complexity_weight", row.score_components.complexity_weight),
    ] {
        if !(0.0..=1.0).contains(&value) || !value.is_finite() {
            issues.push(OptimizationHotspotIssue::new(
                "bad_score_component",
                format!("{path}.score_components.{field}"),
                "score component weights must be finite values between 0 and 1",
            ));
        }
    }
}

fn validate_follow_ups(
    report: &OptimizationHotspotReport,
    issues: &mut Vec<OptimizationHotspotIssue>,
) {
    let threshold_rows = report
        .rows
        .iter()
        .filter(|row| row.priority_score >= HOTSPOT_FOLLOW_UP_THRESHOLD)
        .filter_map(|row| row.follow_up_bead_id.as_deref())
        .collect::<BTreeSet<_>>();
    let row_ids = report
        .rows
        .iter()
        .map(|row| row.hotspot_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut seen_beads = BTreeSet::new();
    for (idx, bead) in report.follow_up_beads.iter().enumerate() {
        let path = format!("$.follow_up_beads[{idx}]");
        if !seen_beads.insert(bead.bead_id.as_str()) {
            issues.push(OptimizationHotspotIssue::new(
                "duplicate_follow_up_bead",
                path.clone(),
                format!("duplicate follow-up bead `{}`", bead.bead_id),
            ));
        }
        if !threshold_rows.contains(bead.bead_id.as_str()) {
            issues.push(OptimizationHotspotIssue::new(
                "follow_up_without_threshold_row",
                path.clone(),
                format!(
                    "follow-up bead `{}` is not required by a threshold-crossing row",
                    bead.bead_id
                ),
            ));
        }
        if !row_ids.contains(bead.hotspot_id.as_str()) {
            issues.push(OptimizationHotspotIssue::new(
                "follow_up_unknown_hotspot",
                path.clone(),
                format!("unknown follow-up hotspot `{}`", bead.hotspot_id),
            ));
        }
        if bead.title.trim().is_empty()
            || bead.one_lever_scope.trim().is_empty()
            || bead.required_evidence.is_empty()
        {
            issues.push(OptimizationHotspotIssue::new(
                "missing_follow_up_field",
                path,
                "follow-up bead must include title, one-lever scope, and required evidence",
            ));
        }
    }
}

fn validate_artifact_refs(
    root: &Path,
    path: &str,
    refs: &[String],
    issues: &mut Vec<OptimizationHotspotIssue>,
) {
    if refs.is_empty() {
        issues.push(OptimizationHotspotIssue::new(
            "missing_artifact_refs",
            path,
            "artifact reference list must not be empty",
        ));
        return;
    }
    for artifact_ref in refs {
        if artifact_ref.trim().is_empty() {
            issues.push(OptimizationHotspotIssue::new(
                "empty_artifact_ref",
                path,
                "artifact reference must be non-empty",
            ));
            continue;
        }
        if should_exist_on_disk(artifact_ref) && !root.join(artifact_ref).exists() {
            issues.push(OptimizationHotspotIssue::new(
                "missing_artifact_ref",
                path,
                format!("artifact ref `{artifact_ref}` is missing"),
            ));
        }
    }
}

fn should_exist_on_disk(artifact_ref: &str) -> bool {
    artifact_ref.starts_with("artifacts/")
        || artifact_ref.starts_with("crates/")
        || artifact_ref.starts_with("scripts/")
        || artifact_ref.starts_with("evidence/")
}

fn summarize_rows(
    rows: &[OptimizationHotspotRow],
    follow_up_beads: &[OptimizationFollowUpBead],
    issue_count: usize,
) -> OptimizationHotspotSummary {
    let observed_family_count = rows
        .iter()
        .map(|row| row.family.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    OptimizationHotspotSummary {
        required_family_count: REQUIRED_HOTSPOT_FAMILIES.len(),
        observed_family_count,
        row_count: rows.len(),
        rows_with_p95: rows.iter().filter(|row| row.p95_ns > 0).count(),
        rows_with_p99: rows.iter().filter(|row| row.p99_ns > 0).count(),
        rows_with_memory: rows
            .iter()
            .filter(|row| row.peak_rss_bytes.is_some_and(|value| value > 0))
            .count(),
        rows_with_profile_artifacts: rows
            .iter()
            .filter(|row| !row.profile_artifact_refs.is_empty())
            .count(),
        rows_with_behavior_proof_templates: rows
            .iter()
            .filter(|row| !row.behavior_proof_template_ref.is_empty())
            .count(),
        rows_with_replay: rows
            .iter()
            .filter(|row| !row.replay_command.is_empty())
            .count(),
        follow_up_threshold_milli: (HOTSPOT_FOLLOW_UP_THRESHOLD * 1000.0) as u32,
        follow_up_required_count: rows
            .iter()
            .filter(|row| row.priority_score >= HOTSPOT_FOLLOW_UP_THRESHOLD)
            .count(),
        follow_up_created_count: follow_up_beads.len(),
        issue_count,
    }
}

fn build_row(root: &Path, definition: HotspotDefinition) -> OptimizationHotspotRow {
    let measured = match definition.family {
        "vmap_multiplier" => measure_hotspot(workload_vmap_multiplier),
        "ad_tape_backward_map" => measure_hotspot(workload_ad_tape_backward_map),
        "tensor_materialization" => measure_hotspot(workload_tensor_materialization),
        "shape_kernels" => measure_hotspot(workload_shape_kernels),
        "cache_key_hashing" => measure_hotspot(workload_cache_key_hashing),
        "egraph_saturation" => measure_hotspot(workload_egraph_saturation),
        "fft_linalg_reduction_mixes" => measure_hotspot(workload_fft_linalg_reduction_mix),
        "durability_encode_decode" => measure_hotspot(|| workload_durability_encode_decode(root)),
        _ => Err(format!("unknown hotspot family `{}`", definition.family)),
    }
    .unwrap_or_else(|_| MeasuredHotspot {
        p50_ns: 1,
        p95_ns: 1,
        p99_ns: 1,
        peak_rss_bytes: sample_process_memory().peak_rss_bytes,
        measurement_backend: sample_process_memory().measurement_backend,
        sample_count: 0,
    });

    OptimizationHotspotRow {
        rank: 0,
        hotspot_id: definition.hotspot_id.to_owned(),
        family: definition.family.to_owned(),
        owner_crate: definition.owner_crate.to_owned(),
        status: "measured".to_owned(),
        p50_ns: measured.p50_ns,
        p95_ns: measured.p95_ns,
        p99_ns: measured.p99_ns,
        peak_rss_bytes: measured.peak_rss_bytes,
        measurement_backend: measured.measurement_backend,
        sample_count: measured.sample_count,
        confidence: definition.confidence,
        priority_score: 0.0,
        score_components: definition.score_components,
        one_lever_candidate: definition.one_lever_candidate.to_owned(),
        behavior_proof_template_ref: definition.behavior_proof_template_ref.to_owned(),
        profile_artifact_refs: definition
            .profile_artifact_refs
            .into_iter()
            .map(str::to_owned)
            .collect(),
        source_evidence_refs: definition
            .source_evidence_refs
            .into_iter()
            .map(str::to_owned)
            .collect(),
        follow_up_bead_id: definition.follow_up_bead_id.map(str::to_owned),
        replay_command: definition.replay_command.to_owned(),
        notes: definition.notes.into_iter().map(str::to_owned).collect(),
    }
}

fn apply_measured_priority_scores(rows: &mut [OptimizationHotspotRow]) {
    let max_p95 = rows.iter().map(|row| row.p95_ns).max().unwrap_or(1).max(1) as f64;
    let max_p99 = rows.iter().map(|row| row.p99_ns).max().unwrap_or(1).max(1) as f64;
    let max_rss = rows
        .iter()
        .filter_map(|row| row.peak_rss_bytes)
        .max()
        .unwrap_or(1)
        .max(1) as f64;

    for row in rows {
        let p95_ratio = row.p95_ns as f64 / max_p95;
        let p99_ratio = row.p99_ns as f64 / max_p99;
        let rss_ratio = row.peak_rss_bytes.unwrap_or(0) as f64 / max_rss;
        let components = &row.score_components;
        let score = components.latency_weight * p95_ratio
            + components.p99_tail_weight * p99_ratio
            + components.memory_weight * rss_ratio
            + components.confidence_weight * row.confidence
            + components.complexity_weight;
        row.priority_score = (score * 100.0).round() / 100.0;
    }
}

fn measure_hotspot<F>(mut workload: F) -> Result<MeasuredHotspot, String>
where
    F: FnMut() -> Result<u64, String>,
{
    const SAMPLES: usize = 32;
    workload()?;
    let before = sample_process_memory();
    let mut durations = Vec::with_capacity(SAMPLES);
    let mut witness = 0_u64;
    for _ in 0..SAMPLES {
        let start = Instant::now();
        witness = witness.saturating_add(workload()?);
        durations.push(start.elapsed().as_nanos().max(1));
    }
    if witness == 0 {
        return Err("hotspot workload produced an empty behavior witness".to_owned());
    }
    durations.sort_unstable();
    let after = sample_process_memory();
    Ok(MeasuredHotspot {
        p50_ns: percentile(&durations, 50),
        p95_ns: percentile(&durations, 95),
        p99_ns: percentile(&durations, 99),
        peak_rss_bytes: after.peak_rss_bytes.or(before.peak_rss_bytes),
        measurement_backend: after.measurement_backend,
        sample_count: durations.len(),
    })
}

fn percentile(sorted: &[u128], percentile: usize) -> u128 {
    let last = sorted.len().saturating_sub(1);
    let idx = ((sorted.len() * percentile).div_ceil(100)).saturating_sub(1);
    sorted[idx.min(last)]
}

fn hotspot_definitions() -> Vec<HotspotDefinition> {
    let proof_template = "artifacts/performance/isomorphism_proof_template.v1.md";
    vec![
        HotspotDefinition {
            hotspot_id: "hotspot-vmap-multiplier-001",
            family: "vmap_multiplier",
            owner_crate: "fj-dispatch",
            confidence: 0.86,
            score_components: score_components(0.80, 0.70, 0.40, 0.95, 1.00),
            one_lever_candidate: "Profile batch-size/rank scaling and choose one BatchTrace vectorization lever only after same-worker evidence.",
            behavior_proof_template_ref: proof_template,
            profile_artifact_refs: vec![
                "artifacts/performance/global_performance_gate.v1.json",
                "artifacts/e2e/e2e_vmap_tensor_ops.e2e.json",
            ],
            source_evidence_refs: vec![
                "crates/fj-dispatch/benches/dispatch_baseline.rs",
                "artifacts/performance/profiling_workflow.md",
            ],
            follow_up_bead_id: Some("frankenjax-cstq.11.1"),
            replay_command: "CARGO_TARGET_DIR=/data/tmp/frankenjax-hotspot-vmap rch exec -- cargo bench -p fj-dispatch --bench dispatch_baseline -- vmap",
            notes: vec![
                "Recent vmap scan/switch work proves this area can yield large wins but also regresses when generic vectorization is guessed.",
                "The follow-up is profile-only until a single lever is justified.",
            ],
        },
        HotspotDefinition {
            hotspot_id: "hotspot-egraph-saturation-001",
            family: "egraph_saturation",
            owner_crate: "fj-egraph",
            confidence: 0.74,
            score_components: score_components(0.62, 0.70, 0.20, 0.74, 0.80),
            one_lever_candidate: "Profile shape-adjacent tensor programs and choose one rewrite or extraction lever with semantics-preservation proof.",
            behavior_proof_template_ref: proof_template,
            profile_artifact_refs: vec![
                "artifacts/performance/benchmark_baselines_v2_2026-03-12.json",
                "crates/fj-egraph/src/lib.rs",
            ],
            source_evidence_refs: vec![
                "crates/fj-conformance/tests/egraph_preserves_semantics.rs",
                "FEATURE_PARITY.md",
            ],
            follow_up_bead_id: Some("frankenjax-cstq.11.2"),
            replay_command: "CARGO_TARGET_DIR=/data/tmp/frankenjax-hotspot-egraph rch exec -- cargo test -p fj-conformance --test egraph_preserves_semantics -- --nocapture",
            notes: vec![
                "E-graph saturation is correctness-sensitive; every change must compare optimized and unoptimized behavior.",
                "Score crosses threshold because shape-aware expansion is a known parity frontier with non-trivial compile risk.",
            ],
        },
        HotspotDefinition {
            hotspot_id: "hotspot-ad-tape-001",
            family: "ad_tape_backward_map",
            owner_crate: "fj-ad",
            confidence: 0.80,
            score_components: score_components(0.55, 0.45, 0.20, 0.80, 0.65),
            one_lever_candidate: "Monitor reverse-mode tape lookup and only open a bead if p99 or allocation evidence crosses the threshold.",
            behavior_proof_template_ref: proof_template,
            profile_artifact_refs: vec![
                "artifacts/performance/global_performance_gate.v1.json",
                "artifacts/e2e/e2e_control_flow_ad.e2e.json",
            ],
            source_evidence_refs: vec![
                "crates/fj-ad/src/lib.rs",
                "crates/fj-conformance/tests/ad_numerical_verification.rs",
            ],
            follow_up_bead_id: None,
            replay_command: "CARGO_TARGET_DIR=/data/tmp/frankenjax-hotspot-ad rch exec -- cargo bench -p fj-api --bench api_overhead -- value_and_grad",
            notes: vec!["Below threshold: keep measured but do not create a speculative bead."],
        },
        HotspotDefinition {
            hotspot_id: "hotspot-fft-linalg-reduction-001",
            family: "fft_linalg_reduction_mixes",
            owner_crate: "fj-lax",
            confidence: 0.78,
            score_components: score_components(0.58, 0.40, 0.30, 0.78, 0.60),
            one_lever_candidate: "Keep FFT/linalg/reduction mixed workloads in the scoreboard until higher-rank oracle fixtures justify one primitive-specific optimization.",
            behavior_proof_template_ref: proof_template,
            profile_artifact_refs: vec![
                "artifacts/performance/global_performance_gate.v1.json",
                "artifacts/e2e/e2e_multirank_fixtures.e2e.json",
            ],
            source_evidence_refs: vec![
                "crates/fj-lax/benches/lax_baseline.rs",
                "crates/fj-conformance/tests/lax_oracle.rs",
            ],
            follow_up_bead_id: None,
            replay_command: "CARGO_TARGET_DIR=/data/tmp/frankenjax-hotspot-lax rch exec -- cargo bench -p fj-lax --bench lax_baseline -- lax_eval",
            notes: vec![
                "Below threshold because recent FFT/vector allocation work already landed; wait for cstq.7 fixture expansion.",
            ],
        },
        HotspotDefinition {
            hotspot_id: "hotspot-tensor-materialization-001",
            family: "tensor_materialization",
            owner_crate: "fj-lax",
            confidence: 0.82,
            score_components: score_components(0.45, 0.35, 0.45, 0.82, 0.35),
            one_lever_candidate: "Monitor tensor output materialization after recent preallocation wins; require new allocation proof before another bead.",
            behavior_proof_template_ref: proof_template,
            profile_artifact_refs: vec![
                "artifacts/performance/memory_performance_gate.v1.json",
                "crates/fj-lax/benches/lax_baseline.rs",
            ],
            source_evidence_refs: vec![
                "crates/fj-lax/src/arithmetic.rs",
                "artifacts/performance/profiling_workflow.md",
            ],
            follow_up_bead_id: None,
            replay_command: "CARGO_TARGET_DIR=/data/tmp/frankenjax-hotspot-tensor rch exec -- cargo bench -p fj-lax --bench lax_baseline -- add_1k",
            notes: vec![
                "Below threshold: current evidence supports guardrail monitoring rather than new work.",
            ],
        },
        HotspotDefinition {
            hotspot_id: "hotspot-durability-encode-decode-001",
            family: "durability_encode_decode",
            owner_crate: "fj-conformance",
            confidence: 0.72,
            score_components: score_components(0.20, 0.15, 0.20, 0.50, 0.20),
            one_lever_candidate: "Track sidecar encode/decode RSS and only optimize after larger artifact profiles show a single bottleneck.",
            behavior_proof_template_ref: proof_template,
            profile_artifact_refs: vec![
                "artifacts/performance/memory_performance_gate.v1.json",
                "artifacts/durability/benchmark_baselines_v2.decode-proof.json",
            ],
            source_evidence_refs: vec![
                "crates/fj-conformance/src/durability.rs",
                "scripts/durability_ci_gate.sh",
            ],
            follow_up_bead_id: None,
            replay_command: "CARGO_TARGET_DIR=/data/tmp/frankenjax-hotspot-durability rch exec -- cargo run -p fj-conformance --bin fj_durability -- pipeline --artifact artifacts/performance/benchmark_baselines_v2_2026-03-12.json --sidecar /data/tmp/frankenjax-hotspot-durability/sidecar.json --report /data/tmp/frankenjax-hotspot-durability/scrub.json --proof /data/tmp/frankenjax-hotspot-durability/proof.json",
            notes: vec![
                "Below threshold: durability is covered by RSS gate and sidecar proof artifacts.",
            ],
        },
        HotspotDefinition {
            hotspot_id: "hotspot-shape-kernels-001",
            family: "shape_kernels",
            owner_crate: "fj-trace",
            confidence: 0.76,
            score_components: score_components(0.35, 0.25, 0.15, 0.76, 0.55),
            one_lever_candidate: "Keep high-rank shape inference under guardrail measurement; do not optimize without a failing scale profile.",
            behavior_proof_template_ref: proof_template,
            profile_artifact_refs: vec![
                "artifacts/performance/global_performance_gate.v1.json",
                "artifacts/testing/logs/fj-trace/fj_trace__tests__prop_shape_inference_deterministic.json",
            ],
            source_evidence_refs: vec![
                "crates/fj-trace/src/lib.rs",
                "crates/fj-lax/src/tensor_ops.rs",
            ],
            follow_up_bead_id: None,
            replay_command: "CARGO_TARGET_DIR=/data/tmp/frankenjax-hotspot-shape rch exec -- cargo test -p fj-trace prop_shape_inference_deterministic -- --nocapture",
            notes: vec![
                "Below threshold: correctness and overflow hardening are more important than speculative shape-kernel speedups.",
            ],
        },
        HotspotDefinition {
            hotspot_id: "hotspot-cache-key-hashing-001",
            family: "cache_key_hashing",
            owner_crate: "fj-cache",
            confidence: 0.88,
            score_components: score_components(0.25, 0.20, 0.10, 0.88, 0.30),
            one_lever_candidate: "Keep streaming cache-key hasher as a regression sentinel; avoid new work unless cache-key p99 regresses.",
            behavior_proof_template_ref: proof_template,
            profile_artifact_refs: vec![
                "artifacts/performance/evidence/ir_core/streaming_cache_key_hasher.json",
                "artifacts/performance/global_performance_gate.v1.json",
            ],
            source_evidence_refs: vec![
                "crates/fj-cache/src/lib.rs",
                "crates/fj-cache/benches/cache_baseline.rs",
            ],
            follow_up_bead_id: None,
            replay_command: "CARGO_TARGET_DIR=/data/tmp/frankenjax-hotspot-cache rch exec -- cargo bench -p fj-cache --bench cache_baseline -- cache_key",
            notes: vec![
                "Below threshold because the streaming hasher optimization already landed with evidence.",
            ],
        },
    ]
}

fn score_components(
    latency_weight: f64,
    p99_tail_weight: f64,
    memory_weight: f64,
    confidence_weight: f64,
    complexity_weight: f64,
) -> HotspotScoreComponents {
    HotspotScoreComponents {
        latency_weight,
        p99_tail_weight,
        memory_weight,
        confidence_weight,
        complexity_weight,
    }
}

fn follow_up_beads() -> Vec<OptimizationFollowUpBead> {
    vec![
        OptimizationFollowUpBead {
            bead_id: "frankenjax-cstq.11.1".to_owned(),
            hotspot_id: "hotspot-vmap-multiplier-001".to_owned(),
            title: "Profile vmap multiplier scaling before next BatchTrace optimization".to_owned(),
            priority: 2,
            one_lever_scope:
                "Choose at most one BatchTrace vectorization or slicing lever after p95/p99/RSS evidence."
                    .to_owned(),
            required_evidence: required_follow_up_evidence(),
        },
        OptimizationFollowUpBead {
            bead_id: "frankenjax-cstq.11.2".to_owned(),
            hotspot_id: "hotspot-egraph-saturation-001".to_owned(),
            title: "Profile egraph saturation shape-aware rewrite expansion".to_owned(),
            priority: 2,
            one_lever_scope:
                "Choose at most one rewrite or extraction lever and prove optimized/unoptimized isomorphism."
                    .to_owned(),
            required_evidence: required_follow_up_evidence(),
        },
    ]
}

fn required_follow_up_evidence() -> Vec<String> {
    vec![
        "same-worker p50/p95/p99 baseline".to_owned(),
        "peak RSS measurement".to_owned(),
        "profile artifact".to_owned(),
        "one-lever implementation note".to_owned(),
        "isomorphism proof".to_owned(),
        "conformance or e2e replay log".to_owned(),
        "post-change rebaseline or rejected-optimization note".to_owned(),
    ]
}

fn workload_vmap_multiplier() -> Result<u64, String> {
    let response = dispatch(dispatch_request(
        ProgramSpec::AddOne,
        &[Transform::Vmap],
        vec![Value::vector_i64(&[1, 2, 3, 4, 5, 6, 7, 8]).map_err(|err| err.to_string())?],
    ))
    .map_err(|err| format!("vmap dispatch failed: {err}"))?;
    Ok(response.outputs.len() as u64 + response.cache_key.len() as u64)
}

fn workload_ad_tape_backward_map() -> Result<u64, String> {
    let response = dispatch(dispatch_request(
        ProgramSpec::SquarePlusLinear,
        &[Transform::Grad],
        vec![Value::scalar_f64(3.0)],
    ))
    .map_err(|err| format!("grad dispatch failed: {err}"))?;
    Ok(response.outputs.len() as u64 + response.cache_key.len() as u64)
}

fn workload_tensor_materialization() -> Result<u64, String> {
    let elements = (0..128).map(Literal::I64).collect::<Vec<_>>();
    let left = Value::Tensor(
        TensorValue::new(DType::I64, Shape::vector(128), elements.clone())
            .map_err(|err| err.to_string())?,
    );
    let right = Value::Tensor(
        TensorValue::new(DType::I64, Shape::vector(128), elements)
            .map_err(|err| err.to_string())?,
    );
    let output = eval_primitive(Primitive::Add, &[left, right], &BTreeMap::new())
        .map_err(|err| format!("tensor add failed: {err}"))?;
    Ok(output
        .as_tensor()
        .map(|tensor| tensor.elements.len() as u64)
        .unwrap_or(0))
}

fn workload_shape_kernels() -> Result<u64, String> {
    let scalar = Value::scalar_i64(7);
    let mut params = BTreeMap::new();
    params.insert("shape".to_owned(), "4,8".to_owned());
    let output = eval_primitive(Primitive::BroadcastInDim, &[scalar], &params)
        .map_err(|err| format!("broadcast_in_dim failed: {err}"))?;
    Ok(output
        .as_tensor()
        .map(|tensor| tensor.elements.len() as u64)
        .unwrap_or(0))
}

fn workload_cache_key_hashing() -> Result<u64, String> {
    let mut compile_options = BTreeMap::new();
    compile_options.insert("egraph_optimize".to_owned(), "false".to_owned());
    let key = build_cache_key(&CacheKeyInput {
        mode: CompatibilityMode::Strict,
        backend: "cpu".to_owned(),
        jaxpr: build_program(ProgramSpec::AddOneMulTwo),
        transform_stack: vec![Transform::Jit, Transform::Vmap],
        compile_options,
        custom_hook: None,
        unknown_incompatible_features: Vec::new(),
    })
    .map_err(|err| format!("cache key failed: {err}"))?;
    Ok(key.digest_hex.len() as u64)
}

fn workload_egraph_saturation() -> Result<u64, String> {
    let input = build_program(ProgramSpec::AddOneMulTwo);
    let optimized = optimize_jaxpr(&input);
    optimized
        .validate_well_formed()
        .map_err(|err| format!("optimized jaxpr invalid: {err}"))?;
    Ok(optimized.equations.len() as u64 + optimized.outvars.len() as u64)
}

fn workload_fft_linalg_reduction_mix() -> Result<u64, String> {
    let fft_input = Value::Tensor(
        TensorValue::new(
            DType::Complex128,
            Shape::vector(4),
            vec![
                Literal::from_complex128(1.0, 0.0),
                Literal::from_complex128(0.0, 0.0),
                Literal::from_complex128(0.0, 0.0),
                Literal::from_complex128(0.0, 0.0),
            ],
        )
        .map_err(|err| err.to_string())?,
    );
    let fft = eval_primitive(Primitive::Fft, &[fft_input], &BTreeMap::new())
        .map_err(|err| format!("fft failed: {err}"))?;

    let chol_input = Value::Tensor(
        TensorValue::new(
            DType::F64,
            Shape { dims: vec![2, 2] },
            vec![
                Literal::from_f64(4.0),
                Literal::from_f64(1.0),
                Literal::from_f64(1.0),
                Literal::from_f64(3.0),
            ],
        )
        .map_err(|err| err.to_string())?,
    );
    let chol = eval_primitive(Primitive::Cholesky, &[chol_input], &BTreeMap::new())
        .map_err(|err| format!("cholesky failed: {err}"))?;

    let reduce_input = Value::Tensor(
        TensorValue::new(
            DType::I64,
            Shape::vector(128),
            (0..128).map(Literal::I64).collect(),
        )
        .map_err(|err| err.to_string())?,
    );
    let reduced = eval_primitive(Primitive::ReduceSum, &[reduce_input], &BTreeMap::new())
        .map_err(|err| format!("reduce_sum failed: {err}"))?;
    Ok(tensor_len(&fft) + tensor_len(&chol) + reduced.as_i64_scalar().map_or(0, |_| 1))
}

fn workload_durability_encode_decode(root: &Path) -> Result<u64, String> {
    let sequence = DURABILITY_PROBE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let dir = root.join(format!(
        "target/optimization-hotspot-durability-probe/{}-{sequence}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).map_err(|err| format!("create durability dir: {err}"))?;
    let artifact = dir.join("probe.json");
    let sidecar = dir.join("probe.sidecar.json");
    let scrub = dir.join("probe.scrub.json");
    let proof = dir.join("probe.proof.json");
    fs::write(
        &artifact,
        br#"{"schema_version":"frankenjax.optimization-hotspot-probe.v1","payload":[1,2,3,4]}"#,
    )
    .map_err(|err| format!("write durability artifact: {err}"))?;
    let manifest = encode_artifact_to_sidecar(&artifact, &sidecar, &SidecarConfig::default())
        .map_err(|err| err.to_string())?;
    let scrub_report = scrub_sidecar(&sidecar, &artifact, &scrub).map_err(|err| err.to_string())?;
    let proof_report =
        generate_decode_proof(&sidecar, &artifact, &proof, 0).map_err(|err| err.to_string())?;
    if !scrub_report.decoded_matches_expected || !proof_report.recovered {
        return Err("durability probe did not recover".to_owned());
    }
    Ok(manifest.total_symbols as u64 + scrub_report.total_symbols as u64)
}

fn tensor_len(value: &Value) -> u64 {
    value
        .as_tensor()
        .map(|tensor| tensor.elements.len() as u64)
        .unwrap_or(0)
}

fn dispatch_request(
    spec: ProgramSpec,
    transforms: &[Transform],
    args: Vec<Value>,
) -> DispatchRequest {
    let mut ledger = TraceTransformLedger::new(build_program(spec));
    for (idx, transform) in transforms.iter().enumerate() {
        ledger.push_transform(
            *transform,
            format!("evidence-{}-{}", transform.as_str(), idx),
        );
    }
    DispatchRequest {
        args,
        ledger,
        mode: CompatibilityMode::Strict,
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: Vec::new(),
    }
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let raw = serde_json::to_string_pretty(value).map_err(std::io::Error::other)?;
    fs::write(path, raw)?;
    Ok(())
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}
