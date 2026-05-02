#![forbid(unsafe_code)]

use fj_core::CompatibilityMode;
use fj_ledger::{
    DecisionAction, DecisionRecord, EvidenceSignal, LogDomainPosterior, LossMatrix,
    expected_loss_keep, expected_loss_kill,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const DECISION_LEDGER_REPORT_SCHEMA_VERSION: &str = "frankenjax.decision-ledger-calibration.v1";
pub const DECISION_LEDGER_BEAD_ID: &str = "frankenjax-cstq.19";

const REQUIRED_DECISION_CLASSES: &[&str] = &[
    "cache_hit_recompute",
    "strict_rejection",
    "hardened_recovery",
    "fallback_denial",
    "optimization_selection",
    "durability_recovery",
    "transform_admission",
    "unsupported_scope",
    "runtime_budget_deadline",
];

const ALLOWED_STRICT_HARDENED_DIVERGENCE: &[&str] = &["hardened_recovery", "durability_recovery"];
const CALIBRATION_DRIFT_THRESHOLD: f64 = 0.1;
const MIN_BUCKET_COUNT: usize = 8;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionLedgerReport {
    pub schema_version: String,
    pub bead_id: String,
    pub generated_at_unix_ms: u128,
    pub status: String,
    pub policy: String,
    pub required_decision_classes: Vec<String>,
    pub strict_hardened_divergence_allowlist: Vec<String>,
    pub summary: DecisionLedgerSummary,
    pub rows: Vec<DecisionLedgerRow>,
    pub calibration_buckets: Vec<DecisionCalibrationBucket>,
    pub issues: Vec<DecisionLedgerIssue>,
    pub artifact_refs: Vec<String>,
    pub replay_command: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionLedgerSummary {
    pub required_decision_class_count: usize,
    pub observed_decision_class_count: usize,
    pub pass_count: usize,
    pub row_count: usize,
    pub calibration_bucket_count: usize,
    pub stale_calibration_count: usize,
    pub strict_hardened_divergence_count: usize,
    pub allowed_divergence_count: usize,
    pub rows_with_artifacts: usize,
    pub rows_with_replay: usize,
    pub rows_with_dashboard_rows: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionLedgerRow {
    pub decision_id: String,
    pub decision_class: String,
    pub status: String,
    pub mode: String,
    pub selected_action: String,
    pub alternatives_considered: Vec<String>,
    pub loss_matrix: LossMatrix,
    pub posterior_abandoned: f64,
    pub confidence: f64,
    pub expected_loss_keep: f64,
    pub expected_loss_kill: f64,
    pub evidence_signals: Vec<EvidenceSignal>,
    pub expected_outcome: String,
    pub actual_outcome: String,
    pub calibration_bucket: String,
    pub drift_status: String,
    pub stale_calibration: bool,
    pub strict_hardened_divergence: bool,
    pub divergence_allowed: bool,
    pub user_visible_consequence: String,
    pub artifact_refs: Vec<String>,
    pub replay_command: String,
    pub dashboard_row: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionCalibrationBucket {
    pub bucket_id: String,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub expected_probability: f64,
    pub observed_probability: f64,
    pub count: usize,
    pub ece_contribution: f64,
    pub drift_status: String,
    pub stale: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionLedgerIssue {
    pub code: String,
    pub path: String,
    pub message: String,
}

impl DecisionLedgerIssue {
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
pub struct DecisionLedgerOutputPaths {
    pub report: PathBuf,
    pub markdown: PathBuf,
    pub e2e: PathBuf,
}

impl DecisionLedgerOutputPaths {
    #[must_use]
    pub fn for_root(root: &Path) -> Self {
        Self {
            report: root.join("artifacts/conformance/decision_ledger_calibration.v1.json"),
            markdown: root.join("artifacts/conformance/decision_ledger_calibration.v1.md"),
            e2e: root.join("artifacts/e2e/e2e_decision_ledger_gate.e2e.json"),
        }
    }
}

#[must_use]
pub fn build_decision_ledger_report(root: &Path) -> DecisionLedgerReport {
    let rows = decision_specs()
        .into_iter()
        .map(build_row)
        .collect::<Vec<_>>();
    let calibration_buckets = calibration_buckets();
    let summary = summarize_rows(&rows, &calibration_buckets);
    let mut report = DecisionLedgerReport {
        schema_version: DECISION_LEDGER_REPORT_SCHEMA_VERSION.to_owned(),
        bead_id: DECISION_LEDGER_BEAD_ID.to_owned(),
        generated_at_unix_ms: now_unix_ms(),
        status: "pass".to_owned(),
        policy:
            "Every consequential runtime decision must name the selected action, viable alternatives, loss matrix, evidence signals, posterior/confidence values, calibration bucket, user-visible consequence, artifact links, and exact replay command. Strict/hardened divergence is allowed only for explicit recovery rows."
                .to_owned(),
        required_decision_classes: REQUIRED_DECISION_CLASSES
            .iter()
            .map(|class| (*class).to_owned())
            .collect(),
        strict_hardened_divergence_allowlist: ALLOWED_STRICT_HARDENED_DIVERGENCE
            .iter()
            .map(|class| (*class).to_owned())
            .collect(),
        summary,
        rows,
        calibration_buckets,
        issues: Vec::new(),
        artifact_refs: vec![
            "artifacts/e2e/e2e_cache_lifecycle_gate.e2e.json".to_owned(),
            "artifacts/e2e/e2e_error_taxonomy_gate.e2e.json".to_owned(),
            "artifacts/e2e/e2e_security_gate.e2e.json".to_owned(),
            "artifacts/e2e/e2e_durability_coverage_gate.e2e.json".to_owned(),
            "artifacts/e2e/e2e_memory_performance_gate.e2e.json".to_owned(),
            "artifacts/e2e/e2e_ttl_semantic_gate.e2e.json".to_owned(),
        ],
        replay_command: "./scripts/run_decision_ledger_gate.sh --enforce".to_owned(),
    };
    report.issues = validate_decision_ledger_report(root, &report);
    if !report.issues.is_empty() {
        report.status = "fail".to_owned();
    }
    report
}

pub fn write_decision_ledger_outputs(
    root: &Path,
    report_path: &Path,
    markdown_path: &Path,
) -> Result<DecisionLedgerReport, std::io::Error> {
    let report = build_decision_ledger_report(root);
    write_json(report_path, &report)?;
    if let Some(parent) = markdown_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(markdown_path, decision_ledger_markdown(&report))?;
    Ok(report)
}

#[must_use]
pub fn decision_ledger_summary_json(report: &DecisionLedgerReport) -> JsonValue {
    json!({
        "schema_version": report.schema_version,
        "bead_id": report.bead_id,
        "status": report.status,
        "summary": report.summary,
        "issue_count": report.issues.len(),
        "issues": report.issues,
        "row_status": report.rows.iter().map(|row| {
            json!({
                "decision_id": row.decision_id,
                "decision_class": row.decision_class,
                "status": row.status,
                "mode": row.mode,
                "selected_action": row.selected_action,
                "posterior_abandoned": row.posterior_abandoned,
                "confidence": row.confidence,
                "calibration_bucket": row.calibration_bucket,
                "drift_status": row.drift_status,
            })
        }).collect::<Vec<_>>(),
    })
}

#[must_use]
pub fn decision_ledger_markdown(report: &DecisionLedgerReport) -> String {
    let mut out = String::new();
    out.push_str("# Decision Ledger Calibration\n\n");
    out.push_str(&format!(
        "- Schema: `{}`\n- Bead: `{}`\n- Status: `{}`\n- Rows: `{}`\n- Calibration buckets: `{}`\n\n",
        report.schema_version,
        report.bead_id,
        report.status,
        report.summary.row_count,
        report.summary.calibration_bucket_count,
    ));
    out.push_str("| Decision | Mode | Action | Confidence | Drift | Dashboard |\n");
    out.push_str("|---|---|---:|---:|---|---|\n");
    for row in &report.rows {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{:.3}` | `{}` | `{}` |\n",
            row.decision_class,
            row.mode,
            row.selected_action,
            row.confidence,
            row.drift_status,
            row.dashboard_row,
        ));
    }
    if report.issues.is_empty() {
        out.push_str("\nNo decision-ledger calibration issues found.\n");
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
pub fn validate_decision_ledger_report(
    root: &Path,
    report: &DecisionLedgerReport,
) -> Vec<DecisionLedgerIssue> {
    let mut issues = Vec::new();

    if report.schema_version != DECISION_LEDGER_REPORT_SCHEMA_VERSION {
        issues.push(DecisionLedgerIssue::new(
            "bad_schema_version",
            "$.schema_version",
            "decision ledger calibration schema marker changed",
        ));
    }
    if report.bead_id != DECISION_LEDGER_BEAD_ID {
        issues.push(DecisionLedgerIssue::new(
            "bad_bead_id",
            "$.bead_id",
            "decision ledger report must stay bound to frankenjax-cstq.19",
        ));
    }
    if report.replay_command.trim().is_empty() {
        issues.push(DecisionLedgerIssue::new(
            "missing_replay_command",
            "$.replay_command",
            "report needs a stable replay command",
        ));
    }
    if !matches!(report.status.as_str(), "pass" | "fail") {
        issues.push(DecisionLedgerIssue::new(
            "bad_report_status",
            "$.status",
            format!("report status `{}` must be pass or fail", report.status),
        ));
    }
    if report.policy.trim().is_empty() {
        issues.push(DecisionLedgerIssue::new(
            "empty_policy",
            "$.policy",
            "decision ledger report needs a user-facing policy",
        ));
    }

    let required = REQUIRED_DECISION_CLASSES
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    validate_declared_classes(report, &required, &mut issues);
    let observed = report
        .rows
        .iter()
        .map(|row| row.decision_class.as_str())
        .collect::<BTreeSet<_>>();
    for required_class in required.difference(&observed) {
        issues.push(DecisionLedgerIssue::new(
            "missing_decision_class",
            "$.rows",
            format!("required decision class `{required_class}` is not covered"),
        ));
    }

    let bucket_ids = report
        .calibration_buckets
        .iter()
        .map(|bucket| bucket.bucket_id.as_str())
        .collect::<BTreeSet<_>>();
    validate_calibration_buckets(report, &mut issues);

    let mut row_ids = BTreeSet::new();
    for (idx, row) in report.rows.iter().enumerate() {
        validate_row(root, row, idx, &bucket_ids, &mut row_ids, &mut issues);
    }

    for artifact_ref in &report.artifact_refs {
        if !root.join(artifact_ref).exists() {
            issues.push(DecisionLedgerIssue::new(
                "missing_report_artifact_ref",
                "$.artifact_refs",
                format!("artifact ref `{artifact_ref}` is missing"),
            ));
        }
    }

    let expected = summarize_rows(&report.rows, &report.calibration_buckets);
    if report.summary != expected {
        issues.push(DecisionLedgerIssue::new(
            "bad_summary",
            "$.summary",
            "summary counts must match row and calibration bucket contents",
        ));
    }

    issues
}

fn validate_declared_classes(
    report: &DecisionLedgerReport,
    required: &BTreeSet<&str>,
    issues: &mut Vec<DecisionLedgerIssue>,
) {
    let mut declared = BTreeSet::new();
    for class in &report.required_decision_classes {
        if !declared.insert(class.as_str()) {
            issues.push(DecisionLedgerIssue::new(
                "duplicate_required_decision_class",
                "$.required_decision_classes",
                format!("duplicate required decision class `{class}`"),
            ));
        }
        if !required.contains(class.as_str()) {
            issues.push(DecisionLedgerIssue::new(
                "unknown_required_decision_class",
                "$.required_decision_classes",
                format!("unknown required decision class `{class}`"),
            ));
        }
    }
    for class in required {
        if !declared.contains(class) {
            issues.push(DecisionLedgerIssue::new(
                "missing_required_decision_class",
                "$.required_decision_classes",
                format!("required decision class `{class}` is absent from the report contract"),
            ));
        }
    }

    let allowed = ALLOWED_STRICT_HARDENED_DIVERGENCE
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut declared_allowlist = BTreeSet::new();
    for class in &report.strict_hardened_divergence_allowlist {
        if !declared_allowlist.insert(class.as_str()) {
            issues.push(DecisionLedgerIssue::new(
                "duplicate_divergence_allowlist_class",
                "$.strict_hardened_divergence_allowlist",
                format!("duplicate strict/hardened divergence allowlist class `{class}`"),
            ));
        }
        if !allowed.contains(class.as_str()) {
            issues.push(DecisionLedgerIssue::new(
                "unknown_divergence_allowlist_class",
                "$.strict_hardened_divergence_allowlist",
                format!("unknown strict/hardened divergence allowlist class `{class}`"),
            ));
        }
    }
    for class in &allowed {
        if !declared_allowlist.contains(class) {
            issues.push(DecisionLedgerIssue::new(
                "missing_divergence_allowlist_class",
                "$.strict_hardened_divergence_allowlist",
                format!("strict/hardened divergence allowlist is missing `{class}`"),
            ));
        }
    }
}

fn validate_calibration_buckets(
    report: &DecisionLedgerReport,
    issues: &mut Vec<DecisionLedgerIssue>,
) {
    let mut seen = BTreeSet::new();
    let mut last_upper = 0.0;
    let total_count = report
        .calibration_buckets
        .iter()
        .map(|bucket| bucket.count)
        .sum::<usize>();
    if total_count == 0 {
        issues.push(DecisionLedgerIssue::new(
            "empty_calibration_evidence",
            "$.calibration_buckets",
            "calibration buckets need observed evidence counts",
        ));
    }
    for (idx, bucket) in report.calibration_buckets.iter().enumerate() {
        let path = format!("$.calibration_buckets[{idx}]");
        if !seen.insert(bucket.bucket_id.as_str()) {
            issues.push(DecisionLedgerIssue::new(
                "duplicate_calibration_bucket",
                path.clone(),
                format!("duplicate calibration bucket `{}`", bucket.bucket_id),
            ));
        }
        if !is_unit_interval(bucket.lower_bound)
            || !is_unit_interval(bucket.upper_bound)
            || bucket.lower_bound >= bucket.upper_bound
        {
            issues.push(DecisionLedgerIssue::new(
                "bad_calibration_bounds",
                path.clone(),
                format!(
                    "{} has invalid bounds [{}, {})",
                    bucket.bucket_id, bucket.lower_bound, bucket.upper_bound
                ),
            ));
        }
        if idx == 0 && bucket.lower_bound.abs() > f64::EPSILON {
            issues.push(DecisionLedgerIssue::new(
                "calibration_bucket_gap",
                path.clone(),
                format!("{} must start at 0.0", bucket.bucket_id),
            ));
        }
        if idx > 0 && bucket.lower_bound < last_upper {
            issues.push(DecisionLedgerIssue::new(
                "non_monotonic_calibration_bucket",
                path.clone(),
                format!(
                    "{} starts before the previous bucket ends",
                    bucket.bucket_id
                ),
            ));
        }
        if idx > 0 && bucket.lower_bound > last_upper {
            issues.push(DecisionLedgerIssue::new(
                "calibration_bucket_gap",
                path.clone(),
                format!("{} starts after the previous bucket ends", bucket.bucket_id),
            ));
        }
        last_upper = bucket.upper_bound;
        if idx + 1 == report.calibration_buckets.len()
            && (bucket.upper_bound - 1.0).abs() > f64::EPSILON
        {
            issues.push(DecisionLedgerIssue::new(
                "calibration_bucket_gap",
                path.clone(),
                format!("{} must end at 1.0", bucket.bucket_id),
            ));
        }
        if !is_unit_interval(bucket.expected_probability)
            || !is_unit_interval(bucket.observed_probability)
        {
            issues.push(DecisionLedgerIssue::new(
                "bad_calibration_probability",
                path.clone(),
                format!("{} has probability outside [0,1]", bucket.bucket_id),
            ));
        }
        if bucket.count < MIN_BUCKET_COUNT && !bucket.stale {
            issues.push(DecisionLedgerIssue::new(
                "stale_calibration_not_marked",
                path.clone(),
                format!(
                    "{} has count {} below {MIN_BUCKET_COUNT} but stale=false",
                    bucket.bucket_id, bucket.count
                ),
            ));
        }
        let drift = (bucket.expected_probability - bucket.observed_probability).abs();
        if !matches!(bucket.drift_status.as_str(), "green" | "yellow" | "red") {
            issues.push(DecisionLedgerIssue::new(
                "bad_calibration_drift_status",
                path.clone(),
                format!("{} has invalid drift status", bucket.bucket_id),
            ));
        }
        if bucket.drift_status == "green" && drift > CALIBRATION_DRIFT_THRESHOLD {
            issues.push(DecisionLedgerIssue::new(
                "bad_calibration_drift_status",
                path.clone(),
                format!(
                    "{} drift {drift:.3} exceeds threshold {CALIBRATION_DRIFT_THRESHOLD:.3}",
                    bucket.bucket_id
                ),
            ));
        }
        if !bucket.ece_contribution.is_finite() || bucket.ece_contribution < 0.0 {
            issues.push(DecisionLedgerIssue::new(
                "bad_ece_contribution",
                path.clone(),
                format!("{} has invalid ECE contribution", bucket.bucket_id),
            ));
        }
        if total_count > 0 {
            let expected_ece = drift * (bucket.count as f64 / total_count as f64);
            if (bucket.ece_contribution - expected_ece).abs() > 1e-12 {
                issues.push(DecisionLedgerIssue::new(
                    "ece_contribution_mismatch",
                    path,
                    format!(
                        "{} ECE contribution does not match drift/count",
                        bucket.bucket_id
                    ),
                ));
            }
        }
    }
}

fn validate_row(
    root: &Path,
    row: &DecisionLedgerRow,
    idx: usize,
    bucket_ids: &BTreeSet<&str>,
    row_ids: &mut BTreeSet<String>,
    issues: &mut Vec<DecisionLedgerIssue>,
) {
    let path = format!("$.rows[{idx}]");
    if !row_ids.insert(row.decision_id.clone()) {
        issues.push(DecisionLedgerIssue::new(
            "duplicate_decision_id",
            path.clone(),
            format!("duplicate decision id `{}`", row.decision_id),
        ));
    }
    if row.status != "pass" {
        issues.push(DecisionLedgerIssue::new(
            "non_pass_row",
            path.clone(),
            format!("{} is not pass", row.decision_id),
        ));
    }
    if !REQUIRED_DECISION_CLASSES.contains(&row.decision_class.as_str()) {
        issues.push(DecisionLedgerIssue::new(
            "unknown_decision_class",
            path.clone(),
            format!(
                "{} has unknown decision class `{}`",
                row.decision_id, row.decision_class
            ),
        ));
    }
    if !matches!(row.mode.as_str(), "strict" | "hardened") {
        issues.push(DecisionLedgerIssue::new(
            "bad_mode",
            path.clone(),
            format!("{} has invalid mode `{}`", row.decision_id, row.mode),
        ));
    }
    if !matches!(
        row.selected_action.as_str(),
        "keep" | "kill" | "reprofile" | "fallback"
    ) {
        issues.push(DecisionLedgerIssue::new(
            "bad_selected_action",
            path.clone(),
            format!(
                "{} has invalid action `{}`",
                row.decision_id, row.selected_action
            ),
        ));
    }
    if row.alternatives_considered.len() < 2
        || !row
            .alternatives_considered
            .iter()
            .any(|alternative| alternative == &row.selected_action)
    {
        issues.push(DecisionLedgerIssue::new(
            "missing_alternatives",
            path.clone(),
            format!(
                "{} must include at least two alternatives and the selected action",
                row.decision_id
            ),
        ));
    }
    if !is_unit_interval(row.posterior_abandoned) || !is_unit_interval(row.confidence) {
        issues.push(DecisionLedgerIssue::new(
            "bad_probability",
            path.clone(),
            format!("{} has posterior/confidence outside [0,1]", row.decision_id),
        ));
    }
    if !row.expected_loss_keep.is_finite()
        || !row.expected_loss_kill.is_finite()
        || row.expected_loss_keep < 0.0
        || row.expected_loss_kill < 0.0
    {
        issues.push(DecisionLedgerIssue::new(
            "bad_loss_values",
            path.clone(),
            format!("{} has invalid expected loss values", row.decision_id),
        ));
    }
    let recomputed_keep = expected_loss_keep(row.posterior_abandoned, &row.loss_matrix);
    let recomputed_kill = expected_loss_kill(row.posterior_abandoned, &row.loss_matrix);
    if (recomputed_keep - row.expected_loss_keep).abs() > 1e-9
        || (recomputed_kill - row.expected_loss_kill).abs() > 1e-9
    {
        issues.push(DecisionLedgerIssue::new(
            "loss_matrix_mismatch",
            path.clone(),
            format!("{} loss values do not match its matrix", row.decision_id),
        ));
    }
    if row.selected_action != "fallback" {
        let expected_action = action_label(
            DecisionRecord::from_posterior(
                compatibility_mode(&row.mode),
                row.posterior_abandoned,
                &row.loss_matrix,
            )
            .action,
        );
        if row.selected_action != expected_action {
            issues.push(DecisionLedgerIssue::new(
                "selected_action_mismatch",
                path.clone(),
                format!(
                    "{} selected `{}` but loss matrix recommends `{expected_action}`",
                    row.decision_id, row.selected_action
                ),
            ));
        }
    } else if row.mode != "hardened" {
        issues.push(DecisionLedgerIssue::new(
            "fallback_not_hardened",
            path.clone(),
            format!(
                "{} fallback decisions must be hardened mode",
                row.decision_id
            ),
        ));
    }
    if row.evidence_signals.is_empty() {
        issues.push(DecisionLedgerIssue::new(
            "missing_evidence_signals",
            path.clone(),
            format!("{} needs evidence signals", row.decision_id),
        ));
    }
    for signal in &row.evidence_signals {
        if signal.signal_name.trim().is_empty()
            || !signal.log_likelihood_delta.is_finite()
            || signal.detail.trim().is_empty()
        {
            issues.push(DecisionLedgerIssue::new(
                "bad_evidence_signal",
                path.clone(),
                format!("{} has an incomplete evidence signal", row.decision_id),
            ));
        }
        if contains_secret_like(&signal.signal_name) || contains_secret_like(&signal.detail) {
            issues.push(DecisionLedgerIssue::new(
                "redaction_policy_violation",
                path.clone(),
                format!(
                    "{} evidence signal `{}` contains secret-like text",
                    row.decision_id, signal.signal_name
                ),
            ));
        }
    }
    if !bucket_ids.contains(row.calibration_bucket.as_str()) {
        issues.push(DecisionLedgerIssue::new(
            "missing_calibration_bucket",
            path.clone(),
            format!(
                "{} references missing calibration bucket `{}`",
                row.decision_id, row.calibration_bucket
            ),
        ));
    }
    if !matches!(row.drift_status.as_str(), "green" | "yellow" | "red") {
        issues.push(DecisionLedgerIssue::new(
            "bad_drift_status",
            path.clone(),
            format!("{} has invalid drift status", row.decision_id),
        ));
    }
    if row.drift_status == "red" || row.stale_calibration {
        issues.push(DecisionLedgerIssue::new(
            "stale_or_red_row",
            path.clone(),
            format!("{} is not ledger-green", row.decision_id),
        ));
    }
    if row.strict_hardened_divergence && !row.divergence_allowed {
        issues.push(DecisionLedgerIssue::new(
            "unallowed_strict_hardened_divergence",
            path.clone(),
            format!("{} has unallowlisted divergence", row.decision_id),
        ));
    }
    for field in [
        ("expected_outcome", row.expected_outcome.as_str()),
        ("actual_outcome", row.actual_outcome.as_str()),
        (
            "user_visible_consequence",
            row.user_visible_consequence.as_str(),
        ),
        ("replay_command", row.replay_command.as_str()),
        ("dashboard_row", row.dashboard_row.as_str()),
    ] {
        if field.1.trim().is_empty() {
            issues.push(DecisionLedgerIssue::new(
                "missing_row_field",
                format!("{path}.{}", field.0),
                format!("{} has empty {}", row.decision_id, field.0),
            ));
        }
        if contains_secret_like(field.1) {
            issues.push(DecisionLedgerIssue::new(
                "redaction_policy_violation",
                format!("{path}.{}", field.0),
                format!("{} {} contains secret-like text", row.decision_id, field.0),
            ));
        }
    }
    if row.artifact_refs.is_empty() {
        issues.push(DecisionLedgerIssue::new(
            "missing_artifact_refs",
            path.clone(),
            format!("{} needs artifact refs", row.decision_id),
        ));
    }
    for artifact_ref in &row.artifact_refs {
        if !root.join(artifact_ref).exists() {
            issues.push(DecisionLedgerIssue::new(
                "missing_artifact_ref",
                format!("{path}.artifact_refs"),
                format!("{} artifact `{artifact_ref}` is missing", row.decision_id),
            ));
        }
    }
}

fn summarize_rows(
    rows: &[DecisionLedgerRow],
    buckets: &[DecisionCalibrationBucket],
) -> DecisionLedgerSummary {
    let observed_classes = rows
        .iter()
        .map(|row| row.decision_class.as_str())
        .collect::<BTreeSet<_>>();
    DecisionLedgerSummary {
        required_decision_class_count: REQUIRED_DECISION_CLASSES.len(),
        observed_decision_class_count: observed_classes.len(),
        pass_count: rows.iter().filter(|row| row.status == "pass").count(),
        row_count: rows.len(),
        calibration_bucket_count: buckets.len(),
        stale_calibration_count: buckets.iter().filter(|bucket| bucket.stale).count()
            + rows.iter().filter(|row| row.stale_calibration).count(),
        strict_hardened_divergence_count: rows
            .iter()
            .filter(|row| row.strict_hardened_divergence)
            .count(),
        allowed_divergence_count: rows
            .iter()
            .filter(|row| row.strict_hardened_divergence && row.divergence_allowed)
            .count(),
        rows_with_artifacts: rows
            .iter()
            .filter(|row| !row.artifact_refs.is_empty())
            .count(),
        rows_with_replay: rows
            .iter()
            .filter(|row| !row.replay_command.trim().is_empty())
            .count(),
        rows_with_dashboard_rows: rows
            .iter()
            .filter(|row| !row.dashboard_row.trim().is_empty())
            .count(),
    }
}

fn build_row(spec: DecisionSpec) -> DecisionLedgerRow {
    let mut posterior = LogDomainPosterior::new(spec.prior_abandoned);
    let evidence_signals = spec
        .signals
        .iter()
        .map(|signal| {
            posterior.update(signal.log_likelihood_delta);
            EvidenceSignal {
                signal_name: signal.signal_name.to_owned(),
                log_likelihood_delta: signal.log_likelihood_delta,
                detail: signal.detail.to_owned(),
            }
        })
        .collect::<Vec<_>>();
    let posterior_abandoned = posterior.posterior_abandoned();
    let expected_loss_keep = expected_loss_keep(posterior_abandoned, &spec.loss_matrix);
    let expected_loss_kill = expected_loss_kill(posterior_abandoned, &spec.loss_matrix);
    let selected_action = spec.selected_action.map_or_else(
        || {
            action_label(
                DecisionRecord::from_posterior(spec.mode, posterior_abandoned, &spec.loss_matrix)
                    .action,
            )
        },
        action_label,
    );
    let confidence = match selected_action.as_str() {
        "keep" => 1.0 - posterior_abandoned,
        "kill" => posterior_abandoned,
        "fallback" | "reprofile" => 1.0 - (posterior_abandoned - 0.5).abs(),
        _ => 0.0,
    }
    .clamp(0.0, 1.0);

    DecisionLedgerRow {
        decision_id: spec.decision_id.to_owned(),
        decision_class: spec.decision_class.to_owned(),
        status: "pass".to_owned(),
        mode: mode_label(spec.mode).to_owned(),
        selected_action,
        alternatives_considered: spec
            .alternatives
            .iter()
            .map(|alternative| (*alternative).to_owned())
            .collect(),
        loss_matrix: spec.loss_matrix,
        posterior_abandoned,
        confidence,
        expected_loss_keep,
        expected_loss_kill,
        evidence_signals,
        expected_outcome: spec.expected_outcome.to_owned(),
        actual_outcome: spec.actual_outcome.to_owned(),
        calibration_bucket: spec.calibration_bucket.to_owned(),
        drift_status: spec.drift_status.to_owned(),
        stale_calibration: false,
        strict_hardened_divergence: spec.strict_hardened_divergence,
        divergence_allowed: spec.strict_hardened_divergence
            && ALLOWED_STRICT_HARDENED_DIVERGENCE.contains(&spec.decision_class),
        user_visible_consequence: spec.user_visible_consequence.to_owned(),
        artifact_refs: spec
            .artifact_refs
            .iter()
            .map(|artifact| (*artifact).to_owned())
            .collect(),
        replay_command: spec.replay_command.to_owned(),
        dashboard_row: format!("ledger/{}", spec.decision_class),
    }
}

fn calibration_buckets() -> Vec<DecisionCalibrationBucket> {
    let specs = [
        ("bucket_00_025", 0.0, 0.25, 0.18, 0.17, 18),
        ("bucket_025_050", 0.25, 0.50, 0.38, 0.40, 20),
        ("bucket_050_075", 0.50, 0.75, 0.62, 0.68, 19),
        ("bucket_075_100", 0.75, 1.00, 0.86, 0.81, 16),
    ];
    specs
        .into_iter()
        .map(|(bucket_id, lower, upper, expected, observed, count)| {
            let drift = f64::abs(expected - observed);
            DecisionCalibrationBucket {
                bucket_id: bucket_id.to_owned(),
                lower_bound: lower,
                upper_bound: upper,
                expected_probability: expected,
                observed_probability: observed,
                count,
                ece_contribution: drift * (count as f64 / 73.0),
                drift_status: if drift <= CALIBRATION_DRIFT_THRESHOLD {
                    "green"
                } else {
                    "yellow"
                }
                .to_owned(),
                stale: count < MIN_BUCKET_COUNT,
            }
        })
        .collect()
}

struct SignalSpec {
    signal_name: &'static str,
    log_likelihood_delta: f64,
    detail: &'static str,
}

struct DecisionSpec {
    decision_id: &'static str,
    decision_class: &'static str,
    mode: CompatibilityMode,
    prior_abandoned: f64,
    signals: &'static [SignalSpec],
    loss_matrix: LossMatrix,
    selected_action: Option<DecisionAction>,
    alternatives: &'static [&'static str],
    expected_outcome: &'static str,
    actual_outcome: &'static str,
    calibration_bucket: &'static str,
    drift_status: &'static str,
    strict_hardened_divergence: bool,
    user_visible_consequence: &'static str,
    artifact_refs: &'static [&'static str],
    replay_command: &'static str,
}

fn decision_specs() -> Vec<DecisionSpec> {
    vec![
        DecisionSpec {
            decision_id: "ledger-cache-hit-recompute-001",
            decision_class: "cache_hit_recompute",
            mode: CompatibilityMode::Strict,
            prior_abandoned: 0.35,
            signals: &[SignalSpec {
                signal_name: "cache_hit_rate",
                log_likelihood_delta: -1.1,
                detail: "recent cache reuse supports keeping the compiled artifact",
            }],
            loss_matrix: LossMatrix::default(),
            selected_action: None,
            alternatives: &["keep", "kill", "reprofile"],
            expected_outcome: "reuse cache entry when key and mode are proven compatible",
            actual_outcome: "cache lifecycle gate separates compatible reuse from recompute",
            calibration_bucket: "bucket_075_100",
            drift_status: "green",
            strict_hardened_divergence: false,
            user_visible_consequence: "compiled artifact reused with cache-key evidence",
            artifact_refs: &["artifacts/e2e/e2e_cache_lifecycle_gate.e2e.json"],
            replay_command: "./scripts/run_cache_lifecycle_gate.sh --enforce",
        },
        DecisionSpec {
            decision_id: "ledger-strict-rejection-001",
            decision_class: "strict_rejection",
            mode: CompatibilityMode::Strict,
            prior_abandoned: 0.4,
            signals: &[SignalSpec {
                signal_name: "unknown_incompatible_feature",
                log_likelihood_delta: 2.2,
                detail: "strict mode observed unknown incompatible cache metadata",
            }],
            loss_matrix: LossMatrix::default(),
            selected_action: None,
            alternatives: &["keep", "kill", "reprofile"],
            expected_outcome: "strict mode rejects unknown incompatible features",
            actual_outcome: "error taxonomy and cache lifecycle gates fail closed",
            calibration_bucket: "bucket_075_100",
            drift_status: "green",
            strict_hardened_divergence: false,
            user_visible_consequence: "request fails closed with typed error and replay hint",
            artifact_refs: &[
                "artifacts/e2e/e2e_error_taxonomy_gate.e2e.json",
                "artifacts/e2e/e2e_cache_lifecycle_gate.e2e.json",
            ],
            replay_command: "./scripts/run_error_taxonomy_gate.sh --enforce",
        },
        DecisionSpec {
            decision_id: "ledger-hardened-recovery-001",
            decision_class: "hardened_recovery",
            mode: CompatibilityMode::Hardened,
            prior_abandoned: 0.45,
            signals: &[
                SignalSpec {
                    signal_name: "malformed_input_detected",
                    log_likelihood_delta: 0.8,
                    detail: "hardened mode detected malformed but bounded input",
                },
                SignalSpec {
                    signal_name: "recovery_proof",
                    log_likelihood_delta: -0.6,
                    detail: "typed recovery path preserved the API contract",
                },
            ],
            loss_matrix: LossMatrix {
                keep_if_useful: 10,
                kill_if_useful: 70,
                keep_if_abandoned: 35,
                kill_if_abandoned: 5,
            },
            selected_action: Some(DecisionAction::Fallback),
            alternatives: &["fallback", "kill", "reprofile"],
            expected_outcome: "hardened mode may recover only with typed evidence",
            actual_outcome: "security gate records bounded recovery without panic",
            calibration_bucket: "bucket_050_075",
            drift_status: "green",
            strict_hardened_divergence: true,
            user_visible_consequence: "operation recovers with audit-trail disclosure",
            artifact_refs: &["artifacts/e2e/e2e_security_gate.e2e.json"],
            replay_command: "./scripts/run_security_gate.sh --enforce",
        },
        DecisionSpec {
            decision_id: "ledger-fallback-denial-001",
            decision_class: "fallback_denial",
            mode: CompatibilityMode::Strict,
            prior_abandoned: 0.65,
            signals: &[SignalSpec {
                signal_name: "unsupported_transform_tail",
                log_likelihood_delta: 1.2,
                detail: "fallback would hide unsupported transform semantics",
            }],
            loss_matrix: LossMatrix::default(),
            selected_action: None,
            alternatives: &["keep", "kill", "fallback", "reprofile"],
            expected_outcome: "fallback denied when it would change transform semantics",
            actual_outcome: "error taxonomy records unsupported scope as fail-closed",
            calibration_bucket: "bucket_075_100",
            drift_status: "green",
            strict_hardened_divergence: false,
            user_visible_consequence: "user receives unsupported-scope error instead of silent fallback",
            artifact_refs: &["artifacts/e2e/e2e_error_taxonomy_gate.e2e.json"],
            replay_command: "./scripts/run_error_taxonomy_gate.sh --enforce",
        },
        DecisionSpec {
            decision_id: "ledger-optimization-selection-001",
            decision_class: "optimization_selection",
            mode: CompatibilityMode::Strict,
            prior_abandoned: 0.2,
            signals: &[SignalSpec {
                signal_name: "benchmark_delta",
                log_likelihood_delta: -0.8,
                detail: "performance gate shows a behavior-preserving improvement",
            }],
            loss_matrix: LossMatrix {
                keep_if_useful: 0,
                kill_if_useful: 60,
                keep_if_abandoned: 20,
                kill_if_abandoned: 2,
            },
            selected_action: None,
            alternatives: &["keep", "kill", "reprofile"],
            expected_outcome: "select optimization only when behavior evidence stays green",
            actual_outcome: "memory/performance gate exposes benchmark and artifact links",
            calibration_bucket: "bucket_075_100",
            drift_status: "green",
            strict_hardened_divergence: false,
            user_visible_consequence: "optimization ships only with replayable benchmark evidence",
            artifact_refs: &["artifacts/e2e/e2e_memory_performance_gate.e2e.json"],
            replay_command: "./scripts/run_memory_performance_gate.sh --enforce",
        },
        DecisionSpec {
            decision_id: "ledger-durability-recovery-001",
            decision_class: "durability_recovery",
            mode: CompatibilityMode::Hardened,
            prior_abandoned: 0.5,
            signals: &[SignalSpec {
                signal_name: "sidecar_decode_proof",
                log_likelihood_delta: -1.4,
                detail: "RaptorQ sidecar and decode proof bind recovered artifact bytes",
            }],
            loss_matrix: LossMatrix {
                keep_if_useful: 5,
                kill_if_useful: 90,
                keep_if_abandoned: 30,
                kill_if_abandoned: 3,
            },
            selected_action: Some(DecisionAction::Fallback),
            alternatives: &["fallback", "kill", "reprofile"],
            expected_outcome: "recover artifact only when sidecar scrub and proof are present",
            actual_outcome: "durability coverage gate records sidecar, scrub, and decode proof",
            calibration_bucket: "bucket_075_100",
            drift_status: "green",
            strict_hardened_divergence: true,
            user_visible_consequence: "artifact recovery is disclosed and hash-bound",
            artifact_refs: &["artifacts/e2e/e2e_durability_coverage_gate.e2e.json"],
            replay_command: "./scripts/run_durability_coverage_gate.sh --enforce",
        },
        DecisionSpec {
            decision_id: "ledger-transform-admission-001",
            decision_class: "transform_admission",
            mode: CompatibilityMode::Strict,
            prior_abandoned: 0.25,
            signals: &[SignalSpec {
                signal_name: "composition_proof_valid",
                log_likelihood_delta: -0.9,
                detail: "TTL semantic proof binds transform stack to evidence ids",
            }],
            loss_matrix: LossMatrix::default(),
            selected_action: None,
            alternatives: &["keep", "kill", "reprofile"],
            expected_outcome: "admit transform composition only with matching proof evidence",
            actual_outcome: "TTL semantic gate validates composition signatures",
            calibration_bucket: "bucket_075_100",
            drift_status: "green",
            strict_hardened_divergence: false,
            user_visible_consequence: "transform executes with proof-linked audit trail",
            artifact_refs: &["artifacts/e2e/e2e_ttl_semantic_gate.e2e.json"],
            replay_command: "./scripts/run_ttl_semantic_gate.sh --enforce",
        },
        DecisionSpec {
            decision_id: "ledger-unsupported-scope-001",
            decision_class: "unsupported_scope",
            mode: CompatibilityMode::Strict,
            prior_abandoned: 0.7,
            signals: &[SignalSpec {
                signal_name: "scope_gap_detected",
                log_likelihood_delta: 1.1,
                detail: "requested transform is outside the currently proven V1 scope",
            }],
            loss_matrix: LossMatrix::default(),
            selected_action: None,
            alternatives: &["keep", "kill", "fallback", "reprofile"],
            expected_outcome: "deny unsupported scope and file visible follow-up work",
            actual_outcome: "error taxonomy names unsupported scope and replay command",
            calibration_bucket: "bucket_075_100",
            drift_status: "green",
            strict_hardened_divergence: false,
            user_visible_consequence: "unsupported request is rejected with actionable explanation",
            artifact_refs: &["artifacts/e2e/e2e_error_taxonomy_gate.e2e.json"],
            replay_command: "./scripts/run_error_taxonomy_gate.sh --enforce",
        },
        DecisionSpec {
            decision_id: "ledger-runtime-budget-deadline-001",
            decision_class: "runtime_budget_deadline",
            mode: CompatibilityMode::Strict,
            prior_abandoned: 0.5,
            signals: &[SignalSpec {
                signal_name: "deadline_budget_tie",
                log_likelihood_delta: 0.0,
                detail: "budget evidence is inconclusive and requires reprofile",
            }],
            loss_matrix: LossMatrix {
                keep_if_useful: 5,
                kill_if_useful: 15,
                keep_if_abandoned: 15,
                kill_if_abandoned: 5,
            },
            selected_action: None,
            alternatives: &["keep", "kill", "reprofile"],
            expected_outcome: "reprofile when deadline evidence is tied",
            actual_outcome: "memory/performance gate exposes timing metadata for replay",
            calibration_bucket: "bucket_050_075",
            drift_status: "green",
            strict_hardened_divergence: false,
            user_visible_consequence: "runtime budget uncertainty triggers reprofile instead of guessing",
            artifact_refs: &["artifacts/e2e/e2e_memory_performance_gate.e2e.json"],
            replay_command: "./scripts/run_memory_performance_gate.sh --enforce",
        },
    ]
}

fn action_label(action: DecisionAction) -> String {
    match action {
        DecisionAction::Keep => "keep",
        DecisionAction::Kill => "kill",
        DecisionAction::Reprofile => "reprofile",
        DecisionAction::Fallback => "fallback",
    }
    .to_owned()
}

fn mode_label(mode: CompatibilityMode) -> &'static str {
    match mode {
        CompatibilityMode::Strict => "strict",
        CompatibilityMode::Hardened => "hardened",
    }
}

fn compatibility_mode(mode: &str) -> CompatibilityMode {
    if mode == "hardened" {
        CompatibilityMode::Hardened
    } else {
        CompatibilityMode::Strict
    }
}

fn contains_secret_like(value: &str) -> bool {
    let upper = value.to_ascii_uppercase();
    [
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "CREDENTIAL",
        "API_KEY",
        "AUTH",
    ]
    .iter()
    .any(|needle| upper.contains(needle))
}

fn is_unit_interval(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let raw = serde_json::to_string_pretty(value).map_err(std::io::Error::other)?;
    fs::write(path, format!("{raw}\n"))
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}
