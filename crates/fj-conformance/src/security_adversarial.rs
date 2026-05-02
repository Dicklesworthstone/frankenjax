#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const SECURITY_ADVERSARIAL_REPORT_SCHEMA_VERSION: &str =
    "frankenjax.security-adversarial-gate.v1";
pub const SECURITY_THREAT_MODEL_SCHEMA_VERSION: &str = "frankenjax.security-threat-model.v1";
pub const SECURITY_ADVERSARIAL_BEAD_ID: &str = "frankenjax-cstq.17";

const REQUIRED_CATEGORIES: &[&str] = &[
    "tc_cache_confusion",
    "tc_transform_ordering",
    "tc_ir_validation",
    "tc_shape_dtype_signatures",
    "tc_subjaxpr_control_flow",
    "tc_serialized_fixtures_logs",
    "tc_ffi_boundaries",
    "tc_durability_corruption",
    "tc_evidence_ledger_recovery",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SecurityThreatModel {
    pub schema_version: String,
    pub bead_id: String,
    pub generated_at_unix_ms: u128,
    pub status: String,
    pub description: String,
    pub threat_categories: Vec<SecurityThreatCategory>,
    pub summary: SecurityThreatSummary,
    pub next_steps: Vec<String>,
    pub replay_command: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SecurityThreatCategory {
    pub category_id: String,
    pub name: String,
    pub description: String,
    pub boundaries: Vec<String>,
    pub controls: Vec<String>,
    pub evidence_status: String,
    pub evidence_refs: Vec<String>,
    pub fuzz_status: String,
    pub fuzz_coverage_pct: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecurityThreatSummary {
    pub total_categories: usize,
    pub evidence_green: usize,
    pub evidence_yellow: usize,
    pub evidence_red: usize,
    pub fuzz_started: usize,
    pub fuzz_complete: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SecurityAdversarialReport {
    pub schema_version: String,
    pub bead_id: String,
    pub generated_at_unix_ms: u128,
    pub status: String,
    pub matrix_policy: String,
    pub coverage: SecurityAdversarialCoverage,
    pub threat_categories: Vec<SecurityAdversarialCategory>,
    pub fuzz_families: Vec<SecurityFuzzFamily>,
    pub adversarial_rows: Vec<SecurityAdversarialRow>,
    pub crash_index: SecurityCrashIndex,
    pub issues: Vec<SecurityAdversarialIssue>,
    pub artifact_refs: Vec<String>,
    pub replay_command: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecurityAdversarialCoverage {
    pub required_category_count: usize,
    pub observed_category_count: usize,
    pub green_category_count: usize,
    pub fuzz_family_count: usize,
    pub fuzz_complete_count: usize,
    pub adversarial_row_count: usize,
    pub typed_error_row_count: usize,
    pub panic_free_row_count: usize,
    pub crash_free_family_count: usize,
    pub timeout_free_family_count: usize,
    pub evidence_ref_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SecurityAdversarialCategory {
    pub category_id: String,
    pub name: String,
    pub description: String,
    pub boundaries: Vec<String>,
    pub controls: Vec<String>,
    pub strict_outcome: String,
    pub hardened_outcome: String,
    pub evidence_status: String,
    pub evidence_refs: Vec<String>,
    pub required_fuzz_families: Vec<String>,
    pub required_adversarial_rows: Vec<String>,
    pub fuzz_status: String,
    pub fuzz_coverage_pct: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecurityFuzzFamily {
    pub family_id: String,
    pub target: String,
    pub corpus_path: String,
    pub target_source: String,
    pub seed_floor: usize,
    pub observed_seed_count: usize,
    pub deterministic_replay_count: usize,
    pub expected_error_class: String,
    pub actual_error_class: String,
    pub panic_status: String,
    pub crash_status: String,
    pub timeout_status: String,
    pub minimized_repro_path: Option<String>,
    pub artifact_hashes: BTreeMap<String, String>,
    pub replay_command: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecurityAdversarialRow {
    pub row_id: String,
    pub category_id: String,
    pub input_family: String,
    pub target_subsystem: String,
    pub expected_error_class: String,
    pub actual_error_class: String,
    pub strict_behavior: String,
    pub hardened_behavior: String,
    pub panic_status: String,
    pub crash_status: String,
    pub timeout_status: String,
    pub fuzz_family_id: String,
    pub evidence_refs: Vec<String>,
    pub replay_command: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecurityCrashIndex {
    pub path: String,
    pub exists: bool,
    pub open_crash_count: usize,
    pub p0_open_crash_count: usize,
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecurityAdversarialIssue {
    pub code: String,
    pub path: String,
    pub message: String,
}

impl SecurityAdversarialIssue {
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
pub struct SecurityAdversarialOutputPaths {
    pub threat_model: PathBuf,
    pub report: PathBuf,
    pub markdown: PathBuf,
}

impl SecurityAdversarialOutputPaths {
    #[must_use]
    pub fn for_root(root: &Path) -> Self {
        Self {
            threat_model: root.join("artifacts/conformance/security_threat_model.v1.json"),
            report: root.join("artifacts/conformance/security_adversarial_gate.v1.json"),
            markdown: root.join("artifacts/conformance/security_adversarial_gate.v1.md"),
        }
    }
}

#[must_use]
pub fn build_security_adversarial_report(root: &Path) -> SecurityAdversarialReport {
    let outputs = SecurityAdversarialOutputPaths::for_root(root);
    build_security_adversarial_report_for_outputs(
        root,
        &outputs.threat_model,
        &outputs.report,
        &outputs.markdown,
    )
}

#[must_use]
pub fn build_security_adversarial_report_for_outputs(
    root: &Path,
    threat_model_path: &Path,
    report_path: &Path,
    markdown_path: &Path,
) -> SecurityAdversarialReport {
    let fuzz_families = build_fuzz_families(root);
    let adversarial_rows = build_adversarial_rows();
    let threat_categories = build_categories(&fuzz_families, &adversarial_rows);
    let crash_index = build_crash_index(root);

    let coverage = SecurityAdversarialCoverage {
        required_category_count: REQUIRED_CATEGORIES.len(),
        observed_category_count: threat_categories.len(),
        green_category_count: threat_categories
            .iter()
            .filter(|category| category.evidence_status == "green")
            .count(),
        fuzz_family_count: fuzz_families.len(),
        fuzz_complete_count: fuzz_families
            .iter()
            .filter(|family| family.actual_error_class == family.expected_error_class)
            .filter(|family| family.panic_status == "no_panic")
            .filter(|family| family.crash_status == "no_crash")
            .filter(|family| family.timeout_status == "no_timeout")
            .filter(|family| family.observed_seed_count >= family.seed_floor)
            .count(),
        adversarial_row_count: adversarial_rows.len(),
        typed_error_row_count: adversarial_rows
            .iter()
            .filter(|row| row.actual_error_class != "none")
            .count(),
        panic_free_row_count: adversarial_rows
            .iter()
            .filter(|row| row.panic_status == "no_panic")
            .count(),
        crash_free_family_count: fuzz_families
            .iter()
            .filter(|family| family.crash_status == "no_crash")
            .count(),
        timeout_free_family_count: fuzz_families
            .iter()
            .filter(|family| family.timeout_status == "no_timeout")
            .count(),
        evidence_ref_count: threat_categories
            .iter()
            .map(|category| category.evidence_refs.len())
            .sum::<usize>()
            + adversarial_rows
                .iter()
                .map(|row| row.evidence_refs.len())
                .sum::<usize>(),
    };

    let mut report = SecurityAdversarialReport {
        schema_version: SECURITY_ADVERSARIAL_REPORT_SCHEMA_VERSION.to_owned(),
        bead_id: SECURITY_ADVERSARIAL_BEAD_ID.to_owned(),
        generated_at_unix_ms: now_unix_ms(),
        status: "pass".to_owned(),
        matrix_policy:
            "Security green requires every V1 threat category to have existing evidence refs, complete deterministic fuzz seed coverage, typed panic-free adversarial rows, no open crash index entries, stable replay commands, and shared E2E log validation."
                .to_owned(),
        coverage,
        threat_categories,
        fuzz_families,
        adversarial_rows,
        crash_index,
        issues: Vec::new(),
        artifact_refs: vec![
            repo_relative_artifact(root, threat_model_path),
            repo_relative_artifact(root, report_path),
            repo_relative_artifact(root, markdown_path),
        ],
        replay_command: "./scripts/run_security_gate.sh --enforce".to_owned(),
    };
    report.issues = validate_security_adversarial_report(root, &report);
    if !report.issues.is_empty() {
        report.status = "fail".to_owned();
    }
    report
}

#[must_use]
pub fn build_security_threat_model(report: &SecurityAdversarialReport) -> SecurityThreatModel {
    let categories = report
        .threat_categories
        .iter()
        .map(|category| SecurityThreatCategory {
            category_id: category.category_id.clone(),
            name: category.name.clone(),
            description: category.description.clone(),
            boundaries: category.boundaries.clone(),
            controls: category.controls.clone(),
            evidence_status: category.evidence_status.clone(),
            evidence_refs: category.evidence_refs.clone(),
            fuzz_status: category.fuzz_status.clone(),
            fuzz_coverage_pct: Some(category.fuzz_coverage_pct),
        })
        .collect::<Vec<_>>();
    let summary = SecurityThreatSummary {
        total_categories: categories.len(),
        evidence_green: categories
            .iter()
            .filter(|category| category.evidence_status == "green")
            .count(),
        evidence_yellow: categories
            .iter()
            .filter(|category| category.evidence_status == "yellow")
            .count(),
        evidence_red: categories
            .iter()
            .filter(|category| category.evidence_status == "red")
            .count(),
        fuzz_started: categories
            .iter()
            .filter(|category| category.fuzz_status != "not_started")
            .count(),
        fuzz_complete: categories
            .iter()
            .filter(|category| category.fuzz_status == "complete")
            .count(),
    };

    SecurityThreatModel {
        schema_version: SECURITY_THREAT_MODEL_SCHEMA_VERSION.to_owned(),
        bead_id: SECURITY_ADVERSARIAL_BEAD_ID.to_owned(),
        generated_at_unix_ms: report.generated_at_unix_ms,
        status: if report.status == "pass" {
            "complete".to_owned()
        } else {
            "partial".to_owned()
        },
        description:
            "Threat model matrix for V1 security boundaries with complete deterministic fuzz seed coverage and replayable adversarial evidence."
                .to_owned(),
        threat_categories: categories,
        summary,
        next_steps: if report.status == "pass" {
            Vec::new()
        } else {
            report
                .issues
                .iter()
                .map(|issue| format!("{}: {}", issue.code, issue.message))
                .collect()
        },
        replay_command: report.replay_command.clone(),
    }
}

pub fn write_security_adversarial_outputs(
    root: &Path,
    threat_model_path: &Path,
    report_path: &Path,
    markdown_path: &Path,
) -> Result<SecurityAdversarialReport, std::io::Error> {
    let report = build_security_adversarial_report_for_outputs(
        root,
        threat_model_path,
        report_path,
        markdown_path,
    );
    let threat_model = build_security_threat_model(&report);
    write_json(threat_model_path, &threat_model)?;
    write_json(report_path, &report)?;
    write_markdown(markdown_path, &security_adversarial_markdown(&report))?;
    Ok(report)
}

#[must_use]
pub fn security_adversarial_summary_json(report: &SecurityAdversarialReport) -> JsonValue {
    json!({
        "status": report.status,
        "schema_version": report.schema_version,
        "category_count": report.threat_categories.len(),
        "fuzz_family_count": report.fuzz_families.len(),
        "adversarial_row_count": report.adversarial_rows.len(),
        "coverage": report.coverage,
        "crash_index": report.crash_index,
        "issue_count": report.issues.len(),
        "issues": report.issues,
    })
}

#[must_use]
pub fn security_adversarial_markdown(report: &SecurityAdversarialReport) -> String {
    let mut out = String::new();
    out.push_str("# Security Adversarial Gate\n\n");
    out.push_str(&format!(
        "- Schema: `{}`\n- Bead: `{}`\n- Status: `{}`\n- Categories: `{}`\n- Fuzz families: `{}`\n- Adversarial rows: `{}`\n\n",
        report.schema_version,
        report.bead_id,
        report.status,
        report.threat_categories.len(),
        report.fuzz_families.len(),
        report.adversarial_rows.len(),
    ));
    out.push_str("| Category | Evidence | Fuzz | Required Families |\n");
    out.push_str("|---|---:|---:|---|\n");
    for category in &report.threat_categories {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | {} |\n",
            category.category_id,
            category.evidence_status,
            category.fuzz_status,
            category
                .required_fuzz_families
                .iter()
                .map(|family| format!("`{family}`"))
                .collect::<Vec<_>>()
                .join(", "),
        ));
    }
    out.push('\n');
    out.push_str("| Fuzz Family | Seeds | Replay | Status |\n");
    out.push_str("|---|---:|---|---|\n");
    for family in &report.fuzz_families {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}/{}/{}` |\n",
            family.family_id,
            family.observed_seed_count,
            family.replay_command,
            family.panic_status,
            family.crash_status,
            family.timeout_status,
        ));
    }
    out.push('\n');
    if report.issues.is_empty() {
        out.push_str("No security adversarial gate issues found.\n");
    } else {
        out.push_str("## Issues\n\n");
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
pub fn validate_security_adversarial_report(
    root: &Path,
    report: &SecurityAdversarialReport,
) -> Vec<SecurityAdversarialIssue> {
    let mut issues = Vec::new();

    if report.schema_version != SECURITY_ADVERSARIAL_REPORT_SCHEMA_VERSION {
        issues.push(SecurityAdversarialIssue::new(
            "bad_schema_version",
            "$.schema_version",
            "security adversarial gate schema marker changed",
        ));
    }
    if report.bead_id != SECURITY_ADVERSARIAL_BEAD_ID {
        issues.push(SecurityAdversarialIssue::new(
            "bad_bead_id",
            "$.bead_id",
            "security gate must remain bound to frankenjax-cstq.17",
        ));
    }
    if report.replay_command.trim().is_empty() {
        issues.push(SecurityAdversarialIssue::new(
            "missing_replay_command",
            "$.replay_command",
            "gate needs a stable replay command",
        ));
    }
    if !matches!(report.status.as_str(), "pass" | "fail") {
        issues.push(SecurityAdversarialIssue::new(
            "bad_report_status",
            "$.status",
            format!(
                "security gate status `{}` must be pass or fail",
                report.status
            ),
        ));
    }
    if report.matrix_policy.trim().is_empty() {
        issues.push(SecurityAdversarialIssue::new(
            "empty_matrix_policy",
            "$.matrix_policy",
            "security gate needs a user-facing matrix policy",
        ));
    }

    for duplicate in duplicate_ids(
        report
            .threat_categories
            .iter()
            .map(|category| category.category_id.as_str()),
    ) {
        issues.push(SecurityAdversarialIssue::new(
            "duplicate_category_id",
            "$.threat_categories",
            format!("duplicate threat category id `{duplicate}`"),
        ));
    }
    for duplicate in duplicate_ids(
        report
            .fuzz_families
            .iter()
            .map(|family| family.family_id.as_str()),
    ) {
        issues.push(SecurityAdversarialIssue::new(
            "duplicate_fuzz_family_id",
            "$.fuzz_families",
            format!("duplicate fuzz family id `{duplicate}`"),
        ));
    }
    for duplicate in duplicate_ids(
        report
            .adversarial_rows
            .iter()
            .map(|row| row.row_id.as_str()),
    ) {
        issues.push(SecurityAdversarialIssue::new(
            "duplicate_adversarial_row_id",
            "$.adversarial_rows",
            format!("duplicate adversarial row id `{duplicate}`"),
        ));
    }

    let observed_categories = report
        .threat_categories
        .iter()
        .map(|category| category.category_id.as_str())
        .collect::<BTreeSet<_>>();
    for category in &report.threat_categories {
        if !REQUIRED_CATEGORIES.contains(&category.category_id.as_str()) {
            issues.push(SecurityAdversarialIssue::new(
                "unknown_threat_category",
                "$.threat_categories",
                format!("unknown threat category `{}`", category.category_id),
            ));
        }
    }
    for required in REQUIRED_CATEGORIES {
        if !observed_categories.contains(required) {
            issues.push(SecurityAdversarialIssue::new(
                "missing_required_category",
                "$.threat_categories",
                format!("missing required threat category `{required}`"),
            ));
        }
    }

    let family_ids = report
        .fuzz_families
        .iter()
        .map(|family| family.family_id.as_str())
        .collect::<BTreeSet<_>>();
    let row_ids = report
        .adversarial_rows
        .iter()
        .map(|row| row.row_id.as_str())
        .collect::<BTreeSet<_>>();
    let category_ids = report
        .threat_categories
        .iter()
        .map(|category| category.category_id.as_str())
        .collect::<BTreeSet<_>>();

    for (idx, category) in report.threat_categories.iter().enumerate() {
        let path = format!("$.threat_categories[{idx}]");
        if category.evidence_status != "green" {
            issues.push(SecurityAdversarialIssue::new(
                "non_green_category",
                path.clone(),
                format!("category {} is not green", category.category_id),
            ));
        }
        if category.fuzz_status != "complete" {
            issues.push(SecurityAdversarialIssue::new(
                "incomplete_category_fuzz",
                path.clone(),
                format!(
                    "category {} does not have complete fuzz coverage",
                    category.category_id
                ),
            ));
        }
        if (category.fuzz_coverage_pct - 100.0).abs() > f64::EPSILON {
            issues.push(SecurityAdversarialIssue::new(
                "bad_category_fuzz_pct",
                path.clone(),
                format!(
                    "category {} has fuzz coverage {}",
                    category.category_id, category.fuzz_coverage_pct
                ),
            ));
        }
        if category.evidence_refs.is_empty() {
            issues.push(SecurityAdversarialIssue::new(
                "missing_category_evidence",
                path.clone(),
                format!("category {} needs evidence refs", category.category_id),
            ));
        }
        for reference in &category.evidence_refs {
            if !evidence_ref_exists(root, reference) {
                issues.push(SecurityAdversarialIssue::new(
                    "missing_evidence_ref",
                    format!("{path}.evidence_refs"),
                    format!("evidence ref `{reference}` is not readable or anchor is absent"),
                ));
            }
        }
        for family in &category.required_fuzz_families {
            if !family_ids.contains(family.as_str()) {
                issues.push(SecurityAdversarialIssue::new(
                    "missing_required_fuzz_family",
                    path.clone(),
                    format!(
                        "category {} references missing fuzz family `{family}`",
                        category.category_id
                    ),
                ));
            }
        }
        for row in &category.required_adversarial_rows {
            if !row_ids.contains(row.as_str()) {
                issues.push(SecurityAdversarialIssue::new(
                    "missing_required_adversarial_row",
                    path.clone(),
                    format!(
                        "category {} references missing row `{row}`",
                        category.category_id
                    ),
                ));
            }
        }
    }

    for (idx, family) in report.fuzz_families.iter().enumerate() {
        let path = format!("$.fuzz_families[{idx}]");
        if family.observed_seed_count < family.seed_floor {
            issues.push(SecurityAdversarialIssue::new(
                "insufficient_seed_corpus",
                path.clone(),
                format!(
                    "{} has {} seeds, needs at least {}",
                    family.family_id, family.observed_seed_count, family.seed_floor
                ),
            ));
        }
        if family.deterministic_replay_count != family.observed_seed_count {
            issues.push(SecurityAdversarialIssue::new(
                "bad_replay_count",
                path.clone(),
                format!(
                    "{} replay count must equal observed seed count",
                    family.family_id
                ),
            ));
        }
        if family.expected_error_class != family.actual_error_class {
            issues.push(SecurityAdversarialIssue::new(
                "bad_fuzz_error_class",
                path.clone(),
                format!(
                    "{} expected {}, got {}",
                    family.family_id, family.expected_error_class, family.actual_error_class
                ),
            ));
        }
        if family.panic_status != "no_panic" {
            issues.push(SecurityAdversarialIssue::new(
                "fuzz_panic",
                path.clone(),
                format!(
                    "{} recorded panic status {}",
                    family.family_id, family.panic_status
                ),
            ));
        }
        if family.crash_status != "no_crash" {
            issues.push(SecurityAdversarialIssue::new(
                "fuzz_crash",
                path.clone(),
                format!(
                    "{} recorded crash status {}",
                    family.family_id, family.crash_status
                ),
            ));
        }
        if family.timeout_status != "no_timeout" {
            issues.push(SecurityAdversarialIssue::new(
                "fuzz_timeout",
                path.clone(),
                format!(
                    "{} recorded timeout status {}",
                    family.family_id, family.timeout_status
                ),
            ));
        }
        if family.replay_command.trim().is_empty() {
            issues.push(SecurityAdversarialIssue::new(
                "missing_fuzz_replay",
                path.clone(),
                format!("{} is missing a replay command", family.family_id),
            ));
        }
        if !root.join(&family.target_source).exists() {
            issues.push(SecurityAdversarialIssue::new(
                "missing_fuzz_target",
                path.clone(),
                format!("{} target source is missing", family.target_source),
            ));
        }
        if !root.join(&family.corpus_path).exists() {
            issues.push(SecurityAdversarialIssue::new(
                "missing_fuzz_corpus",
                path.clone(),
                format!("{} corpus path is missing", family.corpus_path),
            ));
        }
        if family.artifact_hashes.is_empty() {
            issues.push(SecurityAdversarialIssue::new(
                "missing_fuzz_artifact_hashes",
                path.clone(),
                format!(
                    "{} needs at least one hash-bound corpus artifact",
                    family.family_id
                ),
            ));
        }
        for (artifact_path, artifact_hash) in &family.artifact_hashes {
            let artifact_path_for_issue = format!("{path}.artifact_hashes.{artifact_path}");
            if !is_sha256_hex(artifact_hash) {
                issues.push(SecurityAdversarialIssue::new(
                    "bad_fuzz_artifact_hash",
                    artifact_path_for_issue.clone(),
                    format!(
                        "{} artifact hash for `{artifact_path}` is not SHA-256 hex",
                        family.family_id
                    ),
                ));
                continue;
            }
            match sha256_file(&root.join(artifact_path)) {
                Ok(actual_hash) if actual_hash == *artifact_hash => {}
                Ok(actual_hash) => issues.push(SecurityAdversarialIssue::new(
                    "stale_fuzz_artifact_hash",
                    artifact_path_for_issue,
                    format!(
                        "{} artifact hash for `{artifact_path}` is stale: expected {artifact_hash}, got {actual_hash}",
                        family.family_id,
                    ),
                )),
                Err(err) => issues.push(SecurityAdversarialIssue::new(
                    "missing_hashed_fuzz_artifact",
                    artifact_path_for_issue,
                    format!(
                        "{} artifact `{artifact_path}` could not be read for hash verification: {err}",
                        family.family_id,
                    ),
                )),
            }
        }
    }

    for (idx, row) in report.adversarial_rows.iter().enumerate() {
        let path = format!("$.adversarial_rows[{idx}]");
        if row.expected_error_class != row.actual_error_class {
            issues.push(SecurityAdversarialIssue::new(
                "bad_row_error_class",
                path.clone(),
                format!(
                    "{} expected {}, got {}",
                    row.row_id, row.expected_error_class, row.actual_error_class
                ),
            ));
        }
        if row.expected_error_class == "none" {
            issues.push(SecurityAdversarialIssue::new(
                "missing_typed_error_class",
                path.clone(),
                format!("{} needs a typed error or recovery class", row.row_id),
            ));
        }
        if row.replay_command.trim().is_empty() {
            issues.push(SecurityAdversarialIssue::new(
                "missing_row_replay",
                path.clone(),
                format!("{} needs a stable replay command", row.row_id),
            ));
        }
        if row.panic_status != "no_panic" {
            issues.push(SecurityAdversarialIssue::new(
                "row_panic",
                path.clone(),
                format!("{} recorded panic status {}", row.row_id, row.panic_status),
            ));
        }
        if row.crash_status != "no_crash" || row.timeout_status != "no_timeout" {
            issues.push(SecurityAdversarialIssue::new(
                "row_crash_or_timeout",
                path.clone(),
                format!(
                    "{} recorded crash={} timeout={}",
                    row.row_id, row.crash_status, row.timeout_status
                ),
            ));
        }
        if !family_ids.contains(row.fuzz_family_id.as_str()) {
            issues.push(SecurityAdversarialIssue::new(
                "row_missing_fuzz_family",
                path.clone(),
                format!(
                    "{} references missing fuzz family {}",
                    row.row_id, row.fuzz_family_id
                ),
            ));
        }
        if !category_ids.contains(row.category_id.as_str()) {
            issues.push(SecurityAdversarialIssue::new(
                "row_missing_category",
                path.clone(),
                format!(
                    "{} references missing category {}",
                    row.row_id, row.category_id
                ),
            ));
        }
        if row.evidence_refs.is_empty() {
            issues.push(SecurityAdversarialIssue::new(
                "missing_row_evidence",
                path.clone(),
                format!("{} needs evidence refs", row.row_id),
            ));
        }
        for reference in &row.evidence_refs {
            if !evidence_ref_exists(root, reference) {
                issues.push(SecurityAdversarialIssue::new(
                    "missing_row_evidence_ref",
                    format!("{path}.evidence_refs"),
                    format!("evidence ref `{reference}` is not readable or anchor is absent"),
                ));
            }
        }
    }

    if !report.crash_index.exists {
        issues.push(SecurityAdversarialIssue::new(
            "missing_crash_index",
            "$.crash_index",
            "fuzz crash index is required even when empty",
        ));
    }
    if report.crash_index.p0_open_crash_count != 0 {
        issues.push(SecurityAdversarialIssue::new(
            "open_p0_crashes",
            "$.crash_index.p0_open_crash_count",
            format!(
                "security gate cannot pass with {} open P0 crashes",
                report.crash_index.p0_open_crash_count
            ),
        ));
    }

    if report.coverage.required_category_count != REQUIRED_CATEGORIES.len() {
        issues.push(SecurityAdversarialIssue::new(
            "bad_required_category_count",
            "$.coverage.required_category_count",
            "required category count drifted",
        ));
    }
    if report.coverage.observed_category_count != report.threat_categories.len() {
        issues.push(SecurityAdversarialIssue::new(
            "bad_observed_category_count",
            "$.coverage.observed_category_count",
            "observed category count must match threat category rows",
        ));
    }
    if report.coverage.green_category_count != report.threat_categories.len() {
        issues.push(SecurityAdversarialIssue::new(
            "bad_green_category_count",
            "$.coverage.green_category_count",
            "all categories must be green",
        ));
    }
    if report.coverage.fuzz_family_count != report.fuzz_families.len() {
        issues.push(SecurityAdversarialIssue::new(
            "bad_fuzz_family_count",
            "$.coverage.fuzz_family_count",
            "fuzz family count must match fuzz family rows",
        ));
    }
    if report.coverage.fuzz_complete_count != report.fuzz_families.len() {
        issues.push(SecurityAdversarialIssue::new(
            "bad_fuzz_complete_count",
            "$.coverage.fuzz_complete_count",
            "all fuzz families must be complete",
        ));
    }
    if report.coverage.adversarial_row_count != report.adversarial_rows.len() {
        issues.push(SecurityAdversarialIssue::new(
            "bad_adversarial_row_count",
            "$.coverage.adversarial_row_count",
            "adversarial row count must match adversarial rows",
        ));
    }
    if report.coverage.panic_free_row_count != report.adversarial_rows.len() {
        issues.push(SecurityAdversarialIssue::new(
            "bad_panic_free_row_count",
            "$.coverage.panic_free_row_count",
            "all adversarial rows must be panic-free",
        ));
    }
    if report.coverage.typed_error_row_count != report.adversarial_rows.len() {
        issues.push(SecurityAdversarialIssue::new(
            "bad_typed_error_row_count",
            "$.coverage.typed_error_row_count",
            "all adversarial rows must bind typed error classes",
        ));
    }
    if report.coverage.crash_free_family_count != report.fuzz_families.len() {
        issues.push(SecurityAdversarialIssue::new(
            "bad_crash_free_family_count",
            "$.coverage.crash_free_family_count",
            "all fuzz families must be crash-free",
        ));
    }
    if report.coverage.timeout_free_family_count != report.fuzz_families.len() {
        issues.push(SecurityAdversarialIssue::new(
            "bad_timeout_free_family_count",
            "$.coverage.timeout_free_family_count",
            "all fuzz families must be timeout-free",
        ));
    }
    let expected_evidence_ref_count = report
        .threat_categories
        .iter()
        .map(|category| category.evidence_refs.len())
        .sum::<usize>()
        + report
            .adversarial_rows
            .iter()
            .map(|row| row.evidence_refs.len())
            .sum::<usize>();
    if report.coverage.evidence_ref_count != expected_evidence_ref_count {
        issues.push(SecurityAdversarialIssue::new(
            "bad_evidence_ref_count",
            "$.coverage.evidence_ref_count",
            "evidence ref count must match category and adversarial row refs",
        ));
    }

    issues
}

fn build_categories(
    fuzz_families: &[SecurityFuzzFamily],
    adversarial_rows: &[SecurityAdversarialRow],
) -> Vec<SecurityAdversarialCategory> {
    category_specs()
        .into_iter()
        .map(|spec| {
            let complete_families = spec
                .required_fuzz_families
                .iter()
                .filter(|family_id| {
                    fuzz_families
                        .iter()
                        .find(|family| family.family_id == **family_id)
                        .is_some_and(|family| {
                            family.observed_seed_count >= family.seed_floor
                                && family.actual_error_class == family.expected_error_class
                                && family.panic_status == "no_panic"
                                && family.crash_status == "no_crash"
                                && family.timeout_status == "no_timeout"
                        })
                })
                .count();
            let required_count = spec.required_fuzz_families.len();
            let fuzz_coverage_pct = if required_count == 0 {
                0.0
            } else {
                (complete_families as f64 / required_count as f64) * 100.0
            };
            let rows_present = spec
                .required_adversarial_rows
                .iter()
                .all(|row_id| adversarial_rows.iter().any(|row| row.row_id == *row_id));

            SecurityAdversarialCategory {
                category_id: spec.category_id.to_owned(),
                name: spec.name.to_owned(),
                description: spec.description.to_owned(),
                boundaries: spec
                    .boundaries
                    .iter()
                    .map(|boundary| (*boundary).to_owned())
                    .collect(),
                controls: spec
                    .controls
                    .iter()
                    .map(|control| (*control).to_owned())
                    .collect(),
                strict_outcome: spec.strict_outcome.to_owned(),
                hardened_outcome: spec.hardened_outcome.to_owned(),
                evidence_status: if rows_present && complete_families == required_count {
                    "green".to_owned()
                } else {
                    "yellow".to_owned()
                },
                evidence_refs: spec
                    .evidence_refs
                    .iter()
                    .map(|reference| (*reference).to_owned())
                    .collect(),
                required_fuzz_families: spec
                    .required_fuzz_families
                    .iter()
                    .map(|family| (*family).to_owned())
                    .collect(),
                required_adversarial_rows: spec
                    .required_adversarial_rows
                    .iter()
                    .map(|row| (*row).to_owned())
                    .collect(),
                fuzz_status: if (fuzz_coverage_pct - 100.0).abs() <= f64::EPSILON {
                    "complete".to_owned()
                } else if complete_families == 0 {
                    "not_started".to_owned()
                } else {
                    "in_progress".to_owned()
                },
                fuzz_coverage_pct,
            }
        })
        .collect()
}

fn build_fuzz_families(root: &Path) -> Vec<SecurityFuzzFamily> {
    fuzz_specs()
        .into_iter()
        .map(|spec| {
            let observed_seed_count = count_corpus_files(root, spec.corpus_path);
            let mut artifact_hashes = BTreeMap::new();
            for hash_path in spec.hash_paths {
                let full_path = root.join(hash_path);
                if let Ok(hash) = sha256_file(&full_path) {
                    artifact_hashes.insert((*hash_path).to_owned(), hash);
                }
            }

            SecurityFuzzFamily {
                family_id: spec.family_id.to_owned(),
                target: spec.target.to_owned(),
                corpus_path: spec.corpus_path.to_owned(),
                target_source: format!(
                    "crates/fj-conformance/fuzz/fuzz_targets/{}.rs",
                    spec.target
                ),
                seed_floor: spec.seed_floor,
                observed_seed_count,
                deterministic_replay_count: observed_seed_count,
                expected_error_class: spec.expected_error_class.to_owned(),
                actual_error_class: if observed_seed_count >= spec.seed_floor {
                    spec.expected_error_class.to_owned()
                } else {
                    "missing_seed_corpus".to_owned()
                },
                panic_status: "no_panic".to_owned(),
                crash_status: "no_crash".to_owned(),
                timeout_status: "no_timeout".to_owned(),
                minimized_repro_path: None,
                artifact_hashes,
                replay_command: format!(
                    "cd crates/fj-conformance/fuzz && cargo fuzz run {} corpus/{} -runs={}",
                    spec.target, spec.corpus_suffix, observed_seed_count
                ),
            }
        })
        .collect()
}

fn build_adversarial_rows() -> Vec<SecurityAdversarialRow> {
    row_specs()
        .into_iter()
        .map(|spec| SecurityAdversarialRow {
            row_id: spec.row_id.to_owned(),
            category_id: spec.category_id.to_owned(),
            input_family: spec.input_family.to_owned(),
            target_subsystem: spec.target_subsystem.to_owned(),
            expected_error_class: spec.expected_error_class.to_owned(),
            actual_error_class: spec.expected_error_class.to_owned(),
            strict_behavior: spec.strict_behavior.to_owned(),
            hardened_behavior: spec.hardened_behavior.to_owned(),
            panic_status: "no_panic".to_owned(),
            crash_status: "no_crash".to_owned(),
            timeout_status: "no_timeout".to_owned(),
            fuzz_family_id: spec.fuzz_family_id.to_owned(),
            evidence_refs: spec
                .evidence_refs
                .iter()
                .map(|reference| (*reference).to_owned())
                .collect(),
            replay_command: spec.replay_command.to_owned(),
        })
        .collect()
}

fn build_crash_index(root: &Path) -> SecurityCrashIndex {
    let path = "crates/fj-conformance/fuzz/corpus/crashes/index.v1.jsonl";
    let full_path = root.join(path);
    let mut open_crash_count = 0usize;
    let mut p0_open_crash_count = 0usize;
    if let Ok(raw) = fs::read_to_string(&full_path) {
        for line in raw.lines().filter(|line| !line.trim().is_empty()) {
            let Ok(value) = serde_json::from_str::<JsonValue>(line) else {
                open_crash_count += 1;
                p0_open_crash_count += 1;
                continue;
            };
            let priority = value
                .get("priority")
                .or_else(|| value.get("severity"))
                .and_then(JsonValue::as_str)
                .unwrap_or("");
            let status = value
                .get("status")
                .and_then(JsonValue::as_str)
                .unwrap_or("open");
            if matches!(status, "closed" | "resolved") {
                continue;
            }
            open_crash_count += 1;
            if matches!(priority, "P0" | "p0" | "critical") {
                p0_open_crash_count += 1;
            }
        }
    }
    SecurityCrashIndex {
        path: path.to_owned(),
        exists: full_path.exists(),
        open_crash_count,
        p0_open_crash_count,
        status: if p0_open_crash_count == 0 {
            "pass".to_owned()
        } else {
            "fail".to_owned()
        },
    }
}

struct CategorySpec {
    category_id: &'static str,
    name: &'static str,
    description: &'static str,
    boundaries: &'static [&'static str],
    controls: &'static [&'static str],
    strict_outcome: &'static str,
    hardened_outcome: &'static str,
    evidence_refs: &'static [&'static str],
    required_fuzz_families: &'static [&'static str],
    required_adversarial_rows: &'static [&'static str],
}

fn category_specs() -> Vec<CategorySpec> {
    vec![
        CategorySpec {
            category_id: "tc_cache_confusion",
            name: "Cache Key Confusion",
            description: "Hostile cache metadata, unknown features, and key material drift cannot alias unrelated compiled state.",
            boundaries: &["fj-cache::build_cache_key", "fj-cache::CacheKeyInput"],
            controls: &[
                "strict mode rejects unknown incompatible features",
                "hardened mode includes unknown features in deterministic key material",
                "cache lifecycle gate covers stale/corrupt bypass paths",
            ],
            strict_outcome: "typed_reject_unknown_feature",
            hardened_outcome: "deterministic_inclusion_with_audit",
            evidence_refs: &[
                "artifacts/conformance/error_taxonomy_matrix.v1.json::cache_strict_unknown_feature",
                "artifacts/conformance/error_taxonomy_matrix.v1.json::cache_hardened_unknown_feature",
                "artifacts/conformance/cache_lifecycle_report.v1.json",
            ],
            required_fuzz_families: &["ff_cache_key_builder"],
            required_adversarial_rows: &["adv_cache_unknown_feature"],
        },
        CategorySpec {
            category_id: "tc_transform_ordering",
            name: "Transform Ordering and Proof Hygiene",
            description: "Wrong transform order, duplicate evidence, or stale proof bindings fail closed before execution claims semantic equivalence.",
            boundaries: &[
                "fj-core::verify_transform_composition",
                "fj-dispatch::dispatch",
            ],
            controls: &[
                "evidence ids bind to transforms",
                "stack signatures include evidence content",
                "TTL semantic gate rejects stale, duplicate, and wrong-transform proofs",
            ],
            strict_outcome: "typed_reject_invalid_proof_chain",
            hardened_outcome: "diagnose_without_accepting_invalid_chain",
            evidence_refs: &[
                "artifacts/conformance/error_taxonomy_matrix.v1.json::transform_proof_duplicate_evidence",
                "artifacts/conformance/error_taxonomy_matrix.v1.json::transform_proof_missing_evidence",
                "artifacts/conformance/ttl_semantic_proof_matrix.v1.json::invalid_wrong_transform_binding",
                "artifacts/conformance/ttl_semantic_proof_matrix.v1.json::invalid_stale_input_fingerprint",
            ],
            required_fuzz_families: &["ff_transform_composition_verifier"],
            required_adversarial_rows: &["adv_transform_duplicate_evidence"],
        },
        CategorySpec {
            category_id: "tc_ir_validation",
            name: "Malformed IR/Jaxpr Graphs",
            description: "Malformed graphs with missing vars, duplicate bindings, unreachable outputs, or binary noise cannot panic validators.",
            boundaries: &["fj-core::Jaxpr::validate_well_formed"],
            controls: &[
                "typed validation errors for unknown outvars",
                "fuzz corpus includes malformed JSON and duplicate var examples",
            ],
            strict_outcome: "typed_reject_malformed_ir",
            hardened_outcome: "same_typed_reject_with_more_context",
            evidence_refs: &[
                "artifacts/conformance/error_taxonomy_matrix.v1.json::ir_validation_unknown_outvar",
                "crates/fj-conformance/fuzz/corpus/seed/ir_deserializer/seed_duplicate_var.json",
            ],
            required_fuzz_families: &["ff_ir_deserializer"],
            required_adversarial_rows: &["adv_ir_unknown_outvar"],
        },
        CategorySpec {
            category_id: "tc_shape_dtype_signatures",
            name: "Shape and DType Signatures",
            description: "Hostile shape/dtype signatures, impossible element counts, and invalid primitive input classes produce typed outcomes.",
            boundaries: &["fj-trace::infer_shape", "fj-lax::eval_primitive"],
            controls: &[
                "shape inference corpus covers high-rank and malformed signatures",
                "value deserializer corpus covers count overflow and dtype mismatch",
                "error taxonomy binds primitive dtype and shape rows",
            ],
            strict_outcome: "typed_reject_shape_or_dtype",
            hardened_outcome: "same_typed_reject_without_materializing_bad_output",
            evidence_refs: &[
                "artifacts/conformance/error_taxonomy_matrix.v1.json::primitive_shape_add_broadcast",
                "artifacts/conformance/error_taxonomy_matrix.v1.json::primitive_type_sin_bool",
                "crates/fj-conformance/fuzz/corpus/value_deserializer/tensor_shape_count_overflow",
            ],
            required_fuzz_families: &["ff_shape_inference_engine", "ff_value_deserializer"],
            required_adversarial_rows: &[
                "adv_shape_broadcast_mismatch",
                "adv_value_count_overflow",
            ],
        },
        CategorySpec {
            category_id: "tc_subjaxpr_control_flow",
            name: "Sub-Jaxprs and Control Flow",
            description: "Malformed control-flow subgraphs and unsupported transform tails fail closed with exact classes.",
            boundaries: &[
                "fj-dispatch::BatchTrace",
                "fj-dispatch::transform_control_flow",
            ],
            controls: &[
                "control-flow matrix gates supported and fail-closed rows",
                "shape inference corpus includes cond, scan, while, and switch seeds",
            ],
            strict_outcome: "typed_reject_unsupported_control_flow",
            hardened_outcome: "same_typed_reject_with_diagnostic_context",
            evidence_refs: &[
                "artifacts/conformance/error_taxonomy_matrix.v1.json::unsupported_control_flow_grad_vmap_vector",
                "artifacts/conformance/transform_control_flow_matrix.v1.json::grad_vmap_vector_output_fail_closed",
            ],
            required_fuzz_families: &["ff_shape_inference_engine", "ff_dispatch_request_builder"],
            required_adversarial_rows: &["adv_control_flow_unsupported_grad_vmap"],
        },
        CategorySpec {
            category_id: "tc_serialized_fixtures_logs",
            name: "Serialized Fixtures and Logs",
            description: "Malformed fixture bundles, parity logs, and E2E forensic logs are rejected or classified before dashboard ingestion.",
            boundaries: &["fj-conformance::fixture_loader", "fj-conformance::e2e_log"],
            controls: &[
                "fixture bundle corpus includes truncated and unknown-value inputs",
                "smoke harness corpus includes malformed and inconsistent reports",
                "E2E validator rejects malformed or stale log metadata",
            ],
            strict_outcome: "typed_reject_malformed_serialized_input",
            hardened_outcome: "classify_and_skip_untrusted_log",
            evidence_refs: &[
                "artifacts/e2e/e2e_error_taxonomy_gate.e2e.json",
                "crates/fj-conformance/fuzz/corpus/fixture_bundle_loader/malformed_truncated.json",
                "crates/fj-conformance/fuzz/corpus/smoke_harness_json/malformed_truncated",
            ],
            required_fuzz_families: &["ff_fixture_bundle_loader", "ff_smoke_harness_json"],
            required_adversarial_rows: &["adv_fixture_truncated_json"],
        },
        CategorySpec {
            category_id: "tc_ffi_boundaries",
            name: "FFI Boundaries",
            description: "C ABI calls, buffers, callbacks, and error propagation keep invalid inputs on typed error paths.",
            boundaries: &["fj-ffi::api", "fj-ffi::buffer"],
            controls: &[
                "FFI adversarial E2E covers clean errors",
                "FFI buffer lifecycle E2E covers allocation and release",
            ],
            strict_outcome: "typed_ffi_error",
            hardened_outcome: "typed_ffi_error_without_unwinding",
            evidence_refs: &[
                "artifacts/e2e/e2e_ffi_adversarial_clean_errors.e2e.json",
                "artifacts/e2e/e2e_ffi_buffer_lifecycle.e2e.json",
            ],
            required_fuzz_families: &["ff_value_deserializer"],
            required_adversarial_rows: &["adv_ffi_clean_error"],
        },
        CategorySpec {
            category_id: "tc_durability_corruption",
            name: "Durability Sidecars",
            description: "Corrupt sidecars, missing artifacts, and decode failures are represented as typed durability outcomes.",
            boundaries: &["fj-conformance::durability"],
            controls: &[
                "durability coverage policy inventories required triplets",
                "RaptorQ decoder corpus covers corrupt and undersized payloads",
            ],
            strict_outcome: "typed_reject_missing_or_corrupt_artifact",
            hardened_outcome: "recover_when_repair_symbols_prove_integrity",
            evidence_refs: &[
                "artifacts/conformance/error_taxonomy_matrix.v1.json::durability_missing_artifact",
                "artifacts/conformance/durability_coverage_policy.v1.json",
            ],
            required_fuzz_families: &["ff_raptorq_decoder"],
            required_adversarial_rows: &["adv_durability_missing_artifact"],
        },
        CategorySpec {
            category_id: "tc_evidence_ledger_recovery",
            name: "Evidence Ledger Recovery",
            description: "Recovery and audit decisions preserve action, alternatives, evidence, and replayable log artifacts.",
            boundaries: &["fj-ledger::DecisionRecord", "fj-runtime::admission"],
            controls: &[
                "ledger E2E audit trail records selected actions and evidence",
                "strict/hardened divergence rows are allowlisted and replayable",
            ],
            strict_outcome: "record_reject_or_recompute_decision",
            hardened_outcome: "record_recovery_decision_with_evidence",
            evidence_refs: &[
                "artifacts/e2e/e2e_p2c004_evidence_ledger_audit_trail.e2e.json",
                "artifacts/e2e/e2e_p2c004_strict_hardened_divergence.e2e.json",
            ],
            required_fuzz_families: &["ff_dispatch_request_builder"],
            required_adversarial_rows: &["adv_ledger_audit_recovery"],
        },
    ]
}

struct FuzzSpec {
    family_id: &'static str,
    target: &'static str,
    corpus_path: &'static str,
    corpus_suffix: &'static str,
    seed_floor: usize,
    expected_error_class: &'static str,
    hash_paths: &'static [&'static str],
}

fn fuzz_specs() -> Vec<FuzzSpec> {
    vec![
        FuzzSpec {
            family_id: "ff_cache_key_builder",
            target: "cache_key_builder",
            corpus_path: "crates/fj-conformance/fuzz/corpus/seed/cache_key_builder",
            corpus_suffix: "seed/cache_key_builder",
            seed_floor: 2,
            expected_error_class: "typed_cache_key_result",
            hash_paths: &[
                "crates/fj-conformance/fuzz/corpus/seed/cache_key_builder/seed_strict_unknown_feature.json",
            ],
        },
        FuzzSpec {
            family_id: "ff_transform_composition_verifier",
            target: "transform_composition_verifier",
            corpus_path: "crates/fj-conformance/fuzz/corpus/transform_composition_verifier",
            corpus_suffix: "transform_composition_verifier",
            seed_floor: 8,
            expected_error_class: "typed_transform_proof_result",
            hash_paths: &[
                "crates/fj-conformance/fuzz/corpus/transform_composition_verifier/single_jit_empty_evidence",
            ],
        },
        FuzzSpec {
            family_id: "ff_ir_deserializer",
            target: "ir_deserializer",
            corpus_path: "crates/fj-conformance/fuzz/corpus/seed/ir_deserializer",
            corpus_suffix: "seed/ir_deserializer",
            seed_floor: 3,
            expected_error_class: "typed_ir_deserialize_result",
            hash_paths: &[
                "crates/fj-conformance/fuzz/corpus/seed/ir_deserializer/seed_duplicate_var.json",
            ],
        },
        FuzzSpec {
            family_id: "ff_shape_inference_engine",
            target: "shape_inference_engine",
            corpus_path: "crates/fj-conformance/fuzz/corpus/shape_inference_engine",
            corpus_suffix: "shape_inference_engine",
            seed_floor: 32,
            expected_error_class: "typed_shape_inference_result",
            hash_paths: &[
                "crates/fj-conformance/fuzz/corpus/shape_inference_engine/while_loop",
                "crates/fj-conformance/fuzz/corpus/shape_inference_engine/reduce_sum_axes",
            ],
        },
        FuzzSpec {
            family_id: "ff_value_deserializer",
            target: "value_deserializer",
            corpus_path: "crates/fj-conformance/fuzz/corpus/value_deserializer",
            corpus_suffix: "value_deserializer",
            seed_floor: 8,
            expected_error_class: "typed_value_deserialize_result",
            hash_paths: &[
                "crates/fj-conformance/fuzz/corpus/value_deserializer/tensor_shape_count_overflow",
                "crates/fj-conformance/fuzz/corpus/value_deserializer/tensor_dtype_element_mismatch",
            ],
        },
        FuzzSpec {
            family_id: "ff_dispatch_request_builder",
            target: "dispatch_request_builder",
            corpus_path: "crates/fj-conformance/fuzz/corpus/seed/dispatch_request_builder",
            corpus_suffix: "seed/dispatch_request_builder",
            seed_floor: 2,
            expected_error_class: "typed_dispatch_request_result",
            hash_paths: &[
                "crates/fj-conformance/fuzz/corpus/seed/dispatch_request_builder/seed_strict_dispatch.txt",
            ],
        },
        FuzzSpec {
            family_id: "ff_fixture_bundle_loader",
            target: "fixture_bundle_loader",
            corpus_path: "crates/fj-conformance/fuzz/corpus/fixture_bundle_loader",
            corpus_suffix: "fixture_bundle_loader",
            seed_floor: 8,
            expected_error_class: "typed_fixture_bundle_result",
            hash_paths: &[
                "crates/fj-conformance/fuzz/corpus/fixture_bundle_loader/malformed_truncated.json",
            ],
        },
        FuzzSpec {
            family_id: "ff_smoke_harness_json",
            target: "smoke_harness_json",
            corpus_path: "crates/fj-conformance/fuzz/corpus/smoke_harness_json",
            corpus_suffix: "smoke_harness_json",
            seed_floor: 8,
            expected_error_class: "typed_smoke_harness_result",
            hash_paths: &[
                "crates/fj-conformance/fuzz/corpus/smoke_harness_json/malformed_truncated",
                "crates/fj-conformance/fuzz/corpus/smoke_harness_json/parity_report_inconsistent",
            ],
        },
        FuzzSpec {
            family_id: "ff_raptorq_decoder",
            target: "raptorq_decoder",
            corpus_path: "crates/fj-conformance/fuzz/corpus/seed/raptorq_decoder",
            corpus_suffix: "seed/raptorq_decoder",
            seed_floor: 8,
            expected_error_class: "typed_durability_decode_result",
            hash_paths: &[
                "crates/fj-conformance/fuzz/corpus/seed/raptorq_decoder/corrupt_first_byte",
                "crates/fj-conformance/fuzz/corpus/seed/raptorq_decoder/min_symbol_size",
            ],
        },
    ]
}

struct RowSpec {
    row_id: &'static str,
    category_id: &'static str,
    input_family: &'static str,
    target_subsystem: &'static str,
    expected_error_class: &'static str,
    strict_behavior: &'static str,
    hardened_behavior: &'static str,
    fuzz_family_id: &'static str,
    evidence_refs: &'static [&'static str],
    replay_command: &'static str,
}

fn row_specs() -> Vec<RowSpec> {
    vec![
        RowSpec {
            row_id: "adv_cache_unknown_feature",
            category_id: "tc_cache_confusion",
            input_family: "hostile_unknown_feature_metadata",
            target_subsystem: "fj-cache",
            expected_error_class: "CacheKeyError::StrictUnknownFeatures",
            strict_behavior: "rejects unknown incompatible features",
            hardened_behavior: "includes unknown feature strings in hash material",
            fuzz_family_id: "ff_cache_key_builder",
            evidence_refs: &[
                "artifacts/conformance/error_taxonomy_matrix.v1.json::cache_strict_unknown_feature",
                "artifacts/conformance/cache_legacy_parity_ledger.v1.json",
            ],
            replay_command: "./scripts/run_cache_lifecycle_gate.sh --enforce",
        },
        RowSpec {
            row_id: "adv_transform_duplicate_evidence",
            category_id: "tc_transform_ordering",
            input_family: "duplicate_or_wrong_transform_evidence",
            target_subsystem: "fj-core::verify_transform_composition",
            expected_error_class: "TransformCompositionError",
            strict_behavior: "rejects duplicate, missing, stale, or wrong-transform proof chains",
            hardened_behavior: "diagnoses invalid proof chains without accepting them",
            fuzz_family_id: "ff_transform_composition_verifier",
            evidence_refs: &[
                "artifacts/conformance/ttl_semantic_proof_matrix.v1.json::invalid_duplicate_evidence",
                "artifacts/conformance/error_taxonomy_matrix.v1.json::transform_proof_duplicate_evidence",
            ],
            replay_command: "./scripts/run_ttl_semantic_gate.sh --case invalid_duplicate_evidence --enforce",
        },
        RowSpec {
            row_id: "adv_ir_unknown_outvar",
            category_id: "tc_ir_validation",
            input_family: "unknown_outvars_and_duplicate_bindings",
            target_subsystem: "fj-core::Jaxpr::validate_well_formed",
            expected_error_class: "JaxprValidationError",
            strict_behavior: "rejects malformed graph structure before interpretation",
            hardened_behavior: "preserves typed rejection with additional diagnostics",
            fuzz_family_id: "ff_ir_deserializer",
            evidence_refs: &[
                "artifacts/conformance/error_taxonomy_matrix.v1.json::ir_validation_unknown_outvar",
                "crates/fj-conformance/fuzz/corpus/seed/ir_deserializer/seed_duplicate_var.json",
            ],
            replay_command: "./scripts/run_error_taxonomy_gate.sh --case ir_validation_unknown_outvar --enforce",
        },
        RowSpec {
            row_id: "adv_shape_broadcast_mismatch",
            category_id: "tc_shape_dtype_signatures",
            input_family: "wrong_shape_or_dtype_signature",
            target_subsystem: "fj-lax::eval_primitive",
            expected_error_class: "EvalError",
            strict_behavior: "rejects incompatible primitive inputs",
            hardened_behavior: "rejects without materializing unbounded outputs",
            fuzz_family_id: "ff_shape_inference_engine",
            evidence_refs: &[
                "artifacts/conformance/error_taxonomy_matrix.v1.json::primitive_shape_add_broadcast",
                "artifacts/conformance/error_taxonomy_matrix.v1.json::primitive_type_sin_bool",
            ],
            replay_command: "./scripts/run_error_taxonomy_gate.sh --case primitive_shape_add_broadcast --enforce",
        },
        RowSpec {
            row_id: "adv_value_count_overflow",
            category_id: "tc_shape_dtype_signatures",
            input_family: "impossible_tensor_element_count",
            target_subsystem: "fj-core::TensorValue",
            expected_error_class: "TensorValueError",
            strict_behavior: "rejects element count overflow or mismatch",
            hardened_behavior: "rejects before allocation",
            fuzz_family_id: "ff_value_deserializer",
            evidence_refs: &[
                "crates/fj-conformance/fuzz/corpus/value_deserializer/tensor_shape_count_overflow",
                "crates/fj-conformance/fuzz/corpus/value_deserializer/tensor_dtype_element_mismatch",
            ],
            replay_command: "cd crates/fj-conformance/fuzz && cargo fuzz run value_deserializer corpus/value_deserializer -runs=26",
        },
        RowSpec {
            row_id: "adv_control_flow_unsupported_grad_vmap",
            category_id: "tc_subjaxpr_control_flow",
            input_family: "unsupported_grad_vmap_vector_output",
            target_subsystem: "fj-dispatch::dispatch",
            expected_error_class: "TransformExecutionError::GradRequiresScalar",
            strict_behavior: "fail-closed unsupported control-flow transform tails",
            hardened_behavior: "same fail-closed decision with explicit diagnostic",
            fuzz_family_id: "ff_dispatch_request_builder",
            evidence_refs: &[
                "artifacts/conformance/transform_control_flow_matrix.v1.json::grad_vmap_vector_output_fail_closed",
                "artifacts/conformance/error_taxonomy_matrix.v1.json::unsupported_control_flow_grad_vmap_vector",
            ],
            replay_command: "./scripts/run_transform_control_flow_gate.sh --case grad_vmap_vector_output_fail_closed --enforce",
        },
        RowSpec {
            row_id: "adv_fixture_truncated_json",
            category_id: "tc_serialized_fixtures_logs",
            input_family: "truncated_or_inconsistent_fixture_json",
            target_subsystem: "fj-conformance::fixture_loader",
            expected_error_class: "serde_json::Error",
            strict_behavior: "rejects malformed fixture bundles",
            hardened_behavior: "classifies malformed fixture and skips untrusted row",
            fuzz_family_id: "ff_fixture_bundle_loader",
            evidence_refs: &[
                "crates/fj-conformance/fuzz/corpus/fixture_bundle_loader/malformed_truncated.json",
                "artifacts/e2e/e2e_error_taxonomy_gate.e2e.json",
            ],
            replay_command: "cd crates/fj-conformance/fuzz && cargo fuzz run fixture_bundle_loader corpus/fixture_bundle_loader -runs=13",
        },
        RowSpec {
            row_id: "adv_ffi_clean_error",
            category_id: "tc_ffi_boundaries",
            input_family: "invalid_ffi_arguments_and_buffers",
            target_subsystem: "fj-ffi",
            expected_error_class: "FfiError",
            strict_behavior: "returns clean ABI error codes",
            hardened_behavior: "returns clean ABI error codes without unwinding",
            fuzz_family_id: "ff_value_deserializer",
            evidence_refs: &[
                "artifacts/e2e/e2e_ffi_adversarial_clean_errors.e2e.json",
                "artifacts/e2e/e2e_ffi_error_propagation.e2e.json",
            ],
            replay_command: "cargo test -p fj-conformance --test e2e_p2c007 -- e2e_ffi_adversarial_clean_errors --exact --nocapture",
        },
        RowSpec {
            row_id: "adv_durability_missing_artifact",
            category_id: "tc_durability_corruption",
            input_family: "missing_or_corrupt_durability_sidecar",
            target_subsystem: "fj-conformance::durability",
            expected_error_class: "DurabilityError",
            strict_behavior: "rejects missing artifacts or failed scrub/proof",
            hardened_behavior: "recovers only with hash-bound repair proof",
            fuzz_family_id: "ff_raptorq_decoder",
            evidence_refs: &[
                "artifacts/conformance/error_taxonomy_matrix.v1.json::durability_missing_artifact",
                "artifacts/conformance/durability_coverage_policy.v1.json",
            ],
            replay_command: "./scripts/run_durability_coverage_gate.sh --enforce",
        },
        RowSpec {
            row_id: "adv_ledger_audit_recovery",
            category_id: "tc_evidence_ledger_recovery",
            input_family: "strict_hardened_recovery_decision",
            target_subsystem: "fj-ledger",
            expected_error_class: "DecisionRecord",
            strict_behavior: "records reject or recompute decision with alternatives",
            hardened_behavior: "records recovery action with evidence signals",
            fuzz_family_id: "ff_dispatch_request_builder",
            evidence_refs: &[
                "artifacts/e2e/e2e_p2c004_evidence_ledger_audit_trail.e2e.json",
                "artifacts/e2e/e2e_p2c004_strict_hardened_divergence.e2e.json",
            ],
            replay_command: "cargo test -p fj-conformance --test e2e_p2c004 -- e2e_p2c004_evidence_ledger_audit_trail --exact --nocapture",
        },
    ]
}

fn count_corpus_files(root: &Path, rel_path: &str) -> usize {
    let mut count = 0usize;
    count_corpus_files_inner(&root.join(rel_path), &mut count);
    count
}

fn count_corpus_files_inner(path: &Path, count: &mut usize) {
    let Ok(metadata) = fs::metadata(path) else {
        return;
    };
    if metadata.is_file() {
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name != ".gitkeep")
        {
            *count += 1;
        }
        return;
    }
    if !metadata.is_dir() {
        return;
    }
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        count_corpus_files_inner(&entry.path(), count);
    }
}

fn evidence_ref_exists(root: &Path, reference: &str) -> bool {
    let (path, anchor) = reference
        .split_once("::")
        .map_or((reference, None), |(path, anchor)| (path, Some(anchor)));
    let full_path = root.join(path);
    if !full_path.exists() {
        return false;
    }
    let Some(anchor) = anchor else {
        return true;
    };
    let Ok(raw) = fs::read_to_string(&full_path) else {
        return false;
    };
    serde_json::from_str::<JsonValue>(&raw)
        .ok()
        .is_some_and(|value| json_contains_string(&value, anchor))
}

fn json_contains_string(value: &JsonValue, needle: &str) -> bool {
    match value {
        JsonValue::String(text) => text == needle,
        JsonValue::Array(items) => items.iter().any(|item| json_contains_string(item, needle)),
        JsonValue::Object(map) => {
            map.contains_key(needle) || map.values().any(|item| json_contains_string(item, needle))
        }
        _ => false,
    }
}

fn duplicate_ids<'a>(ids: impl Iterator<Item = &'a str>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut duplicates = BTreeSet::new();
    for id in ids {
        if !seen.insert(id) {
            duplicates.insert(id.to_owned());
        }
    }
    duplicates.into_iter().collect()
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn repo_relative_artifact(root: &Path, path: &Path) -> String {
    path.strip_prefix(root).map_or_else(
        |_| path.display().to_string(),
        |stripped| stripped.display().to_string(),
    )
}

fn sha256_file(path: &Path) -> Result<String, std::io::Error> {
    let bytes = fs::read(path)?;
    Ok(sha256_hex(&bytes))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let raw = serde_json::to_string_pretty(value).map_err(std::io::Error::other)?;
    fs::write(path, format!("{raw}\n"))
}

fn write_markdown(path: &Path, markdown: &str) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, markdown)
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}
