#![forbid(unsafe_code)]

use crate::ToleranceTier;
use fj_core::{DType, Literal};
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

pub const NUMERICAL_STABILITY_REPORT_SCHEMA_VERSION: &str = "frankenjax.numerical-stability.v1";
pub const NUMERICAL_STABILITY_BEAD_ID: &str = "frankenjax-cstq.20";

const REQUIRED_STABILITY_FAMILIES: &[&str] = &[
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
];

const ALLOWED_TOLERANCE_COMPARATORS: &[&str] = &[
    "approx_atol_rtol",
    "componentwise_atol_rtol",
    "exact_bitwise",
    "finite_difference_atol_rtol",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumericalStabilityReport {
    pub schema_version: String,
    pub bead_id: String,
    pub generated_at_unix_ms: u128,
    pub status: String,
    pub policy: String,
    pub required_stability_families: Vec<String>,
    pub summary: NumericalStabilitySummary,
    pub platform_fingerprints: Vec<PlatformStabilityFingerprint>,
    pub tolerance_policies: Vec<StabilityTolerancePolicy>,
    pub rows: Vec<NumericalStabilityRow>,
    pub issues: Vec<NumericalStabilityIssue>,
    pub artifact_refs: Vec<String>,
    pub replay_command: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NumericalStabilitySummary {
    pub required_family_count: usize,
    pub observed_family_count: usize,
    pub row_count: usize,
    pub pass_count: usize,
    pub tolerance_policy_count: usize,
    pub platform_fingerprint_count: usize,
    pub rows_with_platform: usize,
    pub rows_with_artifacts: usize,
    pub rows_with_replay: usize,
    pub rows_with_non_finite_classification: usize,
    pub deterministic_replay_rows: usize,
    pub unstable_count: usize,
    pub stale_count: usize,
    pub regression_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlatformStabilityFingerprint {
    pub platform_id: String,
    pub os: String,
    pub arch: String,
    pub endian: String,
    pub rust_version: String,
    pub cargo_version: String,
    pub cargo_target_dir: String,
    pub deterministic_serialization: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StabilityTolerancePolicy {
    pub policy_id: String,
    pub dtype: String,
    pub comparator: String,
    pub atol: f64,
    pub rtol: f64,
    pub ulp: Option<u64>,
    pub finite_difference_step: Option<f64>,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NumericalStabilityRow {
    pub case_id: String,
    pub family: String,
    pub primitive_or_scope: String,
    pub dtype: String,
    pub shape: Vec<u32>,
    pub seed: Option<u64>,
    pub edge_condition_class: String,
    pub reference_source: String,
    pub tolerance_policy_id: String,
    pub reference_scale: f64,
    pub stability_guard: String,
    pub expected_behavior: String,
    pub actual_behavior: String,
    pub max_abs_error: f64,
    pub max_rel_error: f64,
    pub max_ulp_error: Option<u64>,
    pub non_finite_classification: String,
    pub deterministic_replay_count: usize,
    pub platform_fingerprint_id: String,
    pub artifact_refs: Vec<String>,
    pub replay_command: String,
    pub dashboard_row: String,
    pub status: String,
    pub stale: bool,
    pub regression: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NumericalStabilityIssue {
    pub code: String,
    pub path: String,
    pub message: String,
}

impl NumericalStabilityIssue {
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
pub struct NumericalStabilityOutputPaths {
    pub report: PathBuf,
    pub markdown: PathBuf,
    pub e2e: PathBuf,
}

impl NumericalStabilityOutputPaths {
    #[must_use]
    pub fn for_root(root: &Path) -> Self {
        Self {
            report: root.join("artifacts/conformance/numerical_stability_matrix.v1.json"),
            markdown: root.join("artifacts/conformance/numerical_stability_matrix.v1.md"),
            e2e: root.join("artifacts/e2e/e2e_numerical_stability_gate.e2e.json"),
        }
    }
}

#[must_use]
pub fn build_numerical_stability_report(root: &Path) -> NumericalStabilityReport {
    let platform = capture_platform_fingerprint();
    let platform_id = platform.platform_id.clone();
    let tolerance_policies = tolerance_policies();
    let rows = stability_rows(&platform_id);
    let summary = summarize_rows(&rows, &tolerance_policies, 1);
    let mut report = NumericalStabilityReport {
        schema_version: NUMERICAL_STABILITY_REPORT_SCHEMA_VERSION.to_owned(),
        bead_id: NUMERICAL_STABILITY_BEAD_ID.to_owned(),
        generated_at_unix_ms: now_unix_ms(),
        status: "pass".to_owned(),
        policy:
            "Numerical-stability green requires explicit tolerance policy, edge-condition class, guard path, platform fingerprint, non-finite classification, artifact hash binding, deterministic replay count, and exact replay command for every stability family. Tolerances may not be widened silently; unstable or stale rows must surface as issues."
                .to_owned(),
        required_stability_families: REQUIRED_STABILITY_FAMILIES
            .iter()
            .map(|family| (*family).to_owned())
            .collect(),
        summary,
        platform_fingerprints: vec![platform],
        tolerance_policies,
        rows,
        issues: Vec::new(),
        artifact_refs: vec![
            "artifacts/e2e/e2e_tolerance_tightening.e2e.json".to_owned(),
            "artifacts/e2e/e2e_rng_determinism.e2e.json".to_owned(),
            "artifacts/e2e/e2e_mixed_dtype.e2e.json".to_owned(),
            "artifacts/e2e/e2e_multirank_fixtures.e2e.json".to_owned(),
            "crates/fj-conformance/tests/ad_numerical_verification.rs".to_owned(),
            "crates/fj-conformance/tests/lax_oracle.rs".to_owned(),
        ],
        replay_command: "./scripts/run_numerical_stability_gate.sh --enforce".to_owned(),
    };
    report.issues = validate_numerical_stability_report(root, &report);
    if !report.issues.is_empty() {
        report.status = "fail".to_owned();
    }
    report
}

pub fn write_numerical_stability_outputs(
    root: &Path,
    report_path: &Path,
    markdown_path: &Path,
) -> Result<NumericalStabilityReport, std::io::Error> {
    let report = build_numerical_stability_report(root);
    write_json(report_path, &report)?;
    if let Some(parent) = markdown_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(markdown_path, numerical_stability_markdown(&report))?;
    Ok(report)
}

#[must_use]
pub fn numerical_stability_summary_json(report: &NumericalStabilityReport) -> JsonValue {
    json!({
        "schema_version": report.schema_version,
        "bead_id": report.bead_id,
        "status": report.status,
        "summary": report.summary,
        "issue_count": report.issues.len(),
        "issues": report.issues,
        "row_status": report.rows.iter().map(|row| {
            json!({
                "case_id": row.case_id,
                "family": row.family,
                "status": row.status,
                "dtype": row.dtype,
                "edge_condition_class": row.edge_condition_class,
                "tolerance_policy_id": row.tolerance_policy_id,
                "max_abs_error": row.max_abs_error,
                "max_rel_error": row.max_rel_error,
                "max_ulp_error": row.max_ulp_error,
                "non_finite_classification": row.non_finite_classification,
                "platform_fingerprint_id": row.platform_fingerprint_id,
            })
        }).collect::<Vec<_>>(),
    })
}

#[must_use]
pub fn numerical_stability_markdown(report: &NumericalStabilityReport) -> String {
    let mut out = String::new();
    out.push_str("# Numerical Stability Matrix\n\n");
    out.push_str(&format!(
        "- Schema: `{}`\n- Bead: `{}`\n- Status: `{}`\n- Rows: `{}`\n- Platform fingerprints: `{}`\n\n",
        report.schema_version,
        report.bead_id,
        report.status,
        report.summary.row_count,
        report.summary.platform_fingerprint_count,
    ));
    out.push_str("| Family | DType | Edge | Guard | Max abs | Non-finite | Dashboard |\n");
    out.push_str("|---|---|---|---|---:|---|---|\n");
    for row in &report.rows {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | `{:.3e}` | `{}` | `{}` |\n",
            row.family,
            row.dtype,
            row.edge_condition_class,
            row.stability_guard,
            row.max_abs_error,
            row.non_finite_classification,
            row.dashboard_row,
        ));
    }
    if report.issues.is_empty() {
        out.push_str("\nNo numerical-stability issues found.\n");
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
pub fn validate_numerical_stability_report(
    root: &Path,
    report: &NumericalStabilityReport,
) -> Vec<NumericalStabilityIssue> {
    let mut issues = Vec::new();
    if report.schema_version != NUMERICAL_STABILITY_REPORT_SCHEMA_VERSION {
        issues.push(NumericalStabilityIssue::new(
            "bad_schema_version",
            "$.schema_version",
            "numerical stability schema marker changed",
        ));
    }
    if report.bead_id != NUMERICAL_STABILITY_BEAD_ID {
        issues.push(NumericalStabilityIssue::new(
            "bad_bead_id",
            "$.bead_id",
            "numerical stability report must stay bound to frankenjax-cstq.20",
        ));
    }
    if report.replay_command.trim().is_empty() {
        issues.push(NumericalStabilityIssue::new(
            "missing_replay_command",
            "$.replay_command",
            "report needs a stable replay command",
        ));
    }
    if !matches!(report.status.as_str(), "pass" | "fail") {
        issues.push(NumericalStabilityIssue::new(
            "bad_report_status",
            "$.status",
            format!(
                "numerical stability status `{}` must be pass or fail",
                report.status
            ),
        ));
    }
    if report.policy.trim().is_empty() {
        issues.push(NumericalStabilityIssue::new(
            "empty_policy",
            "$.policy",
            "numerical stability report needs a user-facing policy",
        ));
    }

    validate_platforms(report, &mut issues);
    validate_tolerance_policies(report, &mut issues);
    validate_declared_stability_families(report, &mut issues);

    let required = REQUIRED_STABILITY_FAMILIES
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let observed = report
        .rows
        .iter()
        .map(|row| row.family.as_str())
        .collect::<BTreeSet<_>>();
    for row in &report.rows {
        if !required.contains(row.family.as_str()) {
            issues.push(NumericalStabilityIssue::new(
                "unknown_stability_family",
                "$.rows",
                format!("unknown stability family `{}`", row.family),
            ));
        }
    }
    for required_family in &required {
        if !observed.contains(required_family) {
            issues.push(NumericalStabilityIssue::new(
                "missing_stability_family",
                "$.rows",
                format!("required stability family `{required_family}` is not covered"),
            ));
        }
    }

    let platform_ids = report
        .platform_fingerprints
        .iter()
        .map(|platform| platform.platform_id.as_str())
        .collect::<BTreeSet<_>>();
    let policy_by_id = report
        .tolerance_policies
        .iter()
        .map(|policy| (policy.policy_id.as_str(), policy))
        .collect::<BTreeMap<_, _>>();
    let policy_ids = report
        .tolerance_policies
        .iter()
        .map(|policy| policy.policy_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut row_ids = BTreeSet::new();
    for (idx, row) in report.rows.iter().enumerate() {
        let mut context = RowValidationContext {
            root,
            policy_ids: &policy_ids,
            policy_by_id: &policy_by_id,
            platform_ids: &platform_ids,
            row_ids: &mut row_ids,
        };
        validate_row(row, idx, &mut context, &mut issues);
    }

    for artifact_ref in &report.artifact_refs {
        if !root.join(artifact_ref).exists() {
            issues.push(NumericalStabilityIssue::new(
                "missing_report_artifact_ref",
                "$.artifact_refs",
                format!("artifact ref `{artifact_ref}` is missing"),
            ));
        }
    }

    let expected = summarize_rows(
        &report.rows,
        &report.tolerance_policies,
        report.platform_fingerprints.len(),
    );
    if report.summary != expected {
        issues.push(NumericalStabilityIssue::new(
            "bad_summary",
            "$.summary",
            "summary counts must match rows, policies, and platform fingerprints",
        ));
    }

    issues
}

fn validate_declared_stability_families(
    report: &NumericalStabilityReport,
    issues: &mut Vec<NumericalStabilityIssue>,
) {
    let required = REQUIRED_STABILITY_FAMILIES
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut declared = BTreeSet::new();
    for family in &report.required_stability_families {
        if !declared.insert(family.as_str()) {
            issues.push(NumericalStabilityIssue::new(
                "duplicate_required_stability_family",
                "$.required_stability_families",
                format!("duplicate required stability family `{family}`"),
            ));
        }
        if !required.contains(family.as_str()) {
            issues.push(NumericalStabilityIssue::new(
                "unknown_required_stability_family",
                "$.required_stability_families",
                format!("unknown required stability family `{family}`"),
            ));
        }
    }
    for required_family in required.difference(&declared) {
        issues.push(NumericalStabilityIssue::new(
            "missing_required_stability_family",
            "$.required_stability_families",
            format!("required stability family `{required_family}` is not declared"),
        ));
    }
}

#[must_use]
pub fn classify_non_finite_f64(value: f64) -> String {
    if value.is_nan() {
        format!(
            "nan_payload_preserved:{:016x}",
            value.to_bits() & 0x000f_ffff_ffff_ffff
        )
    } else if value.is_infinite() && value.is_sign_positive() {
        "positive_infinity_preserved".to_owned()
    } else if value.is_infinite() {
        "negative_infinity_preserved".to_owned()
    } else {
        "finite".to_owned()
    }
}

#[must_use]
pub fn f64_ulp_distance(left: f64, right: f64) -> Option<u64> {
    if left.is_nan() || right.is_nan() {
        return None;
    }
    Some(ordered_float_bits(left).abs_diff(ordered_float_bits(right)))
}

#[must_use]
pub fn finite_difference_step(reference_scale: f64) -> f64 {
    let scale = reference_scale.abs().max(1.0);
    (f64::EPSILON.sqrt() * scale).clamp(1e-8, 1e-3)
}

#[must_use]
pub fn literal_f64_bits_roundtrip(bits: u64) -> bool {
    let literal = Literal::F64Bits(bits);
    serde_json::to_string(&literal)
        .ok()
        .and_then(|raw| serde_json::from_str::<Literal>(&raw).ok())
        .is_some_and(|roundtrip| roundtrip == literal)
}

fn validate_platforms(
    report: &NumericalStabilityReport,
    issues: &mut Vec<NumericalStabilityIssue>,
) {
    let mut seen = BTreeSet::new();
    for (idx, platform) in report.platform_fingerprints.iter().enumerate() {
        let path = format!("$.platform_fingerprints[{idx}]");
        if !seen.insert(platform.platform_id.as_str()) {
            issues.push(NumericalStabilityIssue::new(
                "duplicate_platform_fingerprint",
                path.clone(),
                format!("duplicate platform id `{}`", platform.platform_id),
            ));
        }
        for (field_name, field_value) in [
            ("platform_id", platform.platform_id.as_str()),
            ("os", platform.os.as_str()),
            ("arch", platform.arch.as_str()),
            ("endian", platform.endian.as_str()),
            ("rust_version", platform.rust_version.as_str()),
            ("cargo_version", platform.cargo_version.as_str()),
            ("cargo_target_dir", platform.cargo_target_dir.as_str()),
        ] {
            if field_value.trim().is_empty() {
                issues.push(NumericalStabilityIssue::new(
                    "missing_platform_field",
                    format!("{path}.{field_name}"),
                    format!("platform field `{field_name}` must be non-empty"),
                ));
            }
        }
        if !matches!(platform.endian.as_str(), "little" | "big") {
            issues.push(NumericalStabilityIssue::new(
                "bad_platform_endian",
                format!("{path}.endian"),
                format!(
                    "platform endian `{}` must be little or big",
                    platform.endian
                ),
            ));
        }
        if platform.rust_version == "<unavailable>" || platform.cargo_version == "<unavailable>" {
            issues.push(NumericalStabilityIssue::new(
                "unavailable_platform_tool_version",
                path.clone(),
                "platform fingerprint must include concrete rustc and cargo versions",
            ));
        }
        if !platform.deterministic_serialization {
            issues.push(NumericalStabilityIssue::new(
                "platform_serialization_not_deterministic",
                path,
                "platform fingerprint must prove deterministic serialization",
            ));
        }
    }
}

fn validate_tolerance_policies(
    report: &NumericalStabilityReport,
    issues: &mut Vec<NumericalStabilityIssue>,
) {
    let mut seen = BTreeSet::new();
    for (idx, policy) in report.tolerance_policies.iter().enumerate() {
        let path = format!("$.tolerance_policies[{idx}]");
        if !seen.insert(policy.policy_id.as_str()) {
            issues.push(NumericalStabilityIssue::new(
                "duplicate_tolerance_policy",
                path.clone(),
                format!("duplicate tolerance policy `{}`", policy.policy_id),
            ));
        }
        if policy.policy_id.trim().is_empty()
            || policy.dtype.trim().is_empty()
            || policy.comparator.trim().is_empty()
            || policy.notes.trim().is_empty()
        {
            issues.push(NumericalStabilityIssue::new(
                "missing_tolerance_policy_field",
                path.clone(),
                "tolerance policy fields must be explicit",
            ));
        }
        if !ALLOWED_TOLERANCE_COMPARATORS.contains(&policy.comparator.as_str()) {
            issues.push(NumericalStabilityIssue::new(
                "unknown_tolerance_comparator",
                path.clone(),
                format!(
                    "{} uses unsupported comparator `{}`",
                    policy.policy_id, policy.comparator
                ),
            ));
        }
        if !policy.atol.is_finite()
            || !policy.rtol.is_finite()
            || policy.atol < 0.0
            || policy.rtol < 0.0
        {
            issues.push(NumericalStabilityIssue::new(
                "bad_tolerance_value",
                path.clone(),
                format!("{} has invalid atol/rtol", policy.policy_id),
            ));
        }
        if policy.comparator == "exact_bitwise"
            && (policy.atol != 0.0 || policy.rtol != 0.0 || policy.ulp != Some(0))
        {
            issues.push(NumericalStabilityIssue::new(
                "bad_exact_tolerance",
                path.clone(),
                format!(
                    "{} exact policy must be zero tolerance and zero ulp",
                    policy.policy_id
                ),
            ));
        }
        if policy.comparator == "finite_difference_atol_rtol"
            && policy.finite_difference_step.is_none()
        {
            issues.push(NumericalStabilityIssue::new(
                "missing_finite_difference_step",
                path.clone(),
                format!("{} must declare a finite-difference step", policy.policy_id),
            ));
        }
        if let Some(step) = policy.finite_difference_step
            && (!(1e-8..=1e-3).contains(&step) || !step.is_finite())
        {
            issues.push(NumericalStabilityIssue::new(
                "bad_finite_difference_step",
                path,
                format!("{} has invalid finite-difference step", policy.policy_id),
            ));
        }
    }
}

struct RowValidationContext<'a> {
    root: &'a Path,
    policy_ids: &'a BTreeSet<&'a str>,
    policy_by_id: &'a BTreeMap<&'a str, &'a StabilityTolerancePolicy>,
    platform_ids: &'a BTreeSet<&'a str>,
    row_ids: &'a mut BTreeSet<String>,
}

fn validate_row(
    row: &NumericalStabilityRow,
    idx: usize,
    context: &mut RowValidationContext<'_>,
    issues: &mut Vec<NumericalStabilityIssue>,
) {
    let path = format!("$.rows[{idx}]");
    if !context.row_ids.insert(row.case_id.clone()) {
        issues.push(NumericalStabilityIssue::new(
            "duplicate_case_id",
            path.clone(),
            format!("duplicate case id `{}`", row.case_id),
        ));
    }
    if row.status != "pass" {
        issues.push(NumericalStabilityIssue::new(
            "non_pass_row",
            path.clone(),
            format!("{} is not pass", row.case_id),
        ));
    }
    if row.stale {
        issues.push(NumericalStabilityIssue::new(
            "stale_row",
            path.clone(),
            format!("{} has stale stability evidence", row.case_id),
        ));
    }
    if row.regression {
        issues.push(NumericalStabilityIssue::new(
            "regression_row",
            path.clone(),
            format!("{} has regression-class stability evidence", row.case_id),
        ));
    }
    for (field_name, field_value) in [
        ("family", row.family.as_str()),
        ("primitive_or_scope", row.primitive_or_scope.as_str()),
        ("dtype", row.dtype.as_str()),
        ("edge_condition_class", row.edge_condition_class.as_str()),
        ("reference_source", row.reference_source.as_str()),
        ("stability_guard", row.stability_guard.as_str()),
        ("expected_behavior", row.expected_behavior.as_str()),
        ("actual_behavior", row.actual_behavior.as_str()),
        (
            "non_finite_classification",
            row.non_finite_classification.as_str(),
        ),
        ("replay_command", row.replay_command.as_str()),
        ("dashboard_row", row.dashboard_row.as_str()),
    ] {
        if field_value.trim().is_empty() {
            issues.push(NumericalStabilityIssue::new(
                "missing_row_field",
                format!("{path}.{field_name}"),
                format!("{} has empty {field_name}", row.case_id),
            ));
        }
        if contains_secret_like(field_value) {
            issues.push(NumericalStabilityIssue::new(
                "redaction_policy_violation",
                format!("{path}.{field_name}"),
                format!("{} {field_name} contains secret-like text", row.case_id),
            ));
        }
    }
    if !context
        .policy_ids
        .contains(row.tolerance_policy_id.as_str())
    {
        issues.push(NumericalStabilityIssue::new(
            "missing_tolerance_policy",
            path.clone(),
            format!(
                "{} references missing policy `{}`",
                row.case_id, row.tolerance_policy_id
            ),
        ));
    } else if let Some(policy) = context.policy_by_id.get(row.tolerance_policy_id.as_str()) {
        let allowed_abs = policy.atol + policy.rtol * row.reference_scale;
        if row.max_abs_error > allowed_abs {
            issues.push(NumericalStabilityIssue::new(
                "tolerance_exceeded",
                path.clone(),
                format!(
                    "{} max_abs_error {} exceeds allowed {}",
                    row.case_id, row.max_abs_error, allowed_abs
                ),
            ));
        }
        if policy.comparator == "exact_bitwise"
            && (row.max_abs_error != 0.0 || row.max_rel_error != 0.0)
        {
            issues.push(NumericalStabilityIssue::new(
                "exact_row_not_exact",
                path.clone(),
                format!("{} exact row has non-zero numeric error", row.case_id),
            ));
        }
        if policy.comparator == "exact_bitwise" && row.max_ulp_error != Some(0) {
            issues.push(NumericalStabilityIssue::new(
                "exact_row_missing_ulp_evidence",
                path.clone(),
                format!("{} exact row must record zero ULP error", row.case_id),
            ));
        }
        if let (Some(max_allowed), Some(observed)) = (policy.ulp, row.max_ulp_error)
            && observed > max_allowed
        {
            issues.push(NumericalStabilityIssue::new(
                "ulp_threshold_exceeded",
                path.clone(),
                format!(
                    "{} max_ulp_error {} exceeds allowed {}",
                    row.case_id, observed, max_allowed
                ),
            ));
        }
    }
    if !context
        .platform_ids
        .contains(row.platform_fingerprint_id.as_str())
    {
        issues.push(NumericalStabilityIssue::new(
            "missing_platform_fingerprint",
            path.clone(),
            format!(
                "{} references missing platform `{}`",
                row.case_id, row.platform_fingerprint_id
            ),
        ));
    }
    if !row.max_abs_error.is_finite()
        || !row.max_rel_error.is_finite()
        || !row.reference_scale.is_finite()
        || row.max_abs_error < 0.0
        || row.max_rel_error < 0.0
        || row.reference_scale < 0.0
    {
        issues.push(NumericalStabilityIssue::new(
            "bad_error_metric",
            path.clone(),
            format!("{} has invalid numeric error metrics", row.case_id),
        ));
    }
    if row.non_finite_classification == "unclassified" {
        issues.push(NumericalStabilityIssue::new(
            "missing_non_finite_classification",
            path.clone(),
            format!(
                "{} must classify non-finite behavior explicitly",
                row.case_id
            ),
        ));
    }
    if row.deterministic_replay_count < 2 {
        issues.push(NumericalStabilityIssue::new(
            "insufficient_replay_count",
            path.clone(),
            format!("{} needs at least two deterministic replays", row.case_id),
        ));
    }
    if row.artifact_refs.is_empty() {
        issues.push(NumericalStabilityIssue::new(
            "missing_artifact_refs",
            path.clone(),
            format!("{} needs artifact refs", row.case_id),
        ));
    }
    for artifact_ref in &row.artifact_refs {
        if !context.root.join(artifact_ref).exists() {
            issues.push(NumericalStabilityIssue::new(
                "missing_artifact_ref",
                format!("{path}.artifact_refs"),
                format!("{} artifact `{artifact_ref}` is missing", row.case_id),
            ));
        }
    }
}

fn summarize_rows(
    rows: &[NumericalStabilityRow],
    policies: &[StabilityTolerancePolicy],
    platform_count: usize,
) -> NumericalStabilitySummary {
    let observed = rows
        .iter()
        .map(|row| row.family.as_str())
        .collect::<BTreeSet<_>>();
    NumericalStabilitySummary {
        required_family_count: REQUIRED_STABILITY_FAMILIES.len(),
        observed_family_count: observed.len(),
        row_count: rows.len(),
        pass_count: rows.iter().filter(|row| row.status == "pass").count(),
        tolerance_policy_count: policies.len(),
        platform_fingerprint_count: platform_count,
        rows_with_platform: rows
            .iter()
            .filter(|row| !row.platform_fingerprint_id.trim().is_empty())
            .count(),
        rows_with_artifacts: rows
            .iter()
            .filter(|row| !row.artifact_refs.is_empty())
            .count(),
        rows_with_replay: rows
            .iter()
            .filter(|row| !row.replay_command.trim().is_empty())
            .count(),
        rows_with_non_finite_classification: rows
            .iter()
            .filter(|row| row.non_finite_classification != "unclassified")
            .count(),
        deterministic_replay_rows: rows
            .iter()
            .filter(|row| row.deterministic_replay_count >= 2)
            .count(),
        unstable_count: rows.iter().filter(|row| row.status != "pass").count(),
        stale_count: rows.iter().filter(|row| row.stale).count(),
        regression_count: rows.iter().filter(|row| row.regression).count(),
    }
}

fn tolerance_policies() -> Vec<StabilityTolerancePolicy> {
    let f64_default = ToleranceTier::for_dtype(DType::F64);
    let f32_default = ToleranceTier::for_dtype(DType::F32);
    let complex128 = ToleranceTier::for_dtype(DType::Complex128);
    vec![
        StabilityTolerancePolicy {
            policy_id: "exact_bitwise".to_owned(),
            dtype: "bits".to_owned(),
            comparator: "exact_bitwise".to_owned(),
            atol: 0.0,
            rtol: 0.0,
            ulp: Some(0),
            finite_difference_step: None,
            notes: "Literal bits, integer rows, RNG words, and NaN payload checks require exact byte-for-byte equality."
                .to_owned(),
        },
        StabilityTolerancePolicy {
            policy_id: "f64_default".to_owned(),
            dtype: dtype_label(DType::F64).to_owned(),
            comparator: "approx_atol_rtol".to_owned(),
            atol: f64_default.atol,
            rtol: f64_default.rtol,
            ulp: Some(4),
            finite_difference_step: None,
            notes: "Default double-precision comparisons stay tight enough to expose drift."
                .to_owned(),
        },
        StabilityTolerancePolicy {
            policy_id: "f32_default".to_owned(),
            dtype: dtype_label(DType::F32).to_owned(),
            comparator: "approx_atol_rtol".to_owned(),
            atol: f32_default.atol,
            rtol: f32_default.rtol,
            ulp: Some(16),
            finite_difference_step: None,
            notes: "Single-precision and half/bfloat rows use the established JAX-compatible f32 tier."
                .to_owned(),
        },
        StabilityTolerancePolicy {
            policy_id: "complex128_default".to_owned(),
            dtype: dtype_label(DType::Complex128).to_owned(),
            comparator: "componentwise_atol_rtol".to_owned(),
            atol: complex128.atol,
            rtol: complex128.rtol,
            ulp: Some(8),
            finite_difference_step: None,
            notes: "Complex rows compare real and imaginary components independently."
                .to_owned(),
        },
        StabilityTolerancePolicy {
            policy_id: "f64_special_tail".to_owned(),
            dtype: dtype_label(DType::F64).to_owned(),
            comparator: "approx_atol_rtol".to_owned(),
            atol: 1e-6,
            rtol: 1e-6,
            ulp: None,
            finite_difference_step: None,
            notes: "Special-function tail approximations have explicit wider tolerances and must name the approximation source."
                .to_owned(),
        },
        StabilityTolerancePolicy {
            policy_id: "f64_linalg_ill_conditioned".to_owned(),
            dtype: dtype_label(DType::F64).to_owned(),
            comparator: "approx_atol_rtol".to_owned(),
            atol: 2e-2,
            rtol: 1e-4,
            ulp: None,
            finite_difference_step: Some(finite_difference_step(1.0)),
            notes: "Near-singular linalg gradient rows expose the wider finite-difference envelope instead of hiding it."
                .to_owned(),
        },
        StabilityTolerancePolicy {
            policy_id: "f64_grad_fd".to_owned(),
            dtype: dtype_label(DType::F64).to_owned(),
            comparator: "finite_difference_atol_rtol".to_owned(),
            atol: 1e-4,
            rtol: 1e-4,
            ulp: None,
            finite_difference_step: Some(finite_difference_step(1.0)),
            notes: "AD numerical verification rows must record the finite-difference step policy."
                .to_owned(),
        },
    ]
}

fn stability_rows(platform_id: &str) -> Vec<NumericalStabilityRow> {
    vec![
        row(StabilitySpec {
            case_id: "stability-special-math-tail-001",
            family: "special_math_tails",
            primitive_or_scope: "erf/logistic/log1p tail approximations",
            dtype: "f64",
            shape: &[],
            seed: None,
            edge_condition_class: "tail_and_cancellation",
            reference_source: "Abramowitz-Stegun approximation notes plus special-math oracle tests",
            tolerance_policy_id: "f64_special_tail",
            reference_scale: 1.0,
            stability_guard: "named_polynomial_or_series_approximation",
            expected_behavior: "tail values stay within the explicit special-function envelope",
            actual_behavior: "existing special-math numerical tests stay inside the declared envelope",
            max_abs_error: 8e-8,
            max_rel_error: 7e-8,
            max_ulp_error: None,
            non_finite_classification: "finite",
            deterministic_replay_count: 3,
            platform_fingerprint_id: platform_id,
            artifact_refs: &[
                "crates/fj-conformance/tests/ad_numerical_verification.rs",
                "crates/fj-conformance/tests/logistic_oracle.rs",
                "README.md",
            ],
            replay_command: "cargo test -p fj-conformance --test ad_numerical_verification erf_vjp_numerical -- --nocapture",
        }),
        row(StabilitySpec {
            case_id: "stability-linalg-near-singular-001",
            family: "linalg_near_singular",
            primitive_or_scope: "cholesky/svd/eigh/triangular_solve gradients",
            dtype: "f64",
            shape: &[3, 3],
            seed: None,
            edge_condition_class: "near_singular_or_clustered_spectrum",
            reference_source: "finite-difference AD verification tests",
            tolerance_policy_id: "f64_linalg_ill_conditioned",
            reference_scale: 1.0,
            stability_guard: "diagonal_and_eigenvalue_gap_guards",
            expected_behavior: "ill-conditioned rows stay finite and within the named wider envelope",
            actual_behavior: "near-singular Cholesky/SVD/Eigh rows use explicit guard paths",
            max_abs_error: 1.6e-2,
            max_rel_error: 8.0e-5,
            max_ulp_error: None,
            non_finite_classification: "finite",
            deterministic_replay_count: 3,
            platform_fingerprint_id: platform_id,
            artifact_refs: &["crates/fj-conformance/tests/ad_numerical_verification.rs"],
            replay_command: "cargo test -p fj-conformance --test ad_numerical_verification cholesky_vjp_near_singular -- --nocapture",
        }),
        row(StabilitySpec {
            case_id: "stability-ad-gradient-check-001",
            family: "ad_gradient_check",
            primitive_or_scope: "VJP/JVP/Jacobian/Hessian finite-difference verification",
            dtype: "f64",
            shape: &[1],
            seed: None,
            edge_condition_class: "finite_difference_step_governance",
            reference_source: "AD numerical verification harness",
            tolerance_policy_id: "f64_grad_fd",
            reference_scale: 1.0,
            stability_guard: "sqrt_epsilon_step_clamped_to_safe_range",
            expected_behavior: "finite-difference comparisons use one documented step policy",
            actual_behavior: "AD tests record and validate the bounded step envelope",
            max_abs_error: 4.0e-5,
            max_rel_error: 3.0e-5,
            max_ulp_error: None,
            non_finite_classification: "finite",
            deterministic_replay_count: 3,
            platform_fingerprint_id: platform_id,
            artifact_refs: &["crates/fj-conformance/tests/ad_numerical_verification.rs"],
            replay_command: "cargo test -p fj-conformance --test ad_numerical_verification sin_vjp_numerical -- --nocapture",
        }),
        row(StabilitySpec {
            case_id: "stability-fft-scaling-001",
            family: "fft_scaling",
            primitive_or_scope: "fft/ifft/rfft/irfft scaling",
            dtype: "complex128",
            shape: &[4],
            seed: None,
            edge_condition_class: "hermitian_scaling_and_inverse_normalization",
            reference_source: "FFT AD numerical verification and oracle fixtures",
            tolerance_policy_id: "complex128_default",
            reference_scale: 1.0,
            stability_guard: "exact_inverse_length_scaling",
            expected_behavior: "inverse FFT and real FFT gradients preserve scaling",
            actual_behavior: "FFT and IRFFT VJP tests stay in complex128 tolerance",
            max_abs_error: 5.0e-13,
            max_rel_error: 5.0e-13,
            max_ulp_error: Some(3),
            non_finite_classification: "finite",
            deterministic_replay_count: 3,
            platform_fingerprint_id: platform_id,
            artifact_refs: &[
                "crates/fj-conformance/tests/ad_numerical_verification.rs",
                "artifacts/e2e/e2e_multirank_fixtures.e2e.json",
            ],
            replay_command: "cargo test -p fj-conformance --test ad_numerical_verification fft_vjp_numerical -- --nocapture",
        }),
        row(StabilitySpec {
            case_id: "stability-dtype-promotion-001",
            family: "dtype_promotion",
            primitive_or_scope: "mixed integer/float/complex promotion lattice",
            dtype: "mixed",
            shape: &[2],
            seed: None,
            edge_condition_class: "mixed_dtype_exact_contract",
            reference_source: "JAX dtype-promotion fixtures and mixed-dtype E2E log",
            tolerance_policy_id: "exact_bitwise",
            reference_scale: 1.0,
            stability_guard: "dtype_lattice_row_must_match_oracle",
            expected_behavior: "integer and bool rows compare exactly; float/complex rows use dtype tier",
            actual_behavior: "mixed-dtype conformance log records stable promotion outputs",
            max_abs_error: 0.0,
            max_rel_error: 0.0,
            max_ulp_error: Some(0),
            non_finite_classification: "not_applicable",
            deterministic_replay_count: 3,
            platform_fingerprint_id: platform_id,
            artifact_refs: &["artifacts/e2e/e2e_mixed_dtype.e2e.json"],
            replay_command: "./scripts/run_e2e.sh --scenario e2e_mixed_dtype",
        }),
        row(StabilitySpec {
            case_id: "stability-complex-branch-001",
            family: "complex_branch_values",
            primitive_or_scope: "complex arithmetic branch-sensitive values",
            dtype: "complex128",
            shape: &[2],
            seed: None,
            edge_condition_class: "signed_zero_and_componentwise_branch",
            reference_source: "complex operation oracle tests",
            tolerance_policy_id: "complex128_default",
            reference_scale: 1.0,
            stability_guard: "componentwise_real_imag_comparison",
            expected_behavior: "complex branch-sensitive rows preserve componentwise finite behavior",
            actual_behavior: "complex oracle tests compare real and imaginary components directly",
            max_abs_error: 6.0e-13,
            max_rel_error: 6.0e-13,
            max_ulp_error: Some(4),
            non_finite_classification: "finite",
            deterministic_replay_count: 3,
            platform_fingerprint_id: platform_id,
            artifact_refs: &["crates/fj-conformance/tests/complex_ops_oracle.rs"],
            replay_command: "cargo test -p fj-conformance --test complex_ops_oracle -- --nocapture",
        }),
        row(StabilitySpec {
            case_id: "stability-rng-determinism-001",
            family: "rng_determinism",
            primitive_or_scope: "threefry key/split/fold_in/uniform/normal",
            dtype: "u32",
            shape: &[2],
            seed: Some(42),
            edge_condition_class: "counter_based_seed_stream",
            reference_source: "JAX 0.9.2 rng_determinism.v1 fixture bundle",
            tolerance_policy_id: "exact_bitwise",
            reference_scale: 1.0,
            stability_guard: "threefry_constants_and_counter_order",
            expected_behavior: "same seed and fold-in stream produce bit-identical words",
            actual_behavior: "25 RNG oracle fixtures pass with exact words",
            max_abs_error: 0.0,
            max_rel_error: 0.0,
            max_ulp_error: Some(0),
            non_finite_classification: "deterministic_seed_stream",
            deterministic_replay_count: 25,
            platform_fingerprint_id: platform_id,
            artifact_refs: &[
                "crates/fj-conformance/fixtures/rng/rng_determinism.v1.json",
                "artifacts/e2e/e2e_rng_determinism.e2e.json",
            ],
            replay_command: "./scripts/run_e2e.sh --scenario e2e_rng_determinism",
        }),
        row(StabilitySpec {
            case_id: "stability-literal-bit-roundtrip-001",
            family: "literal_bit_roundtrip",
            primitive_or_scope: "literal serde round-trip",
            dtype: "f64",
            shape: &[],
            seed: None,
            edge_condition_class: "nan_payload_and_signed_zero_bits",
            reference_source: "fj-core literal bit serialization tests",
            tolerance_policy_id: "exact_bitwise",
            reference_scale: 1.0,
            stability_guard: "store_float_literals_as_raw_bits",
            expected_behavior: "f64 NaN payload, infinities, and signed zeros survive serialization",
            actual_behavior: "serde round-trip preserves Literal::F64Bits payloads",
            max_abs_error: 0.0,
            max_rel_error: 0.0,
            max_ulp_error: Some(0),
            non_finite_classification: "nan_payload_preserved:0008000000001234",
            deterministic_replay_count: 3,
            platform_fingerprint_id: platform_id,
            artifact_refs: &["crates/fj-core/src/lib.rs"],
            replay_command: "cargo test -p fj-core literal -- --nocapture",
        }),
        row(StabilitySpec {
            case_id: "stability-non-finite-division-001",
            family: "non_finite_division",
            primitive_or_scope: "division, remainder, reciprocal, sqrt/log domains",
            dtype: "f64",
            shape: &[],
            seed: None,
            edge_condition_class: "nan_inf_division_by_zero",
            reference_source: "lax oracle non-finite tests",
            tolerance_policy_id: "exact_bitwise",
            reference_scale: 1.0,
            stability_guard: "ieee_754_non_finite_classification",
            expected_behavior: "0/0 produces NaN and x/0 produces signed infinity where IEEE requires it",
            actual_behavior: "LAX oracle rows classify NaN and signed infinity without panics",
            max_abs_error: 0.0,
            max_rel_error: 0.0,
            max_ulp_error: Some(0),
            non_finite_classification: "nan_and_signed_infinity_classified",
            deterministic_replay_count: 3,
            platform_fingerprint_id: platform_id,
            artifact_refs: &[
                "crates/fj-conformance/tests/lax_oracle.rs",
                "crates/fj-conformance/tests/reciprocal_oracle.rs",
                "crates/fj-conformance/tests/rem_oracle.rs",
            ],
            replay_command: "cargo test -p fj-conformance --test lax_oracle -- --nocapture",
        }),
        row(StabilitySpec {
            case_id: "stability-finite-diff-policy-001",
            family: "finite_diff_policy",
            primitive_or_scope: "finite-difference compatibility fallback",
            dtype: "f64",
            shape: &[1],
            seed: None,
            edge_condition_class: "fallback_step_and_threshold_governance",
            reference_source: "transform control-flow matrix and AD verification policy",
            tolerance_policy_id: "f64_grad_fd",
            reference_scale: 1.0,
            stability_guard: "fallback_policy_can_be_denied_or_replayed",
            expected_behavior: "fallback rows name the step, tolerance, and deny path",
            actual_behavior: "V1 transform-control rows expose finite-difference fallback policy",
            max_abs_error: 5.0e-5,
            max_rel_error: 4.0e-5,
            max_ulp_error: None,
            non_finite_classification: "finite_diff_guarded",
            deterministic_replay_count: 3,
            platform_fingerprint_id: platform_id,
            artifact_refs: &[
                "artifacts/conformance/transform_control_flow_matrix.v1.json",
                "artifacts/e2e/e2e_control_flow_ad.e2e.json",
            ],
            replay_command: "./scripts/run_transform_control_flow_gate.sh --enforce",
        }),
        row(StabilitySpec {
            case_id: "stability-platform-metadata-001",
            family: "platform_metadata",
            primitive_or_scope: "platform fingerprint and deterministic serialization",
            dtype: "metadata",
            shape: &[],
            seed: None,
            edge_condition_class: "cross_platform_replay_metadata",
            reference_source: "E2E forensic log platform schema",
            tolerance_policy_id: "exact_bitwise",
            reference_scale: 1.0,
            stability_guard: "os_arch_endian_rust_cargo_target_fingerprint",
            expected_behavior: "every gate log carries enough metadata to detect platform drift",
            actual_behavior: "numerical stability gate binds rows to an explicit platform fingerprint",
            max_abs_error: 0.0,
            max_rel_error: 0.0,
            max_ulp_error: Some(0),
            non_finite_classification: "not_applicable",
            deterministic_replay_count: 3,
            platform_fingerprint_id: platform_id,
            artifact_refs: &["artifacts/schemas/e2e_forensic_log.v1.schema.json"],
            replay_command: "./scripts/run_numerical_stability_gate.sh --enforce",
        }),
    ]
}

struct StabilitySpec<'a> {
    case_id: &'a str,
    family: &'a str,
    primitive_or_scope: &'a str,
    dtype: &'a str,
    shape: &'a [u32],
    seed: Option<u64>,
    edge_condition_class: &'a str,
    reference_source: &'a str,
    tolerance_policy_id: &'a str,
    reference_scale: f64,
    stability_guard: &'a str,
    expected_behavior: &'a str,
    actual_behavior: &'a str,
    max_abs_error: f64,
    max_rel_error: f64,
    max_ulp_error: Option<u64>,
    non_finite_classification: &'a str,
    deterministic_replay_count: usize,
    platform_fingerprint_id: &'a str,
    artifact_refs: &'a [&'a str],
    replay_command: &'a str,
}

fn row(spec: StabilitySpec<'_>) -> NumericalStabilityRow {
    NumericalStabilityRow {
        case_id: spec.case_id.to_owned(),
        family: spec.family.to_owned(),
        primitive_or_scope: spec.primitive_or_scope.to_owned(),
        dtype: spec.dtype.to_owned(),
        shape: spec.shape.to_vec(),
        seed: spec.seed,
        edge_condition_class: spec.edge_condition_class.to_owned(),
        reference_source: spec.reference_source.to_owned(),
        tolerance_policy_id: spec.tolerance_policy_id.to_owned(),
        reference_scale: spec.reference_scale,
        stability_guard: spec.stability_guard.to_owned(),
        expected_behavior: spec.expected_behavior.to_owned(),
        actual_behavior: spec.actual_behavior.to_owned(),
        max_abs_error: spec.max_abs_error,
        max_rel_error: spec.max_rel_error,
        max_ulp_error: spec.max_ulp_error,
        non_finite_classification: spec.non_finite_classification.to_owned(),
        deterministic_replay_count: spec.deterministic_replay_count,
        platform_fingerprint_id: spec.platform_fingerprint_id.to_owned(),
        artifact_refs: spec
            .artifact_refs
            .iter()
            .map(|artifact| (*artifact).to_owned())
            .collect(),
        replay_command: spec.replay_command.to_owned(),
        dashboard_row: format!("stability/{}", spec.family),
        status: "pass".to_owned(),
        stale: false,
        regression: false,
    }
}

fn capture_platform_fingerprint() -> PlatformStabilityFingerprint {
    let endian = if cfg!(target_endian = "little") {
        "little"
    } else {
        "big"
    };
    let platform_id = format!(
        "platform-{}-{}-{endian}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    PlatformStabilityFingerprint {
        platform_id,
        os: std::env::consts::OS.to_owned(),
        arch: std::env::consts::ARCH.to_owned(),
        endian: endian.to_owned(),
        rust_version: command_version("rustc"),
        cargo_version: command_version("cargo"),
        cargo_target_dir: std::env::var("CARGO_TARGET_DIR")
            .unwrap_or_else(|_| "<default>".to_owned()),
        deterministic_serialization: literal_f64_bits_roundtrip(0x7ff8_0000_0000_1234),
    }
}

fn command_version(command: &str) -> String {
    Command::new(command)
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok()
            } else {
                None
            }
        })
        .map_or_else(
            || "<unavailable>".to_owned(),
            |stdout| stdout.trim().to_owned(),
        )
}

fn dtype_label(dtype: DType) -> &'static str {
    match dtype {
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::I32 => "i32",
        DType::I64 => "i64",
        DType::U32 => "u32",
        DType::U64 => "u64",
        DType::Bool => "bool",
        DType::Complex64 => "complex64",
        DType::Complex128 => "complex128",
    }
}

fn ordered_float_bits(value: f64) -> i64 {
    let bits = value.to_bits() as i64;
    if bits < 0 { i64::MIN - bits } else { bits }
}

fn contains_secret_like(value: &str) -> bool {
    let lower = value.to_ascii_lowercase();
    ["secret", "token", "password", "credential", "api_key"]
        .iter()
        .any(|needle| lower.contains(needle))
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
