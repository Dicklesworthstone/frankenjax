use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const ORACLE_RECAPTURE_MATRIX_SCHEMA_VERSION: &str = "frankenjax.oracle-recapture-matrix.v1";
pub const ORACLE_DRIFT_REPORT_SCHEMA_VERSION: &str = "frankenjax.oracle-drift-report.v1";
pub const EXPECTED_ORACLE_CASE_TOTAL: usize = 848;

const REQUIRED_FAMILIES: &[&str] = &[
    "transforms",
    "rng",
    "linalg_fft",
    "composition",
    "dtype_promotion",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyAnchor {
    pub path: String,
    pub symbol: String,
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleFixtureSpec {
    pub family_id: String,
    pub display_name: String,
    pub fixture_path: String,
    pub expected_schema_version: String,
    pub expected_case_count: usize,
    pub expected_oracle_version_prefix: String,
    pub expected_x64_enabled: Option<bool>,
    pub legacy_anchors: Vec<LegacyAnchor>,
    pub recapture_command: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleRecaptureIssue {
    pub code: String,
    pub family_id: Option<String>,
    pub path: String,
    pub message: String,
}

impl OracleRecaptureIssue {
    #[must_use]
    pub fn new(
        code: impl Into<String>,
        family_id: Option<String>,
        path: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code: code.into(),
            family_id,
            path: path.into(),
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleFixtureRow {
    pub family_id: String,
    pub display_name: String,
    pub fixture_path: String,
    pub schema_version: String,
    pub expected_schema_version: String,
    pub expected_case_count: usize,
    pub actual_case_count: usize,
    pub oracle_version: String,
    pub expected_oracle_version_prefix: String,
    pub x64_enabled: Option<bool>,
    pub expected_x64_enabled: Option<bool>,
    pub capture_mode: String,
    pub generated_by: String,
    pub generated_at_unix_ms: Option<u64>,
    pub legacy_anchors: Vec<LegacyAnchor>,
    pub recapture_command: Vec<String>,
    pub fixture_sha256: String,
    pub duplicate_case_ids: Vec<String>,
    pub issue_codes: Vec<String>,
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleRecaptureMatrix {
    pub schema_version: String,
    pub generated_at_unix_ms: u64,
    pub repo_root: String,
    pub expected_total_cases: usize,
    pub actual_total_cases: usize,
    pub required_families: Vec<String>,
    pub rows: Vec<OracleFixtureRow>,
    pub issues: Vec<OracleRecaptureIssue>,
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleDriftRow {
    pub family_id: String,
    pub status: String,
    pub current_case_count: usize,
    pub baseline_case_count: Option<usize>,
    pub current_sha256: String,
    pub baseline_sha256: Option<String>,
    pub issue_codes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleDriftReport {
    pub schema_version: String,
    pub generated_at_unix_ms: u64,
    pub baseline_path: Option<String>,
    pub require_baseline: bool,
    pub expected_total_cases: usize,
    pub actual_total_cases: usize,
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub changed: Vec<String>,
    pub unsupported: Vec<String>,
    pub skipped: Vec<String>,
    pub rows: Vec<OracleDriftRow>,
    pub issues: Vec<OracleRecaptureIssue>,
    pub gate_status: String,
}

#[must_use]
pub fn default_oracle_fixture_specs() -> Vec<OracleFixtureSpec> {
    vec![
        OracleFixtureSpec {
            family_id: "transforms".to_owned(),
            display_name: "Transform jit/grad/vmap/lax/control-flow/mixed-dtype".to_owned(),
            fixture_path:
                "crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json"
                    .to_owned(),
            expected_schema_version: "frankenjax.transform-fixtures.v1".to_owned(),
            expected_case_count: 613,
            expected_oracle_version_prefix: "0.9.2".to_owned(),
            expected_x64_enabled: Some(true),
            legacy_anchors: vec![
                anchor("tests/jax_jit_test.py", "JaxJitTest"),
                anchor("tests/lax_autodiff_test.py", "LAX grad/vjp fixtures"),
                anchor("tests/lax_vmap_test.py", "batching/vmap cases"),
                anchor("tests/lax_test.py", "LAX primitive cases"),
            ],
            recapture_command: vec![
                "python3".to_owned(),
                "crates/fj-conformance/scripts/capture_legacy_fixtures.py".to_owned(),
                "--legacy-root".to_owned(),
                "legacy_jax_code/jax".to_owned(),
                "--output".to_owned(),
                "crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json"
                    .to_owned(),
                "--strict".to_owned(),
            ],
        },
        OracleFixtureSpec {
            family_id: "rng".to_owned(),
            display_name: "RNG determinism key/split/fold_in/uniform/normal".to_owned(),
            fixture_path: "crates/fj-conformance/fixtures/rng/rng_determinism.v1.json".to_owned(),
            expected_schema_version: "frankenjax.rng-fixtures.v1".to_owned(),
            expected_case_count: 25,
            expected_oracle_version_prefix: "0.9.2".to_owned(),
            expected_x64_enabled: Some(true),
            legacy_anchors: vec![anchor(
                "tests/random_test.py",
                "random key/value determinism",
            )],
            recapture_command: vec![
                "python3".to_owned(),
                "crates/fj-conformance/scripts/capture_legacy_fixtures.py".to_owned(),
                "--legacy-root".to_owned(),
                "legacy_jax_code/jax".to_owned(),
                "--output".to_owned(),
                "crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json"
                    .to_owned(),
                "--rng-output".to_owned(),
                "crates/fj-conformance/fixtures/rng/rng_determinism.v1.json".to_owned(),
                "--strict".to_owned(),
            ],
        },
        OracleFixtureSpec {
            family_id: "linalg_fft".to_owned(),
            display_name: "Linear algebra and FFT oracle".to_owned(),
            fixture_path: "crates/fj-conformance/fixtures/linalg_fft_oracle.v1.json".to_owned(),
            expected_schema_version: "frankenjax.linalg-fft-oracle.v2".to_owned(),
            expected_case_count: 33,
            expected_oracle_version_prefix: "0.9.2".to_owned(),
            expected_x64_enabled: Some(true),
            legacy_anchors: vec![
                anchor(
                    "tests/linalg_test.py",
                    "cholesky/qr/svd/eigh/triangular solve",
                ),
                anchor("tests/fft_test.py", "fft/ifft/rfft/irfft"),
            ],
            recapture_command: vec![
                "python3".to_owned(),
                "crates/fj-conformance/scripts/capture_linalg_fft_oracle.py".to_owned(),
                "--legacy-root".to_owned(),
                "legacy_jax_code/jax".to_owned(),
                "--output".to_owned(),
                "crates/fj-conformance/fixtures/linalg_fft_oracle.v1.json".to_owned(),
            ],
        },
        OracleFixtureSpec {
            family_id: "composition".to_owned(),
            display_name: "Transform composition jit+grad/grad+grad/vmap+grad".to_owned(),
            fixture_path: "crates/fj-conformance/fixtures/composition_oracle.v1.json".to_owned(),
            expected_schema_version: "frankenjax.composition-oracle.v1".to_owned(),
            expected_case_count: 15,
            expected_oracle_version_prefix: "0.9.2".to_owned(),
            expected_x64_enabled: Some(true),
            legacy_anchors: vec![
                anchor("tests/api_test.py", "jit/grad/vmap composition semantics"),
                anchor(
                    "tests/lax_autodiff_test.py",
                    "higher-order grad/jacobian/hessian",
                ),
            ],
            recapture_command: vec![
                "python3".to_owned(),
                "crates/fj-conformance/scripts/capture_composition_oracle.py".to_owned(),
                "--output".to_owned(),
                "crates/fj-conformance/fixtures/composition_oracle.v1.json".to_owned(),
            ],
        },
        OracleFixtureSpec {
            family_id: "dtype_promotion".to_owned(),
            display_name: "Dtype promotion add/mul matrix".to_owned(),
            fixture_path: "crates/fj-conformance/fixtures/dtype_promotion_oracle.v1.json"
                .to_owned(),
            expected_schema_version: "frankenjax.dtype-promotion-oracle.v1".to_owned(),
            expected_case_count: 162,
            expected_oracle_version_prefix: "0.9.2".to_owned(),
            expected_x64_enabled: Some(true),
            legacy_anchors: vec![anchor(
                "tests/dtypes_test.py",
                "JAX dtype promotion lattice and weak type rules",
            )],
            recapture_command: vec![
                "python3".to_owned(),
                "crates/fj-conformance/scripts/capture_dtype_promotion_oracle.py".to_owned(),
                "--output".to_owned(),
                "crates/fj-conformance/fixtures/dtype_promotion_oracle.v1.json".to_owned(),
            ],
        },
    ]
}

pub fn build_oracle_recapture_matrix(root: &Path) -> OracleRecaptureMatrix {
    build_oracle_recapture_matrix_from_specs(root, &default_oracle_fixture_specs())
}

pub fn build_oracle_recapture_matrix_from_specs(
    root: &Path,
    specs: &[OracleFixtureSpec],
) -> OracleRecaptureMatrix {
    let rows: Vec<OracleFixtureRow> = specs
        .iter()
        .map(|spec| build_fixture_row(root, spec))
        .collect();
    let actual_total_cases = rows.iter().map(|row| row.actual_case_count).sum();

    let mut matrix = OracleRecaptureMatrix {
        schema_version: ORACLE_RECAPTURE_MATRIX_SCHEMA_VERSION.to_owned(),
        generated_at_unix_ms: now_unix_ms(),
        repo_root: root.display().to_string(),
        expected_total_cases: EXPECTED_ORACLE_CASE_TOTAL,
        actual_total_cases,
        required_families: REQUIRED_FAMILIES
            .iter()
            .map(|family| (*family).to_owned())
            .collect(),
        rows,
        issues: Vec::new(),
        status: "pass".to_owned(),
    };
    matrix.issues = validate_oracle_recapture_matrix(&matrix);
    if !matrix.issues.is_empty() {
        matrix.status = "fail".to_owned();
    }
    matrix
}

#[must_use]
pub fn validate_oracle_recapture_matrix(
    matrix: &OracleRecaptureMatrix,
) -> Vec<OracleRecaptureIssue> {
    let mut issues = Vec::new();

    if matrix.schema_version != ORACLE_RECAPTURE_MATRIX_SCHEMA_VERSION {
        issues.push(OracleRecaptureIssue::new(
            "unsupported_schema_version",
            None,
            "$.schema_version",
            format!(
                "expected {}, got {}",
                ORACLE_RECAPTURE_MATRIX_SCHEMA_VERSION, matrix.schema_version
            ),
        ));
    }

    let mut seen = BTreeSet::new();
    for row in &matrix.rows {
        if !seen.insert(row.family_id.clone()) {
            issues.push(OracleRecaptureIssue::new(
                "duplicate_family",
                Some(row.family_id.clone()),
                "$.rows",
                format!("family {} appears more than once", row.family_id),
            ));
        }
    }

    for family in &matrix.required_families {
        if !seen.contains(family) {
            issues.push(OracleRecaptureIssue::new(
                "missing_family",
                Some(family.clone()),
                "$.rows",
                format!("required family {family} is missing from the recapture matrix"),
            ));
        }
    }

    if matrix.actual_total_cases != matrix.expected_total_cases {
        issues.push(OracleRecaptureIssue::new(
            "total_case_count_mismatch",
            None,
            "$.actual_total_cases",
            format!(
                "expected {} oracle cases, found {}",
                matrix.expected_total_cases, matrix.actual_total_cases
            ),
        ));
    }

    for row in &matrix.rows {
        issues.extend(validate_fixture_row(row));
    }

    issues
}

#[must_use]
pub fn validate_oracle_drift_report(report: &OracleDriftReport) -> Vec<OracleRecaptureIssue> {
    let mut issues = Vec::new();
    if report.schema_version != ORACLE_DRIFT_REPORT_SCHEMA_VERSION {
        issues.push(OracleRecaptureIssue::new(
            "unsupported_schema_version",
            None,
            "$.schema_version",
            format!(
                "expected {}, got {}",
                ORACLE_DRIFT_REPORT_SCHEMA_VERSION, report.schema_version
            ),
        ));
    }
    if report.require_baseline && report.baseline_path.is_none() {
        issues.push(OracleRecaptureIssue::new(
            "missing_baseline",
            None,
            "$.baseline_path",
            "strict drift mode requires an explicit baseline matrix",
        ));
    }
    if report.actual_total_cases != report.expected_total_cases {
        issues.push(OracleRecaptureIssue::new(
            "total_case_count_mismatch",
            None,
            "$.actual_total_cases",
            format!(
                "expected {} oracle cases, found {}",
                report.expected_total_cases, report.actual_total_cases
            ),
        ));
    }
    issues.extend(report.issues.iter().cloned());
    issues
}

#[must_use]
pub fn oracle_drift_report(
    current: &OracleRecaptureMatrix,
    baseline: Option<&OracleRecaptureMatrix>,
    baseline_path: Option<String>,
    require_baseline: bool,
) -> OracleDriftReport {
    let baseline_rows: BTreeMap<&str, &OracleFixtureRow> = baseline
        .map(|matrix| {
            matrix
                .rows
                .iter()
                .map(|row| (row.family_id.as_str(), row))
                .collect()
        })
        .unwrap_or_default();

    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();
    let mut unsupported = Vec::new();
    let mut skipped = Vec::new();
    let mut rows = Vec::new();

    for row in &current.rows {
        let baseline_row = baseline_rows.get(row.family_id.as_str()).copied();
        let mut issue_codes = row.issue_codes.clone();
        let status = if row
            .issue_codes
            .iter()
            .any(|code| code == "missing_recapture_command")
        {
            unsupported.push(row.family_id.clone());
            "unsupported_recapture"
        } else if row.status != "pass" {
            "current_invalid"
        } else if let Some(baseline_row) = baseline_row {
            if baseline_row.fixture_sha256 != row.fixture_sha256
                || baseline_row.actual_case_count != row.actual_case_count
            {
                changed.push(row.family_id.clone());
                issue_codes.push("fixture_drift".to_owned());
                "changed"
            } else {
                "unchanged"
            }
        } else if baseline.is_some() {
            added.push(row.family_id.clone());
            issue_codes.push("added_family".to_owned());
            "added"
        } else {
            skipped.push(row.family_id.clone());
            "baseline_not_provided"
        };

        rows.push(OracleDriftRow {
            family_id: row.family_id.clone(),
            status: status.to_owned(),
            current_case_count: row.actual_case_count,
            baseline_case_count: baseline_row.map(|row| row.actual_case_count),
            current_sha256: row.fixture_sha256.clone(),
            baseline_sha256: baseline_row.map(|row| row.fixture_sha256.clone()),
            issue_codes,
        });
    }

    if let Some(baseline) = baseline {
        let current_families: BTreeSet<&str> = current
            .rows
            .iter()
            .map(|row| row.family_id.as_str())
            .collect();
        for row in &baseline.rows {
            if !current_families.contains(row.family_id.as_str()) {
                removed.push(row.family_id.clone());
                rows.push(OracleDriftRow {
                    family_id: row.family_id.clone(),
                    status: "removed".to_owned(),
                    current_case_count: 0,
                    baseline_case_count: Some(row.actual_case_count),
                    current_sha256: String::new(),
                    baseline_sha256: Some(row.fixture_sha256.clone()),
                    issue_codes: vec!["removed_family".to_owned()],
                });
            }
        }
    }

    let mut issues = current.issues.clone();
    if require_baseline && baseline.is_none() {
        issues.push(OracleRecaptureIssue::new(
            "missing_baseline",
            None,
            "$.baseline_path",
            "strict drift mode requires an explicit baseline matrix",
        ));
    }
    if !added.is_empty() {
        issues.push(OracleRecaptureIssue::new(
            "added_families",
            None,
            "$.added",
            format!("added fixture families: {}", added.join(", ")),
        ));
    }
    if !removed.is_empty() {
        issues.push(OracleRecaptureIssue::new(
            "removed_families",
            None,
            "$.removed",
            format!("removed fixture families: {}", removed.join(", ")),
        ));
    }
    if !changed.is_empty() {
        issues.push(OracleRecaptureIssue::new(
            "changed_families",
            None,
            "$.changed",
            format!("changed fixture families: {}", changed.join(", ")),
        ));
    }

    OracleDriftReport {
        schema_version: ORACLE_DRIFT_REPORT_SCHEMA_VERSION.to_owned(),
        generated_at_unix_ms: now_unix_ms(),
        baseline_path,
        require_baseline,
        expected_total_cases: current.expected_total_cases,
        actual_total_cases: current.actual_total_cases,
        added,
        removed,
        changed,
        unsupported,
        skipped,
        rows,
        gate_status: if issues.is_empty() { "pass" } else { "fail" }.to_owned(),
        issues,
    }
}

pub fn read_oracle_recapture_matrix(path: &Path) -> Result<OracleRecaptureMatrix, std::io::Error> {
    let raw = fs::read_to_string(path)?;
    serde_json::from_str(&raw).map_err(std::io::Error::other)
}

pub fn write_oracle_json<T: Serialize>(path: &Path, value: &T) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let raw = serde_json::to_string_pretty(value).map_err(std::io::Error::other)?;
    fs::write(path, format!("{raw}\n"))
}

#[must_use]
pub fn oracle_recapture_markdown(
    matrix: &OracleRecaptureMatrix,
    drift: &OracleDriftReport,
) -> String {
    let mut out = String::new();
    out.push_str("# Oracle Recapture Matrix\n\n");
    out.push_str(&format!(
        "- matrix status: `{}`\n- drift gate: `{}`\n- cases: `{}/{}`\n\n",
        matrix.status, drift.gate_status, matrix.actual_total_cases, matrix.expected_total_cases
    ));
    out.push_str("| Family | Cases | Oracle | X64 | Status | Recapture |\n");
    out.push_str("|---|---:|---|---|---|---|\n");
    for row in &matrix.rows {
        let recapture = if row.recapture_command.is_empty() {
            "`missing`".to_owned()
        } else {
            format!("`{}`", row.recapture_command.join(" "))
        };
        out.push_str(&format!(
            "| `{}` | {}/{} | `{}` | `{}` | `{}` | {} |\n",
            row.family_id,
            row.actual_case_count,
            row.expected_case_count,
            row.oracle_version,
            row.x64_enabled
                .map(|enabled| enabled.to_string())
                .unwrap_or_else(|| "unknown".to_owned()),
            row.status,
            recapture
        ));
    }
    if !drift.issues.is_empty() {
        out.push_str("\n## Gate Issues\n\n");
        for issue in &drift.issues {
            out.push_str(&format!(
                "- `{}` `{}`: {}\n",
                issue.code,
                issue.family_id.as_deref().unwrap_or("-"),
                issue.message
            ));
        }
    }
    out
}

#[must_use]
pub fn oracle_recapture_summary_json(
    matrix: &OracleRecaptureMatrix,
    drift: &OracleDriftReport,
) -> Value {
    json!({
        "matrix_status": matrix.status,
        "drift_gate_status": drift.gate_status,
        "expected_total_cases": matrix.expected_total_cases,
        "actual_total_cases": matrix.actual_total_cases,
        "required_families": matrix.required_families,
        "issue_codes": drift.issues.iter().map(|issue| issue.code.as_str()).collect::<Vec<_>>(),
        "unsupported": drift.unsupported,
        "changed": drift.changed,
        "removed": drift.removed,
        "added": drift.added,
    })
}

fn build_fixture_row(root: &Path, spec: &OracleFixtureSpec) -> OracleFixtureRow {
    let path = root.join(&spec.fixture_path);
    let mut issue_codes = Vec::new();

    let raw = match fs::read(&path) {
        Ok(raw) => raw,
        Err(_) => {
            issue_codes.push("missing_fixture".to_owned());
            return OracleFixtureRow {
                family_id: spec.family_id.clone(),
                display_name: spec.display_name.clone(),
                fixture_path: spec.fixture_path.clone(),
                schema_version: "missing".to_owned(),
                expected_schema_version: spec.expected_schema_version.clone(),
                expected_case_count: spec.expected_case_count,
                actual_case_count: 0,
                oracle_version: "missing".to_owned(),
                expected_oracle_version_prefix: spec.expected_oracle_version_prefix.clone(),
                x64_enabled: None,
                expected_x64_enabled: spec.expected_x64_enabled,
                capture_mode: "missing".to_owned(),
                generated_by: "missing".to_owned(),
                generated_at_unix_ms: None,
                legacy_anchors: spec.legacy_anchors.clone(),
                recapture_command: spec.recapture_command.clone(),
                fixture_sha256: String::new(),
                duplicate_case_ids: Vec::new(),
                issue_codes,
                status: "fail".to_owned(),
            };
        }
    };

    let fixture_sha256 = sha256_hex(&raw);
    let value = match serde_json::from_slice::<Value>(&raw) {
        Ok(value) => value,
        Err(_) => {
            issue_codes.push("malformed_fixture_json".to_owned());
            Value::Null
        }
    };

    let cases = value
        .get("cases")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if !value.is_null() && !value.get("cases").is_some_and(Value::is_array) {
        issue_codes.push("missing_cases_array".to_owned());
    }

    let duplicate_case_ids = duplicate_case_ids(&cases);
    if !duplicate_case_ids.is_empty() {
        issue_codes.push("duplicate_case_ids".to_owned());
    }
    if cases.len() != spec.expected_case_count {
        issue_codes.push("case_count_mismatch".to_owned());
    }

    let schema_version = string_at(&value, &["schema_version"]).unwrap_or("missing");
    if schema_version != spec.expected_schema_version {
        issue_codes.push("schema_version_mismatch".to_owned());
    }

    let oracle_version = string_at(&value, &["metadata", "jax_version"])
        .or_else(|| string_at(&value, &["jax_version"]))
        .unwrap_or("missing");
    if !oracle_version.starts_with(&spec.expected_oracle_version_prefix) {
        issue_codes.push("stale_oracle_version".to_owned());
    }

    let x64_enabled = bool_at(&value, &["metadata", "x64_enabled"])
        .or_else(|| bool_at(&value, &["x64_enabled"]))
        .or_else(|| bool_at(&value, &["strict_capture"]));
    if let Some(expected) = spec.expected_x64_enabled
        && x64_enabled != Some(expected)
    {
        issue_codes.push("x64_mode_mismatch".to_owned());
    }

    if spec.recapture_command.is_empty() {
        issue_codes.push("missing_recapture_command".to_owned());
    }

    let status = if issue_codes.is_empty() {
        "pass"
    } else {
        "fail"
    }
    .to_owned();

    OracleFixtureRow {
        family_id: spec.family_id.clone(),
        display_name: spec.display_name.clone(),
        fixture_path: spec.fixture_path.clone(),
        schema_version: schema_version.to_owned(),
        expected_schema_version: spec.expected_schema_version.clone(),
        expected_case_count: spec.expected_case_count,
        actual_case_count: cases.len(),
        oracle_version: oracle_version.to_owned(),
        expected_oracle_version_prefix: spec.expected_oracle_version_prefix.clone(),
        x64_enabled,
        expected_x64_enabled: spec.expected_x64_enabled,
        capture_mode: string_at(&value, &["capture_mode"])
            .or_else(|| string_at(&value, &["generated_by"]))
            .unwrap_or("unknown")
            .to_owned(),
        generated_by: string_at(&value, &["generated_by"])
            .unwrap_or("unknown")
            .to_owned(),
        generated_at_unix_ms: value.get("generated_at_unix_ms").and_then(Value::as_u64),
        legacy_anchors: spec.legacy_anchors.clone(),
        recapture_command: spec.recapture_command.clone(),
        fixture_sha256,
        duplicate_case_ids,
        issue_codes,
        status,
    }
}

fn validate_fixture_row(row: &OracleFixtureRow) -> Vec<OracleRecaptureIssue> {
    let mut issues = Vec::new();
    for code in &row.issue_codes {
        issues.push(OracleRecaptureIssue::new(
            code,
            Some(row.family_id.clone()),
            format!("$.rows.{}", row.family_id),
            issue_message(row, code),
        ));
    }
    issues
}

fn issue_message(row: &OracleFixtureRow, code: &str) -> String {
    match code {
        "missing_fixture" => format!("fixture {} is missing", row.fixture_path),
        "malformed_fixture_json" => format!("fixture {} is not valid JSON", row.fixture_path),
        "missing_cases_array" => format!("fixture {} has no cases array", row.fixture_path),
        "duplicate_case_ids" => format!(
            "fixture {} has duplicate case ids: {}",
            row.fixture_path,
            row.duplicate_case_ids.join(", ")
        ),
        "case_count_mismatch" => format!(
            "expected {} cases for {}, found {}",
            row.expected_case_count, row.family_id, row.actual_case_count
        ),
        "schema_version_mismatch" => format!(
            "expected schema {}, found {}",
            row.expected_schema_version, row.schema_version
        ),
        "stale_oracle_version" => format!(
            "expected oracle version prefix {}, found {}",
            row.expected_oracle_version_prefix, row.oracle_version
        ),
        "x64_mode_mismatch" => format!(
            "expected x64 {:?}, found {:?}",
            row.expected_x64_enabled, row.x64_enabled
        ),
        "missing_recapture_command" => {
            "family has no strict automated recapture command".to_owned()
        }
        _ => format!("{} failed validation", row.family_id),
    }
}

fn duplicate_case_ids(cases: &[Value]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut dupes = BTreeSet::new();
    for case in cases {
        if let Some(case_id) = case.get("case_id").and_then(Value::as_str)
            && !seen.insert(case_id.to_owned())
        {
            dupes.insert(case_id.to_owned());
        }
    }
    dupes.into_iter().collect()
}

fn anchor(path: &str, symbol: &str) -> LegacyAnchor {
    LegacyAnchor {
        path: path.to_owned(),
        symbol: symbol.to_owned(),
        required: true,
    }
}

fn string_at<'a>(value: &'a Value, path: &[&str]) -> Option<&'a str> {
    let mut current = value;
    for segment in path {
        current = current.get(*segment)?;
    }
    current.as_str()
}

fn bool_at(value: &Value, path: &[&str]) -> Option<bool> {
    let mut current = value;
    for segment in path {
        current = current.get(*segment)?;
    }
    current.as_bool()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
        .unwrap_or(0)
}
