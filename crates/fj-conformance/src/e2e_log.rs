use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

pub const E2E_FORENSIC_LOG_SCHEMA_VERSION: &str = "frankenjax.e2e-forensic-log.v1";
pub const E2E_LOG_VALIDATION_REPORT_SCHEMA_VERSION: &str =
    "frankenjax.e2e-log-validation-report.v1";

const REQUIRED_TOP_LEVEL_FIELDS: &[&str] = &[
    "schema_version",
    "bead_id",
    "scenario_id",
    "test_id",
    "command",
    "working_dir",
    "environment",
    "feature_flags",
    "fixture_ids",
    "oracle_ids",
    "transform_stack",
    "mode",
    "inputs",
    "expected",
    "actual",
    "tolerance",
    "error",
    "timings",
    "allocations",
    "artifacts",
    "replay_command",
    "status",
    "failure_summary",
    "redactions",
    "metadata",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum E2ECompatibilityMode {
    Strict,
    Hardened,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum E2ELogStatus {
    Pass,
    Fail,
    Skip,
    Timeout,
    Crash,
    Error,
}

impl E2ELogStatus {
    #[must_use]
    pub fn requires_failure_summary(self) -> bool {
        matches!(self, Self::Fail | Self::Timeout | Self::Crash | Self::Error)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct E2EEnvironmentFingerprint {
    pub os: String,
    pub arch: String,
    pub rust_version: String,
    pub cargo_version: String,
    pub cargo_target_dir: String,
    pub env_vars: BTreeMap<String, String>,
    pub timestamp_unix_ms: u64,
}

impl E2EEnvironmentFingerprint {
    #[must_use]
    pub fn capture(env_allowlist: &[&str]) -> Self {
        let env_vars = env_allowlist
            .iter()
            .filter_map(|key| {
                std::env::var(key)
                    .ok()
                    .map(|value| ((*key).to_owned(), value))
            })
            .collect();

        Self {
            os: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            rust_version: command_version("rustc"),
            cargo_version: command_version("cargo"),
            cargo_target_dir: std::env::var("CARGO_TARGET_DIR")
                .unwrap_or_else(|_| "<default>".to_owned()),
            env_vars,
            timestamp_unix_ms: now_unix_ms(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct E2ETolerancePolicy {
    pub policy_id: String,
    pub atol: Option<f64>,
    pub rtol: Option<f64>,
    pub ulp: Option<u64>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct E2EErrorClass {
    pub expected: Option<String>,
    pub actual: Option<String>,
    pub taxonomy_class: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct E2ETimingBreakdown {
    pub setup_ms: u64,
    pub trace_ms: u64,
    pub dispatch_ms: u64,
    pub eval_ms: u64,
    pub verify_ms: u64,
    pub total_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct E2EAllocationCounters {
    pub allocation_count: Option<u64>,
    pub allocated_bytes: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    pub measurement_backend: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct E2EArtifactRef {
    pub kind: String,
    pub path: String,
    pub sha256: String,
    pub required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct E2ERedaction {
    pub path: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct E2EForensicLogV1 {
    pub schema_version: String,
    pub bead_id: String,
    pub scenario_id: String,
    pub test_id: String,
    pub packet_id: Option<String>,
    pub command: Vec<String>,
    pub working_dir: String,
    pub environment: E2EEnvironmentFingerprint,
    pub feature_flags: Vec<String>,
    pub fixture_ids: Vec<String>,
    pub oracle_ids: Vec<String>,
    pub transform_stack: Vec<String>,
    pub mode: E2ECompatibilityMode,
    pub inputs: Value,
    pub expected: Value,
    pub actual: Value,
    pub tolerance: E2ETolerancePolicy,
    pub error: E2EErrorClass,
    pub timings: E2ETimingBreakdown,
    pub allocations: E2EAllocationCounters,
    pub artifacts: Vec<E2EArtifactRef>,
    pub replay_command: String,
    pub status: E2ELogStatus,
    pub failure_summary: Option<String>,
    pub redactions: Vec<E2ERedaction>,
    pub metadata: BTreeMap<String, Value>,
}

impl E2EForensicLogV1 {
    #[must_use]
    pub fn new(
        bead_id: impl Into<String>,
        scenario_id: impl Into<String>,
        test_id: impl Into<String>,
        command: Vec<String>,
        working_dir: impl Into<String>,
        mode: E2ECompatibilityMode,
        status: E2ELogStatus,
    ) -> Self {
        Self {
            schema_version: E2E_FORENSIC_LOG_SCHEMA_VERSION.to_owned(),
            bead_id: bead_id.into(),
            scenario_id: scenario_id.into(),
            test_id: test_id.into(),
            packet_id: None,
            command,
            working_dir: working_dir.into(),
            environment: E2EEnvironmentFingerprint::capture(&[
                "CARGO_TARGET_DIR",
                "RUSTUP_TOOLCHAIN",
            ]),
            feature_flags: Vec::new(),
            fixture_ids: Vec::new(),
            oracle_ids: Vec::new(),
            transform_stack: Vec::new(),
            mode,
            inputs: Value::Null,
            expected: Value::Null,
            actual: Value::Null,
            tolerance: E2ETolerancePolicy {
                policy_id: "exact_or_declared".to_owned(),
                atol: None,
                rtol: None,
                ulp: None,
                notes: None,
            },
            error: E2EErrorClass {
                expected: None,
                actual: None,
                taxonomy_class: "none".to_owned(),
            },
            timings: E2ETimingBreakdown {
                setup_ms: 0,
                trace_ms: 0,
                dispatch_ms: 0,
                eval_ms: 0,
                verify_ms: 0,
                total_ms: 0,
            },
            allocations: E2EAllocationCounters {
                allocation_count: None,
                allocated_bytes: None,
                peak_rss_bytes: None,
                measurement_backend: "not_measured".to_owned(),
            },
            artifacts: Vec::new(),
            replay_command: String::new(),
            status,
            failure_summary: None,
            redactions: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct E2ELogValidationIssue {
    pub code: String,
    pub path: String,
    pub message: String,
}

impl E2ELogValidationIssue {
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct E2ELogFileValidation {
    pub path: String,
    pub status: String,
    pub issues: Vec<E2ELogValidationIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct E2ELogValidationReport {
    pub schema_version: String,
    pub status: String,
    pub checked: usize,
    pub passed: usize,
    pub failed: usize,
    pub logs: Vec<E2ELogFileValidation>,
}

#[must_use]
pub fn classify_process_status(
    exit_code: Option<i32>,
    timed_out: bool,
    crashed: bool,
) -> E2ELogStatus {
    if timed_out {
        return E2ELogStatus::Timeout;
    }
    if crashed {
        return E2ELogStatus::Crash;
    }
    match exit_code {
        Some(0) => E2ELogStatus::Pass,
        Some(_) => E2ELogStatus::Fail,
        None => E2ELogStatus::Error,
    }
}

pub fn validate_e2e_log_str(
    raw: &str,
    artifact_root: &Path,
) -> Result<E2EForensicLogV1, Vec<E2ELogValidationIssue>> {
    let value = match serde_json::from_str::<Value>(raw) {
        Ok(value) => value,
        Err(err) => {
            return Err(vec![E2ELogValidationIssue::new(
                "malformed_json",
                "$",
                format!("log is not valid JSON: {err}"),
            )]);
        }
    };
    validate_e2e_log_value(&value, artifact_root)
}

pub fn validate_e2e_log_value(
    value: &Value,
    artifact_root: &Path,
) -> Result<E2EForensicLogV1, Vec<E2ELogValidationIssue>> {
    let mut issues = Vec::new();
    let Some(object) = value.as_object() else {
        return Err(vec![E2ELogValidationIssue::new(
            "malformed_log",
            "$",
            "top-level log must be a JSON object",
        )]);
    };

    for field in REQUIRED_TOP_LEVEL_FIELDS {
        if !object.contains_key(*field) {
            issues.push(E2ELogValidationIssue::new(
                "missing_required_field",
                format!("$.{field}"),
                format!("required field `{field}` is absent"),
            ));
        }
    }

    let parsed = match serde_json::from_value::<E2EForensicLogV1>(value.clone()) {
        Ok(parsed) => parsed,
        Err(err) => {
            issues.push(E2ELogValidationIssue::new(
                "malformed_log",
                "$",
                format!("log does not match E2EForensicLogV1: {err}"),
            ));
            return Err(issues);
        }
    };

    validate_required_string(
        &mut issues,
        "$.schema_version",
        &parsed.schema_version,
        E2E_FORENSIC_LOG_SCHEMA_VERSION,
    );
    validate_non_empty(&mut issues, "$.bead_id", &parsed.bead_id);
    validate_non_empty(&mut issues, "$.scenario_id", &parsed.scenario_id);
    validate_non_empty(&mut issues, "$.test_id", &parsed.test_id);
    validate_non_empty(&mut issues, "$.working_dir", &parsed.working_dir);

    if parsed.command.is_empty() || parsed.command.iter().any(|arg| arg.trim().is_empty()) {
        issues.push(E2ELogValidationIssue::new(
            "missing_command",
            "$.command",
            "command must list the executable and non-empty arguments",
        ));
    }

    if parsed.replay_command.trim().is_empty() {
        issues.push(E2ELogValidationIssue::new(
            "missing_replay_command",
            "$.replay_command",
            "replay_command must be present so users can reproduce the scenario",
        ));
    }
    if is_redacted_text(&parsed.replay_command) {
        issues.push(E2ELogValidationIssue::new(
            "redacted_replay_command",
            "$.replay_command",
            "replay_command must not be redacted",
        ));
    }

    if parsed.status.requires_failure_summary()
        && parsed
            .failure_summary
            .as_deref()
            .is_none_or(|summary| summary.trim().is_empty())
    {
        issues.push(E2ELogValidationIssue::new(
            "missing_failure_summary",
            "$.failure_summary",
            "failing, timeout, crash, and error logs need a human-readable summary",
        ));
    }

    if parsed.tolerance.policy_id.trim().is_empty() {
        issues.push(E2ELogValidationIssue::new(
            "missing_tolerance_policy",
            "$.tolerance.policy_id",
            "tolerance policy id must be explicit",
        ));
    }

    if parsed.error.taxonomy_class.trim().is_empty() {
        issues.push(E2ELogValidationIssue::new(
            "missing_error_taxonomy",
            "$.error.taxonomy_class",
            "error taxonomy class must be explicit even for passing logs",
        ));
    }

    validate_redaction_policy(value, &mut issues);
    validate_artifacts(&parsed, artifact_root, &mut issues);

    if issues.is_empty() {
        Ok(parsed)
    } else {
        Err(issues)
    }
}

pub fn validate_e2e_log_path(
    path: &Path,
    artifact_root: &Path,
) -> Result<E2EForensicLogV1, Vec<E2ELogValidationIssue>> {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) => {
            return Err(vec![E2ELogValidationIssue::new(
                "read_error",
                path.display().to_string(),
                format!("failed to read log: {err}"),
            )]);
        }
    };
    validate_e2e_log_str(&raw, artifact_root)
}

#[must_use]
pub fn validation_report_for_paths(
    paths: &[PathBuf],
    artifact_root: &Path,
) -> E2ELogValidationReport {
    if paths.is_empty() {
        return E2ELogValidationReport {
            schema_version: E2E_LOG_VALIDATION_REPORT_SCHEMA_VERSION.to_owned(),
            status: "fail".to_owned(),
            checked: 0,
            passed: 0,
            failed: 1,
            logs: vec![E2ELogFileValidation {
                path: "<no e2e logs discovered>".to_owned(),
                status: "fail".to_owned(),
                issues: vec![E2ELogValidationIssue::new(
                    "no_logs_found",
                    "$",
                    "at least one .e2e.json log is required for dashboard ingestion",
                )],
            }],
        };
    }

    let mut logs = Vec::with_capacity(paths.len());
    let mut passed = 0usize;
    let mut failed = 0usize;

    for path in paths {
        match validate_e2e_log_path(path, artifact_root) {
            Ok(_) => {
                passed += 1;
                logs.push(E2ELogFileValidation {
                    path: path.display().to_string(),
                    status: "pass".to_owned(),
                    issues: Vec::new(),
                });
            }
            Err(issues) => {
                failed += 1;
                logs.push(E2ELogFileValidation {
                    path: path.display().to_string(),
                    status: "fail".to_owned(),
                    issues,
                });
            }
        }
    }

    E2ELogValidationReport {
        schema_version: E2E_LOG_VALIDATION_REPORT_SCHEMA_VERSION.to_owned(),
        status: if failed == 0 { "pass" } else { "fail" }.to_owned(),
        checked: paths.len(),
        passed,
        failed,
        logs,
    }
}

pub fn write_e2e_log(path: &Path, log: &E2EForensicLogV1) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let raw = serde_json::to_string_pretty(log).map_err(std::io::Error::other)?;
    fs::write(path, format!("{raw}\n"))
}

pub fn artifact_sha256_hex(path: &Path) -> Result<String, std::io::Error> {
    let bytes = fs::read(path)?;
    Ok(sha256_hex(&bytes))
}

#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn validate_required_string(
    issues: &mut Vec<E2ELogValidationIssue>,
    path: &str,
    actual: &str,
    expected: &str,
) {
    if actual != expected {
        issues.push(E2ELogValidationIssue::new(
            "unsupported_schema_version",
            path,
            format!("expected `{expected}`, got `{actual}`"),
        ));
    }
}

fn validate_non_empty(issues: &mut Vec<E2ELogValidationIssue>, path: &str, actual: &str) {
    if actual.trim().is_empty() {
        issues.push(E2ELogValidationIssue::new(
            "empty_required_field",
            path,
            "field must not be empty",
        ));
    }
}

fn validate_artifacts(
    log: &E2EForensicLogV1,
    artifact_root: &Path,
    issues: &mut Vec<E2ELogValidationIssue>,
) {
    for (idx, artifact) in log.artifacts.iter().enumerate() {
        if artifact.kind.trim().is_empty() {
            issues.push(E2ELogValidationIssue::new(
                "empty_artifact_kind",
                format!("$.artifacts[{idx}].kind"),
                "artifact kind must not be empty",
            ));
        }
        if artifact.path.trim().is_empty() {
            issues.push(E2ELogValidationIssue::new(
                "empty_artifact_path",
                format!("$.artifacts[{idx}].path"),
                "artifact path must not be empty",
            ));
            continue;
        }
        if !is_sha256_hex(&artifact.sha256) {
            issues.push(E2ELogValidationIssue::new(
                "invalid_artifact_hash",
                format!("$.artifacts[{idx}].sha256"),
                "artifact hash must be a lowercase SHA-256 hex digest",
            ));
            continue;
        }

        let path = resolve_artifact_path(artifact_root, &artifact.path);
        match artifact_sha256_hex(&path) {
            Ok(actual) if actual == artifact.sha256 => {}
            Ok(actual) => issues.push(E2ELogValidationIssue::new(
                "stale_artifact_hash",
                format!("$.artifacts[{idx}].sha256"),
                format!(
                    "artifact {} hash mismatch: expected {}, actual {}",
                    path.display(),
                    artifact.sha256,
                    actual
                ),
            )),
            Err(err) if artifact.required => issues.push(E2ELogValidationIssue::new(
                "missing_required_artifact",
                format!("$.artifacts[{idx}].path"),
                format!(
                    "required artifact {} is not readable: {err}",
                    path.display()
                ),
            )),
            Err(_) => {}
        }
    }
}

fn validate_redaction_policy(value: &Value, issues: &mut Vec<E2ELogValidationIssue>) {
    scan_redaction_value(value, "$", issues);
}

fn scan_redaction_value(value: &Value, path: &str, issues: &mut Vec<E2ELogValidationIssue>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let child_path = format!("{path}.{key}");
                if looks_sensitive_key(key) && !value_is_redacted(child) {
                    issues.push(E2ELogValidationIssue::new(
                        "unredacted_sensitive_value",
                        child_path.clone(),
                        format!("sensitive key `{key}` must contain a redacted marker"),
                    ));
                }
                scan_redaction_value(child, &child_path, issues);
            }
        }
        Value::Array(items) => {
            for (idx, child) in items.iter().enumerate() {
                scan_redaction_value(child, &format!("{path}[{idx}]"), issues);
            }
        }
        _ => {}
    }
}

fn value_is_redacted(value: &Value) -> bool {
    value.as_str().is_some_and(is_redacted_text)
}

fn is_redacted_text(text: &str) -> bool {
    let normalized = text.trim().to_ascii_lowercase();
    matches!(
        normalized.as_str(),
        "[redacted]" | "<redacted>" | "redacted" | "***redacted***"
    )
}

fn looks_sensitive_key(key: &str) -> bool {
    let normalized = key.to_ascii_lowercase();
    normalized.contains("secret")
        || normalized.contains("token")
        || normalized.contains("password")
        || normalized.contains("passwd")
        || normalized.contains("private_key")
        || normalized.contains("credential")
}

fn is_sha256_hex(raw: &str) -> bool {
    raw.len() == 64
        && raw
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn resolve_artifact_path(root: &Path, raw: &str) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else {
        root.join(path)
    }
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
        .unwrap_or(0)
}

fn command_version(command: &str) -> String {
    let output = Command::new(command).arg("--version").output();
    match output {
        Ok(result) if result.status.success() => {
            String::from_utf8_lossy(&result.stdout).trim().to_owned()
        }
        _ => format!("{command} <unknown>"),
    }
}
