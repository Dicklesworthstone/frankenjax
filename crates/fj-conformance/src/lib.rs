#![forbid(unsafe_code)]

pub mod durability;

use fj_core::{
    CompatibilityMode, ProgramSpec, TraceTransformLedger, Transform, Value, build_program,
};
use fj_dispatch::{DispatchRequest, dispatch};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct HarnessConfig {
    pub oracle_root: PathBuf,
    pub fixture_root: PathBuf,
    pub strict_mode: bool,
}

impl HarnessConfig {
    #[must_use]
    pub fn default_paths() -> Self {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        Self {
            oracle_root: repo_root.join("legacy_jax_code/jax"),
            fixture_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"),
            strict_mode: true,
        }
    }

    #[must_use]
    pub fn mode(&self) -> CompatibilityMode {
        if self.strict_mode {
            CompatibilityMode::Strict
        } else {
            CompatibilityMode::Hardened
        }
    }
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self::default_paths()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureFamily {
    Jit,
    Grad,
    Vmap,
    Lax,
    Random,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixtureManifestEntry {
    pub family: FixtureFamily,
    pub path_pattern: String,
    pub representative_test: String,
}

#[must_use]
pub fn default_fixture_manifest() -> Vec<FixtureManifestEntry> {
    vec![
        FixtureManifestEntry {
            family: FixtureFamily::Jit,
            path_pattern: "tests/jax_jit_test.py".to_owned(),
            representative_test: "JaxJitTest".to_owned(),
        },
        FixtureManifestEntry {
            family: FixtureFamily::Grad,
            path_pattern: "tests/lax_autodiff_test.py".to_owned(),
            representative_test: "LAX_GRAD_OPS".to_owned(),
        },
        FixtureManifestEntry {
            family: FixtureFamily::Vmap,
            path_pattern: "tests/lax_vmap_test.py".to_owned(),
            representative_test: "LaxVmapTest::_CheckBatching".to_owned(),
        },
        FixtureManifestEntry {
            family: FixtureFamily::Lax,
            path_pattern: "tests/lax_test.py".to_owned(),
            representative_test: "LaxTest::testOp".to_owned(),
        },
        FixtureManifestEntry {
            family: FixtureFamily::Random,
            path_pattern: "tests/random_test.py".to_owned(),
            representative_test: "RandomValuesCase".to_owned(),
        },
    ]
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessReport {
    pub suite: &'static str,
    pub oracle_present: bool,
    pub fixture_count: usize,
    pub manifest_family_count: usize,
    pub strict_mode: bool,
}

#[must_use]
pub fn run_smoke(config: &HarnessConfig) -> HarnessReport {
    let fixture_count = collect_json_fixtures(&config.fixture_root).len();
    let _mode = config.mode();

    HarnessReport {
        suite: "smoke",
        oracle_present: config.oracle_root.exists(),
        fixture_count,
        manifest_family_count: default_fixture_manifest().len(),
        strict_mode: config.strict_mode,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParityCase {
    pub case_id: String,
    pub expected: String,
    pub actual: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParityReport {
    pub total_cases: usize,
    pub matched_cases: usize,
    pub mismatched_cases: usize,
    pub strict_mode: bool,
}

#[must_use]
pub fn evaluate_parity(cases: &[ParityCase], strict_mode: bool) -> ParityReport {
    let matched_cases = cases
        .iter()
        .filter(|case| case.expected == case.actual)
        .count();

    ParityReport {
        total_cases: cases.len(),
        matched_cases,
        mismatched_cases: cases.len().saturating_sub(matched_cases),
        strict_mode,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FixtureNote {
    pub suite: String,
    pub oracle: String,
    pub notes: String,
}

pub fn read_fixture_note(path: &Path) -> Result<FixtureNote, std::io::Error> {
    let raw = fs::read_to_string(path)?;
    let parsed = serde_json::from_str::<FixtureNote>(&raw).map_err(std::io::Error::other)?;
    Ok(parsed)
}

#[must_use]
pub fn collect_json_fixtures(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    collect_json_fixtures_recursive(root, &mut out);
    out.sort();
    out
}

fn collect_json_fixtures_recursive(root: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_json_fixtures_recursive(&path, out);
            continue;
        }

        if path.extension().is_some_and(|ext| ext == "json") {
            out.push(path);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureMode {
    Strict,
    Hardened,
}

impl FixtureMode {
    #[must_use]
    pub fn as_runtime_mode(self) -> CompatibilityMode {
        match self {
            Self::Strict => CompatibilityMode::Strict,
            Self::Hardened => CompatibilityMode::Hardened,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureProgram {
    Add2,
    Square,
    SquarePlusLinear,
    AddOne,
    SinX,
    CosX,
    Dot3,
    ReduceSumVec,
}

impl FixtureProgram {
    #[must_use]
    pub fn as_program_spec(self) -> ProgramSpec {
        match self {
            Self::Add2 => ProgramSpec::Add2,
            Self::Square => ProgramSpec::Square,
            Self::SquarePlusLinear => ProgramSpec::SquarePlusLinear,
            Self::AddOne => ProgramSpec::AddOne,
            Self::SinX => ProgramSpec::SinX,
            Self::CosX => ProgramSpec::CosX,
            Self::Dot3 => ProgramSpec::Dot3,
            Self::ReduceSumVec => ProgramSpec::ReduceSumVec,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureTransform {
    Jit,
    Grad,
    Vmap,
}

impl FixtureTransform {
    #[must_use]
    pub fn as_runtime_transform(self) -> Transform {
        match self {
            Self::Jit => Transform::Jit,
            Self::Grad => Transform::Grad,
            Self::Vmap => Transform::Vmap,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FixtureValue {
    ScalarF64 { value: f64 },
    ScalarI64 { value: i64 },
    VectorF64 { values: Vec<f64> },
    VectorI64 { values: Vec<i64> },
}

impl FixtureValue {
    pub fn to_runtime_value(&self) -> Result<Value, String> {
        match self {
            Self::ScalarF64 { value } => Ok(Value::scalar_f64(*value)),
            Self::ScalarI64 { value } => Ok(Value::scalar_i64(*value)),
            Self::VectorF64 { values } => Value::vector_f64(values)
                .map_err(|err| format!("vector_f64 conversion failed: {err}")),
            Self::VectorI64 { values } => Value::vector_i64(values)
                .map_err(|err| format!("vector_i64 conversion failed: {err}")),
        }
    }

    pub fn approx_matches(&self, actual: &Value, atol: f64, rtol: f64) -> bool {
        match self {
            Self::ScalarF64 { value } => actual
                .as_f64_scalar()
                .is_some_and(|actual_value| approx_equal(*value, actual_value, atol, rtol)),
            Self::ScalarI64 { value } => actual
                .as_scalar_literal()
                .and_then(fj_core::Literal::as_i64)
                .is_some_and(|actual_value| actual_value == *value),
            Self::VectorF64 { values } => actual
                .as_tensor()
                .and_then(|tensor| tensor.to_f64_vec())
                .is_some_and(|actual_values| {
                    values.len() == actual_values.len()
                        && values
                            .iter()
                            .zip(actual_values.iter())
                            .all(|(expected, actual)| approx_equal(*expected, *actual, atol, rtol))
                }),
            Self::VectorI64 { values } => actual.as_tensor().is_some_and(|tensor| {
                if tensor.elements.len() != values.len() {
                    return false;
                }
                tensor
                    .elements
                    .iter()
                    .zip(values.iter())
                    .all(|(actual, expected)| actual.as_i64().is_some_and(|v| v == *expected))
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformFixtureCase {
    pub case_id: String,
    pub family: FixtureFamily,
    pub mode: FixtureMode,
    pub program: FixtureProgram,
    pub transforms: Vec<FixtureTransform>,
    #[serde(default = "default_comparator_kind")]
    pub comparator: ComparatorKind,
    #[serde(default)]
    pub baseline_mismatch: bool,
    #[serde(default)]
    pub flaky: bool,
    #[serde(default)]
    pub simulated_delay_ms: u64,
    pub args: Vec<FixtureValue>,
    pub expected: Vec<FixtureValue>,
    pub atol: f64,
    pub rtol: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformFixtureBundle {
    pub schema_version: String,
    pub generated_by: String,
    pub generated_at_unix_ms: u128,
    pub cases: Vec<TransformFixtureCase>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformCaseReport {
    pub case_id: String,
    pub family: FixtureFamily,
    pub mode: FixtureMode,
    pub comparator: ComparatorKind,
    pub drift_classification: DriftClassification,
    pub matched: bool,
    pub expected_json: String,
    pub actual_json: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformParityReport {
    pub schema_version: String,
    pub total_cases: usize,
    pub matched_cases: usize,
    pub mismatched_cases: usize,
    pub reports: Vec<TransformCaseReport>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchRunnerConfig {
    pub case_timeout: Duration,
}

impl Default for BatchRunnerConfig {
    fn default() -> Self {
        Self {
            case_timeout: default_case_timeout(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleCaptureRequest {
    pub legacy_root: PathBuf,
    pub output_path: PathBuf,
    pub strict: bool,
    pub python_bin: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OracleCaptureResult {
    pub command: String,
    pub stdout: String,
    pub stderr: String,
    pub bundle: TransformFixtureBundle,
}

#[derive(Debug)]
pub enum OracleCaptureError {
    Io(std::io::Error),
    ScriptFailed {
        status: Option<i32>,
        stdout: String,
        stderr: String,
    },
}

impl std::fmt::Display for OracleCaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "oracle capture io error: {err}"),
            Self::ScriptFailed {
                status,
                stdout,
                stderr,
            } => write!(
                f,
                "oracle capture script failed (status={status:?})\nstdout:\n{stdout}\nstderr:\n{stderr}"
            ),
        }
    }
}

impl std::error::Error for OracleCaptureError {}

impl From<std::io::Error> for OracleCaptureError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparatorKind {
    Exact,
    ApproxAtolRtol,
    ShapeOnly,
    TypeOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriftClassification {
    Pass,
    Regression,
    Improvement,
    Flake,
    Timeout,
}

const fn default_comparator_kind() -> ComparatorKind {
    ComparatorKind::ApproxAtolRtol
}

const fn default_case_timeout() -> Duration {
    Duration::from_secs(2)
}

pub fn read_transform_fixture_bundle(
    path: &Path,
) -> Result<TransformFixtureBundle, std::io::Error> {
    let raw = fs::read_to_string(path)?;
    let parsed =
        serde_json::from_str::<TransformFixtureBundle>(&raw).map_err(std::io::Error::other)?;
    Ok(parsed)
}

#[must_use]
pub fn run_transform_fixture_bundle(
    _config: &HarnessConfig,
    bundle: &TransformFixtureBundle,
) -> TransformParityReport {
    let mut reports = Vec::with_capacity(bundle.cases.len());

    for case in &bundle.cases {
        reports.push(run_transform_fixture_case(case));
    }

    let matched_cases = reports.iter().filter(|report| report.matched).count();

    TransformParityReport {
        schema_version: "frankenjax.transform-parity-report.v1".to_owned(),
        total_cases: reports.len(),
        matched_cases,
        mismatched_cases: reports.len().saturating_sub(matched_cases),
        reports,
    }
}

pub fn capture_transform_fixture_bundle_with_oracle(
    request: &OracleCaptureRequest,
) -> Result<OracleCaptureResult, OracleCaptureError> {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("capture_legacy_fixtures.py");
    let python = request
        .python_bin
        .clone()
        .or_else(default_python_for_repo)
        .unwrap_or_else(|| PathBuf::from("python3"));

    let mut cmd = Command::new(&python);
    cmd.arg(&script)
        .arg("--legacy-root")
        .arg(&request.legacy_root)
        .arg("--output")
        .arg(&request.output_path);
    if request.strict {
        cmd.arg("--strict");
    }

    let rendered_cmd = format!(
        "{} {} --legacy-root {} --output {}{}",
        python.display(),
        script.display(),
        request.legacy_root.display(),
        request.output_path.display(),
        if request.strict { " --strict" } else { "" }
    );

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !output.status.success() {
        return Err(OracleCaptureError::ScriptFailed {
            status: output.status.code(),
            stdout,
            stderr,
        });
    }

    let bundle = read_transform_fixture_bundle(&request.output_path)?;
    Ok(OracleCaptureResult {
        command: rendered_cmd,
        stdout,
        stderr,
        bundle,
    })
}

#[must_use]
pub fn run_transform_fixture_bundle_batched(
    _config: &HarnessConfig,
    bundle: &TransformFixtureBundle,
    batch: &BatchRunnerConfig,
) -> TransformParityReport {
    let mut pending = Vec::with_capacity(bundle.cases.len());

    for case in bundle.cases.clone() {
        let expected_json = serde_json::to_string(&case.expected)
            .unwrap_or_else(|err| format!("<expected serialization error: {err}>"));
        let case_for_timeout = case.clone();
        let (tx, rx) = mpsc::channel::<TransformCaseReport>();
        std::thread::spawn(move || {
            let report = run_transform_fixture_case(&case);
            let _ = tx.send(report);
        });
        pending.push((case_for_timeout, expected_json, rx));
    }

    let mut reports = Vec::with_capacity(pending.len());
    for (case, expected_json, rx) in pending {
        match rx.recv_timeout(batch.case_timeout) {
            Ok(report) => reports.push(report),
            Err(_) => reports.push(TransformCaseReport {
                case_id: case.case_id,
                family: case.family,
                mode: case.mode,
                comparator: case.comparator,
                drift_classification: DriftClassification::Timeout,
                matched: false,
                expected_json,
                actual_json: None,
                error: Some(format!(
                    "timeout waiting for case result after {}ms",
                    batch.case_timeout.as_millis()
                )),
            }),
        }
    }

    let matched_cases = reports.iter().filter(|report| report.matched).count();
    TransformParityReport {
        schema_version: "frankenjax.transform-parity-report.v1".to_owned(),
        total_cases: reports.len(),
        matched_cases,
        mismatched_cases: reports.len().saturating_sub(matched_cases),
        reports,
    }
}

pub fn emit_parity_json(report: &TransformParityReport) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(report)
}

#[must_use]
pub fn emit_parity_markdown(report: &TransformParityReport) -> String {
    let mut out = String::new();
    out.push_str("# Transform Parity Report\n\n");
    out.push_str("| Metric | Value |\n");
    out.push_str("|---|---|\n");
    out.push_str(&format!("| Schema | `{}` |\n", report.schema_version));
    out.push_str(&format!("| Total Cases | {} |\n", report.total_cases));
    out.push_str(&format!("| Matched Cases | {} |\n", report.matched_cases));
    out.push_str(&format!(
        "| Mismatched Cases | {} |\n\n",
        report.mismatched_cases
    ));

    out.push_str("| Case ID | Family | Mode | Comparator | Drift | Matched |\n");
    out.push_str("|---|---|---|---|---|---|\n");
    for case in &report.reports {
        out.push_str(&format!(
            "| {} | {:?} | {:?} | {:?} | {:?} | {} |\n",
            case.case_id,
            case.family,
            case.mode,
            case.comparator,
            case.drift_classification,
            case.matched
        ));
    }

    out
}

fn default_python_for_repo() -> Option<PathBuf> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let venv_python = repo_root.join(".venv").join("bin").join("python");
    if venv_python.exists() {
        return Some(venv_python);
    }
    None
}

fn run_transform_fixture_case(case: &TransformFixtureCase) -> TransformCaseReport {
    if case.simulated_delay_ms > 0 {
        std::thread::sleep(Duration::from_millis(case.simulated_delay_ms));
    }

    let runtime_args = case
        .args
        .iter()
        .map(FixtureValue::to_runtime_value)
        .collect::<Result<Vec<_>, _>>();

    let expected_json = serde_json::to_string(&case.expected)
        .unwrap_or_else(|err| format!("<expected serialization error: {err}>"));

    let runtime_args = match runtime_args {
        Ok(args) => args,
        Err(err) => {
            return TransformCaseReport {
                case_id: case.case_id.clone(),
                family: case.family,
                mode: case.mode,
                comparator: case.comparator,
                drift_classification: classify_drift(
                    false,
                    Some("fixture argument conversion failed"),
                    case.baseline_mismatch,
                    case.flaky,
                ),
                matched: false,
                expected_json,
                actual_json: None,
                error: Some(format!("fixture argument conversion failed: {err}")),
            };
        }
    };

    let mut ledger = TraceTransformLedger::new(build_program(case.program.as_program_spec()));
    for (idx, transform) in case.transforms.iter().enumerate() {
        let runtime_transform = transform.as_runtime_transform();
        ledger.push_transform(
            runtime_transform,
            format!(
                "fixture:{}:{}:{}",
                case.case_id,
                runtime_transform.as_str(),
                idx
            ),
        );
    }

    let result = dispatch(DispatchRequest {
        mode: case.mode.as_runtime_mode(),
        ledger,
        args: runtime_args,
        backend: "cpu".to_owned(),
        compile_options: std::collections::BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: Vec::new(),
    });

    match result {
        Ok(response) => {
            let matched = compare_outputs(case, &response.outputs);

            let actual_json = serde_json::to_string(&response.outputs)
                .unwrap_or_else(|err| format!("<actual serialization error: {err}>"));

            TransformCaseReport {
                case_id: case.case_id.clone(),
                family: case.family,
                mode: case.mode,
                comparator: case.comparator,
                drift_classification: classify_drift(
                    matched,
                    if matched {
                        None
                    } else {
                        Some("output mismatch")
                    },
                    case.baseline_mismatch,
                    case.flaky,
                ),
                matched,
                expected_json,
                actual_json: Some(actual_json),
                error: if matched {
                    None
                } else {
                    Some("output mismatch".to_owned())
                },
            }
        }
        Err(err) => TransformCaseReport {
            case_id: case.case_id.clone(),
            family: case.family,
            mode: case.mode,
            comparator: case.comparator,
            drift_classification: classify_drift(
                false,
                Some(&err.to_string()),
                case.baseline_mismatch,
                case.flaky,
            ),
            matched: false,
            expected_json,
            actual_json: None,
            error: Some(err.to_string()),
        },
    }
}

fn approx_equal(expected: f64, actual: f64, atol: f64, rtol: f64) -> bool {
    if expected.is_nan() && actual.is_nan() {
        return true;
    }
    if expected.is_infinite() && actual.is_infinite() && expected.signum() == actual.signum() {
        return true;
    }
    let tolerance = atol + rtol * expected.abs();
    (expected - actual).abs() <= tolerance
}

fn compare_outputs(case: &TransformFixtureCase, actual: &[Value]) -> bool {
    if case.expected.len() != actual.len() {
        return false;
    }

    match case.comparator {
        ComparatorKind::Exact => {
            case.expected
                .iter()
                .zip(actual.iter())
                .all(|(expected, actual_value)| {
                    expected
                        .to_runtime_value()
                        .is_ok_and(|expected_value| expected_value == *actual_value)
                })
        }
        ComparatorKind::ApproxAtolRtol => {
            case.expected
                .iter()
                .zip(actual.iter())
                .all(|(expected, actual_value)| {
                    expected.approx_matches(actual_value, case.atol, case.rtol)
                })
        }
        ComparatorKind::ShapeOnly => {
            case.expected
                .iter()
                .zip(actual.iter())
                .all(|(expected, actual_value)| {
                    value_shape_fingerprint(expected) == value_shape_runtime(actual_value)
                })
        }
        ComparatorKind::TypeOnly => {
            case.expected
                .iter()
                .zip(actual.iter())
                .all(|(expected, actual_value)| {
                    value_type_fingerprint(expected) == value_type_runtime(actual_value)
                })
        }
    }
}

fn value_shape_fingerprint(expected: &FixtureValue) -> String {
    match expected {
        FixtureValue::ScalarF64 { .. } | FixtureValue::ScalarI64 { .. } => "scalar".to_owned(),
        FixtureValue::VectorF64 { values } => format!("vector:{}", values.len()),
        FixtureValue::VectorI64 { values } => format!("vector:{}", values.len()),
    }
}

fn value_shape_runtime(actual: &Value) -> String {
    if let Some(tensor) = actual.as_tensor() {
        if tensor.shape.dims.is_empty() {
            "scalar".to_owned()
        } else {
            format!("vector:{}", tensor.elements.len())
        }
    } else {
        "scalar".to_owned()
    }
}

fn value_type_fingerprint(expected: &FixtureValue) -> &'static str {
    match expected {
        FixtureValue::ScalarF64 { .. } | FixtureValue::VectorF64 { .. } => "f64",
        FixtureValue::ScalarI64 { .. } | FixtureValue::VectorI64 { .. } => "i64",
    }
}

fn value_type_runtime(actual: &Value) -> &'static str {
    if let Some(scalar) = actual.as_scalar_literal() {
        if scalar.as_i64().is_some() {
            return "i64";
        }
        if scalar.as_f64().is_some() {
            return "f64";
        }
    }

    if let Some(tensor) = actual.as_tensor() {
        return match tensor.dtype {
            fj_core::DType::I64 | fj_core::DType::I32 => "i64",
            fj_core::DType::F64 | fj_core::DType::F32 => "f64",
            fj_core::DType::Bool => "bool",
        };
    }

    "unknown"
}

fn classify_drift(
    matched: bool,
    error: Option<&str>,
    baseline_mismatch: bool,
    flaky: bool,
) -> DriftClassification {
    if matched && baseline_mismatch {
        return DriftClassification::Improvement;
    }

    if matched {
        return DriftClassification::Pass;
    }

    if error.is_some_and(|detail| detail.to_ascii_lowercase().contains("timeout")) {
        return DriftClassification::Timeout;
    }

    if flaky {
        return DriftClassification::Flake;
    }

    DriftClassification::Regression
}

#[cfg(test)]
mod tests {
    use super::{
        BatchRunnerConfig, ComparatorKind, DriftClassification, FixtureFamily, FixtureMode,
        FixtureProgram, FixtureTransform, FixtureValue, HarnessConfig, OracleCaptureRequest,
        ParityCase, TransformFixtureBundle, TransformFixtureCase,
        capture_transform_fixture_bundle_with_oracle, classify_drift, collect_json_fixtures,
        default_fixture_manifest, emit_parity_json, emit_parity_markdown, evaluate_parity,
        read_fixture_note, run_smoke, run_transform_fixture_bundle,
        run_transform_fixture_bundle_batched,
    };
    use tempfile::tempdir;

    #[test]
    fn smoke_harness_finds_oracle_and_fixtures() {
        let cfg = HarnessConfig::default_paths();
        let report = run_smoke(&cfg);
        assert!(report.oracle_present, "oracle repo should be present");
        assert!(report.fixture_count >= 1, "expected at least one fixture");
        assert_eq!(report.manifest_family_count, 5);
        assert!(report.strict_mode);
    }

    #[test]
    fn manifest_covers_core_families() {
        let manifest = default_fixture_manifest();
        assert_eq!(manifest.len(), 5);
    }

    #[test]
    fn parity_report_counts_matches() {
        let report = evaluate_parity(
            &[
                ParityCase {
                    case_id: "a".to_owned(),
                    expected: "1".to_owned(),
                    actual: "1".to_owned(),
                },
                ParityCase {
                    case_id: "b".to_owned(),
                    expected: "2".to_owned(),
                    actual: "3".to_owned(),
                },
            ],
            true,
        );

        assert_eq!(report.total_cases, 2);
        assert_eq!(report.matched_cases, 1);
        assert_eq!(report.mismatched_cases, 1);
        assert!(report.strict_mode);
    }

    #[test]
    fn fixture_note_round_trip() {
        let cfg = HarnessConfig::default_paths();
        let note_path = cfg.fixture_root.join("smoke_case.json");
        let note = read_fixture_note(&note_path).expect("fixture note should parse");
        assert_eq!(note.suite, "smoke");
    }

    #[test]
    fn fixture_collection_finds_json() {
        let cfg = HarnessConfig::default_paths();
        let fixtures = collect_json_fixtures(&cfg.fixture_root);
        assert!(!fixtures.is_empty());
    }

    #[test]
    fn transform_bundle_runs_and_matches() {
        let cfg = HarnessConfig::default_paths();
        let bundle = TransformFixtureBundle {
            schema_version: "frankenjax.transform-fixtures.v1".to_owned(),
            generated_by: "unit-test".to_owned(),
            generated_at_unix_ms: 0,
            cases: vec![TransformFixtureCase {
                case_id: "jit_add_scalar".to_owned(),
                family: FixtureFamily::Jit,
                mode: FixtureMode::Strict,
                program: FixtureProgram::Add2,
                transforms: vec![FixtureTransform::Jit],
                comparator: ComparatorKind::ApproxAtolRtol,
                baseline_mismatch: false,
                flaky: false,
                simulated_delay_ms: 0,
                args: vec![
                    FixtureValue::ScalarI64 { value: 2 },
                    FixtureValue::ScalarI64 { value: 5 },
                ],
                expected: vec![FixtureValue::ScalarI64 { value: 7 }],
                atol: 1e-6,
                rtol: 1e-6,
            }],
        };

        let report = run_transform_fixture_bundle(&cfg, &bundle);
        assert_eq!(report.total_cases, 1);
        assert_eq!(report.matched_cases, 1);
        assert_eq!(report.mismatched_cases, 0);
    }

    #[test]
    fn test_conformance_test_log_schema_contract() {
        let fixture_id =
            fj_test_utils::fixture_id_from_json(&("conformance", "smoke")).expect("digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_conformance_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }

    #[test]
    fn test_drift_classification_timeout_and_regression() {
        assert_eq!(
            classify_drift(false, Some("timeout waiting for oracle"), false, false),
            DriftClassification::Timeout
        );
        assert_eq!(
            classify_drift(false, Some("shape mismatch"), false, false),
            DriftClassification::Regression
        );
        assert_eq!(
            classify_drift(true, None, true, false),
            DriftClassification::Improvement
        );
        assert_eq!(
            classify_drift(false, Some("intermittent mismatch"), false, true),
            DriftClassification::Flake
        );
        assert_eq!(
            classify_drift(true, None, false, false),
            DriftClassification::Pass
        );
    }

    #[test]
    fn test_parity_markdown_emitter_includes_summary() {
        let cfg = HarnessConfig::default_paths();
        let bundle = TransformFixtureBundle {
            schema_version: "frankenjax.transform-fixtures.v1".to_owned(),
            generated_by: "unit-test".to_owned(),
            generated_at_unix_ms: 0,
            cases: vec![TransformFixtureCase {
                case_id: "jit_add_scalar_markdown".to_owned(),
                family: FixtureFamily::Jit,
                mode: FixtureMode::Strict,
                program: FixtureProgram::Add2,
                transforms: vec![FixtureTransform::Jit],
                comparator: ComparatorKind::ApproxAtolRtol,
                baseline_mismatch: false,
                flaky: false,
                simulated_delay_ms: 0,
                args: vec![
                    FixtureValue::ScalarI64 { value: 1 },
                    FixtureValue::ScalarI64 { value: 2 },
                ],
                expected: vec![FixtureValue::ScalarI64 { value: 3 }],
                atol: 1e-6,
                rtol: 1e-6,
            }],
        };
        let report = run_transform_fixture_bundle(&cfg, &bundle);
        let markdown = emit_parity_markdown(&report);
        assert!(markdown.contains("Transform Parity Report"));
        assert!(markdown.contains("jit_add_scalar_markdown"));
    }

    #[test]
    fn test_parity_json_emitter_round_trip() {
        let cfg = HarnessConfig::default_paths();
        let bundle = TransformFixtureBundle {
            schema_version: "frankenjax.transform-fixtures.v1".to_owned(),
            generated_by: "unit-test".to_owned(),
            generated_at_unix_ms: 0,
            cases: vec![TransformFixtureCase {
                case_id: "jit_add_scalar_json".to_owned(),
                family: FixtureFamily::Jit,
                mode: FixtureMode::Strict,
                program: FixtureProgram::Add2,
                transforms: vec![FixtureTransform::Jit],
                comparator: ComparatorKind::ApproxAtolRtol,
                baseline_mismatch: false,
                flaky: false,
                simulated_delay_ms: 0,
                args: vec![
                    FixtureValue::ScalarI64 { value: 1 },
                    FixtureValue::ScalarI64 { value: 1 },
                ],
                expected: vec![FixtureValue::ScalarI64 { value: 2 }],
                atol: 1e-6,
                rtol: 1e-6,
            }],
        };
        let report = run_transform_fixture_bundle(&cfg, &bundle);
        let json = emit_parity_json(&report).expect("parity json should serialize");
        let decoded: super::TransformParityReport =
            serde_json::from_str(&json).expect("parity json should parse");
        assert_eq!(decoded.total_cases, 1);
    }

    #[test]
    fn test_comparator_taxonomy_exact_shape_type_and_approx() {
        let cfg = HarnessConfig::default_paths();
        let bundle = TransformFixtureBundle {
            schema_version: "frankenjax.transform-fixtures.v1".to_owned(),
            generated_by: "unit-test".to_owned(),
            generated_at_unix_ms: 0,
            cases: vec![
                TransformFixtureCase {
                    case_id: "cmp_exact".to_owned(),
                    family: FixtureFamily::Jit,
                    mode: FixtureMode::Strict,
                    program: FixtureProgram::Add2,
                    transforms: vec![FixtureTransform::Jit],
                    comparator: ComparatorKind::Exact,
                    baseline_mismatch: false,
                    flaky: false,
                    simulated_delay_ms: 0,
                    args: vec![
                        FixtureValue::ScalarI64 { value: 2 },
                        FixtureValue::ScalarI64 { value: 2 },
                    ],
                    expected: vec![FixtureValue::ScalarI64 { value: 4 }],
                    atol: 0.0,
                    rtol: 0.0,
                },
                TransformFixtureCase {
                    case_id: "cmp_shape_only".to_owned(),
                    family: FixtureFamily::Vmap,
                    mode: FixtureMode::Strict,
                    program: FixtureProgram::AddOne,
                    transforms: vec![FixtureTransform::Vmap],
                    comparator: ComparatorKind::ShapeOnly,
                    baseline_mismatch: false,
                    flaky: false,
                    simulated_delay_ms: 0,
                    args: vec![FixtureValue::VectorI64 {
                        values: vec![10, 20, 30],
                    }],
                    expected: vec![FixtureValue::VectorI64 {
                        values: vec![999, 999, 999],
                    }],
                    atol: 1e-6,
                    rtol: 1e-6,
                },
                TransformFixtureCase {
                    case_id: "cmp_type_only".to_owned(),
                    family: FixtureFamily::Jit,
                    mode: FixtureMode::Strict,
                    program: FixtureProgram::Square,
                    transforms: vec![FixtureTransform::Jit],
                    comparator: ComparatorKind::TypeOnly,
                    baseline_mismatch: false,
                    flaky: false,
                    simulated_delay_ms: 0,
                    args: vec![FixtureValue::ScalarF64 { value: 2.0 }],
                    expected: vec![FixtureValue::ScalarF64 { value: -1234.0 }],
                    atol: 1e-6,
                    rtol: 1e-6,
                },
                TransformFixtureCase {
                    case_id: "cmp_approx".to_owned(),
                    family: FixtureFamily::Grad,
                    mode: FixtureMode::Strict,
                    program: FixtureProgram::Square,
                    transforms: vec![FixtureTransform::Grad],
                    comparator: ComparatorKind::ApproxAtolRtol,
                    baseline_mismatch: false,
                    flaky: false,
                    simulated_delay_ms: 0,
                    args: vec![FixtureValue::ScalarF64 { value: 3.0 }],
                    expected: vec![FixtureValue::ScalarF64 { value: 6.0 }],
                    atol: 1e-3,
                    rtol: 1e-3,
                },
            ],
        };
        let report = run_transform_fixture_bundle(&cfg, &bundle);
        assert_eq!(report.mismatched_cases, 0);
        let comparators = report
            .reports
            .iter()
            .map(|case| case.comparator)
            .collect::<Vec<_>>();
        assert!(comparators.contains(&ComparatorKind::Exact));
        assert!(comparators.contains(&ComparatorKind::ShapeOnly));
        assert!(comparators.contains(&ComparatorKind::TypeOnly));
        assert!(comparators.contains(&ComparatorKind::ApproxAtolRtol));
    }

    #[test]
    fn test_batch_runner_marks_timeout() {
        let cfg = HarnessConfig::default_paths();
        let bundle = TransformFixtureBundle {
            schema_version: "frankenjax.transform-fixtures.v1".to_owned(),
            generated_by: "unit-test".to_owned(),
            generated_at_unix_ms: 0,
            cases: vec![TransformFixtureCase {
                case_id: "batch_timeout".to_owned(),
                family: FixtureFamily::Jit,
                mode: FixtureMode::Strict,
                program: FixtureProgram::Add2,
                transforms: vec![FixtureTransform::Jit],
                comparator: ComparatorKind::ApproxAtolRtol,
                baseline_mismatch: false,
                flaky: false,
                simulated_delay_ms: 50,
                args: vec![
                    FixtureValue::ScalarI64 { value: 1 },
                    FixtureValue::ScalarI64 { value: 2 },
                ],
                expected: vec![FixtureValue::ScalarI64 { value: 3 }],
                atol: 1e-6,
                rtol: 1e-6,
            }],
        };
        let report = run_transform_fixture_bundle_batched(
            &cfg,
            &bundle,
            &BatchRunnerConfig {
                case_timeout: std::time::Duration::from_millis(5),
            },
        );
        assert_eq!(report.total_cases, 1);
        assert_eq!(
            report.reports[0].drift_classification,
            DriftClassification::Timeout
        );
    }

    #[test]
    fn test_oracle_capture_invocation_produces_large_bundle() {
        let cfg = HarnessConfig::default_paths();
        let tmp = tempdir().expect("tempdir should build");
        let output = tmp.path().join("oracle-capture.json");
        let result = capture_transform_fixture_bundle_with_oracle(&OracleCaptureRequest {
            legacy_root: cfg.oracle_root.clone(),
            output_path: output,
            strict: false,
            python_bin: None,
        })
        .expect("oracle capture script should run");
        assert!(
            result.bundle.cases.len() >= 50,
            "expected expanded fixture corpus, got {}",
            result.bundle.cases.len()
        );
    }
}
