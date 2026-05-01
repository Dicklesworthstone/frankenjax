#![forbid(unsafe_code)]

use crate::durability::{DurabilityError, SidecarConfig, encode_artifact_to_sidecar};
use fj_cache::{CacheKeyError, CacheKeyInput, build_cache_key};
use fj_core::{
    Atom, CompatibilityMode, DType, Equation, Jaxpr, JaxprValidationError, Literal, Primitive,
    ProgramSpec, Shape, TensorValue, TraceTransformLedger, Transform, TransformCompositionError,
    Value, VarId, build_program, verify_transform_composition,
};
use fj_dispatch::{DispatchError, DispatchRequest, TransformExecutionError, dispatch};
use fj_interpreters::{InterpreterError, eval_jaxpr};
use fj_lax::{EvalError, eval_primitive};
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};
use smallvec::smallvec;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const ERROR_TAXONOMY_REPORT_SCHEMA_VERSION: &str = "frankenjax.error-taxonomy-matrix.v1";
pub const ERROR_TAXONOMY_BEAD_ID: &str = "frankenjax-cstq.8";

const REQUIRED_CASES: &[&str] = &[
    "ir_validation_unknown_outvar",
    "transform_proof_duplicate_evidence",
    "transform_proof_missing_evidence",
    "primitive_arity_add",
    "primitive_shape_add_broadcast",
    "primitive_type_sin_bool",
    "interpreter_missing_variable",
    "cache_strict_unknown_feature",
    "cache_hardened_unknown_feature",
    "vmap_axis_mismatch",
    "durability_missing_artifact",
    "unsupported_transform_tail_without_fallback",
    "unsupported_control_flow_grad_vmap_vector",
];

const ALLOWED_STRICT_HARDENED_DIVERGENCE_CASES: &[&str] = &[
    "cache_strict_unknown_feature",
    "cache_hardened_unknown_feature",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorTaxonomyCase {
    pub case_id: String,
    pub status: String,
    pub boundary: String,
    pub mode: String,
    pub input_class: String,
    pub expected_error_class: String,
    pub actual_error_class: String,
    pub error_variant: String,
    pub message_shape: String,
    pub actual_message: Option<String>,
    pub replay_hint: String,
    pub strict_behavior: String,
    pub hardened_behavior: String,
    pub strict_hardened_divergence: bool,
    pub divergence_allowed: bool,
    pub panic_status: String,
    pub evidence_refs: Vec<String>,
    pub replay_command: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorTaxonomyIssue {
    pub code: String,
    pub path: String,
    pub message: String,
}

impl ErrorTaxonomyIssue {
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
pub struct ErrorTaxonomyCoverage {
    pub required_case_count: usize,
    pub observed_case_count: usize,
    pub pass_count: usize,
    pub panic_free_count: usize,
    pub typed_error_count: usize,
    pub success_rows: usize,
    pub strict_hardened_divergence_count: usize,
    pub allowed_divergence_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorTaxonomyReport {
    pub schema_version: String,
    pub bead_id: String,
    pub generated_at_unix_ms: u128,
    pub status: String,
    pub matrix_policy: String,
    pub coverage: ErrorTaxonomyCoverage,
    pub strict_hardened_divergence_allowlist: Vec<String>,
    pub cases: Vec<ErrorTaxonomyCase>,
    pub issues: Vec<ErrorTaxonomyIssue>,
    pub artifact_refs: Vec<String>,
    pub replay_command: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorTaxonomyOutputPaths {
    pub report: PathBuf,
    pub markdown: PathBuf,
}

impl ErrorTaxonomyOutputPaths {
    #[must_use]
    pub fn for_root(root: &Path) -> Self {
        Self {
            report: root.join("artifacts/conformance/error_taxonomy_matrix.v1.json"),
            markdown: root.join("artifacts/conformance/error_taxonomy_matrix.v1.md"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ObservedOutcome {
    error_class: String,
    variant: String,
    message_shape: String,
    message: Option<String>,
}

impl ObservedOutcome {
    fn success() -> Self {
        Self {
            error_class: "none".to_owned(),
            variant: "ok".to_owned(),
            message_shape: "ok".to_owned(),
            message: None,
        }
    }
}

struct CaseSpec {
    case_id: &'static str,
    boundary: &'static str,
    mode: &'static str,
    input_class: &'static str,
    expected_error_class: &'static str,
    replay_hint: &'static str,
    strict_behavior: &'static str,
    hardened_behavior: &'static str,
    strict_hardened_divergence: bool,
    evidence_refs: Vec<&'static str>,
    action: fn(&Path) -> ObservedOutcome,
}

#[must_use]
pub fn build_error_taxonomy_report(root: &Path) -> ErrorTaxonomyReport {
    let cases = case_specs()
        .into_iter()
        .map(|spec| build_case(root, spec))
        .collect::<Vec<_>>();
    let pass_count = cases.iter().filter(|case| case.status == "pass").count();
    let panic_free_count = cases
        .iter()
        .filter(|case| case.panic_status == "no_panic")
        .count();
    let typed_error_count = cases
        .iter()
        .filter(|case| case.actual_error_class != "none")
        .count();
    let success_rows = cases
        .iter()
        .filter(|case| case.actual_error_class == "none")
        .count();
    let strict_hardened_divergence_count = cases
        .iter()
        .filter(|case| case.strict_hardened_divergence)
        .count();
    let allowed_divergence_count = cases
        .iter()
        .filter(|case| case.strict_hardened_divergence && case.divergence_allowed)
        .count();

    let coverage = ErrorTaxonomyCoverage {
        required_case_count: REQUIRED_CASES.len(),
        observed_case_count: cases.len(),
        pass_count,
        panic_free_count,
        typed_error_count,
        success_rows,
        strict_hardened_divergence_count,
        allowed_divergence_count,
    };
    let outputs = ErrorTaxonomyOutputPaths::for_root(root);
    let mut report = ErrorTaxonomyReport {
        schema_version: ERROR_TAXONOMY_REPORT_SCHEMA_VERSION.to_owned(),
        bead_id: ERROR_TAXONOMY_BEAD_ID.to_owned(),
        generated_at_unix_ms: now_unix_ms(),
        status: "pass".to_owned(),
        matrix_policy:
            "Every required cross-crate boundary must produce a typed, replayable, panic-free outcome with stable class, message shape, mode tag, boundary tag, and explicit strict/hardened divergence policy."
                .to_owned(),
        coverage,
        strict_hardened_divergence_allowlist: ALLOWED_STRICT_HARDENED_DIVERGENCE_CASES
            .iter()
            .map(|case| (*case).to_owned())
            .collect(),
        cases,
        issues: Vec::new(),
        artifact_refs: vec![
            repo_relative_artifact(&outputs.report),
            repo_relative_artifact(&outputs.markdown),
        ],
        replay_command: "./scripts/run_error_taxonomy_gate.sh --enforce".to_owned(),
    };
    report.issues = validate_error_taxonomy_report(&report);
    if !report.issues.is_empty() {
        report.status = "fail".to_owned();
    }
    report
}

pub fn write_error_taxonomy_outputs(
    root: &Path,
    report_path: &Path,
    markdown_path: &Path,
) -> Result<ErrorTaxonomyReport, std::io::Error> {
    let report = build_error_taxonomy_report(root);
    write_json(report_path, &report)?;
    write_markdown(markdown_path, &error_taxonomy_markdown(&report))?;
    Ok(report)
}

#[must_use]
pub fn error_taxonomy_summary_json(report: &ErrorTaxonomyReport) -> JsonValue {
    json!({
        "status": report.status,
        "schema_version": report.schema_version,
        "case_count": report.cases.len(),
        "coverage": report.coverage,
        "issue_count": report.issues.len(),
        "cases": report.cases.iter().map(|case| {
            json!({
                "case_id": case.case_id,
                "status": case.status,
                "boundary": case.boundary,
                "mode": case.mode,
                "input_class": case.input_class,
                "expected_error_class": case.expected_error_class,
                "actual_error_class": case.actual_error_class,
                "panic_status": case.panic_status,
                "strict_hardened_divergence": case.strict_hardened_divergence,
                "divergence_allowed": case.divergence_allowed,
            })
        }).collect::<Vec<_>>(),
    })
}

#[must_use]
pub fn error_taxonomy_markdown(report: &ErrorTaxonomyReport) -> String {
    let mut out = String::new();
    out.push_str("# Error Taxonomy Matrix Gate\n\n");
    out.push_str(&format!(
        "- schema: `{}`\n- bead: `{}`\n- status: `{}`\n- cases: `{}`\n- typed error rows: `{}`\n- success rows: `{}`\n- strict/hardened divergences: `{}`\n\n",
        report.schema_version,
        report.bead_id,
        report.status,
        report.cases.len(),
        report.coverage.typed_error_count,
        report.coverage.success_rows,
        report.coverage.strict_hardened_divergence_count
    ));
    out.push_str("| Case | Boundary | Mode | Input | Expected | Actual | Panic | Divergence |\n");
    out.push_str("|---|---|---|---|---|---|---|---|\n");
    for case in &report.cases {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
            case.case_id,
            case.boundary,
            case.mode,
            case.input_class.replace('|', "/"),
            case.expected_error_class,
            case.actual_error_class,
            case.panic_status,
            if case.strict_hardened_divergence {
                "allowed"
            } else {
                "none"
            }
        ));
    }

    out.push_str("\n## Strict/Hardened Divergence Allowlist\n\n");
    for case_id in &report.strict_hardened_divergence_allowlist {
        out.push_str(&format!("- `{case_id}`\n"));
    }

    out.push_str("\n## Replay Hints\n\n");
    for case in &report.cases {
        out.push_str(&format!(
            "- `{}`: {} (`{}`)\n",
            case.case_id, case.replay_hint, case.replay_command
        ));
    }

    out.push_str("\n## Issues\n\n");
    if report.issues.is_empty() {
        out.push_str("No error taxonomy matrix issues found.\n");
    } else {
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
pub fn validate_error_taxonomy_report(report: &ErrorTaxonomyReport) -> Vec<ErrorTaxonomyIssue> {
    let mut issues = Vec::new();
    if report.schema_version != ERROR_TAXONOMY_REPORT_SCHEMA_VERSION {
        issues.push(ErrorTaxonomyIssue::new(
            "unsupported_schema_version",
            "$.schema_version",
            format!(
                "expected {}, got {}",
                ERROR_TAXONOMY_REPORT_SCHEMA_VERSION, report.schema_version
            ),
        ));
    }
    if report.bead_id != ERROR_TAXONOMY_BEAD_ID {
        issues.push(ErrorTaxonomyIssue::new(
            "wrong_bead_id",
            "$.bead_id",
            "error taxonomy matrix must remain bound to frankenjax-cstq.8",
        ));
    }

    let allowed = report
        .strict_hardened_divergence_allowlist
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let mut seen = BTreeMap::<&str, usize>::new();
    for (idx, case) in report.cases.iter().enumerate() {
        *seen.entry(case.case_id.as_str()).or_default() += 1;
        let path = format!("$.cases[{idx}]");
        if case.status != "pass" {
            issues.push(ErrorTaxonomyIssue::new(
                "case_failed",
                format!("{path}.status"),
                format!("case `{}` did not pass", case.case_id),
            ));
        }
        if case.expected_error_class != case.actual_error_class {
            issues.push(ErrorTaxonomyIssue::new(
                "error_class_mismatch",
                format!("{path}.actual_error_class"),
                format!(
                    "expected `{}`, got `{}`",
                    case.expected_error_class, case.actual_error_class
                ),
            ));
        }
        if case.panic_status != "no_panic" {
            issues.push(ErrorTaxonomyIssue::new(
                "case_panicked",
                format!("{path}.panic_status"),
                "malformed inputs must return typed results instead of panicking",
            ));
        }
        for (field, value) in [
            ("boundary", case.boundary.as_str()),
            ("mode", case.mode.as_str()),
            ("input_class", case.input_class.as_str()),
            ("error_variant", case.error_variant.as_str()),
            ("message_shape", case.message_shape.as_str()),
            ("replay_hint", case.replay_hint.as_str()),
            ("strict_behavior", case.strict_behavior.as_str()),
            ("hardened_behavior", case.hardened_behavior.as_str()),
            ("replay_command", case.replay_command.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(ErrorTaxonomyIssue::new(
                    "missing_case_field",
                    format!("{path}.{field}"),
                    format!("case `{}` must set `{field}`", case.case_id),
                ));
            }
        }
        if case.evidence_refs.is_empty() {
            issues.push(ErrorTaxonomyIssue::new(
                "missing_evidence_refs",
                format!("{path}.evidence_refs"),
                "every taxonomy row must link to executable source, test, or artifact evidence",
            ));
        }
        if case.strict_hardened_divergence {
            if !allowed.contains(case.case_id.as_str()) || !case.divergence_allowed {
                issues.push(ErrorTaxonomyIssue::new(
                    "unapproved_strict_hardened_divergence",
                    format!("{path}.strict_hardened_divergence"),
                    format!(
                        "strict/hardened divergence for `{}` is not present in the allowlist",
                        case.case_id
                    ),
                ));
            }
        } else if case.divergence_allowed {
            issues.push(ErrorTaxonomyIssue::new(
                "spurious_divergence_allowance",
                format!("{path}.divergence_allowed"),
                "divergence_allowed must only be true for rows that actually diverge",
            ));
        }
    }

    for required in REQUIRED_CASES {
        match seen.get(required).copied().unwrap_or(0) {
            0 => issues.push(ErrorTaxonomyIssue::new(
                "missing_required_case",
                "$.cases",
                format!("missing required error taxonomy case `{required}`"),
            )),
            1 => {}
            count => issues.push(ErrorTaxonomyIssue::new(
                "duplicate_required_case",
                "$.cases",
                format!("case `{required}` appears {count} times"),
            )),
        }
    }

    if report.coverage.required_case_count != REQUIRED_CASES.len() {
        issues.push(ErrorTaxonomyIssue::new(
            "bad_required_case_count",
            "$.coverage.required_case_count",
            "coverage must record the complete required case set",
        ));
    }
    if report.coverage.observed_case_count != report.cases.len() {
        issues.push(ErrorTaxonomyIssue::new(
            "bad_observed_case_count",
            "$.coverage.observed_case_count",
            "observed case count must match cases length",
        ));
    }
    if report.coverage.pass_count
        != report
            .cases
            .iter()
            .filter(|case| case.status == "pass")
            .count()
    {
        issues.push(ErrorTaxonomyIssue::new(
            "bad_pass_count",
            "$.coverage.pass_count",
            "pass_count must match passing rows",
        ));
    }
    if report.replay_command.trim().is_empty() {
        issues.push(ErrorTaxonomyIssue::new(
            "missing_replay_command",
            "$.replay_command",
            "report must include a top-level replay command",
        ));
    }

    issues
}

fn build_case(root: &Path, spec: CaseSpec) -> ErrorTaxonomyCase {
    let caught = catch_unwind(AssertUnwindSafe(|| (spec.action)(root)));
    let (panic_status, observed) = match caught {
        Ok(observed) => ("no_panic".to_owned(), observed),
        Err(_) => (
            "panic".to_owned(),
            ObservedOutcome {
                error_class: "panic".to_owned(),
                variant: "panic".to_owned(),
                message_shape: "panic".to_owned(),
                message: Some("case panicked while evaluating malformed input".to_owned()),
            },
        ),
    };
    let divergence_allowed = spec.strict_hardened_divergence
        && ALLOWED_STRICT_HARDENED_DIVERGENCE_CASES.contains(&spec.case_id);
    let status = if panic_status == "no_panic" && observed.error_class == spec.expected_error_class
    {
        "pass"
    } else {
        "fail"
    };
    ErrorTaxonomyCase {
        case_id: spec.case_id.to_owned(),
        status: status.to_owned(),
        boundary: spec.boundary.to_owned(),
        mode: spec.mode.to_owned(),
        input_class: spec.input_class.to_owned(),
        expected_error_class: spec.expected_error_class.to_owned(),
        actual_error_class: observed.error_class,
        error_variant: observed.variant,
        message_shape: observed.message_shape,
        actual_message: observed.message,
        replay_hint: spec.replay_hint.to_owned(),
        strict_behavior: spec.strict_behavior.to_owned(),
        hardened_behavior: spec.hardened_behavior.to_owned(),
        strict_hardened_divergence: spec.strict_hardened_divergence,
        divergence_allowed,
        panic_status,
        evidence_refs: spec.evidence_refs.into_iter().map(str::to_owned).collect(),
        replay_command: format!(
            "./scripts/run_error_taxonomy_gate.sh --enforce --case {}",
            spec.case_id
        ),
    }
}

fn case_specs() -> Vec<CaseSpec> {
    vec![
        CaseSpec {
            case_id: "ir_validation_unknown_outvar",
            boundary: "fj-core::Jaxpr::validate_well_formed",
            mode: "strict_and_hardened",
            input_class: "jaxpr outvar has no defining input, const, or equation binding",
            expected_error_class: "ir_validation.unknown_outvar",
            replay_hint: "Validate a Jaxpr whose only outvar is v99.",
            strict_behavior: "Rejects malformed IR with JaxprValidationError::UnknownOutvar.",
            hardened_behavior: "Same fail-closed typed rejection; no repair is attempted.",
            strict_hardened_divergence: false,
            evidence_refs: vec!["crates/fj-core/src/lib.rs::JaxprValidationError"],
            action: case_ir_validation_unknown_outvar,
        },
        CaseSpec {
            case_id: "transform_proof_duplicate_evidence",
            boundary: "fj-core::verify_transform_composition",
            mode: "strict_and_hardened",
            input_class: "TraceTransformLedger repeats an evidence id across two transforms",
            expected_error_class: "transform_proof.duplicate_evidence",
            replay_hint: "Verify a JIT/JIT ledger with the same evidence string in both slots.",
            strict_behavior: "Rejects duplicated proof evidence.",
            hardened_behavior: "Same fail-closed typed rejection.",
            strict_hardened_divergence: false,
            evidence_refs: vec!["crates/fj-core/src/lib.rs::TransformCompositionError"],
            action: case_transform_proof_duplicate_evidence,
        },
        CaseSpec {
            case_id: "transform_proof_missing_evidence",
            boundary: "fj-core::verify_transform_composition",
            mode: "strict_and_hardened",
            input_class: "TraceTransformLedger has fewer evidence entries than transforms",
            expected_error_class: "transform_proof.evidence_count_mismatch",
            replay_hint: "Verify a ledger with one JIT transform and zero evidence ids.",
            strict_behavior: "Rejects transform/evidence cardinality mismatch.",
            hardened_behavior: "Same fail-closed typed rejection.",
            strict_hardened_divergence: false,
            evidence_refs: vec!["crates/fj-core/src/lib.rs::verify_transform_composition"],
            action: case_transform_proof_missing_evidence,
        },
        CaseSpec {
            case_id: "primitive_arity_add",
            boundary: "fj-lax::eval_primitive",
            mode: "strict_and_hardened",
            input_class: "add called with one scalar operand",
            expected_error_class: "primitive.arity_mismatch",
            replay_hint: "Evaluate add with a single I64 scalar input.",
            strict_behavior: "Returns EvalError::ArityMismatch.",
            hardened_behavior: "Same typed rejection; no operand synthesis.",
            strict_hardened_divergence: false,
            evidence_refs: vec!["crates/fj-lax/src/arithmetic.rs::eval_binary_elementwise"],
            action: case_primitive_arity_add,
        },
        CaseSpec {
            case_id: "primitive_shape_add_broadcast",
            boundary: "fj-lax::eval_primitive",
            mode: "strict_and_hardened",
            input_class: "add called with non-broadcastable [2] and [3] tensors",
            expected_error_class: "primitive.shape_mismatch",
            replay_hint: "Evaluate add on F64 vectors with lengths 2 and 3.",
            strict_behavior: "Returns EvalError::ShapeMismatch.",
            hardened_behavior: "Same typed rejection; no implicit resize.",
            strict_hardened_divergence: false,
            evidence_refs: vec!["crates/fj-lax/src/arithmetic.rs::broadcast_binary_tensors"],
            action: case_primitive_shape_add_broadcast,
        },
        CaseSpec {
            case_id: "primitive_type_sin_bool",
            boundary: "fj-lax::eval_primitive",
            mode: "strict_and_hardened",
            input_class: "sin called with bool scalar",
            expected_error_class: "primitive.type_mismatch",
            replay_hint: "Evaluate sin(true).",
            strict_behavior: "Returns EvalError::TypeMismatch with numeric-scalar message shape.",
            hardened_behavior: "Same typed rejection; bool is not coerced for unary trig.",
            strict_hardened_divergence: false,
            evidence_refs: vec!["crates/fj-lax/src/arithmetic.rs::eval_unary_elementwise"],
            action: case_primitive_type_sin_bool,
        },
        CaseSpec {
            case_id: "interpreter_missing_variable",
            boundary: "fj-interpreters::eval_jaxpr",
            mode: "strict_and_hardened",
            input_class: "equation references v7 before any binding",
            expected_error_class: "interpreter.missing_variable",
            replay_hint: "Evaluate a one-equation Jaxpr whose add input is an unbound var.",
            strict_behavior: "Returns InterpreterError::MissingVariable.",
            hardened_behavior: "Same typed rejection; no implicit zero/default binding.",
            strict_hardened_divergence: false,
            evidence_refs: vec!["crates/fj-interpreters/src/lib.rs::resolve_equation_inputs"],
            action: case_interpreter_missing_variable,
        },
        CaseSpec {
            case_id: "cache_strict_unknown_feature",
            boundary: "fj-cache::build_cache_key",
            mode: "strict",
            input_class: "cache key input contains unknown incompatible feature marker",
            expected_error_class: "cache.unknown_incompatible_features",
            replay_hint: "Build a strict cache key with unknown_incompatible_features=[future_xla_flag].",
            strict_behavior: "Strict rejects unknown incompatible feature names.",
            hardened_behavior: "Hardened includes unknown names in the deterministic key payload.",
            strict_hardened_divergence: true,
            evidence_refs: vec!["crates/fj-cache/src/lib.rs::build_cache_key"],
            action: case_cache_strict_unknown_feature,
        },
        CaseSpec {
            case_id: "cache_hardened_unknown_feature",
            boundary: "fj-cache::build_cache_key",
            mode: "hardened",
            input_class: "cache key input contains unknown incompatible feature marker",
            expected_error_class: "none",
            replay_hint: "Build a hardened cache key with unknown_incompatible_features=[future_xla_flag].",
            strict_behavior: "Strict rejects the same input.",
            hardened_behavior: "Hardened succeeds while preserving the feature in the hash payload.",
            strict_hardened_divergence: true,
            evidence_refs: vec!["crates/fj-cache/src/lib.rs::canonical_payload"],
            action: case_cache_hardened_unknown_feature,
        },
        CaseSpec {
            case_id: "vmap_axis_mismatch",
            boundary: "fj-dispatch::dispatch",
            mode: "strict_and_hardened",
            input_class: "vmap over two vectors with leading dimensions 2 and 3",
            expected_error_class: "transform_execution.vmap_mismatched_leading_dimension",
            replay_hint: "Dispatch vmap(add) with vector lengths 2 and 3.",
            strict_behavior: "Returns TransformExecutionError::VmapMismatchedLeadingDimension.",
            hardened_behavior: "Same typed rejection; no truncation or padding.",
            strict_hardened_divergence: false,
            evidence_refs: vec!["crates/fj-dispatch/src/lib.rs::execute_vmap"],
            action: case_vmap_axis_mismatch,
        },
        CaseSpec {
            case_id: "durability_missing_artifact",
            boundary: "fj-conformance::durability::encode_artifact_to_sidecar",
            mode: "strict_and_hardened",
            input_class: "durability sidecar generation points at a missing artifact path",
            expected_error_class: "durability.io",
            replay_hint: "Attempt sidecar generation for a missing artifact file.",
            strict_behavior: "Returns DurabilityError::Io from the missing artifact read.",
            hardened_behavior: "Same typed rejection; no empty artifact is synthesized.",
            strict_hardened_divergence: false,
            evidence_refs: vec![
                "crates/fj-conformance/src/durability.rs::encode_artifact_to_sidecar",
            ],
            action: case_durability_missing_artifact,
        },
        CaseSpec {
            case_id: "unsupported_transform_tail_without_fallback",
            boundary: "fj-dispatch::dispatch",
            mode: "strict_and_hardened",
            input_class: "grad(vmap(square)) with finite-difference fallback explicitly denied",
            expected_error_class: "transform_execution.finite_diff_grad_fallback_disabled",
            replay_hint: "Dispatch grad(vmap(square)) with allow_finite_diff_grad_fallback=deny.",
            strict_behavior: "Rejects unsupported transform tail with typed fallback-disabled error.",
            hardened_behavior: "Same fail-closed transform policy.",
            strict_hardened_divergence: false,
            evidence_refs: vec!["crates/fj-dispatch/src/lib.rs::execute_grad"],
            action: case_unsupported_transform_tail_without_fallback,
        },
        CaseSpec {
            case_id: "unsupported_control_flow_grad_vmap_vector",
            boundary: "fj-dispatch::dispatch",
            mode: "strict_and_hardened",
            input_class: "grad(vmap(cond)) row receives a vector first argument in V1 scope",
            expected_error_class: "transform_execution.non_scalar_gradient_input",
            replay_hint: "Dispatch grad(vmap(cond_select)) with a vector first argument.",
            strict_behavior: "Rejects the V1 non-scalar grad input before silently differentiating a vector contract.",
            hardened_behavior: "Same fail-closed transform/control-flow boundary.",
            strict_hardened_divergence: false,
            evidence_refs: vec![
                "crates/fj-dispatch/src/lib.rs::execute_grad",
                "artifacts/conformance/transform_control_flow_matrix.v1.json",
            ],
            action: case_unsupported_control_flow_grad_vmap_vector,
        },
    ]
}

fn case_ir_validation_unknown_outvar(_: &Path) -> ObservedOutcome {
    let jaxpr = Jaxpr::new(Vec::new(), Vec::new(), vec![VarId(99)], Vec::new());
    match jaxpr.validate_well_formed() {
        Ok(()) => ObservedOutcome::success(),
        Err(err) => classify_jaxpr_validation_error(&err),
    }
}

fn case_transform_proof_duplicate_evidence(_: &Path) -> ObservedOutcome {
    let mut ledger = TraceTransformLedger::new(build_program(ProgramSpec::Square));
    ledger.push_transform(Transform::Jit, "jit:duplicate-proof");
    ledger.push_transform(Transform::Jit, "jit:duplicate-proof");
    match verify_transform_composition(&ledger) {
        Ok(_) => ObservedOutcome::success(),
        Err(err) => classify_transform_composition_error(&err),
    }
}

fn case_transform_proof_missing_evidence(_: &Path) -> ObservedOutcome {
    let ledger = TraceTransformLedger {
        root_jaxpr: build_program(ProgramSpec::Square),
        transform_stack: vec![Transform::Jit],
        transform_evidence: Vec::new(),
    };
    match verify_transform_composition(&ledger) {
        Ok(_) => ObservedOutcome::success(),
        Err(err) => classify_transform_composition_error(&err),
    }
}

fn case_primitive_arity_add(_: &Path) -> ObservedOutcome {
    match eval_primitive(Primitive::Add, &[Value::scalar_i64(1)], &BTreeMap::new()) {
        Ok(_) => ObservedOutcome::success(),
        Err(err) => classify_eval_error(&err),
    }
}

fn case_primitive_shape_add_broadcast(_: &Path) -> ObservedOutcome {
    let lhs = tensor_f64(&[2], &[1.0, 2.0]);
    let rhs = tensor_f64(&[3], &[3.0, 4.0, 5.0]);
    match (lhs, rhs) {
        (Ok(lhs), Ok(rhs)) => match eval_primitive(Primitive::Add, &[lhs, rhs], &BTreeMap::new()) {
            Ok(_) => ObservedOutcome::success(),
            Err(err) => classify_eval_error(&err),
        },
        (Err(err), _) | (_, Err(err)) => ObservedOutcome {
            error_class: "fixture_build.tensor".to_owned(),
            variant: "ValueError".to_owned(),
            message_shape: "tensor fixture failed: {detail}".to_owned(),
            message: Some(err.to_string()),
        },
    }
}

fn case_primitive_type_sin_bool(_: &Path) -> ObservedOutcome {
    match eval_primitive(
        Primitive::Sin,
        &[Value::scalar_bool(true)],
        &BTreeMap::new(),
    ) {
        Ok(_) => ObservedOutcome::success(),
        Err(err) => classify_eval_error(&err),
    }
}

fn case_interpreter_missing_variable(_: &Path) -> ObservedOutcome {
    let jaxpr = Jaxpr::new(
        Vec::new(),
        Vec::new(),
        vec![VarId(2)],
        vec![Equation {
            primitive: Primitive::Add,
            inputs: smallvec![Atom::Var(VarId(7)), Atom::Lit(Literal::I64(1))],
            outputs: smallvec![VarId(2)],
            params: BTreeMap::new(),
            effects: Vec::new(),
            sub_jaxprs: Vec::new(),
        }],
    );
    match eval_jaxpr(&jaxpr, &[]) {
        Ok(_) => ObservedOutcome::success(),
        Err(err) => classify_interpreter_error(&err),
    }
}

fn case_cache_strict_unknown_feature(_: &Path) -> ObservedOutcome {
    let input = cache_input(CompatibilityMode::Strict);
    match build_cache_key(&input) {
        Ok(_) => ObservedOutcome::success(),
        Err(err) => classify_cache_key_error(&err),
    }
}

fn case_cache_hardened_unknown_feature(_: &Path) -> ObservedOutcome {
    let input = cache_input(CompatibilityMode::Hardened);
    match build_cache_key(&input) {
        Ok(_) => ObservedOutcome::success(),
        Err(err) => classify_cache_key_error(&err),
    }
}

fn case_vmap_axis_mismatch(_: &Path) -> ObservedOutcome {
    let lhs = Value::vector_f64(&[1.0, 2.0]);
    let rhs = Value::vector_f64(&[3.0, 4.0, 5.0]);
    match (lhs, rhs) {
        (Ok(lhs), Ok(rhs)) => match dispatch(dispatch_request(
            CompatibilityMode::Strict,
            ProgramSpec::Add2,
            &[Transform::Vmap],
            vec![lhs, rhs],
            BTreeMap::new(),
            Vec::new(),
        )) {
            Ok(_) => ObservedOutcome::success(),
            Err(err) => classify_dispatch_error(&err),
        },
        (Err(err), _) | (_, Err(err)) => ObservedOutcome {
            error_class: "fixture_build.tensor".to_owned(),
            variant: "ValueError".to_owned(),
            message_shape: "tensor fixture failed: {detail}".to_owned(),
            message: Some(err.to_string()),
        },
    }
}

fn case_durability_missing_artifact(root: &Path) -> ObservedOutcome {
    let missing_artifact = root.join("artifacts/conformance/error_taxonomy_missing_input.bin");
    let sidecar = root.join("artifacts/conformance/error_taxonomy_missing_input.sidecar.json");
    match encode_artifact_to_sidecar(&missing_artifact, &sidecar, &SidecarConfig::default()) {
        Ok(_) => ObservedOutcome::success(),
        Err(err) => classify_durability_error(&err),
    }
}

fn case_unsupported_transform_tail_without_fallback(_: &Path) -> ObservedOutcome {
    let options = BTreeMap::from([(
        "allow_finite_diff_grad_fallback".to_owned(),
        "deny".to_owned(),
    )]);
    match dispatch(dispatch_request(
        CompatibilityMode::Strict,
        ProgramSpec::Square,
        &[Transform::Grad, Transform::Vmap],
        vec![Value::scalar_f64(3.0)],
        options,
        Vec::new(),
    )) {
        Ok(_) => ObservedOutcome::success(),
        Err(err) => classify_dispatch_error(&err),
    }
}

fn case_unsupported_control_flow_grad_vmap_vector(_: &Path) -> ObservedOutcome {
    let first_arg = Value::vector_f64(&[1.0, 2.0]);
    match first_arg {
        Ok(first_arg) => match dispatch(dispatch_request(
            CompatibilityMode::Strict,
            ProgramSpec::CondSelect,
            &[Transform::Grad, Transform::Vmap],
            vec![first_arg],
            BTreeMap::new(),
            Vec::new(),
        )) {
            Ok(_) => ObservedOutcome::success(),
            Err(err) => classify_dispatch_error(&err),
        },
        Err(err) => ObservedOutcome {
            error_class: "fixture_build.tensor".to_owned(),
            variant: "ValueError".to_owned(),
            message_shape: "tensor fixture failed: {detail}".to_owned(),
            message: Some(err.to_string()),
        },
    }
}

fn classify_jaxpr_validation_error(err: &JaxprValidationError) -> ObservedOutcome {
    let (class, variant, shape) = match err {
        JaxprValidationError::DuplicateBinding { .. } => (
            "ir_validation.duplicate_binding",
            "JaxprValidationError::DuplicateBinding",
            "duplicate binding in {section} for var v{var}",
        ),
        JaxprValidationError::UnboundInputVar { .. } => (
            "ir_validation.unbound_input_var",
            "JaxprValidationError::UnboundInputVar",
            "equation {equation_index} references unbound input var v{var}",
        ),
        JaxprValidationError::OutputShadowsBinding { .. } => (
            "ir_validation.output_shadows_binding",
            "JaxprValidationError::OutputShadowsBinding",
            "equation {equation_index} output var v{var} shadows an existing binding",
        ),
        JaxprValidationError::UnknownOutvar { .. } => (
            "ir_validation.unknown_outvar",
            "JaxprValidationError::UnknownOutvar",
            "outvar v{var} does not have a defining binding",
        ),
    };
    observed(class, variant, shape, err.to_string())
}

fn classify_transform_composition_error(err: &TransformCompositionError) -> ObservedOutcome {
    let (class, variant, shape) = match err {
        TransformCompositionError::EvidenceCountMismatch { .. } => (
            "transform_proof.evidence_count_mismatch",
            "TransformCompositionError::EvidenceCountMismatch",
            "transform/evidence cardinality mismatch: transforms={transform_count}, evidence={evidence_count}",
        ),
        TransformCompositionError::EmptyEvidence { .. } => (
            "transform_proof.empty_evidence",
            "TransformCompositionError::EmptyEvidence",
            "transform evidence at index {index} for {transform} is empty",
        ),
        TransformCompositionError::EvidenceTransformMismatch { .. } => (
            "transform_proof.evidence_transform_mismatch",
            "TransformCompositionError::EvidenceTransformMismatch",
            "transform evidence at index {index} does not bind to {transform}: {evidence}",
        ),
        TransformCompositionError::DuplicateEvidence { .. } => (
            "transform_proof.duplicate_evidence",
            "TransformCompositionError::DuplicateEvidence",
            "transform evidence at index {index} duplicates an earlier evidence id: {evidence}",
        ),
    };
    observed(class, variant, shape, err.to_string())
}

fn classify_eval_error(err: &EvalError) -> ObservedOutcome {
    let (class, variant, shape) = match err {
        EvalError::ArityMismatch { .. } => (
            "primitive.arity_mismatch",
            "EvalError::ArityMismatch",
            "arity mismatch for {primitive}: expected {expected}, got {actual}",
        ),
        EvalError::TypeMismatch { .. } => (
            "primitive.type_mismatch",
            "EvalError::TypeMismatch",
            "type mismatch for {primitive}: {detail}",
        ),
        EvalError::ShapeMismatch { .. } => (
            "primitive.shape_mismatch",
            "EvalError::ShapeMismatch",
            "shape mismatch for {primitive}: left={left_shape} right={right_shape}",
        ),
        EvalError::Unsupported { .. } => (
            "primitive.unsupported",
            "EvalError::Unsupported",
            "unsupported {primitive} behavior: {detail}",
        ),
        EvalError::InvalidTensor(_) => (
            "primitive.invalid_tensor",
            "EvalError::InvalidTensor",
            "invalid tensor: {detail}",
        ),
        EvalError::MaxIterationsExceeded { .. } => (
            "primitive.max_iterations_exceeded",
            "EvalError::MaxIterationsExceeded",
            "{primitive} exceeded max iterations ({max_iterations})",
        ),
        EvalError::ShapeChanged { .. } => (
            "primitive.shape_changed",
            "EvalError::ShapeChanged",
            "{primitive} body changed carry shape: {detail}",
        ),
    };
    observed(class, variant, shape, err.to_string())
}

fn classify_interpreter_error(err: &InterpreterError) -> ObservedOutcome {
    let (class, variant, shape) = match err {
        InterpreterError::InputArity { .. } => (
            "interpreter.input_arity",
            "InterpreterError::InputArity",
            "input arity mismatch: expected {expected}, got {actual}",
        ),
        InterpreterError::ConstArity { .. } => (
            "interpreter.const_arity",
            "InterpreterError::ConstArity",
            "const arity mismatch: expected {expected}, got {actual}",
        ),
        InterpreterError::MissingVariable(_) => (
            "interpreter.missing_variable",
            "InterpreterError::MissingVariable",
            "missing variable v{var}",
        ),
        InterpreterError::UnexpectedOutputArity { .. } => (
            "interpreter.unexpected_output_arity",
            "InterpreterError::UnexpectedOutputArity",
            "primitive {primitive} returned {actual} outputs for {expected} bindings",
        ),
        InterpreterError::InvariantViolation { .. } => (
            "interpreter.invariant_violation",
            "InterpreterError::InvariantViolation",
            "interpreter invariant violated: {detail}",
        ),
        InterpreterError::Primitive(_) => (
            "interpreter.primitive",
            "InterpreterError::Primitive",
            "primitive eval failed: {detail}",
        ),
    };
    observed(class, variant, shape, err.to_string())
}

fn classify_cache_key_error(err: &CacheKeyError) -> ObservedOutcome {
    let (class, variant, shape) = match err {
        CacheKeyError::UnknownIncompatibleFeatures { .. } => (
            "cache.unknown_incompatible_features",
            "CacheKeyError::UnknownIncompatibleFeatures",
            "strict mode rejected unknown incompatible features: {features}",
        ),
    };
    observed(class, variant, shape, err.to_string())
}

fn classify_durability_error(err: &DurabilityError) -> ObservedOutcome {
    let (class, variant, shape) = match err {
        DurabilityError::Io(_) => ("durability.io", "DurabilityError::Io", "io error: {detail}"),
        DurabilityError::Json(_) => (
            "durability.json",
            "DurabilityError::Json",
            "json error: {detail}",
        ),
        DurabilityError::InvalidConfig(_) => (
            "durability.invalid_config",
            "DurabilityError::InvalidConfig",
            "invalid durability config: {detail}",
        ),
        DurabilityError::Encode(_) => (
            "durability.encode",
            "DurabilityError::Encode",
            "encode error: {detail}",
        ),
        DurabilityError::Decode(_) => (
            "durability.decode",
            "DurabilityError::Decode",
            "decode error: {detail}",
        ),
        DurabilityError::Integrity(_) => (
            "durability.integrity",
            "DurabilityError::Integrity",
            "integrity error: {detail}",
        ),
    };
    observed(class, variant, shape, err.to_string())
}

fn classify_dispatch_error(err: &DispatchError) -> ObservedOutcome {
    match err {
        DispatchError::Cache(cache_err) => classify_cache_key_error(cache_err),
        DispatchError::Interpreter(interpreter_err) => classify_interpreter_error(interpreter_err),
        DispatchError::BackendExecution(_) => observed(
            "backend_execution",
            "DispatchError::BackendExecution",
            "backend execution error: {detail}",
            err.to_string(),
        ),
        DispatchError::TransformInvariant(transform_err) => {
            classify_transform_composition_error(transform_err)
        }
        DispatchError::TransformExecution(transform_err) => {
            classify_transform_execution_error(transform_err)
        }
    }
}

fn classify_transform_execution_error(err: &TransformExecutionError) -> ObservedOutcome {
    let (class, variant, shape) = match err {
        TransformExecutionError::EmptyArgumentList { .. } => (
            "transform_execution.empty_argument_list",
            "TransformExecutionError::EmptyArgumentList",
            "{transform} requires at least one argument",
        ),
        TransformExecutionError::NonScalarGradientInput => (
            "transform_execution.non_scalar_gradient_input",
            "TransformExecutionError::NonScalarGradientInput",
            "grad currently requires scalar first input",
        ),
        TransformExecutionError::NonScalarGradientOutput => (
            "transform_execution.non_scalar_gradient_output",
            "TransformExecutionError::NonScalarGradientOutput",
            "grad currently requires scalar first output",
        ),
        TransformExecutionError::VmapRequiresRankOneLeadingArgument => (
            "transform_execution.vmap_requires_rank_one_leading_argument",
            "TransformExecutionError::VmapRequiresRankOneLeadingArgument",
            "vmap currently requires first argument with rank >= 1",
        ),
        TransformExecutionError::VmapMismatchedLeadingDimension { .. } => (
            "transform_execution.vmap_mismatched_leading_dimension",
            "TransformExecutionError::VmapMismatchedLeadingDimension",
            "vmap leading-dimension mismatch: expected {expected}, got {actual}",
        ),
        TransformExecutionError::VmapInconsistentOutputArity { .. } => (
            "transform_execution.vmap_inconsistent_output_arity",
            "TransformExecutionError::VmapInconsistentOutputArity",
            "vmap inner output arity mismatch: expected {expected}, got {actual}",
        ),
        TransformExecutionError::VmapAxesOutOfBounds { .. } => (
            "transform_execution.vmap_axes_out_of_bounds",
            "TransformExecutionError::VmapAxesOutOfBounds",
            "vmap axis {axis} is out of bounds for tensor with {ndim} dimensions",
        ),
        TransformExecutionError::VmapAxesCountMismatch { .. } => (
            "transform_execution.vmap_axes_count_mismatch",
            "TransformExecutionError::VmapAxesCountMismatch",
            "vmap in_axes/out_axes length mismatch: expected {expected}, got {actual}",
        ),
        TransformExecutionError::InvalidVmapAxisSpec { .. } => (
            "transform_execution.invalid_vmap_axis_spec",
            "TransformExecutionError::InvalidVmapAxisSpec",
            "invalid {option} axis spec {value}; expected an integer or \"none\"",
        ),
        TransformExecutionError::VmapUnmappedOutputMismatch => (
            "transform_execution.vmap_unmapped_output_mismatch",
            "TransformExecutionError::VmapUnmappedOutputMismatch",
            "vmap out_axes=none requires the output to be identical across mapped elements",
        ),
        TransformExecutionError::FiniteDiffGradFallbackDisabled { .. } => (
            "transform_execution.finite_diff_grad_fallback_disabled",
            "TransformExecutionError::FiniteDiffGradFallbackDisabled",
            "finite-difference grad fallback is disabled for remaining transform tail [{tail}]",
        ),
        TransformExecutionError::EmptyVmapOutput => (
            "transform_execution.empty_vmap_output",
            "TransformExecutionError::EmptyVmapOutput",
            "vmap received no mapped elements",
        ),
        TransformExecutionError::TensorBuild(_) => (
            "transform_execution.tensor_build",
            "TransformExecutionError::TensorBuild",
            "tensor build error: {detail}",
        ),
    };
    observed(class, variant, shape, err.to_string())
}

fn observed(
    error_class: impl Into<String>,
    variant: impl Into<String>,
    message_shape: impl Into<String>,
    message: String,
) -> ObservedOutcome {
    ObservedOutcome {
        error_class: error_class.into(),
        variant: variant.into(),
        message_shape: message_shape.into(),
        message: Some(message),
    }
}

fn cache_input(mode: CompatibilityMode) -> CacheKeyInput {
    CacheKeyInput {
        mode,
        backend: "cpu".to_owned(),
        jaxpr: build_program(ProgramSpec::Square),
        transform_stack: vec![Transform::Jit],
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec!["future_xla_flag".to_owned()],
    }
}

fn dispatch_request(
    mode: CompatibilityMode,
    program: ProgramSpec,
    transforms: &[Transform],
    args: Vec<Value>,
    compile_options: BTreeMap<String, String>,
    unknown_incompatible_features: Vec<String>,
) -> DispatchRequest {
    DispatchRequest {
        mode,
        ledger: valid_ledger(program, transforms),
        args,
        backend: "cpu".to_owned(),
        compile_options,
        custom_hook: None,
        unknown_incompatible_features,
    }
}

fn valid_ledger(program: ProgramSpec, transforms: &[Transform]) -> TraceTransformLedger {
    let mut ledger = TraceTransformLedger::new(build_program(program));
    for (idx, transform) in transforms.iter().enumerate() {
        ledger.push_transform(
            *transform,
            format!("{}:error-taxonomy-evidence-{idx}", transform.as_str()),
        );
    }
    ledger
}

fn tensor_f64(dims: &[u32], values: &[f64]) -> Result<Value, fj_core::ValueError> {
    let elements = values
        .iter()
        .copied()
        .map(Literal::from_f64)
        .collect::<Vec<_>>();
    TensorValue::new(
        DType::F64,
        Shape {
            dims: dims.to_vec(),
        },
        elements,
    )
    .map(Value::Tensor)
}

fn repo_relative_artifact(path: &Path) -> String {
    let text = path.display().to_string();
    match text.find("artifacts/") {
        Some(idx) => text[idx..].to_owned(),
        None => text,
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), std::io::Error> {
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
        .unwrap_or_default()
        .as_millis()
}
