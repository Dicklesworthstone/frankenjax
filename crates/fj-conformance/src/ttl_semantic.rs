#![forbid(unsafe_code)]

use fj_core::{
    Atom, CompatibilityMode, DType, Equation, Jaxpr, Literal, Primitive, ProgramSpec,
    TraceTransformLedger, Transform, TransformCompositionError, Value, VarId, build_program,
    verify_transform_composition,
};
use fj_dispatch::{DispatchError, DispatchRequest, TransformExecutionError, dispatch};
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};
use sha2::{Digest, Sha256};
use smallvec::smallvec;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const TTL_SEMANTIC_REPORT_SCHEMA_VERSION: &str = "frankenjax.ttl-semantic-proof-matrix.v1";
pub const TTL_SEMANTIC_BEAD_ID: &str = "frankenjax-cstq.3";

const REQUIRED_CASES: &[&str] = &[
    "valid_single_jit_square",
    "valid_single_grad_square",
    "valid_single_vmap_square",
    "valid_jit_grad_fixture",
    "valid_grad_jit_order_sensitive",
    "valid_vmap_grad_fixture",
    "fail_closed_grad_vmap_vector_output",
    "invalid_duplicate_evidence",
    "invalid_missing_evidence",
    "invalid_stale_input_fingerprint",
    "invalid_wrong_transform_binding",
    "invalid_missing_fixture_link",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TtlSemanticProofCase {
    pub case_id: String,
    pub status: String,
    pub proof_kind: String,
    pub compatibility_mode: String,
    pub program: String,
    pub oracle_fixture_id: Option<String>,
    pub fixture_link_required: bool,
    pub transform_stack: Vec<String>,
    pub evidence_ids: Vec<String>,
    pub transform_count: usize,
    pub evidence_count: usize,
    pub input_fingerprint: String,
    pub expected_input_fingerprint: String,
    pub stack_signature: Option<String>,
    pub stack_hash_hex: Option<String>,
    pub output_fingerprint: Option<String>,
    pub expected_output_fingerprint: Option<String>,
    pub output_shapes: Vec<Vec<u32>>,
    pub output_dtypes: Vec<String>,
    pub inputs: JsonValue,
    pub expected_outputs: JsonValue,
    pub actual_outputs: JsonValue,
    pub expected_decision: String,
    pub verifier_decision: String,
    pub rejection_reason: Option<String>,
    pub structural_checks: Vec<String>,
    pub evidence_refs: Vec<String>,
    pub replay_command: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TtlSemanticIssue {
    pub code: String,
    pub path: String,
    pub message: String,
}

impl TtlSemanticIssue {
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
pub struct TtlSemanticCoverage {
    pub required_case_count: usize,
    pub observed_case_count: usize,
    pub pass_count: usize,
    pub accepted_count: usize,
    pub rejected_count: usize,
    pub fixture_linked_count: usize,
    pub structural_replay_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TtlSemanticReport {
    pub schema_version: String,
    pub bead_id: String,
    pub generated_at_unix_ms: u128,
    pub status: String,
    pub matrix_policy: String,
    pub coverage: TtlSemanticCoverage,
    pub cases: Vec<TtlSemanticProofCase>,
    pub issues: Vec<TtlSemanticIssue>,
    pub artifact_refs: Vec<String>,
    pub replay_command: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TtlSemanticOutputPaths {
    pub report: PathBuf,
    pub markdown: PathBuf,
}

impl TtlSemanticOutputPaths {
    #[must_use]
    pub fn for_root(root: &Path) -> Self {
        Self {
            report: root.join("artifacts/conformance/ttl_semantic_proof_matrix.v1.json"),
            markdown: root.join("artifacts/conformance/ttl_semantic_proof_matrix.v1.md"),
        }
    }
}

#[must_use]
pub fn build_ttl_semantic_report(root: &Path) -> TtlSemanticReport {
    let paths = TtlSemanticOutputPaths::for_root(root);
    build_ttl_semantic_report_for_outputs(root, &paths.report, &paths.markdown)
}

#[must_use]
pub fn build_ttl_semantic_report_for_outputs(
    root: &Path,
    report_path: &Path,
    markdown_path: &Path,
) -> TtlSemanticReport {
    let cases = build_cases();
    let pass_count = cases.iter().filter(|case| case.status == "pass").count();
    let accepted_count = cases
        .iter()
        .filter(|case| case.verifier_decision == "accept")
        .count();
    let rejected_count = cases
        .iter()
        .filter(|case| case.verifier_decision == "reject")
        .count();
    let fixture_linked_count = cases
        .iter()
        .filter(|case| {
            case.oracle_fixture_id
                .as_deref()
                .is_some_and(|id| !id.trim().is_empty() && !id.starts_with("analytic:"))
        })
        .count();
    let structural_replay_count = cases
        .iter()
        .filter(|case| {
            case.structural_checks
                .iter()
                .any(|check| check == "shape_dtype_matched")
        })
        .count();

    let mut report = TtlSemanticReport {
        schema_version: TTL_SEMANTIC_REPORT_SCHEMA_VERSION.to_owned(),
        bead_id: TTL_SEMANTIC_BEAD_ID.to_owned(),
        generated_at_unix_ms: now_unix_ms(),
        status: "pass".to_owned(),
        matrix_policy:
            "Strict TTL semantic proof gate: bind canonical input fingerprint, ordered transform stack, evidence ids, stack signature/hash, structural output metadata, and oracle fixture links; invalid proof chains must reject deterministically."
                .to_owned(),
        coverage: TtlSemanticCoverage {
            required_case_count: REQUIRED_CASES.len(),
            observed_case_count: cases.len(),
            pass_count,
            accepted_count,
            rejected_count,
            fixture_linked_count,
            structural_replay_count,
        },
        cases,
        issues: Vec::new(),
        artifact_refs: vec![
            repo_relative_artifact(root, report_path),
            repo_relative_artifact(root, markdown_path),
        ],
        replay_command: "./scripts/run_ttl_semantic_gate.sh --enforce".to_owned(),
    };
    report.issues = validate_ttl_semantic_report(&report);
    if !report.issues.is_empty() {
        report.status = "fail".to_owned();
    }
    report
}

pub fn write_ttl_semantic_outputs(
    root: &Path,
    report_path: &Path,
    markdown_path: &Path,
) -> Result<TtlSemanticReport, std::io::Error> {
    let report = build_ttl_semantic_report_for_outputs(root, report_path, markdown_path);
    write_json(report_path, &report)?;
    write_markdown(markdown_path, &ttl_semantic_markdown(&report))?;
    Ok(report)
}

#[must_use]
pub fn ttl_semantic_summary_json(report: &TtlSemanticReport) -> JsonValue {
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
                "proof_kind": case.proof_kind,
                "compatibility_mode": case.compatibility_mode,
                "oracle_fixture_id": case.oracle_fixture_id,
                "transform_stack": case.transform_stack,
                "expected_decision": case.expected_decision,
                "verifier_decision": case.verifier_decision,
                "rejection_reason": case.rejection_reason,
                "input_fingerprint": case.input_fingerprint,
                "expected_input_fingerprint": case.expected_input_fingerprint,
                "stack_hash_hex": case.stack_hash_hex,
                "output_fingerprint": case.output_fingerprint,
                "output_shapes": case.output_shapes,
                "output_dtypes": case.output_dtypes,
            })
        }).collect::<Vec<_>>(),
    })
}

#[must_use]
pub fn ttl_semantic_markdown(report: &TtlSemanticReport) -> String {
    let mut out = String::new();
    out.push_str("# TTL Semantic Proof Matrix Gate\n\n");
    out.push_str(&format!(
        "- schema: `{}`\n- bead: `{}`\n- status: `{}`\n- accepted rows: `{}`\n- rejected rows: `{}`\n\n",
        report.schema_version,
        report.bead_id,
        report.status,
        report.coverage.accepted_count,
        report.coverage.rejected_count
    ));
    out.push_str("| Case | Status | Decision | Stack | Oracle | Rejection |\n");
    out.push_str("|---|---:|---|---|---|---|\n");
    for case in &report.cases {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
            case.case_id,
            case.status,
            case.verifier_decision,
            case.transform_stack.join(">"),
            case.oracle_fixture_id.as_deref().unwrap_or("n/a"),
            case.rejection_reason.as_deref().unwrap_or("n/a")
        ));
    }
    out.push_str("\n## Issues\n\n");
    if report.issues.is_empty() {
        out.push_str("No TTL semantic proof matrix issues found.\n");
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
pub fn validate_ttl_semantic_report(report: &TtlSemanticReport) -> Vec<TtlSemanticIssue> {
    let mut issues = Vec::new();
    if report.schema_version != TTL_SEMANTIC_REPORT_SCHEMA_VERSION {
        issues.push(TtlSemanticIssue::new(
            "unsupported_schema_version",
            "$.schema_version",
            format!(
                "expected {}, got {}",
                TTL_SEMANTIC_REPORT_SCHEMA_VERSION, report.schema_version
            ),
        ));
    }
    if report.bead_id != TTL_SEMANTIC_BEAD_ID {
        issues.push(TtlSemanticIssue::new(
            "wrong_bead_id",
            "$.bead_id",
            "TTL semantic proof matrix must remain bound to frankenjax-cstq.3",
        ));
    }

    let mut seen = BTreeMap::<&str, usize>::new();
    for (idx, case) in report.cases.iter().enumerate() {
        *seen.entry(case.case_id.as_str()).or_default() += 1;
        let path = format!("$.cases[{idx}]");
        validate_case(case, &path, &mut issues);
    }

    for required in REQUIRED_CASES {
        match seen.get(required).copied() {
            Some(1) => {}
            Some(count) => issues.push(TtlSemanticIssue::new(
                "duplicate_required_case",
                "$.cases",
                format!("required case `{required}` appears {count} times"),
            )),
            None => issues.push(TtlSemanticIssue::new(
                "missing_required_case",
                "$.cases",
                format!("required case `{required}` is absent"),
            )),
        }
    }

    assert_order_sensitive_hashes(report, &mut issues);

    let pass_count = report
        .cases
        .iter()
        .filter(|case| case.status == "pass")
        .count();
    if report.coverage.pass_count != pass_count {
        issues.push(TtlSemanticIssue::new(
            "pass_count_mismatch",
            "$.coverage.pass_count",
            format!("expected {pass_count}, got {}", report.coverage.pass_count),
        ));
    }
    if report.coverage.required_case_count != REQUIRED_CASES.len() {
        issues.push(TtlSemanticIssue::new(
            "required_case_count_mismatch",
            "$.coverage.required_case_count",
            format!("expected {}", REQUIRED_CASES.len()),
        ));
    }
    if report.coverage.observed_case_count != report.cases.len() {
        issues.push(TtlSemanticIssue::new(
            "observed_case_count_mismatch",
            "$.coverage.observed_case_count",
            format!("expected {}", report.cases.len()),
        ));
    }

    issues
}

fn validate_case(case: &TtlSemanticProofCase, path: &str, issues: &mut Vec<TtlSemanticIssue>) {
    if case.status != "pass" {
        issues.push(TtlSemanticIssue::new(
            "case_failed",
            format!("{path}.status"),
            format!("case `{}` did not pass", case.case_id),
        ));
    }
    if case.compatibility_mode != "strict" && case.compatibility_mode != "hardened" {
        issues.push(TtlSemanticIssue::new(
            "invalid_compatibility_mode",
            format!("{path}.compatibility_mode"),
            "compatibility_mode must be strict or hardened",
        ));
    }
    if case.transform_stack.is_empty() {
        issues.push(TtlSemanticIssue::new(
            "missing_transform_stack",
            format!("{path}.transform_stack"),
            "proof rows must bind at least one transform",
        ));
    }
    if case.expected_decision != "accept" && case.expected_decision != "reject" {
        issues.push(TtlSemanticIssue::new(
            "invalid_expected_decision",
            format!("{path}.expected_decision"),
            "expected_decision must be accept or reject",
        ));
    }
    if case.verifier_decision != case.expected_decision {
        issues.push(TtlSemanticIssue::new(
            "verifier_decision_mismatch",
            format!("{path}.verifier_decision"),
            format!(
                "expected {}, got {}",
                case.expected_decision, case.verifier_decision
            ),
        ));
    }
    if case.evidence_refs.is_empty() {
        issues.push(TtlSemanticIssue::new(
            "missing_evidence_refs",
            format!("{path}.evidence_refs"),
            "every row must link to executable evidence",
        ));
    }
    if case.replay_command.trim().is_empty() {
        issues.push(TtlSemanticIssue::new(
            "missing_replay_command",
            format!("{path}.replay_command"),
            "every row must include a replay command",
        ));
    }
    if case.input_fingerprint != case.expected_input_fingerprint
        && case.verifier_decision == "accept"
    {
        issues.push(TtlSemanticIssue::new(
            "accepted_stale_input_fingerprint",
            format!("{path}.expected_input_fingerprint"),
            "accepted proof row has a stale canonical input fingerprint",
        ));
    }
    if case.fixture_link_required
        && case
            .oracle_fixture_id
            .as_deref()
            .is_none_or(|id| id.trim().is_empty())
        && case.verifier_decision == "accept"
    {
        issues.push(TtlSemanticIssue::new(
            "accepted_missing_fixture_link",
            format!("{path}.oracle_fixture_id"),
            "accepted fixture-bound proof row is missing its oracle fixture id",
        ));
    }

    match case.verifier_decision.as_str() {
        "accept" => validate_accepted_case(case, path, issues),
        "reject" => validate_rejected_case(case, path, issues),
        _ => issues.push(TtlSemanticIssue::new(
            "invalid_verifier_decision",
            format!("{path}.verifier_decision"),
            "verifier_decision must be accept or reject",
        )),
    }
}

fn validate_accepted_case(
    case: &TtlSemanticProofCase,
    path: &str,
    issues: &mut Vec<TtlSemanticIssue>,
) {
    if case.rejection_reason.is_some() {
        issues.push(TtlSemanticIssue::new(
            "accepted_case_has_rejection_reason",
            format!("{path}.rejection_reason"),
            "accepted proof rows must not record a rejection reason",
        ));
    }
    if case.stack_signature.as_deref().is_none_or(str::is_empty) {
        issues.push(TtlSemanticIssue::new(
            "missing_stack_signature",
            format!("{path}.stack_signature"),
            "accepted proof rows must include the TTL stack signature",
        ));
    }
    if case.stack_hash_hex.as_deref().is_none_or(str::is_empty) {
        issues.push(TtlSemanticIssue::new(
            "missing_stack_hash",
            format!("{path}.stack_hash_hex"),
            "accepted proof rows must include the TTL stack hash",
        ));
    }
    if case.output_fingerprint != case.expected_output_fingerprint {
        issues.push(TtlSemanticIssue::new(
            "output_fingerprint_mismatch",
            format!("{path}.output_fingerprint"),
            "accepted proof output fingerprint must match expected structural output",
        ));
    }
    if case.output_shapes.is_empty() || case.output_dtypes.is_empty() {
        issues.push(TtlSemanticIssue::new(
            "missing_output_metadata",
            path,
            "accepted proof rows must record output shape and dtype propagation",
        ));
    }
    for required_check in [
        "canonical_input_fingerprint_matched",
        "transform_stack_bound",
        "evidence_ids_bound",
        "shape_dtype_matched",
    ] {
        if !case
            .structural_checks
            .iter()
            .any(|check| check == required_check)
        {
            issues.push(TtlSemanticIssue::new(
                "missing_structural_check",
                format!("{path}.structural_checks"),
                format!("accepted proof row is missing `{required_check}`"),
            ));
        }
    }
}

fn validate_rejected_case(
    case: &TtlSemanticProofCase,
    path: &str,
    issues: &mut Vec<TtlSemanticIssue>,
) {
    if case.rejection_reason.as_deref().is_none_or(str::is_empty) {
        issues.push(TtlSemanticIssue::new(
            "missing_rejection_reason",
            format!("{path}.rejection_reason"),
            "rejected proof rows must record a deterministic rejection reason",
        ));
    }
    if case.case_id == "invalid_duplicate_evidence"
        && !case
            .rejection_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("duplicate_evidence"))
    {
        issues.push(TtlSemanticIssue::new(
            "duplicate_evidence_not_rejected",
            format!("{path}.rejection_reason"),
            "duplicate evidence row must reject with duplicate_evidence",
        ));
    }
    if case.case_id == "invalid_missing_evidence"
        && !case
            .rejection_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("missing_evidence"))
    {
        issues.push(TtlSemanticIssue::new(
            "missing_evidence_not_rejected",
            format!("{path}.rejection_reason"),
            "missing evidence row must reject with missing_evidence",
        ));
    }
    if case.case_id == "invalid_wrong_transform_binding"
        && !case
            .rejection_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("wrong_transform_binding"))
    {
        issues.push(TtlSemanticIssue::new(
            "wrong_transform_binding_not_rejected",
            format!("{path}.rejection_reason"),
            "wrong-transform evidence row must reject with wrong_transform_binding",
        ));
    }
    if case.case_id == "invalid_stale_input_fingerprint"
        && !case
            .rejection_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("stale_input_fingerprint"))
    {
        issues.push(TtlSemanticIssue::new(
            "stale_fingerprint_not_rejected",
            format!("{path}.rejection_reason"),
            "stale input fingerprint row must reject with stale_input_fingerprint",
        ));
    }
    if case.case_id == "invalid_missing_fixture_link"
        && !case
            .rejection_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("missing_oracle_fixture_link"))
    {
        issues.push(TtlSemanticIssue::new(
            "missing_fixture_link_not_rejected",
            format!("{path}.rejection_reason"),
            "missing fixture link row must reject with missing_oracle_fixture_link",
        ));
    }
}

fn assert_order_sensitive_hashes(report: &TtlSemanticReport, issues: &mut Vec<TtlSemanticIssue>) {
    let jit_grad = report
        .cases
        .iter()
        .find(|case| case.case_id == "valid_jit_grad_fixture");
    let grad_jit = report
        .cases
        .iter()
        .find(|case| case.case_id == "valid_grad_jit_order_sensitive");
    if let (Some(jit_grad), Some(grad_jit)) = (jit_grad, grad_jit)
        && jit_grad.stack_hash_hex == grad_jit.stack_hash_hex
    {
        issues.push(TtlSemanticIssue::new(
            "transform_order_hash_not_sensitive",
            "$.cases",
            "jit(grad) and grad(jit) must have distinct stack hashes",
        ));
    }
}

fn build_cases() -> Vec<TtlSemanticProofCase> {
    vec![
        accepted_case(
            "valid_single_jit_square",
            "x^2",
            build_program(ProgramSpec::Square),
            vec![Value::scalar_f64(4.0)],
            vec![Value::scalar_f64(16.0)],
            &[Transform::Jit],
            BTreeMap::new(),
            Some("analytic:jax.jit(lambda x: x*x)(4.0)"),
            false,
            &["crates/fj-dispatch/src/lib.rs::dispatch"],
        ),
        accepted_case(
            "valid_single_grad_square",
            "x^2",
            build_program(ProgramSpec::Square),
            vec![Value::scalar_f64(4.0)],
            vec![Value::scalar_f64(8.0)],
            &[Transform::Grad],
            BTreeMap::new(),
            Some("analytic:jax.grad(lambda x: x*x)(4.0)"),
            false,
            &["crates/fj-core/src/lib.rs::verify_transform_composition"],
        ),
        match (vector_f64(&[1.0, 2.0, 3.0]), vector_f64(&[1.0, 4.0, 9.0])) {
            (Ok(input), Ok(expected)) => accepted_case(
                "valid_single_vmap_square",
                "x^2",
                build_program(ProgramSpec::Square),
                vec![input],
                vec![expected],
                &[Transform::Vmap],
                BTreeMap::new(),
                Some("vmap_square"),
                true,
                &["crates/fj-conformance/fixtures/composition_oracle.v1.json::vmap_square"],
            ),
            (Err(err), _) | (_, Err(err)) => {
                internal_build_error_case("valid_single_vmap_square", "x^2", err)
            }
        },
        accepted_case(
            "valid_jit_grad_fixture",
            "x^2+3x",
            poly_x2_plus_3x_jaxpr(),
            vec![Value::scalar_f64(5.0)],
            vec![Value::scalar_f64(13.0)],
            &[Transform::Jit, Transform::Grad],
            BTreeMap::new(),
            Some("jit_grad_poly_x5.0"),
            true,
            &["crates/fj-conformance/fixtures/composition_oracle.v1.json::jit_grad_poly_x5.0"],
        ),
        accepted_case(
            "valid_grad_jit_order_sensitive",
            "x^2+3x",
            poly_x2_plus_3x_jaxpr(),
            vec![Value::scalar_f64(5.0)],
            vec![Value::scalar_f64(13.0)],
            &[Transform::Grad, Transform::Jit],
            BTreeMap::new(),
            Some("analytic:jax.grad(jax.jit(lambda x: x*x+3*x))(5.0)"),
            false,
            &["crates/fj-dispatch/src/lib.rs::execute_grad"],
        ),
        match (
            vector_f64(&[0.0, 1.0, 2.0, 3.0]),
            vector_f64(&[
                1.0,
                0.540_302_305_868_139_8,
                -0.416_146_836_547_142_4,
                -0.989_992_496_600_445_4,
            ]),
        ) {
            (Ok(input), Ok(expected)) => accepted_case(
                "valid_vmap_grad_fixture",
                "sin(x)",
                sin_jaxpr(),
                vec![input],
                vec![expected],
                &[Transform::Vmap, Transform::Grad],
                BTreeMap::from([("vmap_in_axes".to_owned(), "0".to_owned())]),
                Some("vmap_grad_sin"),
                true,
                &["crates/fj-conformance/fixtures/composition_oracle.v1.json::vmap_grad_sin"],
            ),
            (Err(err), _) | (_, Err(err)) => {
                internal_build_error_case("valid_vmap_grad_fixture", "sin(x)", err)
            }
        },
        fail_closed_grad_vmap_vector_output(),
        invalid_duplicate_evidence(),
        invalid_missing_evidence(),
        invalid_stale_input_fingerprint(),
        invalid_wrong_transform_binding(),
        invalid_missing_fixture_link(),
    ]
}

#[allow(clippy::too_many_arguments)]
fn accepted_case(
    case_id: &'static str,
    program: &'static str,
    jaxpr: Jaxpr,
    args: Vec<Value>,
    expected_outputs: Vec<Value>,
    transforms: &[Transform],
    compile_options: BTreeMap<String, String>,
    oracle_fixture_id: Option<&'static str>,
    fixture_link_required: bool,
    evidence_refs: &[&'static str],
) -> TtlSemanticProofCase {
    let expected_input_fingerprint = jaxpr.canonical_fingerprint().to_owned();
    let mut ledger = TraceTransformLedger::new(jaxpr.clone());
    for (index, transform) in transforms.iter().copied().enumerate() {
        ledger.push_transform(
            transform,
            format!("ttl-semantic-{case_id}-{}-{index}", transform.as_str()),
        );
    }
    let proof = verify_transform_composition(&ledger);
    let inputs = values_json(&args);
    let expected_outputs_json = values_json(&expected_outputs);
    let expected_output_fingerprint = Some(values_fingerprint(&expected_outputs));
    let mut case = base_case(
        case_id,
        "semantic_replay",
        program,
        oracle_fixture_id,
        fixture_link_required,
        &ledger,
        inputs,
        expected_outputs_json,
        evidence_refs,
    );
    case.expected_decision = "accept".to_owned();
    case.expected_input_fingerprint = expected_input_fingerprint;
    case.expected_output_fingerprint = expected_output_fingerprint;

    match proof {
        Ok(proof) => {
            case.stack_signature = Some(proof.stack_signature);
            case.stack_hash_hex = Some(proof.stack_hash_hex);
            case.structural_checks
                .push("transform_stack_bound".to_owned());
            case.structural_checks.push("evidence_ids_bound".to_owned());
            if case.input_fingerprint == case.expected_input_fingerprint {
                case.structural_checks
                    .push("canonical_input_fingerprint_matched".to_owned());
            }
            match dispatch(DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger,
                args,
                backend: "cpu".to_owned(),
                compile_options,
                custom_hook: None,
                unknown_incompatible_features: Vec::new(),
            }) {
                Ok(response) => {
                    case.actual_outputs = values_json(&response.outputs);
                    case.output_fingerprint = Some(values_fingerprint(&response.outputs));
                    case.output_shapes = response.outputs.iter().map(value_shape).collect();
                    case.output_dtypes = response.outputs.iter().map(value_dtype).collect();
                    if values_close(&response.outputs, &expected_outputs, 1e-8)
                        && output_metadata_equal(&response.outputs, &expected_outputs)
                    {
                        case.verifier_decision = "accept".to_owned();
                        case.structural_checks
                            .push("shape_dtype_matched".to_owned());
                        if fixture_link_required {
                            case.structural_checks
                                .push("oracle_fixture_linked".to_owned());
                        }
                    } else {
                        case.verifier_decision = "reject".to_owned();
                        case.rejection_reason =
                            Some("semantic.output_structure_or_value_mismatch".to_owned());
                    }
                }
                Err(err) => {
                    case.verifier_decision = "reject".to_owned();
                    case.rejection_reason = Some(classify_dispatch_error(&err));
                }
            }
        }
        Err(err) => {
            case.verifier_decision = "reject".to_owned();
            case.rejection_reason = Some(classify_transform_error(&err));
        }
    }
    case.status = row_status(&case);
    case
}

fn fail_closed_grad_vmap_vector_output() -> TtlSemanticProofCase {
    let jaxpr = build_program(ProgramSpec::Square);
    let args = match vector_f64(&[1.0, 2.0, 3.0]) {
        Ok(input) => vec![input],
        Err(err) => {
            return internal_build_error_case("fail_closed_grad_vmap_vector_output", "x^2", err);
        }
    };
    let transforms = [Transform::Grad, Transform::Vmap];
    let expected_outputs = Vec::new();
    let mut case = accepted_case(
        "fail_closed_grad_vmap_vector_output",
        "x^2",
        jaxpr,
        args,
        expected_outputs,
        &transforms,
        BTreeMap::new(),
        Some("analytic:strict-fail-closed:jax.grad(jax.vmap(lambda x: x*x))"),
        false,
        &["crates/fj-dispatch/src/lib.rs::execute_grad"],
    );
    case.proof_kind = "valid_proof_fail_closed_semantics".to_owned();
    case.expected_decision = "reject".to_owned();
    case.status = row_status(&case);
    case
}

fn vector_f64(values: &[f64]) -> Result<Value, String> {
    Value::vector_f64(values).map_err(|err| format!("Value::vector_f64 failed: {err}"))
}

fn internal_build_error_case(
    case_id: &'static str,
    program: &'static str,
    detail: String,
) -> TtlSemanticProofCase {
    let ledger = TraceTransformLedger::new(build_program(ProgramSpec::Square));
    let mut case = base_case(
        case_id,
        "internal_fixture_build_error",
        program,
        None,
        false,
        &ledger,
        json!([]),
        json!([]),
        &["crates/fj-conformance/src/ttl_semantic.rs::build_cases"],
    );
    case.expected_decision = "accept".to_owned();
    case.verifier_decision = "reject".to_owned();
    case.rejection_reason = Some(format!("internal.fixture_value_build:{detail}"));
    case.status = row_status(&case);
    case
}

fn invalid_duplicate_evidence() -> TtlSemanticProofCase {
    let mut ledger = TraceTransformLedger::new(build_program(ProgramSpec::Square));
    ledger.push_transform(Transform::Grad, "grad-duplicate");
    ledger.push_transform(Transform::Grad, "grad-duplicate");
    rejected_core_case(
        "invalid_duplicate_evidence",
        "duplicate_evidence_rejection",
        "x^2",
        ledger,
        &["crates/fj-core/src/lib.rs::verify_transform_composition"],
    )
}

fn invalid_missing_evidence() -> TtlSemanticProofCase {
    let ledger = TraceTransformLedger {
        root_jaxpr: build_program(ProgramSpec::Square),
        transform_stack: vec![Transform::Jit],
        transform_evidence: Vec::new(),
    };
    rejected_core_case(
        "invalid_missing_evidence",
        "missing_evidence_rejection",
        "x^2",
        ledger,
        &["crates/fj-core/src/lib.rs::verify_transform_composition"],
    )
}

fn invalid_wrong_transform_binding() -> TtlSemanticProofCase {
    let mut ledger = TraceTransformLedger::new(build_program(ProgramSpec::Square));
    ledger.push_transform(Transform::Jit, "ttl-semantic-grad-0");
    rejected_core_case(
        "invalid_wrong_transform_binding",
        "wrong_transform_binding_rejection",
        "x^2",
        ledger,
        &["crates/fj-core/src/lib.rs::verify_transform_composition"],
    )
}

fn rejected_core_case(
    case_id: &'static str,
    proof_kind: &'static str,
    program: &'static str,
    ledger: TraceTransformLedger,
    evidence_refs: &[&'static str],
) -> TtlSemanticProofCase {
    let inputs = json!([]);
    let expected_outputs = json!([]);
    let mut case = base_case(
        case_id,
        proof_kind,
        program,
        None,
        false,
        &ledger,
        inputs,
        expected_outputs,
        evidence_refs,
    );
    case.expected_decision = "reject".to_owned();
    match verify_transform_composition(&ledger) {
        Ok(proof) => {
            case.stack_signature = Some(proof.stack_signature);
            case.stack_hash_hex = Some(proof.stack_hash_hex);
            case.verifier_decision = "accept".to_owned();
        }
        Err(err) => {
            case.verifier_decision = "reject".to_owned();
            case.rejection_reason = Some(classify_transform_error(&err));
        }
    }
    case.status = row_status(&case);
    case
}

fn invalid_stale_input_fingerprint() -> TtlSemanticProofCase {
    let mut case = accepted_case(
        "invalid_stale_input_fingerprint",
        "x^2",
        build_program(ProgramSpec::Square),
        vec![Value::scalar_f64(4.0)],
        vec![Value::scalar_f64(16.0)],
        &[Transform::Jit],
        BTreeMap::new(),
        Some("analytic:stale-input-fingerprint-regression"),
        false,
        &["crates/fj-core/src/lib.rs::Jaxpr::canonical_fingerprint"],
    );
    case.proof_kind = "stale_input_fingerprint_rejection".to_owned();
    case.expected_input_fingerprint = build_program(ProgramSpec::Add2)
        .canonical_fingerprint()
        .to_owned();
    case.expected_decision = "reject".to_owned();
    case.verifier_decision = "reject".to_owned();
    case.rejection_reason = Some("semantic.stale_input_fingerprint".to_owned());
    case.structural_checks
        .retain(|check| check != "canonical_input_fingerprint_matched");
    case.status = row_status(&case);
    case
}

fn invalid_missing_fixture_link() -> TtlSemanticProofCase {
    let mut case = accepted_case(
        "invalid_missing_fixture_link",
        "x^2+3x",
        poly_x2_plus_3x_jaxpr(),
        vec![Value::scalar_f64(5.0)],
        vec![Value::scalar_f64(13.0)],
        &[Transform::Jit, Transform::Grad],
        BTreeMap::new(),
        None,
        true,
        &["crates/fj-conformance/fixtures/composition_oracle.v1.json::jit_grad_poly_x5.0"],
    );
    case.proof_kind = "missing_fixture_link_rejection".to_owned();
    case.expected_decision = "reject".to_owned();
    case.verifier_decision = "reject".to_owned();
    case.rejection_reason = Some("semantic.missing_oracle_fixture_link".to_owned());
    case.structural_checks
        .retain(|check| check != "oracle_fixture_linked");
    case.status = row_status(&case);
    case
}

#[allow(clippy::too_many_arguments)]
fn base_case(
    case_id: &'static str,
    proof_kind: &'static str,
    program: &'static str,
    oracle_fixture_id: Option<&'static str>,
    fixture_link_required: bool,
    ledger: &TraceTransformLedger,
    inputs: JsonValue,
    expected_outputs: JsonValue,
    evidence_refs: &[&'static str],
) -> TtlSemanticProofCase {
    TtlSemanticProofCase {
        case_id: case_id.to_owned(),
        status: "fail".to_owned(),
        proof_kind: proof_kind.to_owned(),
        compatibility_mode: "strict".to_owned(),
        program: program.to_owned(),
        oracle_fixture_id: oracle_fixture_id.map(str::to_owned),
        fixture_link_required,
        transform_stack: ledger
            .transform_stack
            .iter()
            .map(|transform| transform.as_str().to_owned())
            .collect(),
        evidence_ids: ledger.transform_evidence.clone(),
        transform_count: ledger.transform_stack.len(),
        evidence_count: ledger.transform_evidence.len(),
        input_fingerprint: ledger.root_jaxpr.canonical_fingerprint().to_owned(),
        expected_input_fingerprint: ledger.root_jaxpr.canonical_fingerprint().to_owned(),
        stack_signature: None,
        stack_hash_hex: None,
        output_fingerprint: None,
        expected_output_fingerprint: None,
        output_shapes: Vec::new(),
        output_dtypes: Vec::new(),
        inputs,
        expected_outputs,
        actual_outputs: JsonValue::Null,
        expected_decision: "accept".to_owned(),
        verifier_decision: "reject".to_owned(),
        rejection_reason: None,
        structural_checks: Vec::new(),
        evidence_refs: evidence_refs
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
        replay_command: format!("./scripts/run_ttl_semantic_gate.sh --case {case_id} --enforce"),
    }
}

fn row_status(case: &TtlSemanticProofCase) -> String {
    let decision_matches = case.expected_decision == case.verifier_decision;
    let rejection_shape_ok = case.verifier_decision != "reject" || case.rejection_reason.is_some();
    let acceptance_shape_ok = case.verifier_decision != "accept"
        || (case.rejection_reason.is_none()
            && case.output_fingerprint == case.expected_output_fingerprint
            && !case.output_shapes.is_empty()
            && !case.output_dtypes.is_empty());
    if decision_matches && rejection_shape_ok && acceptance_shape_ok {
        "pass".to_owned()
    } else {
        "fail".to_owned()
    }
}

fn classify_transform_error(err: &TransformCompositionError) -> String {
    match err {
        TransformCompositionError::EvidenceCountMismatch {
            transform_count,
            evidence_count,
        } if evidence_count < transform_count => "transform_invariant.missing_evidence".to_owned(),
        TransformCompositionError::EvidenceCountMismatch { .. } => {
            "transform_invariant.evidence_count_mismatch".to_owned()
        }
        TransformCompositionError::EmptyEvidence { .. } => {
            "transform_invariant.missing_evidence".to_owned()
        }
        TransformCompositionError::EvidenceTransformMismatch { .. } => {
            "transform_invariant.wrong_transform_binding".to_owned()
        }
        TransformCompositionError::DuplicateEvidence { .. } => {
            "transform_invariant.duplicate_evidence".to_owned()
        }
    }
}

fn classify_dispatch_error(err: &DispatchError) -> String {
    match err {
        DispatchError::TransformInvariant(transform_err) => classify_transform_error(transform_err),
        DispatchError::TransformExecution(transform_err) => {
            classify_transform_execution_error(transform_err)
        }
        DispatchError::Cache(_) => "dispatch.cache_error".to_owned(),
        DispatchError::Interpreter(_) => "dispatch.interpreter_error".to_owned(),
        DispatchError::BackendExecution(_) => "dispatch.backend_execution_error".to_owned(),
    }
}

fn classify_transform_execution_error(err: &TransformExecutionError) -> String {
    match err {
        TransformExecutionError::EmptyArgumentList { .. } => {
            "transform_execution.empty_argument_list"
        }
        TransformExecutionError::NonScalarGradientInput => {
            "transform_execution.non_scalar_gradient_input"
        }
        TransformExecutionError::NonScalarGradientOutput => {
            "transform_execution.non_scalar_gradient_output"
        }
        TransformExecutionError::VmapRequiresRankOneLeadingArgument => {
            "transform_execution.vmap_requires_rank_one_leading_argument"
        }
        TransformExecutionError::VmapMismatchedLeadingDimension { .. } => {
            "transform_execution.vmap_mismatched_leading_dimension"
        }
        TransformExecutionError::VmapInconsistentOutputArity { .. } => {
            "transform_execution.vmap_inconsistent_output_arity"
        }
        TransformExecutionError::VmapAxesOutOfBounds { .. } => {
            "transform_execution.vmap_axes_out_of_bounds"
        }
        TransformExecutionError::VmapAxesCountMismatch { .. } => {
            "transform_execution.vmap_axes_count_mismatch"
        }
        TransformExecutionError::InvalidVmapAxisSpec { .. } => {
            "transform_execution.invalid_vmap_axis_spec"
        }
        TransformExecutionError::VmapUnmappedOutputMismatch => {
            "transform_execution.vmap_unmapped_output_mismatch"
        }
        TransformExecutionError::FiniteDiffGradFallbackDisabled { .. } => {
            "transform_execution.finite_diff_grad_fallback_disabled"
        }
        TransformExecutionError::EmptyVmapOutput => "transform_execution.empty_vmap_output",
        TransformExecutionError::TensorBuild(_) => "transform_execution.tensor_build",
    }
    .to_owned()
}

fn poly_x2_plus_3x_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(4)],
        vec![
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Lit(Literal::from_f64(3.0)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
            Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(3))],
                outputs: smallvec![VarId(4)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            },
        ],
    )
}

fn sin_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive: Primitive::Sin,
            inputs: smallvec![Atom::Var(VarId(1))],
            outputs: smallvec![VarId(2)],
            params: BTreeMap::new(),
            effects: Vec::new(),
            sub_jaxprs: Vec::new(),
        }],
    )
}

fn values_json(values: &[Value]) -> JsonValue {
    JsonValue::Array(values.iter().map(value_json).collect())
}

fn value_json(value: &Value) -> JsonValue {
    match value {
        Value::Scalar(literal) => json!({
            "kind": "scalar",
            "dtype": dtype_name(value.dtype()),
            "shape": [],
            "value": literal_json(*literal),
        }),
        Value::Tensor(tensor) => json!({
            "kind": "tensor",
            "dtype": dtype_name(tensor.dtype),
            "shape": tensor.shape.dims,
            "values": tensor.elements.iter().copied().map(literal_json).collect::<Vec<_>>(),
        }),
    }
}

fn literal_json(literal: Literal) -> JsonValue {
    match literal {
        Literal::I64(value) => json!(value),
        Literal::U32(value) => json!(value),
        Literal::U64(value) => json!(value),
        Literal::Bool(value) => json!(value),
        Literal::BF16Bits(bits) => json!(format!("bf16:0x{bits:04x}")),
        Literal::F16Bits(bits) => json!(format!("f16:0x{bits:04x}")),
        Literal::F32Bits(bits) => json!(f32::from_bits(bits)),
        Literal::F64Bits(bits) => json!(f64::from_bits(bits)),
        Literal::Complex64Bits(re, im) => json!({
            "re": f32::from_bits(re),
            "im": f32::from_bits(im),
        }),
        Literal::Complex128Bits(re, im) => json!({
            "re": f64::from_bits(re),
            "im": f64::from_bits(im),
        }),
    }
}

fn values_fingerprint(values: &[Value]) -> String {
    json_fingerprint(&values_json(values))
}

fn json_fingerprint(value: &JsonValue) -> String {
    let bytes = match serde_json::to_vec(value) {
        Ok(bytes) => bytes,
        Err(err) => format!("semantic-json-serialization-error:{err}").into_bytes(),
    };
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn value_shape(value: &Value) -> Vec<u32> {
    match value {
        Value::Scalar(_) => Vec::new(),
        Value::Tensor(tensor) => tensor.shape.dims.clone(),
    }
}

fn value_dtype(value: &Value) -> String {
    dtype_name(value.dtype()).to_owned()
}

fn dtype_name(dtype: DType) -> &'static str {
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

fn values_close(actual: &[Value], expected: &[Value], tolerance: f64) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected.iter())
            .all(|(actual, expected)| value_close(actual, expected, tolerance))
}

fn value_close(actual: &Value, expected: &Value, tolerance: f64) -> bool {
    if actual.dtype() != expected.dtype() || value_shape(actual) != value_shape(expected) {
        return false;
    }
    match (actual, expected) {
        (Value::Scalar(lhs), Value::Scalar(rhs)) => literal_close(*lhs, *rhs, tolerance),
        (Value::Tensor(lhs), Value::Tensor(rhs)) => {
            lhs.elements.len() == rhs.elements.len()
                && lhs
                    .elements
                    .iter()
                    .zip(rhs.elements.iter())
                    .all(|(lhs, rhs)| literal_close(*lhs, *rhs, tolerance))
        }
        _ => false,
    }
}

fn literal_close(actual: Literal, expected: Literal, tolerance: f64) -> bool {
    match (actual.as_f64(), expected.as_f64()) {
        (Some(actual), Some(expected)) => (actual - expected).abs() <= tolerance,
        _ => actual == expected,
    }
}

fn output_metadata_equal(actual: &[Value], expected: &[Value]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected.iter())
            .all(|(actual, expected)| {
                actual.dtype() == expected.dtype() && value_shape(actual) == value_shape(expected)
            })
}

fn repo_relative_artifact(root: &Path, path: &Path) -> String {
    match path.strip_prefix(root) {
        Ok(path) => path.display().to_string(),
        Err(_) => path.display().to_string(),
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
        .map_or(0, |duration| duration.as_millis())
}
