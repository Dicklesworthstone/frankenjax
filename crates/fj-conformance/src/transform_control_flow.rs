#![forbid(unsafe_code)]

use crate::memory_performance::sample_process_memory;
use fj_core::{
    Atom, CompatibilityMode, DType, Equation, Jaxpr, Literal, Primitive, ProgramSpec,
    TraceTransformLedger, Transform, Value, VarId, build_program,
};
use fj_dispatch::{DispatchError, DispatchRequest, TransformExecutionError, dispatch};
use fj_lax::{eval_primitive, eval_scan_functional};
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};
use smallvec::smallvec;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub const TRANSFORM_CONTROL_FLOW_REPORT_SCHEMA_VERSION: &str =
    "frankenjax.transform-control-flow-matrix.v1";
pub const TRANSFORM_CONTROL_FLOW_BEAD_ID: &str = "frankenjax-cstq.2";

const REQUIRED_CASES: &[&str] = &[
    "jit_grad_cond_false",
    "vmap_grad_cond_mixed_predicates",
    "jit_vmap_grad_cond_false",
    "vmap_grad_scan_mul",
    "jit_vmap_grad_scan_mul",
    "vmap_grad_while_mul",
    "vmap_switch_batched_indices",
    "vmap_switch_scalar_index_batched_operand",
    "grad_grad_square",
    "vmap_grad_grad_square",
    "jit_vmap_grad_grad_square",
    "vmap_jit_grad_square",
    "value_and_grad_multi_output",
    "jacobian_quadratic",
    "hessian_quadratic",
    "scan_multi_carry_state",
    "vmap_multi_output_return",
    "jit_mixed_dtype_add",
    "grad_vmap_vector_output_fail_closed",
    "vmap_empty_batch_fail_closed",
    "vmap_out_axes_none_nonconstant_fail_closed",
];

const REQUIRED_PERFORMANCE_SENTINELS: &[&str] = &[
    "perf_vmap_scan_loop_stack",
    "perf_vmap_while_loop_stack",
    "perf_jit_vmap_grad_cond",
    "perf_batched_switch",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformControlFlowCase {
    pub case_id: String,
    pub status: String,
    pub support_status: String,
    pub api_surface: String,
    pub transform_stack: Vec<String>,
    pub control_flow_op: Option<String>,
    pub primitive_ops: Vec<String>,
    pub input_shapes: Vec<Vec<u32>>,
    pub input_dtypes: Vec<String>,
    pub output_shapes: Vec<Vec<u32>>,
    pub output_dtypes: Vec<String>,
    pub oracle_fixture_id: String,
    pub oracle_kind: String,
    pub tolerance_policy: String,
    pub comparison: String,
    pub expected: JsonValue,
    pub actual: JsonValue,
    pub error_class: Option<String>,
    pub error_message: Option<String>,
    pub strict_hardened_rationale: String,
    pub evidence_refs: Vec<String>,
    pub replay_command: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformControlFlowPerformanceSentinel {
    pub workload_id: String,
    pub status: String,
    pub iterations: u32,
    pub transform_stack: Vec<String>,
    pub control_flow_op: Option<String>,
    pub p50_ns: u128,
    pub p95_ns: u128,
    pub p99_ns: u128,
    pub measurement_backend: String,
    pub rss_before_bytes: Option<u64>,
    pub rss_after_bytes: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    pub delta_rss_bytes: Option<i64>,
    pub logical_output_units: u64,
    pub evidence_refs: Vec<String>,
    pub replay_command: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformControlFlowIssue {
    pub code: String,
    pub path: String,
    pub message: String,
}

impl TransformControlFlowIssue {
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransformControlFlowReport {
    pub schema_version: String,
    pub bead_id: String,
    pub generated_at_unix_ms: u128,
    pub status: String,
    pub matrix_policy: String,
    pub supported_rows: usize,
    pub fail_closed_rows: usize,
    pub cases: Vec<TransformControlFlowCase>,
    pub performance_sentinels: Vec<TransformControlFlowPerformanceSentinel>,
    pub issues: Vec<TransformControlFlowIssue>,
    pub artifact_refs: Vec<String>,
    pub replay_command: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformControlFlowOutputPaths {
    pub report: PathBuf,
    pub markdown: PathBuf,
}

impl TransformControlFlowOutputPaths {
    #[must_use]
    pub fn for_root(root: &Path) -> Self {
        let conformance = root.join("artifacts/conformance");
        Self {
            report: conformance.join("transform_control_flow_matrix.v1.json"),
            markdown: conformance.join("transform_control_flow_matrix.v1.md"),
        }
    }
}

pub fn build_transform_control_flow_report(root: &Path) -> TransformControlFlowReport {
    let paths = TransformControlFlowOutputPaths::for_root(root);
    let mut report = TransformControlFlowReport {
        schema_version: TRANSFORM_CONTROL_FLOW_REPORT_SCHEMA_VERSION.to_owned(),
        bead_id: TRANSFORM_CONTROL_FLOW_BEAD_ID.to_owned(),
        generated_at_unix_ms: now_unix_ms(),
        status: "pass".to_owned(),
        matrix_policy:
            "Strict V1 transform/control-flow matrix: supported rows execute; unsupported rows must fail closed with deterministic typed errors"
                .to_owned(),
        supported_rows: 0,
        fail_closed_rows: 0,
        cases: build_matrix_cases(),
        performance_sentinels: build_performance_sentinels(),
        issues: Vec::new(),
        artifact_refs: vec![repo_relative_artifact(&paths.report), repo_relative_artifact(&paths.markdown)],
        replay_command: "./scripts/run_transform_control_flow_gate.sh --enforce".to_owned(),
    };
    report.supported_rows = report
        .cases
        .iter()
        .filter(|case| case.support_status == "supported")
        .count();
    report.fail_closed_rows = report
        .cases
        .iter()
        .filter(|case| case.support_status == "fail_closed")
        .count();
    report.issues = validate_transform_control_flow_report(&report);
    if !report.issues.is_empty() {
        report.status = "fail".to_owned();
    }
    report
}

pub fn write_transform_control_flow_outputs(
    root: &Path,
    report_path: &Path,
    markdown_path: &Path,
) -> Result<TransformControlFlowReport, std::io::Error> {
    let report = build_transform_control_flow_report(root);
    write_json(report_path, &report)?;
    write_markdown(markdown_path, &transform_control_flow_markdown(&report))?;
    Ok(report)
}

#[must_use]
pub fn transform_control_flow_summary_json(report: &TransformControlFlowReport) -> JsonValue {
    json!({
        "status": report.status,
        "schema_version": report.schema_version,
        "case_count": report.cases.len(),
        "supported_rows": report.supported_rows,
        "fail_closed_rows": report.fail_closed_rows,
        "issue_count": report.issues.len(),
        "performance_sentinel_count": report.performance_sentinels.len(),
        "cases": report.cases.iter().map(|case| {
            json!({
                "case_id": case.case_id,
                "status": case.status,
                "support_status": case.support_status,
                "transform_stack": case.transform_stack,
                "control_flow_op": case.control_flow_op,
                "oracle_fixture_id": case.oracle_fixture_id,
                "error_class": case.error_class,
            })
        }).collect::<Vec<_>>(),
        "performance_sentinels": report.performance_sentinels.iter().map(|sentinel| {
            json!({
                "workload_id": sentinel.workload_id,
                "status": sentinel.status,
                "iterations": sentinel.iterations,
                "p50_ns": sentinel.p50_ns,
                "p95_ns": sentinel.p95_ns,
                "p99_ns": sentinel.p99_ns,
                "peak_rss_bytes": sentinel.peak_rss_bytes,
                "delta_rss_bytes": sentinel.delta_rss_bytes,
            })
        }).collect::<Vec<_>>(),
    })
}

#[must_use]
pub fn transform_control_flow_markdown(report: &TransformControlFlowReport) -> String {
    let mut out = String::new();
    out.push_str("# Transform Control-Flow Matrix Gate\n\n");
    out.push_str(&format!(
        "- schema: `{}`\n- bead: `{}`\n- status: `{}`\n- supported rows: `{}`\n- fail-closed rows: `{}`\n\n",
        report.schema_version,
        report.bead_id,
        report.status,
        report.supported_rows,
        report.fail_closed_rows
    ));
    out.push_str("| Case | Support | Status | Stack | Control flow | Oracle | Comparison |\n");
    out.push_str("|---|---|---:|---|---|---|---|\n");
    for case in &report.cases {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
            case.case_id,
            case.support_status,
            case.status,
            case.transform_stack.join(">"),
            case.control_flow_op.as_deref().unwrap_or("n/a"),
            case.oracle_fixture_id,
            case.comparison.replace('|', "/")
        ));
    }
    out.push_str("\n## Performance Sentinels\n\n");
    out.push_str(
        "| Workload | Status | Iterations | p50 ns | p95 ns | p99 ns | Peak RSS bytes |\n",
    );
    out.push_str("|---|---:|---:|---:|---:|---:|---:|\n");
    for sentinel in &report.performance_sentinels {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
            sentinel.workload_id,
            sentinel.status,
            sentinel.iterations,
            sentinel.p50_ns,
            sentinel.p95_ns,
            sentinel.p99_ns,
            sentinel
                .peak_rss_bytes
                .map_or_else(|| "n/a".to_owned(), |value| value.to_string())
        ));
    }
    out.push_str("\n## Issues\n\n");
    if report.issues.is_empty() {
        out.push_str("No transform control-flow matrix issues found.\n");
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
pub fn validate_transform_control_flow_report(
    report: &TransformControlFlowReport,
) -> Vec<TransformControlFlowIssue> {
    let mut issues = Vec::new();
    if report.schema_version != TRANSFORM_CONTROL_FLOW_REPORT_SCHEMA_VERSION {
        issues.push(TransformControlFlowIssue::new(
            "unsupported_schema_version",
            "$.schema_version",
            format!(
                "expected {}, got {}",
                TRANSFORM_CONTROL_FLOW_REPORT_SCHEMA_VERSION, report.schema_version
            ),
        ));
    }
    if report.bead_id != TRANSFORM_CONTROL_FLOW_BEAD_ID {
        issues.push(TransformControlFlowIssue::new(
            "wrong_bead_id",
            "$.bead_id",
            "transform control-flow matrix must remain bound to frankenjax-cstq.2",
        ));
    }

    let mut seen = BTreeMap::<&str, usize>::new();
    for (idx, case) in report.cases.iter().enumerate() {
        *seen.entry(case.case_id.as_str()).or_default() += 1;
        let path = format!("$.cases[{idx}]");
        if case.status != "pass" {
            issues.push(TransformControlFlowIssue::new(
                "case_failed",
                format!("{path}.status"),
                format!("case `{}` did not pass", case.case_id),
            ));
        }
        if case.support_status != "supported" && case.support_status != "fail_closed" {
            issues.push(TransformControlFlowIssue::new(
                "invalid_support_status",
                format!("{path}.support_status"),
                "support_status must be supported or fail_closed",
            ));
        }
        if case.oracle_fixture_id.trim().is_empty() {
            issues.push(TransformControlFlowIssue::new(
                "missing_oracle_fixture_id",
                format!("{path}.oracle_fixture_id"),
                "every row must identify an oracle fixture, analytic oracle, or explicit unsupported contract",
            ));
        }
        if case.evidence_refs.is_empty() {
            issues.push(TransformControlFlowIssue::new(
                "missing_evidence_refs",
                format!("{path}.evidence_refs"),
                "every row must link to executable evidence",
            ));
        }
        if case.replay_command.trim().is_empty() {
            issues.push(TransformControlFlowIssue::new(
                "missing_replay_command",
                format!("{path}.replay_command"),
                "every row must include a replay command",
            ));
        }
        if case.support_status == "supported" {
            if case.output_dtypes.is_empty() {
                issues.push(TransformControlFlowIssue::new(
                    "missing_output_metadata",
                    format!("{path}.output_dtypes"),
                    "supported rows must record output dtype metadata",
                ));
            }
            if case.error_class.is_some() {
                issues.push(TransformControlFlowIssue::new(
                    "supported_row_has_error",
                    format!("{path}.error_class"),
                    "supported rows must not record an error class",
                ));
            }
        } else {
            if case.error_class.as_deref().is_none_or(str::is_empty) {
                issues.push(TransformControlFlowIssue::new(
                    "missing_fail_closed_error_class",
                    format!("{path}.error_class"),
                    "unsupported rows must fail closed with a typed error class",
                ));
            }
            if case.strict_hardened_rationale.trim().is_empty() {
                issues.push(TransformControlFlowIssue::new(
                    "missing_fail_closed_rationale",
                    format!("{path}.strict_hardened_rationale"),
                    "unsupported rows must explain the strict/hardened rationale",
                ));
            }
        }
    }

    for required in REQUIRED_CASES {
        match seen.get(required).copied() {
            Some(1) => {}
            Some(count) => issues.push(TransformControlFlowIssue::new(
                "duplicate_required_case",
                "$.cases",
                format!("required case `{required}` appears {count} times"),
            )),
            None => issues.push(TransformControlFlowIssue::new(
                "missing_required_case",
                "$.cases",
                format!("required case `{required}` is absent"),
            )),
        }
    }

    let mut seen_sentinels = BTreeMap::<&str, usize>::new();
    for (idx, sentinel) in report.performance_sentinels.iter().enumerate() {
        *seen_sentinels
            .entry(sentinel.workload_id.as_str())
            .or_default() += 1;
        let path = format!("$.performance_sentinels[{idx}]");
        if sentinel.status != "pass" {
            issues.push(TransformControlFlowIssue::new(
                "performance_sentinel_failed",
                format!("{path}.status"),
                format!("sentinel `{}` did not pass", sentinel.workload_id),
            ));
        }
        if sentinel.iterations == 0 {
            issues.push(TransformControlFlowIssue::new(
                "zero_performance_iterations",
                format!("{path}.iterations"),
                "performance sentinel must execute at least once",
            ));
        }
        if sentinel.p50_ns == 0 || sentinel.p95_ns == 0 || sentinel.p99_ns == 0 {
            issues.push(TransformControlFlowIssue::new(
                "missing_latency_percentile",
                path.clone(),
                "performance sentinel must record non-zero p50/p95/p99 nanoseconds",
            ));
        }
        if sentinel.measurement_backend == "unavailable" || sentinel.measurement_backend.is_empty()
        {
            issues.push(TransformControlFlowIssue::new(
                "memory_not_measured",
                format!("{path}.measurement_backend"),
                "performance sentinel must record RSS when procfs is available",
            ));
        }
        if sentinel.logical_output_units == 0 {
            issues.push(TransformControlFlowIssue::new(
                "empty_performance_witness",
                format!("{path}.logical_output_units"),
                "performance sentinel must produce a behavior witness",
            ));
        }
        if sentinel.evidence_refs.is_empty() {
            issues.push(TransformControlFlowIssue::new(
                "missing_performance_evidence",
                format!("{path}.evidence_refs"),
                "performance sentinel must include evidence refs",
            ));
        }
    }
    for required in REQUIRED_PERFORMANCE_SENTINELS {
        match seen_sentinels.get(required).copied() {
            Some(1) => {}
            Some(count) => issues.push(TransformControlFlowIssue::new(
                "duplicate_required_performance_sentinel",
                "$.performance_sentinels",
                format!("required sentinel `{required}` appears {count} times"),
            )),
            None => issues.push(TransformControlFlowIssue::new(
                "missing_required_performance_sentinel",
                "$.performance_sentinels",
                format!("required sentinel `{required}` is absent"),
            )),
        }
    }

    let supported_rows = report
        .cases
        .iter()
        .filter(|case| case.support_status == "supported")
        .count();
    let fail_closed_rows = report
        .cases
        .iter()
        .filter(|case| case.support_status == "fail_closed")
        .count();
    if report.supported_rows != supported_rows {
        issues.push(TransformControlFlowIssue::new(
            "supported_row_count_mismatch",
            "$.supported_rows",
            format!("expected {supported_rows}, got {}", report.supported_rows),
        ));
    }
    if report.fail_closed_rows != fail_closed_rows {
        issues.push(TransformControlFlowIssue::new(
            "fail_closed_row_count_mismatch",
            "$.fail_closed_rows",
            format!(
                "expected {fail_closed_rows}, got {}",
                report.fail_closed_rows
            ),
        ));
    }

    issues
}

fn build_matrix_cases() -> Vec<TransformControlFlowCase> {
    vec![
        case_jit_grad_cond_false(),
        case_vmap_grad_cond_mixed(),
        case_jit_vmap_grad_cond_false(),
        case_vmap_grad_scan_mul(),
        case_jit_vmap_grad_scan_mul(),
        case_vmap_grad_while_mul(),
        case_vmap_switch_batched_indices(),
        case_vmap_switch_scalar_index_batched_operand(),
        case_grad_grad_square(),
        case_vmap_grad_grad_square(),
        case_jit_vmap_grad_grad_square(),
        case_vmap_jit_grad_square(),
        case_value_and_grad_multi_output(),
        case_jacobian_quadratic(),
        case_hessian_quadratic(),
        case_scan_multi_carry_state(),
        case_vmap_multi_output_return(),
        case_jit_mixed_dtype_add(),
        case_grad_vmap_vector_output_fail_closed(),
        case_vmap_empty_batch_fail_closed(),
        case_vmap_out_axes_none_nonconstant_fail_closed(),
    ]
}

fn case_jit_grad_cond_false() -> TransformControlFlowCase {
    let inputs = vec![Value::scalar_f64(5.0), Value::scalar_bool(false)];
    let transforms = [Transform::Jit, Transform::Grad];
    match dispatch_outputs(
        grad_cond_jaxpr(),
        inputs.clone(),
        &transforms,
        BTreeMap::new(),
    ) {
        Ok(outputs) => {
            let actual = scalar_f64(&outputs, 0);
            supported_case(
                CaseMetadata::new(
                    "jit_grad_cond_false",
                    "fj-dispatch",
                    &transforms,
                    Some("cond"),
                    &["mul", "cond"],
                    &inputs,
                    "analytic:jax.grad(lambda x: lax.cond(False, x, x*x))(5.0)",
                    "analytic_jax_semantics",
                    &[
                        "crates/fj-conformance/tests/control_flow_conformance.rs::test_jit_grad_cond",
                        "crates/fj-conformance/tests/control_flow_conformance.rs::e2e_control_flow_ad_oracle",
                    ],
                ),
                json!(10.0),
                json!(actual),
                &outputs,
                (actual - 10.0).abs() <= 1e-6,
                "absolute_error <= 1e-6",
            )
        }
        Err(err) => supported_error_case(
            "jit_grad_cond_false",
            "fj-dispatch",
            &transforms,
            Some("cond"),
            &["mul", "cond"],
            &inputs,
            "analytic:jax.grad(lambda x: lax.cond(False, x, x*x))(5.0)",
            err,
        ),
    }
}

fn case_vmap_grad_cond_mixed() -> TransformControlFlowCase {
    let inputs = vec![
        Value::vector_f64(&[3.0, 5.0, 7.0]).expect("vector should build"),
        Value::vector_bool(&[true, false, true]).expect("bool vector should build"),
    ];
    let transforms = [Transform::Vmap, Transform::Grad];
    let options = BTreeMap::from([("vmap_in_axes".to_owned(), "0,0".to_owned())]);
    match dispatch_outputs(grad_cond_jaxpr(), inputs.clone(), &transforms, options) {
        Ok(outputs) => {
            let actual = tensor_f64_vec(&outputs, 0);
            supported_case(
                CaseMetadata::new(
                    "vmap_grad_cond_mixed_predicates",
                    "fj-dispatch",
                    &transforms,
                    Some("cond"),
                    &["mul", "cond"],
                    &inputs,
                    "analytic:jax.vmap(jax.grad(cond_square_identity))([3,5,7],[T,F,T])",
                    "analytic_jax_semantics",
                    &[
                        "crates/fj-conformance/tests/control_flow_conformance.rs::test_vmap_grad_cond_mixed",
                    ],
                ),
                json!([1.0, 10.0, 1.0]),
                json!(actual),
                &outputs,
                f64_vec_close(&actual, &[1.0, 10.0, 1.0], 1e-6),
                "all absolute_errors <= 1e-6",
            )
        }
        Err(err) => supported_error_case(
            "vmap_grad_cond_mixed_predicates",
            "fj-dispatch",
            &transforms,
            Some("cond"),
            &["mul", "cond"],
            &inputs,
            "analytic:jax.vmap(jax.grad(cond_square_identity))([3,5,7],[T,F,T])",
            err,
        ),
    }
}

fn case_jit_vmap_grad_cond_false() -> TransformControlFlowCase {
    let inputs = vec![
        Value::vector_f64(&[2.0, 4.0]).expect("vector should build"),
        Value::vector_bool(&[false, false]).expect("bool vector should build"),
    ];
    let transforms = [Transform::Jit, Transform::Vmap, Transform::Grad];
    let options = BTreeMap::from([("vmap_in_axes".to_owned(), "0,0".to_owned())]);
    match dispatch_outputs(grad_cond_jaxpr(), inputs.clone(), &transforms, options) {
        Ok(outputs) => {
            let actual = tensor_f64_vec(&outputs, 0);
            supported_case(
                CaseMetadata::new(
                    "jit_vmap_grad_cond_false",
                    "fj-dispatch",
                    &transforms,
                    Some("cond"),
                    &["mul", "cond"],
                    &inputs,
                    "analytic:jax.jit(jax.vmap(jax.grad(cond_square_identity)))",
                    "analytic_jax_semantics",
                    &[
                        "crates/fj-conformance/tests/control_flow_conformance.rs::test_jit_vmap_grad_cond",
                    ],
                ),
                json!([4.0, 8.0]),
                json!(actual),
                &outputs,
                f64_vec_close(&actual, &[4.0, 8.0], 1e-6),
                "all absolute_errors <= 1e-6",
            )
        }
        Err(err) => supported_error_case(
            "jit_vmap_grad_cond_false",
            "fj-dispatch",
            &transforms,
            Some("cond"),
            &["mul", "cond"],
            &inputs,
            "analytic:jax.jit(jax.vmap(jax.grad(cond_square_identity)))",
            err,
        ),
    }
}

fn case_vmap_grad_scan_mul() -> TransformControlFlowCase {
    let inputs = vec![
        Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector should build"),
        Value::vector_f64(&[2.0, 5.0]).expect("xs vector should build"),
    ];
    let transforms = [Transform::Vmap, Transform::Grad];
    let options = BTreeMap::from([("vmap_in_axes".to_owned(), "0,none".to_owned())]);
    match dispatch_outputs(
        scan_jaxpr("mul", false),
        inputs.clone(),
        &transforms,
        options,
    ) {
        Ok(outputs) => {
            let actual = tensor_f64_vec(&outputs, 0);
            supported_case(
                CaseMetadata::new(
                    "vmap_grad_scan_mul",
                    "fj-dispatch",
                    &transforms,
                    Some("scan"),
                    &["scan:mul"],
                    &inputs,
                    "analytic:jax.vmap(jax.grad(lambda init: lax.scan(mul, init, [2,5])))",
                    "analytic_jax_semantics",
                    &[
                        "crates/fj-conformance/tests/control_flow_conformance.rs::test_vmap_grad_scan",
                    ],
                ),
                json!([10.0, 10.0, 10.0]),
                json!(actual),
                &outputs,
                f64_vec_close(&actual, &[10.0, 10.0, 10.0], 1e-6),
                "all absolute_errors <= 1e-6",
            )
        }
        Err(err) => supported_error_case(
            "vmap_grad_scan_mul",
            "fj-dispatch",
            &transforms,
            Some("scan"),
            &["scan:mul"],
            &inputs,
            "analytic:jax.vmap(jax.grad(lambda init: lax.scan(mul, init, [2,5])))",
            err,
        ),
    }
}

fn case_jit_vmap_grad_scan_mul() -> TransformControlFlowCase {
    let inputs = vec![
        Value::vector_f64(&[1.0, 5.0]).expect("vector should build"),
        Value::vector_f64(&[3.0, 4.0]).expect("xs vector should build"),
    ];
    let transforms = [Transform::Jit, Transform::Vmap, Transform::Grad];
    let options = BTreeMap::from([("vmap_in_axes".to_owned(), "0,none".to_owned())]);
    match dispatch_outputs(
        scan_jaxpr("mul", false),
        inputs.clone(),
        &transforms,
        options,
    ) {
        Ok(outputs) => {
            let actual = tensor_f64_vec(&outputs, 0);
            supported_case(
                CaseMetadata::new(
                    "jit_vmap_grad_scan_mul",
                    "fj-dispatch",
                    &transforms,
                    Some("scan"),
                    &["scan:mul"],
                    &inputs,
                    "analytic:jax.jit(jax.vmap(jax.grad(lambda init: lax.scan(mul, init, [3,4]))))",
                    "analytic_jax_semantics",
                    &[
                        "crates/fj-conformance/tests/control_flow_conformance.rs::test_jit_vmap_grad_scan",
                    ],
                ),
                json!([12.0, 12.0]),
                json!(actual),
                &outputs,
                f64_vec_close(&actual, &[12.0, 12.0], 1e-6),
                "all absolute_errors <= 1e-6",
            )
        }
        Err(err) => supported_error_case(
            "jit_vmap_grad_scan_mul",
            "fj-dispatch",
            &transforms,
            Some("scan"),
            &["scan:mul"],
            &inputs,
            "analytic:jax.jit(jax.vmap(jax.grad(lambda init: lax.scan(mul, init, [3,4]))))",
            err,
        ),
    }
}

fn case_vmap_grad_while_mul() -> TransformControlFlowCase {
    let inputs = vec![
        Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector should build"),
        Value::scalar_f64(2.0),
        Value::scalar_f64(8.0),
    ];
    let transforms = [Transform::Vmap, Transform::Grad];
    let options = BTreeMap::from([("vmap_in_axes".to_owned(), "0,none,none".to_owned())]);
    match dispatch_outputs(
        while_jaxpr("mul", "lt", 16),
        inputs.clone(),
        &transforms,
        options,
    ) {
        Ok(outputs) => {
            let actual = tensor_f64_vec(&outputs, 0);
            supported_case(
                CaseMetadata::new(
                    "vmap_grad_while_mul",
                    "fj-dispatch",
                    &transforms,
                    Some("while"),
                    &["while:mul:lt"],
                    &inputs,
                    "analytic:jax.vmap(jax.grad(lambda init: lax.while_loop(init * 2 < 8)))",
                    "analytic_jax_semantics",
                    &[
                        "crates/fj-conformance/tests/control_flow_conformance.rs::test_vmap_grad_while_mul",
                    ],
                ),
                json!([8.0, 4.0, 4.0]),
                json!(actual),
                &outputs,
                f64_vec_close(&actual, &[8.0, 4.0, 4.0], 1e-6),
                "all absolute_errors <= 1e-6",
            )
        }
        Err(err) => supported_error_case(
            "vmap_grad_while_mul",
            "fj-dispatch",
            &transforms,
            Some("while"),
            &["while:mul:lt"],
            &inputs,
            "analytic:jax.vmap(jax.grad(lambda init: lax.while_loop(init * 2 < 8)))",
            err,
        ),
    }
}

fn case_vmap_switch_batched_indices() -> TransformControlFlowCase {
    let inputs = vec![
        Value::vector_i64(&[0, 1, 2]).expect("branch vector should build"),
        Value::vector_i64(&[5, 6, 7]).expect("operand vector should build"),
    ];
    let transforms = [Transform::Vmap];
    match dispatch_outputs(
        switch_control_flow_jaxpr(),
        inputs.clone(),
        &transforms,
        BTreeMap::new(),
    ) {
        Ok(outputs) => {
            let actual = tensor_i64_vec(&outputs, 0);
            supported_case(
                CaseMetadata::new(
                    "vmap_switch_batched_indices",
                    "fj-dispatch",
                    &transforms,
                    Some("switch"),
                    &["switch", "add", "mul"],
                    &inputs,
                    "analytic:jax.vmap(lax.switch)([0,1,2],[5,6,7])",
                    "analytic_jax_semantics",
                    &[
                        "crates/fj-conformance/tests/control_flow_conformance.rs::test_vmap_switch_dispatch_with_batched_indices",
                    ],
                ),
                json!([5, 12, 49]),
                json!(actual),
                &outputs,
                actual == [5, 12, 49],
                "exact integer vector match",
            )
        }
        Err(err) => supported_error_case(
            "vmap_switch_batched_indices",
            "fj-dispatch",
            &transforms,
            Some("switch"),
            &["switch", "add", "mul"],
            &inputs,
            "analytic:jax.vmap(lax.switch)([0,1,2],[5,6,7])",
            err,
        ),
    }
}

fn case_vmap_switch_scalar_index_batched_operand() -> TransformControlFlowCase {
    let inputs = vec![
        Value::scalar_i64(1),
        Value::vector_i64(&[2, 3, 4]).expect("operand vector should build"),
    ];
    let transforms = [Transform::Vmap];
    let options = BTreeMap::from([("vmap_in_axes".to_owned(), "none,0".to_owned())]);
    match dispatch_outputs(
        switch_control_flow_jaxpr(),
        inputs.clone(),
        &transforms,
        options,
    ) {
        Ok(outputs) => {
            let actual = tensor_i64_vec(&outputs, 0);
            supported_case(
                CaseMetadata::new(
                    "vmap_switch_scalar_index_batched_operand",
                    "fj-dispatch",
                    &transforms,
                    Some("switch"),
                    &["switch", "add"],
                    &inputs,
                    "analytic:jax.vmap(lambda x: lax.switch(1, branches, x))([2,3,4])",
                    "analytic_jax_semantics",
                    &[
                        "crates/fj-conformance/tests/control_flow_conformance.rs::test_vmap_switch_dispatch_with_scalar_index_and_batched_operand",
                    ],
                ),
                json!([4, 6, 8]),
                json!(actual),
                &outputs,
                actual == [4, 6, 8],
                "exact integer vector match",
            )
        }
        Err(err) => supported_error_case(
            "vmap_switch_scalar_index_batched_operand",
            "fj-dispatch",
            &transforms,
            Some("switch"),
            &["switch", "add"],
            &inputs,
            "analytic:jax.vmap(lambda x: lax.switch(1, branches, x))([2,3,4])",
            err,
        ),
    }
}

fn case_grad_grad_square() -> TransformControlFlowCase {
    let inputs = vec![Value::scalar_f64(5.0)];
    let transforms = [Transform::Grad, Transform::Grad];
    match dispatch_outputs(
        build_program(ProgramSpec::SquarePlusLinear),
        inputs.clone(),
        &transforms,
        BTreeMap::new(),
    ) {
        Ok(outputs) => {
            let actual = scalar_f64(&outputs, 0);
            supported_case(
                CaseMetadata::new(
                    "grad_grad_square",
                    "fj-dispatch",
                    &transforms,
                    None,
                    &["mul", "add"],
                    &inputs,
                    "fixtures/composition_oracle.v1.json#grad_grad_poly",
                    "jax_0.9_fixture",
                    &[
                        "crates/fj-conformance/tests/composition_oracle_parity.rs::composition_oracle_grad_grad",
                        "crates/fj-conformance/tests/control_flow_conformance.rs::test_grad_grad_square_plus_linear",
                    ],
                ),
                json!(2.0),
                json!(actual),
                &outputs,
                (actual - 2.0).abs() <= 1e-6,
                "absolute_error <= 1e-6",
            )
        }
        Err(err) => supported_error_case(
            "grad_grad_square",
            "fj-dispatch",
            &transforms,
            None,
            &["mul", "add"],
            &inputs,
            "fixtures/composition_oracle.v1.json#grad_grad_poly",
            err,
        ),
    }
}

fn case_vmap_grad_grad_square() -> TransformControlFlowCase {
    let inputs = vec![Value::vector_f64(&[1.0, 3.0, 5.0, 10.0]).expect("vector")];
    let transforms = [Transform::Vmap, Transform::Grad, Transform::Grad];
    let options = BTreeMap::from([("vmap_in_axes".to_owned(), "0".to_owned())]);
    match dispatch_outputs(
        build_program(ProgramSpec::Square),
        inputs.clone(),
        &transforms,
        options,
    ) {
        Ok(outputs) => {
            let actual = tensor_f64_vec(&outputs, 0);
            supported_case(
                CaseMetadata::new(
                    "vmap_grad_grad_square",
                    "fj-dispatch",
                    &transforms,
                    None,
                    &["mul"],
                    &inputs,
                    "analytic:jax.vmap(jax.grad(jax.grad(lambda x: x*x)))",
                    "analytic_jax_semantics",
                    &[
                        "crates/fj-conformance/tests/control_flow_conformance.rs::test_vmap_grad_grad_square",
                    ],
                ),
                json!([2.0, 2.0, 2.0, 2.0]),
                json!(actual),
                &outputs,
                f64_vec_close(&actual, &[2.0, 2.0, 2.0, 2.0], 1e-6),
                "all absolute_errors <= 1e-6",
            )
        }
        Err(err) => supported_error_case(
            "vmap_grad_grad_square",
            "fj-dispatch",
            &transforms,
            None,
            &["mul"],
            &inputs,
            "analytic:jax.vmap(jax.grad(jax.grad(lambda x: x*x)))",
            err,
        ),
    }
}

fn case_jit_vmap_grad_grad_square() -> TransformControlFlowCase {
    let inputs = vec![Value::vector_f64(&[2.0, 8.0]).expect("vector")];
    let transforms = [
        Transform::Jit,
        Transform::Vmap,
        Transform::Grad,
        Transform::Grad,
    ];
    let options = BTreeMap::from([("vmap_in_axes".to_owned(), "0".to_owned())]);
    match dispatch_outputs(
        build_program(ProgramSpec::Square),
        inputs.clone(),
        &transforms,
        options,
    ) {
        Ok(outputs) => {
            let actual = tensor_f64_vec(&outputs, 0);
            supported_case(
                CaseMetadata::new(
                    "jit_vmap_grad_grad_square",
                    "fj-dispatch",
                    &transforms,
                    None,
                    &["mul"],
                    &inputs,
                    "analytic:jax.jit(jax.vmap(jax.grad(jax.grad(lambda x: x*x))))",
                    "analytic_jax_semantics",
                    &[
                        "crates/fj-conformance/tests/control_flow_conformance.rs::test_jit_vmap_grad_grad_square",
                    ],
                ),
                json!([2.0, 2.0]),
                json!(actual),
                &outputs,
                f64_vec_close(&actual, &[2.0, 2.0], 1e-6),
                "all absolute_errors <= 1e-6",
            )
        }
        Err(err) => supported_error_case(
            "jit_vmap_grad_grad_square",
            "fj-dispatch",
            &transforms,
            None,
            &["mul"],
            &inputs,
            "analytic:jax.jit(jax.vmap(jax.grad(jax.grad(lambda x: x*x))))",
            err,
        ),
    }
}

fn case_vmap_jit_grad_square() -> TransformControlFlowCase {
    let inputs = vec![Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector")];
    let transforms = [Transform::Vmap, Transform::Jit, Transform::Grad];
    let options = BTreeMap::from([("vmap_in_axes".to_owned(), "0".to_owned())]);
    match dispatch_outputs(
        build_program(ProgramSpec::Square),
        inputs.clone(),
        &transforms,
        options,
    ) {
        Ok(outputs) => {
            let actual = tensor_f64_vec(&outputs, 0);
            supported_case(
                CaseMetadata::new(
                    "vmap_jit_grad_square",
                    "fj-dispatch",
                    &transforms,
                    None,
                    &["mul"],
                    &inputs,
                    "analytic:jax.vmap(jax.jit(jax.grad(lambda x: x*x)))",
                    "analytic_jax_semantics",
                    &[
                        "crates/fj-conformance/tests/control_flow_conformance.rs::test_vmap_jit_grad_square",
                    ],
                ),
                json!([2.0, 4.0, 6.0]),
                json!(actual),
                &outputs,
                f64_vec_close(&actual, &[2.0, 4.0, 6.0], 1e-6),
                "all absolute_errors <= 1e-6",
            )
        }
        Err(err) => supported_error_case(
            "vmap_jit_grad_square",
            "fj-dispatch",
            &transforms,
            None,
            &["mul"],
            &inputs,
            "analytic:jax.vmap(jax.jit(jax.grad(lambda x: x*x)))",
            err,
        ),
    }
}

fn case_value_and_grad_multi_output() -> TransformControlFlowCase {
    let inputs = vec![Value::scalar_f64(3.0)];
    let transforms = [Transform::Grad];
    let result = fj_ad::value_and_grad_jaxpr(&build_program(ProgramSpec::AddOneMulTwo), &inputs);
    match result {
        Ok((values, grads)) => {
            let mut outputs = values;
            outputs.extend(grads);
            let actual = values_to_json(&outputs);
            supported_case(
                CaseMetadata::new(
                    "value_and_grad_multi_output",
                    "fj-ad",
                    &transforms,
                    None,
                    &["add", "mul"],
                    &inputs,
                    "fixtures/composition_oracle.v1.json#value_and_grad_poly",
                    "jax_0.9_fixture",
                    &[
                        "crates/fj-ad/src/lib.rs::value_and_grad_returns_all_outputs",
                        "crates/fj-api/tests/api_transform_suite.rs::api_value_and_grad_square",
                    ],
                ),
                json!([4.0, 6.0, 1.0]),
                actual.clone(),
                &outputs,
                actual == json!([4.0, 6.0, 1.0]),
                "exact value_and_grad output vector match",
            )
        }
        Err(err) => supported_manual_error_case(
            "value_and_grad_multi_output",
            "fj-ad",
            &transforms,
            None,
            &["add", "mul"],
            &inputs,
            "fixtures/composition_oracle.v1.json#value_and_grad_poly",
            "ad_error",
            err.to_string(),
        ),
    }
}

fn case_jacobian_quadratic() -> TransformControlFlowCase {
    let inputs = vec![Value::scalar_f64(3.0)];
    let result = fj_ad::jacobian_jaxpr(&build_program(ProgramSpec::Square), &inputs);
    match result {
        Ok(output) => {
            let outputs = vec![output];
            let actual = tensor_f64_vec(&outputs, 0);
            supported_case(
                CaseMetadata::new(
                    "jacobian_quadratic",
                    "fj-ad",
                    &[],
                    None,
                    &["mul"],
                    &inputs,
                    "fixtures/composition_oracle.v1.json#jacobian_quadratic",
                    "jax_0.9_fixture",
                    &[
                        "crates/fj-conformance/tests/composition_oracle_parity.rs::fixture_covers_all_composition_types",
                        "crates/fj-ad/src/lib.rs::test_jacobian_two_outputs_two_inputs",
                    ],
                ),
                json!([6.0]),
                json!(actual),
                &outputs,
                f64_vec_close(&actual, &[6.0], 1e-6),
                "all absolute_errors <= 1e-6",
            )
        }
        Err(err) => supported_manual_error_case(
            "jacobian_quadratic",
            "fj-ad",
            &[],
            None,
            &["mul"],
            &inputs,
            "fixtures/composition_oracle.v1.json#jacobian_quadratic",
            "ad_error",
            err.to_string(),
        ),
    }
}

fn case_hessian_quadratic() -> TransformControlFlowCase {
    let inputs = vec![Value::scalar_f64(3.0)];
    let result = fj_ad::hessian_jaxpr(&build_program(ProgramSpec::Square), &inputs);
    match result {
        Ok(output) => {
            let outputs = vec![output];
            let actual = tensor_f64_vec(&outputs, 0);
            supported_case(
                CaseMetadata::new(
                    "hessian_quadratic",
                    "fj-ad",
                    &[],
                    None,
                    &["mul"],
                    &inputs,
                    "fixtures/composition_oracle.v1.json#hessian_quadratic",
                    "jax_0.9_fixture",
                    &[
                        "crates/fj-conformance/tests/composition_oracle_parity.rs::fixture_covers_all_composition_types",
                        "crates/fj-ad/src/lib.rs::test_hessian_x2y",
                    ],
                ),
                json!([2.0]),
                json!(actual),
                &outputs,
                f64_vec_close(&actual, &[2.0], 1e-3),
                "all absolute_errors <= 1e-3",
            )
        }
        Err(err) => supported_manual_error_case(
            "hessian_quadratic",
            "fj-ad",
            &[],
            None,
            &["mul"],
            &inputs,
            "fixtures/composition_oracle.v1.json#hessian_quadratic",
            "ad_error",
            err.to_string(),
        ),
    }
}

fn case_scan_multi_carry_state() -> TransformControlFlowCase {
    let inputs = vec![
        Value::scalar_f64(0.0),
        Value::scalar_f64(0.0),
        Value::vector_f64(&[10.0, 20.0, 30.0]).expect("vector should build"),
    ];
    let xs = inputs[2].clone();
    let result = eval_scan_functional(
        vec![inputs[0].clone(), inputs[1].clone()],
        &xs,
        |carry, x| {
            let new_count = eval_primitive(
                Primitive::Add,
                &[carry[0].clone(), Value::scalar_f64(1.0)],
                &BTreeMap::new(),
            )?;
            let new_sum = eval_primitive(
                Primitive::Add,
                &[carry[1].clone(), x.clone()],
                &BTreeMap::new(),
            )?;
            Ok((vec![new_count, new_sum], vec![x]))
        },
        false,
    );
    match result {
        Ok((carry, ys)) => {
            let outputs = carry
                .iter()
                .cloned()
                .chain(ys.iter().cloned())
                .collect::<Vec<_>>();
            let actual = json!({
                "carry": values_to_json(&carry),
                "ys": values_to_json(&ys),
            });
            supported_case(
                CaseMetadata::new(
                    "scan_multi_carry_state",
                    "fj-lax",
                    &[],
                    Some("scan"),
                    &["scan:functional", "add"],
                    &inputs,
                    "analytic:lax.scan multi-carry count/sum",
                    "analytic_jax_semantics",
                    &["crates/fj-lax/src/lib.rs::test_scan_multi_carry"],
                ),
                json!({"carry": [3.0, 60.0], "ys": [[10.0, 20.0, 30.0]]}),
                actual.clone(),
                &outputs,
                actual == json!({"carry": [3.0, 60.0], "ys": [[10.0, 20.0, 30.0]]}),
                "exact multi-carry and stacked-output match",
            )
        }
        Err(err) => supported_manual_error_case(
            "scan_multi_carry_state",
            "fj-lax",
            &[],
            Some("scan"),
            &["scan:functional", "add"],
            &inputs,
            "analytic:lax.scan multi-carry count/sum",
            "lax_eval_error",
            err.to_string(),
        ),
    }
}

fn case_vmap_multi_output_return() -> TransformControlFlowCase {
    let inputs = vec![Value::vector_i64(&[1, 2, 3]).expect("vector should build")];
    let transforms = [Transform::Vmap];
    match dispatch_outputs(
        build_program(ProgramSpec::AddOneMulTwo),
        inputs.clone(),
        &transforms,
        BTreeMap::new(),
    ) {
        Ok(outputs) => {
            let actual = values_to_json(&outputs);
            supported_case(
                CaseMetadata::new(
                    "vmap_multi_output_return",
                    "fj-dispatch",
                    &transforms,
                    None,
                    &["add", "mul"],
                    &inputs,
                    "analytic:jax.vmap(lambda x: (x+1, x*2))",
                    "analytic_jax_semantics",
                    &["crates/fj-core/src/lib.rs::ProgramSpec::AddOneMulTwo"],
                ),
                json!([[2, 3, 4], [2, 4, 6]]),
                actual.clone(),
                &outputs,
                actual == json!([[2, 3, 4], [2, 4, 6]]),
                "exact multi-output vector match",
            )
        }
        Err(err) => supported_error_case(
            "vmap_multi_output_return",
            "fj-dispatch",
            &transforms,
            None,
            &["add", "mul"],
            &inputs,
            "analytic:jax.vmap(lambda x: (x+1, x*2))",
            err,
        ),
    }
}

fn case_jit_mixed_dtype_add() -> TransformControlFlowCase {
    let inputs = vec![Value::scalar_i64(2), Value::scalar_f64(0.5)];
    let transforms = [Transform::Jit];
    match dispatch_outputs(
        build_program(ProgramSpec::Add2),
        inputs.clone(),
        &transforms,
        BTreeMap::new(),
    ) {
        Ok(outputs) => {
            let actual = scalar_f64(&outputs, 0);
            supported_case(
                CaseMetadata::new(
                    "jit_mixed_dtype_add",
                    "fj-dispatch",
                    &transforms,
                    None,
                    &["add"],
                    &inputs,
                    "analytic:jax.jit(lambda x, y: x + y)(int64, float64)",
                    "analytic_jax_semantics",
                    &["crates/fj-lax/src/arithmetic.rs::mixed scalar promotion"],
                ),
                json!(2.5),
                json!(actual),
                &outputs,
                (actual - 2.5).abs() <= 1e-9,
                "absolute_error <= 1e-9 and output dtype f64",
            )
        }
        Err(err) => supported_error_case(
            "jit_mixed_dtype_add",
            "fj-dispatch",
            &transforms,
            None,
            &["add"],
            &inputs,
            "analytic:jax.jit(lambda x, y: x + y)(int64, float64)",
            err,
        ),
    }
}

fn case_grad_vmap_vector_output_fail_closed() -> TransformControlFlowCase {
    let inputs = vec![Value::vector_i64(&[1, 2, 3]).expect("vector should build")];
    let transforms = [Transform::Grad, Transform::Vmap];
    let result = dispatch_outputs(
        build_program(ProgramSpec::AddOne),
        inputs.clone(),
        &transforms,
        BTreeMap::new(),
    );
    fail_closed_case(
        "grad_vmap_vector_output_fail_closed",
        "fj-dispatch",
        &transforms,
        None,
        &["add"],
        &inputs,
        "unsupported:v1_grad_requires_scalar_first_input_before_vmap_tail",
        result,
        "transform_execution.non_scalar_gradient_input",
        "Strict mode rejects grad(vmap(...)) over vector first inputs instead of silently differentiating a non-scalar contract; hardened mode keeps the same fail-closed boundary.",
    )
}

fn case_vmap_empty_batch_fail_closed() -> TransformControlFlowCase {
    let inputs = vec![Value::vector_i64(&[]).expect("empty vector should build")];
    let transforms = [Transform::Vmap];
    let result = dispatch_outputs(
        build_program(ProgramSpec::AddOne),
        inputs.clone(),
        &transforms,
        BTreeMap::new(),
    );
    fail_closed_case(
        "vmap_empty_batch_fail_closed",
        "fj-dispatch",
        &transforms,
        None,
        &["add"],
        &inputs,
        "unsupported:v1_empty_vmap_output_has_no_materialized_batch_shape",
        result,
        "transform_execution.empty_vmap_output",
        "Strict mode rejects empty vmap batches until an explicit empty-output shape contract is carried through dispatch; hardened mode must not synthesize data.",
    )
}

fn case_vmap_out_axes_none_nonconstant_fail_closed() -> TransformControlFlowCase {
    let inputs = vec![Value::vector_i64(&[1, 2, 3]).expect("vector should build")];
    let transforms = [Transform::Vmap];
    let options = BTreeMap::from([("vmap_out_axes".to_owned(), "none".to_owned())]);
    let result = dispatch_outputs(
        build_program(ProgramSpec::AddOne),
        inputs.clone(),
        &transforms,
        options,
    );
    fail_closed_case(
        "vmap_out_axes_none_nonconstant_fail_closed",
        "fj-dispatch",
        &transforms,
        None,
        &["add"],
        &inputs,
        "unsupported:v1_out_axes_none_requires_identical_outputs",
        result,
        "transform_execution.vmap_unmapped_output_mismatch",
        "Strict mode refuses to collapse non-identical mapped outputs into an unmapped scalar; hardened mode preserves this deterministic mismatch error.",
    )
}

fn build_performance_sentinels() -> Vec<TransformControlFlowPerformanceSentinel> {
    vec![
        measure_sentinel(
            "perf_vmap_scan_loop_stack",
            &[Transform::Vmap],
            Some("scan"),
            16,
            || {
                let outputs = dispatch_outputs(
                    scan_jaxpr("add", false),
                    vec![
                        Value::vector_i64(&[0, 10, 20, 30]).map_err(|err| err.to_string())?,
                        Value::vector_i64(&[1, 2, 3]).map_err(|err| err.to_string())?,
                    ],
                    &[Transform::Vmap],
                    BTreeMap::from([
                        ("vmap_in_axes".to_owned(), "0,none".to_owned()),
                        ("vmap_out_axes".to_owned(), "0".to_owned()),
                    ]),
                )
                .map_err(|err| err.to_string())?;
                Ok(outputs.iter().map(value_output_units).sum())
            },
            &[
                "benchmark:transform_control_flow/vmap_scan_loop_stack",
                "crates/fj-conformance/tests/control_flow_conformance.rs::test_scan_fixture_accumulate",
            ],
        ),
        measure_sentinel(
            "perf_vmap_while_loop_stack",
            &[Transform::Vmap],
            Some("while"),
            16,
            || {
                let outputs = dispatch_outputs(
                    while_jaxpr("add", "lt", 16),
                    vec![
                        Value::vector_i64(&[0, 2, 4]).map_err(|err| err.to_string())?,
                        Value::scalar_i64(3),
                        Value::scalar_i64(10),
                    ],
                    &[Transform::Vmap],
                    BTreeMap::from([
                        ("vmap_in_axes".to_owned(), "0,none,none".to_owned()),
                        ("vmap_out_axes".to_owned(), "0".to_owned()),
                    ]),
                )
                .map_err(|err| err.to_string())?;
                Ok(outputs.iter().map(value_output_units).sum())
            },
            &[
                "benchmark:transform_control_flow/vmap_while_loop_stack",
                "crates/fj-conformance/tests/control_flow_conformance.rs::test_while_fixture_countdown",
            ],
        ),
        measure_sentinel(
            "perf_jit_vmap_grad_cond",
            &[Transform::Jit, Transform::Vmap, Transform::Grad],
            Some("cond"),
            16,
            || {
                let outputs = dispatch_outputs(
                    grad_cond_jaxpr(),
                    vec![
                        Value::vector_f64(&[2.0, 4.0]).map_err(|err| err.to_string())?,
                        Value::vector_bool(&[false, false]).map_err(|err| err.to_string())?,
                    ],
                    &[Transform::Jit, Transform::Vmap, Transform::Grad],
                    BTreeMap::from([("vmap_in_axes".to_owned(), "0,0".to_owned())]),
                )
                .map_err(|err| err.to_string())?;
                Ok(outputs.iter().map(value_output_units).sum())
            },
            &[
                "benchmark:transform_control_flow/jit_vmap_grad_cond",
                "crates/fj-conformance/tests/control_flow_conformance.rs::test_jit_vmap_grad_cond",
            ],
        ),
        measure_sentinel(
            "perf_batched_switch",
            &[Transform::Vmap],
            Some("switch"),
            16,
            || {
                let outputs = dispatch_outputs(
                    switch_control_flow_jaxpr(),
                    vec![
                        Value::vector_i64(&[0, 1, 2]).map_err(|err| err.to_string())?,
                        Value::vector_i64(&[5, 6, 7]).map_err(|err| err.to_string())?,
                    ],
                    &[Transform::Vmap],
                    BTreeMap::new(),
                )
                .map_err(|err| err.to_string())?;
                Ok(outputs.iter().map(value_output_units).sum())
            },
            &[
                "benchmark:transform_control_flow/batched_switch",
                "crates/fj-conformance/tests/control_flow_conformance.rs::test_vmap_switch_dispatch_with_batched_indices",
            ],
        ),
    ]
}

struct CaseMetadata {
    case_id: String,
    api_surface: String,
    transform_stack: Vec<String>,
    control_flow_op: Option<String>,
    primitive_ops: Vec<String>,
    input_shapes: Vec<Vec<u32>>,
    input_dtypes: Vec<String>,
    oracle_fixture_id: String,
    oracle_kind: String,
    evidence_refs: Vec<String>,
}

impl CaseMetadata {
    #[allow(clippy::too_many_arguments)]
    fn new(
        case_id: &str,
        api_surface: &str,
        transforms: &[Transform],
        control_flow_op: Option<&str>,
        primitive_ops: &[&str],
        inputs: &[Value],
        oracle_fixture_id: &str,
        oracle_kind: &str,
        evidence_refs: &[&str],
    ) -> Self {
        Self {
            case_id: case_id.to_owned(),
            api_surface: api_surface.to_owned(),
            transform_stack: transforms_to_names(transforms),
            control_flow_op: control_flow_op.map(ToOwned::to_owned),
            primitive_ops: primitive_ops.iter().map(|op| (*op).to_owned()).collect(),
            input_shapes: inputs.iter().map(value_shape).collect(),
            input_dtypes: inputs.iter().map(value_dtype_name).collect(),
            oracle_fixture_id: oracle_fixture_id.to_owned(),
            oracle_kind: oracle_kind.to_owned(),
            evidence_refs: evidence_refs
                .iter()
                .map(|item| (*item).to_owned())
                .collect(),
        }
    }
}

fn supported_case(
    metadata: CaseMetadata,
    expected: JsonValue,
    actual: JsonValue,
    outputs: &[Value],
    pass: bool,
    comparison: &str,
) -> TransformControlFlowCase {
    TransformControlFlowCase {
        case_id: metadata.case_id,
        status: if pass { "pass" } else { "fail" }.to_owned(),
        support_status: "supported".to_owned(),
        api_surface: metadata.api_surface,
        transform_stack: metadata.transform_stack,
        control_flow_op: metadata.control_flow_op,
        primitive_ops: metadata.primitive_ops,
        input_shapes: metadata.input_shapes,
        input_dtypes: metadata.input_dtypes,
        output_shapes: outputs.iter().map(value_shape).collect(),
        output_dtypes: outputs.iter().map(value_dtype_name).collect(),
        oracle_fixture_id: metadata.oracle_fixture_id,
        oracle_kind: metadata.oracle_kind,
        tolerance_policy: "strict_numeric_or_exact_declared_per_row".to_owned(),
        comparison: comparison.to_owned(),
        expected,
        actual,
        error_class: None,
        error_message: None,
        strict_hardened_rationale:
            "Supported V1 row executes under strict mode and hardened mode must preserve the same observable transform semantics."
                .to_owned(),
        evidence_refs: metadata.evidence_refs,
        replay_command: "./scripts/run_transform_control_flow_gate.sh --enforce".to_owned(),
    }
}

#[allow(clippy::too_many_arguments)]
fn supported_error_case(
    case_id: &str,
    api_surface: &str,
    transforms: &[Transform],
    control_flow_op: Option<&str>,
    primitive_ops: &[&str],
    inputs: &[Value],
    oracle_fixture_id: &str,
    err: DispatchError,
) -> TransformControlFlowCase {
    supported_manual_error_case(
        case_id,
        api_surface,
        transforms,
        control_flow_op,
        primitive_ops,
        inputs,
        oracle_fixture_id,
        &dispatch_error_class(&err),
        err.to_string(),
    )
}

#[allow(clippy::too_many_arguments)]
fn supported_manual_error_case(
    case_id: &str,
    api_surface: &str,
    transforms: &[Transform],
    control_flow_op: Option<&str>,
    primitive_ops: &[&str],
    inputs: &[Value],
    oracle_fixture_id: &str,
    error_class: &str,
    error_message: String,
) -> TransformControlFlowCase {
    TransformControlFlowCase {
        case_id: case_id.to_owned(),
        status: "fail".to_owned(),
        support_status: "supported".to_owned(),
        api_surface: api_surface.to_owned(),
        transform_stack: transforms_to_names(transforms),
        control_flow_op: control_flow_op.map(ToOwned::to_owned),
        primitive_ops: primitive_ops.iter().map(|op| (*op).to_owned()).collect(),
        input_shapes: inputs.iter().map(value_shape).collect(),
        input_dtypes: inputs.iter().map(value_dtype_name).collect(),
        output_shapes: Vec::new(),
        output_dtypes: Vec::new(),
        oracle_fixture_id: oracle_fixture_id.to_owned(),
        oracle_kind: "expected_supported_row".to_owned(),
        tolerance_policy: "strict_numeric_or_exact_declared_per_row".to_owned(),
        comparison: "supported row errored before comparison".to_owned(),
        expected: JsonValue::Null,
        actual: JsonValue::Null,
        error_class: Some(error_class.to_owned()),
        error_message: Some(error_message),
        strict_hardened_rationale: "Supported V1 row must execute; any error is a matrix failure."
            .to_owned(),
        evidence_refs: vec![
            "artifacts/conformance/transform_control_flow_matrix.v1.json".to_owned(),
        ],
        replay_command: "./scripts/run_transform_control_flow_gate.sh --enforce".to_owned(),
    }
}

#[allow(clippy::too_many_arguments)]
fn fail_closed_case(
    case_id: &str,
    api_surface: &str,
    transforms: &[Transform],
    control_flow_op: Option<&str>,
    primitive_ops: &[&str],
    inputs: &[Value],
    oracle_fixture_id: &str,
    result: Result<Vec<Value>, DispatchError>,
    expected_error_class: &str,
    rationale: &str,
) -> TransformControlFlowCase {
    match result {
        Ok(outputs) => TransformControlFlowCase {
            case_id: case_id.to_owned(),
            status: "fail".to_owned(),
            support_status: "fail_closed".to_owned(),
            api_surface: api_surface.to_owned(),
            transform_stack: transforms_to_names(transforms),
            control_flow_op: control_flow_op.map(ToOwned::to_owned),
            primitive_ops: primitive_ops.iter().map(|op| (*op).to_owned()).collect(),
            input_shapes: inputs.iter().map(value_shape).collect(),
            input_dtypes: inputs.iter().map(value_dtype_name).collect(),
            output_shapes: outputs.iter().map(value_shape).collect(),
            output_dtypes: outputs.iter().map(value_dtype_name).collect(),
            oracle_fixture_id: oracle_fixture_id.to_owned(),
            oracle_kind: "unsupported_v1_contract".to_owned(),
            tolerance_policy: "typed_error_exact_match".to_owned(),
            comparison: "unsupported row unexpectedly executed".to_owned(),
            expected: json!({"error_class": expected_error_class}),
            actual: values_to_json(&outputs),
            error_class: None,
            error_message: None,
            strict_hardened_rationale: rationale.to_owned(),
            evidence_refs: vec![
                "crates/fj-dispatch/src/lib.rs::TransformExecutionError".to_owned(),
            ],
            replay_command: "./scripts/run_transform_control_flow_gate.sh --enforce".to_owned(),
        },
        Err(err) => {
            let actual_error_class = dispatch_error_class(&err);
            TransformControlFlowCase {
                case_id: case_id.to_owned(),
                status: if actual_error_class == expected_error_class {
                    "pass"
                } else {
                    "fail"
                }
                .to_owned(),
                support_status: "fail_closed".to_owned(),
                api_surface: api_surface.to_owned(),
                transform_stack: transforms_to_names(transforms),
                control_flow_op: control_flow_op.map(ToOwned::to_owned),
                primitive_ops: primitive_ops.iter().map(|op| (*op).to_owned()).collect(),
                input_shapes: inputs.iter().map(value_shape).collect(),
                input_dtypes: inputs.iter().map(value_dtype_name).collect(),
                output_shapes: Vec::new(),
                output_dtypes: Vec::new(),
                oracle_fixture_id: oracle_fixture_id.to_owned(),
                oracle_kind: "unsupported_v1_contract".to_owned(),
                tolerance_policy: "typed_error_exact_match".to_owned(),
                comparison: format!(
                    "expected typed error {expected_error_class}, got {actual_error_class}"
                ),
                expected: json!({"error_class": expected_error_class}),
                actual: json!({"error_class": actual_error_class}),
                error_class: Some(actual_error_class),
                error_message: Some(err.to_string()),
                strict_hardened_rationale: rationale.to_owned(),
                evidence_refs: vec![
                    "crates/fj-dispatch/src/lib.rs::TransformExecutionError".to_owned(),
                ],
                replay_command: "./scripts/run_transform_control_flow_gate.sh --enforce".to_owned(),
            }
        }
    }
}

fn measure_sentinel<F>(
    workload_id: &str,
    transforms: &[Transform],
    control_flow_op: Option<&str>,
    iterations: u32,
    mut workload: F,
    evidence_refs: &[&str],
) -> TransformControlFlowPerformanceSentinel
where
    F: FnMut() -> Result<u64, String>,
{
    let before = sample_process_memory();
    let mut durations = Vec::with_capacity(iterations as usize);
    let mut logical_output_units = 0_u64;
    let mut error = None;

    for _ in 0..iterations {
        let start = Instant::now();
        match workload() {
            Ok(units) => logical_output_units = logical_output_units.saturating_add(units),
            Err(err) => {
                error = Some(err);
                break;
            }
        }
        durations.push(start.elapsed().as_nanos().max(1));
    }

    let after = sample_process_memory();
    let mut sorted = durations.clone();
    sorted.sort_unstable();
    let p50_ns = percentile_ns(&sorted, 50);
    let p95_ns = percentile_ns(&sorted, 95);
    let p99_ns = percentile_ns(&sorted, 99);
    let peak = after.peak_rss_bytes.or(before.peak_rss_bytes);
    let delta_rss = before
        .current_rss_bytes
        .zip(after.current_rss_bytes)
        .map(|(before, after)| after as i64 - before as i64);
    let status = if error.is_none()
        && !durations.is_empty()
        && p50_ns > 0
        && p95_ns > 0
        && p99_ns > 0
        && logical_output_units > 0
        && after.measurement_backend != "unavailable"
    {
        "pass"
    } else {
        "fail"
    };

    TransformControlFlowPerformanceSentinel {
        workload_id: workload_id.to_owned(),
        status: status.to_owned(),
        iterations,
        transform_stack: transforms_to_names(transforms),
        control_flow_op: control_flow_op.map(ToOwned::to_owned),
        p50_ns,
        p95_ns,
        p99_ns,
        measurement_backend: after.measurement_backend,
        rss_before_bytes: before.current_rss_bytes,
        rss_after_bytes: after.current_rss_bytes,
        peak_rss_bytes: peak,
        delta_rss_bytes: delta_rss,
        logical_output_units,
        evidence_refs: evidence_refs
            .iter()
            .map(|item| (*item).to_owned())
            .collect(),
        replay_command: "./scripts/run_transform_control_flow_gate.sh --enforce".to_owned(),
        error,
    }
}

fn dispatch_outputs(
    jaxpr: Jaxpr,
    args: Vec<Value>,
    transforms: &[Transform],
    compile_options: BTreeMap<String, String>,
) -> Result<Vec<Value>, DispatchError> {
    let mut ledger = TraceTransformLedger::new(jaxpr);
    for (idx, transform) in transforms.iter().enumerate() {
        ledger.push_transform(
            *transform,
            format!("transform-control-flow-{}-{idx}", transform.as_str()),
        );
    }
    dispatch(DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger,
        args,
        backend: "cpu".to_owned(),
        compile_options,
        custom_hook: None,
        unknown_incompatible_features: vec![],
    })
    .map(|response| response.outputs)
}

fn grad_cond_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(4)],
        vec![
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
            Equation {
                primitive: Primitive::Cond,
                inputs: smallvec![
                    Atom::Var(VarId(2)),
                    Atom::Var(VarId(1)),
                    Atom::Var(VarId(3))
                ],
                outputs: smallvec![VarId(4)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            },
        ],
    )
}

fn scan_jaxpr(body_op: &str, reverse: bool) -> Jaxpr {
    let mut params = BTreeMap::from([("body_op".to_owned(), body_op.to_owned())]);
    if reverse {
        params.insert("reverse".to_owned(), "true".to_owned());
    }
    Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3)],
        vec![Equation {
            primitive: Primitive::Scan,
            inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec![VarId(3)],
            params,
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    )
}

fn while_jaxpr(body_op: &str, cond_op: &str, max_iter: usize) -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1), VarId(2), VarId(3)],
        vec![],
        vec![VarId(4)],
        vec![Equation {
            primitive: Primitive::While,
            inputs: smallvec![
                Atom::Var(VarId(1)),
                Atom::Var(VarId(2)),
                Atom::Var(VarId(3))
            ],
            outputs: smallvec![VarId(4)],
            params: BTreeMap::from([
                ("body_op".to_owned(), body_op.to_owned()),
                ("cond_op".to_owned(), cond_op.to_owned()),
                ("max_iter".to_owned(), max_iter.to_string()),
            ]),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    )
}

fn switch_branch_identity_jaxpr() -> Jaxpr {
    Jaxpr::new(vec![VarId(1)], vec![], vec![VarId(1)], vec![])
}

fn switch_branch_self_binary_jaxpr(primitive: Primitive) -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive,
            inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
            outputs: smallvec![VarId(2)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    )
}

fn switch_control_flow_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3)],
        vec![Equation {
            primitive: Primitive::Switch,
            inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec![VarId(3)],
            params: BTreeMap::from([("num_branches".to_owned(), "3".to_owned())]),
            effects: vec![],
            sub_jaxprs: vec![
                switch_branch_identity_jaxpr(),
                switch_branch_self_binary_jaxpr(Primitive::Add),
                switch_branch_self_binary_jaxpr(Primitive::Mul),
            ],
        }],
    )
}

fn transforms_to_names(transforms: &[Transform]) -> Vec<String> {
    transforms
        .iter()
        .map(|transform| transform.as_str().to_owned())
        .collect()
}

fn value_shape(value: &Value) -> Vec<u32> {
    match value {
        Value::Scalar(_) => Vec::new(),
        Value::Tensor(tensor) => tensor.shape.dims.clone(),
    }
}

fn value_dtype_name(value: &Value) -> String {
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

fn values_to_json(values: &[Value]) -> JsonValue {
    JsonValue::Array(values.iter().map(value_to_json).collect())
}

fn value_to_json(value: &Value) -> JsonValue {
    match value {
        Value::Scalar(lit) => literal_to_json(*lit),
        Value::Tensor(tensor) => {
            if tensor.shape.rank() == 1 {
                JsonValue::Array(
                    tensor
                        .elements
                        .iter()
                        .map(|lit| literal_to_json(*lit))
                        .collect(),
                )
            } else {
                json!({
                    "dtype": dtype_name(tensor.dtype),
                    "shape": tensor.shape.dims,
                    "elements": tensor.elements.iter().map(|lit| literal_to_json(*lit)).collect::<Vec<_>>(),
                })
            }
        }
    }
}

fn literal_to_json(lit: Literal) -> JsonValue {
    match lit {
        Literal::I64(value) => json!(value),
        Literal::U32(value) => json!(value),
        Literal::U64(value) => json!(value),
        Literal::Bool(value) => json!(value),
        Literal::BF16Bits(_) | Literal::F16Bits(_) | Literal::F32Bits(_) | Literal::F64Bits(_) => {
            json!(lit.as_f64().expect("float literal should convert to f64"))
        }
        Literal::Complex64Bits(_, _) => {
            let (re, im) = lit
                .as_complex64()
                .expect("complex64 literal should convert to pair");
            json!({"re": re, "im": im})
        }
        Literal::Complex128Bits(_, _) => {
            let (re, im) = lit
                .as_complex128()
                .expect("complex128 literal should convert to pair");
            json!({"re": re, "im": im})
        }
    }
}

fn scalar_f64(outputs: &[Value], index: usize) -> f64 {
    outputs
        .get(index)
        .and_then(Value::as_f64_scalar)
        .unwrap_or(f64::NAN)
}

fn tensor_f64_vec(outputs: &[Value], index: usize) -> Vec<f64> {
    outputs
        .get(index)
        .and_then(Value::as_tensor)
        .and_then(|tensor| tensor.to_f64_vec())
        .unwrap_or_default()
}

fn tensor_i64_vec(outputs: &[Value], index: usize) -> Vec<i64> {
    outputs
        .get(index)
        .and_then(Value::as_tensor)
        .map(|tensor| {
            tensor
                .elements
                .iter()
                .filter_map(|lit| lit.as_i64())
                .collect()
        })
        .unwrap_or_default()
}

fn f64_vec_close(actual: &[f64], expected: &[f64], tolerance: f64) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected.iter())
            .all(|(actual, expected)| (*actual - *expected).abs() <= tolerance)
}

fn value_output_units(value: &Value) -> u64 {
    match value {
        Value::Scalar(_) => 1,
        Value::Tensor(tensor) => tensor.len() as u64,
    }
}

fn dispatch_error_class(err: &DispatchError) -> String {
    match err {
        DispatchError::TransformExecution(transform_error) => {
            format!(
                "transform_execution.{}",
                transform_error_class(transform_error)
            )
        }
        DispatchError::Cache(_) => "cache".to_owned(),
        DispatchError::Interpreter(_) => "interpreter".to_owned(),
        DispatchError::BackendExecution(_) => "backend_execution".to_owned(),
        DispatchError::TransformInvariant(_) => "transform_invariant".to_owned(),
    }
}

fn transform_error_class(err: &TransformExecutionError) -> &'static str {
    match err {
        TransformExecutionError::EmptyArgumentList { .. } => "empty_argument_list",
        TransformExecutionError::NonScalarGradientInput => "non_scalar_gradient_input",
        TransformExecutionError::NonScalarGradientOutput => "non_scalar_gradient_output",
        TransformExecutionError::VmapRequiresRankOneLeadingArgument => {
            "vmap_requires_rank_one_leading_argument"
        }
        TransformExecutionError::VmapMismatchedLeadingDimension { .. } => {
            "vmap_mismatched_leading_dimension"
        }
        TransformExecutionError::VmapInconsistentOutputArity { .. } => {
            "vmap_inconsistent_output_arity"
        }
        TransformExecutionError::VmapAxesOutOfBounds { .. } => "vmap_axes_out_of_bounds",
        TransformExecutionError::VmapAxesCountMismatch { .. } => "vmap_axes_count_mismatch",
        TransformExecutionError::InvalidVmapAxisSpec { .. } => "invalid_vmap_axis_spec",
        TransformExecutionError::VmapUnmappedOutputMismatch => "vmap_unmapped_output_mismatch",
        TransformExecutionError::FiniteDiffGradFallbackDisabled { .. } => {
            "finite_diff_grad_fallback_disabled"
        }
        TransformExecutionError::EmptyVmapOutput => "empty_vmap_output",
        TransformExecutionError::TensorBuild(_) => "tensor_build",
    }
}

fn percentile_ns(sorted: &[u128], percentile: u32) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let max_index = sorted.len() - 1;
    let index = ((max_index as f64) * (f64::from(percentile) / 100.0)).ceil() as usize;
    sorted[index.min(max_index)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn validator_rejects_missing_required_rows() {
        let mut report = TransformControlFlowReport {
            schema_version: TRANSFORM_CONTROL_FLOW_REPORT_SCHEMA_VERSION.to_owned(),
            bead_id: TRANSFORM_CONTROL_FLOW_BEAD_ID.to_owned(),
            generated_at_unix_ms: 0,
            status: "pass".to_owned(),
            matrix_policy: "test".to_owned(),
            supported_rows: 0,
            fail_closed_rows: 0,
            cases: Vec::new(),
            performance_sentinels: Vec::new(),
            issues: Vec::new(),
            artifact_refs: Vec::new(),
            replay_command: "replay".to_owned(),
        };
        report.issues = validate_transform_control_flow_report(&report);
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.code == "missing_required_case")
        );
        assert!(
            report
                .issues
                .iter()
                .any(|issue| issue.code == "missing_required_performance_sentinel")
        );
    }

    #[test]
    fn current_matrix_report_passes() {
        let temp = tempfile::tempdir().expect("tempdir");
        let report = build_transform_control_flow_report(temp.path());
        assert_eq!(
            report.schema_version,
            TRANSFORM_CONTROL_FLOW_REPORT_SCHEMA_VERSION
        );
        assert_eq!(report.bead_id, TRANSFORM_CONTROL_FLOW_BEAD_ID);
        assert_eq!(report.status, "pass", "report: {report:#?}");
        assert!(report.issues.is_empty(), "issues: {:#?}", report.issues);
        let case_ids = report
            .cases
            .iter()
            .map(|case| case.case_id.as_str())
            .collect::<BTreeSet<_>>();
        for required in REQUIRED_CASES {
            assert!(case_ids.contains(required), "missing {required}");
        }
    }
}
