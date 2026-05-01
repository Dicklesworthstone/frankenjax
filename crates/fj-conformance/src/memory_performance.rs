#![forbid(unsafe_code)]

use crate::durability::{
    SidecarConfig, encode_artifact_to_sidecar, generate_decode_proof, scrub_sidecar,
};
use fj_cache::{CacheKey, CacheKeyInput, CacheLookup, CacheManager, build_cache_key};
use fj_core::{
    CompatibilityMode, DType, Literal, Primitive, ProgramSpec, Shape, TensorValue,
    TraceTransformLedger, Transform, Value, build_program,
};
use fj_dispatch::{DispatchRequest, dispatch};
use fj_lax::eval_primitive;
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const MEMORY_PERFORMANCE_REPORT_SCHEMA_VERSION: &str = "frankenjax.memory-performance-gate.v1";
pub const DEFAULT_PEAK_RSS_BUDGET_BYTES: u64 = 1_073_741_824;

const REQUIRED_WORKLOADS: &[&str] = &[
    "trace_canonical_fingerprint",
    "dispatch_jit_scalar",
    "ad_grad_scalar",
    "vmap_vector_add_one",
    "fft_complex_vector",
    "linalg_cholesky_matrix",
    "cache_hit_miss",
    "durability_sidecar_round_trip",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryProbeSample {
    pub measurement_backend: String,
    pub current_rss_bytes: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryWorkloadMeasurement {
    pub phase_id: String,
    pub workload_id: String,
    pub status: String,
    pub iterations: u32,
    pub measurement_backend: String,
    pub peak_rss_budget_bytes: u64,
    pub rss_before_bytes: Option<u64>,
    pub rss_after_bytes: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    pub delta_rss_bytes: Option<i64>,
    pub logical_output_units: u64,
    pub evidence_refs: Vec<String>,
    pub replay_command: String,
    pub notes: Vec<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPerformanceIssue {
    pub code: String,
    pub path: String,
    pub message: String,
}

impl MemoryPerformanceIssue {
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
pub struct MemoryPerformanceReport {
    pub schema_version: String,
    pub bead_id: String,
    pub generated_at_unix_ms: u128,
    pub status: String,
    pub measurement_policy: String,
    pub measurement_backend: String,
    pub peak_rss_budget_bytes: u64,
    pub workloads: Vec<MemoryWorkloadMeasurement>,
    pub issues: Vec<MemoryPerformanceIssue>,
    pub artifact_refs: Vec<String>,
    pub replay_command: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryGateOutputPaths {
    pub report: PathBuf,
    pub markdown: PathBuf,
    pub durability_probe: PathBuf,
    pub durability_sidecar: PathBuf,
    pub durability_scrub: PathBuf,
    pub durability_proof: PathBuf,
}

impl MemoryGateOutputPaths {
    #[must_use]
    pub fn for_root(root: &Path) -> Self {
        let performance = root.join("artifacts/performance");
        Self {
            report: performance.join("memory_performance_gate.v1.json"),
            markdown: performance.join("memory_performance_gate.v1.md"),
            durability_probe: performance.join("memory_durability_probe.v1.json"),
            durability_sidecar: performance.join("memory_durability_probe.v1.sidecar.json"),
            durability_scrub: performance.join("memory_durability_probe.v1.scrub.json"),
            durability_proof: performance.join("memory_durability_probe.v1.proof.json"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WorkloadOutcome {
    logical_output_units: u64,
    evidence_refs: Vec<String>,
}

pub fn build_memory_performance_report(root: &Path) -> MemoryPerformanceReport {
    let paths = MemoryGateOutputPaths::for_root(root);
    let mut runner = MemoryWorkloadRunner::new(paths);
    let workloads = vec![
        runner.measure(
            "trace",
            "trace_canonical_fingerprint",
            32,
            workload_trace_canonical_fingerprint,
        ),
        runner.measure(
            "compile_dispatch",
            "dispatch_jit_scalar",
            16,
            workload_dispatch_jit_scalar,
        ),
        runner.measure("ad", "ad_grad_scalar", 16, workload_ad_grad_scalar),
        runner.measure(
            "vmap",
            "vmap_vector_add_one",
            16,
            workload_vmap_vector_add_one,
        ),
        runner.measure("fft", "fft_complex_vector", 16, workload_fft_complex_vector),
        runner.measure(
            "linalg",
            "linalg_cholesky_matrix",
            16,
            workload_linalg_cholesky_matrix,
        ),
        runner.measure("cache", "cache_hit_miss", 32, workload_cache_hit_miss),
        runner.measure(
            "durability",
            "durability_sidecar_round_trip",
            1,
            workload_durability_sidecar_round_trip,
        ),
    ];

    let measurement_backend = workloads
        .iter()
        .find_map(|workload| {
            (workload.measurement_backend != "unavailable")
                .then(|| workload.measurement_backend.clone())
        })
        .unwrap_or_else(|| "unavailable".to_owned());

    let mut report = MemoryPerformanceReport {
        schema_version: MEMORY_PERFORMANCE_REPORT_SCHEMA_VERSION.to_owned(),
        bead_id: "frankenjax-cstq.4".to_owned(),
        generated_at_unix_ms: now_unix_ms(),
        status: "pass".to_owned(),
        measurement_policy:
            "Linux procfs VmRSS/VmHWM smoke workloads; no synthetic allocation counts".to_owned(),
        measurement_backend,
        peak_rss_budget_bytes: DEFAULT_PEAK_RSS_BUDGET_BYTES,
        workloads,
        issues: Vec::new(),
        artifact_refs: runner.artifact_refs,
        replay_command: "./scripts/run_memory_performance_gate.sh --enforce".to_owned(),
    };
    report.issues = validate_memory_performance_report(&report);
    if !report.issues.is_empty() {
        report.status = "fail".to_owned();
    }
    report
}

pub fn write_memory_performance_outputs(
    root: &Path,
    report_path: &Path,
    markdown_path: &Path,
) -> Result<MemoryPerformanceReport, std::io::Error> {
    let report = build_memory_performance_report(root);
    write_json(report_path, &report)?;
    write_markdown(markdown_path, &memory_performance_markdown(&report))?;
    Ok(report)
}

#[must_use]
pub fn memory_performance_summary_json(report: &MemoryPerformanceReport) -> JsonValue {
    let max_peak_rss_bytes = report
        .workloads
        .iter()
        .filter_map(|workload| workload.peak_rss_bytes)
        .max();
    json!({
        "status": report.status,
        "schema_version": report.schema_version,
        "workload_count": report.workloads.len(),
        "issue_count": report.issues.len(),
        "measurement_backend": report.measurement_backend,
        "max_peak_rss_bytes": max_peak_rss_bytes,
        "workloads": report.workloads.iter().map(|workload| {
            json!({
                "phase_id": workload.phase_id,
                "workload_id": workload.workload_id,
                "status": workload.status,
                "iterations": workload.iterations,
                "peak_rss_bytes": workload.peak_rss_bytes,
                "delta_rss_bytes": workload.delta_rss_bytes,
                "logical_output_units": workload.logical_output_units,
            })
        }).collect::<Vec<_>>(),
    })
}

#[must_use]
pub fn memory_performance_markdown(report: &MemoryPerformanceReport) -> String {
    let mut out = String::new();
    out.push_str("# Memory Performance Gate\n\n");
    out.push_str(&format!(
        "- schema: `{}`\n- bead: `{}`\n- status: `{}`\n- backend: `{}`\n- peak RSS budget: `{}` bytes\n\n",
        report.schema_version,
        report.bead_id,
        report.status,
        report.measurement_backend,
        report.peak_rss_budget_bytes
    ));
    out.push_str("| Phase | Workload | Status | Iterations | Peak RSS bytes | Delta RSS bytes | Evidence |\n");
    out.push_str("|---|---|---:|---:|---:|---:|---|\n");
    for workload in &report.workloads {
        out.push_str(&format!(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |\n",
            workload.phase_id,
            workload.workload_id,
            workload.status,
            workload.iterations,
            workload
                .peak_rss_bytes
                .map_or_else(|| "n/a".to_owned(), |value| value.to_string()),
            workload
                .delta_rss_bytes
                .map_or_else(|| "n/a".to_owned(), |value| value.to_string()),
            workload.evidence_refs.join(", ").replace('|', "/")
        ));
    }
    out.push_str("\n## Issues\n\n");
    if report.issues.is_empty() {
        out.push_str("No memory performance issues found.\n");
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
pub fn validate_memory_performance_report(
    report: &MemoryPerformanceReport,
) -> Vec<MemoryPerformanceIssue> {
    let mut issues = Vec::new();
    if report.schema_version != MEMORY_PERFORMANCE_REPORT_SCHEMA_VERSION {
        issues.push(MemoryPerformanceIssue::new(
            "unsupported_schema_version",
            "$.schema_version",
            format!(
                "expected {}, got {}",
                MEMORY_PERFORMANCE_REPORT_SCHEMA_VERSION, report.schema_version
            ),
        ));
    }
    if report.bead_id != "frankenjax-cstq.4" {
        issues.push(MemoryPerformanceIssue::new(
            "wrong_bead_id",
            "$.bead_id",
            "memory performance gate must remain bound to frankenjax-cstq.4",
        ));
    }
    if report.peak_rss_budget_bytes == 0 {
        issues.push(MemoryPerformanceIssue::new(
            "empty_peak_rss_budget",
            "$.peak_rss_budget_bytes",
            "peak RSS budget must be positive",
        ));
    }

    let mut seen = BTreeMap::<&str, usize>::new();
    for (idx, workload) in report.workloads.iter().enumerate() {
        *seen.entry(workload.workload_id.as_str()).or_default() += 1;
        let path = format!("$.workloads[{idx}]");
        if workload.status != "pass" {
            issues.push(MemoryPerformanceIssue::new(
                "workload_failed",
                format!("{path}.status"),
                format!("workload `{}` did not pass", workload.workload_id),
            ));
        }
        if workload.iterations == 0 {
            issues.push(MemoryPerformanceIssue::new(
                "zero_iterations",
                format!("{path}.iterations"),
                "memory workload must execute at least once",
            ));
        }
        if workload.measurement_backend == "not_measured"
            || workload.measurement_backend == "unavailable"
            || workload.measurement_backend.is_empty()
        {
            issues.push(MemoryPerformanceIssue::new(
                "memory_not_measured",
                format!("{path}.measurement_backend"),
                "workload must use a concrete RSS measurement backend",
            ));
        }
        match workload.peak_rss_bytes {
            Some(peak) if peak > 0 => {
                if peak > workload.peak_rss_budget_bytes {
                    issues.push(MemoryPerformanceIssue::new(
                        "peak_rss_budget_exceeded",
                        format!("{path}.peak_rss_bytes"),
                        format!(
                            "workload `{}` peak RSS {peak} exceeds budget {}",
                            workload.workload_id, workload.peak_rss_budget_bytes
                        ),
                    ));
                }
            }
            _ => issues.push(MemoryPerformanceIssue::new(
                "missing_peak_rss",
                format!("{path}.peak_rss_bytes"),
                "workload must record non-zero peak RSS bytes",
            )),
        }
        if workload.logical_output_units == 0 {
            issues.push(MemoryPerformanceIssue::new(
                "empty_workload_output",
                format!("{path}.logical_output_units"),
                "workload must produce a non-empty behavior witness",
            ));
        }
        if workload.evidence_refs.is_empty() {
            issues.push(MemoryPerformanceIssue::new(
                "missing_evidence_refs",
                format!("{path}.evidence_refs"),
                "workload must link to a benchmark, API surface, or artifact witness",
            ));
        }
        if workload.replay_command.is_empty() {
            issues.push(MemoryPerformanceIssue::new(
                "missing_replay_command",
                format!("{path}.replay_command"),
                "workload must include a replay command",
            ));
        }
    }

    for required in REQUIRED_WORKLOADS {
        match seen.get(required).copied() {
            Some(1) => {}
            Some(count) => issues.push(MemoryPerformanceIssue::new(
                "duplicate_required_workload",
                "$.workloads",
                format!("required workload `{required}` appears {count} times"),
            )),
            None => issues.push(MemoryPerformanceIssue::new(
                "missing_required_workload",
                "$.workloads",
                format!("required workload `{required}` is absent"),
            )),
        }
    }

    issues
}

#[must_use]
pub fn sample_process_memory() -> MemoryProbeSample {
    match fs::read_to_string("/proc/self/status") {
        Ok(raw) => MemoryProbeSample {
            measurement_backend: "linux_procfs_status_vm_hwm".to_owned(),
            current_rss_bytes: parse_proc_status_kib(&raw, "VmRSS:"),
            peak_rss_bytes: parse_proc_status_kib(&raw, "VmHWM:"),
        },
        Err(_) => MemoryProbeSample {
            measurement_backend: "unavailable".to_owned(),
            current_rss_bytes: None,
            peak_rss_bytes: None,
        },
    }
}

#[must_use]
pub fn parse_proc_status_kib(raw: &str, key: &str) -> Option<u64> {
    raw.lines().find_map(|line| {
        let rest = line.strip_prefix(key)?;
        let kib = rest.split_whitespace().next()?.parse::<u64>().ok()?;
        kib.checked_mul(1024)
    })
}

struct MemoryWorkloadRunner {
    paths: MemoryGateOutputPaths,
    artifact_refs: Vec<String>,
}

impl MemoryWorkloadRunner {
    fn new(paths: MemoryGateOutputPaths) -> Self {
        Self {
            paths,
            artifact_refs: Vec::new(),
        }
    }

    fn measure<F>(
        &mut self,
        phase_id: &str,
        workload_id: &str,
        iterations: u32,
        mut workload: F,
    ) -> MemoryWorkloadMeasurement
    where
        F: FnMut(&MemoryGateOutputPaths) -> Result<WorkloadOutcome, String>,
    {
        let before = sample_process_memory();
        let mut logical_output_units = 0_u64;
        let mut evidence_refs = Vec::new();
        let mut error = None;

        for _ in 0..iterations {
            match workload(&self.paths) {
                Ok(outcome) => {
                    logical_output_units =
                        logical_output_units.saturating_add(outcome.logical_output_units);
                    for evidence_ref in outcome.evidence_refs {
                        if !evidence_refs.contains(&evidence_ref) {
                            evidence_refs.push(evidence_ref);
                        }
                    }
                }
                Err(err) => {
                    error = Some(err);
                    break;
                }
            }
        }

        for evidence_ref in &evidence_refs {
            if evidence_ref.starts_with("artifacts/") && !self.artifact_refs.contains(evidence_ref)
            {
                self.artifact_refs.push(evidence_ref.clone());
            }
        }

        let after = sample_process_memory();
        let peak = after.peak_rss_bytes.or(before.peak_rss_bytes);
        let delta_rss = before
            .current_rss_bytes
            .zip(after.current_rss_bytes)
            .map(|(before, after)| after as i64 - before as i64);
        let status = if error.is_none()
            && peak.is_some_and(|peak| peak > 0 && peak <= DEFAULT_PEAK_RSS_BUDGET_BYTES)
            && after.measurement_backend != "unavailable"
            && logical_output_units > 0
        {
            "pass"
        } else {
            "fail"
        };
        let mut notes = vec![
            "RSS is sampled from the gate process before and after the workload".to_owned(),
            "Allocation counts are intentionally not synthesized".to_owned(),
        ];
        if delta_rss == Some(0) {
            notes.push("VmRSS delta is zero because the workload reused process memory".to_owned());
        }

        MemoryWorkloadMeasurement {
            phase_id: phase_id.to_owned(),
            workload_id: workload_id.to_owned(),
            status: status.to_owned(),
            iterations,
            measurement_backend: after.measurement_backend,
            peak_rss_budget_bytes: DEFAULT_PEAK_RSS_BUDGET_BYTES,
            rss_before_bytes: before.current_rss_bytes,
            rss_after_bytes: after.current_rss_bytes,
            peak_rss_bytes: peak,
            delta_rss_bytes: delta_rss,
            logical_output_units,
            evidence_refs,
            replay_command: "./scripts/run_memory_performance_gate.sh --enforce".to_owned(),
            notes,
            error,
        }
    }
}

fn workload_trace_canonical_fingerprint(
    _paths: &MemoryGateOutputPaths,
) -> Result<WorkloadOutcome, String> {
    let jaxpr = build_program(ProgramSpec::SquarePlusLinear);
    jaxpr
        .validate_well_formed()
        .map_err(|err| format!("jaxpr validation failed: {err}"))?;
    Ok(WorkloadOutcome {
        logical_output_units: jaxpr.canonical_fingerprint().len() as u64,
        evidence_refs: vec!["benchmark:jaxpr_fingerprint/canonical_fingerprint/10".to_owned()],
    })
}

fn workload_dispatch_jit_scalar(_paths: &MemoryGateOutputPaths) -> Result<WorkloadOutcome, String> {
    let response = dispatch(dispatch_request(
        ProgramSpec::Add2,
        &[Transform::Jit],
        vec![Value::scalar_i64(3), Value::scalar_i64(4)],
    ))
    .map_err(|err| format!("dispatch jit failed: {err}"))?;
    Ok(WorkloadOutcome {
        logical_output_units: response.outputs.len() as u64 + response.cache_key.len() as u64,
        evidence_refs: vec!["benchmark:dispatch_latency/jit/scalar_add".to_owned()],
    })
}

fn workload_ad_grad_scalar(_paths: &MemoryGateOutputPaths) -> Result<WorkloadOutcome, String> {
    let response = dispatch(dispatch_request(
        ProgramSpec::Square,
        &[Transform::Grad],
        vec![Value::scalar_f64(3.0)],
    ))
    .map_err(|err| format!("dispatch grad failed: {err}"))?;
    let derivative = response
        .outputs
        .first()
        .and_then(Value::as_f64_scalar)
        .ok_or_else(|| "grad output was not a scalar f64".to_owned())?;
    if (derivative - 6.0).abs() > 1e-3 {
        return Err(format!("grad derivative mismatch: {derivative}"));
    }
    Ok(WorkloadOutcome {
        logical_output_units: response.outputs.len() as u64 + 6,
        evidence_refs: vec!["benchmark:dispatch_latency/grad/scalar_square".to_owned()],
    })
}

fn workload_vmap_vector_add_one(_paths: &MemoryGateOutputPaths) -> Result<WorkloadOutcome, String> {
    let response = dispatch(dispatch_request(
        ProgramSpec::AddOne,
        &[Transform::Vmap],
        vec![Value::vector_i64(&[1, 2, 3, 4]).map_err(|err| err.to_string())?],
    ))
    .map_err(|err| format!("dispatch vmap failed: {err}"))?;
    let tensor = response
        .outputs
        .first()
        .and_then(Value::as_tensor)
        .ok_or_else(|| "vmap output was not a tensor".to_owned())?;
    Ok(WorkloadOutcome {
        logical_output_units: tensor.elements.len() as u64,
        evidence_refs: vec!["benchmark:dispatch_latency/vmap/vector_add_one".to_owned()],
    })
}

fn workload_fft_complex_vector(_paths: &MemoryGateOutputPaths) -> Result<WorkloadOutcome, String> {
    let input = Value::Tensor(
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
    let output = eval_primitive(Primitive::Fft, &[input], &BTreeMap::new())
        .map_err(|err| format!("fft failed: {err}"))?;
    let tensor = output
        .as_tensor()
        .ok_or_else(|| "fft output was not a tensor".to_owned())?;
    Ok(WorkloadOutcome {
        logical_output_units: tensor.elements.len() as u64,
        evidence_refs: vec!["surface:lax_fft".to_owned()],
    })
}

fn workload_linalg_cholesky_matrix(
    _paths: &MemoryGateOutputPaths,
) -> Result<WorkloadOutcome, String> {
    let input = Value::Tensor(
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
    let output = eval_primitive(Primitive::Cholesky, &[input], &BTreeMap::new())
        .map_err(|err| format!("cholesky failed: {err}"))?;
    let tensor = output
        .as_tensor()
        .ok_or_else(|| "cholesky output was not a tensor".to_owned())?;
    Ok(WorkloadOutcome {
        logical_output_units: tensor.elements.len() as u64,
        evidence_refs: vec!["surface:lax_linalg_cholesky".to_owned()],
    })
}

fn workload_cache_hit_miss(_paths: &MemoryGateOutputPaths) -> Result<WorkloadOutcome, String> {
    let mut manager = CacheManager::in_memory();
    let key_input = CacheKeyInput {
        mode: CompatibilityMode::Strict,
        backend: "cpu".to_owned(),
        jaxpr: build_program(ProgramSpec::Add2),
        transform_stack: vec![Transform::Jit],
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    };
    let key = build_cache_key(&key_input).map_err(|err| format!("cache key failed: {err}"))?;
    manager.put(&key, vec![1_u8; 64]);
    let hit_len = match manager.get(&key) {
        CacheLookup::Hit { data } => data.len(),
        other => return Err(format!("expected cache hit, got {other:?}")),
    };
    let miss = CacheKey {
        namespace: "fjx",
        digest_hex: "memory_gate_missing_key".to_owned(),
    };
    match manager.get(&miss) {
        CacheLookup::Miss => {}
        other => return Err(format!("expected cache miss, got {other:?}")),
    }
    Ok(WorkloadOutcome {
        logical_output_units: hit_len as u64 + 1,
        evidence_refs: vec![
            "benchmark:cache_subsystem/cache_lookup/hit/in_memory".to_owned(),
            "benchmark:cache_subsystem/cache_lookup/miss/in_memory".to_owned(),
        ],
    })
}

fn workload_durability_sidecar_round_trip(
    paths: &MemoryGateOutputPaths,
) -> Result<WorkloadOutcome, String> {
    let payload = json!({
        "schema_version": "frankenjax.memory-durability-probe.v1",
        "bead_id": "frankenjax-cstq.4",
        "payload": "rss gate durability smoke artifact",
    });
    write_json(&paths.durability_probe, &payload).map_err(|err| err.to_string())?;
    let manifest = encode_artifact_to_sidecar(
        &paths.durability_probe,
        &paths.durability_sidecar,
        &SidecarConfig::default(),
    )
    .map_err(|err| format!("sidecar generation failed: {err}"))?;
    let scrub = scrub_sidecar(
        &paths.durability_sidecar,
        &paths.durability_probe,
        &paths.durability_scrub,
    )
    .map_err(|err| format!("sidecar scrub failed: {err}"))?;
    if !scrub.decoded_matches_expected {
        return Err("sidecar scrub did not match expected artifact".to_owned());
    }
    let proof = generate_decode_proof(
        &paths.durability_sidecar,
        &paths.durability_probe,
        &paths.durability_proof,
        1,
    )
    .map_err(|err| format!("decode proof failed: {err}"))?;
    if !proof.recovered {
        return Err("decode proof did not recover after source drop".to_owned());
    }
    Ok(WorkloadOutcome {
        logical_output_units: manifest.total_symbols as u64 + proof.dropped_symbols.len() as u64,
        evidence_refs: vec![
            repo_relative_artifact(&paths.durability_probe),
            repo_relative_artifact(&paths.durability_sidecar),
            repo_relative_artifact(&paths.durability_scrub),
            repo_relative_artifact(&paths.durability_proof),
        ],
    })
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
            format!("memory-gate-{}-{idx}", transform.as_str()),
        );
    }
    DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger,
        args,
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    }
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

    #[test]
    fn proc_status_parser_converts_kib_to_bytes() {
        let raw = "Name:\tfj\nVmRSS:\t   1234 kB\nVmHWM:\t   5678 kB\n";
        assert_eq!(parse_proc_status_kib(raw, "VmRSS:"), Some(1_263_616));
        assert_eq!(parse_proc_status_kib(raw, "VmHWM:"), Some(5_814_272));
        assert_eq!(parse_proc_status_kib(raw, "Missing:"), None);
    }

    #[test]
    fn validator_rejects_unmeasured_rows() {
        let mut report = MemoryPerformanceReport {
            schema_version: MEMORY_PERFORMANCE_REPORT_SCHEMA_VERSION.to_owned(),
            bead_id: "frankenjax-cstq.4".to_owned(),
            generated_at_unix_ms: 0,
            status: "pass".to_owned(),
            measurement_policy: "test".to_owned(),
            measurement_backend: "linux_procfs_status_vm_hwm".to_owned(),
            peak_rss_budget_bytes: DEFAULT_PEAK_RSS_BUDGET_BYTES,
            workloads: REQUIRED_WORKLOADS
                .iter()
                .map(|id| MemoryWorkloadMeasurement {
                    phase_id: "test".to_owned(),
                    workload_id: (*id).to_owned(),
                    status: "pass".to_owned(),
                    iterations: 1,
                    measurement_backend: "linux_procfs_status_vm_hwm".to_owned(),
                    peak_rss_budget_bytes: DEFAULT_PEAK_RSS_BUDGET_BYTES,
                    rss_before_bytes: Some(1),
                    rss_after_bytes: Some(2),
                    peak_rss_bytes: Some(2),
                    delta_rss_bytes: Some(1),
                    logical_output_units: 1,
                    evidence_refs: vec!["benchmark:test".to_owned()],
                    replay_command: "replay".to_owned(),
                    notes: Vec::new(),
                    error: None,
                })
                .collect(),
            issues: Vec::new(),
            artifact_refs: Vec::new(),
            replay_command: "replay".to_owned(),
        };
        assert!(validate_memory_performance_report(&report).is_empty());
        report.workloads[0].measurement_backend = "not_measured".to_owned();
        let issues = validate_memory_performance_report(&report);
        assert!(
            issues
                .iter()
                .any(|issue| issue.code == "memory_not_measured")
        );
    }

    #[test]
    fn current_report_passes_on_linux_procfs() {
        let temp = tempfile::tempdir().expect("tempdir");
        let report = build_memory_performance_report(temp.path());
        assert_eq!(
            report.schema_version,
            MEMORY_PERFORMANCE_REPORT_SCHEMA_VERSION
        );
        assert_eq!(report.bead_id, "frankenjax-cstq.4");
        assert_eq!(report.workloads.len(), REQUIRED_WORKLOADS.len());
        assert_eq!(report.status, "pass", "report: {report:#?}");
        assert!(report.issues.is_empty());
    }
}
