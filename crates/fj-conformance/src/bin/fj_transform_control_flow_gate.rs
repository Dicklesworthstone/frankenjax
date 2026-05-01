#![forbid(unsafe_code)]

use fj_conformance::e2e_log::{
    E2EAllocationCounters, E2EArtifactRef, E2ECompatibilityMode, E2EErrorClass, E2EForensicLogV1,
    E2ELogStatus, E2ETolerancePolicy, artifact_sha256_hex, write_e2e_log,
};
use fj_conformance::transform_control_flow::{
    TransformControlFlowReport, transform_control_flow_summary_json,
    write_transform_control_flow_outputs,
};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
struct Args {
    root: PathBuf,
    report: PathBuf,
    markdown: PathBuf,
    e2e: PathBuf,
    enforce: bool,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut root = std::env::current_dir().map_err(|err| format!("current dir: {err}"))?;
        let mut report = None;
        let mut markdown = None;
        let mut e2e = None;
        let mut enforce = false;

        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--root" => root = PathBuf::from(next_value(&mut iter, "--root")?),
                "--report" => report = Some(PathBuf::from(next_value(&mut iter, "--report")?)),
                "--markdown" => {
                    markdown = Some(PathBuf::from(next_value(&mut iter, "--markdown")?));
                }
                "--e2e" => e2e = Some(PathBuf::from(next_value(&mut iter, "--e2e")?)),
                "--enforce" => enforce = true,
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument `{arg}`")),
            }
        }

        Ok(Self {
            report: report.unwrap_or_else(|| {
                root.join("artifacts/conformance/transform_control_flow_matrix.v1.json")
            }),
            markdown: markdown.unwrap_or_else(|| {
                root.join("artifacts/conformance/transform_control_flow_matrix.v1.md")
            }),
            e2e: e2e.unwrap_or_else(|| {
                root.join("artifacts/e2e/e2e_transform_control_flow_gate.e2e.json")
            }),
            root,
            enforce,
        })
    }
}

fn main() {
    let args = match Args::parse() {
        Ok(args) => args,
        Err(err) => {
            eprintln!("error: {err}");
            print_usage();
            std::process::exit(2);
        }
    };

    let report =
        match write_transform_control_flow_outputs(&args.root, &args.report, &args.markdown) {
            Ok(report) => report,
            Err(err) => {
                eprintln!("error: failed to write transform control-flow outputs: {err}");
                std::process::exit(1);
            }
        };

    if let Err(err) = write_forensic_log(&args, &report) {
        eprintln!("error: failed to write E2E forensic log: {err}");
        std::process::exit(1);
    }

    println!(
        "transform control-flow gate: status={} cases={} supported={} fail_closed={} sentinels={} issues={}",
        report.status,
        report.cases.len(),
        report.supported_rows,
        report.fail_closed_rows,
        report.performance_sentinels.len(),
        report.issues.len()
    );
    for case in &report.cases {
        println!(
            "  case {}: support={} status={} stack={} error={:?}",
            case.case_id,
            case.support_status,
            case.status,
            case.transform_stack.join(">"),
            case.error_class
        );
    }
    for sentinel in &report.performance_sentinels {
        println!(
            "  perf {}: status={} p50={} p95={} p99={} peak_rss={:?}",
            sentinel.workload_id,
            sentinel.status,
            sentinel.p50_ns,
            sentinel.p95_ns,
            sentinel.p99_ns,
            sentinel.peak_rss_bytes
        );
    }

    if args.enforce && report.status != "pass" {
        std::process::exit(1);
    }
}

fn write_forensic_log(
    args: &Args,
    report: &TransformControlFlowReport,
) -> Result<(), std::io::Error> {
    let status = if report.status == "pass" {
        E2ELogStatus::Pass
    } else {
        E2ELogStatus::Fail
    };
    let max_peak_rss_bytes = report
        .performance_sentinels
        .iter()
        .filter_map(|sentinel| sentinel.peak_rss_bytes)
        .max();
    let mut log = E2EForensicLogV1::new(
        "frankenjax-cstq.2",
        "e2e_transform_control_flow_gate",
        "fj_transform_control_flow_gate",
        std::env::args().collect(),
        args.root.display().to_string(),
        E2ECompatibilityMode::Strict,
        status,
    );
    log.fixture_ids = report
        .cases
        .iter()
        .map(|case| case.case_id.clone())
        .collect();
    log.oracle_ids = report
        .cases
        .iter()
        .map(|case| case.oracle_fixture_id.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    log.transform_stack = vec![
        "jit".to_owned(),
        "grad".to_owned(),
        "vmap".to_owned(),
        "value_and_grad".to_owned(),
        "jacobian".to_owned(),
        "hessian".to_owned(),
        "cond".to_owned(),
        "scan".to_owned(),
        "while".to_owned(),
        "switch".to_owned(),
    ];
    log.inputs = serde_json::json!({
        "workspace_root": args.root.display().to_string(),
        "matrix_policy": report.matrix_policy,
        "case_ids": report.cases.iter().map(|case| case.case_id.clone()).collect::<Vec<_>>(),
        "performance_workloads": report.performance_sentinels.iter().map(|sentinel| sentinel.workload_id.clone()).collect::<Vec<_>>(),
    });
    log.expected = serde_json::json!({
        "gate_status": "pass",
        "required_case_count": report.cases.len(),
        "required_performance_sentinels": report.performance_sentinels.len(),
        "unsupported_rows": "typed_fail_closed",
    });
    log.actual = transform_control_flow_summary_json(report);
    log.tolerance = E2ETolerancePolicy {
        policy_id: "exact_matrix_coverage_with_declared_numeric_tolerances".to_owned(),
        atol: Some(1e-6),
        rtol: Some(0.0),
        ulp: None,
        notes: Some(
            "Each matrix row records its own exact or numeric comparison; unsupported rows require exact typed error classes"
                .to_owned(),
        ),
    };
    log.error = E2EErrorClass {
        expected: None,
        actual: if report.status == "pass" {
            None
        } else {
            Some(
                report
                    .issues
                    .iter()
                    .map(|issue| issue.code.as_str())
                    .collect::<Vec<_>>()
                    .join(","),
            )
        },
        taxonomy_class: if report.status == "pass" {
            "none".to_owned()
        } else {
            "transform_control_flow_matrix_gate".to_owned()
        },
    };
    log.allocations = E2EAllocationCounters {
        allocation_count: None,
        allocated_bytes: None,
        peak_rss_bytes: max_peak_rss_bytes,
        measurement_backend: report
            .performance_sentinels
            .iter()
            .find_map(|sentinel| {
                (sentinel.measurement_backend != "unavailable")
                    .then(|| sentinel.measurement_backend.clone())
            })
            .unwrap_or_else(|| "unavailable".to_owned()),
    };
    log.artifacts = vec![
        artifact_ref(
            &args.root,
            &args.report,
            "transform_control_flow_matrix_report",
        )?,
        artifact_ref(
            &args.root,
            &args.markdown,
            "transform_control_flow_matrix_markdown",
        )?,
    ];
    log.replay_command = std::env::args().collect::<Vec<_>>().join(" ");
    if status.requires_failure_summary() {
        log.failure_summary = Some(
            "transform control-flow matrix gate found a missing row, failed supported case, unsupported row without typed fail-closed behavior, or missing performance evidence"
                .to_owned(),
        );
    }
    log.metadata.insert(
        "bead".to_owned(),
        serde_json::json!({
            "id": "frankenjax-cstq.2",
            "title": "Close advanced transform-control-flow composition parity"
        }),
    );
    write_e2e_log(&args.e2e, &log)
}

fn artifact_ref(root: &Path, path: &Path, kind: &str) -> Result<E2EArtifactRef, std::io::Error> {
    Ok(E2EArtifactRef {
        kind: kind.to_owned(),
        path: repo_relative(root, path),
        sha256: artifact_sha256_hex(path)?,
        required: true,
    })
}

fn repo_relative(root: &Path, path: &Path) -> String {
    match path.strip_prefix(root) {
        Ok(path) => path.display().to_string(),
        Err(_) => path.display().to_string(),
    }
}

fn next_value(
    iter: &mut impl Iterator<Item = String>,
    flag: &'static str,
) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn print_usage() {
    eprintln!(
        "Usage: fj_transform_control_flow_gate [--root <dir>] [--report <json>] [--markdown <md>] [--e2e <json>] [--enforce]"
    );
}
