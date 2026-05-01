#![forbid(unsafe_code)]

use fj_conformance::e2e_log::{
    E2EAllocationCounters, E2EArtifactRef, E2ECompatibilityMode, E2EErrorClass, E2EForensicLogV1,
    E2ELogStatus, E2ETolerancePolicy, artifact_sha256_hex, write_e2e_log,
};
use fj_conformance::memory_performance::{
    MemoryPerformanceReport, memory_performance_summary_json, write_memory_performance_outputs,
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
                root.join("artifacts/performance/memory_performance_gate.v1.json")
            }),
            markdown: markdown.unwrap_or_else(|| {
                root.join("artifacts/performance/memory_performance_gate.v1.md")
            }),
            e2e: e2e
                .unwrap_or_else(|| root.join("artifacts/e2e/e2e_memory_performance_gate.e2e.json")),
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

    let report = match write_memory_performance_outputs(&args.root, &args.report, &args.markdown) {
        Ok(report) => report,
        Err(err) => {
            eprintln!("error: failed to write memory performance outputs: {err}");
            std::process::exit(1);
        }
    };

    if let Err(err) = write_forensic_log(&args, &report) {
        eprintln!("error: failed to write E2E forensic log: {err}");
        std::process::exit(1);
    }

    println!(
        "memory performance gate: status={} workloads={} issues={} backend={}",
        report.status,
        report.workloads.len(),
        report.issues.len(),
        report.measurement_backend
    );
    for workload in &report.workloads {
        println!(
            "  {} {}: status={} peak_rss={:?} delta_rss={:?}",
            workload.phase_id,
            workload.workload_id,
            workload.status,
            workload.peak_rss_bytes,
            workload.delta_rss_bytes
        );
    }

    if args.enforce && report.status != "pass" {
        std::process::exit(1);
    }
}

fn write_forensic_log(args: &Args, report: &MemoryPerformanceReport) -> Result<(), std::io::Error> {
    let status = if report.status == "pass" {
        E2ELogStatus::Pass
    } else {
        E2ELogStatus::Fail
    };
    let max_peak_rss_bytes = report
        .workloads
        .iter()
        .filter_map(|workload| workload.peak_rss_bytes)
        .max();
    let mut log = E2EForensicLogV1::new(
        "frankenjax-cstq.4",
        "e2e_memory_performance_gate",
        "fj_memory_performance_gate",
        std::env::args().collect(),
        args.root.display().to_string(),
        E2ECompatibilityMode::Strict,
        status,
    );
    log.fixture_ids = report
        .workloads
        .iter()
        .map(|workload| workload.workload_id.clone())
        .collect();
    log.oracle_ids = vec![
        "artifacts/performance/benchmark_baselines_v2_2026-03-12.json".to_owned(),
        "linux:/proc/self/status".to_owned(),
    ];
    log.transform_stack = vec![
        "trace".to_owned(),
        "jit".to_owned(),
        "grad".to_owned(),
        "vmap".to_owned(),
        "lax_fft".to_owned(),
        "lax_linalg".to_owned(),
        "cache".to_owned(),
        "durability".to_owned(),
    ];
    log.inputs = serde_json::json!({
        "workspace_root": args.root.display().to_string(),
        "workload_ids": report.workloads.iter().map(|workload| workload.workload_id.clone()).collect::<Vec<_>>(),
        "measurement_backend": report.measurement_backend,
        "peak_rss_budget_bytes": report.peak_rss_budget_bytes,
    });
    log.expected = serde_json::json!({
        "gate_status": "pass",
        "workloads": 8,
        "measurement_backend": "linux_procfs_status_vm_hwm",
        "allocation_count_policy": "not_synthesized",
    });
    log.actual = memory_performance_summary_json(report);
    log.tolerance = E2ETolerancePolicy {
        policy_id: "rss_budget_with_exact_workload_coverage".to_owned(),
        atol: None,
        rtol: None,
        ulp: None,
        notes: Some(
            "workload coverage and behavior witnesses are exact; RSS values are host measurements bounded by the report budget"
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
            "memory_performance_gate".to_owned()
        },
    };
    log.allocations = E2EAllocationCounters {
        allocation_count: None,
        allocated_bytes: None,
        peak_rss_bytes: max_peak_rss_bytes,
        measurement_backend: report.measurement_backend.clone(),
    };
    log.artifacts = vec![
        artifact_ref(&args.root, &args.report, "memory_performance_report")?,
        artifact_ref(&args.root, &args.markdown, "memory_performance_markdown")?,
    ];
    log.replay_command = stable_replay_command(args);
    if status.requires_failure_summary() {
        log.failure_summary = Some(
            "memory performance gate found missing RSS evidence, failed workload behavior, or a budget breach; see report artifact"
                .to_owned(),
        );
    }
    log.metadata.insert(
        "bead".to_owned(),
        serde_json::json!({
            "id": "frankenjax-cstq.4",
            "title": "Add memory and allocation performance gate"
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

fn stable_replay_command(args: &Args) -> String {
    let mut parts = vec!["./scripts/run_memory_performance_gate.sh".to_owned()];
    let default_report = args
        .root
        .join("artifacts/performance/memory_performance_gate.v1.json");
    let default_markdown = args
        .root
        .join("artifacts/performance/memory_performance_gate.v1.md");
    let default_e2e = args
        .root
        .join("artifacts/e2e/e2e_memory_performance_gate.e2e.json");

    if args.report != default_report {
        push_flag_value(
            &mut parts,
            "--report",
            repo_relative(&args.root, &args.report),
        );
    }
    if args.markdown != default_markdown {
        push_flag_value(
            &mut parts,
            "--markdown",
            repo_relative(&args.root, &args.markdown),
        );
    }
    if args.e2e != default_e2e {
        push_flag_value(&mut parts, "--e2e", repo_relative(&args.root, &args.e2e));
    }
    if args.enforce {
        parts.push("--enforce".to_owned());
    }
    parts.join(" ")
}

fn push_flag_value(parts: &mut Vec<String>, flag: &str, value: String) {
    parts.push(flag.to_owned());
    parts.push(shell_word(&value));
}

fn shell_word(value: &str) -> String {
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '/' | '_' | '-' | '=' | ':'))
    {
        value.to_owned()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
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
        "Usage: fj_memory_performance_gate [--root <path>] [--report <json>] [--markdown <md>] [--e2e <json>] [--enforce]"
    );
}
