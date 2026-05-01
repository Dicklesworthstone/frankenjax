#![forbid(unsafe_code)]

use fj_conformance::architecture_decision::{
    ArchitectureBoundaryReport, architecture_boundary_summary_json,
    write_architecture_boundary_outputs,
};
use fj_conformance::e2e_log::{
    E2EArtifactRef, E2ECompatibilityMode, E2EErrorClass, E2EForensicLogV1, E2ELogStatus,
    E2ETolerancePolicy, artifact_sha256_hex, write_e2e_log,
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
                    markdown = Some(PathBuf::from(next_value(&mut iter, "--markdown")?))
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
                root.join("artifacts/conformance/architecture_boundary_decision.v1.json")
            }),
            markdown: markdown.unwrap_or_else(|| {
                root.join("artifacts/conformance/architecture_boundary_decision.v1.md")
            }),
            e2e: e2e.unwrap_or_else(|| {
                root.join("artifacts/e2e/e2e_architecture_boundary_gate.e2e.json")
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

    let report = match write_architecture_boundary_outputs(&args.root, &args.report, &args.markdown)
    {
        Ok(report) => report,
        Err(err) => {
            eprintln!("error: failed to write architecture boundary outputs: {err}");
            std::process::exit(1);
        }
    };

    if let Err(err) = write_forensic_log(&args, &report) {
        eprintln!("error: failed to write E2E forensic log: {err}");
        std::process::exit(1);
    }

    println!(
        "architecture boundary gate: status={} crates={} edges={} decisions={} issues={}",
        report.status,
        report.crate_count,
        report.normal_edges.len(),
        report.decisions.len(),
        report.issues.len()
    );
    for decision in &report.decisions {
        println!(
            "  {:?} {}: {}",
            decision.decision, decision.boundary_id, decision.user_outcome
        );
    }

    if args.enforce && report.status != "pass" {
        std::process::exit(1);
    }
}

fn write_forensic_log(
    args: &Args,
    report: &ArchitectureBoundaryReport,
) -> Result<(), std::io::Error> {
    let status = if report.status == "pass" {
        E2ELogStatus::Pass
    } else {
        E2ELogStatus::Fail
    };
    let mut log = E2EForensicLogV1::new(
        "frankenjax-cstq.12",
        "e2e_architecture_boundary_gate",
        "fj_architecture_boundary_gate",
        std::env::args().collect(),
        args.root.display().to_string(),
        E2ECompatibilityMode::Strict,
        status,
    );
    log.fixture_ids = report
        .decisions
        .iter()
        .map(|decision| decision.boundary_id.clone())
        .collect();
    log.oracle_ids = vec![
        "EXISTING_JAX_STRUCTURE.md".to_owned(),
        "PROPOSED_ARCHITECTURE.md".to_owned(),
        "cargo metadata --no-deps".to_owned(),
    ];
    log.transform_stack = vec![
        "architecture".to_owned(),
        "facade".to_owned(),
        "lowering".to_owned(),
        "runtime_backend".to_owned(),
    ];
    log.inputs = serde_json::json!({
        "workspace_root": report.workspace_root,
        "crate_count": report.crate_count,
        "normal_edge_count": report.normal_edges.len(),
        "decision_ids": report.decisions.iter().map(|decision| decision.boundary_id.clone()).collect::<Vec<_>>(),
    });
    log.expected = serde_json::json!({
        "gate_status": "pass",
        "dependency_cycles": 0,
        "layer_violations": 0,
        "missing_decision_fields": 0,
    });
    log.actual = architecture_boundary_summary_json(report);
    log.tolerance = E2ETolerancePolicy {
        policy_id: "exact_architecture_boundary_gate".to_owned(),
        atol: None,
        rtol: None,
        ulp: None,
        notes: Some(
            "workspace graph, boundary owners, rejected alternatives, and guardrails are exact"
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
            "architecture_boundary_gate".to_owned()
        },
    };
    log.artifacts = vec![
        artifact_ref(&args.root, &args.report, "architecture_boundary_report")?,
        artifact_ref(&args.root, &args.markdown, "architecture_boundary_markdown")?,
    ];
    log.replay_command = stable_replay_command(args);
    if status.requires_failure_summary() {
        log.failure_summary = Some(
            "architecture boundary gate found dependency, owner, or decision-record issues; see report artifact"
                .to_owned(),
        );
    }
    log.metadata.insert(
        "bead".to_owned(),
        serde_json::json!({
            "id": "frankenjax-cstq.12",
            "title": "Decide transform/lowering/API facade crate boundary extraction"
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
    let mut parts = vec!["./scripts/run_architecture_boundary_gate.sh".to_owned()];
    let default_report = args
        .root
        .join("artifacts/conformance/architecture_boundary_decision.v1.json");
    let default_markdown = args
        .root
        .join("artifacts/conformance/architecture_boundary_decision.v1.md");
    let default_e2e = args
        .root
        .join("artifacts/e2e/e2e_architecture_boundary_gate.e2e.json");

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

fn next_value(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    let value = iter
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))?;
    if value.starts_with('-') {
        return Err(format!("{flag} requires a value, got flag `{value}`"));
    }
    Ok(value)
}

fn print_usage() {
    eprintln!(
        "Usage: fj_architecture_boundary_gate [--root <repo>] [--report <json>] [--markdown <md>] [--e2e <json>] [--enforce]"
    );
}
