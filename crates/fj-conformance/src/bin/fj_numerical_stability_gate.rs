#![forbid(unsafe_code)]

use fj_conformance::e2e_log::{
    E2EArtifactRef, E2ECompatibilityMode, E2EErrorClass, E2EForensicLogV1, E2ELogStatus,
    E2ETolerancePolicy, artifact_sha256_hex, write_e2e_log,
};
use fj_conformance::numerical_stability::{
    NumericalStabilityOutputPaths, numerical_stability_summary_json,
    write_numerical_stability_outputs,
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

        let defaults = NumericalStabilityOutputPaths::for_root(&root);
        Ok(Self {
            root,
            report: report.unwrap_or(defaults.report),
            markdown: markdown.unwrap_or(defaults.markdown),
            e2e: e2e.unwrap_or(defaults.e2e),
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

    let report = match write_numerical_stability_outputs(&args.root, &args.report, &args.markdown) {
        Ok(report) => report,
        Err(err) => {
            eprintln!("error: failed to write numerical stability outputs: {err}");
            std::process::exit(1);
        }
    };
    if let Err(err) = write_forensic_log(&args, &report) {
        eprintln!("error: failed to write E2E forensic log: {err}");
        std::process::exit(1);
    }

    println!(
        "numerical stability gate: status={} rows={} policies={} platforms={} deterministic_rows={} issues={}",
        report.status,
        report.summary.row_count,
        report.summary.tolerance_policy_count,
        report.summary.platform_fingerprint_count,
        report.summary.deterministic_replay_rows,
        report.issues.len(),
    );
    match serde_json::to_string_pretty(&numerical_stability_summary_json(&report)) {
        Ok(raw) => println!("{raw}"),
        Err(err) => eprintln!("warning: failed to render summary json: {err}"),
    }

    if args.enforce && report.status != "pass" {
        std::process::exit(1);
    }
}

fn write_forensic_log(
    args: &Args,
    report: &fj_conformance::numerical_stability::NumericalStabilityReport,
) -> Result<(), std::io::Error> {
    let status = if report.status == "pass" {
        E2ELogStatus::Pass
    } else {
        E2ELogStatus::Fail
    };
    let mut log = E2EForensicLogV1::new(
        "frankenjax-cstq.20",
        "e2e_numerical_stability_gate",
        "fj_numerical_stability_gate",
        std::env::args().collect(),
        args.root.display().to_string(),
        E2ECompatibilityMode::Strict,
        status,
    );
    log.fixture_ids = report.rows.iter().map(|row| row.case_id.clone()).collect();
    log.oracle_ids = report
        .rows
        .iter()
        .flat_map(|row| row.artifact_refs.iter().cloned())
        .collect();
    log.inputs = serde_json::json!({
        "required_stability_families": report.required_stability_families,
        "report_path": repo_relative(&args.root, &args.report),
        "platform_fingerprints": report.platform_fingerprints,
    });
    log.expected = serde_json::json!({
        "gate_status": "pass",
        "required_stability_families": report.summary.required_family_count,
        "stale_rows": 0,
        "regression_rows": 0,
        "unstable_rows": 0,
    });
    log.actual = numerical_stability_summary_json(report);
    log.tolerance = E2ETolerancePolicy {
        policy_id: "exact_numerical_stability_gate".to_owned(),
        atol: None,
        rtol: None,
        ulp: None,
        notes: Some(
            "Gate status is exact; per-row numeric tolerances live in the numerical_stability_matrix report."
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
            "numerical_stability_gate".to_owned()
        },
    };
    log.artifacts = vec![
        artifact_ref(&args.root, &args.report, "numerical_stability_report")?,
        artifact_ref(&args.root, &args.markdown, "numerical_stability_markdown")?,
    ];
    log.replay_command = stable_replay_command(args);
    if status.requires_failure_summary() {
        log.failure_summary = Some(
            "numerical stability gate found missing tolerance policy, platform metadata, non-finite classification, replay command, artifact binding, or regression/stale evidence"
                .to_owned(),
        );
    }
    log.metadata.insert(
        "bead".to_owned(),
        serde_json::json!({
            "id": "frankenjax-cstq.20",
            "title": "Prove numerical stability and platform determinism gates"
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

fn stable_replay_command(args: &Args) -> String {
    let defaults = NumericalStabilityOutputPaths::for_root(&args.root);
    let mut parts = vec!["./scripts/run_numerical_stability_gate.sh".to_owned()];
    if args.report != defaults.report {
        push_flag_value(
            &mut parts,
            "--report",
            repo_relative(&args.root, &args.report),
        );
    }
    if args.markdown != defaults.markdown {
        push_flag_value(
            &mut parts,
            "--markdown",
            repo_relative(&args.root, &args.markdown),
        );
    }
    if args.e2e != defaults.e2e {
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

fn repo_relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root).map_or_else(
        |_| path.display().to_string(),
        |stripped| stripped.display().to_string(),
    )
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
        "Usage: fj_numerical_stability_gate [--root <repo>] [--report <json>] [--markdown <md>] [--e2e <json>] [--enforce]"
    );
}
