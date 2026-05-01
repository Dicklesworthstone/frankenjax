#![forbid(unsafe_code)]

use fj_conformance::e2e_log::{
    E2EArtifactRef, E2ECompatibilityMode, E2EErrorClass, E2EForensicLogV1, E2ELogStatus,
    E2ETolerancePolicy, artifact_sha256_hex, write_e2e_log,
};
use fj_conformance::oracle_recapture::{
    build_oracle_recapture_matrix, oracle_drift_report, oracle_recapture_markdown,
    oracle_recapture_summary_json, read_oracle_recapture_matrix, validate_oracle_drift_report,
    write_oracle_json,
};
use serde_json::json;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
struct Args {
    root: PathBuf,
    matrix: PathBuf,
    drift: PathBuf,
    markdown: PathBuf,
    e2e: PathBuf,
    baseline: Option<PathBuf>,
    require_baseline: bool,
    enforce: bool,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut root = std::env::current_dir().map_err(|err| format!("current dir: {err}"))?;
        let mut matrix = None;
        let mut drift = None;
        let mut markdown = None;
        let mut e2e = None;
        let mut baseline = None;
        let mut require_baseline = false;
        let mut enforce = false;

        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--root" => root = PathBuf::from(next_value(&mut iter, "--root")?),
                "--matrix" => matrix = Some(PathBuf::from(next_value(&mut iter, "--matrix")?)),
                "--drift" => drift = Some(PathBuf::from(next_value(&mut iter, "--drift")?)),
                "--markdown" => {
                    markdown = Some(PathBuf::from(next_value(&mut iter, "--markdown")?));
                }
                "--e2e" => e2e = Some(PathBuf::from(next_value(&mut iter, "--e2e")?)),
                "--baseline" => {
                    baseline = Some(PathBuf::from(next_value(&mut iter, "--baseline")?))
                }
                "--require-baseline" => require_baseline = true,
                "--enforce" => enforce = true,
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument `{arg}`")),
            }
        }

        Ok(Self {
            matrix: matrix.unwrap_or_else(|| {
                root.join("artifacts/conformance/oracle_recapture_matrix.v1.json")
            }),
            drift: drift
                .unwrap_or_else(|| root.join("artifacts/conformance/oracle_drift_report.v1.json")),
            markdown: markdown.unwrap_or_else(|| {
                root.join("artifacts/conformance/oracle_recapture_matrix.v1.md")
            }),
            e2e: e2e
                .unwrap_or_else(|| root.join("artifacts/e2e/e2e_oracle_recapture_gate.e2e.json")),
            root,
            baseline,
            require_baseline,
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

    let matrix = build_oracle_recapture_matrix(&args.root);
    let baseline = match args.baseline.as_deref() {
        Some(path) => match read_oracle_recapture_matrix(path) {
            Ok(matrix) => Some(matrix),
            Err(err) => {
                eprintln!("error: failed to read baseline {}: {err}", path.display());
                std::process::exit(1);
            }
        },
        None => None,
    };
    let baseline_path = args
        .baseline
        .as_ref()
        .map(|path| repo_relative(&args.root, path));
    let drift = oracle_drift_report(
        &matrix,
        baseline.as_ref(),
        baseline_path,
        args.require_baseline,
    );
    let drift_issues = validate_oracle_drift_report(&drift);
    let markdown = oracle_recapture_markdown(&matrix, &drift);

    if let Err(err) = write_outputs(&args, &matrix, &drift, &markdown) {
        eprintln!("error: failed to write oracle recapture outputs: {err}");
        std::process::exit(1);
    }

    if let Err(err) = write_forensic_log(&args, &matrix, &drift) {
        eprintln!("error: failed to write E2E forensic log: {err}");
        std::process::exit(1);
    }

    println!(
        "oracle recapture gate: matrix={} drift={} cases={}/{} issues={}",
        matrix.status,
        drift.gate_status,
        matrix.actual_total_cases,
        matrix.expected_total_cases,
        drift_issues.len()
    );
    for issue in &drift.issues {
        println!(
            "  {} {}: {}",
            issue.code,
            issue.family_id.as_deref().unwrap_or("-"),
            issue.message
        );
    }

    if args.enforce && (matrix.status != "pass" || drift.gate_status != "pass") {
        std::process::exit(1);
    }
}

fn write_outputs(
    args: &Args,
    matrix: &fj_conformance::oracle_recapture::OracleRecaptureMatrix,
    drift: &fj_conformance::oracle_recapture::OracleDriftReport,
    markdown: &str,
) -> Result<(), std::io::Error> {
    write_oracle_json(&args.matrix, matrix)?;
    write_oracle_json(&args.drift, drift)?;
    if let Some(parent) = args.markdown.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.markdown, markdown)
}

fn write_forensic_log(
    args: &Args,
    matrix: &fj_conformance::oracle_recapture::OracleRecaptureMatrix,
    drift: &fj_conformance::oracle_recapture::OracleDriftReport,
) -> Result<(), std::io::Error> {
    let status = if matrix.status == "pass" && drift.gate_status == "pass" {
        E2ELogStatus::Pass
    } else {
        E2ELogStatus::Fail
    };
    let mut log = E2EForensicLogV1::new(
        "frankenjax-cstq.1",
        "e2e_oracle_recapture_gate",
        "fj_oracle_recapture_gate",
        std::env::args().collect(),
        args.root.display().to_string(),
        E2ECompatibilityMode::Strict,
        status,
    );
    log.fixture_ids = matrix
        .rows
        .iter()
        .map(|row| row.family_id.clone())
        .collect();
    log.oracle_ids = matrix
        .rows
        .iter()
        .map(|row| format!("{}:{}", row.family_id, row.oracle_version))
        .collect();
    log.expected = json!({
        "total_cases": matrix.expected_total_cases,
        "families": matrix.required_families,
        "gate_status": "pass"
    });
    log.actual = oracle_recapture_summary_json(matrix, drift);
    log.tolerance = E2ETolerancePolicy {
        policy_id: "exact_case_count_and_hash".to_owned(),
        atol: None,
        rtol: None,
        ulp: None,
        notes: Some(
            "fixture case counts, schema versions, oracle versions, and hashes are exact"
                .to_owned(),
        ),
    };
    log.error = E2EErrorClass {
        expected: None,
        actual: if drift.issues.is_empty() {
            None
        } else {
            Some(
                drift
                    .issues
                    .iter()
                    .map(|issue| issue.code.as_str())
                    .collect::<Vec<_>>()
                    .join(","),
            )
        },
        taxonomy_class: if drift.issues.is_empty() {
            "none".to_owned()
        } else {
            "oracle_recapture_drift".to_owned()
        },
    };
    log.artifacts = vec![
        artifact_ref(&args.root, &args.matrix, "oracle_recapture_matrix")?,
        artifact_ref(&args.root, &args.drift, "oracle_drift_report")?,
        artifact_ref(&args.root, &args.markdown, "oracle_recapture_markdown")?,
    ];
    log.replay_command = stable_replay_command(args);
    if status.requires_failure_summary() {
        log.failure_summary = Some(format!(
            "oracle recapture gate found {} issue(s); see drift report",
            drift.issues.len()
        ));
    }
    log.metadata.insert(
        "bead".to_owned(),
        json!({
            "id": "frankenjax-cstq.1",
            "title": "Build oracle parity recapture matrix and drift gate"
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
    let mut parts = vec!["./scripts/run_oracle_recapture_gate.sh".to_owned()];
    let default_matrix = args
        .root
        .join("artifacts/conformance/oracle_recapture_matrix.v1.json");
    let default_drift = args
        .root
        .join("artifacts/conformance/oracle_drift_report.v1.json");
    let default_markdown = args
        .root
        .join("artifacts/conformance/oracle_recapture_matrix.v1.md");
    let default_e2e = args
        .root
        .join("artifacts/e2e/e2e_oracle_recapture_gate.e2e.json");

    if args.matrix != default_matrix {
        push_flag_value(
            &mut parts,
            "--matrix",
            repo_relative(&args.root, &args.matrix),
        );
    }
    if args.drift != default_drift {
        push_flag_value(
            &mut parts,
            "--drift",
            repo_relative(&args.root, &args.drift),
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
    if let Some(baseline) = &args.baseline {
        push_flag_value(
            &mut parts,
            "--baseline",
            repo_relative(&args.root, baseline),
        );
    }
    if args.require_baseline {
        parts.push("--require-baseline".to_owned());
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
        "Usage: fj_oracle_recapture_gate [--root <repo>] [--matrix <json>] [--drift <json>] [--markdown <md>] [--e2e <json>] [--baseline <matrix.json>] [--require-baseline] [--enforce]"
    );
}
