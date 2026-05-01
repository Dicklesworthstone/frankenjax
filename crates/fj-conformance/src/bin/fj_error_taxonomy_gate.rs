#![forbid(unsafe_code)]

use fj_conformance::e2e_log::{
    E2EAllocationCounters, E2EArtifactRef, E2ECompatibilityMode, E2EErrorClass, E2EForensicLogV1,
    E2ELogStatus, E2ETolerancePolicy, artifact_sha256_hex, write_e2e_log,
};
use fj_conformance::error_taxonomy::{
    ERROR_TAXONOMY_BEAD_ID, ErrorTaxonomyReport, error_taxonomy_summary_json,
    write_error_taxonomy_outputs,
};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
struct Args {
    root: PathBuf,
    report: PathBuf,
    markdown: PathBuf,
    e2e: PathBuf,
    case_filter: Option<String>,
    enforce: bool,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut root = std::env::current_dir().map_err(|err| format!("current dir: {err}"))?;
        let mut report = None;
        let mut markdown = None;
        let mut e2e = None;
        let mut case_filter = None;
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
                "--case" => case_filter = Some(next_value(&mut iter, "--case")?),
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
                root.join("artifacts/conformance/error_taxonomy_matrix.v1.json")
            }),
            markdown: markdown
                .unwrap_or_else(|| root.join("artifacts/conformance/error_taxonomy_matrix.v1.md")),
            e2e: e2e.unwrap_or_else(|| root.join("artifacts/e2e/e2e_error_taxonomy_gate.e2e.json")),
            root,
            case_filter,
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

    let report = match write_error_taxonomy_outputs(&args.root, &args.report, &args.markdown) {
        Ok(report) => report,
        Err(err) => {
            eprintln!("error: failed to write error taxonomy outputs: {err}");
            std::process::exit(1);
        }
    };

    if let Some(case_id) = args.case_filter.as_deref()
        && !report.cases.iter().any(|case| case.case_id == case_id)
    {
        eprintln!("error: unknown error taxonomy case `{case_id}`");
        std::process::exit(2);
    }

    if let Err(err) = write_forensic_log(&args, &report) {
        eprintln!("error: failed to write E2E forensic log: {err}");
        std::process::exit(1);
    }

    println!(
        "error taxonomy gate: status={} cases={} typed_errors={} success_rows={} divergences={} issues={}",
        report.status,
        report.cases.len(),
        report.coverage.typed_error_count,
        report.coverage.success_rows,
        report.coverage.strict_hardened_divergence_count,
        report.issues.len()
    );
    for case in &report.cases {
        if args
            .case_filter
            .as_deref()
            .is_none_or(|filter| filter == case.case_id)
        {
            println!(
                "  case {}: boundary={} mode={} expected={} actual={} panic={} status={}",
                case.case_id,
                case.boundary,
                case.mode,
                case.expected_error_class,
                case.actual_error_class,
                case.panic_status,
                case.status
            );
        }
    }

    if args.enforce && report.status != "pass" {
        std::process::exit(1);
    }
}

fn write_forensic_log(args: &Args, report: &ErrorTaxonomyReport) -> Result<(), std::io::Error> {
    let status = if report.status == "pass" {
        E2ELogStatus::Pass
    } else {
        E2ELogStatus::Fail
    };
    let mut log = E2EForensicLogV1::new(
        ERROR_TAXONOMY_BEAD_ID,
        "e2e_error_taxonomy_gate",
        "fj_error_taxonomy_gate",
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
    log.oracle_ids = vec!["local:error-taxonomy-matrix.v1".to_owned()];
    log.transform_stack = vec![
        "jit".to_owned(),
        "grad".to_owned(),
        "vmap".to_owned(),
        "cache".to_owned(),
        "durability".to_owned(),
    ];
    log.inputs = serde_json::json!({
        "workspace_root": args.root.display().to_string(),
        "matrix_policy": report.matrix_policy,
        "case_filter": args.case_filter,
        "case_ids": report.cases.iter().map(|case| case.case_id.clone()).collect::<Vec<_>>(),
        "boundaries": report.cases.iter().map(|case| case.boundary.clone()).collect::<std::collections::BTreeSet<_>>(),
    });
    log.expected = serde_json::json!({
        "gate_status": "pass",
        "required_case_count": report.coverage.required_case_count,
        "panic_status": "no_panic",
        "strict_hardened_divergence_allowlist": report.strict_hardened_divergence_allowlist,
    });
    log.actual = error_taxonomy_summary_json(report);
    log.tolerance = E2ETolerancePolicy {
        policy_id: "exact_error_taxonomy_classes_and_replay_metadata".to_owned(),
        atol: None,
        rtol: None,
        ulp: None,
        notes: Some(
            "Every taxonomy row compares exact expected/actual class names and requires replay metadata."
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
            "error_taxonomy_matrix_gate".to_owned()
        },
    };
    log.allocations = E2EAllocationCounters {
        allocation_count: None,
        allocated_bytes: None,
        peak_rss_bytes: None,
        measurement_backend: "not_measured".to_owned(),
    };
    log.artifacts = vec![
        artifact_ref(&args.root, &args.report, "error_taxonomy_matrix_report")?,
        artifact_ref(&args.root, &args.markdown, "error_taxonomy_matrix_markdown")?,
    ];
    log.replay_command = std::env::args().collect::<Vec<_>>().join(" ");
    if status.requires_failure_summary() {
        log.failure_summary = Some(
            "error taxonomy gate found a missing row, mismatched typed class, panic, missing replay metadata, or unapproved strict/hardened divergence"
                .to_owned(),
        );
    }
    log.metadata.insert(
        "bead".to_owned(),
        serde_json::json!({
            "id": ERROR_TAXONOMY_BEAD_ID,
            "title": "Unify cross-crate error taxonomy conformance"
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
        "Usage: fj_error_taxonomy_gate [--root <dir>] [--report <json>] [--markdown <md>] [--e2e <json>] [--case <case_id>] [--enforce]"
    );
}
