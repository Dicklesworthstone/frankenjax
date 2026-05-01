#![forbid(unsafe_code)]

use fj_conformance::cache_lifecycle::{
    cache_lifecycle_summary_json, write_cache_lifecycle_outputs,
};
use fj_conformance::e2e_log::{
    E2EArtifactRef, E2ECompatibilityMode, E2EErrorClass, E2EForensicLogV1, E2ELogStatus,
    E2ETolerancePolicy, artifact_sha256_hex, write_e2e_log,
};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
struct Args {
    root: PathBuf,
    ledger: PathBuf,
    report: PathBuf,
    markdown: PathBuf,
    e2e: PathBuf,
    enforce: bool,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut root = std::env::current_dir().map_err(|err| format!("current dir: {err}"))?;
        let mut ledger = None;
        let mut report = None;
        let mut markdown = None;
        let mut e2e = None;
        let mut enforce = false;

        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--root" => root = PathBuf::from(next_value(&mut iter, "--root")?),
                "--ledger" => ledger = Some(PathBuf::from(next_value(&mut iter, "--ledger")?)),
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
            ledger: ledger.unwrap_or_else(|| {
                root.join("artifacts/conformance/cache_legacy_parity_ledger.v1.json")
            }),
            report: report.unwrap_or_else(|| {
                root.join("artifacts/conformance/cache_lifecycle_report.v1.json")
            }),
            markdown: markdown.unwrap_or_else(|| {
                root.join("artifacts/conformance/cache_legacy_parity_ledger.v1.md")
            }),
            e2e: e2e
                .unwrap_or_else(|| root.join("artifacts/e2e/e2e_cache_lifecycle_gate.e2e.json")),
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

    let (ledger, report) =
        match write_cache_lifecycle_outputs(&args.root, &args.ledger, &args.report, &args.markdown)
        {
            Ok(outputs) => outputs,
            Err(err) => {
                eprintln!("error: failed to write cache lifecycle outputs: {err}");
                std::process::exit(1);
            }
        };

    if let Err(err) = write_forensic_log(&args, &ledger, &report) {
        eprintln!("error: failed to write E2E forensic log: {err}");
        std::process::exit(1);
    }

    println!(
        "cache lifecycle gate: status={} ledger_rows={} scenarios={} ledger_issues={}",
        report.status,
        ledger.rows.len(),
        report.scenarios.len(),
        report.ledger_issues.len()
    );
    for scenario in &report.scenarios {
        println!(
            "  {} {}: {}",
            scenario.status, scenario.scenario_id, scenario.actual
        );
    }

    if args.enforce && report.status != "pass" {
        std::process::exit(1);
    }
}

fn write_forensic_log(
    args: &Args,
    ledger: &fj_cache::legacy_parity::CacheLegacyParityLedger,
    report: &fj_conformance::cache_lifecycle::CacheLifecycleReport,
) -> Result<(), std::io::Error> {
    let status = if report.status == "pass" {
        E2ELogStatus::Pass
    } else {
        E2ELogStatus::Fail
    };
    let mut log = E2EForensicLogV1::new(
        "frankenjax-cstq.6",
        "e2e_cache_lifecycle_gate",
        "fj_cache_lifecycle_gate",
        std::env::args().collect(),
        args.root.display().to_string(),
        E2ECompatibilityMode::Strict,
        status,
    );
    log.fixture_ids = report
        .scenarios
        .iter()
        .map(|scenario| scenario.scenario_id.clone())
        .collect();
    log.oracle_ids = ledger
        .rows
        .iter()
        .map(|row| row.legacy_anchor.clone())
        .collect();
    log.transform_stack = vec![
        "jit".to_owned(),
        "grad,vmap".to_owned(),
        "vmap,grad".to_owned(),
    ];
    log.inputs = serde_json::json!({
        "ledger_schema": ledger.schema_version,
        "key_namespace": ledger.key_namespace,
        "scenario_ids": report.scenarios.iter().map(|scenario| scenario.scenario_id.clone()).collect::<Vec<_>>(),
    });
    log.expected = serde_json::json!({
        "gate_status": "pass",
        "ledger_issues": 0,
        "all_scenarios": "pass",
    });
    log.actual = cache_lifecycle_summary_json(ledger, report);
    log.tolerance = E2ETolerancePolicy {
        policy_id: "exact_cache_key_lifecycle".to_owned(),
        atol: None,
        rtol: None,
        ulp: None,
        notes: Some(
            "cache key equality, inequality, rejection, corruption, and write-blocking outcomes are exact"
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
                    .scenarios
                    .iter()
                    .filter(|scenario| scenario.status != "pass")
                    .map(|scenario| scenario.scenario_id.as_str())
                    .collect::<Vec<_>>()
                    .join(","),
            )
        },
        taxonomy_class: if report.status == "pass" {
            "none".to_owned()
        } else {
            "cache_lifecycle_gate".to_owned()
        },
    };
    log.artifacts = vec![
        artifact_ref(&args.root, &args.ledger, "cache_legacy_parity_ledger")?,
        artifact_ref(&args.root, &args.report, "cache_lifecycle_report")?,
        artifact_ref(&args.root, &args.markdown, "cache_legacy_parity_markdown")?,
    ];
    log.replay_command = std::env::args().collect::<Vec<_>>().join(" ");
    if status.requires_failure_summary() {
        log.failure_summary = Some(
            "cache lifecycle gate found ledger or scenario failures; see report artifact"
                .to_owned(),
        );
    }
    log.metadata.insert(
        "bead".to_owned(),
        serde_json::json!({
            "id": "frankenjax-cstq.6",
            "title": "Create cache-key legacy parity ledger and lifecycle conformance"
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

fn next_value(iter: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    iter.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn print_usage() {
    eprintln!(
        "Usage: fj_cache_lifecycle_gate [--root <repo>] [--ledger <json>] [--report <json>] [--markdown <md>] [--e2e <json>] [--enforce]"
    );
}
