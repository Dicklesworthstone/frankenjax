#![forbid(unsafe_code)]

use fj_conformance::e2e_log::{E2ELogValidationReport, validation_report_for_paths};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
struct Args {
    root: PathBuf,
    output: Option<PathBuf>,
    json: bool,
    paths: Vec<PathBuf>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut root = std::env::current_dir().map_err(|err| format!("current dir: {err}"))?;
        let mut output = None;
        let mut json = false;
        let mut paths = Vec::new();

        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--root" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--root requires a path".to_owned())?;
                    root = PathBuf::from(value);
                }
                "--output" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--output requires a path".to_owned())?;
                    output = Some(PathBuf::from(value));
                }
                "--json" => json = true,
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                _ if arg.starts_with('-') => return Err(format!("unknown argument `{arg}`")),
                _ => paths.push(PathBuf::from(arg)),
            }
        }

        if paths.is_empty() {
            paths.push(root.join("artifacts/e2e/e2e_forensic_log_contract_bootstrap.e2e.json"));
        }

        Ok(Self {
            root,
            output,
            json,
            paths,
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

    let paths = expand_inputs(&args.paths);
    let report = validation_report_for_paths(&paths, &args.root);

    if let Some(path) = &args.output
        && let Err(err) = write_report(path, &report)
    {
        eprintln!("error: failed to write {}: {err}", path.display());
        std::process::exit(1);
    }

    if args.json {
        match serde_json::to_string_pretty(&report) {
            Ok(raw) => println!("{raw}"),
            Err(err) => {
                eprintln!("error: failed to serialize validation report: {err}");
                std::process::exit(1);
            }
        }
    } else {
        print_text_report(&report);
    }

    if report.failed > 0 {
        std::process::exit(1);
    }
}

fn print_usage() {
    eprintln!(
        "Usage: fj_e2e_log_check [--root <repo>] [--output <report.json>] [--json] [log-or-dir ...]"
    );
}

fn expand_inputs(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for path in paths {
        if path.is_dir() {
            collect_e2e_logs(path, &mut out);
        } else {
            out.push(path.clone());
        }
    }
    out.sort();
    out
}

fn collect_e2e_logs(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_e2e_logs(&path, out);
        } else if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".e2e.json"))
        {
            out.push(path);
        }
    }
}

fn write_report(path: &Path, report: &E2ELogValidationReport) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let raw = serde_json::to_string_pretty(report).map_err(std::io::Error::other)?;
    fs::write(path, format!("{raw}\n"))
}

fn print_text_report(report: &E2ELogValidationReport) {
    println!(
        "E2E log validation: {} checked, {} passed, {} failed",
        report.checked, report.passed, report.failed
    );
    for log in &report.logs {
        if log.issues.is_empty() {
            println!("PASS {}", log.path);
            continue;
        }
        println!("FAIL {}", log.path);
        for issue in &log.issues {
            println!("  {} {}: {}", issue.code, issue.path, issue.message);
        }
    }
}
