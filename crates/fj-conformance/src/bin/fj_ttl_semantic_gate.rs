#![forbid(unsafe_code)]

use fj_conformance::e2e_log::{
    E2EAllocationCounters, E2EArtifactRef, E2ECompatibilityMode, E2EErrorClass, E2EForensicLogV1,
    E2ELogStatus, E2ETolerancePolicy, artifact_sha256_hex, write_e2e_log,
};
use fj_conformance::ttl_semantic::{
    TTL_SEMANTIC_BEAD_ID, TtlSemanticReport, ttl_semantic_summary_json, write_ttl_semantic_outputs,
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
                root.join("artifacts/conformance/ttl_semantic_proof_matrix.v1.json")
            }),
            markdown: markdown.unwrap_or_else(|| {
                root.join("artifacts/conformance/ttl_semantic_proof_matrix.v1.md")
            }),
            e2e: e2e.unwrap_or_else(|| root.join("artifacts/e2e/e2e_ttl_semantic_gate.e2e.json")),
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

    let report = match write_ttl_semantic_outputs(&args.root, &args.report, &args.markdown) {
        Ok(report) => report,
        Err(err) => {
            eprintln!("error: failed to write TTL semantic outputs: {err}");
            std::process::exit(1);
        }
    };
    if let Some(case_id) = args.case_filter.as_deref()
        && !report.cases.iter().any(|case| case.case_id == case_id)
    {
        eprintln!("error: unknown TTL semantic case `{case_id}`");
        std::process::exit(2);
    }

    if let Err(err) = write_forensic_log(&args, &report) {
        eprintln!("error: failed to write E2E forensic log: {err}");
        std::process::exit(1);
    }

    println!(
        "TTL semantic gate: status={} cases={} accepted={} rejected={} fixture_links={} structural_replays={} issues={}",
        report.status,
        report.cases.len(),
        report.coverage.accepted_count,
        report.coverage.rejected_count,
        report.coverage.fixture_linked_count,
        report.coverage.structural_replay_count,
        report.issues.len()
    );
    for case in &report.cases {
        if args
            .case_filter
            .as_deref()
            .is_none_or(|filter| filter == case.case_id)
        {
            println!(
                "  case {}: kind={} mode={} decision={} expected={} stack={} rejection={:?}",
                case.case_id,
                case.proof_kind,
                case.compatibility_mode,
                case.verifier_decision,
                case.expected_decision,
                case.transform_stack.join(">"),
                case.rejection_reason
            );
        }
    }

    if args.enforce && report.status != "pass" {
        std::process::exit(1);
    }
}

fn write_forensic_log(args: &Args, report: &TtlSemanticReport) -> Result<(), std::io::Error> {
    let status = if report.status == "pass" {
        E2ELogStatus::Pass
    } else {
        E2ELogStatus::Fail
    };
    let mut log = E2EForensicLogV1::new(
        TTL_SEMANTIC_BEAD_ID,
        "e2e_ttl_semantic_gate",
        "fj_ttl_semantic_gate",
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
        .filter_map(|case| case.oracle_fixture_id.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    log.transform_stack = vec!["jit".to_owned(), "grad".to_owned(), "vmap".to_owned()];
    log.inputs = serde_json::json!({
        "workspace_root": args.root.display().to_string(),
        "matrix_policy": report.matrix_policy,
        "case_filter": args.case_filter,
        "proof_inputs": report.cases.iter().map(|case| {
            serde_json::json!({
                "case_id": case.case_id,
                "program": case.program,
                "inputs": case.inputs,
                "expected_input_fingerprint": case.expected_input_fingerprint,
                "input_fingerprint": case.input_fingerprint,
                "evidence_ids": case.evidence_ids,
                "stack_signature": case.stack_signature,
                "stack_hash_hex": case.stack_hash_hex,
            })
        }).collect::<Vec<_>>(),
    });
    log.expected = serde_json::json!({
        "gate_status": "pass",
        "required_case_count": report.coverage.required_case_count,
        "accepted_rows": report.coverage.accepted_count,
        "rejected_rows": report.coverage.rejected_count,
        "strict_invalid_proofs": [
            "duplicate_evidence",
            "missing_evidence",
            "stale_input_fingerprint",
            "wrong_transform_binding",
            "missing_oracle_fixture_link"
        ],
    });
    log.actual = ttl_semantic_summary_json(report);
    log.tolerance = E2ETolerancePolicy {
        policy_id: "exact_proof_decisions_with_f64_tolerance_for_oracle_replay".to_owned(),
        atol: Some(1e-8),
        rtol: Some(0.0),
        ulp: None,
        notes: Some(
            "Proof metadata comparisons are exact; numeric replay rows use absolute f64 tolerance while preserving shape and dtype equality."
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
            "ttl_semantic_proof_matrix_gate".to_owned()
        },
    };
    log.allocations = E2EAllocationCounters {
        allocation_count: None,
        allocated_bytes: None,
        peak_rss_bytes: None,
        measurement_backend: "not_measured".to_owned(),
    };
    log.artifacts = vec![
        artifact_ref(&args.root, &args.report, "ttl_semantic_proof_matrix_report")?,
        artifact_ref(
            &args.root,
            &args.markdown,
            "ttl_semantic_proof_matrix_markdown",
        )?,
    ];
    log.replay_command = stable_replay_command(args);
    if status.requires_failure_summary() {
        log.failure_summary = Some(
            "TTL semantic gate found a missing row, mismatched verifier decision, accepted stale proof, missing fixture link, or structural output mismatch"
                .to_owned(),
        );
    }
    log.metadata.insert(
        "bead".to_owned(),
        serde_json::json!({
            "id": TTL_SEMANTIC_BEAD_ID,
            "title": "Add TTL semantic verifier with structural oracle replay"
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
    let mut parts = vec!["./scripts/run_ttl_semantic_gate.sh".to_owned()];
    let default_report = args
        .root
        .join("artifacts/conformance/ttl_semantic_proof_matrix.v1.json");
    let default_markdown = args
        .root
        .join("artifacts/conformance/ttl_semantic_proof_matrix.v1.md");
    let default_e2e = args
        .root
        .join("artifacts/e2e/e2e_ttl_semantic_gate.e2e.json");

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
    if let Some(case_id) = args.case_filter.as_deref() {
        push_flag_value(&mut parts, "--case", case_id.to_owned());
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
        "Usage: fj_ttl_semantic_gate [--root <dir>] [--report <json>] [--markdown <md>] [--e2e <json>] [--case <case_id>] [--enforce]"
    );
}
