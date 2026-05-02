#![forbid(unsafe_code)]

use fj_conformance::e2e_log::{
    E2EAllocationCounters, E2EArtifactRef, E2ECompatibilityMode, E2EErrorClass, E2EForensicLogV1,
    E2ELogStatus, E2ETolerancePolicy, artifact_sha256_hex, write_e2e_log,
};
use fj_conformance::security_adversarial::{
    SECURITY_ADVERSARIAL_BEAD_ID, SecurityAdversarialOutputPaths, SecurityAdversarialReport,
    security_adversarial_summary_json, write_security_adversarial_outputs,
};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
struct Args {
    root: PathBuf,
    threat_model: PathBuf,
    report: PathBuf,
    markdown: PathBuf,
    e2e: PathBuf,
    category_filter: Option<String>,
    family_filter: Option<String>,
    enforce: bool,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut root = std::env::current_dir().map_err(|err| format!("current dir: {err}"))?;
        let mut threat_model = None;
        let mut report = None;
        let mut markdown = None;
        let mut e2e = None;
        let mut category_filter = None;
        let mut family_filter = None;
        let mut enforce = false;

        let mut iter = std::env::args().skip(1);
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--root" => root = PathBuf::from(next_value(&mut iter, "--root")?),
                "--threat-model" => {
                    threat_model = Some(PathBuf::from(next_value(&mut iter, "--threat-model")?));
                }
                "--report" => report = Some(PathBuf::from(next_value(&mut iter, "--report")?)),
                "--markdown" => {
                    markdown = Some(PathBuf::from(next_value(&mut iter, "--markdown")?));
                }
                "--e2e" => e2e = Some(PathBuf::from(next_value(&mut iter, "--e2e")?)),
                "--category" => category_filter = Some(next_value(&mut iter, "--category")?),
                "--family" => family_filter = Some(next_value(&mut iter, "--family")?),
                "--enforce" => enforce = true,
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument `{arg}`")),
            }
        }

        Ok(Self {
            threat_model: threat_model.unwrap_or_else(|| {
                root.join("artifacts/conformance/security_threat_model.v1.json")
            }),
            report: report.unwrap_or_else(|| {
                root.join("artifacts/conformance/security_adversarial_gate.v1.json")
            }),
            markdown: markdown.unwrap_or_else(|| {
                root.join("artifacts/conformance/security_adversarial_gate.v1.md")
            }),
            e2e: e2e.unwrap_or_else(|| root.join("artifacts/e2e/e2e_security_gate.e2e.json")),
            root,
            category_filter,
            family_filter,
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

    let report = match write_security_adversarial_outputs(
        &args.root,
        &args.threat_model,
        &args.report,
        &args.markdown,
    ) {
        Ok(report) => report,
        Err(err) => {
            eprintln!("error: failed to write security gate outputs: {err}");
            std::process::exit(1);
        }
    };

    if let Some(category_id) = args.category_filter.as_deref()
        && !report
            .threat_categories
            .iter()
            .any(|category| category.category_id == category_id)
    {
        eprintln!("error: unknown security category `{category_id}`");
        std::process::exit(2);
    }
    if let Some(family_id) = args.family_filter.as_deref()
        && !report
            .fuzz_families
            .iter()
            .any(|family| family.family_id == family_id)
    {
        eprintln!("error: unknown fuzz family `{family_id}`");
        std::process::exit(2);
    }

    if let Err(err) = write_forensic_log(&args, &report) {
        eprintln!("error: failed to write E2E forensic log: {err}");
        std::process::exit(1);
    }

    println!(
        "security adversarial gate: status={} categories={} fuzz_families={} rows={} open_p0_crashes={} issues={}",
        report.status,
        report.threat_categories.len(),
        report.fuzz_families.len(),
        report.adversarial_rows.len(),
        report.crash_index.p0_open_crash_count,
        report.issues.len()
    );
    for category in &report.threat_categories {
        if args
            .category_filter
            .as_deref()
            .is_none_or(|filter| filter == category.category_id)
        {
            println!(
                "  category {}: evidence={} fuzz={} families={} rows={}",
                category.category_id,
                category.evidence_status,
                category.fuzz_status,
                category.required_fuzz_families.join(","),
                category.required_adversarial_rows.join(",")
            );
        }
    }
    for family in &report.fuzz_families {
        if args
            .family_filter
            .as_deref()
            .is_none_or(|filter| filter == family.family_id)
        {
            println!(
                "  fuzz {}: target={} seeds={} panic={} crash={} timeout={}",
                family.family_id,
                family.target,
                family.observed_seed_count,
                family.panic_status,
                family.crash_status,
                family.timeout_status
            );
        }
    }

    if args.enforce && report.status != "pass" {
        std::process::exit(1);
    }
}

fn write_forensic_log(
    args: &Args,
    report: &SecurityAdversarialReport,
) -> Result<(), std::io::Error> {
    let status = if report.status == "pass" {
        E2ELogStatus::Pass
    } else {
        E2ELogStatus::Fail
    };
    let mut log = E2EForensicLogV1::new(
        SECURITY_ADVERSARIAL_BEAD_ID,
        "e2e_security_gate",
        "fj_security_gate",
        std::env::args().collect(),
        args.root.display().to_string(),
        E2ECompatibilityMode::Strict,
        status,
    );
    log.fixture_ids = report
        .adversarial_rows
        .iter()
        .map(|row| row.row_id.clone())
        .collect();
    log.oracle_ids = vec!["local:security-adversarial-gate.v1".to_owned()];
    log.transform_stack = vec![
        "cache".to_owned(),
        "jit".to_owned(),
        "grad".to_owned(),
        "vmap".to_owned(),
        "ffi".to_owned(),
        "durability".to_owned(),
    ];
    log.inputs = serde_json::json!({
        "workspace_root": args.root.display().to_string(),
        "matrix_policy": report.matrix_policy,
        "category_filter": args.category_filter,
        "family_filter": args.family_filter,
        "threat_categories": report.threat_categories.iter().map(|category| {
            serde_json::json!({
                "category_id": category.category_id,
                "boundaries": category.boundaries,
                "required_fuzz_families": category.required_fuzz_families,
                "required_adversarial_rows": category.required_adversarial_rows,
            })
        }).collect::<Vec<_>>(),
        "fuzz_families": report.fuzz_families.iter().map(|family| {
            serde_json::json!({
                "family_id": family.family_id,
                "target": family.target,
                "corpus_path": family.corpus_path,
                "target_source": family.target_source,
                "seed_floor": family.seed_floor,
                "observed_seed_count": family.observed_seed_count,
                "deterministic_replay_count": family.deterministic_replay_count,
                "expected_error_class": family.expected_error_class,
                "actual_error_class": family.actual_error_class,
                "panic_status": family.panic_status,
                "crash_status": family.crash_status,
                "timeout_status": family.timeout_status,
                "minimized_repro_path": family.minimized_repro_path,
                "artifact_hashes": family.artifact_hashes,
                "replay_command": family.replay_command,
            })
        }).collect::<Vec<_>>(),
        "adversarial_rows": report.adversarial_rows.iter().map(|row| {
            serde_json::json!({
                "row_id": row.row_id,
                "category_id": row.category_id,
                "input_family": row.input_family,
                "target_subsystem": row.target_subsystem,
                "expected_error_class": row.expected_error_class,
                "actual_error_class": row.actual_error_class,
                "strict_behavior": row.strict_behavior,
                "hardened_behavior": row.hardened_behavior,
                "panic_status": row.panic_status,
                "crash_status": row.crash_status,
                "timeout_status": row.timeout_status,
                "fuzz_family_id": row.fuzz_family_id,
                "evidence_refs": row.evidence_refs,
                "replay_command": row.replay_command,
            })
        }).collect::<Vec<_>>(),
        "case_count": report.adversarial_rows.len(),
    });
    log.expected = serde_json::json!({
        "gate_status": "pass",
        "required_category_count": report.coverage.required_category_count,
        "all_categories_green": true,
        "all_fuzz_families_complete": true,
        "panic_status": "no_panic",
        "crash_status": "no_crash",
        "timeout_status": "no_timeout",
        "p0_open_crashes": 0,
    });
    log.actual = security_adversarial_summary_json(report);
    log.tolerance = E2ETolerancePolicy {
        policy_id: "exact_security_evidence_and_deterministic_fuzz_inventory".to_owned(),
        atol: None,
        rtol: None,
        ulp: None,
        notes: Some(
            "Security gate comparisons are exact: evidence refs must exist, fuzz corpus inventory must meet seed floors, rows must be typed and panic-free, and open P0 crash count must be zero."
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
            "security_adversarial_gate".to_owned()
        },
    };
    log.allocations = E2EAllocationCounters {
        allocation_count: None,
        allocated_bytes: None,
        peak_rss_bytes: None,
        measurement_backend: "not_measured".to_owned(),
    };
    log.artifacts = vec![
        artifact_ref(&args.root, &args.threat_model, "security_threat_model")?,
        artifact_ref(&args.root, &args.report, "security_adversarial_report")?,
        artifact_ref(&args.root, &args.markdown, "security_adversarial_markdown")?,
    ];
    log.replay_command = stable_replay_command(args);
    if status.requires_failure_summary() {
        log.failure_summary = Some(
            "security gate found missing threat evidence, incomplete fuzz corpus coverage, non-typed adversarial rows, open P0 crashes, or stale artifact refs"
                .to_owned(),
        );
    }
    log.metadata.insert(
        "bead".to_owned(),
        serde_json::json!({
            "id": SECURITY_ADVERSARIAL_BEAD_ID,
            "title": "Prove security and adversarial fuzz gates"
        }),
    );
    log.metadata.insert(
        "crash_index".to_owned(),
        serde_json::json!(report.crash_index),
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
    let defaults = SecurityAdversarialOutputPaths::for_root(&args.root);
    let mut parts = vec!["./scripts/run_security_gate.sh".to_owned()];
    if args.threat_model != defaults.threat_model {
        push_flag_value(
            &mut parts,
            "--threat-model",
            repo_relative(&args.root, &args.threat_model),
        );
    }
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
    let default_e2e = args.root.join("artifacts/e2e/e2e_security_gate.e2e.json");
    if args.e2e != default_e2e {
        push_flag_value(&mut parts, "--e2e", repo_relative(&args.root, &args.e2e));
    }
    if let Some(category) = args.category_filter.as_deref() {
        push_flag_value(&mut parts, "--category", category.to_owned());
    }
    if let Some(family) = args.family_filter.as_deref() {
        push_flag_value(&mut parts, "--family", family.to_owned());
    }
    parts.push("--enforce".to_owned());
    parts.join(" ")
}

fn repo_relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root).map_or_else(
        |_| path.display().to_string(),
        |stripped| stripped.display().to_string(),
    )
}

fn push_flag_value(parts: &mut Vec<String>, flag: &str, value: String) {
    parts.push(flag.to_owned());
    parts.push(shell_quote(&value));
}

fn shell_quote(value: &str) -> String {
    if value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b'/'))
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
    if value.starts_with("--") {
        return Err(format!("{flag} requires a value, got flag `{value}`"));
    }
    Ok(value)
}

fn print_usage() {
    eprintln!(
        "Usage: fj_security_gate [--root <dir>] [--threat-model <json>] [--report <json>] [--markdown <md>] [--e2e <json>] [--category <id>] [--family <id>] [--enforce]"
    );
}
