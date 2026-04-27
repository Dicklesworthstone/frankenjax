#![forbid(unsafe_code)]

use fj_conformance::durability::{
    SidecarConfig, encode_artifact_to_sidecar, generate_decode_proof, scrub_sidecar,
};
use std::path::PathBuf;

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        return Err(usage());
    }

    let command = args.remove(0);
    match command.as_str() {
        "generate" => cmd_generate(args),
        "scrub" => cmd_scrub(args),
        "proof" => cmd_proof(args),
        "pipeline" => cmd_pipeline(args),
        "batch" => cmd_batch(args),
        "verify-only" => cmd_verify_only(args),
        _ => Err(usage()),
    }
}

fn cmd_generate(args: Vec<String>) -> Result<(), String> {
    let artifact = required_path_flag(&args, "--artifact")?;
    let sidecar = required_path_flag(&args, "--sidecar")?;

    let symbol_size = optional_u16_flag(&args, "--symbol-size")?.unwrap_or(256);
    let max_block_size = optional_usize_flag(&args, "--max-block-size")?.unwrap_or(1024 * 1024);
    let repair_overhead = optional_f64_flag(&args, "--repair-overhead")?.unwrap_or(1.1);

    let manifest = encode_artifact_to_sidecar(
        &artifact,
        &sidecar,
        &SidecarConfig {
            symbol_size,
            max_block_size,
            repair_overhead,
        },
    )
    .map_err(|err| err.to_string())?;

    println!(
        "generated sidecar: {} (total symbols={}, source={}, repair={})",
        sidecar.display(),
        manifest.total_symbols,
        manifest.source_symbols,
        manifest.repair_symbols
    );

    Ok(())
}

fn cmd_scrub(args: Vec<String>) -> Result<(), String> {
    let artifact = required_path_flag(&args, "--artifact")?;
    let sidecar = required_path_flag(&args, "--sidecar")?;
    let report = required_path_flag(&args, "--report")?;

    let scrub = scrub_sidecar(&sidecar, &artifact, &report).map_err(|err| err.to_string())?;
    println!(
        "scrub report: {} (match={})",
        report.display(),
        scrub.decoded_matches_expected
    );
    Ok(())
}

fn cmd_proof(args: Vec<String>) -> Result<(), String> {
    let artifact = required_path_flag(&args, "--artifact")?;
    let sidecar = required_path_flag(&args, "--sidecar")?;
    let proof = required_path_flag(&args, "--proof")?;
    let drop_source_count = optional_usize_flag(&args, "--drop-source")?.unwrap_or(1);

    let result = generate_decode_proof(&sidecar, &artifact, &proof, drop_source_count)
        .map_err(|err| err.to_string())?;
    println!(
        "decode proof: {} (recovered={}, dropped={})",
        proof.display(),
        result.recovered,
        result.dropped_symbols.len()
    );
    Ok(())
}

fn cmd_pipeline(args: Vec<String>) -> Result<(), String> {
    let artifact = required_path_flag(&args, "--artifact")?;
    let sidecar = required_path_flag(&args, "--sidecar")?;
    let report = required_path_flag(&args, "--report")?;
    let proof = required_path_flag(&args, "--proof")?;

    let symbol_size = optional_u16_flag(&args, "--symbol-size")?.unwrap_or(256);
    let max_block_size = optional_usize_flag(&args, "--max-block-size")?.unwrap_or(1024 * 1024);
    let repair_overhead = optional_f64_flag(&args, "--repair-overhead")?.unwrap_or(1.1);
    let drop_source_count = optional_usize_flag(&args, "--drop-source")?.unwrap_or(1);

    encode_artifact_to_sidecar(
        &artifact,
        &sidecar,
        &SidecarConfig {
            symbol_size,
            max_block_size,
            repair_overhead,
        },
    )
    .map_err(|err| err.to_string())?;

    scrub_sidecar(&sidecar, &artifact, &report).map_err(|err| err.to_string())?;
    generate_decode_proof(&sidecar, &artifact, &proof, drop_source_count)
        .map_err(|err| err.to_string())?;

    println!(
        "pipeline complete: sidecar={}, report={}, proof={}",
        sidecar.display(),
        report.display(),
        proof.display()
    );
    Ok(())
}

fn required_path_flag(args: &[String], flag: &str) -> Result<PathBuf, String> {
    let value = required_string_flag(args, flag)?;
    Ok(PathBuf::from(value))
}

fn required_string_flag(args: &[String], flag: &str) -> Result<String, String> {
    for idx in 0..args.len() {
        if args[idx] == flag {
            if let Some(value) = args.get(idx + 1) {
                return Ok(value.clone());
            }
            return Err(format!("missing value for {flag}"));
        }
    }
    Err(format!("missing required flag {flag}"))
}

fn optional_usize_flag(args: &[String], flag: &str) -> Result<Option<usize>, String> {
    optional_string_flag(args, flag)?
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|err| format!("invalid {flag}: {err}"))
        })
        .transpose()
}

fn optional_u16_flag(args: &[String], flag: &str) -> Result<Option<u16>, String> {
    optional_string_flag(args, flag)?
        .map(|value| {
            value
                .parse::<u16>()
                .map_err(|err| format!("invalid {flag}: {err}"))
        })
        .transpose()
}

fn optional_f64_flag(args: &[String], flag: &str) -> Result<Option<f64>, String> {
    optional_string_flag(args, flag)?
        .map(|value| {
            value
                .parse::<f64>()
                .map_err(|err| format!("invalid {flag}: {err}"))
        })
        .transpose()
}

fn optional_string_flag(args: &[String], flag: &str) -> Result<Option<String>, String> {
    for idx in 0..args.len() {
        if args[idx] == flag {
            if let Some(value) = args.get(idx + 1) {
                return Ok(Some(value.clone()));
            }
            return Err(format!("missing value for {flag}"));
        }
    }
    Ok(None)
}

fn cmd_batch(args: Vec<String>) -> Result<(), String> {
    let artifact_dir = required_path_flag(&args, "--dir")?;
    let output_dir = required_path_flag(&args, "--output")?;
    let pattern = optional_string_flag(&args, "--pattern")?.unwrap_or_else(|| "*.json".to_owned());
    let drop_source_count = optional_usize_flag(&args, "--drop-source")?.unwrap_or(2);
    let json_output = args.iter().any(|a| a == "--json");

    std::fs::create_dir_all(&output_dir)
        .map_err(|e| format!("failed to create output dir: {e}"))?;

    let entries: Vec<_> = std::fs::read_dir(&artifact_dir)
        .map_err(|e| format!("failed to read dir {}: {e}", artifact_dir.display()))?
        .filter_map(Result::ok)
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            glob_match(&name, &pattern)
        })
        .collect();

    let mut results = Vec::new();
    let mut pass_count = 0_usize;
    let mut fail_count = 0_usize;

    for entry in &entries {
        let artifact = entry.path();
        let stem = artifact
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_owned());

        let sidecar = output_dir.join(format!("{stem}.sidecar.json"));
        let report = output_dir.join(format!("{stem}.scrub.json"));
        let proof = output_dir.join(format!("{stem}.proof.json"));

        let pipeline_result = (|| -> Result<(), String> {
            encode_artifact_to_sidecar(&artifact, &sidecar, &SidecarConfig::default())
                .map_err(|e| e.to_string())?;

            scrub_sidecar(&sidecar, &artifact, &report).map_err(|e| e.to_string())?;

            let decode = generate_decode_proof(&sidecar, &artifact, &proof, drop_source_count)
                .map_err(|e| e.to_string())?;

            if !decode.recovered {
                return Err(format!("decode proof failed for {}", artifact.display()));
            }
            Ok(())
        })();

        match &pipeline_result {
            Ok(()) => {
                pass_count += 1;
                if !json_output {
                    println!("  PASS: {}", artifact.display());
                }
            }
            Err(e) => {
                fail_count += 1;
                if !json_output {
                    eprintln!("  FAIL: {} — {e}", artifact.display());
                }
            }
        }

        results.push(serde_json::json!({
            "artifact": artifact.display().to_string(),
            "status": if pipeline_result.is_ok() { "pass" } else { "fail" },
            "error": pipeline_result.err(),
        }));
    }

    if json_output {
        let report = serde_json::json!({
            "schema_version": "frankenjax.durability-batch.v1",
            "total": entries.len(),
            "passed": pass_count,
            "failed": fail_count,
            "drop_source_count": drop_source_count,
            "results": results,
        });
        let rendered = serde_json::to_string_pretty(&report)
            .map_err(|err| format!("serialize batch report: {err}"))?;
        println!("{rendered}");
    } else {
        println!(
            "batch complete: {} total, {} passed, {} failed",
            entries.len(),
            pass_count,
            fail_count
        );
    }

    if fail_count > 0 {
        Err(format!(
            "{fail_count} artifact(s) failed durability pipeline"
        ))
    } else {
        Ok(())
    }
}

fn cmd_verify_only(args: Vec<String>) -> Result<(), String> {
    let artifact_dir = required_path_flag(&args, "--dir")?;
    let json_output = args.iter().any(|a| a == "--json");

    let entries: Vec<_> = std::fs::read_dir(&artifact_dir)
        .map_err(|e| format!("failed to read dir: {e}"))?
        .filter_map(Result::ok)
        .filter(|e| {
            e.path().extension().is_some_and(|ext| ext == "json")
                && !e.file_name().to_string_lossy().contains(".sidecar.")
                && !e.file_name().to_string_lossy().contains(".scrub.")
                && !e.file_name().to_string_lossy().contains(".proof.")
        })
        .collect();

    let mut pass_count = 0_usize;
    let mut fail_count = 0_usize;
    let mut missing_count = 0_usize;
    let mut results = Vec::new();

    for entry in &entries {
        let artifact = entry.path();
        let stem = artifact
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();

        let sidecar = artifact.with_file_name(format!("{stem}.sidecar.json"));
        if !sidecar.exists() {
            missing_count += 1;
            results.push(serde_json::json!({
                "artifact": artifact.display().to_string(),
                "status": "missing_sidecar",
            }));
            continue;
        }

        let scrub_result = scrub_sidecar(
            &sidecar,
            &artifact,
            &artifact.with_file_name(format!("{stem}.verify.json")),
        );

        match scrub_result {
            Ok(report) if report.decoded_matches_expected => {
                pass_count += 1;
                results.push(serde_json::json!({
                    "artifact": artifact.display().to_string(),
                    "status": "pass",
                }));
            }
            Ok(_) => {
                fail_count += 1;
                results.push(serde_json::json!({
                    "artifact": artifact.display().to_string(),
                    "status": "integrity_fail",
                }));
            }
            Err(e) => {
                fail_count += 1;
                results.push(serde_json::json!({
                    "artifact": artifact.display().to_string(),
                    "status": "error",
                    "error": e.to_string(),
                }));
            }
        }
    }

    if json_output {
        let report = serde_json::json!({
            "schema_version": "frankenjax.durability-verify.v1",
            "total": entries.len(),
            "passed": pass_count,
            "failed": fail_count,
            "missing_sidecar": missing_count,
            "results": results,
        });
        let rendered = serde_json::to_string_pretty(&report)
            .map_err(|err| format!("serialize verify report: {err}"))?;
        println!("{rendered}");
    } else {
        println!(
            "verify: {} artifacts, {} passed, {} failed, {} missing sidecar",
            entries.len(),
            pass_count,
            fail_count,
            missing_count
        );
    }

    if fail_count > 0 {
        Err(format!(
            "{fail_count} artifact(s) failed integrity verification"
        ))
    } else {
        Ok(())
    }
}

fn glob_match(name: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(suffix) = pattern.strip_prefix('*') {
        return name.ends_with(suffix);
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return name.starts_with(prefix);
    }
    name == pattern
}

fn usage() -> String {
    [
        "usage:",
        "  fj_durability generate --artifact <path> --sidecar <path> [--symbol-size <u16>] [--max-block-size <usize>] [--repair-overhead <f64>]",
        "  fj_durability scrub --artifact <path> --sidecar <path> --report <path>",
        "  fj_durability proof --artifact <path> --sidecar <path> --proof <path> [--drop-source <usize>]",
        "  fj_durability pipeline --artifact <path> --sidecar <path> --report <path> --proof <path> [opts...]",
        "  fj_durability batch --dir <dir> --output <dir> [--pattern <glob>] [--drop-source <N>] [--json]",
        "  fj_durability verify-only --dir <dir> [--json]",
    ]
    .join("\n")
}
