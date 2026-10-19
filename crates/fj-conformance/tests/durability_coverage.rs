//! Durability coverage tests.
//!
//! Verifies that all long-lived conformance fixtures pass the full
//! RaptorQ durability pipeline: generate → scrub → decode proof.
//! This ensures that fixture bundles underpinning correctness guarantees
//! are protected against bit rot and storage corruption.

use fj_conformance::durability::{
    SidecarConfig, encode_artifact_to_sidecar, generate_decode_proof, scrub_sidecar,
};
use std::path::{Path, PathBuf};
use tempfile::tempdir;

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

/// Run the full durability pipeline on a single artifact file.
/// Returns Ok(()) if generate + scrub + decode-proof all succeed.
fn run_durability_pipeline(artifact_path: &Path) -> Result<(), String> {
    let tmp = tempdir().map_err(|e| format!("tempdir: {e}"))?;
    let stem = artifact_path
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .to_string();

    let sidecar = tmp.path().join(format!("{stem}.sidecar.json"));
    let scrub_report = tmp.path().join(format!("{stem}.scrub.json"));
    let proof = tmp.path().join(format!("{stem}.proof.json"));

    // Generate sidecar
    let manifest = encode_artifact_to_sidecar(artifact_path, &sidecar, &SidecarConfig::default())
        .map_err(|e| format!("generate failed for {}: {e}", artifact_path.display()))?;

    assert!(
        manifest.total_symbols > 0,
        "sidecar should have symbols for {}",
        artifact_path.display()
    );
    assert!(
        manifest.source_symbols > 0,
        "sidecar should have source symbols for {}",
        artifact_path.display()
    );
    assert!(
        manifest.repair_symbols > 0,
        "sidecar should have repair symbols for {}",
        artifact_path.display()
    );

    // Scrub: verify sidecar decodes to match artifact
    let scrub = scrub_sidecar(&sidecar, artifact_path, &scrub_report)
        .map_err(|e| format!("scrub failed for {}: {e}", artifact_path.display()))?;

    assert!(
        scrub.decoded_matches_expected,
        "scrub integrity check failed for {}",
        artifact_path.display()
    );

    // Decode proof: drop 2 source symbols, verify recovery
    let decode = generate_decode_proof(&sidecar, artifact_path, &proof, 2)
        .map_err(|e| format!("proof failed for {}: {e}", artifact_path.display()))?;

    assert!(
        decode.recovered,
        "decode proof recovery failed for {} (dropped {} symbols)",
        artifact_path.display(),
        decode.dropped_symbols.len()
    );

    Ok(())
}

fn collect_json_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(collect_json_files(&path));
            } else if path.extension().is_some_and(|ext| ext == "json") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

#[test]
fn all_conformance_fixtures_pass_durability_pipeline() {
    let root = fixture_root();
    let fixtures = collect_json_files(&root);

    assert!(
        !fixtures.is_empty(),
        "expected at least one fixture in {}",
        root.display()
    );

    let mut failures = Vec::new();
    for fixture in &fixtures {
        if let Err(e) = run_durability_pipeline(fixture) {
            failures.push(e);
        }
    }

    assert!(
        failures.is_empty(),
        "durability pipeline failures:\n{}",
        failures.join("\n")
    );
}

#[test]
fn smoke_fixture_durability_round_trip() {
    let smoke_path = fixture_root().join("smoke_case.json");
    if !smoke_path.exists() {
        eprintln!("SKIP: smoke_case.json not found");
        return;
    }
    run_durability_pipeline(&smoke_path).expect("smoke fixture durability pipeline should pass");
}

#[test]
fn rng_fixture_durability_round_trip() {
    let rng_path = fixture_root().join("rng/rng_determinism.v1.json");
    if !rng_path.exists() {
        eprintln!("SKIP: rng_determinism.v1.json not found");
        return;
    }
    run_durability_pipeline(&rng_path).expect("RNG fixture durability pipeline should pass");
}

#[test]
fn transform_fixture_durability_round_trip() {
    let transform_path = fixture_root().join("transforms/legacy_transform_cases.v1.json");
    if !transform_path.exists() {
        eprintln!("SKIP: legacy_transform_cases.v1.json not found");
        return;
    }
    run_durability_pipeline(&transform_path)
        .expect("transform fixture durability pipeline should pass");
}

/// Verify that the CI reliability budgets artifact passes durability.
#[test]
fn ci_reliability_budgets_durability() {
    let budgets_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../artifacts/ci/reliability_budgets.v1.json");
    if !budgets_path.exists() {
        eprintln!("SKIP: reliability_budgets.v1.json not found");
        return;
    }
    run_durability_pipeline(&budgets_path)
        .expect("reliability budgets durability pipeline should pass");
}

/// Verify that the conformance parity report artifact passes durability.
#[test]
fn conformance_parity_report_durability() {
    let parity_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../artifacts/conformance/v1_parity_report.json");
    if !parity_path.exists() {
        eprintln!("SKIP: v1_parity_report.json not found");
        return;
    }
    run_durability_pipeline(&parity_path).expect("parity report durability pipeline should pass");
}

fn sha256_hex_file(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).expect("file must be readable for sha256");
    let digest = Sha256::digest(&bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn resolve_triplet_file(root: &Path, artifact: &Path, suffix: &str) -> Option<PathBuf> {
    let stem = artifact.file_stem()?.to_string_lossy();
    let inline = artifact.with_file_name(format!("{stem}.{suffix}.json"));
    if inline.exists() {
        return Some(inline);
    }
    let sub = root
        .join("artifacts/durability")
        .join(stem.as_ref())
        .join(format!("{stem}.{suffix}.json"));
    if sub.exists() {
        return Some(sub);
    }
    let flat = root
        .join("artifacts/durability")
        .join(format!("{stem}.{suffix}.json"));
    if flat.exists() {
        return Some(flat);
    }
    None
}

#[test]
fn all_required_policy_artifacts_have_fresh_on_disk_triplets() {
    use fj_conformance::durability::{DecodeProof, ScrubReport, SidecarManifest};

    let root = repo_root();
    let policy_path = root.join("artifacts/conformance/durability_coverage_policy.v1.json");
    let policy_raw =
        std::fs::read_to_string(&policy_path).expect("durability_coverage_policy.v1.json exists");
    let policy: serde_json::Value =
        serde_json::from_str(&policy_raw).expect("valid durability coverage policy JSON");
    assert_eq!(
        policy["coverage_target"]["required_artifacts_coverage_pct"],
        100
    );
    assert_eq!(policy["coverage_target"]["sha256_freshness_required"], true);

    let mut required_artifacts = collect_json_files(&root.join("crates/fj-conformance/fixtures"));
    for rel_dir in [
        "artifacts/conformance",
        "artifacts/ci",
        "artifacts/performance",
    ] {
        let dir = root.join(rel_dir);
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && path.extension().is_some_and(|ext| ext == "json") {
                    let name = path.file_name().unwrap().to_string_lossy();
                    if !name.ends_with(".sidecar.json")
                        && !name.ends_with(".scrub.json")
                        && !name.ends_with(".proof.json")
                        && !name.ends_with(".decode-proof.json")
                        && !name.ends_with(".verify.json")
                    {
                        required_artifacts.push(path);
                    }
                }
            }
        }
    }
    required_artifacts.sort();
    assert!(
        required_artifacts.len() >= 38,
        "expected at least 38 required long-lived artifacts, found {}",
        required_artifacts.len()
    );

    let mut failures = Vec::new();
    for artifact in &required_artifacts {
        let actual_sha = sha256_hex_file(artifact);
        let Some(sidecar_path) = resolve_triplet_file(&root, artifact, "sidecar") else {
            failures.push(format!("missing sidecar for {}", artifact.display()));
            continue;
        };
        let Some(scrub_path) = resolve_triplet_file(&root, artifact, "scrub") else {
            failures.push(format!("missing scrub report for {}", artifact.display()));
            continue;
        };
        let Some(proof_path) = resolve_triplet_file(&root, artifact, "proof") else {
            failures.push(format!("missing decode proof for {}", artifact.display()));
            continue;
        };

        let sidecar: SidecarManifest = serde_json::from_str(
            &std::fs::read_to_string(&sidecar_path).expect("sidecar readable"),
        )
        .expect("valid SidecarManifest");
        let scrub: ScrubReport =
            serde_json::from_str(&std::fs::read_to_string(&scrub_path).expect("scrub readable"))
                .expect("valid ScrubReport");
        let proof: DecodeProof =
            serde_json::from_str(&std::fs::read_to_string(&proof_path).expect("proof readable"))
                .expect("valid DecodeProof");

        if sidecar.artifact_sha256_hex != actual_sha {
            failures.push(format!(
                "stale sidecar SHA-256 for {}: sidecar={} actual={}",
                artifact.display(),
                sidecar.artifact_sha256_hex,
                actual_sha
            ));
        }
        if !scrub.decoded_matches_expected || scrub.decoded_sha256_hex != actual_sha {
            failures.push(format!(
                "stale or failed scrub report for {}: scrub={} actual={}",
                artifact.display(),
                scrub.decoded_sha256_hex,
                actual_sha
            ));
        }
        if !proof.recovered || proof.decoded_sha256_hex.as_deref() != Some(actual_sha.as_str()) {
            failures.push(format!(
                "stale or unrecovered decode proof for {}: proof={:?} actual={}",
                artifact.display(),
                proof.decoded_sha256_hex,
                actual_sha
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "on-disk durability coverage / freshness failures:\n{}",
        failures.join("\n")
    );
}

#[test]
fn all_on_disk_durability_sidecars_match_source_sha256() {
    use fj_conformance::durability::SidecarManifest;

    let root = repo_root();
    let durability_dir = root.join("artifacts/durability");
    let all_json = collect_json_files(&durability_dir);
    let sidecars: Vec<_> = all_json
        .into_iter()
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().ends_with(".sidecar.json"))
        })
        .collect();
    assert!(
        !sidecars.is_empty(),
        "expected sidecar files in artifacts/durability"
    );

    let mut stale = Vec::new();
    for sc_path in &sidecars {
        let manifest: SidecarManifest =
            serde_json::from_str(&std::fs::read_to_string(sc_path).expect("sidecar readable"))
                .expect("valid SidecarManifest");
        let rel = manifest
            .artifact_path
            .strip_prefix("/data/projects/frankenjax/")
            .or_else(|| manifest.artifact_path.strip_prefix("/dp/frankenjax/"))
            .unwrap_or(&manifest.artifact_path);
        let source_path = root.join(rel);
        if !source_path.exists() {
            stale.push(format!(
                "{} references missing source {}",
                sc_path.display(),
                source_path.display()
            ));
            continue;
        }
        let actual_sha = sha256_hex_file(&source_path);
        if manifest.artifact_sha256_hex != actual_sha {
            stale.push(format!(
                "{} has stale SHA-256 {} != {} ({})",
                sc_path.display(),
                manifest.artifact_sha256_hex,
                actual_sha,
                rel
            ));
        }
    }

    assert!(
        stale.is_empty(),
        "found stale sidecars in artifacts/durability:\n{}",
        stale.join("\n")
    );
}

#[test]
fn tampered_artifact_fails_scrub_and_sha256_freshness() {
    let tmp = tempdir().expect("tempdir");
    let artifact = tmp.path().join("sample.json");
    let sidecar = tmp.path().join("sample.sidecar.json");
    let scrub_report = tmp.path().join("sample.scrub.json");

    std::fs::write(&artifact, br#"{"value": 42}"#).expect("write artifact");
    let manifest =
        encode_artifact_to_sidecar(&artifact, &sidecar, &SidecarConfig::default()).expect("encode");

    // Tamper with the artifact on disk without updating the sidecar
    std::fs::write(&artifact, br#"{"value": 43}"#).expect("tamper artifact");
    let tampered_sha = sha256_hex_file(&artifact);
    assert_ne!(
        manifest.artifact_sha256_hex, tampered_sha,
        "tampered artifact must diverge from sidecar SHA-256"
    );

    let scrub_res = scrub_sidecar(&sidecar, &artifact, &scrub_report);
    assert!(
        matches!(
            scrub_res,
            Err(fj_conformance::durability::DurabilityError::Integrity(_))
        ),
        "scrub must fail with Integrity error when artifact is modified after sidecar generation, got {scrub_res:?}"
    );
}
