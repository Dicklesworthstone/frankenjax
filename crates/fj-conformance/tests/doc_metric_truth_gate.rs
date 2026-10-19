#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use fj_core::Primitive;
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn sha256_hex(path: &Path) -> String {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
    format!("{:x}", Sha256::digest(&bytes))
}

fn count_enum_variants(source: &str, enum_name: &str) -> usize {
    let header = format!("pub enum {enum_name} {{");
    let start = source
        .find(&header)
        .unwrap_or_else(|| panic!("missing `{header}` in source"));
    let body_plus = &source[start + header.len()..];
    let end = body_plus
        .find("\n}")
        .unwrap_or_else(|| panic!("missing closing brace for `{header}`"));
    let body = &body_plus[..end];
    body.lines()
        .filter_map(|line| {
            let trimmed = line.split("//").next().unwrap_or("").trim();
            if trimmed.is_empty() || trimmed.starts_with("#[") {
                None
            } else if trimmed.ends_with(',') {
                Some(())
            } else {
                None
            }
        })
        .count()
}

fn count_workspace_members(cargo_toml: &str) -> usize {
    let start = cargo_toml
        .find("members = [")
        .expect("Cargo.toml should contain workspace members list");
    let rest = &cargo_toml[start + "members = [".len()..];
    let end = rest
        .find(']')
        .expect("Cargo.toml workspace members list should close with `]`");
    rest[..end]
        .lines()
        .filter(|line| {
            let trimmed = line.split('#').next().unwrap_or("").trim();
            trimmed.starts_with('"')
        })
        .count()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LiveMetrics {
    workspace_crates: usize,
    canonical_primitives: usize,
    v1_local_primitives: usize,
    pmap_collectives: usize,
    ffi_primitives: usize,
    total_primitive_enum_variants: usize,
    dtypes: usize,
    program_specs: usize,
    egraph_safe_rules: usize,
    egraph_unsafe_rules: usize,
    egraph_total_rules: usize,
}

fn measure_live_metrics(root: &Path) -> LiveMetrics {
    let cargo_toml = fs::read_to_string(root.join("Cargo.toml")).expect("read Cargo.toml");
    let core_src = fs::read_to_string(root.join("crates/fj-core/src/lib.rs"))
        .expect("read crates/fj-core/src/lib.rs");

    let workspace_crates = count_workspace_members(&cargo_toml);
    let canonical_primitives = Primitive::ALL.len();
    let pmap_collectives = Primitive::PMAP_COLLECTIVES.len();
    let ffi_primitives = Primitive::FFI_PRIMITIVES.len();
    let v1_local_primitives = canonical_primitives - pmap_collectives;
    let total_primitive_enum_variants = count_enum_variants(&core_src, "Primitive");
    let dtypes = count_enum_variants(&core_src, "DType");
    let program_specs = count_enum_variants(&core_src, "ProgramSpec");

    let egraph_safe_rules = fj_egraph::algebraic_rules().len();
    let egraph_total_rules =
        fj_egraph::algebraic_rules_with_config(&fj_egraph::OptimizationConfig::aggressive()).len();
    let egraph_unsafe_rules = egraph_total_rules - egraph_safe_rules;

    LiveMetrics {
        workspace_crates,
        canonical_primitives,
        v1_local_primitives,
        pmap_collectives,
        ffi_primitives,
        total_primitive_enum_variants,
        dtypes,
        program_specs,
        egraph_safe_rules,
        egraph_unsafe_rules,
        egraph_total_rules,
    }
}

fn validate_doc_bundle(
    metrics: &LiveMetrics,
    readme: &str,
    agents: &str,
    feature_parity: &str,
    parity_coverage: &str,
    comprehensive_spec: &str,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    let required_readme = [
        format!("{} member crates", metrics.workspace_crates),
        format!("{} workspace crates", metrics.workspace_crates),
        format!("{} canonical primitive", metrics.canonical_primitives),
        format!("{} V1 local", metrics.v1_local_primitives),
        format!("{} pmap collectives", metrics.pmap_collectives),
        format!("{} enum variants", metrics.total_primitive_enum_variants),
        format!(
            "{} pre-defined test program specifications",
            metrics.program_specs
        ),
        format!("{} IEEE-754-safe", metrics.egraph_safe_rules),
        format!("{} total", metrics.egraph_total_rules),
    ];
    for needle in &required_readme {
        if !readme.contains(needle) {
            errors.push(format!(
                "README.md missing live metric substring `{needle}`"
            ));
        }
    }

    let required_agents = [
        format!("{} member crates", metrics.workspace_crates),
        format!(
            "{} canonical primitive variants",
            metrics.canonical_primitives
        ),
        format!("{} V1 local", metrics.v1_local_primitives),
        format!("{} pmap collectives", metrics.pmap_collectives),
        format!("{} `ProgramSpec` variants", metrics.program_specs),
        format!(
            "{} IEEE-754-safe active rules / {} total",
            metrics.egraph_safe_rules, metrics.egraph_total_rules
        ),
        "/data/projects/frankenjax".to_owned(),
        ".beads/issues.jsonl".to_owned(),
    ];
    for needle in &required_agents {
        if !agents.contains(needle) {
            errors.push(format!(
                "AGENTS.md missing live metric substring `{needle}`"
            ));
        }
    }

    let required_feature_parity = [
        format!("{} workspace crates", metrics.workspace_crates),
        format!(
            "{} canonical primitive variants",
            metrics.canonical_primitives
        ),
        format!("{} V1 local", metrics.v1_local_primitives),
        format!("{} pmap collectives", metrics.pmap_collectives),
        format!("{} ops", metrics.total_primitive_enum_variants),
        format!("{} `ProgramSpec` variants", metrics.program_specs),
        format!(
            "{} active IEEE-754-safe e-graph rules",
            metrics.egraph_safe_rules
        ),
    ];
    for needle in &required_feature_parity {
        if !feature_parity.contains(needle) {
            errors.push(format!(
                "docs/planning/FEATURE_PARITY.md missing live metric substring `{needle}`"
            ));
        }
    }

    let required_parity_coverage = [
        format!(
            "{} canonical primitive variants",
            metrics.canonical_primitives
        ),
        format!("{} V1 local", metrics.v1_local_primitives),
        format!("{} pmap collectives", metrics.pmap_collectives),
        format!(
            "{} total `Primitive` enum variants",
            metrics.total_primitive_enum_variants
        ),
    ];
    for needle in &required_parity_coverage {
        if !parity_coverage.contains(needle) {
            errors.push(format!(
                "docs/planning/PARITY-COVERAGE.md missing live metric substring `{needle}`"
            ));
        }
    }

    let required_spec = [
        format!("{} workspace member crates", metrics.workspace_crates),
        "/data/projects/frankenjax/legacy_jax_code/jax".to_owned(),
    ];
    for needle in &required_spec {
        if !comprehensive_spec.contains(needle) {
            errors.push(format!(
                "docs/planning/COMPREHENSIVE_SPEC_FOR_FRANKENJAX_V1.md missing `{needle}`"
            ));
        }
    }

    let banned_stale_strings = [
        "primitives-118_variants",
        "113%2F113",
        "118 Primitive variants",
        "118 primitive variants",
        "86 algebraic rewrite rules",
        "86 rewrite rules",
        "86 rules, equality saturation",
        "87 algebraic rewrite rules",
        "(110 ops)",
        "across 15 crates",
        "across 16 crates",
        "145+ pre-defined test programs",
        "/dp/frankenjax",
        ".beads/beads.jsonl",
    ];
    for (doc_name, content) in [
        ("README.md", readme),
        ("AGENTS.md", agents),
        ("docs/planning/FEATURE_PARITY.md", feature_parity),
        ("docs/planning/PARITY-COVERAGE.md", parity_coverage),
        (
            "docs/planning/COMPREHENSIVE_SPEC_FOR_FRANKENJAX_V1.md",
            comprehensive_spec,
        ),
    ] {
        for banned in banned_stale_strings {
            if content.contains(banned) {
                errors.push(format!(
                    "{doc_name} still contains stale metric/path substring `{banned}`"
                ));
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[test]
fn live_codebase_metrics_match_canonical_invariants() {
    let root = repo_root();
    let metrics = measure_live_metrics(&root);

    assert_eq!(metrics.workspace_crates, 17, "workspace crate count");
    assert_eq!(
        metrics.canonical_primitives, 162,
        "Primitive::ALL canonical count"
    );
    assert_eq!(
        metrics.pmap_collectives, 5,
        "Primitive::PMAP_COLLECTIVES count"
    );
    assert_eq!(metrics.v1_local_primitives, 157, "V1 local primitive count");
    assert_eq!(metrics.ffi_primitives, 1, "Primitive::FFI_PRIMITIVES count");
    assert_eq!(
        metrics.total_primitive_enum_variants,
        metrics.canonical_primitives + metrics.ffi_primitives,
        "Primitive enum variant count must equal Primitive::ALL + Primitive::FFI_PRIMITIVES"
    );
    assert_eq!(metrics.dtypes, 11, "DType enum variant count");
    assert_eq!(metrics.program_specs, 121, "ProgramSpec enum variant count");
    assert_eq!(
        metrics.egraph_safe_rules, 30,
        "fj_egraph::rules() safe rule count"
    );
    assert_eq!(
        metrics.egraph_unsafe_rules, 61,
        "fj_egraph::numerically_unsafe_rules() count"
    );
    assert_eq!(metrics.egraph_total_rules, 91, "total e-graph rule count");
}

#[test]
fn public_docs_match_live_codebase_metrics() {
    let root = repo_root();
    let metrics = measure_live_metrics(&root);

    let readme = fs::read_to_string(root.join("README.md")).expect("read README.md");
    let agents = fs::read_to_string(root.join("AGENTS.md")).expect("read AGENTS.md");
    let feature_parity = fs::read_to_string(root.join("docs/planning/FEATURE_PARITY.md"))
        .expect("read docs/planning/FEATURE_PARITY.md");
    let parity_coverage = fs::read_to_string(root.join("docs/planning/PARITY-COVERAGE.md"))
        .expect("read docs/planning/PARITY-COVERAGE.md");
    let comprehensive_spec =
        fs::read_to_string(root.join("docs/planning/COMPREHENSIVE_SPEC_FOR_FRANKENJAX_V1.md"))
            .expect("read docs/planning/COMPREHENSIVE_SPEC_FOR_FRANKENJAX_V1.md");

    if let Err(errors) = validate_doc_bundle(
        &metrics,
        &readme,
        &agents,
        &feature_parity,
        &parity_coverage,
        &comprehensive_spec,
    ) {
        panic!(
            "documentation metric drift detected:\n- {}",
            errors.join("\n- ")
        );
    }
}

#[test]
fn doc_metric_truth_gate_rejects_stale_metric_drift() {
    let root = repo_root();
    let metrics = measure_live_metrics(&root);

    let readme = fs::read_to_string(root.join("README.md")).expect("read README.md");
    let agents = fs::read_to_string(root.join("AGENTS.md")).expect("read AGENTS.md");
    let feature_parity = fs::read_to_string(root.join("docs/planning/FEATURE_PARITY.md"))
        .expect("read docs/planning/FEATURE_PARITY.md");
    let parity_coverage = fs::read_to_string(root.join("docs/planning/PARITY-COVERAGE.md"))
        .expect("read docs/planning/PARITY-COVERAGE.md");
    let comprehensive_spec =
        fs::read_to_string(root.join("docs/planning/COMPREHENSIVE_SPEC_FOR_FRANKENJAX_V1.md"))
            .expect("read docs/planning/COMPREHENSIVE_SPEC_FOR_FRANKENJAX_V1.md");

    let drifted_readme = readme.replace("162 canonical primitive", "118 Primitive variants");
    let errs = validate_doc_bundle(
        &metrics,
        &drifted_readme,
        &agents,
        &feature_parity,
        &parity_coverage,
        &comprehensive_spec,
    )
    .expect_err("validator must reject stale primitive count in README.md");
    assert!(
        errs.iter().any(|e| e.contains("README.md")),
        "expected README.md drift error, got: {errs:?}"
    );
}

#[test]
fn parity_validation_scripts_and_vision_evidence_map_are_green() {
    let root = repo_root();
    let temp_out = tempfile::tempdir().expect("create temp dir for parity validation outputs");

    let status = Command::new("bash")
        .arg(root.join("scripts/validate_feature_parity.sh"))
        .env("FJ_FEATURE_PARITY_OUT_DIR", temp_out.path())
        .current_dir(&root)
        .status()
        .expect("run scripts/validate_feature_parity.sh");
    assert!(
        status.success(),
        "scripts/validate_feature_parity.sh should exit 0"
    );

    let map_path = root.join("artifacts/conformance/vision_evidence_map.v1.json");
    let map_raw = fs::read_to_string(&map_path).expect("read vision_evidence_map.v1.json");
    let map_json: JsonValue =
        serde_json::from_str(&map_raw).expect("parse vision_evidence_map.v1.json");

    let summary = &map_json["summary"];
    assert_eq!(
        summary["red_count"].as_u64(),
        Some(0),
        "vision_evidence_map.v1.json must have 0 red claims"
    );
    assert_eq!(
        summary["yellow_count"].as_u64(),
        Some(0),
        "vision_evidence_map.v1.json must have 0 yellow claims"
    );
    assert_eq!(
        summary["green_count"].as_u64(),
        summary["total_claims"].as_u64(),
        "all claims in vision_evidence_map.v1.json must be green"
    );

    let docs = map_json["source_docs"]
        .as_array()
        .expect("source_docs array");
    for doc in docs {
        let rel = match doc["path"].as_str().expect("doc path") {
            "FEATURE_PARITY.md" => "docs/planning/FEATURE_PARITY.md",
            other => other,
        };
        let actual_sha = sha256_hex(&root.join(rel));
        let recorded_sha = doc["sha256"].as_str().expect("doc sha256");
        assert_eq!(
            recorded_sha, actual_sha,
            "vision_evidence_map.v1.json source_docs sha256 stale for {rel}"
        );
    }
}
