#![forbid(unsafe_code)]

use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_json(path: &Path) -> Value {
    let raw = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|err| panic!("failed to parse {}: {err}", path.display()))
}

fn validate_schema_examples(name: &str) {
    let root = repo_root();
    let schema_path = root.join(format!("artifacts/schemas/{name}.schema.json"));
    let valid_path = root.join(format!("artifacts/examples/{name}.example.json"));
    let invalid_path = root.join(format!(
        "artifacts/examples/invalid/{name}.missing-required.example.json"
    ));

    let schema = read_json(&schema_path);
    let validator = jsonschema::validator_for(&schema)
        .unwrap_or_else(|err| panic!("schema {} failed to compile: {err}", schema_path.display()));

    let valid_instance = read_json(&valid_path);
    let valid_errors = validator
        .iter_errors(&valid_instance)
        .map(|err| err.to_string())
        .collect::<Vec<_>>();
    assert!(
        valid_errors.is_empty(),
        "expected valid example {} to pass validation, got errors: {}",
        valid_path.display(),
        valid_errors.join(" | ")
    );

    let invalid_instance = read_json(&invalid_path);
    let invalid_errors = validator
        .iter_errors(&invalid_instance)
        .map(|err| err.to_string())
        .collect::<Vec<_>>();
    assert!(
        !invalid_errors.is_empty(),
        "expected invalid example {} to fail validation",
        invalid_path.display()
    );
}

fn validate_schema_instance(schema_name: &str, instance_rel_path: &str) {
    let root = repo_root();
    let schema_path = root.join(format!("artifacts/schemas/{schema_name}.schema.json"));
    let instance_path = root.join(instance_rel_path);
    let schema = read_json(&schema_path);
    let instance = read_json(&instance_path);
    let validator = jsonschema::validator_for(&schema)
        .unwrap_or_else(|err| panic!("schema {} failed to compile: {err}", schema_path.display()));
    let errors = validator
        .iter_errors(&instance)
        .map(|err| err.to_string())
        .collect::<Vec<_>>();
    assert!(
        errors.is_empty(),
        "expected instance {} to pass {} validation, got errors: {}",
        instance_path.display(),
        schema_name,
        errors.join(" | ")
    );
}

#[test]
fn all_v1_artifact_schemas_have_valid_and_invalid_examples() {
    let schemas = [
        "legacy_anchor_map.v1",
        "contract_table.v1",
        "fixture_manifest.v1",
        "parity_gate.v1",
        "risk_note.v1",
        "compatibility_matrix.v1",
        "test_log.v1",
        "failure_diagnostic.v1",
    ];

    for schema_name in schemas {
        validate_schema_examples(schema_name);
    }
}

#[test]
fn canonical_phase2c_security_artifacts_validate_against_v1_schemas() {
    validate_schema_instance(
        "compatibility_matrix.v1",
        "artifacts/phase2c/global/compatibility_matrix.v1.json",
    );
    validate_schema_instance(
        "legacy_anchor_map.v1",
        "artifacts/phase2c/FJ-P2C-FOUNDATION/legacy_anchor_map.v1.json",
    );
    validate_schema_instance(
        "legacy_anchor_map.v1",
        "artifacts/phase2c/FJ-P2C-001/legacy_anchor_map.v1.json",
    );
    validate_schema_instance(
        "contract_table.v1",
        "artifacts/phase2c/FJ-P2C-001/contract_table.v1.json",
    );
    validate_schema_instance(
        "risk_note.v1",
        "artifacts/phase2c/FJ-P2C-FOUNDATION/risk_note.v1.json",
    );
}

#[test]
fn all_phase2c_packets_have_normative_artifact_topology() {
    let root = repo_root();
    let required_files = [
        "legacy_anchor_map.v1.json",
        "contract_table.v1.json",
        "fixture_manifest.json",
        "parity_gate.yaml",
        "risk_note.md",
        "parity_report.json",
        "parity_report.raptorq.json",
        "parity_report.decode_proof.json",
    ];

    for packet_number in 1..=8 {
        let packet_id = format!("FJ-P2C-{packet_number:03}");
        let packet_dir = root.join("artifacts/phase2c").join(&packet_id);
        for required_file in required_files {
            let path = packet_dir.join(required_file);
            assert!(path.exists(), "missing {}", path.display());
        }

        let manifest = read_json(&packet_dir.join("fixture_manifest.json"));
        assert_eq!(
            manifest["schema_version"], "frankenjax.fixture-manifest.v1",
            "bad fixture manifest schema for {packet_id}"
        );
        assert_eq!(manifest["packet_id"], packet_id);
        assert!(
            manifest["fixtures"]
                .as_array()
                .is_some_and(|items| !items.is_empty()),
            "fixture manifest must list packet evidence for {packet_id}"
        );

        let parity_report = read_json(&packet_dir.join("parity_report.json"));
        assert_eq!(
            parity_report["schema_version"], "frankenjax.phase2c-parity-report.v1",
            "bad parity report schema for {packet_id}"
        );
        assert_eq!(parity_report["packet_id"], packet_id);
        assert_eq!(
            parity_report["status"], "pass_with_tracked_residual_risks",
            "packet parity report must preserve tracked residual risk status for {packet_id}"
        );

        let sidecar = read_json(&packet_dir.join("parity_report.raptorq.json"));
        assert_eq!(
            sidecar["schema_version"], "frankenjax.sidecar.v1",
            "bad RaptorQ sidecar schema for {packet_id}"
        );

        let proof = read_json(&packet_dir.join("parity_report.decode_proof.json"));
        assert!(
            proof["recovered"].as_bool().unwrap_or(false),
            "decode proof did not recover for {packet_id}"
        );

        let gate = fs::read_to_string(packet_dir.join("parity_gate.yaml"))
            .unwrap_or_else(|err| panic!("failed to read parity gate for {packet_id}: {err}"));
        assert!(
            gate.contains("overall_status: pass"),
            "parity gate must record pass status for {packet_id}"
        );
        assert!(
            gate.contains("G2-durability"),
            "parity gate must include durability gate for {packet_id}"
        );

        let risk_note = fs::read_to_string(packet_dir.join("risk_note.md"))
            .unwrap_or_else(|err| panic!("failed to read risk note for {packet_id}: {err}"));
        assert!(
            risk_note.contains("Tracking bead:"),
            "risk note must link residual risk to a bead for {packet_id}"
        );
    }
}

#[test]
fn global_performance_gate_covers_required_phases() {
    let root = repo_root();
    let gate_path = root.join("artifacts/performance/global_performance_gate.v1.json");
    let gate = read_json(&gate_path);
    assert_eq!(
        gate["schema_version"], "frankenjax.global-performance-gate.v1",
        "global performance gate schema marker changed"
    );
    assert_eq!(
        gate["baseline_ref"], "artifacts/performance/benchmark_baselines_v2_2026-03-12.json",
        "global performance gate must point at the normalized baseline artifact"
    );
    assert_eq!(
        gate["regression_threshold_percent"], 5.0,
        "global performance gate must preserve the 5% regression threshold"
    );
    assert!(
        root.join(
            gate["baseline_ref"]
                .as_str()
                .expect("baseline ref must be a string")
        )
        .exists(),
        "global performance gate baseline artifact does not exist"
    );
    assert!(
        root.join(
            gate["regression_report_ref"]
                .as_str()
                .expect("regression report ref must be a string")
        )
        .exists(),
        "global performance gate regression report artifact does not exist"
    );

    let policy = gate["policy"]
        .as_object()
        .expect("policy must be an object");
    for key in [
        "profile_first",
        "one_optimization_lever_per_change",
        "behavior_proof_required",
        "risk_note_required_for_regression",
    ] {
        assert_eq!(
            policy.get(key).and_then(Value::as_bool),
            Some(true),
            "policy flag {key} must be true"
        );
    }

    let phases = gate["phases"].as_array().expect("phases must be an array");
    let phase_ids = phases
        .iter()
        .map(|phase| {
            phase["phase_id"]
                .as_str()
                .expect("phase_id must be a string")
                .to_owned()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        phase_ids,
        BTreeSet::from([
            "cold_cache".to_owned(),
            "compile_dispatch".to_owned(),
            "execute".to_owned(),
            "memory".to_owned(),
            "trace".to_owned(),
            "warm_cache".to_owned(),
        ]),
        "global performance gate must cover every required phase exactly once"
    );

    for phase in phases {
        let phase_id = phase["phase_id"]
            .as_str()
            .expect("phase_id must be a string");
        let status = phase["status"].as_str().expect("status must be a string");
        let benchmarks = phase["benchmarks"]
            .as_array()
            .expect("benchmarks must be an array");

        match status {
            "measured" => {
                assert!(
                    !benchmarks.is_empty(),
                    "measured phase {phase_id} must reference at least one benchmark"
                );
                for benchmark in benchmarks {
                    assert!(
                        benchmark["suite"]
                            .as_str()
                            .is_some_and(|suite| !suite.is_empty()),
                        "benchmark suite must be non-empty for phase {phase_id}"
                    );
                    assert!(
                        benchmark["bench_id"]
                            .as_str()
                            .is_some_and(|id| !id.is_empty()),
                        "benchmark id must be non-empty for phase {phase_id}"
                    );
                    assert!(
                        benchmark["p50_ns"]
                            .as_f64()
                            .is_some_and(|value| value > 0.0),
                        "benchmark p50_ns must be positive for phase {phase_id}"
                    );
                    assert!(
                        benchmark["p95_ns"]
                            .as_f64()
                            .is_some_and(|value| value > 0.0),
                        "benchmark p95_ns must be positive for phase {phase_id}"
                    );
                }
            }
            "not_measured" => {
                assert_eq!(
                    phase_id, "memory",
                    "only memory may remain an explicit unmeasured coverage gap"
                );
                assert!(
                    benchmarks.is_empty(),
                    "unmeasured phase {phase_id} must not invent benchmark values"
                );
                assert!(
                    phase["required_next_evidence"]
                        .as_array()
                        .is_some_and(|items| !items.is_empty()),
                    "unmeasured phase {phase_id} must list next evidence requirements"
                );
            }
            other => panic!("unexpected performance phase status {other:?} for {phase_id}"),
        }
    }
}
