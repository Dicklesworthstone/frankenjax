#![forbid(unsafe_code)]

use fj_conformance::error_taxonomy::{ErrorTaxonomyReport, validate_error_taxonomy_report};
use fj_conformance::transform_control_flow::{
    TransformControlFlowReport, validate_transform_control_flow_report,
};
use fj_conformance::ttl_semantic::{TtlSemanticReport, validate_ttl_semantic_report};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_json(path: &Path) -> Value {
    let raw = fs::read_to_string(path).expect("failed to read JSON artifact");
    serde_json::from_str(&raw).expect("failed to parse JSON artifact")
}

fn validate_schema_examples(name: &str) {
    let root = repo_root();
    let schema_path = root.join(format!("artifacts/schemas/{name}.schema.json"));
    let valid_path = root.join(format!("artifacts/examples/{name}.example.json"));
    let invalid_path = root.join(format!(
        "artifacts/examples/invalid/{name}.missing-required.example.json"
    ));

    let schema = read_json(&schema_path);
    let validator = jsonschema::validator_for(&schema).expect("schema failed to compile");

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
    let validator = jsonschema::validator_for(&schema).expect("schema failed to compile");
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
        "e2e_forensic_log.v1",
        "failure_diagnostic.v1",
        "vision_evidence_map.v1",
        "security_threat_model.v1",
        "onboarding_command_inventory.v1",
        "decision_ledger_calibration.v1",
        "numerical_stability_matrix.v1",
        "optimization_hotspot_scoreboard.v1",
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
            .expect("failed to read parity gate");
        assert!(
            gate.contains("overall_status: pass"),
            "parity gate must record pass status for {packet_id}"
        );
        assert!(
            gate.contains("G2-durability"),
            "parity gate must include durability gate for {packet_id}"
        );

        let risk_note =
            fs::read_to_string(packet_dir.join("risk_note.md")).expect("failed to read risk note");
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
    let memory_gate_ref = gate["memory_gate_ref"]
        .as_str()
        .expect("memory gate ref must be a string");
    let memory_gate_path = root.join(memory_gate_ref);
    assert!(
        memory_gate_path.exists(),
        "global performance gate memory artifact does not exist"
    );
    let memory_gate = read_json(&memory_gate_path);
    assert_eq!(
        memory_gate["schema_version"], "frankenjax.memory-performance-gate.v1",
        "memory performance gate schema marker changed"
    );
    assert_eq!(
        memory_gate["status"], "pass",
        "memory performance gate must pass before the global gate can reference it"
    );
    let hotspot_scoreboard_ref = gate["hotspot_scoreboard_ref"]
        .as_str()
        .expect("hotspot scoreboard ref must be a string");
    let hotspot_scoreboard_path = root.join(hotspot_scoreboard_ref);
    assert!(
        hotspot_scoreboard_path.exists(),
        "global performance gate hotspot scoreboard artifact does not exist"
    );
    let hotspot_scoreboard = read_json(&hotspot_scoreboard_path);
    assert_eq!(
        hotspot_scoreboard["schema_version"], "frankenjax.optimization-hotspot-scoreboard.v1",
        "optimization hotspot scoreboard schema marker changed"
    );
    assert_eq!(
        hotspot_scoreboard["status"], "pass",
        "optimization hotspot scoreboard must pass before the global gate can reference it"
    );

    let policy = gate["policy"]
        .as_object()
        .expect("policy must be an object");
    for key in [
        "profile_first",
        "one_optimization_lever_per_change",
        "behavior_proof_required",
        "risk_note_required_for_regression",
        "hotspot_scoreboard_required",
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

        assert_ne!(
            status, "not_measured",
            "performance phase {phase_id} is still unmeasured; memory must be covered by the RSS gate"
        );
        assert_eq!(
            status, "measured",
            "unexpected performance phase status {status:?} for {phase_id}"
        );
        if status == "measured" {
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
                if phase_id == "memory" {
                    assert!(
                        benchmark["peak_rss_bytes"]
                            .as_u64()
                            .is_some_and(|value| value > 0),
                        "memory benchmark must record non-zero peak RSS"
                    );
                } else {
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
        }

        if phase_id == "memory" {
            assert_eq!(
                phase["gate_kind"], "rss_budget",
                "memory phase must use the RSS budget gate"
            );
            assert_eq!(
                phase["report_ref"], gate["memory_gate_ref"],
                "memory phase must point at the memory gate artifact"
            );
            let memory_workload_ids = memory_gate["workloads"]
                .as_array()
                .expect("memory gate workloads must be an array")
                .iter()
                .map(|workload| {
                    workload["workload_id"]
                        .as_str()
                        .expect("memory workload id must be a string")
                        .to_owned()
                })
                .collect::<BTreeSet<_>>();
            for benchmark in benchmarks {
                let bench_id = benchmark["bench_id"]
                    .as_str()
                    .expect("memory bench id must be a string");
                assert!(
                    memory_workload_ids.contains(bench_id),
                    "memory benchmark {bench_id} must come from the memory gate workloads"
                );
                assert!(
                    benchmark["peak_rss_bytes"]
                        .as_u64()
                        .is_some_and(|value| value > 0),
                    "memory benchmark must record non-zero peak RSS"
                );
                assert!(
                    benchmark["measurement_backend"]
                        .as_str()
                        .is_some_and(|backend| backend != "not_measured" && !backend.is_empty()),
                    "memory benchmark must record concrete measurement backend"
                );
            }
        }
    }
}

#[test]
fn optimization_hotspot_scoreboard_validates_against_schema() {
    let root = repo_root();
    let report_path = root.join("artifacts/performance/optimization_hotspot_scoreboard.v1.json");
    let report = read_json(&report_path);
    assert_eq!(
        report["schema_version"], "frankenjax.optimization-hotspot-scoreboard.v1",
        "optimization hotspot scoreboard schema marker changed"
    );
    assert_eq!(
        report["bead_id"], "frankenjax-cstq.11",
        "optimization hotspot scoreboard must stay bound to frankenjax-cstq.11"
    );
    assert_eq!(report["status"], "pass");
    validate_schema_instance(
        "optimization_hotspot_scoreboard.v1",
        "artifacts/performance/optimization_hotspot_scoreboard.v1.json",
    );
}

#[test]
fn transform_control_flow_matrix_artifact_covers_required_rows() {
    let root = repo_root();
    let matrix_path = root.join("artifacts/conformance/transform_control_flow_matrix.v1.json");
    let matrix = read_json(&matrix_path);
    assert_eq!(
        matrix["schema_version"], "frankenjax.transform-control-flow-matrix.v1",
        "transform control-flow matrix schema marker changed"
    );
    assert_eq!(
        matrix["bead_id"], "frankenjax-cstq.2",
        "matrix must stay bound to the closing bead"
    );
    assert_eq!(
        matrix["status"], "pass",
        "transform control-flow matrix must pass"
    );

    let parsed: TransformControlFlowReport =
        serde_json::from_value(matrix).expect("matrix artifact should parse");
    let issues = validate_transform_control_flow_report(&parsed);
    assert!(issues.is_empty(), "matrix validation issues: {issues:#?}");

    let case_ids = parsed
        .cases
        .iter()
        .map(|case| case.case_id.as_str())
        .collect::<BTreeSet<_>>();
    for required in [
        "jit_vmap_grad_cond_false",
        "jit_vmap_grad_scan_mul",
        "vmap_grad_while_mul",
        "value_and_grad_multi_output",
        "jacobian_quadratic",
        "hessian_quadratic",
        "scan_multi_carry_state",
        "vmap_multi_output_return",
        "jit_mixed_dtype_add",
        "grad_vmap_vector_output_fail_closed",
        "vmap_empty_batch_fail_closed",
        "vmap_out_axes_none_nonconstant_fail_closed",
    ] {
        assert!(case_ids.contains(required), "missing matrix row {required}");
    }

    let sentinel_ids = parsed
        .performance_sentinels
        .iter()
        .map(|sentinel| sentinel.workload_id.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        sentinel_ids,
        BTreeSet::from([
            "perf_batched_switch",
            "perf_jit_vmap_grad_cond",
            "perf_vmap_scan_loop_stack",
            "perf_vmap_while_loop_stack",
        ]),
        "performance sentinels must cover the required transform-control-flow workloads"
    );
    assert!(
        parsed
            .performance_sentinels
            .iter()
            .all(|sentinel| sentinel.p50_ns > 0
                && sentinel.p95_ns >= sentinel.p50_ns
                && sentinel.p99_ns >= sentinel.p95_ns
                && sentinel.peak_rss_bytes.is_some_and(|rss| rss > 0)),
        "performance sentinels must record ordered p50/p95/p99 and non-zero peak RSS"
    );
}

#[test]
fn error_taxonomy_matrix_artifact_covers_required_rows() {
    let root = repo_root();
    let matrix_path = root.join("artifacts/conformance/error_taxonomy_matrix.v1.json");
    let matrix = read_json(&matrix_path);
    assert_eq!(
        matrix["schema_version"], "frankenjax.error-taxonomy-matrix.v1",
        "error taxonomy matrix schema marker changed"
    );
    assert_eq!(
        matrix["bead_id"], "frankenjax-cstq.8",
        "matrix must stay bound to the error taxonomy bead"
    );
    assert_eq!(matrix["status"], "pass", "error taxonomy matrix must pass");

    let parsed: ErrorTaxonomyReport =
        serde_json::from_value(matrix).expect("matrix artifact should parse");
    let issues = validate_error_taxonomy_report(&parsed);
    assert!(issues.is_empty(), "matrix validation issues: {issues:#?}");

    let case_ids = parsed
        .cases
        .iter()
        .map(|case| case.case_id.as_str())
        .collect::<BTreeSet<_>>();
    for required in [
        "ir_validation_unknown_outvar",
        "transform_proof_duplicate_evidence",
        "transform_proof_missing_evidence",
        "primitive_arity_add",
        "primitive_shape_add_broadcast",
        "primitive_type_sin_bool",
        "interpreter_missing_variable",
        "cache_strict_unknown_feature",
        "cache_hardened_unknown_feature",
        "vmap_axis_mismatch",
        "durability_missing_artifact",
        "unsupported_transform_tail_without_fallback",
        "unsupported_control_flow_grad_vmap_vector",
    ] {
        assert!(case_ids.contains(required), "missing matrix row {required}");
    }
    assert_eq!(
        parsed.coverage.panic_free_count,
        parsed.cases.len(),
        "every taxonomy row must be panic-free"
    );
    assert!(
        parsed
            .cases
            .iter()
            .all(|case| case.expected_error_class == case.actual_error_class
                && !case.replay_command.trim().is_empty()
                && !case.evidence_refs.is_empty()),
        "rows must have exact typed classes, replay commands, and evidence refs"
    );
}

#[test]
fn ttl_semantic_matrix_artifact_covers_required_rows() {
    let root = repo_root();
    let matrix_path = root.join("artifacts/conformance/ttl_semantic_proof_matrix.v1.json");
    let matrix = read_json(&matrix_path);
    assert_eq!(
        matrix["schema_version"], "frankenjax.ttl-semantic-proof-matrix.v1",
        "TTL semantic matrix schema marker changed"
    );
    assert_eq!(
        matrix["bead_id"], "frankenjax-cstq.3",
        "matrix must stay bound to the TTL semantic verifier bead"
    );
    assert_eq!(matrix["status"], "pass", "TTL semantic matrix must pass");

    let parsed: TtlSemanticReport =
        serde_json::from_value(matrix).expect("matrix artifact should parse");
    let issues = validate_ttl_semantic_report(&parsed);
    assert!(issues.is_empty(), "matrix validation issues: {issues:#?}");

    let case_ids = parsed
        .cases
        .iter()
        .map(|case| case.case_id.as_str())
        .collect::<BTreeSet<_>>();
    for required in [
        "valid_single_jit_square",
        "valid_single_grad_square",
        "valid_single_vmap_square",
        "valid_jit_grad_fixture",
        "valid_grad_jit_order_sensitive",
        "valid_vmap_grad_fixture",
        "fail_closed_grad_vmap_vector_output",
        "invalid_duplicate_evidence",
        "invalid_missing_evidence",
        "invalid_stale_input_fingerprint",
        "invalid_wrong_transform_binding",
        "invalid_missing_fixture_link",
    ] {
        assert!(case_ids.contains(required), "missing matrix row {required}");
    }

    assert_eq!(
        parsed.coverage.accepted_count, 6,
        "semantic proof gate must keep the valid transform replay rows accepted"
    );
    assert!(
        parsed.coverage.rejected_count >= 6,
        "semantic proof gate must keep invalid/fail-closed rows rejected"
    );
    assert!(
        parsed
            .cases
            .iter()
            .filter(|case| case.verifier_decision == "accept")
            .all(
                |case| case.output_fingerprint == case.expected_output_fingerprint
                    && !case.output_shapes.is_empty()
                    && !case.output_dtypes.is_empty()
            ),
        "accepted rows must bind output fingerprints, shapes, and dtypes"
    );
}

#[test]
fn vision_evidence_map_artifact_validates_against_schema() {
    let root = repo_root();
    let map_path = root.join("artifacts/conformance/vision_evidence_map.v1.json");
    if !map_path.exists() {
        return;
    }
    validate_schema_instance(
        "vision_evidence_map.v1",
        "artifacts/conformance/vision_evidence_map.v1.json",
    );
    let map = read_json(&map_path);
    assert_eq!(
        map["schema_version"], "frankenjax.vision-evidence-map.v1",
        "vision evidence map schema marker changed"
    );
    assert_eq!(
        map["bead_id"], "frankenjax-cstq.14",
        "map must stay bound to the vision-evidence bootstrap bead"
    );
    let claims = map["claims"].as_array().expect("claims must be an array");
    assert!(!claims.is_empty(), "map must contain at least one claim");
    for claim in claims {
        let status = claim["status"].as_str().unwrap();
        assert!(
            ["red", "yellow", "green"].contains(&status),
            "claim status must be red, yellow, or green"
        );
        let evidence = claim["evidence"]
            .as_array()
            .expect("evidence must be array");
        assert!(
            !(status == "green" && evidence.is_empty()),
            "green claim {} must have at least one evidence reference",
            claim["claim_id"]
        );
    }
    let summary = &map["summary"];
    let total = summary["total_claims"].as_i64().unwrap();
    let red = summary["red_count"].as_i64().unwrap();
    let yellow = summary["yellow_count"].as_i64().unwrap();
    let green = summary["green_count"].as_i64().unwrap();
    assert_eq!(
        total,
        red + yellow + green,
        "summary counts must add up to total"
    );
}

#[test]
fn security_threat_model_artifact_validates_against_schema() {
    let root = repo_root();
    let model_path = root.join("artifacts/conformance/security_threat_model.v1.json");
    if !model_path.exists() {
        return;
    }
    validate_schema_instance(
        "security_threat_model.v1",
        "artifacts/conformance/security_threat_model.v1.json",
    );
    let model = read_json(&model_path);
    assert_eq!(
        model["schema_version"], "frankenjax.security-threat-model.v1",
        "security threat model schema marker changed"
    );
    assert_eq!(
        model["bead_id"], "frankenjax-cstq.17",
        "model must stay bound to the security gate bead"
    );
    let categories = model["threat_categories"]
        .as_array()
        .expect("threat_categories must be an array");
    assert!(
        !categories.is_empty(),
        "model must contain at least one threat category"
    );
    for category in categories {
        let evidence_status = category["evidence_status"].as_str().unwrap();
        assert!(
            ["red", "yellow", "green"].contains(&evidence_status),
            "category evidence_status must be red, yellow, or green"
        );
        let evidence_refs = category["evidence_refs"]
            .as_array()
            .expect("evidence_refs must be array");
        assert!(
            !(evidence_status == "green" && evidence_refs.is_empty()),
            "green category {} must have at least one evidence ref",
            category["category_id"]
        );
    }
    let summary = &model["summary"];
    let total = summary["total_categories"].as_i64().unwrap();
    let green = summary["evidence_green"].as_i64().unwrap();
    let yellow = summary["evidence_yellow"].as_i64().unwrap();
    let red = summary["evidence_red"].as_i64().unwrap();
    assert_eq!(
        total,
        green + yellow + red,
        "summary counts must add up to total_categories"
    );
}

#[test]
fn onboarding_command_inventory_validates_against_schema() {
    let root = repo_root();
    let inv_path = root.join("artifacts/conformance/onboarding_command_inventory.v1.json");
    if !inv_path.exists() {
        return;
    }
    validate_schema_instance(
        "onboarding_command_inventory.v1",
        "artifacts/conformance/onboarding_command_inventory.v1.json",
    );
    let inv = read_json(&inv_path);
    assert_eq!(
        inv["schema_version"], "frankenjax.onboarding-command-inventory.v1",
        "onboarding command inventory schema marker changed"
    );
    assert_eq!(
        inv["bead_id"], "frankenjax-cstq.18",
        "inventory must stay bound to the onboarding bead"
    );
    let commands = inv["commands"].as_array();
    assert!(commands.is_some(), "commands must be an array");
    let Some(commands) = commands else {
        return;
    };
    assert!(
        !commands.is_empty(),
        "inventory must contain at least one command"
    );
    for cmd in commands {
        let evidence_status = cmd["evidence_status"].as_str();
        assert!(
            evidence_status.is_some_and(|status| ["red", "yellow", "green"].contains(&status)),
            "command evidence_status must be red, yellow, or green"
        );
        let replay_cmd = cmd["replay_command"].as_str().unwrap_or_default();
        assert!(
            !replay_cmd.is_empty(),
            "command {} must have a replay_command",
            cmd["command_id"]
        );
    }
    let summary = &inv["summary"];
    let status_total = summary["green_count"]
        .as_i64()
        .zip(summary["yellow_count"].as_i64())
        .zip(summary["red_count"].as_i64())
        .map(|((green, yellow), red)| green + yellow + red);
    assert_eq!(
        summary["total_commands"].as_i64(),
        status_total,
        "summary counts must add up to total_commands"
    );
}

#[test]
fn decision_ledger_calibration_validates_against_schema() {
    let root = repo_root();
    let report_path = root.join("artifacts/conformance/decision_ledger_calibration.v1.json");
    if !report_path.exists() {
        return;
    }
    validate_schema_instance(
        "decision_ledger_calibration.v1",
        "artifacts/conformance/decision_ledger_calibration.v1.json",
    );
    let report = read_json(&report_path);
    assert_eq!(
        report["schema_version"], "frankenjax.decision-ledger-calibration.v1",
        "decision ledger calibration schema marker changed"
    );
    assert_eq!(
        report["bead_id"], "frankenjax-cstq.19",
        "decision ledger report must stay bound to the ledger bead"
    );
    let rows = report["rows"].as_array();
    assert!(rows.is_some(), "rows must be an array");
    let Some(rows) = rows else {
        return;
    };
    assert!(!rows.is_empty(), "decision ledger report needs rows");
    for row in rows {
        assert!(
            row["alternatives_considered"]
                .as_array()
                .is_some_and(|alternatives| alternatives.len() >= 2),
            "decision row must contain at least two alternatives"
        );
        assert!(
            row["evidence_signals"]
                .as_array()
                .is_some_and(|signals| !signals.is_empty()),
            "decision row must contain evidence signals"
        );
        assert!(
            !row["replay_command"]
                .as_str()
                .unwrap_or_default()
                .is_empty(),
            "decision row must have replay command"
        );
    }
}

#[test]
fn numerical_stability_matrix_validates_against_schema() {
    let root = repo_root();
    let report_path = root.join("artifacts/conformance/numerical_stability_matrix.v1.json");
    if !report_path.exists() {
        return;
    }
    validate_schema_instance(
        "numerical_stability_matrix.v1",
        "artifacts/conformance/numerical_stability_matrix.v1.json",
    );
    let report = read_json(&report_path);
    assert_eq!(
        report["schema_version"], "frankenjax.numerical-stability.v1",
        "numerical stability schema marker changed"
    );
    assert_eq!(
        report["bead_id"], "frankenjax-cstq.20",
        "numerical stability report must stay bound to the stability bead"
    );
    let rows = report["rows"].as_array();
    assert!(rows.is_some(), "rows must be an array");
    let Some(rows) = rows else {
        return;
    };
    assert!(!rows.is_empty(), "numerical stability report needs rows");
    for row in rows {
        assert!(
            !row["tolerance_policy_id"]
                .as_str()
                .unwrap_or_default()
                .is_empty(),
            "stability row must name a tolerance policy"
        );
        assert!(
            !row["platform_fingerprint_id"]
                .as_str()
                .unwrap_or_default()
                .is_empty(),
            "stability row must bind platform metadata"
        );
        assert!(
            row["artifact_refs"]
                .as_array()
                .is_some_and(|artifacts| !artifacts.is_empty()),
            "stability row must contain artifact refs"
        );
        assert!(
            !row["replay_command"]
                .as_str()
                .unwrap_or_default()
                .is_empty(),
            "stability row must have replay command"
        );
    }
}
