#![forbid(unsafe_code)]

use fj_conformance::{HarnessConfig, read_transform_fixture_bundle, run_transform_fixture_bundle};
use fj_core::{
    Atom, CompatibilityMode, DType, Equation, Jaxpr, Literal, Primitive, ProgramSpec, Shape,
    TensorValue, TraceTransformLedger, Transform, Value, VarId, build_program,
    verify_transform_composition,
};
use fj_dispatch::{DispatchError, DispatchRequest, TransformExecutionError, dispatch};
use serde::Serialize;
use serde_json::{Value as JsonValue, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize)]
struct E2EForensicLogV1 {
    schema_version: &'static str,
    scenario_id: String,
    test_id: String,
    packet_id: String,
    fixture_id: String,
    seed: Option<u64>,
    mode: String,
    ts_utc_unix_ms: u128,
    env: JsonValue,
    artifact_refs: Vec<String>,
    replay_command: String,
    input_capture: JsonValue,
    intermediate_states: Vec<JsonValue>,
    output_capture: Option<JsonValue>,
    result: String,
    duration_ms: u128,
    details: Option<String>,
}

#[derive(Debug)]
struct ScenarioOutcome {
    output_capture: JsonValue,
    artifact_refs: Vec<String>,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn artifact_dir() -> PathBuf {
    if let Ok(path) = std::env::var("FJ_E2E_ARTIFACT_DIR") {
        return PathBuf::from(path);
    }
    repo_root().join("artifacts/e2e")
}

fn replay_command(scenario_id: &str) -> String {
    format!(
        "cargo test -p fj-conformance --test e2e -- {} --exact --nocapture",
        scenario_id
    )
}

fn fixture_id(input_capture: &JsonValue) -> String {
    fj_test_utils::fixture_id_from_json(input_capture)
        .unwrap_or_else(|_| "fixture-id-error".to_owned())
}

fn env_fingerprint() -> JsonValue {
    json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "rust_toolchain": std::env::var("RUSTUP_TOOLCHAIN").unwrap_or_else(|_| "unknown".to_owned()),
    })
}

fn write_log(log: &E2EForensicLogV1) {
    let dir = artifact_dir();
    fs::create_dir_all(&dir).expect("e2e artifact dir should be creatable");
    let path = dir.join(format!("{}.e2e.json", log.scenario_id));
    let raw = serde_json::to_string_pretty(log).expect("log serialization should succeed");
    if std::env::var_os("FJ_E2E_ARTIFACT_DIR").is_none()
        && let Ok(existing_raw) = fs::read_to_string(&path)
        && let Ok(existing) = serde_json::from_str::<JsonValue>(&existing_raw)
        && existing.get("scenario_id").and_then(JsonValue::as_str) == Some(&log.scenario_id)
        && existing.get("fixture_id").and_then(JsonValue::as_str) == Some(&log.fixture_id)
        && existing.get("result").and_then(JsonValue::as_str) == Some(&log.result)
    {
        return;
    }
    fs::write(&path, raw).expect("e2e forensic log write should succeed");
}

fn run_e2e_scenario(
    scenario_id: &str,
    packet_id: &str,
    mode: CompatibilityMode,
    input_capture: JsonValue,
    run: impl FnOnce(&mut Vec<JsonValue>) -> Result<ScenarioOutcome, String>,
) {
    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after epoch")
        .as_millis();
    let start = Instant::now();
    let mut states = Vec::new();

    let result = run(&mut states);
    let duration_ms = start.elapsed().as_millis();

    let mut log = E2EForensicLogV1 {
        schema_version: "frankenjax.e2e.log.v1",
        scenario_id: scenario_id.to_owned(),
        test_id: format!("{}::{}", module_path!(), scenario_id),
        packet_id: packet_id.to_owned(),
        fixture_id: fixture_id(&input_capture),
        seed: None,
        mode: match mode {
            CompatibilityMode::Strict => "strict".to_owned(),
            CompatibilityMode::Hardened => "hardened".to_owned(),
        },
        ts_utc_unix_ms: started,
        env: env_fingerprint(),
        artifact_refs: Vec::new(),
        replay_command: replay_command(scenario_id),
        input_capture,
        intermediate_states: states,
        output_capture: None,
        result: "fail".to_owned(),
        duration_ms,
        details: None,
    };

    match result {
        Ok(outcome) => {
            log.result = "pass".to_owned();
            log.output_capture = Some(outcome.output_capture);
            log.artifact_refs = outcome.artifact_refs;
            write_log(&log);
        }
        Err(detail) => {
            log.details = Some(detail.clone());
            write_log(&log);
            panic!("{} failed: {}", scenario_id, detail);
        }
    }
}

fn ledger(program: ProgramSpec, transforms: &[Transform]) -> TraceTransformLedger {
    let mut ttl = TraceTransformLedger::new(build_program(program));
    for (idx, transform) in transforms.iter().enumerate() {
        ttl.push_transform(*transform, format!("e2e-{}-{}", idx, transform.as_str()));
    }
    ttl
}

#[test]
fn e2e_p2c001_full_dispatch_pipeline() {
    let cfg = HarnessConfig::default_paths();
    let fixture_path = cfg
        .fixture_root
        .join("transforms")
        .join("legacy_transform_cases.v1.json");

    run_e2e_scenario(
        "e2e_p2c001_full_dispatch_pipeline",
        "P2C-001",
        CompatibilityMode::Strict,
        json!({
            "fixture_path": fixture_path.display().to_string(),
            "strict_mode": true,
        }),
        |states| {
            states.push(json!({
                "step": "fixture_exists",
                "exists": fixture_path.exists(),
            }));

            let bundle = read_transform_fixture_bundle(&fixture_path)
                .map_err(|err| format!("failed to load fixture bundle: {err}"))?;
            states.push(json!({
                "step": "bundle_loaded",
                "case_count": bundle.cases.len(),
                "schema_version": bundle.schema_version,
            }));

            let report = run_transform_fixture_bundle(&cfg, &bundle);
            states.push(json!({
                "step": "parity_run_complete",
                "matched_cases": report.matched_cases,
                "mismatched_cases": report.mismatched_cases,
            }));

            if report.total_cases != bundle.cases.len() {
                return Err(format!(
                    "report case count mismatch: report={} bundle={}",
                    report.total_cases,
                    bundle.cases.len()
                ));
            }
            if report.mismatched_cases != 0 {
                return Err(format!(
                    "expected zero mismatches, got {}",
                    report.mismatched_cases
                ));
            }

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "total_cases": report.total_cases,
                    "matched_cases": report.matched_cases,
                    "mismatched_cases": report.mismatched_cases,
                }),
                artifact_refs: vec![fixture_path.display().to_string()],
            })
        },
    );
}

#[test]
fn e2e_p2c001_transform_order_enforcement() {
    run_e2e_scenario(
        "e2e_p2c001_transform_order_enforcement",
        "P2C-001",
        CompatibilityMode::Strict,
        json!({
            "good_stack": ["vmap", "grad"],
            "bad_stack": ["grad", "vmap"],
            "input_vector": [1.0, 2.0, 3.0],
        }),
        |states| {
            let args = vec![Value::vector_f64(&[1.0, 2.0, 3.0]).map_err(|err| err.to_string())?];
            let good = dispatch(DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger: ledger(ProgramSpec::Square, &[Transform::Vmap, Transform::Grad]),
                args: args.clone(),
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: Vec::new(),
            })
            .map_err(|err| format!("good transform order failed: {err}"))?;

            let good_values = good.outputs[0]
                .as_tensor()
                .and_then(TensorValue::to_f64_vec)
                .ok_or_else(|| "good path did not produce numeric tensor".to_owned())?;
            states.push(json!({
                "step": "good_stack_executed",
                "outputs": good_values,
            }));

            let bad = dispatch(DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger: ledger(ProgramSpec::Square, &[Transform::Grad, Transform::Vmap]),
                args,
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: Vec::new(),
            });

            match bad {
                Err(DispatchError::TransformExecution(
                    TransformExecutionError::NonScalarGradientInput,
                )) => {
                    states.push(json!({
                        "step": "bad_stack_rejected",
                        "error": "NonScalarGradientInput",
                    }));
                }
                Err(other) => {
                    return Err(format!("unexpected bad-stack rejection class: {other}"));
                }
                Ok(response) => {
                    return Err(format!(
                        "bad transform order unexpectedly succeeded with outputs={:?}",
                        response.outputs
                    ));
                }
            }

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "good_outputs": good_values,
                    "bad_stack_rejection": "NonScalarGradientInput",
                }),
                artifact_refs: vec![],
            })
        },
    );
}

#[test]
fn e2e_p2c001_vmap_rank2_tensor_support() {
    run_e2e_scenario(
        "e2e_p2c001_vmap_rank2_tensor_support",
        "P2C-001",
        CompatibilityMode::Strict,
        json!({
            "input_shape": [2, 2],
            "input_values": [1, 2, 3, 4],
            "transform_stack": ["vmap"],
        }),
        |states| {
            let matrix = Value::Tensor(
                TensorValue::new(
                    DType::I64,
                    Shape { dims: vec![2, 2] },
                    vec![
                        fj_core::Literal::I64(1),
                        fj_core::Literal::I64(2),
                        fj_core::Literal::I64(3),
                        fj_core::Literal::I64(4),
                    ],
                )
                .map_err(|err| err.to_string())?,
            );

            let response = dispatch(DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
                args: vec![matrix],
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: Vec::new(),
            })
            .map_err(|err| format!("vmap rank2 dispatch failed: {err}"))?;

            let out_tensor = response.outputs[0]
                .as_tensor()
                .ok_or_else(|| "expected tensor output".to_owned())?;
            let out_values = out_tensor
                .elements
                .iter()
                .map(|lit| lit.as_i64().ok_or_else(|| "non-i64 element".to_owned()))
                .collect::<Result<Vec<_>, _>>()?;

            states.push(json!({
                "step": "rank2_vmap_output",
                "shape": out_tensor.shape.dims,
                "values": out_values,
            }));

            if out_tensor.shape.dims != vec![2, 2] {
                return Err(format!(
                    "unexpected output shape {:?}",
                    out_tensor.shape.dims
                ));
            }
            if out_values != vec![2, 3, 4, 5] {
                return Err(format!("unexpected output values {:?}", out_values));
            }

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "shape": out_tensor.shape.dims,
                    "values": out_values,
                }),
                artifact_refs: vec![],
            })
        },
    );
}

// ===========================================================================
// bd-3dl.12.7: Required E2E Scenarios for IR Core
// ===========================================================================

// ---------------------------------------------------------------------------
// Scenario 1: trace_to_ir_roundtrip
// ---------------------------------------------------------------------------

#[test]
fn e2e_p2c001_trace_to_ir_roundtrip() {
    run_e2e_scenario(
        "e2e_p2c001_trace_to_ir_roundtrip",
        "P2C-001",
        CompatibilityMode::Strict,
        json!({
            "program": "SquarePlusLinear",
            "description": "f(x) = x^2 + x, verify Jaxpr structure + fingerprint stability",
        }),
        |states| {
            // Step 1: Build the IR for f(x) = x^2 + x
            let jaxpr = build_program(ProgramSpec::SquarePlusLinear);
            states.push(json!({
                "step": "jaxpr_built",
                "invars": jaxpr.invars.len(),
                "outvars": jaxpr.outvars.len(),
                "equations": jaxpr.equations.len(),
            }));

            // Step 2: Verify structure matches expectations
            if jaxpr.invars.len() != 1 {
                return Err(format!(
                    "expected 1 input variable, got {}",
                    jaxpr.invars.len()
                ));
            }
            if jaxpr.outvars.len() != 1 {
                return Err(format!(
                    "expected 1 output variable, got {}",
                    jaxpr.outvars.len()
                ));
            }

            // Step 3: Verify well-formedness
            jaxpr
                .validate_well_formed()
                .map_err(|e| format!("jaxpr validation failed: {e}"))?;
            states.push(json!({"step": "well_formed_check_passed"}));

            // Step 4: Verify fingerprint stability
            let fp1 = jaxpr.canonical_fingerprint();
            let fp2 = jaxpr.canonical_fingerprint();
            let fp3 = jaxpr.canonical_fingerprint();
            if fp1 != fp2 || fp2 != fp3 {
                return Err(format!("fingerprint unstable: {fp1} vs {fp2} vs {fp3}"));
            }
            states.push(json!({
                "step": "fingerprint_stable",
                "fingerprint": fp1,
                "length": fp1.len(),
            }));

            // Step 5: Rebuild from same spec, verify fingerprint identical
            let jaxpr2 = build_program(ProgramSpec::SquarePlusLinear);
            let fp_rebuilt = jaxpr2.canonical_fingerprint();
            if fp1 != fp_rebuilt {
                return Err(format!(
                    "fingerprint diverged after rebuild: {fp1} vs {fp_rebuilt}"
                ));
            }
            states.push(json!({"step": "rebuild_fingerprint_matches"}));

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "invars": jaxpr.invars.len(),
                    "outvars": jaxpr.outvars.len(),
                    "equations": jaxpr.equations.len(),
                    "fingerprint": fp1,
                    "fingerprint_stable": true,
                }),
                artifact_refs: vec![],
            })
        },
    );
}

// ---------------------------------------------------------------------------
// Scenario 2: transform_stack_composition
// ---------------------------------------------------------------------------

#[test]
fn e2e_p2c001_transform_stack_composition() {
    run_e2e_scenario(
        "e2e_p2c001_transform_stack_composition",
        "P2C-001",
        CompatibilityMode::Strict,
        json!({
            "program": "Square",
            "transforms": ["Jit", "Grad"],
            "input": 3.0,
            "expected_output": 6.0,
            "description": "jit(grad(x^2))(3.0) = 6.0",
        }),
        |states| {
            // Step 1: Build TTL with jit(grad(f))
            let ttl = ledger(ProgramSpec::Square, &[Transform::Jit, Transform::Grad]);
            states.push(json!({
                "step": "ttl_built",
                "transforms": ["Jit", "Grad"],
                "root_jaxpr_equations": ttl.root_jaxpr.equations.len(),
            }));

            // Step 2: Verify composition proof passes
            let proof = verify_transform_composition(&ttl)
                .map_err(|e| format!("composition proof failed: {e}"))?;
            states.push(json!({
                "step": "composition_proof_passed",
                "proof_hash": format!("{:?}", proof),
            }));

            // Step 3: Execute through full dispatch pipeline
            let resp = dispatch(DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger: ttl,
                args: vec![Value::scalar_f64(3.0)],
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: Vec::new(),
            })
            .map_err(|e| format!("dispatch failed: {e}"))?;

            let output = resp.outputs[0]
                .as_f64_scalar()
                .ok_or("output is not f64 scalar")?;
            states.push(json!({
                "step": "dispatch_complete",
                "output": output,
                "cache_key": resp.cache_key,
                "ledger_entries": resp.evidence_ledger.len(),
            }));

            // Step 4: Compare against analytical expected value
            let expected = 6.0; // d/dx(x^2) = 2x, at x=3 => 6
            if (output - expected).abs() > 1e-3 {
                return Err(format!(
                    "analytical mismatch: expected {expected}, got {output}"
                ));
            }

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "output": output,
                    "expected": expected,
                    "delta": (output - expected).abs(),
                    "cache_key": resp.cache_key,
                }),
                artifact_refs: vec![],
            })
        },
    );
}

// ---------------------------------------------------------------------------
// Scenario 3: strict_vs_hardened_mode_split
// ---------------------------------------------------------------------------

#[test]
fn e2e_p2c001_strict_vs_hardened_mode_split() {
    run_e2e_scenario(
        "e2e_p2c001_strict_vs_hardened_mode_split",
        "P2C-001",
        CompatibilityMode::Strict,
        json!({
            "program": "Add2",
            "unknown_features": ["experimental_feature"],
            "description": "strict mode rejects, hardened accepts unknown features",
        }),
        |states| {
            let unknown = vec!["experimental_feature".to_owned()];

            // Strict mode: should reject with CacheKeyError
            let strict_result = dispatch(DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
                args: vec![Value::scalar_i64(1), Value::scalar_i64(2)],
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: unknown.clone(),
            });

            match &strict_result {
                Err(e) => {
                    let msg = format!("{e}");
                    states.push(json!({
                        "step": "strict_mode_rejected",
                        "error": msg,
                        "mentions_feature": msg.contains("experimental_feature"),
                    }));
                    if !msg.contains("experimental_feature") {
                        return Err(format!(
                            "strict mode error should mention feature name: {msg}"
                        ));
                    }
                }
                Ok(_) => {
                    return Err(
                        "strict mode should reject unknown features but accepted".to_owned()
                    );
                }
            }

            // Hardened mode: should accept with feature included in key
            let hardened_result = dispatch(DispatchRequest {
                mode: CompatibilityMode::Hardened,
                ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
                args: vec![Value::scalar_i64(1), Value::scalar_i64(2)],
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: unknown.clone(),
            })
            .map_err(|e| format!("hardened mode rejected unexpectedly: {e}"))?;

            states.push(json!({
                "step": "hardened_mode_accepted",
                "cache_key": hardened_result.cache_key,
                "output": hardened_result.outputs[0].as_scalar_literal().and_then(|l| l.as_i64()),
                "ledger_entries": hardened_result.evidence_ledger.len(),
            }));

            // Also dispatch without unknown features for key comparison
            let clean_result = dispatch(DispatchRequest {
                mode: CompatibilityMode::Hardened,
                ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
                args: vec![Value::scalar_i64(1), Value::scalar_i64(2)],
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: Vec::new(),
            })
            .map_err(|e| format!("clean dispatch failed: {e}"))?;

            // Cache keys should differ (unknown feature changes hash)
            let keys_differ = hardened_result.cache_key != clean_result.cache_key;
            states.push(json!({
                "step": "cache_key_comparison",
                "hardened_key": hardened_result.cache_key,
                "clean_key": clean_result.cache_key,
                "keys_differ": keys_differ,
            }));

            if !keys_differ {
                return Err("unknown features should change cache key".to_owned());
            }

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "strict_rejected": true,
                    "hardened_accepted": true,
                    "cache_keys_differ_with_unknown_features": keys_differ,
                    "hardened_cache_key": hardened_result.cache_key,
                }),
                artifact_refs: vec![],
            })
        },
    );
}

// ---------------------------------------------------------------------------
// Scenario 4: ir_determinism_under_replay
// ---------------------------------------------------------------------------

#[test]
fn e2e_p2c001_ir_determinism_under_replay() {
    run_e2e_scenario(
        "e2e_p2c001_ir_determinism_under_replay",
        "P2C-001",
        CompatibilityMode::Strict,
        json!({
            "program": "SquarePlusLinear",
            "transforms": ["Jit", "Grad"],
            "replay_count": 100,
            "description": "100x replay, verify fingerprints + composition proofs identical",
        }),
        |states| {
            let replay_count = 100_usize;
            let mut fingerprints: Vec<String> = Vec::with_capacity(replay_count);
            let mut proofs = Vec::with_capacity(replay_count);

            for i in 0..replay_count {
                let jaxpr = build_program(ProgramSpec::SquarePlusLinear);
                let fp = jaxpr.canonical_fingerprint().to_owned();
                fingerprints.push(fp);

                let ttl = ledger(
                    ProgramSpec::SquarePlusLinear,
                    &[Transform::Jit, Transform::Grad],
                );
                let proof = verify_transform_composition(&ttl)
                    .map_err(|e| format!("composition proof failed at iteration {i}: {e}"))?;
                proofs.push(format!("{proof:?}"));
            }

            // Check all fingerprints identical
            let first_fp = &fingerprints[0];
            let fp_drift: Vec<usize> = fingerprints
                .iter()
                .enumerate()
                .filter(|(_, fp)| *fp != first_fp)
                .map(|(i, _)| i)
                .collect();

            states.push(json!({
                "step": "fingerprint_check",
                "total_runs": replay_count,
                "drift_count": fp_drift.len(),
                "drift_indices": &fp_drift[..fp_drift.len().min(5)],
                "fingerprint_sample": first_fp,
            }));

            if !fp_drift.is_empty() {
                return Err(format!(
                    "fingerprint drift detected in {} of {} runs: indices {:?}",
                    fp_drift.len(),
                    replay_count,
                    fp_drift
                ));
            }

            // Check all composition proofs identical
            let first_proof = &proofs[0];
            let proof_drift: Vec<usize> = proofs
                .iter()
                .enumerate()
                .filter(|(_, p)| *p != first_proof)
                .map(|(i, _)| i)
                .collect();

            states.push(json!({
                "step": "proof_check",
                "drift_count": proof_drift.len(),
                "proof_sample": first_proof,
            }));

            if !proof_drift.is_empty() {
                return Err(format!(
                    "composition proof drift in {} of {} runs",
                    proof_drift.len(),
                    replay_count
                ));
            }

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "replay_count": replay_count,
                    "fingerprint_deterministic": true,
                    "proof_deterministic": true,
                    "fingerprint": first_fp,
                }),
                artifact_refs: vec![],
            })
        },
    );
}

// ---------------------------------------------------------------------------
// Scenario 5: large_jaxpr_stress
// ---------------------------------------------------------------------------

fn build_chain_jaxpr(n: usize) -> Jaxpr {
    let mut equations = Vec::with_capacity(n);
    for i in 0..n {
        let input_var = VarId((i + 1) as u32);
        let output_var = VarId((i + 2) as u32);
        equations.push(Equation {
            primitive: Primitive::Add,
            inputs: smallvec::smallvec![Atom::Var(input_var), Atom::Lit(Literal::I64(1))],
            outputs: smallvec::smallvec![output_var],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        });
    }
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId((n + 1) as u32)],
        equations,
    )
}

#[test]
fn e2e_p2c001_large_jaxpr_stress() {
    run_e2e_scenario(
        "e2e_p2c001_large_jaxpr_stress",
        "P2C-001",
        CompatibilityMode::Strict,
        json!({
            "equation_count": 1000,
            "fingerprint_budget_ms": 100,
            "description": "1000-equation chain, verify fingerprint timing and composition",
        }),
        |states| {
            let n = 1000_usize;
            let jaxpr = build_chain_jaxpr(n);

            // Verify well-formedness
            jaxpr
                .validate_well_formed()
                .map_err(|e| format!("large jaxpr validation failed: {e}"))?;
            states.push(json!({
                "step": "well_formed",
                "equation_count": jaxpr.equations.len(),
            }));

            // Fingerprint timing
            let fp_start = Instant::now();
            let fp = jaxpr.canonical_fingerprint();
            let fp_ms = fp_start.elapsed().as_millis();

            states.push(json!({
                "step": "fingerprint_computed",
                "fingerprint_ms": fp_ms,
                "fingerprint_len": fp.len(),
            }));

            if fp_ms > 100 {
                return Err(format!("fingerprint took {fp_ms}ms, budget is 100ms"));
            }

            // Composition proof still valid with transforms
            let ttl = {
                let mut t = TraceTransformLedger::new(jaxpr.clone());
                t.push_transform(Transform::Jit, "stress-jit".to_owned());
                t
            };
            let proof = verify_transform_composition(&ttl)
                .map_err(|e| format!("composition proof failed: {e}"))?;
            states.push(json!({
                "step": "composition_proof_passed",
                "proof": format!("{proof:?}"),
            }));

            // Evaluate to verify correctness
            let eval_start = Instant::now();
            let result = fj_interpreters::eval_jaxpr(&jaxpr, &[Value::scalar_i64(0)])
                .map_err(|e| format!("eval failed: {e}"))?;
            let eval_ms = eval_start.elapsed().as_millis();

            let output = result[0]
                .as_scalar_literal()
                .and_then(|l| l.as_i64())
                .ok_or("expected i64 scalar output")?;

            states.push(json!({
                "step": "evaluation_complete",
                "output": output,
                "expected": n,
                "eval_ms": eval_ms,
            }));

            if output != n as i64 {
                return Err(format!("expected output {n}, got {output}"));
            }

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "equation_count": n,
                    "fingerprint_ms": fp_ms,
                    "eval_ms": eval_ms,
                    "output_correct": true,
                    "composition_valid": true,
                }),
                artifact_refs: vec![],
            })
        },
    );
}
