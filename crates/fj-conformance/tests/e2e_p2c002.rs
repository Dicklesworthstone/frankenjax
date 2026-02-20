#![forbid(unsafe_code)]

//! FJ-P2C-002-G: E2E Scenario Scripts + Replay/Forensics Logging
//! for the API transform front-door (jit/grad/vmap/value_and_grad).

use fj_api::{ApiError, compose, grad, jit, value_and_grad, vmap};
use fj_core::{CompatibilityMode, ProgramSpec, Transform, Value, build_program};
use serde::Serialize;
use serde_json::{Value as JsonValue, json};
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ============================================================================
// Forensic logging infrastructure (shared pattern with e2e.rs)
// ============================================================================

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

fn artifact_dir() -> PathBuf {
    if let Ok(path) = std::env::var("FJ_E2E_ARTIFACT_DIR") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("artifacts/e2e")
}

fn replay_command(scenario_id: &str) -> String {
    format!(
        "cargo test -p fj-conformance --test e2e_p2c002 -- {} --exact --nocapture",
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
    fs::write(&path, raw).expect("e2e forensic log write should succeed");
}

fn run_e2e_scenario(
    scenario_id: &str,
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
        packet_id: "P2C-002".to_owned(),
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

// ============================================================================
// Scenario 1: user_api_jit_basic
// ============================================================================

#[test]
fn e2e_p2c002_user_api_jit_basic() {
    run_e2e_scenario(
        "e2e_p2c002_user_api_jit_basic",
        CompatibilityMode::Strict,
        json!({
            "program": "Add2",
            "inputs": [7, 13],
            "description": "jit(f)(7, 13) == eval_jaxpr(f, 7, 13)",
        }),
        |states| {
            let jaxpr = build_program(ProgramSpec::Add2);
            let args = vec![Value::scalar_i64(7), Value::scalar_i64(13)];

            // Step 1: Call through public API
            let api_result = jit(jaxpr.clone())
                .call(args.clone())
                .map_err(|e| format!("jit API call failed: {e}"))?;
            states.push(json!({
                "step": "api_call_complete",
                "output_count": api_result.len(),
            }));

            // Step 2: Call through direct interpreter
            let direct_result = fj_interpreters::eval_jaxpr(&jaxpr, &args)
                .map_err(|e| format!("direct eval failed: {e}"))?;
            states.push(json!({
                "step": "direct_eval_complete",
                "output_count": direct_result.len(),
            }));

            // Step 3: Compare
            if api_result != direct_result {
                return Err(format!(
                    "API result {:?} != direct result {:?}",
                    api_result, direct_result
                ));
            }

            let output_val = api_result[0]
                .as_scalar_literal()
                .and_then(|l| l.as_i64())
                .ok_or("expected i64 scalar")?;
            states.push(json!({
                "step": "comparison_passed",
                "output": output_val,
                "expected": 20,
            }));

            if output_val != 20 {
                return Err(format!("expected 20, got {output_val}"));
            }

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "api_output": output_val,
                    "direct_output": output_val,
                    "match": true,
                }),
                artifact_refs: vec![],
            })
        },
    );
}

// ============================================================================
// Scenario 2: user_api_grad_polynomial
// ============================================================================

#[test]
fn e2e_p2c002_user_api_grad_polynomial() {
    run_e2e_scenario(
        "e2e_p2c002_user_api_grad_polynomial",
        CompatibilityMode::Strict,
        json!({
            "program": "Square",
            "input": 2.0,
            "expected_gradient": 4.0,
            "description": "grad(x^2)(2.0) = 4.0, verified through public API",
        }),
        |states| {
            let jaxpr = build_program(ProgramSpec::Square);
            let x = 2.0_f64;

            // Step 1: Call grad through public API
            let grad_result = grad(jaxpr.clone())
                .call(vec![Value::scalar_f64(x)])
                .map_err(|e| format!("grad API call failed: {e}"))?;
            let gradient = grad_result[0]
                .as_f64_scalar()
                .ok_or("expected f64 scalar gradient")?;
            states.push(json!({
                "step": "grad_api_call_complete",
                "gradient": gradient,
            }));

            // Step 2: Verify against analytical value (d/dx x^2 = 2x)
            let expected = 2.0 * x;
            let delta = (gradient - expected).abs();
            states.push(json!({
                "step": "analytical_comparison",
                "expected": expected,
                "actual": gradient,
                "delta": delta,
                "tolerance": 1e-3,
            }));

            if delta > 1e-3 {
                return Err(format!(
                    "grad(x^2)({x}) = {gradient}, expected {expected} (delta={delta})"
                ));
            }

            // Step 3: Also verify value_and_grad consistency
            let (val, grd) = value_and_grad(jaxpr)
                .call(vec![Value::scalar_f64(x)])
                .map_err(|e| format!("value_and_grad failed: {e}"))?;
            let value = val[0].as_f64_scalar().ok_or("expected f64 value")?;
            let vag_gradient = grd[0].as_f64_scalar().ok_or("expected f64 gradient")?;
            states.push(json!({
                "step": "value_and_grad_cross_check",
                "value": value,
                "gradient": vag_gradient,
                "expected_value": x * x,
            }));

            if (value - x * x).abs() > 1e-6 {
                return Err(format!(
                    "value_and_grad value mismatch: {value} vs {}",
                    x * x
                ));
            }
            if (vag_gradient - gradient).abs() > 1e-6 {
                return Err(format!(
                    "value_and_grad gradient mismatch: {vag_gradient} vs {gradient}"
                ));
            }

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "gradient": gradient,
                    "expected": expected,
                    "delta": delta,
                    "value_and_grad_consistent": true,
                }),
                artifact_refs: vec![],
            })
        },
    );
}

// ============================================================================
// Scenario 3: user_api_vmap_batch
// ============================================================================

#[test]
fn e2e_p2c002_user_api_vmap_batch() {
    run_e2e_scenario(
        "e2e_p2c002_user_api_vmap_batch",
        CompatibilityMode::Strict,
        json!({
            "program": "AddOne",
            "batch": [10, 20, 30, 40, 50],
            "expected": [11, 21, 31, 41, 51],
            "description": "vmap(add_one)([10,20,30,40,50]) = [11,21,31,41,51]",
        }),
        |states| {
            let jaxpr = build_program(ProgramSpec::AddOne);
            let batch = vec![10, 20, 30, 40, 50];
            let input = Value::vector_i64(&batch).map_err(|e| e.to_string())?;

            // Step 1: Call vmap through public API
            let vmap_result = vmap(jaxpr.clone())
                .call(vec![input])
                .map_err(|e| format!("vmap API call failed: {e}"))?;
            let tensor = vmap_result[0].as_tensor().ok_or("expected tensor output")?;
            let vals: Vec<i64> = tensor
                .elements
                .iter()
                .map(|l| l.as_i64().ok_or("expected i64 element"))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?;
            states.push(json!({
                "step": "vmap_api_call_complete",
                "output_len": vals.len(),
                "output": vals,
            }));

            // Step 2: Verify each element against direct eval
            let expected: Vec<i64> = batch.iter().map(|x| x + 1).collect();
            for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
                let direct = fj_interpreters::eval_jaxpr(&jaxpr, &[Value::scalar_i64(batch[i])])
                    .map_err(|e| format!("direct eval [{i}] failed: {e}"))?;
                let direct_val = direct[0]
                    .as_scalar_literal()
                    .and_then(|l| l.as_i64())
                    .ok_or(format!("expected i64 scalar at index {i}"))?;

                if got != exp || got != direct_val {
                    return Err(format!(
                        "mismatch at [{i}]: vmap={got}, expected={exp}, direct={direct_val}"
                    ));
                }
            }
            states.push(json!({
                "step": "elementwise_comparison_passed",
                "all_match": true,
            }));

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "batch_size": batch.len(),
                    "output": vals,
                    "expected": expected,
                    "all_match": true,
                }),
                artifact_refs: vec![],
            })
        },
    );
}

// ============================================================================
// Scenario 4: user_api_stacking
// ============================================================================

#[test]
fn e2e_p2c002_user_api_stacking() {
    run_e2e_scenario(
        "e2e_p2c002_user_api_stacking",
        CompatibilityMode::Strict,
        json!({
            "program": "Square",
            "transforms": ["Jit", "Grad"],
            "input": 5.0,
            "expected": 10.0,
            "description": "jit(grad(x^2))(5.0) = 10.0 via compose_grad builder",
        }),
        |states| {
            let jaxpr = build_program(ProgramSpec::Square);
            let x = 5.0_f64;

            // Step 1: Call via builder pattern
            let builder_result = jit(jaxpr.clone())
                .compose_grad()
                .call(vec![Value::scalar_f64(x)])
                .map_err(|e| format!("jit.compose_grad API call failed: {e}"))?;
            let builder_val = builder_result[0]
                .as_f64_scalar()
                .ok_or("expected f64 scalar from builder")?;
            states.push(json!({
                "step": "builder_pattern_call",
                "output": builder_val,
            }));

            // Step 2: Call via compose() helper
            let compose_result = compose(jaxpr.clone(), vec![Transform::Jit, Transform::Grad])
                .call(vec![Value::scalar_f64(x)])
                .map_err(|e| format!("compose API call failed: {e}"))?;
            let compose_val = compose_result[0]
                .as_f64_scalar()
                .ok_or("expected f64 scalar from compose")?;
            states.push(json!({
                "step": "compose_helper_call",
                "output": compose_val,
            }));

            // Step 3: Call via standalone grad
            let grad_result = grad(jaxpr)
                .call(vec![Value::scalar_f64(x)])
                .map_err(|e| format!("standalone grad failed: {e}"))?;
            let grad_val = grad_result[0]
                .as_f64_scalar()
                .ok_or("expected f64 scalar from grad")?;
            states.push(json!({
                "step": "standalone_grad_call",
                "output": grad_val,
            }));

            // Step 4: Cross-validate all three paths
            let expected = 2.0 * x;
            let paths = [
                ("builder", builder_val),
                ("compose", compose_val),
                ("standalone_grad", grad_val),
            ];
            for (name, val) in &paths {
                if (val - expected).abs() > 1e-3 {
                    return Err(format!("{name} path: got {val}, expected {expected}"));
                }
            }

            // Also check builder == compose exactly
            if (builder_val - compose_val).abs() > 1e-10 {
                return Err(format!(
                    "builder ({builder_val}) != compose ({compose_val})"
                ));
            }

            states.push(json!({
                "step": "cross_validation_passed",
                "builder": builder_val,
                "compose": compose_val,
                "grad": grad_val,
                "expected": expected,
            }));

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "builder_output": builder_val,
                    "compose_output": compose_val,
                    "grad_output": grad_val,
                    "expected": expected,
                    "all_match": true,
                }),
                artifact_refs: vec![],
            })
        },
    );
}

// ============================================================================
// Scenario 5: user_api_error_messages
// ============================================================================

#[test]
fn e2e_p2c002_user_api_error_messages() {
    run_e2e_scenario(
        "e2e_p2c002_user_api_error_messages",
        CompatibilityMode::Strict,
        json!({
            "description": "Invalid compositions produce user-friendly errors, not panics",
            "test_cases": [
                "grad with vector input",
                "grad with empty args",
                "vmap with scalar input",
                "grad(vmap(f)) invalid composition",
            ],
        }),
        |states| {
            // Case 1: grad with vector input → GradRequiresScalar
            let jaxpr = build_program(ProgramSpec::Square);
            let err1 = grad(jaxpr)
                .call(vec![
                    Value::vector_f64(&[1.0, 2.0]).map_err(|e| e.to_string())?,
                ])
                .expect_err("grad with vector should fail");
            let msg1 = format!("{err1}");
            let is_grad_scalar = matches!(err1, ApiError::GradRequiresScalar { .. });
            states.push(json!({
                "step": "grad_vector_error",
                "error_message": msg1,
                "is_grad_requires_scalar": is_grad_scalar,
                "message_is_user_friendly": !msg1.contains("DispatchError"),
            }));
            if !is_grad_scalar {
                return Err(format!("expected GradRequiresScalar, got: {msg1}"));
            }
            if msg1.contains("DispatchError") {
                return Err(format!("error message leaks internal type: {msg1}"));
            }

            // Case 2: grad with empty args → EvalError
            let jaxpr = build_program(ProgramSpec::Square);
            let err2 = grad(jaxpr)
                .call(vec![])
                .expect_err("grad with no args should fail");
            let msg2 = format!("{err2}");
            states.push(json!({
                "step": "grad_empty_args_error",
                "error_message": msg2,
                "is_eval_error": matches!(err2, ApiError::EvalError { .. }),
            }));

            // Case 3: vmap with scalar → EvalError
            let jaxpr = build_program(ProgramSpec::AddOne);
            let err3 = vmap(jaxpr)
                .call(vec![Value::scalar_i64(42)])
                .expect_err("vmap with scalar should fail");
            let msg3 = format!("{err3}");
            states.push(json!({
                "step": "vmap_scalar_error",
                "error_message": msg3,
                "is_eval_error": matches!(err3, ApiError::EvalError { .. }),
            }));

            // Case 4: grad(vmap(f)) with vector → GradRequiresScalar
            let jaxpr = build_program(ProgramSpec::Square);
            let err4 = compose(jaxpr, vec![Transform::Grad, Transform::Vmap])
                .call(vec![
                    Value::vector_f64(&[1.0, 2.0]).map_err(|e| e.to_string())?,
                ])
                .expect_err("grad(vmap(f)) should fail");
            let msg4 = format!("{err4}");
            let is_grad_scalar4 = matches!(err4, ApiError::GradRequiresScalar { .. });
            states.push(json!({
                "step": "grad_vmap_error",
                "error_message": msg4,
                "is_grad_requires_scalar": is_grad_scalar4,
            }));
            if !is_grad_scalar4 {
                return Err(format!(
                    "expected GradRequiresScalar for grad(vmap), got: {msg4}"
                ));
            }

            // Verify none of the error messages contain panic-like internal details
            for (name, msg) in [
                ("grad_vector", &msg1),
                ("grad_empty", &msg2),
                ("vmap_scalar", &msg3),
                ("grad_vmap", &msg4),
            ] {
                if msg.contains("panicked") || msg.contains("thread '") {
                    return Err(format!("{name} error looks like a panic: {msg}"));
                }
            }

            states.push(json!({
                "step": "all_errors_user_friendly",
                "total_cases": 4,
            }));

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "cases_validated": 4,
                    "grad_vector": { "type": "GradRequiresScalar", "message": msg1 },
                    "grad_empty": { "type": "EvalError", "message": msg2 },
                    "vmap_scalar": { "type": "EvalError", "message": msg3 },
                    "grad_vmap": { "type": "GradRequiresScalar", "message": msg4 },
                    "all_user_friendly": true,
                }),
                artifact_refs: vec![],
            })
        },
    );
}

// ============================================================================
// Scenario 6: user_api_mode_and_backend_configuration
// ============================================================================

#[test]
fn e2e_p2c002_user_api_mode_and_backend_configuration() {
    run_e2e_scenario(
        "e2e_p2c002_user_api_mode_and_backend_config",
        CompatibilityMode::Strict,
        json!({
            "description": "Verify mode/backend configuration flows through API correctly",
            "modes": ["strict", "hardened"],
        }),
        |states| {
            let jaxpr = build_program(ProgramSpec::Add2);
            let args = vec![Value::scalar_i64(3), Value::scalar_i64(4)];

            // Step 1: Default (strict) mode
            let strict_result = jit(jaxpr.clone())
                .call(args.clone())
                .map_err(|e| format!("strict jit failed: {e}"))?;
            let strict_val = strict_result[0]
                .as_scalar_literal()
                .and_then(|l| l.as_i64())
                .ok_or("expected i64 from strict")?;
            states.push(json!({
                "step": "strict_mode_call",
                "output": strict_val,
            }));

            // Step 2: Hardened mode
            let hardened_result = jit(jaxpr.clone())
                .with_mode(CompatibilityMode::Hardened)
                .call(args.clone())
                .map_err(|e| format!("hardened jit failed: {e}"))?;
            let hardened_val = hardened_result[0]
                .as_scalar_literal()
                .and_then(|l| l.as_i64())
                .ok_or("expected i64 from hardened")?;
            states.push(json!({
                "step": "hardened_mode_call",
                "output": hardened_val,
            }));

            // Step 3: Both should produce same output
            if strict_val != hardened_val {
                return Err(format!(
                    "mode split: strict={strict_val}, hardened={hardened_val}"
                ));
            }
            if strict_val != 7 {
                return Err(format!("expected 7, got {strict_val}"));
            }

            // Step 4: Hardened grad
            let jaxpr_sq = build_program(ProgramSpec::Square);
            let hardened_grad = grad(jaxpr_sq)
                .with_mode(CompatibilityMode::Hardened)
                .call(vec![Value::scalar_f64(3.0)])
                .map_err(|e| format!("hardened grad failed: {e}"))?;
            let grad_val = hardened_grad[0]
                .as_f64_scalar()
                .ok_or("expected f64 grad")?;
            states.push(json!({
                "step": "hardened_grad_call",
                "gradient": grad_val,
                "expected": 6.0,
            }));

            if (grad_val - 6.0).abs() > 1e-3 {
                return Err(format!("hardened grad: expected 6.0, got {grad_val}"));
            }

            // Step 5: Composed transform with mode
            let jaxpr_sq2 = build_program(ProgramSpec::Square);
            let composed_result = jit(jaxpr_sq2)
                .compose_grad()
                .with_mode(CompatibilityMode::Hardened)
                .call(vec![Value::scalar_f64(4.0)])
                .map_err(|e| format!("composed hardened failed: {e}"))?;
            let composed_val = composed_result[0]
                .as_f64_scalar()
                .ok_or("expected f64 from composed")?;
            states.push(json!({
                "step": "composed_hardened_call",
                "output": composed_val,
                "expected": 8.0,
            }));

            if (composed_val - 8.0).abs() > 1e-3 {
                return Err(format!(
                    "composed hardened: expected 8.0, got {composed_val}"
                ));
            }

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "strict_output": strict_val,
                    "hardened_output": hardened_val,
                    "hardened_grad": grad_val,
                    "composed_hardened": composed_val,
                    "modes_consistent": true,
                }),
                artifact_refs: vec![],
            })
        },
    );
}
