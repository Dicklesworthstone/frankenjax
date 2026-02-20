#![forbid(unsafe_code)]

//! FJ-P2C-004-G: E2E Scenario Scripts + Replay/Forensics Logging
//! for dispatch/AD/effects runtime.

use fj_core::{
    CompatibilityMode, ProgramSpec, TraceTransformLedger, Transform, Value, build_program,
};
use fj_dispatch::{DispatchRequest, dispatch};
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

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn artifact_dir() -> PathBuf {
    let dir = repo_root().join("artifacts/e2e");
    fs::create_dir_all(&dir).ok();
    dir
}

fn ledger(program: ProgramSpec, transforms: &[Transform]) -> TraceTransformLedger {
    let mut l = TraceTransformLedger::new(build_program(program));
    for (i, t) in transforms.iter().enumerate() {
        l.push_transform(*t, format!("evidence-{}-{}", t.as_str(), i));
    }
    l
}

fn run_e2e_scenario(
    scenario_id: &str,
    mode: CompatibilityMode,
    body: impl FnOnce(&mut Vec<JsonValue>) -> (JsonValue, Option<JsonValue>, String),
) {
    let start = Instant::now();
    let mut intermediate_states = Vec::new();
    let (input_capture, output_capture, result) = body(&mut intermediate_states);
    let duration = start.elapsed();

    let log = E2EForensicLogV1 {
        schema_version: "frankenjax.e2e-forensic-log.v1",
        scenario_id: scenario_id.to_owned(),
        test_id: format!("e2e_p2c004_{scenario_id}"),
        packet_id: "FJ-P2C-004".to_owned(),
        fixture_id: format!("p2c004-e2e-{scenario_id}"),
        seed: None,
        mode: format!("{mode:?}"),
        ts_utc_unix_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis(),
        env: json!({
            "rust_version": env!("CARGO_PKG_VERSION"),
            "packet": "FJ-P2C-004",
        }),
        artifact_refs: vec![
            "crates/fj-dispatch/src/lib.rs".to_owned(),
            "crates/fj-ad/src/lib.rs".to_owned(),
        ],
        replay_command: format!(
            "cargo test -p fj-conformance --test e2e_p2c004 -- e2e_p2c004_{scenario_id} --nocapture"
        ),
        input_capture,
        intermediate_states,
        output_capture,
        result: result.clone(),
        duration_ms: duration.as_millis(),
        details: None,
    };

    let path = artifact_dir().join(format!("e2e_p2c004_{scenario_id}.e2e.json"));
    let json = serde_json::to_string_pretty(&log).expect("serialize");
    fs::write(&path, &json).expect("write forensic log");

    assert_eq!(result, "PASS", "scenario {scenario_id} failed");
}

// ── Scenario 1: full_dispatch_pipeline ─────────────────────────────

#[test]
fn e2e_p2c004_full_dispatch_pipeline() {
    run_e2e_scenario(
        "full_dispatch_pipeline",
        CompatibilityMode::Strict,
        |states| {
            // jit(grad(f)) for f(x) = x²
            let request = DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger: ledger(ProgramSpec::Square, &[Transform::Jit, Transform::Grad]),
                args: vec![Value::scalar_f64(3.0)],
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: vec![],
            };

            states.push(json!({"phase": "request_built", "transforms": ["jit", "grad"]}));

            let response = dispatch(request).expect("dispatch should succeed");

            states.push(json!({
                "phase": "dispatch_complete",
                "output_count": response.outputs.len(),
                "cache_key": response.cache_key,
                "ledger_entries": response.evidence_ledger.len(),
            }));

            let derivative = response.outputs[0].as_f64_scalar().unwrap();
            assert!((derivative - 6.0).abs() < 1e-3);
            assert!(response.cache_key.starts_with("fjx-"));
            assert_eq!(response.evidence_ledger.len(), 1);

            (
                json!({"program": "Square", "transforms": ["Jit", "Grad"], "x": 3.0}),
                Some(json!({"derivative": derivative, "cache_key": response.cache_key})),
                "PASS".to_owned(),
            )
        },
    );
}

// ── Scenario 2: transform_composition_matrix ───────────────────────

#[test]
fn e2e_p2c004_transform_composition_matrix() {
    run_e2e_scenario(
        "transform_composition_matrix",
        CompatibilityMode::Strict,
        |states| {
            let compositions: Vec<(&str, Vec<Transform>, Vec<Value>)> = vec![
                (
                    "jit_only",
                    vec![Transform::Jit],
                    vec![Value::scalar_i64(5), Value::scalar_i64(3)],
                ),
                (
                    "grad_only",
                    vec![Transform::Grad],
                    vec![Value::scalar_f64(4.0)],
                ),
                (
                    "vmap_only",
                    vec![Transform::Vmap],
                    vec![Value::vector_i64(&[1, 2, 3]).unwrap()],
                ),
                (
                    "jit_grad",
                    vec![Transform::Jit, Transform::Grad],
                    vec![Value::scalar_f64(2.0)],
                ),
                (
                    "jit_vmap",
                    vec![Transform::Jit, Transform::Vmap],
                    vec![Value::vector_i64(&[1, 2]).unwrap()],
                ),
                (
                    "vmap_grad",
                    vec![Transform::Vmap, Transform::Grad],
                    vec![Value::vector_f64(&[1.0, 2.0]).unwrap()],
                ),
                (
                    "jit_vmap_grad",
                    vec![Transform::Jit, Transform::Vmap, Transform::Grad],
                    vec![Value::vector_f64(&[1.0, 2.0]).unwrap()],
                ),
            ];

            let mut cache_keys = Vec::new();

            for (name, transforms, args) in &compositions {
                let program = if transforms.contains(&Transform::Grad)
                    && !transforms.contains(&Transform::Vmap)
                {
                    ProgramSpec::Square
                } else if transforms.contains(&Transform::Vmap) {
                    if transforms.contains(&Transform::Grad) {
                        ProgramSpec::Square
                    } else {
                        ProgramSpec::AddOne
                    }
                } else {
                    ProgramSpec::Add2
                };

                let result = dispatch(DispatchRequest {
                    mode: CompatibilityMode::Strict,
                    ledger: ledger(program, transforms),
                    args: args.clone(),
                    backend: "cpu".to_owned(),
                    compile_options: BTreeMap::new(),
                    custom_hook: None,
                    unknown_incompatible_features: vec![],
                });

                let (status, key) = match result {
                    Ok(r) => {
                        cache_keys.push(r.cache_key.clone());
                        ("OK", r.cache_key)
                    }
                    Err(e) => ("ERR", format!("{e}")),
                };

                states.push(json!({
                    "composition": name,
                    "status": status,
                    "cache_key": key,
                }));
            }

            // Verify all cache keys are distinct
            let unique: std::collections::HashSet<_> = cache_keys.iter().collect();
            assert_eq!(unique.len(), cache_keys.len(), "cache key collision");

            (
                json!({"compositions": compositions.len()}),
                Some(json!({"distinct_keys": cache_keys.len()})),
                "PASS".to_owned(),
            )
        },
    );
}

// ── Scenario 3: strict_hardened_dispatch_divergence ─────────────────

#[test]
fn e2e_p2c004_strict_hardened_divergence() {
    run_e2e_scenario(
        "strict_hardened_divergence",
        CompatibilityMode::Strict,
        |states| {
            let features = vec!["future.sharding.v2".to_owned()];

            // Strict mode: should fail
            let strict_result = dispatch(DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
                args: vec![Value::scalar_i64(10), Value::scalar_i64(20)],
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: features.clone(),
            });

            states.push(json!({
                "mode": "strict",
                "result": if strict_result.is_err() { "REJECTED" } else { "ACCEPTED" },
            }));
            assert!(
                strict_result.is_err(),
                "strict must reject unknown features"
            );

            // Hardened mode: should succeed
            let hardened_result = dispatch(DispatchRequest {
                mode: CompatibilityMode::Hardened,
                ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
                args: vec![Value::scalar_i64(10), Value::scalar_i64(20)],
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: features.clone(),
            })
            .expect("hardened should accept");

            states.push(json!({
                "mode": "hardened",
                "result": "ACCEPTED",
                "output": format!("{:?}", hardened_result.outputs),
                "cache_key": hardened_result.cache_key,
            }));

            assert_eq!(hardened_result.outputs, vec![Value::scalar_i64(30)]);

            (
                json!({"features": features}),
                Some(json!({"strict": "rejected", "hardened": "accepted"})),
                "PASS".to_owned(),
            )
        },
    );
}

// ── Scenario 4: evidence_ledger_audit_trail ─────────────────────────

#[test]
fn e2e_p2c004_evidence_ledger_audit_trail() {
    run_e2e_scenario(
        "evidence_ledger_audit_trail",
        CompatibilityMode::Strict,
        |states| {
            let programs = [
                (
                    ProgramSpec::Add2,
                    &[Transform::Jit][..],
                    vec![Value::scalar_i64(1), Value::scalar_i64(2)],
                ),
                (
                    ProgramSpec::Square,
                    &[Transform::Grad][..],
                    vec![Value::scalar_f64(3.0)],
                ),
                (
                    ProgramSpec::AddOne,
                    &[Transform::Vmap][..],
                    vec![Value::vector_i64(&[1, 2, 3]).unwrap()],
                ),
                (
                    ProgramSpec::Square,
                    &[Transform::Jit, Transform::Grad][..],
                    vec![Value::scalar_f64(5.0)],
                ),
                (
                    ProgramSpec::AddOne,
                    &[Transform::Jit, Transform::Vmap][..],
                    vec![Value::vector_i64(&[10, 20]).unwrap()],
                ),
            ];

            for (i, (program, transforms, args)) in programs.iter().enumerate() {
                let r = dispatch(DispatchRequest {
                    mode: CompatibilityMode::Strict,
                    ledger: ledger(*program, transforms),
                    args: args.clone(),
                    backend: "cpu".to_owned(),
                    compile_options: BTreeMap::new(),
                    custom_hook: None,
                    unknown_incompatible_features: vec![],
                })
                .expect("dispatch should succeed");

                assert_eq!(r.evidence_ledger.len(), 1);
                let entry = &r.evidence_ledger.entries()[0];
                assert_eq!(entry.signals.len(), 4);
                assert_eq!(entry.decision_id, r.cache_key);

                states.push(json!({
                    "dispatch": i,
                    "ledger_entries": r.evidence_ledger.len(),
                    "signal_count": entry.signals.len(),
                    "decision_id": entry.decision_id,
                }));
            }

            (
                json!({"dispatch_count": programs.len()}),
                Some(json!({"all_ledgers_valid": true})),
                "PASS".to_owned(),
            )
        },
    );
}

// ── Scenario 5: dispatch_under_load ─────────────────────────────────

#[test]
fn e2e_p2c004_dispatch_under_load() {
    run_e2e_scenario("dispatch_under_load", CompatibilityMode::Strict, |states| {
        let n = 1000;
        let mut latencies_ns = Vec::with_capacity(n);
        let mut last_key = String::new();
        let mut last_output = vec![];
        let mut drift_count = 0;

        for i in 0..n {
            let start = Instant::now();
            let r = dispatch(DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger: ledger(ProgramSpec::Add2, &[Transform::Jit]),
                args: vec![Value::scalar_i64(10), Value::scalar_i64(20)],
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: vec![],
            })
            .expect("dispatch should succeed");
            latencies_ns.push(start.elapsed().as_nanos());

            if i == 0 {
                last_key = r.cache_key.clone();
                last_output = r.outputs.clone();
            } else {
                assert_eq!(r.cache_key, last_key, "cache key drift at i={i}");
                if r.outputs != last_output {
                    drift_count += 1;
                }
            }
        }

        latencies_ns.sort();
        let p50 = latencies_ns[n / 2];
        let p95 = latencies_ns[n * 95 / 100];
        let p99 = latencies_ns[n * 99 / 100];

        states.push(json!({
            "iterations": n,
            "p50_ns": p50,
            "p95_ns": p95,
            "p99_ns": p99,
            "drift_count": drift_count,
        }));

        assert_eq!(drift_count, 0, "output drift detected");

        (
            json!({"iterations": n}),
            Some(json!({"p50_ns": p50, "p95_ns": p95, "p99_ns": p99, "drift_count": drift_count})),
            "PASS".to_owned(),
        )
    });
}

// ── Scenario 6: adversarial_dispatch_inputs ─────────────────────────

#[test]
fn e2e_p2c004_adversarial_dispatch_inputs() {
    run_e2e_scenario(
        "adversarial_dispatch_inputs",
        CompatibilityMode::Strict,
        |states| {
            let mut pass_count = 0;

            // 1. Empty transform stack — should succeed (direct eval)
            let r = dispatch(DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger: { TraceTransformLedger::new(build_program(ProgramSpec::Add2)) },
                args: vec![Value::scalar_i64(1), Value::scalar_i64(2)],
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: vec![],
            });
            states.push(json!({"case": "empty_transform_stack", "result": if r.is_ok() { "ok" } else { "err" }}));
            assert!(r.is_ok());
            pass_count += 1;

            // 2. Transform with empty evidence — should fail composition proof
            let mut ttl = TraceTransformLedger::new(build_program(ProgramSpec::Add2));
            ttl.push_transform(Transform::Jit, "".to_owned()); // empty evidence
            let r = dispatch(DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger: ttl,
                args: vec![Value::scalar_i64(1), Value::scalar_i64(2)],
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: vec![],
            });
            states.push(json!({"case": "empty_evidence", "result": if r.is_err() { "rejected" } else { "accepted" }}));
            assert!(r.is_err(), "empty evidence should be rejected");
            pass_count += 1;

            // 3. Grad with vector input — should fail (grad requires scalar)
            let r = dispatch(DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger: ledger(ProgramSpec::Square, &[Transform::Grad]),
                args: vec![Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap()],
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: vec![],
            });
            states.push(json!({"case": "grad_vector_input", "result": if r.is_err() { "rejected" } else { "accepted" }}));
            assert!(r.is_err(), "vector grad input should be rejected");
            pass_count += 1;

            // 4. Vmap with empty batch — should fail
            let r = dispatch(DispatchRequest {
                mode: CompatibilityMode::Strict,
                ledger: ledger(ProgramSpec::AddOne, &[Transform::Vmap]),
                args: vec![],
                backend: "cpu".to_owned(),
                compile_options: BTreeMap::new(),
                custom_hook: None,
                unknown_incompatible_features: vec![],
            });
            states.push(json!({"case": "empty_vmap_args", "result": if r.is_err() { "rejected" } else { "accepted" }}));
            assert!(r.is_err());
            pass_count += 1;

            (
                json!({"adversarial_cases": 4}),
                Some(json!({"passed": pass_count, "total": 4})),
                "PASS".to_owned(),
            )
        },
    );
}
