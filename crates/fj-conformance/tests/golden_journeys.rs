//! bd-3dl.20: User Workflow Scenario Corpus + Golden Journeys
//!
//! 9 golden journey scenarios reflecting real JAX user patterns:
//!   1. Basic Transform — jit a simple function, check output matches eager
//!   2. Gradient Computation — grad of polynomial, verify against analytical derivative
//!   3. Batched Computation — vmap over batch dimension, compare to manual loop
//!   4. Transform Composition — jit(grad(f)), vmap(grad(f)) with order checking
//!   5. Cache Hit/Miss — repeat dispatch, verify cache key determinism
//!   6. Error Recovery — malformed input, shape mismatch, error message quality
//!   7. Large Program — 100+ equation Jaxpr, verify correctness and timing
//!   8. Transcendental Gradients — inverse hyperbolic function derivatives
//!   9. Ledger Inspection — multi-dispatch session, inspect decision ledger

#![forbid(unsafe_code)]

use fj_core::{
    Atom, CompatibilityMode, Equation, Jaxpr, Literal, Primitive, ProgramSpec,
    TraceTransformLedger, Transform, Value, VarId, build_program, verify_transform_composition,
};
use fj_dispatch::{DispatchRequest, dispatch};
use serde::Serialize;
use serde_json::{Value as JsonValue, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Infrastructure (mirrors e2e.rs patterns)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct GoldenJourneyLog {
    schema_version: &'static str,
    scenario_id: String,
    scenario_category: String,
    ts_utc_unix_ms: u128,
    replay_command: String,
    input_capture: JsonValue,
    assertions: Vec<JourneyAssertion>,
    output_capture: JsonValue,
    result: String,
    duration_ms: u128,
}

#[derive(Debug, Clone, Serialize)]
struct JourneyAssertion {
    name: String,
    passed: bool,
    detail: String,
}

fn artifact_dir() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    root.join("artifacts/e2e/golden_journeys")
}

fn write_journey_log(log: &GoldenJourneyLog) {
    let dir = artifact_dir();
    fs::create_dir_all(&dir).expect("golden journey artifact dir should be creatable");
    let path = dir.join(format!("{}.golden.json", log.scenario_id));
    let raw = serde_json::to_string_pretty(log).expect("journey log serialization should succeed");
    fs::write(&path, raw).expect("golden journey log write should succeed");
}

fn make_ledger(spec: ProgramSpec, transforms: &[Transform]) -> TraceTransformLedger {
    let mut ledger = TraceTransformLedger::new(build_program(spec));
    for (idx, t) in transforms.iter().enumerate() {
        ledger.push_transform(*t, format!("gj-{}-{idx}", t.as_str()));
    }
    ledger
}

fn make_request(spec: ProgramSpec, transforms: &[Transform], args: Vec<Value>) -> DispatchRequest {
    DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger: make_ledger(spec, transforms),
        args,
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    }
}

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

// ---------------------------------------------------------------------------
// Journey 1: Basic Transform — jit(add2)(3, 5) = 8
// ---------------------------------------------------------------------------

#[test]
fn golden_journey_01_basic_jit_transform() {
    let start = Instant::now();
    let mut assertions = Vec::new();

    // Setup: jit(add2) with scalar inputs
    let resp = dispatch(make_request(
        ProgramSpec::Add2,
        &[Transform::Jit],
        vec![Value::scalar_i64(3), Value::scalar_i64(5)],
    ))
    .expect("jit dispatch should succeed");

    // Assert output correctness
    let output_val = resp.outputs[0].as_scalar_literal().and_then(|l| l.as_i64());
    assertions.push(JourneyAssertion {
        name: "output_equals_8".into(),
        passed: output_val == Some(8),
        detail: format!("expected 8, got {:?}", output_val),
    });

    // Assert cache key is present
    assertions.push(JourneyAssertion {
        name: "cache_key_present".into(),
        passed: resp.cache_key.starts_with("fjx-"),
        detail: format!("cache_key={}", resp.cache_key),
    });

    // Assert evidence ledger non-empty
    assertions.push(JourneyAssertion {
        name: "evidence_ledger_populated".into(),
        passed: resp.evidence_ledger.len() == 1,
        detail: format!("ledger_len={}", resp.evidence_ledger.len()),
    });

    // Compare with eager (non-jit) evaluation
    let eager_resp = dispatch(make_request(
        ProgramSpec::Add2,
        &[], // no transforms = eager
        vec![Value::scalar_i64(3), Value::scalar_i64(5)],
    ))
    .expect("eager dispatch should succeed");
    let eager_val = eager_resp.outputs[0]
        .as_scalar_literal()
        .and_then(|l| l.as_i64());
    assertions.push(JourneyAssertion {
        name: "jit_matches_eager".into(),
        passed: output_val == eager_val,
        detail: format!("jit={:?}, eager={:?}", output_val, eager_val),
    });

    let all_passed = assertions.iter().all(|a| a.passed);
    let duration_ms = start.elapsed().as_millis();

    write_journey_log(&GoldenJourneyLog {
        schema_version: "frankenjax.golden-journey.v1",
        scenario_id: "gj_01_basic_jit_transform".into(),
        scenario_category: "basic_transform".into(),
        ts_utc_unix_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis(),
        replay_command: "cargo test -p fj-conformance --test golden_journeys -- golden_journey_01 --exact --nocapture".into(),
        input_capture: json!({"program": "Add2", "transforms": ["Jit"], "args": [3, 5]}),
        assertions: assertions.clone(),
        output_capture: json!({"output": output_val, "cache_key": resp.cache_key}),
        result: if all_passed { "pass" } else { "fail" }.into(),
        duration_ms,
    });

    for a in &assertions {
        assert!(a.passed, "assertion '{}' failed: {}", a.name, a.detail);
    }
}

// ---------------------------------------------------------------------------
// Journey 2: Gradient Computation — grad(square)(x) = 2*x
// ---------------------------------------------------------------------------

#[test]
fn golden_journey_02_gradient_computation() {
    let start = Instant::now();
    let mut assertions = Vec::new();

    let test_inputs: Vec<f64> = vec![1.0, 3.0, 5.0, -2.0, 0.0];

    for &x in &test_inputs {
        let resp = dispatch(make_request(
            ProgramSpec::Square,
            &[Transform::Grad],
            vec![Value::scalar_f64(x)],
        ))
        .expect("grad dispatch should succeed");

        let derivative = resp.outputs[0].as_f64_scalar().expect("grad returns f64");
        let expected = 2.0 * x;
        let matches = (derivative - expected).abs() < 1e-4;

        assertions.push(JourneyAssertion {
            name: format!("grad_square_at_{x}"),
            passed: matches,
            detail: format!("expected {expected}, got {derivative}"),
        });
    }

    let all_passed = assertions.iter().all(|a| a.passed);
    let duration_ms = start.elapsed().as_millis();

    write_journey_log(&GoldenJourneyLog {
        schema_version: "frankenjax.golden-journey.v1",
        scenario_id: "gj_02_gradient_computation".into(),
        scenario_category: "gradient_computation".into(),
        ts_utc_unix_ms: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis(),
        replay_command: "cargo test -p fj-conformance --test golden_journeys -- golden_journey_02 --exact --nocapture".into(),
        input_capture: json!({"program": "Square", "transforms": ["Grad"], "inputs": test_inputs}),
        assertions: assertions.clone(),
        output_capture: json!({"verified_inputs": test_inputs.len()}),
        result: if all_passed { "pass" } else { "fail" }.into(),
        duration_ms,
    });

    for a in &assertions {
        assert!(a.passed, "assertion '{}' failed: {}", a.name, a.detail);
    }
}

// ---------------------------------------------------------------------------
// Journey 3: Batched Computation — vmap(add_one) vs manual loop
// ---------------------------------------------------------------------------

#[test]
fn golden_journey_03_batched_computation() {
    let start = Instant::now();
    let mut assertions = Vec::new();

    let input_data = vec![10, 20, 30, 40, 50];
    let vec_arg = Value::vector_i64(&input_data).expect("vector should build");

    // vmap execution
    let resp = dispatch(make_request(
        ProgramSpec::AddOne,
        &[Transform::Vmap],
        vec![vec_arg],
    ))
    .expect("vmap dispatch should succeed");

    let vmap_output = resp.outputs[0]
        .as_tensor()
        .expect("vmap output should be tensor");
    let vmap_values: Vec<i64> = vmap_output
        .elements
        .iter()
        .map(|l| l.as_i64().unwrap())
        .collect();

    // Manual loop comparison
    let manual_values: Vec<i64> = input_data.iter().map(|x| x + 1).collect();

    assertions.push(JourneyAssertion {
        name: "vmap_matches_manual_loop".into(),
        passed: vmap_values == manual_values,
        detail: format!("vmap={:?}, manual={:?}", vmap_values, manual_values),
    });

    assertions.push(JourneyAssertion {
        name: "output_length_matches_input".into(),
        passed: vmap_values.len() == input_data.len(),
        detail: format!(
            "output_len={}, input_len={}",
            vmap_values.len(),
            input_data.len()
        ),
    });

    let all_passed = assertions.iter().all(|a| a.passed);
    let duration_ms = start.elapsed().as_millis();

    write_journey_log(&GoldenJourneyLog {
        schema_version: "frankenjax.golden-journey.v1",
        scenario_id: "gj_03_batched_computation".into(),
        scenario_category: "batched_computation".into(),
        ts_utc_unix_ms: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis(),
        replay_command: "cargo test -p fj-conformance --test golden_journeys -- golden_journey_03 --exact --nocapture".into(),
        input_capture: json!({"program": "AddOne", "transforms": ["Vmap"], "input_data": input_data}),
        assertions: assertions.clone(),
        output_capture: json!({"vmap_output": vmap_values, "manual_output": manual_values}),
        result: if all_passed { "pass" } else { "fail" }.into(),
        duration_ms,
    });

    for a in &assertions {
        assert!(a.passed, "assertion '{}' failed: {}", a.name, a.detail);
    }
}

// ---------------------------------------------------------------------------
// Journey 4: Transform Composition — order-sensitive dispatch
// ---------------------------------------------------------------------------

#[test]
fn golden_journey_04_transform_composition() {
    let start = Instant::now();
    let mut assertions = Vec::new();

    // jit(grad(square))(3.0) = 6.0
    let resp_jg = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Jit, Transform::Grad],
        vec![Value::scalar_f64(3.0)],
    ))
    .expect("jit(grad(f)) should succeed");
    let jg_val = resp_jg.outputs[0].as_f64_scalar().unwrap();

    assertions.push(JourneyAssertion {
        name: "jit_grad_square_at_3".into(),
        passed: (jg_val - 6.0).abs() < 1e-4,
        detail: format!("expected 6.0, got {jg_val}"),
    });

    // vmap(grad(square))([1.0, 2.0, 3.0]) = [2.0, 4.0, 6.0]
    let resp_vg = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Vmap, Transform::Grad],
        vec![Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap()],
    ))
    .expect("vmap(grad(f)) should succeed");
    let vg_tensor = resp_vg.outputs[0].as_tensor().unwrap();
    let vg_values = vg_tensor.to_f64_vec().unwrap();

    assertions.push(JourneyAssertion {
        name: "vmap_grad_square_at_1_2_3".into(),
        passed: vg_values.len() == 3
            && (vg_values[0] - 2.0).abs() < 1e-3
            && (vg_values[1] - 4.0).abs() < 1e-3
            && (vg_values[2] - 6.0).abs() < 1e-3,
        detail: format!("expected [2.0,4.0,6.0], got {:?}", vg_values),
    });

    // grad(vmap(f)) should fail — wrong ordering
    let bad_result = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Grad, Transform::Vmap],
        vec![Value::vector_f64(&[1.0, 2.0]).unwrap()],
    ));
    assertions.push(JourneyAssertion {
        name: "grad_vmap_rejects_wrong_order".into(),
        passed: bad_result.is_err(),
        detail: format!("result={:?}", bad_result.is_err()),
    });

    // Composition proofs are valid for accepted orderings
    let proof_jg = verify_transform_composition(&make_ledger(
        ProgramSpec::Square,
        &[Transform::Jit, Transform::Grad],
    ));
    assertions.push(JourneyAssertion {
        name: "composition_proof_jit_grad_valid".into(),
        passed: proof_jg.is_ok(),
        detail: format!("{:?}", proof_jg),
    });

    let all_passed = assertions.iter().all(|a| a.passed);
    let duration_ms = start.elapsed().as_millis();

    write_journey_log(&GoldenJourneyLog {
        schema_version: "frankenjax.golden-journey.v1",
        scenario_id: "gj_04_transform_composition".into(),
        scenario_category: "transform_composition".into(),
        ts_utc_unix_ms: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis(),
        replay_command: "cargo test -p fj-conformance --test golden_journeys -- golden_journey_04 --exact --nocapture".into(),
        input_capture: json!({"compositions_tested": ["jit+grad", "vmap+grad", "grad+vmap(rejected)"]}),
        assertions: assertions.clone(),
        output_capture: json!({"jit_grad_at_3": jg_val, "vmap_grad": vg_values}),
        result: if all_passed { "pass" } else { "fail" }.into(),
        duration_ms,
    });

    for a in &assertions {
        assert!(a.passed, "assertion '{}' failed: {}", a.name, a.detail);
    }
}

// ---------------------------------------------------------------------------
// Journey 5: Cache Hit/Miss — repeat dispatch, verify cache key determinism
// ---------------------------------------------------------------------------

#[test]
fn golden_journey_05_cache_hit_miss() {
    let start = Instant::now();
    let mut assertions = Vec::new();

    // Run same dispatch 10 times, collect cache keys
    let mut cache_keys = Vec::new();
    for _ in 0..10 {
        let resp = dispatch(make_request(
            ProgramSpec::Add2,
            &[Transform::Jit],
            vec![Value::scalar_i64(7), Value::scalar_i64(11)],
        ))
        .expect("repeated dispatch should succeed");
        cache_keys.push(resp.cache_key.clone());
    }

    // All cache keys should be identical (deterministic)
    let all_same = cache_keys.windows(2).all(|w| w[0] == w[1]);
    assertions.push(JourneyAssertion {
        name: "cache_keys_deterministic_10x".into(),
        passed: all_same,
        detail: format!(
            "unique_keys={}",
            cache_keys
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
        ),
    });

    // Different inputs should produce different cache keys
    let resp_other = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Jit],
        vec![Value::scalar_i64(7)],
    ))
    .expect("different program dispatch should succeed");
    assertions.push(JourneyAssertion {
        name: "different_program_different_key".into(),
        passed: resp_other.cache_key != cache_keys[0],
        detail: format!(
            "add2_key={}, square_key={}",
            cache_keys[0], resp_other.cache_key
        ),
    });

    // Different transforms should produce different cache keys
    let resp_grad = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Grad],
        vec![Value::scalar_f64(3.0)],
    ))
    .expect("grad dispatch should succeed");
    assertions.push(JourneyAssertion {
        name: "different_transform_different_key".into(),
        passed: resp_grad.cache_key != resp_other.cache_key,
        detail: format!(
            "jit_key={}, grad_key={}",
            resp_other.cache_key, resp_grad.cache_key
        ),
    });

    let all_passed = assertions.iter().all(|a| a.passed);
    let duration_ms = start.elapsed().as_millis();

    write_journey_log(&GoldenJourneyLog {
        schema_version: "frankenjax.golden-journey.v1",
        scenario_id: "gj_05_cache_hit_miss".into(),
        scenario_category: "cache_hit_miss".into(),
        ts_utc_unix_ms: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis(),
        replay_command: "cargo test -p fj-conformance --test golden_journeys -- golden_journey_05 --exact --nocapture".into(),
        input_capture: json!({"repeat_count": 10, "programs": ["Add2", "Square"]}),
        assertions: assertions.clone(),
        output_capture: json!({"cache_key_sample": &cache_keys[0]}),
        result: if all_passed { "pass" } else { "fail" }.into(),
        duration_ms,
    });

    for a in &assertions {
        assert!(a.passed, "assertion '{}' failed: {}", a.name, a.detail);
    }
}

// ---------------------------------------------------------------------------
// Journey 6: Error Recovery — malformed input, actionable error messages
// ---------------------------------------------------------------------------

#[test]
fn golden_journey_06_error_recovery() {
    let start = Instant::now();
    let mut assertions = Vec::new();

    // 6a: grad requires scalar input, tensor input should fail
    let err_tensor_grad = dispatch(make_request(
        ProgramSpec::Square,
        &[Transform::Grad],
        vec![Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap()],
    ));
    assertions.push(JourneyAssertion {
        name: "grad_rejects_tensor_input".into(),
        passed: err_tensor_grad.is_err(),
        detail: format!("error={:?}", err_tensor_grad.err()),
    });

    // 6b: vmap requires tensor input, not scalar
    let err_scalar_vmap = dispatch(make_request(
        ProgramSpec::AddOne,
        &[Transform::Vmap],
        vec![Value::scalar_i64(42)],
    ));
    assertions.push(JourneyAssertion {
        name: "vmap_rejects_scalar_input".into(),
        passed: err_scalar_vmap.is_err(),
        detail: format!("error={:?}", err_scalar_vmap.err()),
    });

    // 6c: strict mode rejects unknown features with actionable message
    let err_strict = dispatch(DispatchRequest {
        mode: CompatibilityMode::Strict,
        ledger: make_ledger(ProgramSpec::Add2, &[Transform::Jit]),
        args: vec![Value::scalar_i64(1), Value::scalar_i64(2)],
        backend: "cpu".to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec!["future.feature.v99".into()],
    });
    let err_msg = format!("{}", err_strict.as_ref().unwrap_err());
    assertions.push(JourneyAssertion {
        name: "strict_rejects_unknown_features".into(),
        passed: err_strict.is_err(),
        detail: err_msg.clone(),
    });
    assertions.push(JourneyAssertion {
        name: "error_message_mentions_feature_name".into(),
        passed: err_msg.contains("future.feature.v99"),
        detail: format!(
            "message includes feature name: {}",
            err_msg.contains("future.feature.v99")
        ),
    });

    // 6d: wrong arity — add2 expects 2 args, give 1
    let err_arity = dispatch(make_request(
        ProgramSpec::Add2,
        &[Transform::Jit],
        vec![Value::scalar_i64(3)], // missing second arg
    ));
    assertions.push(JourneyAssertion {
        name: "wrong_arity_produces_error".into(),
        passed: err_arity.is_err(),
        detail: format!("error={:?}", err_arity.err()),
    });

    let all_passed = assertions.iter().all(|a| a.passed);
    let duration_ms = start.elapsed().as_millis();

    write_journey_log(&GoldenJourneyLog {
        schema_version: "frankenjax.golden-journey.v1",
        scenario_id: "gj_06_error_recovery".into(),
        scenario_category: "error_recovery".into(),
        ts_utc_unix_ms: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis(),
        replay_command: "cargo test -p fj-conformance --test golden_journeys -- golden_journey_06 --exact --nocapture".into(),
        input_capture: json!({"error_cases": ["tensor_grad", "scalar_vmap", "strict_unknown_features", "wrong_arity"]}),
        assertions: assertions.clone(),
        output_capture: json!({"all_errors_caught": all_passed}),
        result: if all_passed { "pass" } else { "fail" }.into(),
        duration_ms,
    });

    for a in &assertions {
        assert!(a.passed, "assertion '{}' failed: {}", a.name, a.detail);
    }
}

// ---------------------------------------------------------------------------
// Journey 7: Large Program — 100+ equations, correctness and timing
// ---------------------------------------------------------------------------

#[test]
fn golden_journey_07_large_program() {
    let start = Instant::now();
    let mut assertions = Vec::new();

    let n = 200;
    let jaxpr = build_chain_jaxpr(n);

    // Verify well-formedness
    let valid = jaxpr.validate_well_formed();
    assertions.push(JourneyAssertion {
        name: "large_jaxpr_well_formed".into(),
        passed: valid.is_ok(),
        detail: format!("equations={n}, valid={:?}", valid),
    });

    // Evaluate — chain of 200 +1 operations from 0 should give 200
    let eval_start = Instant::now();
    let result = fj_interpreters::eval_jaxpr(&jaxpr, &[Value::scalar_i64(0)])
        .expect("large jaxpr eval should succeed");
    let eval_ms = eval_start.elapsed().as_millis();

    let output_val = result[0].as_scalar_literal().and_then(|l| l.as_i64());
    assertions.push(JourneyAssertion {
        name: "large_jaxpr_correct_output".into(),
        passed: output_val == Some(n as i64),
        detail: format!("expected {n}, got {:?}", output_val),
    });

    // Timing: 200 equations should complete in under 30ms (3x headroom for CI)
    assertions.push(JourneyAssertion {
        name: "large_jaxpr_timing_under_30ms".into(),
        passed: eval_ms < 30,
        detail: format!("eval_ms={eval_ms}"),
    });

    // Fingerprint should be non-empty and computed
    let fp_start = Instant::now();
    let fp = jaxpr.canonical_fingerprint();
    let fp_ms = fp_start.elapsed().as_millis();

    assertions.push(JourneyAssertion {
        name: "large_jaxpr_fingerprint_non_empty".into(),
        passed: !fp.is_empty(),
        detail: format!("fingerprint_len={}", fp.len()),
    });

    // Fingerprint should complete in under 10ms (3x headroom)
    assertions.push(JourneyAssertion {
        name: "fingerprint_timing_under_10ms".into(),
        passed: fp_ms < 10,
        detail: format!("fingerprint_ms={fp_ms}"),
    });

    let all_passed = assertions.iter().all(|a| a.passed);
    let duration_ms = start.elapsed().as_millis();

    write_journey_log(&GoldenJourneyLog {
        schema_version: "frankenjax.golden-journey.v1",
        scenario_id: "gj_07_large_program".into(),
        scenario_category: "large_program".into(),
        ts_utc_unix_ms: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis(),
        replay_command: "cargo test -p fj-conformance --test golden_journeys -- golden_journey_07 --exact --nocapture".into(),
        input_capture: json!({"equation_count": n, "program": "chain_add"}),
        assertions: assertions.clone(),
        output_capture: json!({"output": output_val, "eval_ms": eval_ms, "fingerprint_ms": fp_ms}),
        result: if all_passed { "pass" } else { "fail" }.into(),
        duration_ms,
    });

    for a in &assertions {
        assert!(a.passed, "assertion '{}' failed: {}", a.name, a.detail);
    }
}

// ---------------------------------------------------------------------------
// Journey 8: Transcendental Gradients — inverse hyperbolic functions
// ---------------------------------------------------------------------------

#[test]
fn golden_journey_08_transcendental_gradients() {
    let start = Instant::now();
    let mut assertions = Vec::new();

    // Test grad of inverse hyperbolic functions against analytical derivatives
    // asinh'(x) = 1/sqrt(x^2 + 1)
    // acosh'(x) = 1/sqrt(x^2 - 1), x > 1
    // atanh'(x) = 1/(1 - x^2), |x| < 1

    // asinh at multiple points
    for &x in &[0.5f64, 1.0, 2.0, -1.5] {
        let resp = dispatch(make_request(
            ProgramSpec::LaxAsinh,
            &[Transform::Grad],
            vec![Value::scalar_f64(x)],
        ))
        .expect("grad(asinh) should succeed");

        let derivative = resp.outputs[0].as_f64_scalar().expect("grad returns f64");
        let expected = 1.0 / (x * x + 1.0).sqrt();
        let matches = (derivative - expected).abs() < 1e-4;

        assertions.push(JourneyAssertion {
            name: format!("grad_asinh_at_{x}"),
            passed: matches,
            detail: format!("expected {expected}, got {derivative}"),
        });
    }

    // acosh at x > 1 (domain constraint)
    for &x in &[1.5f64, 2.0, 3.0] {
        let resp = dispatch(make_request(
            ProgramSpec::LaxAcosh,
            &[Transform::Grad],
            vec![Value::scalar_f64(x)],
        ))
        .expect("grad(acosh) should succeed");

        let derivative = resp.outputs[0].as_f64_scalar().expect("grad returns f64");
        let expected = 1.0 / (x * x - 1.0).sqrt();
        let matches = (derivative - expected).abs() < 1e-4;

        assertions.push(JourneyAssertion {
            name: format!("grad_acosh_at_{x}"),
            passed: matches,
            detail: format!("expected {expected}, got {derivative}"),
        });
    }

    // atanh at |x| < 1 (domain constraint)
    for &x in &[0.3f64, 0.5, -0.4, 0.8] {
        let resp = dispatch(make_request(
            ProgramSpec::LaxAtanh,
            &[Transform::Grad],
            vec![Value::scalar_f64(x)],
        ))
        .expect("grad(atanh) should succeed");

        let derivative = resp.outputs[0].as_f64_scalar().expect("grad returns f64");
        let expected = 1.0 / (1.0 - x * x);
        let matches = (derivative - expected).abs() < 1e-4;

        assertions.push(JourneyAssertion {
            name: format!("grad_atanh_at_{x}"),
            passed: matches,
            detail: format!("expected {expected}, got {derivative}"),
        });
    }

    let all_passed = assertions.iter().all(|a| a.passed);
    let duration_ms = start.elapsed().as_millis();

    write_journey_log(&GoldenJourneyLog {
        schema_version: "frankenjax.golden-journey.v1",
        scenario_id: "gj_08_transcendental_gradients".into(),
        scenario_category: "transcendental_gradients".into(),
        ts_utc_unix_ms: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis(),
        replay_command: "cargo test -p fj-conformance --test golden_journeys -- golden_journey_08_transcendental --exact --nocapture".into(),
        input_capture: json!({"functions": ["asinh", "acosh", "atanh"], "transform": "Grad"}),
        assertions: assertions.clone(),
        output_capture: json!({"total_assertions": assertions.len(), "all_passed": all_passed}),
        result: if all_passed { "pass" } else { "fail" }.into(),
        duration_ms,
    });

    for a in &assertions {
        assert!(a.passed, "assertion '{}' failed: {}", a.name, a.detail);
    }
}

// ---------------------------------------------------------------------------
// Journey 9: Ledger Inspection — multi-dispatch session consistency
// ---------------------------------------------------------------------------

#[test]
fn golden_journey_09_ledger_inspection() {
    let start = Instant::now();
    let mut assertions = Vec::new();

    // Run multiple dispatches and collect evidence ledgers
    let dispatches: Vec<(ProgramSpec, &[Transform], Vec<Value>)> = vec![
        (
            ProgramSpec::Add2,
            &[Transform::Jit][..],
            vec![Value::scalar_i64(1), Value::scalar_i64(2)],
        ),
        (
            ProgramSpec::Square,
            &[Transform::Grad][..],
            vec![Value::scalar_f64(4.0)],
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
    ];

    let mut all_ledgers = Vec::new();
    let mut all_cache_keys = Vec::new();

    for (spec, transforms, args) in &dispatches {
        let resp = dispatch(make_request(*spec, transforms, args.clone()))
            .expect("dispatch should succeed");
        all_cache_keys.push(resp.cache_key.clone());
        all_ledgers.push(resp.evidence_ledger);
    }

    // Each dispatch should produce exactly one ledger entry
    for (i, ledger) in all_ledgers.iter().enumerate() {
        assertions.push(JourneyAssertion {
            name: format!("ledger_{i}_has_one_entry"),
            passed: ledger.len() == 1,
            detail: format!("dispatch_{i} ledger_len={}", ledger.len()),
        });
    }

    // All cache keys should be unique (different programs/transforms)
    let unique_keys: std::collections::HashSet<&String> = all_cache_keys.iter().collect();
    assertions.push(JourneyAssertion {
        name: "all_cache_keys_unique".into(),
        passed: unique_keys.len() == all_cache_keys.len(),
        detail: format!(
            "total={}, unique={}",
            all_cache_keys.len(),
            unique_keys.len()
        ),
    });

    // Verify ledger consistency: decision_id matches cache key
    for (i, (ledger, key)) in all_ledgers.iter().zip(all_cache_keys.iter()).enumerate() {
        let entries = ledger.entries();
        let decision_id = &entries[0].decision_id;
        assertions.push(JourneyAssertion {
            name: format!("ledger_{i}_decision_id_matches_cache_key"),
            passed: decision_id == key,
            detail: format!("decision_id={}, cache_key={}", decision_id, key),
        });
    }

    let all_passed = assertions.iter().all(|a| a.passed);
    let duration_ms = start.elapsed().as_millis();

    write_journey_log(&GoldenJourneyLog {
        schema_version: "frankenjax.golden-journey.v1",
        scenario_id: "gj_09_ledger_inspection".into(),
        scenario_category: "ledger_inspection".into(),
        ts_utc_unix_ms: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis(),
        replay_command: "cargo test -p fj-conformance --test golden_journeys -- golden_journey_09 --exact --nocapture".into(),
        input_capture: json!({"dispatch_count": dispatches.len(), "programs": ["Add2", "Square", "AddOne", "Square"]}),
        assertions: assertions.clone(),
        output_capture: json!({"cache_keys": all_cache_keys, "ledger_lengths": all_ledgers.iter().map(|l| l.len()).collect::<Vec<_>>()}),
        result: if all_passed { "pass" } else { "fail" }.into(),
        duration_ms,
    });

    for a in &assertions {
        assert!(a.passed, "assertion '{}' failed: {}", a.name, a.detail);
    }
}
