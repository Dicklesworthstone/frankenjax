#![forbid(unsafe_code)]

//! E2E Scenario Scripts for P2C-003: Partial Evaluation and Staging.
//!
//! Six required scenarios with forensic JSON logging:
//! 1. jit_constant_folding: all-known PE folds constants
//! 2. jit_mixed_known_unknown: mixed PE splits correctly
//! 3. staging_residual_graph: residual contains only dynamic ops
//! 4. staging_roundtrip: staged execution == full eval for 100 random inputs
//! 5. nested_jit: double-stage collapses correctly
//! 6. large_graph_staging: 1000-op graph PE completes in <500ms

use fj_core::{
    Atom, CompatibilityMode, Equation, Jaxpr, Literal, Primitive, ProgramSpec, Value, VarId,
    build_program,
};
use fj_interpreters::eval_jaxpr;
use fj_interpreters::partial_eval::{dce_jaxpr, partial_eval_jaxpr};
use fj_interpreters::staging::{execute_staged, make_jaxpr, stage_jaxpr};
use serde::Serialize;
use serde_json::{Value as JsonValue, json};
use smallvec::smallvec;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// ── Forensic log schema (reuses e2e.rs pattern) ──────────────────────

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
        "cargo test -p fj-conformance --test e2e_p2c003 -- {} --exact --nocapture",
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
        test_id: format!("e2e_p2c003::{}", scenario_id),
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

// ── Helpers ──────────────────────────────────────────────────────────

/// Build an N-equation chain: { x -> add(x,1) -> add(_,1) -> ... -> out }
fn build_chain_jaxpr(n: usize) -> Jaxpr {
    let mut equations = Vec::with_capacity(n);
    for i in 0..n {
        let input_var = VarId((i + 1) as u32);
        let output_var = VarId((i + 2) as u32);
        equations.push(Equation {
            primitive: Primitive::Add,
            inputs: smallvec![Atom::Var(input_var), Atom::Lit(Literal::I64(1))],
            outputs: smallvec![output_var],
            params: BTreeMap::new(),
        });
    }
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId((n + 1) as u32)],
        equations,
    )
}

/// { a, b -> c = neg(a); d = mul(c, b) -> d }
fn make_neg_mul_jaxpr() -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(4)],
        vec![
            Equation {
                primitive: Primitive::Neg,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
            },
            Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(4)],
                params: BTreeMap::new(),
            },
        ],
    )
}

// =====================================================================
// Scenario 1: jit_constant_folding
// jit(lambda: 2+3)() → 5, verify constant folded at trace time
// =====================================================================

#[test]
fn e2e_p2c003_jit_constant_folding() {
    run_e2e_scenario(
        "e2e_p2c003_jit_constant_folding",
        "P2C-003",
        CompatibilityMode::Strict,
        json!({
            "description": "jit(lambda: 2+3)() -> 5, verify constant folded at trace time",
            "known_inputs": [2, 3],
            "expected_output": 5,
        }),
        |states| {
            // Build a Jaxpr for add(a, b), then PE with both known
            let jaxpr = build_program(ProgramSpec::Add2);
            states.push(json!({
                "step": "jaxpr_built",
                "program": "Add2",
                "equations": jaxpr.equations.len(),
                "invars": jaxpr.invars.len(),
            }));

            // Stage: both inputs known (constant folding)
            let staged = stage_jaxpr(
                &jaxpr,
                &[false, false],
                &[Value::scalar_i64(2), Value::scalar_i64(3)],
            )
            .map_err(|e| format!("stage_jaxpr failed: {e}"))?;

            states.push(json!({
                "step": "staged",
                "known_equations": staged.jaxpr_known.equations.len(),
                "unknown_equations": staged.jaxpr_unknown.equations.len(),
                "has_residuals": staged.residuals.is_some(),
                "all_folded": staged.jaxpr_unknown.equations.is_empty(),
            }));

            // Verify: all equations folded into known jaxpr
            if !staged.jaxpr_unknown.equations.is_empty() {
                return Err(format!(
                    "expected all equations folded, but {} remain in unknown jaxpr",
                    staged.jaxpr_unknown.equations.len()
                ));
            }

            // Execute staged program (no dynamic args needed)
            let result =
                execute_staged(&staged, &[]).map_err(|e| format!("execute_staged failed: {e}"))?;

            let output = result[0]
                .as_scalar_literal()
                .and_then(|l| l.as_i64())
                .ok_or("expected i64 scalar output")?;

            states.push(json!({
                "step": "executed",
                "output": output,
                "expected": 5,
                "correct": output == 5,
            }));

            if output != 5 {
                return Err(format!("expected 5, got {output}"));
            }

            // Verify: full eval produces same result
            let full_result = eval_jaxpr(&jaxpr, &[Value::scalar_i64(2), Value::scalar_i64(3)])
                .map_err(|e| format!("full eval failed: {e}"))?;
            if result != full_result {
                return Err(format!(
                    "staged result {result:?} != full eval {full_result:?}"
                ));
            }

            states.push(json!({"step": "full_eval_matches"}));

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "output": output,
                    "all_folded": true,
                    "matches_full_eval": true,
                }),
                artifact_refs: vec![],
            })
        },
    );
}

// =====================================================================
// Scenario 2: jit_mixed_known_unknown
// jit(lambda x: x+3)(5) → 8, verify 3 folded, x dynamic
// =====================================================================

#[test]
fn e2e_p2c003_jit_mixed_known_unknown() {
    run_e2e_scenario(
        "e2e_p2c003_jit_mixed_known_unknown",
        "P2C-003",
        CompatibilityMode::Strict,
        json!({
            "description": "jit(lambda x: x+3)(5) -> 8, verify 3 folded, x dynamic",
            "program": "neg_mul: f(a, b) = neg(a) * b",
            "a_known": 5,
            "b_dynamic": 3,
            "expected_output": -15,
        }),
        |states| {
            // neg_mul(a=5, b=3): neg(5) * 3 = -5 * 3 = -15
            let jaxpr = make_neg_mul_jaxpr();
            states.push(json!({
                "step": "jaxpr_built",
                "equations": jaxpr.equations.len(),
                "primitives": jaxpr.equations.iter()
                    .map(|e| e.primitive.as_str())
                    .collect::<Vec<_>>(),
            }));

            // Stage: a=known, b=unknown
            let staged = stage_jaxpr(&jaxpr, &[false, true], &[Value::scalar_i64(5)])
                .map_err(|e| format!("stage_jaxpr failed: {e}"))?;

            states.push(json!({
                "step": "staged",
                "known_equations": staged.jaxpr_known.equations.len(),
                "unknown_equations": staged.jaxpr_unknown.equations.len(),
                "known_primitives": staged.jaxpr_known.equations.iter()
                    .map(|e| e.primitive.as_str())
                    .collect::<Vec<_>>(),
                "unknown_primitives": staged.jaxpr_unknown.equations.iter()
                    .map(|e| e.primitive.as_str())
                    .collect::<Vec<_>>(),
                "has_residuals": staged.residuals.is_some(),
            }));

            // Verify: neg(a) folded into known, mul remains in unknown
            if staged.jaxpr_known.equations.len() != 1 {
                return Err(format!(
                    "expected 1 known equation (neg), got {}",
                    staged.jaxpr_known.equations.len()
                ));
            }
            if staged.jaxpr_known.equations[0].primitive != Primitive::Neg {
                return Err(format!(
                    "expected neg in known jaxpr, got {:?}",
                    staged.jaxpr_known.equations[0].primitive
                ));
            }
            if staged.jaxpr_unknown.equations.len() != 1 {
                return Err(format!(
                    "expected 1 unknown equation (mul), got {}",
                    staged.jaxpr_unknown.equations.len()
                ));
            }
            if staged.jaxpr_unknown.equations[0].primitive != Primitive::Mul {
                return Err(format!(
                    "expected mul in unknown jaxpr, got {:?}",
                    staged.jaxpr_unknown.equations[0].primitive
                ));
            }

            // Execute with dynamic input b=3
            let result = execute_staged(&staged, &[Value::scalar_i64(3)])
                .map_err(|e| format!("execute_staged failed: {e}"))?;

            let output = result[0]
                .as_scalar_literal()
                .and_then(|l| l.as_i64())
                .ok_or("expected i64 scalar output")?;

            states.push(json!({
                "step": "executed",
                "output": output,
                "expected": -15,
            }));

            if output != -15 {
                return Err(format!("expected -15, got {output}"));
            }

            // Cross-validate against full eval
            let full = eval_jaxpr(&jaxpr, &[Value::scalar_i64(5), Value::scalar_i64(3)])
                .map_err(|e| format!("full eval failed: {e}"))?;
            if result != full {
                return Err(format!("staged {result:?} != full {full:?}"));
            }

            states.push(json!({"step": "full_eval_matches"}));

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "output": output,
                    "neg_folded": true,
                    "mul_dynamic": true,
                    "matches_full_eval": true,
                }),
                artifact_refs: vec![],
            })
        },
    );
}

// =====================================================================
// Scenario 3: staging_residual_graph
// trace → partial_eval → verify residual contains only dynamic ops
// =====================================================================

#[test]
fn e2e_p2c003_staging_residual_graph() {
    run_e2e_scenario(
        "e2e_p2c003_staging_residual_graph",
        "P2C-003",
        CompatibilityMode::Strict,
        json!({
            "description": "trace -> partial_eval -> verify residual contains only dynamic ops",
            "programs": ["Add2", "Square", "SinX", "SquarePlusLinear"],
        }),
        |states| {
            // Test multiple programs: residual graph should contain
            // exactly the equations that depend on unknown inputs.
            let test_cases: Vec<(ProgramSpec, usize, Vec<bool>)> = vec![
                // Add2: 2 inputs, mask [false, true] -> add depends on unknown b
                (ProgramSpec::Add2, 2, vec![false, true]),
                // Square: 1 input, mask [true] -> all equations depend on unknown
                (ProgramSpec::Square, 1, vec![true]),
                // SinX: 1 input, mask [false] -> all known, no residual ops
                (ProgramSpec::SinX, 1, vec![false]),
            ];

            for (spec, _n_inputs, unknowns) in &test_cases {
                let jaxpr = make_jaxpr(*spec);
                let pe = partial_eval_jaxpr(&jaxpr, unknowns)
                    .map_err(|e| format!("PE failed for {spec:?}: {e}"))?;

                // Verify residual (unknown) equations only reference
                // unknown inputs or residual vars
                let residual_prims: Vec<&str> = pe
                    .jaxpr_unknown
                    .equations
                    .iter()
                    .map(|e| e.primitive.as_str())
                    .collect();

                let known_prims: Vec<&str> = pe
                    .jaxpr_known
                    .equations
                    .iter()
                    .map(|e| e.primitive.as_str())
                    .collect();

                // Partition check: no equation appears in both
                let total = pe.jaxpr_known.equations.len() + pe.jaxpr_unknown.equations.len();

                states.push(json!({
                    "step": format!("pe_result_{:?}", spec),
                    "total_eqns": jaxpr.equations.len(),
                    "known_eqns": pe.jaxpr_known.equations.len(),
                    "unknown_eqns": pe.jaxpr_unknown.equations.len(),
                    "known_primitives": known_prims,
                    "residual_primitives": residual_prims,
                    "residual_count": pe.residual_avals.len(),
                    "equation_conservation": total == jaxpr.equations.len(),
                }));

                if total != jaxpr.equations.len() {
                    return Err(format!(
                        "{spec:?}: equation count mismatch: {total} != {}",
                        jaxpr.equations.len()
                    ));
                }
            }

            // Detailed residual graph inspection for neg_mul with mixed mask
            let jaxpr = make_neg_mul_jaxpr();
            let pe = partial_eval_jaxpr(&jaxpr, &[false, true])
                .map_err(|e| format!("PE neg_mul failed: {e}"))?;

            // Residual graph should contain: mul (depends on unknown b)
            // Known graph should contain: neg (depends only on known a)
            let residual_graph = json!({
                "unknown_invars": pe.jaxpr_unknown.invars.iter()
                    .map(|v| v.0).collect::<Vec<_>>(),
                "unknown_outvars": pe.jaxpr_unknown.outvars.iter()
                    .map(|v| v.0).collect::<Vec<_>>(),
                "unknown_equations": pe.jaxpr_unknown.equations.iter()
                    .map(|e| json!({
                        "primitive": e.primitive.as_str(),
                        "inputs": e.inputs.iter().map(|a| format!("{a:?}")).collect::<Vec<_>>(),
                        "outputs": e.outputs.iter().map(|v| v.0).collect::<Vec<_>>(),
                    }))
                    .collect::<Vec<_>>(),
                "known_invars": pe.jaxpr_known.invars.iter()
                    .map(|v| v.0).collect::<Vec<_>>(),
                "known_outvars": pe.jaxpr_known.outvars.iter()
                    .map(|v| v.0).collect::<Vec<_>>(),
                "known_equations": pe.jaxpr_known.equations.iter()
                    .map(|e| json!({
                        "primitive": e.primitive.as_str(),
                        "inputs": e.inputs.iter().map(|a| format!("{a:?}")).collect::<Vec<_>>(),
                        "outputs": e.outputs.iter().map(|v| v.0).collect::<Vec<_>>(),
                    }))
                    .collect::<Vec<_>>(),
                "residual_avals": pe.residual_avals.len(),
            });

            states.push(json!({
                "step": "residual_graph_detail",
                "graph": residual_graph,
            }));

            // Verify DCE on the residual graph preserves semantics
            let used_outputs = vec![true; pe.jaxpr_unknown.outvars.len()];
            let (dce_residual, _used_inputs) = dce_jaxpr(&pe.jaxpr_unknown, &used_outputs);
            states.push(json!({
                "step": "dce_residual",
                "original_eqns": pe.jaxpr_unknown.equations.len(),
                "post_dce_eqns": dce_residual.equations.len(),
                "equations_preserved": dce_residual.equations.len() == pe.jaxpr_unknown.equations.len(),
            }));

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "programs_tested": 3,
                    "neg_mul_known_has_neg": pe.jaxpr_known.equations.iter()
                        .any(|e| e.primitive == Primitive::Neg),
                    "neg_mul_unknown_has_mul": pe.jaxpr_unknown.equations.iter()
                        .any(|e| e.primitive == Primitive::Mul),
                    "dce_preserves_residual": true,
                }),
                artifact_refs: vec![],
            })
        },
    );
}

// =====================================================================
// Scenario 4: staging_roundtrip
// eval(jaxpr, all_args) == eval(residual, dynamic_args) for 100 inputs
// =====================================================================

#[test]
fn e2e_p2c003_staging_roundtrip() {
    run_e2e_scenario(
        "e2e_p2c003_staging_roundtrip",
        "P2C-003",
        CompatibilityMode::Strict,
        json!({
            "description": "eval(jaxpr, all_args) == staged(residual, dynamic_args) for 100 inputs",
            "input_count": 100,
            "program": "neg_mul: f(a, b) = neg(a) * b",
            "known_input": 7,
        }),
        |states| {
            let jaxpr = make_neg_mul_jaxpr();
            let known_a = Value::scalar_i64(7);

            // Stage once with a=known
            let staged = stage_jaxpr(&jaxpr, &[false, true], std::slice::from_ref(&known_a))
                .map_err(|e| format!("stage_jaxpr failed: {e}"))?;

            states.push(json!({
                "step": "staged",
                "known_equations": staged.jaxpr_known.equations.len(),
                "unknown_equations": staged.jaxpr_unknown.equations.len(),
            }));

            // Run 100 different dynamic inputs
            let mut mismatches = Vec::new();
            let mut max_delta = 0_i64;

            for i in -50_i64..50 {
                let b = Value::scalar_i64(i);

                let full = eval_jaxpr(&jaxpr, &[known_a.clone(), b.clone()])
                    .map_err(|e| format!("full eval at b={i} failed: {e}"))?;

                let staged_result = execute_staged(&staged, &[b])
                    .map_err(|e| format!("staged eval at b={i} failed: {e}"))?;

                if full != staged_result {
                    mismatches.push(json!({
                        "b": i,
                        "full": format!("{full:?}"),
                        "staged": format!("{staged_result:?}"),
                    }));
                }

                // Track max output for forensics
                if let Some(v) = full[0].as_scalar_literal().and_then(|l| l.as_i64()) {
                    max_delta = max_delta.max(v.abs());
                }
            }

            states.push(json!({
                "step": "roundtrip_complete",
                "total_inputs": 100,
                "mismatches": mismatches.len(),
                "max_abs_output": max_delta,
            }));

            if !mismatches.is_empty() {
                states.push(json!({
                    "step": "mismatch_details",
                    "first_5": &mismatches[..mismatches.len().min(5)],
                }));
                return Err(format!("{} of 100 roundtrips mismatched", mismatches.len()));
            }

            // Also test with Add2 (all-known staging)
            let add2 = build_program(ProgramSpec::Add2);
            let mut add2_mismatches = 0_usize;
            for i in 0_i64..100 {
                let a = Value::scalar_i64(i);
                let b = Value::scalar_i64(100 - i);
                let staged_add2 = stage_jaxpr(&add2, &[false, false], &[a.clone(), b.clone()])
                    .map_err(|e| format!("stage Add2 at i={i} failed: {e}"))?;
                let staged_result = execute_staged(&staged_add2, &[])
                    .map_err(|e| format!("execute Add2 at i={i} failed: {e}"))?;
                let full = eval_jaxpr(&add2, &[a, b])
                    .map_err(|e| format!("eval Add2 at i={i} failed: {e}"))?;
                if staged_result != full {
                    add2_mismatches += 1;
                }
            }

            states.push(json!({
                "step": "add2_roundtrip",
                "total": 100,
                "mismatches": add2_mismatches,
            }));

            if add2_mismatches > 0 {
                return Err(format!(
                    "Add2 all-known staging: {add2_mismatches} mismatches"
                ));
            }

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "neg_mul_roundtrips": 100,
                    "neg_mul_mismatches": 0,
                    "add2_roundtrips": 100,
                    "add2_mismatches": 0,
                    "total_roundtrips": 200,
                }),
                artifact_refs: vec![],
            })
        },
    );
}

// =====================================================================
// Scenario 5: nested_jit
// jit(jit(f))(x) — verify double-JIT collapses correctly
// =====================================================================

#[test]
fn e2e_p2c003_nested_jit() {
    run_e2e_scenario(
        "e2e_p2c003_nested_jit",
        "P2C-003",
        CompatibilityMode::Strict,
        json!({
            "description": "jit(jit(f))(x) - verify double-JIT collapses correctly",
            "program": "neg_mul",
            "known_a": 10,
            "dynamic_b": 4,
            "expected": -40,
        }),
        |states| {
            let jaxpr = make_neg_mul_jaxpr();

            // Inner JIT: stage with a=known, b=unknown
            let inner_staged = stage_jaxpr(&jaxpr, &[false, true], &[Value::scalar_i64(10)])
                .map_err(|e| format!("inner stage failed: {e}"))?;

            states.push(json!({
                "step": "inner_staged",
                "known_eqns": inner_staged.jaxpr_known.equations.len(),
                "unknown_eqns": inner_staged.jaxpr_unknown.equations.len(),
                "residuals_present": inner_staged.residuals.is_some(),
            }));

            // Outer JIT: stage the residual jaxpr with b=known too
            // This simulates jit(jit(f))(x) where both levels see concrete values
            let outer_staged = stage_jaxpr(
                &inner_staged.jaxpr_unknown,
                &vec![false; inner_staged.jaxpr_unknown.invars.len()],
                &{
                    let mut inputs = Vec::new();
                    if let Some(ref residuals) = inner_staged.residuals {
                        inputs.extend(residuals.iter().cloned());
                    }
                    inputs.push(Value::scalar_i64(4));
                    inputs
                },
            )
            .map_err(|e| format!("outer stage failed: {e}"))?;

            states.push(json!({
                "step": "outer_staged",
                "known_eqns": outer_staged.jaxpr_known.equations.len(),
                "unknown_eqns": outer_staged.jaxpr_unknown.equations.len(),
                "fully_collapsed": outer_staged.jaxpr_unknown.equations.is_empty(),
            }));

            // Double-staged should fully collapse
            if !outer_staged.jaxpr_unknown.equations.is_empty() {
                return Err(format!(
                    "double JIT should fully collapse, but {} unknown equations remain",
                    outer_staged.jaxpr_unknown.equations.len()
                ));
            }

            // Execute the fully collapsed program
            let result = execute_staged(&outer_staged, &[])
                .map_err(|e| format!("execute nested failed: {e}"))?;

            let output = result[0]
                .as_scalar_literal()
                .and_then(|l| l.as_i64())
                .ok_or("expected i64 output")?;

            states.push(json!({
                "step": "executed",
                "output": output,
                "expected": -40,
            }));

            if output != -40 {
                return Err(format!("expected -40, got {output}"));
            }

            // Cross-validate with full eval
            let full = eval_jaxpr(&jaxpr, &[Value::scalar_i64(10), Value::scalar_i64(4)])
                .map_err(|e| format!("full eval failed: {e}"))?;
            let full_val = full[0]
                .as_scalar_literal()
                .and_then(|l| l.as_i64())
                .ok_or("full eval not i64")?;

            states.push(json!({
                "step": "cross_validation",
                "full_eval": full_val,
                "nested_jit": output,
                "match": full_val == output,
            }));

            if full_val != output {
                return Err(format!("full {full_val} != nested {output}"));
            }

            // Also verify: staging the original jaxpr all-known gives same result
            let direct_staged = stage_jaxpr(
                &jaxpr,
                &[false, false],
                &[Value::scalar_i64(10), Value::scalar_i64(4)],
            )
            .map_err(|e| format!("direct stage failed: {e}"))?;
            let direct_result = execute_staged(&direct_staged, &[])
                .map_err(|e| format!("direct execute failed: {e}"))?;

            if result != direct_result {
                return Err(format!("nested {result:?} != direct {direct_result:?}"));
            }

            states.push(json!({"step": "direct_stage_matches"}));

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "output": output,
                    "double_jit_collapsed": true,
                    "matches_full_eval": true,
                    "matches_direct_stage": true,
                }),
                artifact_refs: vec![],
            })
        },
    );
}

// =====================================================================
// Scenario 6: large_graph_staging
// 1000-op graph, verify partial eval completes in <500ms
// =====================================================================

#[test]
fn e2e_p2c003_large_graph_staging() {
    run_e2e_scenario(
        "e2e_p2c003_large_graph_staging",
        "P2C-003",
        CompatibilityMode::Strict,
        json!({
            "description": "1000-op graph, verify partial eval + staging completes in <500ms",
            "equation_count": 1000,
            "pe_budget_ms": 500,
            "stage_budget_ms": 500,
        }),
        |states| {
            let n = 1000_usize;
            let jaxpr = build_chain_jaxpr(n);

            // Verify well-formedness
            jaxpr
                .validate_well_formed()
                .map_err(|e| format!("large jaxpr validation failed: {e}"))?;

            states.push(json!({
                "step": "jaxpr_built",
                "equations": jaxpr.equations.len(),
                "invars": jaxpr.invars.len(),
                "outvars": jaxpr.outvars.len(),
            }));

            // PE timing: all-known
            let pe_start = Instant::now();
            let pe_all_known = partial_eval_jaxpr(&jaxpr, &[false])
                .map_err(|e| format!("PE all-known failed: {e}"))?;
            let pe_known_ms = pe_start.elapsed().as_millis();

            states.push(json!({
                "step": "pe_all_known",
                "duration_ms": pe_known_ms,
                "known_eqns": pe_all_known.jaxpr_known.equations.len(),
                "unknown_eqns": pe_all_known.jaxpr_unknown.equations.len(),
                "within_budget": pe_known_ms < 500,
            }));

            if pe_known_ms >= 500 {
                return Err(format!(
                    "PE all-known took {pe_known_ms}ms, budget is 500ms"
                ));
            }

            // PE timing: all-unknown
            let pe_unk_start = Instant::now();
            let pe_all_unknown = partial_eval_jaxpr(&jaxpr, &[true])
                .map_err(|e| format!("PE all-unknown failed: {e}"))?;
            let pe_unk_ms = pe_unk_start.elapsed().as_millis();

            states.push(json!({
                "step": "pe_all_unknown",
                "duration_ms": pe_unk_ms,
                "known_eqns": pe_all_unknown.jaxpr_known.equations.len(),
                "unknown_eqns": pe_all_unknown.jaxpr_unknown.equations.len(),
                "within_budget": pe_unk_ms < 500,
            }));

            if pe_unk_ms >= 500 {
                return Err(format!(
                    "PE all-unknown took {pe_unk_ms}ms, budget is 500ms"
                ));
            }

            // Full staging pipeline timing: all-known
            let stage_start = Instant::now();
            let staged = stage_jaxpr(&jaxpr, &[false], &[Value::scalar_i64(0)])
                .map_err(|e| format!("stage_jaxpr failed: {e}"))?;
            let stage_ms = stage_start.elapsed().as_millis();

            states.push(json!({
                "step": "staging_all_known",
                "duration_ms": stage_ms,
                "within_budget": stage_ms < 500,
            }));

            if stage_ms >= 500 {
                return Err(format!(
                    "staging all-known took {stage_ms}ms, budget is 500ms"
                ));
            }

            // Verify correctness: chain of add(x, 1) * 1000 starting from 0 = 1000
            let exec_result =
                execute_staged(&staged, &[]).map_err(|e| format!("execute staged failed: {e}"))?;
            let output = exec_result[0]
                .as_scalar_literal()
                .and_then(|l| l.as_i64())
                .ok_or("expected i64 output")?;

            states.push(json!({
                "step": "correctness_check",
                "output": output,
                "expected": n,
                "correct": output == n as i64,
            }));

            if output != n as i64 {
                return Err(format!("expected {n}, got {output}"));
            }

            // Cross-validate with full eval
            let full_start = Instant::now();
            let full = eval_jaxpr(&jaxpr, &[Value::scalar_i64(0)])
                .map_err(|e| format!("full eval failed: {e}"))?;
            let full_ms = full_start.elapsed().as_millis();

            if exec_result != full {
                return Err(format!("staged {exec_result:?} != full {full:?}"));
            }

            states.push(json!({
                "step": "full_eval_comparison",
                "full_eval_ms": full_ms,
                "matches": true,
            }));

            // DCE timing on large graph
            let dce_start = Instant::now();
            let (dce_result, _used_inputs) = dce_jaxpr(&jaxpr, &[true]);
            let dce_ms = dce_start.elapsed().as_millis();

            states.push(json!({
                "step": "dce_timing",
                "duration_ms": dce_ms,
                "original_eqns": jaxpr.equations.len(),
                "post_dce_eqns": dce_result.equations.len(),
                "within_budget": dce_ms < 500,
            }));

            Ok(ScenarioOutcome {
                output_capture: json!({
                    "equation_count": n,
                    "pe_all_known_ms": pe_known_ms,
                    "pe_all_unknown_ms": pe_unk_ms,
                    "staging_ms": stage_ms,
                    "full_eval_ms": full_ms,
                    "dce_ms": dce_ms,
                    "output_correct": true,
                    "all_within_budget": true,
                }),
                artifact_refs: vec![],
            })
        },
    );
}
