#![forbid(unsafe_code)]

//! Partial evaluation: split a Jaxpr into known and unknown sub-Jaxprs.
//!
//! Given a Jaxpr and a boolean mask indicating which inputs are known (concrete)
//! vs unknown (abstract), partial evaluation produces:
//! - `jaxpr_known`: equations whose inputs are all derivable from known values
//! - `jaxpr_unknown`: residual equations that depend on unknown inputs
//! - `residuals`: intermediate values produced by jaxpr_known and consumed by jaxpr_unknown
//!
//! Invariant: eval(jaxpr_known, known_inputs) ++ eval(jaxpr_unknown, residuals ++ unknown_inputs)
//!            == eval(original_jaxpr, all_inputs)

use fj_core::{AbstractValue, Atom, DType, Equation, Jaxpr, Shape, Value, VarId};

/// Classification of a value during partial evaluation.
#[derive(Debug, Clone)]
pub enum PartialVal {
    /// Value is concretely known at trace time.
    Known(Value),
    /// Value is abstract (unknown) — only its type signature is available.
    Unknown(AbstractValue),
}

impl PartialVal {
    /// Returns `true` if this value is concretely known.
    pub fn is_known(&self) -> bool {
        matches!(self, PartialVal::Known(_))
    }

    /// Returns the known value, if available.
    pub fn get_known(&self) -> Option<&Value> {
        match self {
            PartialVal::Known(v) => Some(v),
            PartialVal::Unknown(_) => None,
        }
    }

    /// Returns the abstract value (type signature) regardless of known/unknown status.
    pub fn get_aval(&self) -> AbstractValue {
        match self {
            PartialVal::Known(v) => abstract_value_of(v),
            PartialVal::Unknown(aval) => aval.clone(),
        }
    }
}

/// Result of partial evaluation on a Jaxpr.
#[derive(Debug, Clone)]
pub struct PartialEvalResult {
    /// Jaxpr containing only equations with all-known inputs.
    /// Outputs = original known outputs ++ residual values.
    pub jaxpr_known: Jaxpr,

    /// Constants for jaxpr_known's constvars (values hoisted from known inputs).
    pub known_consts: Vec<Value>,

    /// Jaxpr containing equations that depend on unknown inputs.
    /// Inputs = residuals (from jaxpr_known) ++ original unknown inputs.
    pub jaxpr_unknown: Jaxpr,

    /// Which of the original Jaxpr's outputs are unknown.
    pub out_unknowns: Vec<bool>,

    /// Abstract values of residual intermediate values passed between the two jaxprs.
    pub residual_avals: Vec<AbstractValue>,
}

/// Errors that can occur during partial evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartialEvalError {
    /// Input mask length doesn't match Jaxpr input count.
    InputMaskMismatch { expected: usize, actual: usize },
    /// A variable referenced in an equation was not defined.
    UndefinedVariable(VarId),
    /// Residual type mismatch between known outputs and unknown inputs.
    ResidualTypeMismatch { index: usize },
}

impl std::fmt::Display for PartialEvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputMaskMismatch { expected, actual } => {
                write!(
                    f,
                    "input mask length mismatch: jaxpr has {} inputs, mask has {} entries",
                    expected, actual
                )
            }
            Self::UndefinedVariable(var) => write!(f, "undefined variable v{}", var.0),
            Self::ResidualTypeMismatch { index } => {
                write!(f, "residual type mismatch at index {}", index)
            }
        }
    }
}

impl std::error::Error for PartialEvalError {}

/// Partially evaluate a Jaxpr given a mask of which inputs are unknown.
///
/// This is the core partial evaluation routine. It walks the equations in order,
/// classifying each as "known" (all inputs known) or "unknown" (any input unknown),
/// and produces two sub-Jaxprs plus residual metadata.
///
/// # Arguments
/// * `jaxpr` - The Jaxpr to partially evaluate.
/// * `unknowns` - Boolean mask: `true` means the corresponding input is unknown.
///
/// # Returns
/// A `PartialEvalResult` containing the known and unknown sub-Jaxprs.
pub fn partial_eval_jaxpr(
    jaxpr: &Jaxpr,
    unknowns: &[bool],
) -> Result<PartialEvalResult, PartialEvalError> {
    if unknowns.len() != jaxpr.invars.len() {
        return Err(PartialEvalError::InputMaskMismatch {
            expected: jaxpr.invars.len(),
            actual: unknowns.len(),
        });
    }

    let any_unknown = unknowns.iter().any(|u| *u);

    // Fast path: all-known — everything goes to known jaxpr, no residuals.
    if !any_unknown {
        let out_unknowns = vec![false; jaxpr.outvars.len()];
        return Ok(PartialEvalResult {
            jaxpr_known: Jaxpr::new(
                jaxpr.invars.clone(),
                jaxpr.constvars.clone(),
                jaxpr.outvars.clone(),
                jaxpr.equations.clone(),
            ),
            known_consts: vec![],
            jaxpr_unknown: Jaxpr::new(vec![], vec![], vec![], vec![]),
            out_unknowns,
            residual_avals: vec![],
        });
    }

    // Fast path: all-unknown — everything goes to unknown jaxpr, no residuals.
    let all_unknown = unknowns.iter().all(|u| *u);
    if all_unknown {
        let out_unknowns = vec![true; jaxpr.outvars.len()];
        return Ok(PartialEvalResult {
            jaxpr_known: Jaxpr::new(vec![], jaxpr.constvars.clone(), vec![], vec![]),
            known_consts: vec![],
            jaxpr_unknown: Jaxpr::new(
                jaxpr.invars.clone(),
                vec![],
                jaxpr.outvars.clone(),
                jaxpr.equations.clone(),
            ),
            out_unknowns,
            residual_avals: vec![],
        });
    }

    // Mixed case: use VarId-indexed bitset for O(1) lookups.
    let max_var_id = max_var_in_jaxpr(jaxpr);
    let bitset_len = max_var_id + 1;

    let mut is_unknown_var = vec![false; bitset_len];
    for (var, &is_unk) in jaxpr.invars.iter().zip(unknowns.iter()) {
        if is_unk {
            is_unknown_var[var.0 as usize] = true;
        }
    }

    // Classify equations and collect residuals.
    let n_eqns = jaxpr.equations.len();
    let mut known_eqns: Vec<Equation> = Vec::with_capacity(n_eqns);
    let mut unknown_eqns: Vec<Equation> = Vec::with_capacity(n_eqns);
    let mut residual_vars: Vec<VarId> = Vec::new();
    let mut is_residual = vec![false; bitset_len];

    for eqn in &jaxpr.equations {
        let any_input_unknown = eqn.inputs.iter().any(|atom| match atom {
            Atom::Var(v) => is_unknown_var[v.0 as usize],
            Atom::Lit(_) => false,
        });

        if any_input_unknown {
            for out_var in &eqn.outputs {
                is_unknown_var[out_var.0 as usize] = true;
            }
            unknown_eqns.push(eqn.clone());

            for atom in &eqn.inputs {
                if let Atom::Var(v) = atom {
                    let idx = v.0 as usize;
                    if !is_unknown_var[idx] && !is_residual[idx] {
                        residual_vars.push(*v);
                        is_residual[idx] = true;
                    }
                }
            }
        } else {
            known_eqns.push(eqn.clone());
        }
    }

    // Second pass: identify known-equation outputs consumed by unknown equations.
    let mut is_unknown_input = vec![false; bitset_len];
    for eqn in &unknown_eqns {
        for atom in &eqn.inputs {
            if let Atom::Var(v) = atom {
                is_unknown_input[v.0 as usize] = true;
            }
        }
    }

    for eqn in &known_eqns {
        for out_var in &eqn.outputs {
            let idx = out_var.0 as usize;
            if is_unknown_input[idx] && !is_residual[idx] {
                residual_vars.push(*out_var);
                is_residual[idx] = true;
            }
        }
    }

    for (var, &is_unk) in jaxpr.invars.iter().zip(unknowns.iter()) {
        let idx = var.0 as usize;
        if !is_unk && is_unknown_input[idx] && !is_residual[idx] {
            residual_vars.push(*var);
            is_residual[idx] = true;
        }
    }

    for var in &jaxpr.constvars {
        let idx = var.0 as usize;
        if is_unknown_input[idx] && !is_residual[idx] {
            residual_vars.push(*var);
            is_residual[idx] = true;
        }
    }

    // Build known Jaxpr.
    let known_invars: Vec<VarId> = jaxpr
        .invars
        .iter()
        .zip(unknowns.iter())
        .filter(|(_, is_unknown)| !**is_unknown)
        .map(|(v, _)| *v)
        .collect();

    let known_outvars: Vec<VarId> = {
        let mut outs: Vec<VarId> = jaxpr
            .outvars
            .iter()
            .filter(|v| !is_unknown_var[v.0 as usize])
            .copied()
            .collect();
        // Append residual outputs.
        outs.extend(residual_vars.iter().copied());
        outs
    };

    let jaxpr_known = Jaxpr::new(
        known_invars,
        jaxpr.constvars.clone(),
        known_outvars,
        known_eqns,
    );

    // Build unknown Jaxpr.
    // Inputs: residual vars ++ original unknown inputs.
    // We keep original VarIds so equations can reference them without remapping.
    let mut unknown_invars: Vec<VarId> = Vec::new();

    // Residual inputs come first (same VarIds as in original jaxpr).
    for &res_var in &residual_vars {
        unknown_invars.push(res_var);
    }

    // Then original unknown inputs.
    for (var, &is_unknown) in jaxpr.invars.iter().zip(unknowns.iter()) {
        if is_unknown {
            unknown_invars.push(*var);
        }
    }

    let unknown_outvars: Vec<VarId> = jaxpr
        .outvars
        .iter()
        .filter(|v| is_unknown_var[v.0 as usize])
        .copied()
        .collect();

    let jaxpr_unknown = Jaxpr::new(unknown_invars, vec![], unknown_outvars, unknown_eqns);

    // Determine which original outputs are unknown.
    let out_unknowns: Vec<bool> = jaxpr
        .outvars
        .iter()
        .map(|v| is_unknown_var[v.0 as usize])
        .collect();

    // Compute residual abstract values.
    let residual_avals: Vec<AbstractValue> = residual_vars
        .iter()
        .map(|_| AbstractValue {
            dtype: DType::F64,
            shape: Shape::scalar(),
        })
        .collect();

    Ok(PartialEvalResult {
        jaxpr_known,
        known_consts: vec![],
        jaxpr_unknown,
        out_unknowns,
        residual_avals,
    })
}

/// Dead code elimination on a Jaxpr.
///
/// Given a Jaxpr and a mask of which outputs are used, removes equations
/// that don't contribute to any used output. Preserves equation ordering.
///
/// Returns the pruned Jaxpr and a mask of which inputs are still needed.
pub fn dce_jaxpr(jaxpr: &Jaxpr, used_outputs: &[bool]) -> (Jaxpr, Vec<bool>) {
    let max_var = max_var_in_jaxpr(jaxpr);
    let bitset_len = max_var + 1;

    // Backward pass: determine which variables are needed via indexed bitset.
    let mut needed = vec![false; bitset_len];

    for (var, &used) in jaxpr.outvars.iter().zip(used_outputs.iter()) {
        if used {
            needed[var.0 as usize] = true;
        }
    }

    // Walk equations in reverse, marking inputs of needed equations.
    let mut keep_eqn = vec![false; jaxpr.equations.len()];
    for (i, eqn) in jaxpr.equations.iter().enumerate().rev() {
        let outputs_needed = eqn.outputs.iter().any(|v| needed[v.0 as usize]);
        if outputs_needed {
            keep_eqn[i] = true;
            for atom in &eqn.inputs {
                if let Atom::Var(v) = atom {
                    needed[v.0 as usize] = true;
                }
            }
        }
    }

    let retained_eqns: Vec<Equation> = jaxpr
        .equations
        .iter()
        .zip(keep_eqn.iter())
        .filter(|(_, keep)| **keep)
        .map(|(eqn, _)| eqn.clone())
        .collect();

    let used_inputs: Vec<bool> = jaxpr.invars.iter().map(|v| needed[v.0 as usize]).collect();

    let retained_outvars: Vec<VarId> = jaxpr
        .outvars
        .iter()
        .zip(used_outputs.iter())
        .filter(|(_, &used)| used)
        .map(|(v, _)| *v)
        .collect();

    let new_jaxpr = Jaxpr::new(
        jaxpr.invars.clone(),
        jaxpr.constvars.clone(),
        retained_outvars,
        retained_eqns,
    );

    (new_jaxpr, used_inputs)
}

/// Compute the maximum VarId index in a Jaxpr (for bitset sizing).
fn max_var_in_jaxpr(jaxpr: &Jaxpr) -> usize {
    let mut max_id: u32 = 0;
    for v in &jaxpr.invars {
        max_id = max_id.max(v.0);
    }
    for v in &jaxpr.constvars {
        max_id = max_id.max(v.0);
    }
    for v in &jaxpr.outvars {
        max_id = max_id.max(v.0);
    }
    for eqn in &jaxpr.equations {
        for atom in &eqn.inputs {
            if let Atom::Var(v) = atom {
                max_id = max_id.max(v.0);
            }
        }
        for v in &eqn.outputs {
            max_id = max_id.max(v.0);
        }
    }
    max_id as usize
}

/// Extract an abstract value (dtype + shape) from a concrete Value.
fn abstract_value_of(value: &Value) -> AbstractValue {
    match value {
        Value::Scalar(lit) => {
            let dtype = match lit {
                fj_core::Literal::I64(_) => DType::I64,
                fj_core::Literal::Bool(_) => DType::Bool,
                fj_core::Literal::F64Bits(_) => DType::F64,
            };
            AbstractValue {
                dtype,
                shape: Shape::scalar(),
            }
        }
        Value::Tensor(t) => AbstractValue {
            dtype: t.dtype,
            shape: t.shape.clone(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fj_core::{Equation, Jaxpr, Literal, Primitive, ProgramSpec, VarId, build_program};
    use serde::Serialize;
    use smallvec::smallvec;
    use std::any::Any;
    use std::collections::BTreeMap;
    use std::fs;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    const PACKET_ID: &str = "FJ-P2C-003";
    const SUITE_ID: &str = "fj-interpreters";

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn test_log_path(test_id: &str) -> PathBuf {
        let file_name = test_id.replace("::", "__");
        repo_root()
            .join("artifacts")
            .join("testing")
            .join("logs")
            .join(SUITE_ID)
            .join(format!("{file_name}.json"))
    }

    fn replay_command(test_id: &str) -> String {
        format!("cargo test -p fj-interpreters --lib {test_id} -- --exact --nocapture")
    }

    fn duration_ms(start: Instant) -> u64 {
        u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    fn write_log(path: &Path, log: &fj_test_utils::TestLogV1) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| format!("log dir create failed: {err}"))?;
        }
        let payload = serde_json::to_string_pretty(log)
            .map_err(|err| format!("log serialize failed: {err}"))?;
        fs::write(path, payload).map_err(|err| format!("log write failed: {err}"))
    }

    fn panic_payload_to_string(payload: &(dyn Any + Send)) -> String {
        if let Some(msg) = payload.downcast_ref::<String>() {
            return msg.clone();
        }
        if let Some(msg) = payload.downcast_ref::<&str>() {
            return (*msg).to_owned();
        }
        "non-string panic payload".to_owned()
    }

    fn run_logged_test<Fixture, F>(
        test_name: &str,
        fixture: &Fixture,
        mode: fj_test_utils::TestMode,
        body: F,
    ) where
        Fixture: Serialize,
        F: FnOnce() -> Result<Vec<String>, String> + std::panic::UnwindSafe,
    {
        let overall_start = Instant::now();
        let setup_start = Instant::now();
        let fixture_id = fj_test_utils::fixture_id_from_json(fixture).expect("fixture digest");
        let test_id = fj_test_utils::test_id(module_path!(), test_name);
        let mut log = fj_test_utils::TestLogV1::unit(
            test_id.clone(),
            fixture_id,
            mode,
            fj_test_utils::TestResult::Fail,
        );
        log.phase_timings.setup_ms = duration_ms(setup_start);

        let execute_start = Instant::now();
        let outcome = catch_unwind(AssertUnwindSafe(body));
        log.phase_timings.execute_ms = duration_ms(execute_start);

        let verify_start = Instant::now();
        let mut panic_payload: Option<Box<dyn Any + Send>> = None;
        let mut failure_detail: Option<String> = None;

        match outcome {
            Ok(Ok(mut artifact_refs)) => {
                log.result = fj_test_utils::TestResult::Pass;
                artifact_refs.push(format!("packet:{PACKET_ID}"));
                artifact_refs.push(format!("replay: {}", replay_command(&test_id)));
                log.artifact_refs = artifact_refs;
                log.details = Some(format!(
                    "packet_id={PACKET_ID};suite_id={SUITE_ID};result=pass"
                ));
            }
            Ok(Err(detail)) => {
                failure_detail = Some(detail.clone());
                log.result = fj_test_utils::TestResult::Fail;
                log.artifact_refs = vec![
                    format!("packet:{PACKET_ID}"),
                    format!("replay: {}", replay_command(&test_id)),
                ];
                log.details = Some(detail);
            }
            Err(payload) => {
                let detail = panic_payload_to_string(payload.as_ref());
                failure_detail = Some(detail.clone());
                log.result = fj_test_utils::TestResult::Fail;
                log.artifact_refs = vec![
                    format!("packet:{PACKET_ID}"),
                    format!("replay: {}", replay_command(&test_id)),
                ];
                log.details = Some(detail);
                panic_payload = Some(payload);
            }
        }
        log.phase_timings.verify_ms = duration_ms(verify_start);

        let log_path = test_log_path(&test_id);
        log.artifact_refs.push(log_path.display().to_string());
        log.duration_ms = duration_ms(overall_start);

        let teardown_start = Instant::now();
        write_log(&log_path, &log).expect("test log write should succeed");
        log.phase_timings.teardown_ms = duration_ms(teardown_start);
        log.duration_ms = duration_ms(overall_start);
        write_log(&log_path, &log).expect("test log rewrite should succeed");

        if let Some(payload) = panic_payload {
            std::panic::resume_unwind(payload);
        }
        if let Some(detail) = failure_detail {
            panic!("{detail}");
        }
    }

    // ── Fixture Jaxpr builders ──────────────────────────────────────

    /// { a, b -> c = add(a, b); d = mul(c, b) -> d }
    fn make_add_chain_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
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

    /// { a -> b = neg(a); c = abs(b); d = sin(c) -> d }
    fn make_deep_chain_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                },
                Equation {
                    primitive: Primitive::Abs,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                },
                Equation {
                    primitive: Primitive::Sin,
                    inputs: smallvec![Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                },
            ],
        )
    }

    /// { a, b -> c = add(a, b); d = neg(a) -> c, d } (two independent outputs)
    fn make_multi_output_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3), VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                },
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                },
            ],
        )
    }

    /// { a -> b = add(a, lit(1)) -> b } (equation with literal input)
    fn make_literal_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        )
    }

    /// Empty Jaxpr: { a -> a } (identity, no equations)
    fn make_empty_jaxpr() -> Jaxpr {
        Jaxpr::new(vec![VarId(1)], vec![], vec![VarId(1)], vec![])
    }

    /// Single-equation Jaxpr: { a -> b = neg(a) -> b }
    fn make_single_eqn_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Neg,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        )
    }

    /// { a, b, c -> d = add(a, b); e = mul(d, c); f = neg(a) -> e, f }
    fn make_three_input_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1), VarId(2), VarId(3)],
            vec![],
            vec![VarId(5), VarId(6)],
            vec![
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(4)), Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(5)],
                    params: BTreeMap::new(),
                },
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(6)],
                    params: BTreeMap::new(),
                },
            ],
        )
    }

    // ══════════════════════════════════════════════════════════════════
    // Category 1: Constant Folding (all-known → fully evaluated)
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_pe_constant_fold_add_chain() {
        run_logged_test(
            "test_pe_constant_fold_add_chain",
            &("pe", "const_fold", "add_chain"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_add_chain_jaxpr();
                let result = partial_eval_jaxpr(&jaxpr, &[false, false]).unwrap();
                assert_eq!(result.jaxpr_known.equations.len(), 2);
                assert_eq!(result.jaxpr_unknown.equations.len(), 0);
                assert_eq!(result.out_unknowns, vec![false]);
                assert!(result.residual_avals.is_empty());
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_constant_fold_single_eqn() {
        run_logged_test(
            "test_pe_constant_fold_single_eqn",
            &("pe", "const_fold", "single"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_single_eqn_jaxpr();
                let result = partial_eval_jaxpr(&jaxpr, &[false]).unwrap();
                assert_eq!(result.jaxpr_known.equations.len(), 1);
                assert_eq!(result.jaxpr_unknown.equations.len(), 0);
                assert_eq!(result.out_unknowns, vec![false]);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_constant_fold_deep_chain() {
        run_logged_test(
            "test_pe_constant_fold_deep_chain",
            &("pe", "const_fold", "deep"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_deep_chain_jaxpr();
                let result = partial_eval_jaxpr(&jaxpr, &[false]).unwrap();
                assert_eq!(result.jaxpr_known.equations.len(), 3);
                assert_eq!(result.jaxpr_unknown.equations.len(), 0);
                assert_eq!(result.out_unknowns, vec![false]);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_constant_fold_multi_output() {
        run_logged_test(
            "test_pe_constant_fold_multi_output",
            &("pe", "const_fold", "multi_out"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_multi_output_jaxpr();
                let result = partial_eval_jaxpr(&jaxpr, &[false, false]).unwrap();
                assert_eq!(result.jaxpr_known.equations.len(), 2);
                assert_eq!(result.jaxpr_unknown.equations.len(), 0);
                assert_eq!(result.out_unknowns, vec![false, false]);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_constant_fold_with_literal() {
        run_logged_test(
            "test_pe_constant_fold_with_literal",
            &("pe", "const_fold", "literal"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_literal_jaxpr();
                let result = partial_eval_jaxpr(&jaxpr, &[false]).unwrap();
                assert_eq!(result.jaxpr_known.equations.len(), 1);
                assert_eq!(result.jaxpr_unknown.equations.len(), 0);
                assert_eq!(result.out_unknowns, vec![false]);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_constant_fold_program_spec_add2() {
        run_logged_test(
            "test_pe_constant_fold_program_spec_add2",
            &("pe", "const_fold", "program_spec_add2"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = build_program(ProgramSpec::Add2);
                let result = partial_eval_jaxpr(&jaxpr, &[false, false]).unwrap();
                assert_eq!(result.jaxpr_known.equations.len(), jaxpr.equations.len());
                assert_eq!(result.jaxpr_unknown.equations.len(), 0);
                assert!(result.out_unknowns.iter().all(|u| !u));
                Ok(vec![])
            },
        );
    }

    // ══════════════════════════════════════════════════════════════════
    // Category 2: Full Residual (all-unknown → residual matches original)
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_pe_full_residual_add_chain() {
        run_logged_test(
            "test_pe_full_residual_add_chain",
            &("pe", "full_residual", "add_chain"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_add_chain_jaxpr();
                let result = partial_eval_jaxpr(&jaxpr, &[true, true]).unwrap();
                assert_eq!(result.jaxpr_known.equations.len(), 0);
                assert_eq!(result.jaxpr_unknown.equations.len(), jaxpr.equations.len());
                assert!(result.out_unknowns.iter().all(|u| *u));
                assert!(result.residual_avals.is_empty());
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_full_residual_deep_chain() {
        run_logged_test(
            "test_pe_full_residual_deep_chain",
            &("pe", "full_residual", "deep_chain"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_deep_chain_jaxpr();
                let result = partial_eval_jaxpr(&jaxpr, &[true]).unwrap();
                assert_eq!(result.jaxpr_known.equations.len(), 0);
                assert_eq!(result.jaxpr_unknown.equations.len(), 3);
                assert_eq!(result.out_unknowns, vec![true]);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_full_residual_preserves_primitives() {
        run_logged_test(
            "test_pe_full_residual_preserves_primitives",
            &("pe", "full_residual", "primitives"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_add_chain_jaxpr();
                let result = partial_eval_jaxpr(&jaxpr, &[true, true]).unwrap();
                let orig_prims: Vec<_> = jaxpr.equations.iter().map(|e| e.primitive).collect();
                let residual_prims: Vec<_> = result
                    .jaxpr_unknown
                    .equations
                    .iter()
                    .map(|e| e.primitive)
                    .collect();
                assert_eq!(orig_prims, residual_prims);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_full_residual_three_inputs() {
        run_logged_test(
            "test_pe_full_residual_three_inputs",
            &("pe", "full_residual", "three_inputs"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_three_input_jaxpr();
                let result = partial_eval_jaxpr(&jaxpr, &[true, true, true]).unwrap();
                assert_eq!(result.jaxpr_known.equations.len(), 0);
                assert_eq!(result.jaxpr_unknown.equations.len(), 3);
                assert!(result.out_unknowns.iter().all(|u| *u));
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_full_residual_program_specs() {
        run_logged_test(
            "test_pe_full_residual_program_specs",
            &("pe", "full_residual", "program_specs"),
            fj_test_utils::TestMode::Strict,
            || {
                let specs = [
                    (ProgramSpec::Add2, 2),
                    (ProgramSpec::Square, 1),
                    (ProgramSpec::SinX, 1),
                    (ProgramSpec::CosX, 1),
                    (ProgramSpec::AddOne, 1),
                ];
                for (spec, n_inputs) in specs {
                    let jaxpr = build_program(spec);
                    let unknowns = vec![true; n_inputs];
                    let result = partial_eval_jaxpr(&jaxpr, &unknowns).unwrap();
                    assert_eq!(
                        result.jaxpr_unknown.equations.len(),
                        jaxpr.equations.len(),
                        "full-residual mismatch for {:?}",
                        spec
                    );
                }
                Ok(vec![])
            },
        );
    }

    // ══════════════════════════════════════════════════════════════════
    // Category 3: Mixed (known constants + unknown runtime values)
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_pe_mixed_known_unknown_split() {
        run_logged_test(
            "test_pe_mixed_known_unknown_split",
            &("pe", "mixed", "split"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_neg_mul_jaxpr();
                let result = partial_eval_jaxpr(&jaxpr, &[false, true]).unwrap();
                // neg(a) is known; mul(neg(a), b) is unknown
                assert_eq!(result.jaxpr_known.equations.len(), 1);
                assert_eq!(result.jaxpr_unknown.equations.len(), 1);
                assert_eq!(result.jaxpr_known.equations[0].primitive, Primitive::Neg);
                assert_eq!(result.jaxpr_unknown.equations[0].primitive, Primitive::Mul);
                assert!(!result.residual_avals.is_empty());
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_mixed_three_inputs_partial() {
        run_logged_test(
            "test_pe_mixed_three_inputs_partial",
            &("pe", "mixed", "three_inputs"),
            fj_test_utils::TestMode::Strict,
            || {
                // a known, b known, c unknown
                let jaxpr = make_three_input_jaxpr();
                let result = partial_eval_jaxpr(&jaxpr, &[false, false, true]).unwrap();
                // add(a,b) is known; mul(d,c) is unknown; neg(a) is known
                assert_eq!(result.jaxpr_known.equations.len(), 2); // add + neg
                assert_eq!(result.jaxpr_unknown.equations.len(), 1); // mul
                // First output (mul result) unknown, second (neg) known
                assert_eq!(result.out_unknowns, vec![true, false]);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_mixed_preserves_equation_order() {
        run_logged_test(
            "test_pe_mixed_preserves_equation_order",
            &("pe", "mixed", "order"),
            fj_test_utils::TestMode::Strict,
            || {
                // Verify topological ordering in both sub-jaxprs
                let jaxpr = make_neg_mul_jaxpr();
                let result = partial_eval_jaxpr(&jaxpr, &[false, true]).unwrap();
                // Known jaxpr: neg comes before any dependent
                assert_eq!(result.jaxpr_known.equations[0].primitive, Primitive::Neg);
                // Unknown jaxpr: mul is the only equation
                assert_eq!(result.jaxpr_unknown.equations[0].primitive, Primitive::Mul);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_mixed_residual_count_matches() {
        run_logged_test(
            "test_pe_mixed_residual_count_matches",
            &("pe", "mixed", "residual_count"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_neg_mul_jaxpr();
                let result = partial_eval_jaxpr(&jaxpr, &[false, true]).unwrap();
                // Residual vars in known outputs should match residual_avals count
                let known_out_count = result.jaxpr_known.outvars.len();
                let known_direct_outs: usize = result.out_unknowns.iter().filter(|u| !**u).count();
                let residual_count = known_out_count - known_direct_outs;
                assert_eq!(residual_count, result.residual_avals.len());
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_mixed_literal_inputs_always_known() {
        run_logged_test(
            "test_pe_mixed_literal_inputs_always_known",
            &("pe", "mixed", "literal"),
            fj_test_utils::TestMode::Strict,
            || {
                // { a -> b = add(a, lit(1)) -> b }; a unknown
                let jaxpr = make_literal_jaxpr();
                let result = partial_eval_jaxpr(&jaxpr, &[true]).unwrap();
                // Even though literal is known, the equation has an unknown var input
                assert_eq!(result.jaxpr_unknown.equations.len(), 1);
                assert_eq!(result.out_unknowns, vec![true]);
                Ok(vec![])
            },
        );
    }

    // ══════════════════════════════════════════════════════════════════
    // Category 4: Staging pipeline (PE + eval equivalence)
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_pe_staging_equivalence_neg_mul() {
        run_logged_test(
            "test_pe_staging_equivalence_neg_mul",
            &("pe", "staging", "neg_mul"),
            fj_test_utils::TestMode::Strict,
            || {
                use crate::{eval_jaxpr, eval_jaxpr_with_consts};

                let jaxpr = make_neg_mul_jaxpr();
                let a = Value::scalar_i64(5);
                let b = Value::scalar_i64(3);

                // Full eval: neg(5) * 3 = -5 * 3 = -15
                let full_result = eval_jaxpr(&jaxpr, &[a.clone(), b.clone()]).unwrap();

                // Staged: PE with a=known, b=unknown
                let pe = partial_eval_jaxpr(&jaxpr, &[false, true]).unwrap();
                let known_outs =
                    eval_jaxpr_with_consts(&pe.jaxpr_known, &pe.known_consts, &[a]).unwrap();
                // known_outs should contain residuals; feed them + unknown into jaxpr_unknown
                let mut unknown_inputs = known_outs;
                unknown_inputs.push(b);
                let staged_result = eval_jaxpr(&pe.jaxpr_unknown, &unknown_inputs).unwrap();

                assert_eq!(full_result, staged_result);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_staging_equivalence_all_known() {
        run_logged_test(
            "test_pe_staging_equivalence_all_known",
            &("pe", "staging", "all_known"),
            fj_test_utils::TestMode::Strict,
            || {
                use crate::eval_jaxpr;

                let jaxpr = build_program(ProgramSpec::Add2);
                let args = [Value::scalar_i64(10), Value::scalar_i64(20)];
                let full_result = eval_jaxpr(&jaxpr, &args).unwrap();

                let pe = partial_eval_jaxpr(&jaxpr, &[false, false]).unwrap();
                // All known: jaxpr_known has all equations, jaxpr_unknown is empty
                let known_result = eval_jaxpr(&pe.jaxpr_known, &args).unwrap();
                // known_result includes original outputs (no residuals for all-known)
                // The first outputs correspond to the original non-unknown outvars
                assert_eq!(full_result[0], known_result[0]);
                Ok(vec![])
            },
        );
    }

    // ══════════════════════════════════════════════════════════════════
    // Category 5: JIT integration (via staging module)
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_pe_staging_module_roundtrip() {
        run_logged_test(
            "test_pe_staging_module_roundtrip",
            &("pe", "jit", "roundtrip"),
            fj_test_utils::TestMode::Strict,
            || {
                use crate::staging::{execute_staged, stage_jaxpr};

                let jaxpr = make_neg_mul_jaxpr();
                let staged = stage_jaxpr(&jaxpr, &[false, true], &[Value::scalar_i64(5)]).unwrap();
                let result = execute_staged(&staged, &[Value::scalar_i64(3)]).unwrap();

                // neg(5) * 3 = -15
                let full = crate::eval_jaxpr(&jaxpr, &[Value::scalar_i64(5), Value::scalar_i64(3)])
                    .unwrap();
                assert_eq!(result, full);
                Ok(vec![])
            },
        );
    }

    // ══════════════════════════════════════════════════════════════════
    // Category 6: Edge cases
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_pe_edge_empty_jaxpr_known() {
        run_logged_test(
            "test_pe_edge_empty_jaxpr_known",
            &("pe", "edge", "empty_known"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_empty_jaxpr();
                let result = partial_eval_jaxpr(&jaxpr, &[false]).unwrap();
                assert_eq!(result.jaxpr_known.equations.len(), 0);
                assert_eq!(result.jaxpr_unknown.equations.len(), 0);
                assert_eq!(result.out_unknowns, vec![false]);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_edge_empty_jaxpr_unknown() {
        run_logged_test(
            "test_pe_edge_empty_jaxpr_unknown",
            &("pe", "edge", "empty_unknown"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_empty_jaxpr();
                let result = partial_eval_jaxpr(&jaxpr, &[true]).unwrap();
                assert_eq!(result.jaxpr_known.equations.len(), 0);
                assert_eq!(result.jaxpr_unknown.equations.len(), 0);
                assert_eq!(result.out_unknowns, vec![true]);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_edge_single_equation_known() {
        run_logged_test(
            "test_pe_edge_single_equation_known",
            &("pe", "edge", "single_known"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_single_eqn_jaxpr();
                let result = partial_eval_jaxpr(&jaxpr, &[false]).unwrap();
                assert_eq!(result.jaxpr_known.equations.len(), 1);
                assert_eq!(result.jaxpr_unknown.equations.len(), 0);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_edge_single_equation_unknown() {
        run_logged_test(
            "test_pe_edge_single_equation_unknown",
            &("pe", "edge", "single_unknown"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_single_eqn_jaxpr();
                let result = partial_eval_jaxpr(&jaxpr, &[true]).unwrap();
                assert_eq!(result.jaxpr_known.equations.len(), 0);
                assert_eq!(result.jaxpr_unknown.equations.len(), 1);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_edge_mask_too_short() {
        run_logged_test(
            "test_pe_edge_mask_too_short",
            &("pe", "edge", "mask_short"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_add_chain_jaxpr();
                let err = partial_eval_jaxpr(&jaxpr, &[false]).unwrap_err();
                assert_eq!(
                    err,
                    PartialEvalError::InputMaskMismatch {
                        expected: 2,
                        actual: 1,
                    }
                );
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_edge_mask_too_long() {
        run_logged_test(
            "test_pe_edge_mask_too_long",
            &("pe", "edge", "mask_long"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_add_chain_jaxpr();
                let err = partial_eval_jaxpr(&jaxpr, &[false, true, false]).unwrap_err();
                assert_eq!(
                    err,
                    PartialEvalError::InputMaskMismatch {
                        expected: 2,
                        actual: 3,
                    }
                );
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_edge_empty_mask() {
        run_logged_test(
            "test_pe_edge_empty_mask",
            &("pe", "edge", "empty_mask"),
            fj_test_utils::TestMode::Strict,
            || {
                // Jaxpr with no inputs, no equations — just constvars
                let jaxpr = Jaxpr::new(vec![], vec![], vec![], vec![]);
                let result = partial_eval_jaxpr(&jaxpr, &[]).unwrap();
                assert_eq!(result.jaxpr_known.equations.len(), 0);
                assert_eq!(result.jaxpr_unknown.equations.len(), 0);
                Ok(vec![])
            },
        );
    }

    // ── DCE edge cases ──────────────────────────────────────────────

    #[test]
    fn test_dce_removes_unused_equations() {
        run_logged_test(
            "test_dce_removes_unused_equations",
            &("dce", "basic", "removes_unused"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = Jaxpr::new(
                    vec![VarId(1)],
                    vec![],
                    vec![VarId(2), VarId(3)],
                    vec![
                        Equation {
                            primitive: Primitive::Neg,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(2)],
                            params: BTreeMap::new(),
                        },
                        Equation {
                            primitive: Primitive::Abs,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(3)],
                            params: BTreeMap::new(),
                        },
                    ],
                );
                let (pruned, used_inputs) = dce_jaxpr(&jaxpr, &[true, false]);
                assert_eq!(pruned.equations.len(), 1);
                assert_eq!(pruned.equations[0].primitive, Primitive::Neg);
                assert_eq!(used_inputs, vec![true]);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_dce_keeps_chain_dependencies() {
        run_logged_test(
            "test_dce_keeps_chain_dependencies",
            &("dce", "basic", "keeps_chain"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_add_chain_jaxpr();
                let (pruned, used_inputs) = dce_jaxpr(&jaxpr, &[true]);
                assert_eq!(pruned.equations.len(), 2);
                assert_eq!(used_inputs, vec![true, true]);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_dce_all_unused_removes_everything() {
        run_logged_test(
            "test_dce_all_unused_removes_everything",
            &("dce", "edge", "all_unused"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_multi_output_jaxpr();
                let (pruned, used_inputs) = dce_jaxpr(&jaxpr, &[false, false]);
                assert_eq!(pruned.equations.len(), 0);
                assert_eq!(used_inputs, vec![false, false]);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_dce_all_used_keeps_everything() {
        run_logged_test(
            "test_dce_all_used_keeps_everything",
            &("dce", "edge", "all_used"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_multi_output_jaxpr();
                let (pruned, used_inputs) = dce_jaxpr(&jaxpr, &[true, true]);
                assert_eq!(pruned.equations.len(), 2);
                assert_eq!(used_inputs, vec![true, true]);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_dce_empty_jaxpr() {
        run_logged_test(
            "test_dce_empty_jaxpr",
            &("dce", "edge", "empty"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_empty_jaxpr();
                let (pruned, used_inputs) = dce_jaxpr(&jaxpr, &[true]);
                assert_eq!(pruned.equations.len(), 0);
                assert_eq!(used_inputs, vec![true]);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_dce_selective_multi_output() {
        run_logged_test(
            "test_dce_selective_multi_output",
            &("dce", "basic", "selective"),
            fj_test_utils::TestMode::Strict,
            || {
                // Only second output used: neg(a) needed but not add(a,b)
                let jaxpr = make_multi_output_jaxpr();
                let (pruned, used_inputs) = dce_jaxpr(&jaxpr, &[false, true]);
                assert_eq!(pruned.equations.len(), 1);
                assert_eq!(pruned.equations[0].primitive, Primitive::Neg);
                // Only first input (a) needed for neg; b not needed
                assert_eq!(used_inputs, vec![true, false]);
                Ok(vec![])
            },
        );
    }

    // ── PartialVal unit tests ───────────────────────────────────────

    #[test]
    fn test_partial_val_known_accessors() {
        run_logged_test(
            "test_partial_val_known_accessors",
            &("partial_val", "known"),
            fj_test_utils::TestMode::Strict,
            || {
                let pv = PartialVal::Known(Value::scalar_i64(42));
                assert!(pv.is_known());
                assert_eq!(pv.get_known(), Some(&Value::scalar_i64(42)));
                let aval = pv.get_aval();
                assert_eq!(aval.dtype, DType::I64);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_partial_val_unknown_accessors() {
        run_logged_test(
            "test_partial_val_unknown_accessors",
            &("partial_val", "unknown"),
            fj_test_utils::TestMode::Strict,
            || {
                let aval = AbstractValue {
                    dtype: DType::F64,
                    shape: Shape::scalar(),
                };
                let pv = PartialVal::Unknown(aval.clone());
                assert!(!pv.is_known());
                assert_eq!(pv.get_known(), None);
                assert_eq!(pv.get_aval(), aval);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_partial_val_abstract_value_of_types() {
        run_logged_test(
            "test_partial_val_abstract_value_of_types",
            &("partial_val", "aval_types"),
            fj_test_utils::TestMode::Strict,
            || {
                let pv_i64 = PartialVal::Known(Value::scalar_i64(1));
                assert_eq!(pv_i64.get_aval().dtype, DType::I64);

                let pv_f64 = PartialVal::Known(Value::scalar_f64(1.0));
                assert_eq!(pv_f64.get_aval().dtype, DType::F64);

                let pv_bool = PartialVal::Known(Value::scalar_bool(true));
                assert_eq!(pv_bool.get_aval().dtype, DType::Bool);
                Ok(vec![])
            },
        );
    }

    // ── Error Display coverage ──────────────────────────────────────

    #[test]
    fn test_pe_error_display_variants() {
        run_logged_test(
            "test_pe_error_display_variants",
            &("pe", "errors", "display"),
            fj_test_utils::TestMode::Strict,
            || {
                let e1 = PartialEvalError::InputMaskMismatch {
                    expected: 3,
                    actual: 1,
                };
                assert!(format!("{e1}").contains("3"));
                assert!(format!("{e1}").contains("1"));

                let e2 = PartialEvalError::UndefinedVariable(VarId(42));
                assert!(format!("{e2}").contains("42"));

                let e3 = PartialEvalError::ResidualTypeMismatch { index: 7 };
                assert!(format!("{e3}").contains("7"));

                // Error trait impl
                let _: &dyn std::error::Error = &e1;
                Ok(vec![])
            },
        );
    }

    // ── Schema contract test ────────────────────────────────────────

    #[test]
    fn test_pe_test_log_schema_contract() {
        let fixture_id =
            fj_test_utils::fixture_id_from_json(&("pe", "schema")).expect("fixture digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_pe_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }

    // ══════════════════════════════════════════════════════════════════
    // Category 7: Property Tests
    // ══════════════════════════════════════════════════════════════════

    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(proptest::test_runner::Config::with_cases(
                fj_test_utils::property_test_case_count()
            ))]

            /// PE equation conservation: known + unknown eqn counts == original.
            #[test]
            fn prop_pe_equation_count_conservation(
                a_unknown in proptest::bool::ANY,
                b_unknown in proptest::bool::ANY,
            ) {
                let _seed = fj_test_utils::capture_proptest_seed();
                let jaxpr = make_add_chain_jaxpr();
                let result = partial_eval_jaxpr(&jaxpr, &[a_unknown, b_unknown]).unwrap();
                let total = result.jaxpr_known.equations.len()
                    + result.jaxpr_unknown.equations.len();
                // In general total >= original because some eqns may appear only in one.
                // But our PE partitions: total == original
                prop_assert_eq!(total, jaxpr.equations.len());
            }

            /// All-known always yields empty unknown jaxpr.
            #[test]
            fn prop_pe_all_known_yields_empty_unknown(
                n_eqns in 1_usize..=3,
            ) {
                let _seed = fj_test_utils::capture_proptest_seed();
                let jaxpr = match n_eqns {
                    1 => make_single_eqn_jaxpr(),
                    2 => make_add_chain_jaxpr(),
                    _ => make_deep_chain_jaxpr(),
                };
                let unknowns = vec![false; jaxpr.invars.len()];
                let result = partial_eval_jaxpr(&jaxpr, &unknowns).unwrap();
                prop_assert_eq!(result.jaxpr_unknown.equations.len(), 0);
                prop_assert!(result.out_unknowns.iter().all(|u| !u));
            }

            /// All-unknown always yields empty known jaxpr.
            #[test]
            fn prop_pe_all_unknown_yields_empty_known(
                n_eqns in 1_usize..=3,
            ) {
                let _seed = fj_test_utils::capture_proptest_seed();
                let jaxpr = match n_eqns {
                    1 => make_single_eqn_jaxpr(),
                    2 => make_add_chain_jaxpr(),
                    _ => make_deep_chain_jaxpr(),
                };
                let unknowns = vec![true; jaxpr.invars.len()];
                let result = partial_eval_jaxpr(&jaxpr, &unknowns).unwrap();
                prop_assert_eq!(result.jaxpr_known.equations.len(), 0);
                prop_assert!(result.out_unknowns.iter().all(|u| *u));
            }

            /// DCE never increases equation count.
            #[test]
            fn prop_dce_never_increases_equations(
                use_first in proptest::bool::ANY,
                use_second in proptest::bool::ANY,
            ) {
                let _seed = fj_test_utils::capture_proptest_seed();
                let jaxpr = make_multi_output_jaxpr();
                let (pruned, _) = dce_jaxpr(&jaxpr, &[use_first, use_second]);
                prop_assert!(pruned.equations.len() <= jaxpr.equations.len());
            }

            /// DCE with all outputs used preserves all equations in chains.
            #[test]
            fn prop_dce_all_used_preserves_chain(
                _dummy in 0..10_u32,
            ) {
                let _seed = fj_test_utils::capture_proptest_seed();
                let jaxpr = make_add_chain_jaxpr();
                let used = vec![true; jaxpr.outvars.len()];
                let (pruned, inputs_used) = dce_jaxpr(&jaxpr, &used);
                prop_assert_eq!(pruned.equations.len(), jaxpr.equations.len());
                prop_assert!(inputs_used.iter().all(|u| *u));
            }

            /// PE staging semantic equivalence: stage(known,unknown) == full_eval.
            #[test]
            fn prop_pe_staging_semantic_equivalence(
                a in -100_i64..100,
                b in -100_i64..100,
            ) {
                let _seed = fj_test_utils::capture_proptest_seed();
                let jaxpr = make_neg_mul_jaxpr();
                let va = Value::scalar_i64(a);
                let vb = Value::scalar_i64(b);

                let full = crate::eval_jaxpr(&jaxpr, &[va.clone(), vb.clone()]).unwrap();

                let pe = partial_eval_jaxpr(&jaxpr, &[false, true]).unwrap();
                let known_outs = crate::eval_jaxpr_with_consts(
                    &pe.jaxpr_known, &pe.known_consts, &[va],
                ).unwrap();
                let mut unk_inputs = known_outs;
                unk_inputs.push(vb);
                let staged = crate::eval_jaxpr(&pe.jaxpr_unknown, &unk_inputs).unwrap();

                prop_assert_eq!(full, staged);
            }

            /// out_unknowns length matches original outvar count.
            #[test]
            fn prop_pe_out_unknowns_length(
                a_unk in proptest::bool::ANY,
                b_unk in proptest::bool::ANY,
            ) {
                let _seed = fj_test_utils::capture_proptest_seed();
                let jaxpr = make_multi_output_jaxpr();
                let result = partial_eval_jaxpr(&jaxpr, &[a_unk, b_unk]).unwrap();
                prop_assert_eq!(result.out_unknowns.len(), jaxpr.outvars.len());
            }
        }
    }
}
