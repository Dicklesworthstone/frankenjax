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

use fj_core::{AbstractValue, Atom, DType, Equation, Jaxpr, Primitive, Shape, Value, VarId};
use std::cell::RefCell;

const DCE_ALL_USED_LINEAR_CHAIN_CACHE_LIMIT: usize = 32;

thread_local! {
    static DCE_ALL_USED_LINEAR_CHAIN_CACHE: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

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
    /// Const value count doesn't match Jaxpr constvar count.
    ConstArity { expected: usize, actual: usize },
    /// A variable referenced in an equation was not defined.
    UndefinedVariable(VarId),
    /// Residual type mismatch between known outputs and unknown inputs.
    ResidualTypeMismatch { index: usize },
    /// Shape inference failed while computing residual abstract values.
    ShapeInference {
        primitive: Primitive,
        detail: String,
    },
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
            Self::ConstArity { expected, actual } => {
                write!(
                    f,
                    "const value count mismatch: jaxpr has {} constvars, values has {} entries",
                    expected, actual
                )
            }
            Self::UndefinedVariable(var) => write!(f, "undefined variable v{}", var.0),
            Self::ResidualTypeMismatch { index } => {
                write!(f, "residual type mismatch at index {}", index)
            }
            Self::ShapeInference { primitive, detail } => {
                write!(
                    f,
                    "shape inference failed for {}: {}",
                    primitive.as_str(),
                    detail
                )
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
    partial_eval_jaxpr_typed_with_consts(jaxpr, &[], unknowns, None)
}

/// Partially evaluate a Jaxpr with explicit external const values.
///
/// Use this for closed Jaxprs produced by tracing: the `const_values` slice must
/// align one-to-one with `jaxpr.constvars`.
pub fn partial_eval_jaxpr_with_consts(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    unknowns: &[bool],
) -> Result<PartialEvalResult, PartialEvalError> {
    partial_eval_jaxpr_typed_with_consts(jaxpr, const_values, unknowns, None)
}

/// Partially evaluate a Jaxpr with optional input abstract values for proper
/// residual typing.
///
/// When `in_avals` is provided, residual abstract values are computed from the
/// input types and equation output inference. Without it, residuals default to
/// F64 scalar (legacy behavior).
pub fn partial_eval_jaxpr_typed(
    jaxpr: &Jaxpr,
    unknowns: &[bool],
    in_avals: Option<&[AbstractValue]>,
) -> Result<PartialEvalResult, PartialEvalError> {
    partial_eval_jaxpr_typed_with_consts(jaxpr, &[], unknowns, in_avals)
}

/// Partially evaluate a Jaxpr with explicit external const values and optional
/// input abstract values for residual typing.
pub fn partial_eval_jaxpr_typed_with_consts(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    unknowns: &[bool],
    in_avals: Option<&[AbstractValue]>,
) -> Result<PartialEvalResult, PartialEvalError> {
    if const_values.len() != jaxpr.constvars.len() {
        return Err(PartialEvalError::ConstArity {
            expected: jaxpr.constvars.len(),
            actual: const_values.len(),
        });
    }

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
            known_consts: const_values.to_vec(),
            jaxpr_unknown: Jaxpr::new(vec![], vec![], vec![], vec![]),
            out_unknowns,
            residual_avals: vec![],
        });
    }

    // Fast path: all-unknown (only safe if there are no constvars to thread as residuals)
    let all_unknown = unknowns.iter().all(|u| *u);
    if all_unknown && jaxpr.constvars.is_empty() {
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

    if const_values.is_empty()
        && jaxpr.constvars.is_empty()
        && in_avals.is_none()
        && let Some(result) = try_partial_eval_two_eq_mixed_residual(jaxpr, unknowns)
    {
        return Ok(result);
    }

    // Mixed case: use VarId-indexed bitset for O(1) lookups.
    let max_var_id = max_var_in_jaxpr(jaxpr);
    let bitset_len = max_var_id + 1;

    // Build optional var→aval map for residual typing.
    let mut var_aval: Vec<Option<AbstractValue>> = vec![None; bitset_len];
    for (var, value) in jaxpr.constvars.iter().zip(const_values.iter()) {
        var_aval[var.0 as usize] = Some(abstract_value_of(value));
    }
    if let Some(avals) = in_avals {
        for (var, aval) in jaxpr.invars.iter().zip(avals.iter()) {
            var_aval[var.0 as usize] = Some(aval.clone());
        }
    }

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
            // Propagate types through known equations for residual typing.
            if in_avals.is_some() {
                let input_avals: Vec<AbstractValue> = eqn
                    .inputs
                    .iter()
                    .filter_map(|atom| match atom {
                        Atom::Var(v) => var_aval[v.0 as usize].clone(),
                        Atom::Lit(lit) => Some(abstract_value_of_literal(lit)),
                    })
                    .collect();
                if !input_avals.is_empty() {
                    let out_avals = infer_equation_output_avals(eqn, &input_avals)?;
                    for (out_var, out_aval) in eqn.outputs.iter().zip(out_avals) {
                        var_aval[out_var.0 as usize] = Some(out_aval);
                    }
                }
            }
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

    // Compute residual abstract values from var_aval map when available,
    // falling back to F64 scalar for unknown types.
    let default_aval = AbstractValue {
        dtype: DType::F64,
        shape: Shape::scalar(),
    };
    let residual_avals: Vec<AbstractValue> = residual_vars
        .iter()
        .map(|v| {
            var_aval[v.0 as usize]
                .clone()
                .unwrap_or_else(|| default_aval.clone())
        })
        .collect();

    Ok(PartialEvalResult {
        jaxpr_known,
        known_consts: const_values.to_vec(),
        jaxpr_unknown,
        out_unknowns,
        residual_avals,
    })
}

fn try_partial_eval_two_eq_mixed_residual(
    jaxpr: &Jaxpr,
    unknowns: &[bool],
) -> Option<PartialEvalResult> {
    if jaxpr.invars.len() != 2
        || unknowns.len() != 2
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 2
    {
        return None;
    }

    let (known_var, unknown_var) = match unknowns {
        [false, true] => (jaxpr.invars[0], jaxpr.invars[1]),
        [true, false] => (jaxpr.invars[1], jaxpr.invars[0]),
        _ => return None,
    };
    let [known_eqn, unknown_eqn] = jaxpr.equations.as_slice() else {
        return None;
    };
    let [residual_var] = known_eqn.outputs.as_slice() else {
        return None;
    };
    let [unknown_out] = unknown_eqn.outputs.as_slice() else {
        return None;
    };
    if *unknown_out != jaxpr.outvars[0] {
        return None;
    }

    let known_inputs_are_known = known_eqn.inputs.iter().all(|atom| match atom {
        Atom::Var(var) => *var == known_var,
        Atom::Lit(_) => true,
    });
    if !known_inputs_are_known {
        return None;
    }

    let mut sees_residual = false;
    let mut sees_unknown = false;
    for atom in &unknown_eqn.inputs {
        let Atom::Var(var) = atom else {
            continue;
        };
        if *var == *residual_var {
            sees_residual = true;
        } else if *var == unknown_var {
            sees_unknown = true;
        } else {
            return None;
        }
    }
    if !sees_residual || !sees_unknown {
        return None;
    }

    Some(PartialEvalResult {
        jaxpr_known: Jaxpr::new(
            vec![known_var],
            vec![],
            vec![*residual_var],
            vec![known_eqn.clone()],
        ),
        known_consts: vec![],
        jaxpr_unknown: Jaxpr::new(
            vec![*residual_var, unknown_var],
            vec![],
            vec![*unknown_out],
            vec![unknown_eqn.clone()],
        ),
        out_unknowns: vec![true],
        residual_avals: vec![AbstractValue {
            dtype: DType::F64,
            shape: Shape::scalar(),
        }],
    })
}

/// Dead code elimination on a Jaxpr.
///
/// Given a Jaxpr and a mask of which outputs are used, removes equations
/// that don't contribute to any used output. Preserves equation ordering.
///
/// Returns the pruned Jaxpr and a mask of which inputs are still needed.
pub fn dce_jaxpr(jaxpr: &Jaxpr, used_outputs: &[bool]) -> (Jaxpr, Vec<bool>) {
    let all_outputs_used =
        used_outputs.len() == jaxpr.outvars.len() && used_outputs.iter().all(|used| *used);
    if all_outputs_used && dce_all_used_linear_chain_cache_hit(jaxpr) {
        return (jaxpr.clone(), vec![true]);
    }

    if let Some(result) = try_dce_all_used_linear_chain(jaxpr, used_outputs) {
        if all_outputs_used {
            remember_dce_all_used_linear_chain(jaxpr);
        }
        return result;
    }

    let max_var = max_var_in_jaxpr(jaxpr);
    let bitset_len = max_var + 1;

    // Backward pass: determine which variables are needed via indexed bitset.
    let mut needed = vec![false; bitset_len];

    for (var, &used) in jaxpr.outvars.iter().zip(used_outputs.iter()) {
        if used {
            needed[var.0 as usize] = true;
        }
    }

    let retained_eqns = if jaxpr.equations.is_empty() {
        let mut retained_eqns = Vec::with_capacity(jaxpr.equations.len());
        for eqn in jaxpr.equations.iter().rev() {
            let outputs_needed = eqn.outputs.iter().any(|v| needed[v.0 as usize]);
            if outputs_needed {
                for atom in &eqn.inputs {
                    if let Atom::Var(v) = atom {
                        needed[v.0 as usize] = true;
                    }
                }
                retained_eqns.push(eqn.clone());
            }
        }
        retained_eqns.reverse();
        retained_eqns.into()
    } else {
        let mut retain_eqn = vec![false; jaxpr.equations.len()];
        let mut retained_count = 0;
        for (eqn_index, eqn) in jaxpr.equations.iter().enumerate().rev() {
            let outputs_needed = eqn.outputs.iter().any(|v| needed[v.0 as usize]);
            // An effectful equation must be retained even when its outputs are dead:
            // dropping it would discard the side effect. This matches JAX's dce_jaxpr,
            // which keeps any equation whose effects are non-empty. Its inputs then
            // become live too (they are needed to perform the effect).
            if outputs_needed || !eqn.effects.is_empty() {
                retain_eqn[eqn_index] = true;
                retained_count += 1;
                for atom in &eqn.inputs {
                    if let Atom::Var(v) = atom {
                        needed[v.0 as usize] = true;
                    }
                }
            }
        }

        if retained_count == jaxpr.equations.len() {
            jaxpr.equations.clone()
        } else {
            jaxpr
                .equations
                .iter()
                .zip(retain_eqn.iter())
                .filter(|(_, retained)| **retained)
                .map(|(eqn, _)| eqn.clone())
                .collect()
        }
    };

    let used_inputs: Vec<bool> = jaxpr.invars.iter().map(|v| needed[v.0 as usize]).collect();

    let retained_outvars: Vec<VarId> = jaxpr
        .outvars
        .iter()
        .zip(used_outputs.iter())
        .filter(|&(_, &used)| used)
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

fn dce_all_used_linear_chain_cache_hit(jaxpr: &Jaxpr) -> bool {
    let fingerprint = jaxpr.canonical_fingerprint();
    DCE_ALL_USED_LINEAR_CHAIN_CACHE
        .with(|cache| cache.borrow().iter().any(|cached| cached == fingerprint))
}

fn remember_dce_all_used_linear_chain(jaxpr: &Jaxpr) {
    let fingerprint = jaxpr.canonical_fingerprint();
    DCE_ALL_USED_LINEAR_CHAIN_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.iter().any(|cached| cached == fingerprint) {
            return;
        }
        if cache.len() == DCE_ALL_USED_LINEAR_CHAIN_CACHE_LIMIT {
            cache.remove(0);
        }
        cache.push(fingerprint.to_owned());
    });
}

fn try_dce_all_used_linear_chain(
    jaxpr: &Jaxpr,
    used_outputs: &[bool],
) -> Option<(Jaxpr, Vec<bool>)> {
    if used_outputs.len() != jaxpr.outvars.len()
        || !used_outputs.iter().all(|used| *used)
        || jaxpr.invars.len() != 1
        || !jaxpr.constvars.is_empty()
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.is_empty()
    {
        return None;
    }

    let mut current = jaxpr.invars[0];
    for eqn in &jaxpr.equations {
        let [output] = eqn.outputs.as_slice() else {
            return None;
        };

        let mut consumes_current = false;
        for atom in &eqn.inputs {
            let Atom::Var(var) = atom else {
                continue;
            };
            if *var != current {
                return None;
            }
            consumes_current = true;
        }
        if !consumes_current {
            return None;
        }

        current = *output;
    }

    if current != jaxpr.outvars[0] {
        return None;
    }

    Some((jaxpr.clone(), vec![true]))
}

/// Infer the output abstract value of an equation from the first input's aval.
///
/// For most element-wise primitives the output type/shape matches the input.
/// Comparison primitives always output Bool. Reductions honour the "axes" param.
/// Shape-manipulation ops (Reshape, Slice, Transpose) parse their params.
/// Dot always produces a scalar.
/// Right-aligned numpy broadcast of two dim lists (missing axes treated as 1).
/// Correct for compatible shapes; best-effort (max) otherwise.
fn broadcast_dims(a: &[u32], b: &[u32]) -> Vec<u32> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| {
            let ad = if i + a.len() < n {
                1
            } else {
                a[i + a.len() - n]
            };
            let bd = if i + b.len() < n {
                1
            } else {
                b[i + b.len() - n]
            };
            ad.max(bd)
        })
        .collect()
}

/// BroadcastedIota takes NO inputs — its aval comes entirely from params.
fn infer_broadcasted_iota_aval(eqn: &Equation) -> Result<AbstractValue, PartialEvalError> {
    let primitive = Primitive::BroadcastedIota;
    let dims = parse_required_u32_list_param(eqn, "shape")?;

    let dimension = match eqn.params.get("dimension") {
        Some(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return Err(PartialEvalError::ShapeInference {
                    primitive,
                    detail: "invalid dimension: ''".to_owned(),
                });
            }
            trimmed
                .parse::<usize>()
                .map_err(|_| PartialEvalError::ShapeInference {
                    primitive,
                    detail: format!("invalid dimension: '{raw}'"),
                })?
        }
        None => 0,
    };
    if !dims.is_empty() && dimension >= dims.len() {
        return Err(PartialEvalError::ShapeInference {
            primitive,
            detail: format!(
                "dimension {dimension} out of bounds for rank {}",
                dims.len()
            ),
        });
    }

    let dtype = match eqn.params.get("dtype") {
        Some(raw) => dtype_from_name(raw).ok_or_else(|| PartialEvalError::ShapeInference {
            primitive,
            detail: format!("invalid dtype: '{raw}'"),
        })?,
        None => DType::I64,
    };
    if matches!(dtype, DType::Bool) {
        return Err(PartialEvalError::ShapeInference {
            primitive,
            detail: "broadcasted_iota does not accept bool dtype".to_owned(),
        });
    }

    Ok(AbstractValue {
        dtype,
        shape: Shape { dims },
    })
}

fn parse_required_u32_list_param(
    eqn: &Equation,
    key: &str,
) -> Result<Vec<u32>, PartialEvalError> {
    let raw = eqn
        .params
        .get(key)
        .ok_or_else(|| PartialEvalError::ShapeInference {
            primitive: eqn.primitive,
            detail: format!("missing required param '{key}'"),
        })?;
    parse_u32_list_param(eqn.primitive, key, raw)
}

fn parse_u32_list_param(
    primitive: Primitive,
    key: &str,
    raw: &str,
) -> Result<Vec<u32>, PartialEvalError> {
    raw.split(',')
        .map(str::trim)
        .map(|piece| {
            piece
                .parse::<u32>()
                .map_err(|_| PartialEvalError::ShapeInference {
                    primitive,
                    detail: format!("invalid u32 in param '{key}': '{piece}'"),
                })
        })
        .collect()
}

fn parse_usize_list_param(
    primitive: Primitive,
    key: &str,
    raw: &str,
) -> Result<Vec<usize>, PartialEvalError> {
    raw.split(',')
        .map(str::trim)
        .map(|piece| {
            piece
                .parse::<usize>()
                .map_err(|_| PartialEvalError::ShapeInference {
                    primitive,
                    detail: format!("invalid usize in param '{key}': '{piece}'"),
                })
        })
        .collect()
}

fn infer_broadcast_in_dim_shape(
    eqn: &Equation,
    input: &AbstractValue,
) -> Result<Shape, PartialEvalError> {
    let target_dims = parse_required_u32_list_param(eqn, "shape")?;
    let target_rank = target_dims.len();
    let input_rank = input.shape.rank();
    let broadcast_dims = if let Some(raw) = eqn.params.get("broadcast_dimensions") {
        parse_usize_list_param(eqn.primitive, "broadcast_dimensions", raw)?
    } else {
        if input_rank > target_rank {
            return Err(PartialEvalError::ShapeInference {
                primitive: eqn.primitive,
                detail: format!("input rank {input_rank} exceeds output rank {target_rank}"),
            });
        }
        (target_rank - input_rank..target_rank).collect()
    };

    if broadcast_dims.len() != input_rank {
        return Err(PartialEvalError::ShapeInference {
            primitive: eqn.primitive,
            detail: format!(
                "broadcast_dimensions length {} must equal input rank {input_rank}",
                broadcast_dims.len()
            ),
        });
    }

    let mut seen = Vec::with_capacity(broadcast_dims.len());
    for (input_axis, &target_axis) in broadcast_dims.iter().enumerate() {
        if target_axis >= target_rank {
            return Err(PartialEvalError::ShapeInference {
                primitive: eqn.primitive,
                detail: format!("target axis {target_axis} out of range"),
            });
        }
        if seen.contains(&target_axis) {
            return Err(PartialEvalError::ShapeInference {
                primitive: eqn.primitive,
                detail: "broadcast_dimensions must be unique".to_owned(),
            });
        }
        seen.push(target_axis);

        let input_dim = input.shape.dims[input_axis];
        let target_dim = target_dims[target_axis];
        if input_dim != 1 && input_dim != target_dim {
            return Err(PartialEvalError::ShapeInference {
                primitive: eqn.primitive,
                detail: format!(
                    "cannot broadcast input dim {input_dim} into target dim {target_dim} at axis {target_axis}"
                ),
            });
        }
    }

    Ok(Shape { dims: target_dims })
}

fn infer_slice_shape(eqn: &Equation, input: &AbstractValue) -> Result<Shape, PartialEvalError> {
    let rank = input.shape.rank();
    let starts = parse_required_u32_list_param(eqn, "start_indices")?;
    let limits = parse_required_u32_list_param(eqn, "limit_indices")?;
    let strides = match eqn.params.get("strides") {
        Some(raw) => parse_u32_list_param(eqn.primitive, "strides", raw)?,
        None => vec![1; rank],
    };

    if starts.len() != rank || limits.len() != rank {
        return Err(PartialEvalError::ShapeInference {
            primitive: eqn.primitive,
            detail: format!(
                "slice start/limit rank mismatch starts={} limits={} rank={rank}",
                starts.len(),
                limits.len(),
            ),
        });
    }
    if strides.len() != rank {
        return Err(PartialEvalError::ShapeInference {
            primitive: eqn.primitive,
            detail: format!(
                "slice strides rank mismatch strides={} rank={rank}",
                strides.len()
            ),
        });
    }

    let mut dims = Vec::with_capacity(rank);
    for axis in 0..rank {
        let start = starts[axis];
        let limit = limits[axis];
        let stride = strides[axis];
        let bound = input.shape.dims[axis];
        if stride == 0 {
            return Err(PartialEvalError::ShapeInference {
                primitive: eqn.primitive,
                detail: format!("slice stride on axis {axis} must be positive"),
            });
        }
        if start > limit || limit > bound {
            return Err(PartialEvalError::ShapeInference {
                primitive: eqn.primitive,
                detail: format!(
                    "invalid slice bounds axis {axis}: start={start} limit={limit} bound={bound}"
                ),
            });
        }
        dims.push((limit - start).div_ceil(stride));
    }

    Ok(Shape { dims })
}

/// Complex(re, im): dtype (F32,F32)→Complex64 else Complex128; shape = broadcast.
fn infer_complex_aval(input_avals: &[AbstractValue]) -> AbstractValue {
    let dtype = match (
        input_avals.first().map(|v| v.dtype),
        input_avals.get(1).map(|v| v.dtype),
    ) {
        (Some(DType::F32), Some(DType::F32)) => DType::Complex64,
        _ => DType::Complex128,
    };
    let a = input_avals
        .first()
        .map(|v| v.shape.dims.as_slice())
        .unwrap_or(&[]);
    let b = input_avals
        .get(1)
        .map(|v| v.shape.dims.as_slice())
        .unwrap_or(&[]);
    AbstractValue {
        dtype,
        shape: Shape {
            dims: broadcast_dims(a, b),
        },
    }
}

/// DotGeneral output aval, mirroring fj-trace's `infer_dot_general`: the output
/// shape is `[lhs batch dims] ++ [lhs free dims] ++ [rhs free dims]` and the
/// dtype is the promotion of the two operand dtypes. DotGeneral needs BOTH
/// operands, so it lives in the multi-input dispatcher (like Complex/Solve);
/// the single-input catch-all previously typed a residual matmul with the LHS
/// shape, one rank-pair too wide. Falls back to the first input aval when the
/// arity is wrong (best-effort: staging must not panic on a malformed residual).
fn infer_dot_general_aval(eqn: &Equation, input_avals: &[AbstractValue]) -> AbstractValue {
    let (Some(lhs), Some(rhs)) = (input_avals.first(), input_avals.get(1)) else {
        return input_avals.first().cloned().unwrap_or(AbstractValue {
            dtype: DType::F64,
            shape: Shape::scalar(),
        });
    };

    let parse_dims = |key: &str| -> Vec<usize> {
        eqn.params
            .get(key)
            .map(|s| {
                s.trim_matches(|c| c == '[' || c == ']')
                    .split(',')
                    .filter(|x| !x.trim().is_empty())
                    .filter_map(|x| x.trim().parse::<usize>().ok())
                    .collect()
            })
            .unwrap_or_default()
    };
    let lhs_contracting = parse_dims("lhs_contracting_dims");
    let rhs_contracting = parse_dims("rhs_contracting_dims");
    let lhs_batch = parse_dims("lhs_batch_dims");
    let rhs_batch = parse_dims("rhs_batch_dims");

    let mut out_dims: Vec<u32> = Vec::new();
    for &b in &lhs_batch {
        if let Some(&d) = lhs.shape.dims.get(b) {
            out_dims.push(d);
        }
    }
    for (i, &d) in lhs.shape.dims.iter().enumerate() {
        if !lhs_contracting.contains(&i) && !lhs_batch.contains(&i) {
            out_dims.push(d);
        }
    }
    for (i, &d) in rhs.shape.dims.iter().enumerate() {
        if !rhs_contracting.contains(&i) && !rhs_batch.contains(&i) {
            out_dims.push(d);
        }
    }

    let dtype = fj_lax::promote_dtype_public(lhs.dtype, rhs.dtype);
    AbstractValue {
        dtype,
        shape: Shape { dims: out_dims },
    }
}

fn infer_equation_output_avals(
    eqn: &Equation,
    input_avals: &[AbstractValue],
) -> Result<Vec<AbstractValue>, PartialEvalError> {
    use fj_core::Primitive::*;

    // AUTHORITATIVE PATH: delegate to fj-trace's `infer_output_avals` — the SAME
    // inference tracing uses — for EVERY op, so staging and tracing agree by
    // construction (no more two-layers drift). fj-trace is exhaustive over
    // Primitive and returns the correct output count for multi-output ops too.
    // The local arms below are now only a best-effort FALLBACK for residuals
    // fj-trace validates and REJECTS (it is stricter than the lenient staging
    // path, which must never fail). Covers BroadcastedIota (0 inputs) as well.
    if let Some(avals) = delegate_infer_to_trace(eqn, input_avals) {
        return Ok(avals);
    }

    // BroadcastedIota takes NO inputs, so it must be typed before the
    // first-input guard below (which would otherwise return an empty aval list).
    if eqn.primitive == BroadcastedIota {
        return Ok(vec![infer_broadcasted_iota_aval(eqn)?; eqn.outputs.len()]);
    }

    let Some(first_input) = input_avals.first() else {
        return Ok(vec![]);
    };

    match eqn.primitive {
        Qr => Ok(infer_qr_output_avals(first_input, eqn)),
        Svd => Ok(infer_svd_output_avals(first_input, eqn)),
        Eigh => Ok(infer_eigh_output_avals(first_input)),
        Slogdet => Ok(infer_slogdet_output_avals(first_input)),
        Eig => Ok(infer_eig_output_avals(first_input)),
        TopK => Ok(infer_topk_output_avals(first_input, eqn)),
        Solve => Ok(infer_solve_output_avals(input_avals)),
        // Complex is binary — the dtype/shape depend on BOTH inputs, which the
        // single-input infer_equation_output_aval can't see.
        Complex => Ok(vec![infer_complex_aval(input_avals); eqn.outputs.len()]),
        // DotGeneral's output shape/dtype depend on BOTH operands (and the
        // contracting/batch dimension_numbers params), so it likewise needs the
        // multi-input path. The catch-all typed a residual matmul as the LHS.
        DotGeneral => Ok(vec![
            infer_dot_general_aval(eqn, input_avals);
            eqn.outputs.len()
        ]),
        _ => {
            let out_aval = infer_equation_output_aval(eqn, first_input)?;
            Ok(vec![out_aval; eqn.outputs.len()])
        }
    }
}

/// Delegate a residual equation's aval inference to fj-trace's authoritative
/// `infer_output_avals` (the single source of truth that tracing uses), bridging
/// `fj_core::AbstractValue` ↔ `fj_trace::ShapedArray` (both are `{dtype, shape}`).
/// Returns `None` on any inference error or output-arity mismatch so the caller
/// can fall back to local best-effort typing — residual staging must never fail.
fn delegate_infer_to_trace(
    eqn: &Equation,
    input_avals: &[AbstractValue],
) -> Option<Vec<AbstractValue>> {
    let trace_inputs: Vec<fj_trace::ShapedArray> = input_avals
        .iter()
        .map(|av| fj_trace::ShapedArray {
            dtype: av.dtype,
            shape: av.shape.clone(),
        })
        .collect();
    let out = fj_trace::infer_output_avals(eqn.primitive, &trace_inputs, &eqn.params).ok()?;
    if out.len() != eqn.outputs.len() {
        return None;
    }
    Some(
        out.into_iter()
            .map(|sa| AbstractValue {
                dtype: sa.dtype,
                shape: sa.shape,
            })
            .collect(),
    )
}

/// Parse a `new_dtype`/`dtype` param string into a `DType`, mirroring
/// fj-trace's `parse_dtype_name` (the authoritative trace parser, so the
/// accepted spellings stay in lockstep). Returns None on an unknown name —
/// callers fall back to a shape/dtype-preserving default (staging is
/// best-effort and must not panic on a residual it can't precisely type).
fn dtype_from_name(raw: &str) -> Option<DType> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "bf16" | "bfloat16" => Some(DType::BF16),
        "f16" | "float16" => Some(DType::F16),
        "f32" | "float32" => Some(DType::F32),
        "f64" | "float64" => Some(DType::F64),
        "i32" => Some(DType::I32),
        "i64" => Some(DType::I64),
        "u32" => Some(DType::U32),
        "u64" => Some(DType::U64),
        "bool" => Some(DType::Bool),
        "complex64" => Some(DType::Complex64),
        "complex128" => Some(DType::Complex128),
        _ => None,
    }
}

fn infer_equation_output_aval(
    eqn: &Equation,
    first_input: &AbstractValue,
) -> Result<AbstractValue, PartialEvalError> {
    use fj_core::Primitive::*;
    let out_aval = match eqn.primitive {
        // Comparisons always produce Bool
        Eq | Ne | Lt | Le | Gt | Ge => AbstractValue {
            dtype: DType::Bool,
            shape: first_input.shape.clone(),
        },
        // Reductions: honour "axes" param when present
        ReduceSum | ReduceMax | ReduceMin | ReduceProd | ReduceAnd | ReduceOr | ReduceXor => {
            let reduced_shape = match eqn.params.get("axes") {
                Some(axes_str) if !axes_str.trim().is_empty() => {
                    reduction_shape_from_axes(eqn.primitive, &first_input.shape, axes_str)?
                }
                Some(_) => first_input.shape.clone(),
                _ => Shape::scalar(),
            };
            AbstractValue {
                dtype: first_input.dtype,
                shape: reduced_shape,
            }
        }
        // Dot product always produces a scalar
        Dot => AbstractValue {
            dtype: first_input.dtype,
            shape: Shape::scalar(),
        },
        // Det preserves the input's float/complex dtype (int -> F64), matching
        // eval_det and fj-trace's infer_det. The prior hardcoded F64 widened f32
        // determinants in partial-eval (silent dtype divergence from eval).
        Det => AbstractValue {
            dtype: det_scalar_dtype(first_input.dtype),
            shape: Shape::scalar(),
        },
        // associative_scan is a prefix scan in place: same shape and dtype.
        AssociativeScan => first_input.clone(),
        // Reshape: parse "new_shape" param, including JAX/NumPy's single
        // inferred -1 dimension. Invalid fallback params should not silently
        // drop dimensions by filtering failed u32 parses.
        Reshape => {
            let shape = match eqn.params.get("new_shape") {
                Some(raw_shape) => {
                    infer_reshape_shape(eqn.primitive, &first_input.shape, raw_shape)?
                }
                None => first_input.shape.clone(),
            };
            AbstractValue {
                dtype: first_input.dtype,
                shape,
            }
        }
        // Slice: shape = limit_indices - start_indices
        Slice => {
            let shape = infer_slice_shape(eqn, first_input)?;
            AbstractValue {
                dtype: first_input.dtype,
                shape,
            }
        }
        // Transpose: permute dims according to "permutation" param
        Transpose => {
            let shape = eqn
                .params
                .get("permutation")
                .map(|s| {
                    let perm: Vec<usize> = s
                        .split(',')
                        .filter_map(|p| p.trim().parse::<usize>().ok())
                        .collect();
                    let dims: Vec<u32> = perm
                        .iter()
                        .filter_map(|&i| first_input.shape.dims.get(i).copied())
                        .collect();
                    Shape { dims }
                })
                .unwrap_or_else(|| {
                    // Default: reverse dims
                    let mut dims = first_input.shape.dims.clone();
                    dims.reverse();
                    Shape { dims }
                });
            AbstractValue {
                dtype: first_input.dtype,
                shape,
            }
        }
        // BroadcastInDim: parse "shape" param for target shape
        BroadcastInDim => {
            let shape = infer_broadcast_in_dim_shape(eqn, first_input)?;
            AbstractValue {
                dtype: first_input.dtype,
                shape,
            }
        }
        // Concatenate: sum along concat dimension
        Concatenate => {
            // Without access to all inputs, we can only approximate.
            // The dtype is correct; the shape is best-effort from first input.
            first_input.clone()
        }
        // Pad: expand each axis by low/high edges and interior spacing.
        Pad => {
            let parse_list = |key: &str| {
                eqn.params.get(key).map(|s| {
                    s.split(',')
                        .filter_map(|d| d.trim().parse::<u32>().ok())
                        .collect::<Vec<_>>()
                })
            };

            let rank = first_input.shape.rank();
            let shape = match (
                parse_list("padding_low"),
                parse_list("padding_high"),
                parse_list("padding_interior"),
            ) {
                (Some(lows), Some(highs), Some(interiors))
                    if lows.len() == rank && highs.len() == rank && interiors.len() == rank =>
                {
                    let dims = first_input
                        .shape
                        .dims
                        .iter()
                        .enumerate()
                        .map(|(i, &dim)| {
                            let interior_span = dim.saturating_sub(1).saturating_mul(interiors[i]);
                            lows[i]
                                .saturating_add(dim)
                                .saturating_add(interior_span)
                                .saturating_add(highs[i])
                        })
                        .collect();
                    Shape { dims }
                }
                _ => first_input.shape.clone(),
            };

            AbstractValue {
                dtype: first_input.dtype,
                shape,
            }
        }
        // DynamicSlice: output shape = slice_sizes param
        DynamicSlice => {
            let shape = eqn
                .params
                .get("slice_sizes")
                .map(|s| {
                    let dims: Vec<u32> = s
                        .split(',')
                        .filter_map(|d| d.trim().parse::<u32>().ok())
                        .collect();
                    Shape { dims }
                })
                .unwrap_or_else(|| first_input.shape.clone());
            AbstractValue {
                dtype: first_input.dtype,
                shape,
            }
        }
        // Clamp: output shape matches first input
        Clamp => first_input.clone(),
        // Iota: output from params (no real input to infer from)
        Iota => {
            let length = eqn
                .params
                .get("length")
                .and_then(|s| s.trim().parse::<u32>().ok())
                .unwrap_or(0);
            let dtype_str = eqn.params.get("dtype").map(String::as_str).unwrap_or("I64");
            let dtype = match dtype_str {
                "F64" | "f64" => DType::F64,
                _ => DType::I64,
            };
            AbstractValue {
                dtype,
                shape: Shape::vector(length),
            }
        }
        // dtype-changing, shape-preserving unary ops: the catch-all kept the
        // INPUT dtype, so a staged residual carried the wrong element type.
        ConvertElementType | BitcastConvertType => {
            let dtype = eqn
                .params
                .get("new_dtype")
                .and_then(|s| dtype_from_name(s))
                .unwrap_or(first_input.dtype);
            AbstractValue {
                dtype,
                shape: first_input.shape.clone(),
            }
        }
        // Real/Imag: complex operand → its real component dtype.
        Real | Imag => {
            let dtype = match first_input.dtype {
                DType::Complex64 => DType::F32,
                DType::Complex128 => DType::F64,
                other => other,
            };
            AbstractValue {
                dtype,
                shape: first_input.shape.clone(),
            }
        }
        // Unary predicates always produce Bool.
        IsFinite | IsNan | IsInf | Signbit => AbstractValue {
            dtype: DType::Bool,
            shape: first_input.shape.clone(),
        },
        // ExpandDims: insert a size-1 axis (normalize a negative axis against
        // rank+1, matching numpy/jnp expand_dims and fj-trace). Without this the
        // catch-all returned the INPUT shape — a residual typed one rank too low.
        ExpandDims => {
            let rank = first_input.shape.rank() as i64;
            let raw_axis: i64 = eqn
                .params
                .get("axis")
                .and_then(|s| s.split(',').next())
                .and_then(|s| s.trim().parse::<i64>().ok())
                .unwrap_or(0);
            let norm = if raw_axis < 0 {
                raw_axis + rank + 1
            } else {
                raw_axis
            };
            let mut dims = first_input.shape.dims.clone();
            let axis = norm.clamp(0, rank) as usize;
            dims.insert(axis, 1);
            AbstractValue {
                dtype: first_input.dtype,
                shape: Shape { dims },
            }
        }
        // Squeeze: drop the listed dims (negative-normalized against rank), or all
        // size-1 dims if unspecified. Catch-all previously kept the input shape.
        Squeeze => {
            let rank = first_input.shape.rank() as i64;
            let drop: Vec<usize> = match eqn.params.get("dimensions") {
                Some(s) if !s.trim().is_empty() => s
                    .split(',')
                    .filter_map(|d| d.trim().parse::<i64>().ok())
                    .map(|d| (if d < 0 { d + rank } else { d }).clamp(0, rank.max(0)) as usize)
                    .collect(),
                _ => first_input
                    .shape
                    .dims
                    .iter()
                    .enumerate()
                    .filter(|&(_, &d)| d == 1)
                    .map(|(i, _)| i)
                    .collect(),
            };
            let dims: Vec<u32> = first_input
                .shape
                .dims
                .iter()
                .enumerate()
                .filter(|(i, _)| !drop.contains(i))
                .map(|(_, &d)| d)
                .collect();
            AbstractValue {
                dtype: first_input.dtype,
                shape: Shape { dims },
            }
        }
        // Argmax/Argmin: drop the reduced axis; output is ALWAYS I64 indices. The
        // catch-all kept the input shape AND dtype — wrong on both counts.
        Argmax | Argmin => {
            let rank = first_input.shape.rank() as i64;
            let shape = if rank == 0 {
                Shape::scalar()
            } else {
                let raw_axis: i64 = eqn
                    .params
                    .get("axis")
                    .and_then(|s| s.trim().parse::<i64>().ok())
                    .unwrap_or(rank - 1);
                let norm = if raw_axis < 0 {
                    raw_axis + rank
                } else {
                    raw_axis
                };
                let mut dims = first_input.shape.dims.clone();
                if norm >= 0 && (norm as usize) < dims.len() {
                    dims.remove(norm as usize);
                }
                Shape { dims }
            };
            AbstractValue {
                dtype: DType::I64,
                shape,
            }
        }
        // Tile: NumPy/JAX-style rank promotion left-pads the shorter side with
        // ones before multiplying dimensions. Catch-all kept the input shape.
        Tile => {
            let reps: Vec<u32> = eqn
                .params
                .get("reps")
                .map(|s| {
                    s.split(',')
                        .filter_map(|r| r.trim().parse::<u32>().ok())
                        .collect()
                })
                .unwrap_or_default();
            let in_dims = &first_input.shape.dims;
            let rank = in_dims.len().max(reps.len());
            let mut tile_dims = Vec::with_capacity(rank);
            tile_dims.resize(rank - in_dims.len(), 1);
            tile_dims.extend_from_slice(in_dims);

            let mut tile_reps = Vec::with_capacity(rank);
            tile_reps.resize(rank - reps.len(), 1);
            tile_reps.extend(reps);

            let dims: Vec<u32> = tile_dims
                .iter()
                .zip(&tile_reps)
                .map(|(&d, &r)| {
                    d.checked_mul(r)
                        .ok_or_else(|| PartialEvalError::ShapeInference {
                            primitive: eqn.primitive,
                            detail: "tile result dimension overflows u32".to_owned(),
                        })
                })
                .collect::<Result<_, _>>()?;
            AbstractValue {
                dtype: first_input.dtype,
                shape: Shape { dims },
            }
        }
        // OneHot: insert a num_classes axis (default last; negative normalized
        // against the output rank); output dtype from "dtype" (default F64). The
        // catch-all kept the index shape AND its integer dtype.
        OneHot => {
            let num_classes: u32 = eqn
                .params
                .get("num_classes")
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0);
            let dtype = eqn
                .params
                .get("dtype")
                .and_then(|s| dtype_from_name(s))
                .unwrap_or(DType::F64);
            let mut out_dims = first_input.shape.dims.clone();
            let output_rank = out_dims.len() + 1;
            let raw_axis: i64 = eqn
                .params
                .get("axis")
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or((output_rank - 1) as i64);
            let norm = if raw_axis < 0 {
                raw_axis + output_rank as i64
            } else {
                raw_axis
            };
            let axis = norm.clamp(0, (output_rank - 1) as i64) as usize;
            out_dims.insert(axis, num_classes);
            AbstractValue {
                dtype,
                shape: Shape { dims: out_dims },
            }
        }
        // Split: mirror fj-trace's single-output rule — an EVEN split packs a
        // [num_sections, section_size] pair at the split axis; an UNEVEN split is
        // best-effort to the first section's shape. Catch-all kept the full input
        // shape. Falls back to the input aval on inconsistent params (no panic).
        Split => {
            let rank = first_input.shape.rank();
            let dims = &first_input.shape.dims;
            let raw_axis: i64 = eqn
                .params
                .get("axis")
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0);
            let axis_i = if raw_axis < 0 {
                raw_axis + rank as i64
            } else {
                raw_axis
            };
            if rank == 0 || axis_i < 0 || axis_i >= rank as i64 {
                first_input.clone()
            } else {
                let axis = axis_i as usize;
                let axis_size = dims[axis];
                let sizes: Vec<u32> =
                    if let Some(s) = eqn.params.get("sizes").filter(|s| !s.trim().is_empty()) {
                        s.split(',')
                            .filter_map(|x| x.trim().parse::<u32>().ok())
                            .collect()
                    } else if let Some(ns) = eqn
                        .params
                        .get("num_sections")
                        .and_then(|s| s.trim().parse::<u32>().ok())
                        .filter(|&n| n > 0)
                    {
                        if axis_size.is_multiple_of(ns) {
                            vec![axis_size / ns; ns as usize]
                        } else {
                            Vec::new()
                        }
                    } else {
                        vec![axis_size]
                    };

                if sizes.is_empty() || sizes.iter().sum::<u32>() != axis_size {
                    first_input.clone()
                } else if sizes.len() == 1 || sizes.windows(2).all(|w| w[0] == w[1]) {
                    let mut new_dims = Vec::with_capacity(dims.len() + 1);
                    for (i, &d) in dims.iter().enumerate() {
                        if i == axis {
                            new_dims.push(sizes.len() as u32);
                            new_dims.push(sizes[0]);
                        } else {
                            new_dims.push(d);
                        }
                    }
                    AbstractValue {
                        dtype: first_input.dtype,
                        shape: Shape { dims: new_dims },
                    }
                } else {
                    let mut new_dims = dims.clone();
                    new_dims[axis] = sizes[0];
                    AbstractValue {
                        dtype: first_input.dtype,
                        shape: Shape { dims: new_dims },
                    }
                }
            }
        }
        // Argsort: shape is preserved but the output is ALWAYS I64 index data;
        // the catch-all kept the input's float dtype, mistyping a residual sort.
        // (Plain Sort preserves both dtype and shape, so it stays on the default.)
        Argsort => AbstractValue {
            dtype: DType::I64,
            shape: first_input.shape.clone(),
        },
        // Most element-wise ops preserve dtype and shape
        _ => first_input.clone(),
    };
    Ok(out_aval)
}

fn infer_reshape_shape(
    primitive: Primitive,
    input_shape: &Shape,
    raw_shape: &str,
) -> Result<Shape, PartialEvalError> {
    let mut inferred_axis = None;
    let mut known_product = 1_u64;
    let mut dims = Vec::new();

    for (idx, piece) in raw_shape
        .split(',')
        .map(str::trim)
        .filter(|piece| !piece.is_empty())
        .enumerate()
    {
        let dim = piece
            .parse::<i64>()
            .map_err(|_| PartialEvalError::ShapeInference {
                primitive,
                detail: format!("invalid reshape dimension: {piece}"),
            })?;
        if dim == -1 {
            if inferred_axis.is_some() {
                return Err(PartialEvalError::ShapeInference {
                    primitive,
                    detail: "only one -1 inferred axis allowed".to_owned(),
                });
            }
            inferred_axis = Some(idx);
            dims.push(0);
            continue;
        }
        if dim < 0 {
            return Err(PartialEvalError::ShapeInference {
                primitive,
                detail: format!("reshape dimension must be non-negative, got {dim}"),
            });
        }

        let dim_u32 = u32::try_from(dim).map_err(|_| PartialEvalError::ShapeInference {
            primitive,
            detail: format!("reshape dimension out of range: {dim}"),
        })?;
        known_product = known_product
            .checked_mul(u64::from(dim_u32))
            .ok_or_else(|| PartialEvalError::ShapeInference {
                primitive,
                detail: "reshape target element count overflow".to_owned(),
            })?;
        dims.push(dim_u32);
    }

    let input_elements =
        input_shape
            .element_count()
            .ok_or_else(|| PartialEvalError::ShapeInference {
                primitive,
                detail: "input shape overflow".to_owned(),
            })?;

    if let Some(infer_idx) = inferred_axis {
        if known_product == 0 || !input_elements.is_multiple_of(known_product) {
            return Err(PartialEvalError::ShapeInference {
                primitive,
                detail: format!(
                    "cannot infer reshape dimension: input elements {input_elements} not divisible by {known_product}"
                ),
            });
        }
        let inferred = input_elements / known_product;
        dims[infer_idx] =
            u32::try_from(inferred).map_err(|_| PartialEvalError::ShapeInference {
                primitive,
                detail: format!("inferred reshape dimension out of range: {inferred}"),
            })?;
    } else if known_product != input_elements {
        return Err(PartialEvalError::ShapeInference {
            primitive,
            detail: format!(
                "reshape element mismatch: input={input_elements} target={known_product}"
            ),
        });
    }

    Ok(Shape { dims })
}

fn reduction_shape_from_axes(
    primitive: Primitive,
    input_shape: &Shape,
    axes_str: &str,
) -> Result<Shape, PartialEvalError> {
    let rank = input_shape.rank();
    let mut axes = Vec::new();
    for piece in axes_str.split(',').map(str::trim) {
        let axis = piece
            .parse::<i64>()
            .map_err(|_| PartialEvalError::ShapeInference {
                primitive,
                detail: format!("invalid axis value: {piece}"),
            })?;
        let normalized = if axis < 0 { rank as i64 + axis } else { axis };
        if normalized < 0 || normalized >= rank as i64 {
            return Err(PartialEvalError::ShapeInference {
                primitive,
                detail: format!("axis {axis} out of bounds for rank {rank}"),
            });
        }

        let normalized = normalized as usize;
        if axes.contains(&normalized) {
            return Err(PartialEvalError::ShapeInference {
                primitive,
                detail: format!("duplicate value in axes: {axis}"),
            });
        }
        axes.push(normalized);
    }

    axes.sort_unstable();
    let remaining = input_shape
        .dims
        .iter()
        .enumerate()
        .filter(|(i, _)| !axes.contains(i))
        .map(|(_, &d)| d)
        .collect();
    Ok(Shape { dims: remaining })
}

fn infer_qr_output_avals(input: &AbstractValue, eqn: &Equation) -> Vec<AbstractValue> {
    let Some((&m, &n)) = input.shape.dims.first().zip(input.shape.dims.get(1)) else {
        return vec![input.clone(); eqn.outputs.len()];
    };
    let k = m.min(n);
    let full_matrices = eqn
        .params
        .get("full_matrices")
        .is_some_and(|v| v.trim() == "true");
    let q_cols = if full_matrices { m } else { k };
    let r_rows = if full_matrices { m } else { k };
    vec![
        AbstractValue {
            dtype: input.dtype,
            shape: Shape {
                dims: vec![m, q_cols],
            },
        },
        AbstractValue {
            dtype: input.dtype,
            shape: Shape {
                dims: vec![r_rows, n],
            },
        },
    ]
}

fn infer_svd_output_avals(input: &AbstractValue, eqn: &Equation) -> Vec<AbstractValue> {
    let Some((&m, &n)) = input.shape.dims.first().zip(input.shape.dims.get(1)) else {
        return vec![input.clone(); eqn.outputs.len()];
    };
    let k = m.min(n);
    let full_matrices = eqn
        .params
        .get("full_matrices")
        .is_some_and(|v| v.trim() == "true");
    let u_cols = if full_matrices { m } else { k };
    let vt_rows = if full_matrices { n } else { k };
    vec![
        AbstractValue {
            dtype: input.dtype,
            shape: Shape {
                dims: vec![m, u_cols],
            },
        },
        AbstractValue {
            dtype: input.dtype,
            shape: Shape { dims: vec![k] },
        },
        AbstractValue {
            dtype: input.dtype,
            shape: Shape {
                dims: vec![vt_rows, n],
            },
        },
    ]
}

fn infer_eigh_output_avals(input: &AbstractValue) -> Vec<AbstractValue> {
    let Some(&n) = input.shape.dims.first() else {
        return vec![input.clone(), input.clone()];
    };
    vec![
        AbstractValue {
            dtype: input.dtype,
            shape: Shape { dims: vec![n] },
        },
        AbstractValue {
            dtype: input.dtype,
            shape: Shape { dims: vec![n, n] },
        },
    ]
}

/// Determinant scalar dtype, matching eval_det / fj-trace infer_det: complex ->
/// same complex; real float -> same float; integer -> F64 (int linalg -> float).
fn det_scalar_dtype(input: DType) -> DType {
    match input {
        DType::Complex64 => DType::Complex64,
        DType::Complex128 => DType::Complex128,
        DType::F32 => DType::F32,
        DType::BF16 => DType::BF16,
        DType::F16 => DType::F16,
        _ => DType::F64,
    }
}

fn infer_slogdet_output_avals(input: &AbstractValue) -> Vec<AbstractValue> {
    // eval_slogdet preserves dtype: sign keeps det_scalar_dtype; logabsdet is real
    // (complex -> real-component float). Prior hardcoded F64 widened f32 slogdet.
    let logabsdet_dtype = match input.dtype {
        DType::Complex64 => DType::F32,
        DType::Complex128 => DType::F64,
        other => det_scalar_dtype(other),
    };
    vec![
        AbstractValue {
            dtype: det_scalar_dtype(input.dtype),
            shape: Shape::scalar(),
        },
        AbstractValue {
            dtype: logabsdet_dtype,
            shape: Shape::scalar(),
        },
    ]
}

fn infer_eig_output_avals(input: &AbstractValue) -> Vec<AbstractValue> {
    // eval_eig returns (eigenvalues [n], eigenvectors [n, n]) as Complex128.
    let Some(&n) = input.shape.dims.first() else {
        return vec![input.clone(), input.clone()];
    };
    vec![
        AbstractValue {
            dtype: DType::Complex128,
            shape: Shape { dims: vec![n] },
        },
        AbstractValue {
            dtype: DType::Complex128,
            shape: Shape { dims: vec![n, n] },
        },
    ]
}

fn infer_topk_output_avals(input: &AbstractValue, eqn: &Equation) -> Vec<AbstractValue> {
    // eval_top_k returns (values: same dtype, indices: I64), both with the last
    // axis replaced by k.
    let rank = input.shape.dims.len();
    if rank == 0 {
        return vec![input.clone(); eqn.outputs.len()];
    }
    let k = eqn
        .params
        .get("k")
        .and_then(|s| s.trim().parse::<u32>().ok())
        .unwrap_or(input.shape.dims[rank - 1]);
    let mut out_dims = input.shape.dims.clone();
    out_dims[rank - 1] = k;
    vec![
        AbstractValue {
            dtype: input.dtype,
            shape: Shape {
                dims: out_dims.clone(),
            },
        },
        AbstractValue {
            dtype: DType::I64,
            shape: Shape { dims: out_dims },
        },
    ]
}

fn infer_solve_output_avals(input_avals: &[AbstractValue]) -> Vec<AbstractValue> {
    // eval_solve(A, b) -> x with the shape of b and a float-promoted dtype.
    let (Some(a), Some(b)) = (input_avals.first(), input_avals.get(1)) else {
        return input_avals.first().cloned().into_iter().collect();
    };
    let dtype = match fj_lax::promote_dtype_public(a.dtype, b.dtype) {
        dt @ (DType::F16 | DType::BF16 | DType::F32 | DType::F64) => dt,
        _ => DType::F64,
    };
    vec![AbstractValue {
        dtype,
        shape: b.shape.clone(),
    }]
}

fn abstract_value_of_literal(lit: &fj_core::Literal) -> AbstractValue {
    AbstractValue {
        dtype: match lit {
            fj_core::Literal::I32(_) => DType::I32,
            fj_core::Literal::I64(_) => DType::I64,
            fj_core::Literal::U32(_) => DType::U32,
            fj_core::Literal::U64(_) => DType::U64,
            fj_core::Literal::Bool(_) => DType::Bool,
            fj_core::Literal::BF16Bits(_) => DType::BF16,
            fj_core::Literal::F16Bits(_) => DType::F16,
            fj_core::Literal::F32Bits(_) => DType::F32,
            fj_core::Literal::F64Bits(_) => DType::F64,
            fj_core::Literal::Complex64Bits(..) => DType::Complex64,
            fj_core::Literal::Complex128Bits(..) => DType::Complex128,
        },
        shape: Shape::scalar(),
    }
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
                fj_core::Literal::I32(_) => DType::I32,
                fj_core::Literal::I64(_) => DType::I64,
                fj_core::Literal::U32(_) => DType::U32,
                fj_core::Literal::U64(_) => DType::U64,
                fj_core::Literal::Bool(_) => DType::Bool,
                fj_core::Literal::BF16Bits(_) => DType::BF16,
                fj_core::Literal::F16Bits(_) => DType::F16,
                fj_core::Literal::F32Bits(_) => DType::F32,
                fj_core::Literal::F64Bits(_) => DType::F64,
                fj_core::Literal::Complex64Bits(..) => DType::Complex64,
                fj_core::Literal::Complex128Bits(..) => DType::Complex128,
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
            std::panic::panic_any(detail);
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
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
            ],
        )
    }

    fn make_linear_add_chain_jaxpr(n: usize) -> Jaxpr {
        let mut equations = Vec::with_capacity(n);
        for i in 0..n {
            let input_var = VarId((i + 1) as u32);
            let output_var = VarId((i + 2) as u32);
            equations.push(Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(input_var), Atom::Lit(Literal::I64(1))],
                outputs: smallvec![output_var],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
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
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
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
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Abs,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Sin,
                    inputs: smallvec![Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
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
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
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
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        )
    }

    /// { c; a, b -> d = add(c, a); e = mul(c, b); f = add(d, e) -> f }
    fn make_const_mixed_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![VarId(10)],
            vec![VarId(5)],
            vec![
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(10)), Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(10)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(4))],
                    outputs: smallvec![VarId(5)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
            ],
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
                effects: vec![],
                sub_jaxprs: vec![],
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
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(4)), Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(5)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(6)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
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

    #[test]
    fn test_pe_constvars_all_known_threads_known_consts() {
        run_logged_test(
            "test_pe_constvars_all_known_threads_known_consts",
            &("pe", "constvars", "all_known"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_const_mixed_jaxpr();
                let consts = vec![Value::scalar_i64(2)];
                let pe = partial_eval_jaxpr_with_consts(&jaxpr, &consts, &[false, false]).unwrap();

                assert_eq!(pe.jaxpr_known.constvars, vec![VarId(10)]);
                assert_eq!(pe.known_consts, consts);
                assert!(pe.jaxpr_unknown.equations.is_empty());

                let known = crate::eval_jaxpr_with_consts(
                    &pe.jaxpr_known,
                    &pe.known_consts,
                    &[Value::scalar_i64(5), Value::scalar_i64(3)],
                )
                .unwrap();
                assert_eq!(known, vec![Value::scalar_i64(13)]);
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

    #[test]
    fn test_pe_constvars_all_unknown_residualizes_consts() {
        run_logged_test(
            "test_pe_constvars_all_unknown_residualizes_consts",
            &("pe", "constvars", "all_unknown"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_const_mixed_jaxpr();
                let pe =
                    partial_eval_jaxpr_with_consts(&jaxpr, &[Value::scalar_i64(2)], &[true, true])
                        .unwrap();

                assert!(pe.jaxpr_known.equations.is_empty());
                assert_eq!(pe.jaxpr_known.outvars, vec![VarId(10)]);
                assert!(pe.jaxpr_unknown.constvars.is_empty());
                assert_eq!(pe.jaxpr_unknown.invars, vec![VarId(10), VarId(1), VarId(2)]);
                assert_eq!(pe.residual_avals[0].dtype, DType::I64);

                let residuals =
                    crate::eval_jaxpr_with_consts(&pe.jaxpr_known, &pe.known_consts, &[]).unwrap();
                let mut unknown_inputs = residuals;
                unknown_inputs.extend([Value::scalar_i64(5), Value::scalar_i64(3)]);
                let staged = crate::eval_jaxpr(&pe.jaxpr_unknown, &unknown_inputs).unwrap();
                let full = crate::eval_jaxpr_with_consts(
                    &jaxpr,
                    &[Value::scalar_i64(2)],
                    &[Value::scalar_i64(5), Value::scalar_i64(3)],
                )
                .unwrap();
                assert_eq!(staged, full);
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
    fn test_pe_typed_residual_preserves_i64_dtype() {
        run_logged_test(
            "test_pe_typed_residual_preserves_i64_dtype",
            &("pe", "mixed", "typed_residual"),
            fj_test_utils::TestMode::Strict,
            || {
                // a(known, I64) -> neg(a) = v2 -> mul(v2, b(unknown)) = v3
                // Residual v2 should have I64 dtype when typed PE is used
                let jaxpr = make_neg_mul_jaxpr();
                let in_avals = vec![
                    AbstractValue {
                        dtype: DType::I64,
                        shape: Shape::scalar(),
                    },
                    AbstractValue {
                        dtype: DType::I64,
                        shape: Shape::scalar(),
                    },
                ];
                let result = partial_eval_jaxpr_typed(&jaxpr, &[false, true], Some(&in_avals))
                    .map_err(|err| err.to_string())?;
                assert!(!result.residual_avals.is_empty(), "should have residuals");
                assert_eq!(
                    result.residual_avals[0].dtype,
                    DType::I64,
                    "residual should preserve I64 dtype"
                );
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_typed_residual_comparison_produces_bool() {
        run_logged_test(
            "test_pe_typed_residual_comparison_produces_bool",
            &("pe", "mixed", "typed_comparison"),
            fj_test_utils::TestMode::Strict,
            || {
                // a(known, F64) -> eq(a, a) = v2(Bool) -> select(v2, b(unknown), b) = v3
                // Residual v2 should have Bool dtype
                let jaxpr = Jaxpr::new(
                    vec![VarId(1), VarId(2)],
                    vec![],
                    vec![VarId(4)],
                    vec![
                        Equation {
                            primitive: Primitive::Eq,
                            inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(3)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Select,
                            inputs: smallvec![
                                Atom::Var(VarId(3)),
                                Atom::Var(VarId(2)),
                                Atom::Var(VarId(2))
                            ],
                            outputs: smallvec![VarId(4)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );
                let in_avals = vec![
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape::scalar(),
                    },
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape::scalar(),
                    },
                ];
                let result = partial_eval_jaxpr_typed(&jaxpr, &[false, true], Some(&in_avals))
                    .map_err(|err| err.to_string())?;
                // v3 is produced by Eq → Bool, and consumed by Select (unknown)
                let eq_residual = result
                    .residual_avals
                    .iter()
                    .find(|a| a.dtype == DType::Bool);
                assert!(
                    eq_residual.is_some(),
                    "comparison residual should have Bool dtype, got: {:?}",
                    result.residual_avals
                );
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_constvars_mixed_residualizes_const_and_known_output() {
        run_logged_test(
            "test_pe_constvars_mixed_residualizes_const_and_known_output",
            &("pe", "constvars", "mixed"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_const_mixed_jaxpr();
                let in_avals = vec![
                    AbstractValue {
                        dtype: DType::I64,
                        shape: Shape::scalar(),
                    },
                    AbstractValue {
                        dtype: DType::I64,
                        shape: Shape::scalar(),
                    },
                ];
                let pe = partial_eval_jaxpr_typed_with_consts(
                    &jaxpr,
                    &[Value::scalar_i64(2)],
                    &[false, true],
                    Some(&in_avals),
                )
                .unwrap();

                assert_eq!(pe.jaxpr_known.equations.len(), 1);
                assert_eq!(pe.jaxpr_unknown.equations.len(), 2);
                assert_eq!(pe.jaxpr_unknown.invars, vec![VarId(10), VarId(3), VarId(2)]);
                assert_eq!(pe.residual_avals.len(), 2);
                assert!(
                    pe.residual_avals
                        .iter()
                        .all(|aval| aval.dtype == DType::I64)
                );

                let known_outs = crate::eval_jaxpr_with_consts(
                    &pe.jaxpr_known,
                    &pe.known_consts,
                    &[Value::scalar_i64(5)],
                )
                .unwrap();
                assert_eq!(known_outs, vec![Value::scalar_i64(2), Value::scalar_i64(7)]);

                let mut unknown_inputs = known_outs;
                unknown_inputs.push(Value::scalar_i64(3));
                let staged = crate::eval_jaxpr(&pe.jaxpr_unknown, &unknown_inputs).unwrap();
                let full = crate::eval_jaxpr_with_consts(
                    &jaxpr,
                    &[Value::scalar_i64(2)],
                    &[Value::scalar_i64(5), Value::scalar_i64(3)],
                )
                .unwrap();
                assert_eq!(staged, full);
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

    #[test]
    fn pe_all_known_matches_eval_for_diverse_programs() {
        use crate::eval_jaxpr;
        // Extends test_pe_staging_equivalence_all_known (Add2 only): with all inputs
        // known (unknowns = all false), eval(jaxpr_known, args) must equal
        // eval_jaxpr(jaxpr, args) for diverse ops/dtypes — the all-known partial-eval
        // path must not diverge from the eager interpreter.
        let bin = |prim: Primitive| {
            Jaxpr::new(
                vec![VarId(1), VarId(2)],
                vec![],
                vec![VarId(3)],
                vec![Equation {
                    primitive: prim,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                }],
            )
        };
        let un = |prim: Primitive| {
            Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(2)],
                vec![Equation {
                    primitive: prim,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                }],
            )
        };
        let check = |label: &str, jaxpr: &Jaxpr, args: &[Value], unknowns: &[bool]| {
            let full = eval_jaxpr(jaxpr, args).expect("eval_jaxpr");
            let pe = partial_eval_jaxpr(jaxpr, unknowns).expect("partial_eval");
            let known = eval_jaxpr(&pe.jaxpr_known, args).expect("eval jaxpr_known");
            assert_eq!(
                full[0], known[0],
                "{label}: all-known partial_eval(jaxpr_known) != eval_jaxpr"
            );
        };
        check(
            "f64_mul",
            &bin(Primitive::Mul),
            &[Value::scalar_f64(3.0), Value::scalar_f64(7.0)],
            &[false, false],
        );
        check(
            "i64_sub",
            &bin(Primitive::Sub),
            &[Value::scalar_i64(10), Value::scalar_i64(3)],
            &[false, false],
        );
        check(
            "f64_exp",
            &un(Primitive::Exp),
            &[Value::scalar_f64(1.0)],
            &[false],
        );
        check(
            "f64_sqrt",
            &un(Primitive::Sqrt),
            &[Value::scalar_f64(4.0)],
            &[false],
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
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Abs,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(3)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
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
    fn test_dce_retains_effectful_equation_with_dead_output() {
        run_logged_test(
            "test_dce_retains_effectful_equation_with_dead_output",
            &("dce", "effects", "retains_effectful"),
            fj_test_utils::TestMode::Strict,
            || {
                // eq0 (Neg, effectful) produces VarId(2), which is UNUSED; eq1 (Add)
                // produces the live output VarId(3). DCE must NOT drop the effectful eq0
                // even though its output is dead — dropping it would discard the side
                // effect. Matches JAX dce_jaxpr, which keeps any equation with effects.
                let jaxpr = Jaxpr::new(
                    vec![VarId(1)],
                    vec![],
                    vec![VarId(3)],
                    vec![
                        Equation {
                            primitive: Primitive::Neg,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(2)],
                            params: BTreeMap::new(),
                            effects: vec!["io".to_owned()],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(3)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );
                let (pruned, used_inputs) = dce_jaxpr(&jaxpr, &[true]);
                assert_eq!(
                    pruned.equations.len(),
                    2,
                    "effectful equation must be retained despite its dead output"
                );
                assert!(
                    pruned
                        .equations
                        .iter()
                        .any(|e| e.primitive == Primitive::Neg && !e.effects.is_empty()),
                    "the effectful Neg was dropped by DCE"
                );
                // The effectful equation's input is live (needed to perform the effect).
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
    fn test_dce_all_used_large_chain_golden_hash() {
        let jaxpr = make_linear_add_chain_jaxpr(1000);
        let (pruned, used_inputs) = dce_jaxpr(&jaxpr, &[true]);

        assert_eq!(pruned.invars, jaxpr.invars);
        assert_eq!(pruned.outvars, jaxpr.outvars);
        assert_eq!(pruned.equations.len(), 1000);
        assert_eq!(used_inputs, vec![true]);
        for (idx, eqn) in pruned.equations.iter().enumerate() {
            assert_eq!(eqn.primitive, Primitive::Add);
            assert_eq!(eqn.outputs.as_slice(), &[VarId((idx + 2) as u32)]);
        }

        let digest =
            fj_test_utils::fixture_id_from_json(&(&pruned, &used_inputs)).expect("DCE digest");
        assert_eq!(
            digest,
            "3729e2d5cc19c0abec46fb5b188cc7576b9853ee7d0cd523f3656b1ac57e8ad8"
        );

        let (cached_pruned, cached_used_inputs) = dce_jaxpr(&jaxpr, &[true]);
        assert_eq!(cached_pruned, pruned);
        assert_eq!(cached_used_inputs, used_inputs);
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

                let e3 = PartialEvalError::ConstArity {
                    expected: 2,
                    actual: 1,
                };
                assert!(format!("{e3}").contains("2"));
                assert!(format!("{e3}").contains("1"));
                assert!(format!("{e3}").contains("const"));

                let e4 = PartialEvalError::ResidualTypeMismatch { index: 7 };
                assert!(format!("{e4}").contains("7"));

                let e5 = PartialEvalError::ShapeInference {
                    primitive: Primitive::ReduceSum,
                    detail: "duplicate value in axes: -1".to_owned(),
                };
                assert!(format!("{e5}").contains("reduce_sum"));
                assert!(format!("{e5}").contains("duplicate"));

                // Error trait impl
                let _: &dyn std::error::Error = &e1;
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_constvars_require_const_values() {
        run_logged_test(
            "test_pe_constvars_require_const_values",
            &("pe", "constvars", "arity"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = make_const_mixed_jaxpr();
                let err = partial_eval_jaxpr(&jaxpr, &[false, true]).unwrap_err();
                assert_eq!(
                    err,
                    PartialEvalError::ConstArity {
                        expected: 1,
                        actual: 0,
                    }
                );
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
    // Category 6b: Typed PE — axis-aware reductions, Dot, shape ops
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn test_pe_typed_reduce_axis0_preserves_reduced_shape() {
        run_logged_test(
            "test_pe_typed_reduce_axis0_preserves_reduced_shape",
            &("pe", "typed", "reduce_axis0"),
            fj_test_utils::TestMode::Strict,
            || {
                // a(known, F64, [3,4]) -> reduce_sum(a, axes="0") = v2([4])
                //   -> mul(v2, b(unknown)) = v3
                // Residual v2 should have shape [4], not scalar
                let mut reduce_params = BTreeMap::new();
                reduce_params.insert("axes".to_owned(), "0".to_owned());
                let jaxpr = Jaxpr::new(
                    vec![VarId(1), VarId(2)],
                    vec![],
                    vec![VarId(4)],
                    vec![
                        Equation {
                            primitive: Primitive::ReduceSum,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(3)],
                            params: reduce_params,
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Mul,
                            inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                            outputs: smallvec![VarId(4)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );
                let in_avals = vec![
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![3, 4] },
                    },
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![4] },
                    },
                ];
                let result = partial_eval_jaxpr_typed(&jaxpr, &[false, true], Some(&in_avals))
                    .map_err(|err| err.to_string())?;
                assert!(!result.residual_avals.is_empty(), "should have residuals");
                assert_eq!(
                    result.residual_avals[0].shape.dims,
                    vec![4],
                    "axis-0 reduction of [3,4] should produce shape [4]"
                );
                assert_eq!(result.residual_avals[0].dtype, DType::F64);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_typed_reduce_axis1_preserves_reduced_shape() {
        run_logged_test(
            "test_pe_typed_reduce_axis1_preserves_reduced_shape",
            &("pe", "typed", "reduce_axis1"),
            fj_test_utils::TestMode::Strict,
            || {
                // a(known, I64, [2,3]) -> reduce_max(a, axes="1") = v2([2])
                //   -> add(v2, b(unknown)) = v3
                let mut reduce_params = BTreeMap::new();
                reduce_params.insert("axes".to_owned(), "1".to_owned());
                let jaxpr = Jaxpr::new(
                    vec![VarId(1), VarId(2)],
                    vec![],
                    vec![VarId(4)],
                    vec![
                        Equation {
                            primitive: Primitive::ReduceMax,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(3)],
                            params: reduce_params,
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                            outputs: smallvec![VarId(4)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );
                let in_avals = vec![
                    AbstractValue {
                        dtype: DType::I64,
                        shape: Shape { dims: vec![2, 3] },
                    },
                    AbstractValue {
                        dtype: DType::I64,
                        shape: Shape { dims: vec![2] },
                    },
                ];
                let result = partial_eval_jaxpr_typed(&jaxpr, &[false, true], Some(&in_avals))
                    .map_err(|err| err.to_string())?;
                assert!(!result.residual_avals.is_empty());
                assert_eq!(
                    result.residual_avals[0].shape.dims,
                    vec![2],
                    "axis-1 reduction of [2,3] should produce shape [2]"
                );
                assert_eq!(result.residual_avals[0].dtype, DType::I64);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_typed_reduce_negative_axis_preserves_reduced_shape() {
        run_logged_test(
            "test_pe_typed_reduce_negative_axis_preserves_reduced_shape",
            &("pe", "typed", "reduce_negative_axis"),
            fj_test_utils::TestMode::Strict,
            || {
                // a(known, F64, [2,3]) -> reduce_sum(a, axes="-1") = v3([2])
                //   -> add(v3, b(unknown)) = v4
                let mut reduce_params = BTreeMap::new();
                reduce_params.insert("axes".to_owned(), "-1".to_owned());
                let jaxpr = Jaxpr::new(
                    vec![VarId(1), VarId(2)],
                    vec![],
                    vec![VarId(4)],
                    vec![
                        Equation {
                            primitive: Primitive::ReduceSum,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(3)],
                            params: reduce_params,
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                            outputs: smallvec![VarId(4)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );
                let in_avals = vec![
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![2, 3] },
                    },
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![2] },
                    },
                ];
                let result = partial_eval_jaxpr_typed(&jaxpr, &[false, true], Some(&in_avals))
                    .map_err(|err| err.to_string())?;
                assert_eq!(
                    result.residual_avals,
                    vec![AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![2] },
                    }],
                    "axis -1 reduction of [2,3] should produce shape [2]"
                );
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_typed_bitwise_reduce_axis_preserves_reduced_shape() {
        run_logged_test(
            "test_pe_typed_bitwise_reduce_axis_preserves_reduced_shape",
            &("pe", "typed", "bitwise_reduce_axis"),
            fj_test_utils::TestMode::Strict,
            || {
                // a(known, Bool, [2,3]) -> reduce_or(a, axes="1") = v3([2])
                //   -> eq(v3, b(unknown)) = v4
                let mut reduce_params = BTreeMap::new();
                reduce_params.insert("axes".to_owned(), "1".to_owned());
                let jaxpr = Jaxpr::new(
                    vec![VarId(1), VarId(2)],
                    vec![],
                    vec![VarId(4)],
                    vec![
                        Equation {
                            primitive: Primitive::ReduceOr,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(3)],
                            params: reduce_params,
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Eq,
                            inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                            outputs: smallvec![VarId(4)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );
                let in_avals = vec![
                    AbstractValue {
                        dtype: DType::Bool,
                        shape: Shape { dims: vec![2, 3] },
                    },
                    AbstractValue {
                        dtype: DType::Bool,
                        shape: Shape { dims: vec![2] },
                    },
                ];
                let result = partial_eval_jaxpr_typed(&jaxpr, &[false, true], Some(&in_avals))
                    .map_err(|err| err.to_string())?;
                assert_eq!(
                    result.residual_avals,
                    vec![AbstractValue {
                        dtype: DType::Bool,
                        shape: Shape { dims: vec![2] },
                    }],
                    "bitwise axis reduction of [2,3] should produce shape [2]"
                );
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_typed_reduce_rejects_duplicate_axes_after_normalization() {
        run_logged_test(
            "test_pe_typed_reduce_rejects_duplicate_axes_after_normalization",
            &("pe", "typed", "reduce_duplicate_axes"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut reduce_params = BTreeMap::new();
                reduce_params.insert("axes".to_owned(), "1,-1".to_owned());
                let jaxpr = Jaxpr::new(
                    vec![VarId(1), VarId(2)],
                    vec![],
                    vec![VarId(4)],
                    vec![
                        Equation {
                            primitive: Primitive::ReduceSum,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(3)],
                            params: reduce_params,
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                            outputs: smallvec![VarId(4)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );
                let in_avals = vec![
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![2, 3] },
                    },
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![2] },
                    },
                ];

                let err = match partial_eval_jaxpr_typed(&jaxpr, &[false, true], Some(&in_avals)) {
                    Ok(result) => {
                        return Err(format!(
                            "expected duplicate-axis error, got result: {result:?}"
                        ));
                    }
                    Err(err) => err,
                };
                match err {
                    PartialEvalError::ShapeInference { primitive, detail } => {
                        assert_eq!(primitive, Primitive::ReduceSum);
                        assert!(
                            detail.contains("duplicate value in axes"),
                            "unexpected detail: {detail}"
                        );
                    }
                    other => return Err(format!("unexpected error: {other}")),
                }
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_typed_full_reduce_still_produces_scalar() {
        run_logged_test(
            "test_pe_typed_full_reduce_still_produces_scalar",
            &("pe", "typed", "full_reduce"),
            fj_test_utils::TestMode::Strict,
            || {
                // No axes param: full reduction to scalar (backwards compat)
                let jaxpr = Jaxpr::new(
                    vec![VarId(1), VarId(2)],
                    vec![],
                    vec![VarId(4)],
                    vec![
                        Equation {
                            primitive: Primitive::ReduceSum,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(3)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                            outputs: smallvec![VarId(4)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );
                let in_avals = vec![
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![3, 4] },
                    },
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape::scalar(),
                    },
                ];
                let result =
                    partial_eval_jaxpr_typed(&jaxpr, &[false, true], Some(&in_avals)).unwrap();
                assert!(!result.residual_avals.is_empty());
                assert!(
                    result.residual_avals[0].shape.dims.is_empty(),
                    "full reduction should produce scalar shape"
                );
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_typed_dot_produces_scalar() {
        run_logged_test(
            "test_pe_typed_dot_produces_scalar",
            &("pe", "typed", "dot_scalar"),
            fj_test_utils::TestMode::Strict,
            || {
                // a(known, F64, [3]) dot b_known -> v2(scalar) -> add(v2, c(unknown)) = v3
                let jaxpr = Jaxpr::new(
                    vec![VarId(1), VarId(2), VarId(5)],
                    vec![],
                    vec![VarId(4)],
                    vec![
                        Equation {
                            primitive: Primitive::Dot,
                            inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                            outputs: smallvec![VarId(3)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(5))],
                            outputs: smallvec![VarId(4)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );
                let in_avals = vec![
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape::vector(3),
                    },
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape::vector(3),
                    },
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape::scalar(),
                    },
                ];
                let result =
                    partial_eval_jaxpr_typed(&jaxpr, &[false, false, true], Some(&in_avals))
                        .unwrap();
                assert!(!result.residual_avals.is_empty());
                assert!(
                    result.residual_avals[0].shape.dims.is_empty(),
                    "dot product should produce scalar residual"
                );
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_typed_reshape_propagates_new_shape() {
        run_logged_test(
            "test_pe_typed_reshape_propagates_new_shape",
            &("pe", "typed", "reshape"),
            fj_test_utils::TestMode::Strict,
            || {
                // a(known, F64, [6]) -> reshape(a, new_shape="2,3") = v2([2,3])
                //   -> add(v2, b(unknown)) = v3
                let mut reshape_params = BTreeMap::new();
                reshape_params.insert("new_shape".to_owned(), "2,3".to_owned());
                let jaxpr = Jaxpr::new(
                    vec![VarId(1), VarId(2)],
                    vec![],
                    vec![VarId(4)],
                    vec![
                        Equation {
                            primitive: Primitive::Reshape,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(3)],
                            params: reshape_params,
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                            outputs: smallvec![VarId(4)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );
                let in_avals = vec![
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape::vector(6),
                    },
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![2, 3] },
                    },
                ];
                let result =
                    partial_eval_jaxpr_typed(&jaxpr, &[false, true], Some(&in_avals)).unwrap();
                assert!(!result.residual_avals.is_empty());
                assert_eq!(
                    result.residual_avals[0].shape.dims,
                    vec![2, 3],
                    "reshape residual should have new shape [2,3]"
                );
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_typed_slice_computes_shape_from_params() {
        run_logged_test(
            "test_pe_typed_slice_computes_shape_from_params",
            &("pe", "typed", "slice"),
            fj_test_utils::TestMode::Strict,
            || {
                // a(known, I64, [4,6]) -> slice(a, start="1,2", limit="3,5") = v2([2,3])
                //   -> add(v2, b(unknown)) = v3
                let mut slice_params = BTreeMap::new();
                slice_params.insert("start_indices".to_owned(), "1,2".to_owned());
                slice_params.insert("limit_indices".to_owned(), "3,5".to_owned());
                let jaxpr = Jaxpr::new(
                    vec![VarId(1), VarId(2)],
                    vec![],
                    vec![VarId(4)],
                    vec![
                        Equation {
                            primitive: Primitive::Slice,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(3)],
                            params: slice_params,
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                            outputs: smallvec![VarId(4)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );
                let in_avals = vec![
                    AbstractValue {
                        dtype: DType::I64,
                        shape: Shape { dims: vec![4, 6] },
                    },
                    AbstractValue {
                        dtype: DType::I64,
                        shape: Shape { dims: vec![2, 3] },
                    },
                ];
                let result =
                    partial_eval_jaxpr_typed(&jaxpr, &[false, true], Some(&in_avals)).unwrap();
                assert!(!result.residual_avals.is_empty());
                assert_eq!(
                    result.residual_avals[0].shape.dims,
                    vec![2, 3],
                    "slice [1:3, 2:5] should produce shape [2,3]"
                );
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_typed_transpose_permutes_shape() {
        run_logged_test(
            "test_pe_typed_transpose_permutes_shape",
            &("pe", "typed", "transpose"),
            fj_test_utils::TestMode::Strict,
            || {
                // a(known, F64, [2,3]) -> transpose(a, perm="1,0") = v2([3,2])
                //   -> add(v2, b(unknown)) = v3
                let mut transpose_params = BTreeMap::new();
                transpose_params.insert("permutation".to_owned(), "1,0".to_owned());
                let jaxpr = Jaxpr::new(
                    vec![VarId(1), VarId(2)],
                    vec![],
                    vec![VarId(4)],
                    vec![
                        Equation {
                            primitive: Primitive::Transpose,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(3)],
                            params: transpose_params,
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                            outputs: smallvec![VarId(4)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );
                let in_avals = vec![
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![2, 3] },
                    },
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![3, 2] },
                    },
                ];
                let result =
                    partial_eval_jaxpr_typed(&jaxpr, &[false, true], Some(&in_avals)).unwrap();
                assert!(!result.residual_avals.is_empty());
                assert_eq!(
                    result.residual_avals[0].shape.dims,
                    vec![3, 2],
                    "transpose [2,3] with perm [1,0] should produce [3,2]"
                );
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_typed_dot_general_contracts_to_matmul_shape() {
        run_logged_test(
            "test_pe_typed_dot_general_contracts_to_matmul_shape",
            &("pe", "typed", "dot_general"),
            fj_test_utils::TestMode::Strict,
            || {
                // lhs(known [2,3]) ·_{1,0} rhs(known [3,4]) = v4([2,4], known) ->
                //   add(v4, c(unknown [2,4])) = v5. The dot output is residualized
                //   into the unknown partition; its aval must be [2,4], not the LHS
                //   [2,3] the single-input catch-all used to return.
                let mut dot_params = BTreeMap::new();
                dot_params.insert("lhs_contracting_dims".to_owned(), "1".to_owned());
                dot_params.insert("rhs_contracting_dims".to_owned(), "0".to_owned());
                let jaxpr = Jaxpr::new(
                    vec![VarId(1), VarId(2), VarId(3)],
                    vec![],
                    vec![VarId(5)],
                    vec![
                        Equation {
                            primitive: Primitive::DotGeneral,
                            inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                            outputs: smallvec![VarId(4)],
                            params: dot_params,
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![Atom::Var(VarId(4)), Atom::Var(VarId(3))],
                            outputs: smallvec![VarId(5)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );
                let in_avals = vec![
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![2, 3] },
                    },
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![3, 4] },
                    },
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![2, 4] },
                    },
                ];
                let result =
                    partial_eval_jaxpr_typed(&jaxpr, &[false, false, true], Some(&in_avals))
                        .unwrap();
                assert!(!result.residual_avals.is_empty());
                assert_eq!(
                    result.residual_avals[0].shape.dims,
                    vec![2, 4],
                    "dot_general [2,3]·[3,4] contracting (1,0) should produce [2,4]"
                );
                assert_eq!(result.residual_avals[0].dtype, DType::F64);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_typed_argsort_residual_is_i64() {
        run_logged_test(
            "test_pe_typed_argsort_residual_is_i64",
            &("pe", "typed", "argsort"),
            fj_test_utils::TestMode::Strict,
            || {
                // argsort(a(known, F64, [5])) = v3(I64, known) -> add(v3, b(unknown,
                // I64, [5])) = v4. The residualized argsort output must be I64 index
                // data; the catch-all used to copy the input's F64 dtype.
                let mut sort_params = BTreeMap::new();
                sort_params.insert("axis".to_owned(), "0".to_owned());
                let jaxpr = Jaxpr::new(
                    vec![VarId(1), VarId(2)],
                    vec![],
                    vec![VarId(4)],
                    vec![
                        Equation {
                            primitive: Primitive::Argsort,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(3)],
                            params: sort_params,
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                            outputs: smallvec![VarId(4)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );
                let in_avals = vec![
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![5] },
                    },
                    AbstractValue {
                        dtype: DType::I64,
                        shape: Shape { dims: vec![5] },
                    },
                ];
                let result =
                    partial_eval_jaxpr_typed(&jaxpr, &[false, true], Some(&in_avals)).unwrap();
                assert!(!result.residual_avals.is_empty());
                assert_eq!(
                    result.residual_avals[0].dtype,
                    DType::I64,
                    "argsort output is I64 index data, not the input's F64"
                );
                assert_eq!(result.residual_avals[0].shape.dims, vec![5]);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_typed_reduce_window_delegates_window_geometry() {
        run_logged_test(
            "test_pe_typed_reduce_window_delegates_window_geometry",
            &("pe", "typed", "reduce_window"),
            fj_test_utils::TestMode::Strict,
            || {
                // reduce_window(a(known F64 [4,4]), window 2x2 stride 2x2 VALID)
                //   = v3([2,2], known) -> add(v3, b(unknown [2,2])) = v4. The window
                // geometry is delegated to fj-trace's authoritative inference; the
                // single-input catch-all used to keep the [4,4] input shape.
                let mut rw_params = BTreeMap::new();
                rw_params.insert("window_dimensions".to_owned(), "2,2".to_owned());
                rw_params.insert("window_strides".to_owned(), "2,2".to_owned());
                rw_params.insert("padding".to_owned(), "valid".to_owned());
                let jaxpr = Jaxpr::new(
                    vec![VarId(1), VarId(2)],
                    vec![],
                    vec![VarId(4)],
                    vec![
                        Equation {
                            primitive: Primitive::ReduceWindow,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(3)],
                            params: rw_params,
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                            outputs: smallvec![VarId(4)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );
                let in_avals = vec![
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![4, 4] },
                    },
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![2, 2] },
                    },
                ];
                let result =
                    partial_eval_jaxpr_typed(&jaxpr, &[false, true], Some(&in_avals)).unwrap();
                assert!(!result.residual_avals.is_empty());
                assert_eq!(
                    result.residual_avals[0].shape.dims,
                    vec![2, 2],
                    "reduce_window 2x2/stride2 VALID over [4,4] should produce [2,2]"
                );
                Ok(vec![])
            },
        );
    }

    #[test]
    fn delegation_stays_live_for_common_staged_ops() {
        // GUARD for the structural single-source-of-truth (commits 3ac6dab0 /
        // 8b46a23d): partial_eval now delegates residual aval inference to
        // fj-trace's authoritative `infer_output_avals` for EVERY op, falling back
        // to the local best-effort arms ONLY when fj-trace returns Err. If fj-trace
        // ever regresses to erroring on a COMMON op, delegation would silently drop
        // to the crude fallback and the two-layers drift could creep back. This
        // asserts delegation returns Some (i.e. fj-trace handled it) for a
        // representative op of each arity/shape class, with a couple of exact-shape
        // spot checks — so a silent fallback regression fails loudly here.
        let av = |dtype: DType, dims: Vec<u32>| AbstractValue {
            dtype,
            shape: Shape { dims },
        };
        let eqn = |primitive: Primitive, params: &[(&str, &str)]| Equation {
            primitive,
            inputs: smallvec![],
            outputs: smallvec![VarId(0)],
            params: params
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                .collect(),
            effects: vec![],
            sub_jaxprs: vec![],
        };
        let f64v = |dims: Vec<u32>| av(DType::F64, dims);

        // (label, eqn, input avals) — every one must delegate (Some).
        let cases: Vec<(&str, Equation, Vec<AbstractValue>)> = vec![
            ("exp", eqn(Primitive::Exp, &[]), vec![f64v(vec![3])]),
            (
                "add",
                eqn(Primitive::Add, &[]),
                vec![f64v(vec![3]), f64v(vec![3])],
            ),
            (
                "reduce_sum",
                eqn(Primitive::ReduceSum, &[("axes", "0")]),
                vec![f64v(vec![2, 3])],
            ),
            (
                "reshape",
                eqn(Primitive::Reshape, &[("new_shape", "6")]),
                vec![f64v(vec![2, 3])],
            ),
            (
                "transpose",
                eqn(Primitive::Transpose, &[("permutation", "1,0")]),
                vec![f64v(vec![2, 3])],
            ),
            (
                "expand_dims",
                eqn(Primitive::ExpandDims, &[("axis", "-1")]),
                vec![f64v(vec![3])],
            ),
            (
                "argsort",
                eqn(Primitive::Argsort, &[("axis", "0")]),
                vec![f64v(vec![5])],
            ),
        ];
        for (label, equation, avals) in &cases {
            assert!(
                delegate_infer_to_trace(equation, avals).is_some(),
                "delegation to fj-trace must stay live for '{label}' (silent fallback = drift risk)"
            );
        }

        // Exact spot checks: delegation produces the authoritative shapes/dtypes.
        let argsort = delegate_infer_to_trace(&cases[6].1, &cases[6].2).unwrap();
        assert_eq!(argsort[0].dtype, DType::I64, "argsort indices are I64");
        let dot = delegate_infer_to_trace(
            &eqn(
                Primitive::DotGeneral,
                &[("lhs_contracting_dims", "1"), ("rhs_contracting_dims", "0")],
            ),
            &[f64v(vec![2, 3]), f64v(vec![3, 4])],
        )
        .expect("dot_general delegates");
        assert_eq!(
            dot[0].shape.dims,
            vec![2, 4],
            "dot_general [2,3]·[3,4]→[2,4]"
        );
    }

    #[test]
    fn test_pe_typed_broadcast_in_dim_uses_target_shape() {
        run_logged_test(
            "test_pe_typed_broadcast_in_dim_uses_target_shape",
            &("pe", "typed", "broadcast"),
            fj_test_utils::TestMode::Strict,
            || {
                // a(known, F64, [3]) -> broadcast_in_dim(a, shape="2,3") = v2([2,3])
                //   -> add(v2, b(unknown)) = v3
                let mut bcast_params = BTreeMap::new();
                bcast_params.insert("shape".to_owned(), "2,3".to_owned());
                bcast_params.insert("broadcast_dimensions".to_owned(), "1".to_owned());
                let jaxpr = Jaxpr::new(
                    vec![VarId(1), VarId(2)],
                    vec![],
                    vec![VarId(4)],
                    vec![
                        Equation {
                            primitive: Primitive::BroadcastInDim,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(3)],
                            params: bcast_params,
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                            outputs: smallvec![VarId(4)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );
                let in_avals = vec![
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape::vector(3),
                    },
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![2, 3] },
                    },
                ];
                let result =
                    partial_eval_jaxpr_typed(&jaxpr, &[false, true], Some(&in_avals)).unwrap();
                assert!(!result.residual_avals.is_empty());
                assert_eq!(
                    result.residual_avals[0].shape.dims,
                    vec![2, 3],
                    "broadcast_in_dim should use target shape [2,3]"
                );
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_typed_svd_singular_values_keep_vector_shape() {
        run_logged_test(
            "test_pe_typed_svd_singular_values_keep_vector_shape",
            &("pe", "typed", "svd_s"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = Jaxpr::new(
                    vec![VarId(1), VarId(5)],
                    vec![],
                    vec![VarId(6)],
                    vec![
                        Equation {
                            primitive: Primitive::Svd,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(2), VarId(3), VarId(4)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(5))],
                            outputs: smallvec![VarId(6)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );
                let in_avals = vec![
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![3, 2] },
                    },
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![2] },
                    },
                ];
                let result =
                    partial_eval_jaxpr_typed(&jaxpr, &[false, true], Some(&in_avals)).unwrap();
                assert_eq!(
                    result.residual_avals,
                    vec![AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![2] },
                    }],
                    "SVD singular values should stage as a length-k vector residual"
                );
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_typed_qr_r_output_keeps_factor_shape() {
        run_logged_test(
            "test_pe_typed_qr_r_output_keeps_factor_shape",
            &("pe", "typed", "qr_r"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = Jaxpr::new(
                    vec![VarId(1), VarId(4)],
                    vec![],
                    vec![VarId(5)],
                    vec![
                        Equation {
                            primitive: Primitive::Qr,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(2), VarId(3)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(4))],
                            outputs: smallvec![VarId(5)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );
                let in_avals = vec![
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![3, 2] },
                    },
                    AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![2, 2] },
                    },
                ];
                let result =
                    partial_eval_jaxpr_typed(&jaxpr, &[false, true], Some(&in_avals)).unwrap();
                assert_eq!(
                    result.residual_avals,
                    vec![AbstractValue {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![2, 2] },
                    }],
                    "QR residual should preserve R's k-by-n shape"
                );
                Ok(vec![])
            },
        );
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

    // ── Diamond DAG and dependency chain edge cases (frankenjax-8u3) ──

    #[test]
    fn test_pe_diamond_dag_known_unknown_split() {
        // Diamond: v3 = neg(x), v4 = abs(x), v5 = add(v3, v4)
        // When x is unknown, all equations should be in jaxpr_unknown
        run_logged_test(
            "test_pe_diamond_dag_known_unknown_split",
            &("pe", "edge", "diamond_unknown"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = Jaxpr::new(
                    vec![VarId(1)],
                    vec![],
                    vec![VarId(4)],
                    vec![
                        Equation {
                            primitive: Primitive::Neg,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(2)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Abs,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(3)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(3))],
                            outputs: smallvec![VarId(4)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );

                // x unknown (true=unknown): all equations should be residual
                let result = partial_eval_jaxpr(&jaxpr, &[true]).unwrap();
                assert_eq!(
                    result.jaxpr_known.equations.len(),
                    0,
                    "no known equations when input is unknown"
                );
                assert_eq!(
                    result.jaxpr_unknown.equations.len(),
                    3,
                    "all 3 equations should be in unknown jaxpr"
                );
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_diamond_dag_known_input() {
        // Same diamond DAG but with known input — all should constant-fold
        run_logged_test(
            "test_pe_diamond_dag_known_input",
            &("pe", "edge", "diamond_known"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = Jaxpr::new(
                    vec![VarId(1)],
                    vec![],
                    vec![VarId(4)],
                    vec![
                        Equation {
                            primitive: Primitive::Neg,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(2)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Abs,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(3)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(3))],
                            outputs: smallvec![VarId(4)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );

                // x known (false=known): all equations should be in jaxpr_known
                let result = partial_eval_jaxpr(&jaxpr, &[false]).unwrap();
                assert_eq!(
                    result.jaxpr_known.equations.len(),
                    3,
                    "all equations should be known-folded"
                );
                assert_eq!(
                    result.jaxpr_unknown.equations.len(),
                    0,
                    "no unknown equations"
                );
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_mixed_known_unknown_residual_chain() {
        // Two inputs: a (known), b (unknown)
        // v3 = neg(a)   → known
        // v4 = add(v3, b) → unknown (depends on known residual + unknown input)
        // Residual chain: v3 flows from known to unknown
        run_logged_test(
            "test_pe_mixed_known_unknown_residual_chain",
            &("pe", "edge", "residual_chain"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = Jaxpr::new(
                    vec![VarId(1), VarId(2)],
                    vec![],
                    vec![VarId(4)],
                    vec![
                        Equation {
                            primitive: Primitive::Neg,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(3)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                            outputs: smallvec![VarId(4)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );

                // a=known (false), b=unknown (true)
                let result = partial_eval_jaxpr(&jaxpr, &[false, true]).unwrap();
                assert_eq!(result.jaxpr_known.equations.len(), 1, "neg(a) is known");
                assert_eq!(
                    result.jaxpr_unknown.equations.len(),
                    1,
                    "add(v3, b) is unknown"
                );
                assert!(
                    !result.residual_avals.is_empty(),
                    "v3 should be a residual flowing from known to unknown"
                );
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_two_eq_mixed_residual_fast_path_golden() {
        run_logged_test(
            "test_pe_two_eq_mixed_residual_fast_path_golden",
            &("pe", "perf", "two_eq_mixed_residual"),
            fj_test_utils::TestMode::Strict,
            || {
                let known = VarId(1);
                let unknown = VarId(2);
                let residual = VarId(3);
                let out = VarId(4);
                let jaxpr = Jaxpr::new(
                    vec![known, unknown],
                    vec![],
                    vec![out],
                    vec![
                        Equation {
                            primitive: Primitive::Neg,
                            inputs: smallvec![Atom::Var(known)],
                            outputs: smallvec![residual],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Mul,
                            inputs: smallvec![Atom::Var(residual), Atom::Var(unknown)],
                            outputs: smallvec![out],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );

                let result = partial_eval_jaxpr(&jaxpr, &[false, true]).unwrap();
                assert_eq!(result.jaxpr_known.invars, vec![known]);
                assert!(result.jaxpr_known.constvars.is_empty());
                assert_eq!(result.jaxpr_known.outvars, vec![residual]);
                assert_eq!(result.jaxpr_known.equations.len(), 1);
                assert_eq!(result.jaxpr_known.equations[0].primitive, Primitive::Neg);
                assert_eq!(result.known_consts, Vec::<Value>::new());
                assert_eq!(result.jaxpr_unknown.invars, vec![residual, unknown]);
                assert!(result.jaxpr_unknown.constvars.is_empty());
                assert_eq!(result.jaxpr_unknown.outvars, vec![out]);
                assert_eq!(result.jaxpr_unknown.equations.len(), 1);
                assert_eq!(result.jaxpr_unknown.equations[0].primitive, Primitive::Mul);
                assert_eq!(result.out_unknowns, vec![true]);
                assert_eq!(
                    result.residual_avals,
                    vec![AbstractValue {
                        dtype: DType::F64,
                        shape: Shape::scalar(),
                    }]
                );

                let digest = fj_test_utils::fixture_id_from_json(&(
                    "frankenjax-4kwjw",
                    &result.jaxpr_known,
                    &result.jaxpr_unknown,
                    &result.out_unknowns,
                    &result.residual_avals,
                ))
                .expect("two-equation PE golden rows should hash");
                eprintln!("two-equation mixed partial-eval golden digest: {digest}");
                assert_eq!(
                    digest,
                    "f51e1a62763e23c83ec7a1433ef7e3ec3e1e9122a78edcc2559f1e5d4f97e88d"
                );
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_deep_dependency_chain_all_unknown() {
        // 5-deep chain: v2=neg(x), v3=abs(v2), v4=neg(v3), v5=abs(v4), v6=neg(v5)
        // All unknown when x is unknown
        run_logged_test(
            "test_pe_deep_dependency_chain_all_unknown",
            &("pe", "edge", "deep_chain_unknown"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = Jaxpr::new(
                    vec![VarId(1)],
                    vec![],
                    vec![VarId(6)],
                    vec![
                        Equation {
                            primitive: Primitive::Neg,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(2)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Abs,
                            inputs: smallvec![Atom::Var(VarId(2))],
                            outputs: smallvec![VarId(3)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Neg,
                            inputs: smallvec![Atom::Var(VarId(3))],
                            outputs: smallvec![VarId(4)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Abs,
                            inputs: smallvec![Atom::Var(VarId(4))],
                            outputs: smallvec![VarId(5)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Neg,
                            inputs: smallvec![Atom::Var(VarId(5))],
                            outputs: smallvec![VarId(6)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );

                let result = partial_eval_jaxpr(&jaxpr, &[true]).unwrap();
                assert_eq!(result.jaxpr_known.equations.len(), 0);
                assert_eq!(result.jaxpr_unknown.equations.len(), 5);
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_literal_mixed_with_unknown_variable() {
        // v3 = add(42, x) where x is unknown — literal is always known
        // The equation should be in unknown because x is unknown
        run_logged_test(
            "test_pe_literal_mixed_with_unknown",
            &("pe", "edge", "literal_unknown_mix"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = Jaxpr::new(
                    vec![VarId(1)],
                    vec![],
                    vec![VarId(2)],
                    vec![Equation {
                        primitive: Primitive::Add,
                        inputs: smallvec![
                            Atom::Lit(fj_core::Literal::I64(42)),
                            Atom::Var(VarId(1))
                        ],
                        outputs: smallvec![VarId(2)],
                        params: BTreeMap::new(),
                        effects: vec![],
                        sub_jaxprs: vec![],
                    }],
                );

                let result = partial_eval_jaxpr(&jaxpr, &[true]).unwrap();
                assert_eq!(
                    result.jaxpr_unknown.equations.len(),
                    1,
                    "literal + unknown → unknown equation"
                );
                Ok(vec![])
            },
        );
    }

    #[test]
    fn test_pe_semantic_equivalence_diamond() {
        // Verify that splitting and re-evaluating a diamond DAG gives the same result
        run_logged_test(
            "test_pe_semantic_equivalence_diamond",
            &("pe", "edge", "diamond_semantic"),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = Jaxpr::new(
                    vec![VarId(1), VarId(2)],
                    vec![],
                    vec![VarId(5)],
                    vec![
                        Equation {
                            primitive: Primitive::Neg,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(3)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Abs,
                            inputs: smallvec![Atom::Var(VarId(2))],
                            outputs: smallvec![VarId(4)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(4))],
                            outputs: smallvec![VarId(5)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                    ],
                );

                let a = Value::scalar_f64(3.0);
                let b = Value::scalar_f64(-7.0);

                // Full evaluation
                let full = crate::eval_jaxpr(&jaxpr, &[a.clone(), b.clone()]).unwrap();

                // Partial eval: a=known (false), b=unknown (true)
                let pe = partial_eval_jaxpr(&jaxpr, &[false, true]).unwrap();
                let known_outs =
                    crate::eval_jaxpr_with_consts(&pe.jaxpr_known, &pe.known_consts, &[a]).unwrap();
                let mut unk_inputs = known_outs;
                unk_inputs.push(b);
                let staged = crate::eval_jaxpr(&pe.jaxpr_unknown, &unk_inputs).unwrap();

                assert_eq!(
                    full, staged,
                    "partial eval split-and-recombine must match full eval"
                );
                Ok(vec![])
            },
        );
    }

    #[test]
    fn infer_shape_changing_ops_residual_avals() {
        // Guards the partial_eval residual-typing path (the two-layers class):
        // shape/dtype-changing ops must not fall through to the catch-all, which
        // returned the INPUT aval and silently mistyped staged residuals.
        fn av(dims: &[u32], dtype: DType) -> AbstractValue {
            AbstractValue {
                dtype,
                shape: Shape {
                    dims: dims.to_vec(),
                },
            }
        }
        fn eqn(prim: Primitive, params: &[(&str, &str)]) -> Equation {
            let mut p = BTreeMap::new();
            for (k, v) in params {
                p.insert((*k).to_string(), (*v).to_string());
            }
            Equation {
                primitive: prim,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: p,
                effects: vec![],
                sub_jaxprs: vec![],
            }
        }

        // ExpandDims axis=-1 on [2,3] -> [2,3,1] (catch-all wrongly kept [2,3]).
        let out = infer_equation_output_aval(
            &eqn(Primitive::ExpandDims, &[("axis", "-1")]),
            &av(&[2, 3], DType::F64),
        )
        .unwrap();
        assert_eq!(out.shape.dims, vec![2, 3, 1]);
        assert_eq!(out.dtype, DType::F64);

        // Squeeze dimensions=1 on [2,1,3] -> [2,3]; default (all size-1) -> drops.
        let out = infer_equation_output_aval(
            &eqn(Primitive::Squeeze, &[("dimensions", "1")]),
            &av(&[2, 1, 3], DType::F64),
        )
        .unwrap();
        assert_eq!(out.shape.dims, vec![2, 3]);
        let out =
            infer_equation_output_aval(&eqn(Primitive::Squeeze, &[]), &av(&[1, 4, 1], DType::F64))
                .unwrap();
        assert_eq!(out.shape.dims, vec![4]);

        // Argmax/Argmin: drop the axis AND retype to I64 (catch-all kept [4,3]/F64).
        let out = infer_equation_output_aval(
            &eqn(Primitive::Argmax, &[("axis", "0")]),
            &av(&[4, 3], DType::F64),
        )
        .unwrap();
        assert_eq!(out.shape.dims, vec![3]);
        assert_eq!(out.dtype, DType::I64);
        let out =
            infer_equation_output_aval(&eqn(Primitive::Argmin, &[]), &av(&[4, 3], DType::F64))
                .unwrap();
        assert_eq!(out.shape.dims, vec![4]);
        assert_eq!(out.dtype, DType::I64);

        // ConvertElementType: dtype from "new_dtype", shape preserved (catch-all
        // wrongly kept the input dtype).
        let out = infer_equation_output_aval(
            &eqn(Primitive::ConvertElementType, &[("new_dtype", "i32")]),
            &av(&[2, 3], DType::F64),
        )
        .unwrap();
        assert_eq!(out.dtype, DType::I32);
        assert_eq!(out.shape.dims, vec![2, 3]);

        // Real of Complex128 -> F64 (shape preserved).
        let out =
            infer_equation_output_aval(&eqn(Primitive::Real, &[]), &av(&[4], DType::Complex128))
                .unwrap();
        assert_eq!(out.dtype, DType::F64);
        assert_eq!(out.shape.dims, vec![4]);

        // Imag of Complex64 -> F32.
        let out =
            infer_equation_output_aval(&eqn(Primitive::Imag, &[]), &av(&[2], DType::Complex64))
                .unwrap();
        assert_eq!(out.dtype, DType::F32);

        // Unary predicates -> Bool, shape preserved.
        for prim in [
            Primitive::IsFinite,
            Primitive::IsNan,
            Primitive::IsInf,
            Primitive::Signbit,
        ] {
            let out =
                infer_equation_output_aval(&eqn(prim, &[]), &av(&[2, 2], DType::F32)).unwrap();
            assert_eq!(out.dtype, DType::Bool, "{prim:?} must be Bool");
            assert_eq!(out.shape.dims, vec![2, 2]);
        }

        // Tile [2,3] reps=2,3 -> [4,9]; scalar reps=4 -> [4].
        let out = infer_equation_output_aval(
            &eqn(Primitive::Tile, &[("reps", "2,3")]),
            &av(&[2, 3], DType::F64),
        )
        .unwrap();
        assert_eq!(out.shape.dims, vec![4, 9]);
        let out = infer_equation_output_aval(
            &eqn(Primitive::Tile, &[("reps", "4")]),
            &av(&[], DType::F64),
        )
        .unwrap();
        assert_eq!(out.shape.dims, vec![4]);
        let out = infer_equation_output_aval(
            &eqn(Primitive::Tile, &[("reps", "2")]),
            &av(&[2, 3], DType::F64),
        )
        .unwrap();
        assert_eq!(out.shape.dims, vec![2, 6]);
        let out = infer_equation_output_aval(
            &eqn(Primitive::Tile, &[("reps", "3,2")]),
            &av(&[2], DType::F64),
        )
        .unwrap();
        assert_eq!(out.shape.dims, vec![3, 4]);
        let out = infer_equation_output_aval(
            &eqn(Primitive::Tile, &[("reps", "0,2")]),
            &av(&[2, 3], DType::F64),
        )
        .unwrap();
        assert_eq!(out.shape.dims, vec![0, 6]);

        // Reshape fallback honors inferred dimensions and zero-sized shapes
        // instead of dropping non-u32 tokens like "-1".
        let out = infer_equation_output_aval(
            &eqn(Primitive::Reshape, &[("new_shape", "2,-1")]),
            &av(&[2, 3], DType::F64),
        )
        .unwrap();
        assert_eq!(out.shape.dims, vec![2, 3]);
        let out = infer_equation_output_aval(
            &eqn(Primitive::Reshape, &[("new_shape", "0,3")]),
            &av(&[0], DType::F64),
        )
        .unwrap();
        assert_eq!(out.shape.dims, vec![0, 3]);
        let err = infer_equation_output_aval(
            &eqn(Primitive::Reshape, &[("new_shape", "2,-1,-1")]),
            &av(&[2, 3], DType::F64),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            PartialEvalError::ShapeInference {
                primitive: Primitive::Reshape,
                ..
            }
        ));
        let err = infer_equation_output_aval(
            &eqn(Primitive::Reshape, &[("new_shape", "4,2")]),
            &av(&[2, 3], DType::F64),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            PartialEvalError::ShapeInference {
                primitive: Primitive::Reshape,
                ..
            }
        ));

        // Slice fallback honors strides and rejects malformed params instead of
        // dropping bad tokens or keeping the input shape.
        let out = infer_equation_output_aval(
            &eqn(
                Primitive::Slice,
                &[
                    ("start_indices", "1"),
                    ("limit_indices", "8"),
                    ("strides", "2"),
                ],
            ),
            &av(&[10], DType::F64),
        )
        .unwrap();
        assert_eq!(out.shape.dims, vec![4]);

        for (label, params, input, expected_detail) in [
            (
                "bad slice start token",
                &[("start_indices", "0,bad"), ("limit_indices", "2,3")][..],
                av(&[4, 6], DType::F64),
                "invalid u32 in param 'start_indices'",
            ),
            (
                "missing slice limits",
                &[("start_indices", "0")][..],
                av(&[4], DType::F64),
                "missing required param 'limit_indices'",
            ),
            (
                "slice rank mismatch",
                &[("start_indices", "0"), ("limit_indices", "2,3")][..],
                av(&[4, 6], DType::F64),
                "slice start/limit rank mismatch",
            ),
            (
                "zero slice stride",
                &[
                    ("start_indices", "0"),
                    ("limit_indices", "3"),
                    ("strides", "0"),
                ][..],
                av(&[4], DType::F64),
                "slice stride on axis 0 must be positive",
            ),
            (
                "slice limit before start",
                &[("start_indices", "3"), ("limit_indices", "2")][..],
                av(&[4], DType::F64),
                "invalid slice bounds axis 0",
            ),
        ] {
            let err = infer_equation_output_aval(&eqn(Primitive::Slice, params), &input)
                .unwrap_err();
            match err {
                PartialEvalError::ShapeInference { primitive, detail } => {
                    assert_eq!(primitive, Primitive::Slice, "{label}");
                    assert!(
                        detail.contains(expected_detail),
                        "{label}: unexpected detail: {detail}"
                    );
                }
                other => panic!("{label}: unexpected error: {other}"),
            }
        }

        // OneHot [2] num_classes=5 default axis -> [2,5] dtype F64 (catch-all kept
        // the [2] index shape and its dtype).
        let out = infer_equation_output_aval(
            &eqn(Primitive::OneHot, &[("num_classes", "5")]),
            &av(&[2], DType::I64),
        )
        .unwrap();
        assert_eq!(out.shape.dims, vec![2, 5]);
        assert_eq!(out.dtype, DType::F64);
        // axis=0 -> [5,2].
        let out = infer_equation_output_aval(
            &eqn(Primitive::OneHot, &[("num_classes", "5"), ("axis", "0")]),
            &av(&[2], DType::I64),
        )
        .unwrap();
        assert_eq!(out.shape.dims, vec![5, 2]);

        for (label, params, input, expected_detail) in [
            (
                "bad broadcast shape token",
                &[("shape", "2,bad,3"), ("broadcast_dimensions", "1")][..],
                av(&[3], DType::F64),
                "invalid u32 in param 'shape'",
            ),
            (
                "missing broadcast shape",
                &[("broadcast_dimensions", "0")][..],
                av(&[3], DType::F64),
                "missing required param 'shape'",
            ),
            (
                "duplicate broadcast target axis",
                &[("shape", "2,3"), ("broadcast_dimensions", "0,0")][..],
                av(&[1, 1], DType::F64),
                "broadcast_dimensions must be unique",
            ),
            (
                "broadcast target axis out of bounds",
                &[("shape", "2,3"), ("broadcast_dimensions", "2")][..],
                av(&[3], DType::F64),
                "target axis 2 out of range",
            ),
            (
                "incompatible broadcast dimension",
                &[("shape", "2,3"), ("broadcast_dimensions", "0")][..],
                av(&[4], DType::F64),
                "cannot broadcast input dim 4 into target dim 2",
            ),
        ] {
            let err = infer_equation_output_aval(
                &eqn(Primitive::BroadcastInDim, params),
                &input,
            )
            .unwrap_err();
            match err {
                PartialEvalError::ShapeInference { primitive, detail } => {
                    assert_eq!(primitive, Primitive::BroadcastInDim, "{label}");
                    assert!(
                        detail.contains(expected_detail),
                        "{label}: unexpected detail: {detail}"
                    );
                }
                other => panic!("{label}: unexpected error: {other}"),
            }
        }

        // Plural-fn paths: BroadcastedIota (0 inputs) + Complex (binary).
        fn eqn_n(prim: Primitive, outs: usize, params: &[(&str, &str)]) -> Equation {
            let mut p = BTreeMap::new();
            for (k, v) in params {
                p.insert((*k).to_string(), (*v).to_string());
            }
            Equation {
                primitive: prim,
                inputs: smallvec![],
                outputs: (0..outs).map(|i| VarId(i as u32 + 1)).collect(),
                params: p,
                effects: vec![],
                sub_jaxprs: vec![],
            }
        }

        // BroadcastedIota shape=3,4 dtype=f32 -> [3,4] F32.
        let outs = infer_equation_output_avals(
            &eqn_n(
                Primitive::BroadcastedIota,
                1,
                &[("shape", "3,4"), ("dtype", "f32")],
            ),
            &[],
        )
        .unwrap();
        assert_eq!(outs[0].shape.dims, vec![3, 4]);
        assert_eq!(outs[0].dtype, DType::F32);

        for (label, params, expected_detail) in [
            (
                "bad shape token",
                &[("shape", "3,bad,4"), ("dtype", "f32")][..],
                "invalid u32 in param 'shape'",
            ),
            (
                "dimension out of bounds",
                &[("shape", "3,4"), ("dimension", "2")][..],
                "dimension 2 out of bounds for rank 2",
            ),
            (
                "bool dtype",
                &[("shape", "3,4"), ("dtype", "bool")][..],
                "broadcasted_iota does not accept bool dtype",
            ),
        ] {
            let err = infer_equation_output_avals(
                &eqn_n(Primitive::BroadcastedIota, 1, params),
                &[],
            )
            .unwrap_err();
            match err {
                PartialEvalError::ShapeInference { primitive, detail } => {
                    assert_eq!(primitive, Primitive::BroadcastedIota, "{label}");
                    assert!(
                        detail.contains(expected_detail),
                        "{label}: unexpected detail: {detail}"
                    );
                }
                other => panic!("{label}: unexpected error: {other}"),
            }
        }

        // Complex(f32,f32) -> Complex64; broadcast [3,1] with [4] -> [3,4].
        let outs = infer_equation_output_avals(
            &eqn_n(Primitive::Complex, 1, &[]),
            &[av(&[3, 1], DType::F32), av(&[4], DType::F32)],
        )
        .unwrap();
        assert_eq!(outs[0].dtype, DType::Complex64);
        assert_eq!(outs[0].shape.dims, vec![3, 4]);
        // (f64,f64) -> Complex128.
        let outs = infer_equation_output_avals(
            &eqn_n(Primitive::Complex, 1, &[]),
            &[av(&[2], DType::F64), av(&[2], DType::F64)],
        )
        .unwrap();
        assert_eq!(outs[0].dtype, DType::Complex128);

        // Split: even num_sections=3 on [6,2] axis 0 -> packed [3,2,2]
        // (mirrors fj-trace's even-split packed output).
        let out = infer_equation_output_aval(
            &eqn(Primitive::Split, &[("axis", "0"), ("num_sections", "3")]),
            &av(&[6, 2], DType::F64),
        )
        .unwrap();
        assert_eq!(out.shape.dims, vec![3, 2, 2]);
        // even sizes=2,2 on [4] axis 0 -> packed [2,2].
        let out = infer_equation_output_aval(
            &eqn(Primitive::Split, &[("axis", "0"), ("sizes", "2,2")]),
            &av(&[4], DType::F64),
        )
        .unwrap();
        assert_eq!(out.shape.dims, vec![2, 2]);
        // uneven sizes=2,3 on [5,4] axis 0 -> first section [2,4].
        let out = infer_equation_output_aval(
            &eqn(Primitive::Split, &[("axis", "0"), ("sizes", "2,3")]),
            &av(&[5, 4], DType::F64),
        )
        .unwrap();
        assert_eq!(out.shape.dims, vec![2, 4]);
        // negative axis=-1 even num_sections=2 on [2,4] -> packed [2,2,2].
        let out = infer_equation_output_aval(
            &eqn(Primitive::Split, &[("axis", "-1"), ("num_sections", "2")]),
            &av(&[2, 4], DType::F64),
        )
        .unwrap();
        assert_eq!(out.shape.dims, vec![2, 2, 2]);

        // det/slogdet of an f32 matrix must stage f32 output avals (validates the
        // delegate_infer_to_trace path returns the f32-preserving fj-trace inference,
        // matching eval — the prior hardcoded-F64 inference diverged for f32 inputs).
        let det =
            infer_equation_output_avals(&eqn_n(Primitive::Det, 1, &[]), &[av(&[3, 3], DType::F32)])
                .unwrap();
        assert_eq!(det.len(), 1);
        assert_eq!(det[0].dtype, DType::F32);
        assert_eq!(det[0].shape.dims, Vec::<u32>::new());
        let sld = infer_equation_output_avals(
            &eqn_n(Primitive::Slogdet, 2, &[]),
            &[av(&[3, 3], DType::F32)],
        )
        .unwrap();
        assert_eq!(sld.len(), 2);
        assert_eq!(sld[0].dtype, DType::F32);
        assert_eq!(sld[1].dtype, DType::F32);
        // f64 input still stages f64 (unchanged).
        let det64 =
            infer_equation_output_avals(&eqn_n(Primitive::Det, 1, &[]), &[av(&[2, 2], DType::F64)])
                .unwrap();
        assert_eq!(det64[0].dtype, DType::F64);
    }
}
