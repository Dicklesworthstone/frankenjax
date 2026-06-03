//! CPU execution engine with dependency-aware equation scheduling.
//!
//! The CpuBackend provides the Backend trait implementation for host-CPU
//! execution. Pure equations whose inputs are already available are evaluated
//! in dependency waves, with sufficiently large waves evaluated in parallel.
//!
//! Contract: p2c006.strict.inv001 (CPU always available).

use fj_core::{Atom, DType, Equation, Jaxpr, Literal, Primitive, Shape, TensorValue, Value, VarId};
use fj_interpreters::{InterpreterError, eval_equation_outputs, eval_equation_single};
use fj_runtime::backend::{Backend, BackendCapabilities, BackendError};
use fj_runtime::buffer::Buffer;
use fj_runtime::device::{DeviceId, DeviceInfo, Platform};
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use std::collections::BTreeSet;

// The scan scheduler stays faster for medium/wide DAGs; dependency counts win
// once repeated ready scans dominate long pure segments.
const DEPENDENCY_COUNT_MIN_SEGMENT_LEN: usize = 128;
const SCALAR_PARALLEL_READY_WAVE_MIN_LEN: usize = 256;
const TENSOR_PARALLEL_READY_WAVE_MIN_ELEMENTS: usize = 4_096;
const TENSOR_PARALLEL_INPUT_MIN_ELEMENTS: usize = 1_024;
const F64_TENSOR_DAG_MIN_SEGMENT_LEN: usize = 128;
const F64_TENSOR_DAG_MAX_DIRECT_ELEMENTS: usize = 128;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ReadyWaveCost {
    tensor_element_work: usize,
    max_tensor_elements: usize,
}

fn equation_inputs_ready(equation: &Equation, env: &HashMap<VarId, Value>) -> bool {
    equation.inputs.iter().all(|atom| match atom {
        Atom::Var(var) => env.contains_key(var),
        Atom::Lit(_) => true,
    })
}

fn first_missing_input_var(equation: &Equation, env: &HashMap<VarId, Value>) -> Option<VarId> {
    equation.inputs.iter().find_map(|atom| match atom {
        Atom::Var(var) if !env.contains_key(var) => Some(*var),
        _ => None,
    })
}

fn is_scheduler_barrier(equation: &Equation) -> bool {
    !equation.effects.is_empty() || !equation.sub_jaxprs.is_empty() || equation.outputs.len() > 1
}

fn evaluate_equation_multi(
    equation: &Equation,
    env: &HashMap<VarId, Value>,
) -> Result<Vec<Value>, InterpreterError> {
    eval_equation_outputs(equation, env)
}

fn evaluate_equation(
    equation: &Equation,
    env: &HashMap<VarId, Value>,
) -> Result<Value, InterpreterError> {
    // Single-output fast path: avoids the one-element Vec<Value> that
    // eval_primitive_multi allocates per equation. Bit-for-bit identical to
    // the previous evaluate_equation_multi + arity-check + pop.
    eval_equation_single(equation, env)
}

fn single_output_var(equation: &Equation) -> Result<VarId, InterpreterError> {
    equation
        .outputs
        .first()
        .copied()
        .ok_or(InterpreterError::UnexpectedOutputArity {
            primitive: equation.primitive,
            expected: 1,
            actual: 0,
        })
}

fn malformed_scheduler_jaxpr(detail: String) -> InterpreterError {
    InterpreterError::InvariantViolation {
        detail: format!("CPU scheduler received malformed Jaxpr: {detail}"),
    }
}

fn validate_scheduler_bindings(jaxpr: &Jaxpr) -> Result<(), InterpreterError> {
    let mut bindings = BTreeSet::new();

    for var in &jaxpr.invars {
        if !bindings.insert(*var) {
            return Err(malformed_scheduler_jaxpr(format!(
                "duplicate input binding v{}",
                var.0
            )));
        }
    }

    for var in &jaxpr.constvars {
        if !bindings.insert(*var) {
            return Err(malformed_scheduler_jaxpr(format!(
                "duplicate const binding v{}",
                var.0
            )));
        }
    }

    for (equation_index, equation) in jaxpr.equations.iter().enumerate() {
        for out_var in &equation.outputs {
            if !bindings.insert(*out_var) {
                return Err(malformed_scheduler_jaxpr(format!(
                    "duplicate equation output binding v{} at equation {equation_index}",
                    out_var.0
                )));
            }
        }
    }

    let mut seen_outvars = BTreeSet::new();
    for out_var in &jaxpr.outvars {
        if !seen_outvars.insert(*out_var) {
            return Err(malformed_scheduler_jaxpr(format!(
                "duplicate output reference v{}",
                out_var.0
            )));
        }
        if !bindings.contains(out_var) {
            return Err(malformed_scheduler_jaxpr(format!(
                "unknown output reference v{}",
                out_var.0
            )));
        }
    }

    Ok(())
}

fn atom_tensor_element_count(atom: &Atom, env: &HashMap<VarId, Value>) -> Option<usize> {
    match atom {
        Atom::Var(var) => env.get(var).and_then(|value| match value {
            Value::Tensor(tensor) => Some(tensor.len()),
            Value::Scalar(_) => None,
        }),
        Atom::Lit(_) => None,
    }
}

fn ready_wave_cost(
    jaxpr: &Jaxpr,
    env: &HashMap<VarId, Value>,
    ready_indices: &[usize],
) -> ReadyWaveCost {
    ready_indices
        .iter()
        .fold(ReadyWaveCost::default(), |mut cost, idx| {
            let equation_work = jaxpr.equations[*idx]
                .inputs
                .iter()
                .filter_map(|atom| atom_tensor_element_count(atom, env))
                .max()
                .unwrap_or(0);
            cost.tensor_element_work = cost.tensor_element_work.saturating_add(equation_work);
            cost.max_tensor_elements = cost.max_tensor_elements.max(equation_work);
            cost
        })
}

fn should_parallelize_ready_wave_by_cost(
    ready_len: usize,
    tensor_element_work: usize,
    max_tensor_elements: usize,
    available_threads: usize,
) -> bool {
    ready_len > 1
        && available_threads > 1
        && (ready_len >= SCALAR_PARALLEL_READY_WAVE_MIN_LEN
            || tensor_element_work >= TENSOR_PARALLEL_READY_WAVE_MIN_ELEMENTS
            || max_tensor_elements >= TENSOR_PARALLEL_INPUT_MIN_ELEMENTS)
}

fn should_parallelize_ready_wave(
    jaxpr: &Jaxpr,
    env: &HashMap<VarId, Value>,
    ready_indices: &[usize],
) -> bool {
    let cost = ready_wave_cost(jaxpr, env, ready_indices);
    should_parallelize_ready_wave_by_cost(
        ready_indices.len(),
        cost.tensor_element_work,
        cost.max_tensor_elements,
        rayon::current_num_threads(),
    )
}

fn execute_ready_wave(
    jaxpr: &Jaxpr,
    env: &mut HashMap<VarId, Value>,
    executed: &mut [bool],
    remaining: &mut usize,
    ready_indices: &[usize],
) -> Result<(), InterpreterError> {
    if let [idx] = ready_indices {
        let eqn = &jaxpr.equations[*idx];
        let output = evaluate_equation(eqn, env)?;
        let out_var = single_output_var(eqn)?;
        env.insert(out_var, output);
        executed[*idx] = true;
        *remaining -= 1;
        return Ok(());
    }

    let should_parallelize = should_parallelize_ready_wave(jaxpr, env, ready_indices);

    if should_parallelize {
        // No env.clone() needed: the parallel phase only reads from env.
        // The shared borrow is released after collect() before we mutate env below.
        let env_ref = &*env;
        let mut evaluated = ready_indices
            .par_iter()
            .map(|idx| {
                let eqn = &jaxpr.equations[*idx];
                let output = evaluate_equation(eqn, env_ref)?;
                let out_var = single_output_var(eqn)?;
                Ok((*idx, out_var, output))
            })
            .collect::<Result<Vec<_>, InterpreterError>>()?;

        evaluated.sort_by_key(|(idx, _, _)| *idx);
        for (idx, out_var, out_value) in evaluated {
            env.insert(out_var, out_value);
            executed[idx] = true;
            *remaining -= 1;
        }
    } else {
        for idx in ready_indices {
            let eqn = &jaxpr.equations[*idx];
            let output = evaluate_equation(eqn, env)?;
            let out_var = single_output_var(eqn)?;
            env.insert(out_var, output);
            executed[*idx] = true;
            *remaining -= 1;
        }
    }

    Ok(())
}

fn first_missing_input_in_segment(
    jaxpr: &Jaxpr,
    env: &HashMap<VarId, Value>,
    executed: &[bool],
    segment_start: usize,
    segment_end: usize,
) -> Result<VarId, InterpreterError> {
    for (idx, done) in executed
        .iter()
        .enumerate()
        .take(segment_end)
        .skip(segment_start)
    {
        if *done {
            continue;
        }
        let equation = &jaxpr.equations[idx];
        if let Some(missing) = first_missing_input_var(equation, env) {
            return Ok(missing);
        }
        if let Some(out_var) = equation.outputs.first() {
            return Ok(*out_var);
        }
    }
    Err(InterpreterError::InvariantViolation {
        detail: format!(
            "scheduler could not identify missing input in segment {segment_start}..{segment_end}"
        ),
    })
}

fn execute_scan_segment(
    jaxpr: &Jaxpr,
    segment_start: usize,
    segment_end: usize,
    env: &mut HashMap<VarId, Value>,
    executed: &mut [bool],
    remaining: &mut usize,
    max_ready_wave: &mut usize,
) -> Result<(), InterpreterError> {
    let mut segment_remaining = segment_end - segment_start;
    while segment_remaining > 0 {
        let mut ready_indices = Vec::new();
        for (idx, done) in executed
            .iter()
            .enumerate()
            .take(segment_end)
            .skip(segment_start)
        {
            if *done {
                continue;
            }
            let equation = &jaxpr.equations[idx];
            if equation_inputs_ready(equation, env) {
                ready_indices.push(idx);
            }
        }

        if ready_indices.is_empty() {
            return Err(InterpreterError::MissingVariable(
                first_missing_input_in_segment(jaxpr, env, executed, segment_start, segment_end)?,
            ));
        }

        *max_ready_wave = (*max_ready_wave).max(ready_indices.len());
        let ready_len = ready_indices.len();
        execute_ready_wave(jaxpr, env, executed, remaining, &ready_indices)?;
        segment_remaining -= ready_len;
    }

    Ok(())
}

fn execute_pure_segment(
    jaxpr: &Jaxpr,
    segment_start: usize,
    segment_end: usize,
    env: &mut HashMap<VarId, Value>,
    executed: &mut [bool],
    remaining: &mut usize,
    max_ready_wave: &mut usize,
) -> Result<(), InterpreterError> {
    let segment_len = segment_end - segment_start;
    if execute_linear_topological_segment(
        jaxpr,
        segment_start,
        segment_end,
        env,
        executed,
        remaining,
        max_ready_wave,
    )? {
        return Ok(());
    }

    let mut producer_local_by_var: HashMap<VarId, usize> =
        HashMap::with_capacity_and_hasher(segment_len, Default::default());
    for idx in segment_start..segment_end {
        producer_local_by_var.insert(
            single_output_var(&jaxpr.equations[idx])?,
            idx - segment_start,
        );
    }

    let mut pending_inputs = vec![0_usize; segment_len];
    let mut consumers_by_producer = vec![Vec::new(); segment_len];
    let mut ready_indices = Vec::new();

    for idx in segment_start..segment_end {
        let local_idx = idx - segment_start;
        let equation = &jaxpr.equations[idx];
        for atom in &equation.inputs {
            if let Atom::Var(var) = atom {
                if env.contains_key(var) {
                    continue;
                }
                pending_inputs[local_idx] += 1;
                if let Some(producer_local_idx) = producer_local_by_var.get(var) {
                    consumers_by_producer[*producer_local_idx].push(local_idx);
                }
            }
        }

        if pending_inputs[local_idx] == 0 {
            ready_indices.push(idx);
        }
    }

    let mut segment_remaining = segment_len;
    while segment_remaining > 0 {
        if ready_indices.is_empty() {
            return Err(InterpreterError::MissingVariable(
                first_missing_input_in_segment(jaxpr, env, executed, segment_start, segment_end)?,
            ));
        }

        *max_ready_wave = (*max_ready_wave).max(ready_indices.len());
        let ready_len = ready_indices.len();
        execute_ready_wave(jaxpr, env, executed, remaining, &ready_indices)?;
        segment_remaining -= ready_len;

        let mut next_ready = Vec::new();
        for idx in &ready_indices {
            let producer_local_idx = *idx - segment_start;
            for &local_idx in &consumers_by_producer[producer_local_idx] {
                if executed[segment_start + local_idx] || pending_inputs[local_idx] == 0 {
                    continue;
                }
                pending_inputs[local_idx] -= 1;
                if pending_inputs[local_idx] == 0 {
                    next_ready.push(segment_start + local_idx);
                }
            }
        }
        next_ready.sort_unstable();
        ready_indices = next_ready;
    }

    Ok(())
}

fn execute_linear_topological_segment(
    jaxpr: &Jaxpr,
    segment_start: usize,
    segment_end: usize,
    env: &mut HashMap<VarId, Value>,
    _executed: &mut [bool],
    remaining: &mut usize,
    max_ready_wave: &mut usize,
) -> Result<bool, InterpreterError> {
    let segment_len = segment_end - segment_start;
    if execute_terminal_i64_add_chain_segment(
        jaxpr,
        segment_start,
        segment_end,
        env,
        remaining,
        max_ready_wave,
    )? {
        return Ok(true);
    }
    if execute_terminal_i64_add_dag_segment(
        jaxpr,
        segment_start,
        segment_end,
        env,
        remaining,
        max_ready_wave,
    )? {
        return Ok(true);
    }
    if execute_terminal_f64_tensor_add_dag_segment(
        jaxpr,
        segment_start,
        segment_end,
        env,
        remaining,
        max_ready_wave,
    )? {
        return Ok(true);
    }

    let mut previous_output = None;
    for equation in &jaxpr.equations[segment_start..segment_end] {
        for atom in &equation.inputs {
            if let Atom::Var(var) = atom
                && !env.contains_key(var)
                && Some(*var) != previous_output
            {
                return Ok(false);
            }
        }
        previous_output = Some(single_output_var(equation)?);
    }

    *max_ready_wave = (*max_ready_wave).max(1);
    for equation in &jaxpr.equations[segment_start..segment_end] {
        let output = evaluate_equation(equation, env)?;
        let out_var = single_output_var(equation)?;
        env.insert(out_var, output);
    }
    *remaining -= segment_len;

    Ok(true)
}

#[derive(Clone, Copy)]
enum F64TensorSource<'a> {
    Dense(&'a [f64]),
    Literals(&'a [Literal]),
}

impl F64TensorSource<'_> {
    fn len(self) -> usize {
        match self {
            Self::Dense(values) => values.len(),
            Self::Literals(values) => values.len(),
        }
    }

    fn is_dense(self) -> bool {
        matches!(self, Self::Dense(_))
    }

    fn value_at(self, idx: usize) -> Option<f64> {
        match self {
            Self::Dense(values) => Some(values[idx]),
            Self::Literals(values) => values[idx].as_f64(),
        }
    }
}

#[derive(Clone, Copy)]
enum F64TensorDagAtom {
    Lit(f64),
    Slot(usize),
}

impl F64TensorDagAtom {
    fn value(self, slots: &[f64]) -> f64 {
        match self {
            Self::Lit(value) => value,
            Self::Slot(idx) => slots[idx],
        }
    }
}

struct F64TensorAddStep {
    left: F64TensorDagAtom,
    right: F64TensorDagAtom,
    out_idx: usize,
}

struct F64TensorDagOutputInfo {
    shape: Option<Shape>,
    len: Option<usize>,
    all_source_tensors_dense: bool,
}

fn var_slot_index(var: VarId, slot_count: usize) -> Option<usize> {
    let idx = usize::try_from(var.0).ok()?;
    (idx < slot_count).then_some(idx)
}

fn f64_tensor_source(tensor: &TensorValue) -> Option<F64TensorSource<'_>> {
    if tensor.dtype != DType::F64 {
        return None;
    }
    if let Some(values) = tensor.elements.as_f64_slice() {
        return Some(F64TensorSource::Dense(values));
    }

    let literals = tensor.elements.as_slice();
    literals
        .iter()
        .all(|literal| matches!(literal, Literal::F64Bits(_)))
        .then_some(F64TensorSource::Literals(literals))
}

fn f64_tensor_atom_source(
    atom: &Atom,
    source_slots: &[Option<F64TensorSource<'_>>],
    source_shapes: &[Option<&Shape>],
    present: &[bool],
    levels: &[usize],
    output_info: &mut F64TensorDagOutputInfo,
) -> Option<(F64TensorDagAtom, usize)> {
    match atom {
        Atom::Lit(literal) => Some((F64TensorDagAtom::Lit(literal.as_f64()?), 0)),
        Atom::Var(var) => {
            let idx = var_slot_index(*var, present.len())?;
            if !present[idx] {
                return None;
            }
            if let Some(source) = source_slots[idx] {
                let shape = source_shapes[idx]?;
                match &mut output_info.shape {
                    Some(existing) if existing != shape => return None,
                    Some(_) => {}
                    None => output_info.shape = Some(shape.clone()),
                }
                match &mut output_info.len {
                    Some(existing) if *existing != source.len() => return None,
                    Some(_) => {}
                    None => output_info.len = Some(source.len()),
                }
                output_info.all_source_tensors_dense &= source.is_dense();
            }
            Some((F64TensorDagAtom::Slot(idx), levels[idx]))
        }
    }
}

fn max_slot_index_for_add_dag(
    jaxpr: &Jaxpr,
    segment_start: usize,
    segment_end: usize,
) -> Option<usize> {
    let mut max_slot = 0_usize;
    for var in jaxpr.invars.iter().chain(jaxpr.outvars.iter()) {
        max_slot = max_slot.max(usize::try_from(var.0).ok()?);
    }

    for equation in &jaxpr.equations[segment_start..segment_end] {
        for atom in &equation.inputs {
            if let Atom::Var(var) = atom {
                max_slot = max_slot.max(usize::try_from(var.0).ok()?);
            }
        }
        for var in &equation.outputs {
            max_slot = max_slot.max(usize::try_from(var.0).ok()?);
        }
    }

    Some(max_slot)
}

fn i64_atom_slot_value(atom: &Atom, values: &[i64], present: &[bool]) -> Option<(i64, usize)> {
    match atom {
        Atom::Lit(Literal::I64(value)) => Some((*value, 0)),
        Atom::Lit(_) => None,
        Atom::Var(var) => {
            let idx = var_slot_index(*var, values.len())?;
            present[idx].then_some((values[idx], idx))
        }
    }
}

fn max_slot_index_for_i64_add_dag(
    jaxpr: &Jaxpr,
    segment_start: usize,
    segment_end: usize,
) -> Option<usize> {
    max_slot_index_for_add_dag(jaxpr, segment_start, segment_end)
}

fn execute_terminal_i64_add_dag_segment(
    jaxpr: &Jaxpr,
    segment_start: usize,
    segment_end: usize,
    env: &mut HashMap<VarId, Value>,
    remaining: &mut usize,
    max_ready_wave: &mut usize,
) -> Result<bool, InterpreterError> {
    let segment_len = segment_end - segment_start;
    if segment_len == 0 || segment_end != jaxpr.equations.len() || jaxpr.outvars.len() != 1 {
        return Ok(false);
    }

    let Some(max_slot) = max_slot_index_for_i64_add_dag(jaxpr, segment_start, segment_end) else {
        return Ok(false);
    };
    let Some(slot_count) = max_slot.checked_add(1) else {
        return Ok(false);
    };
    let dense_slot_limit = (jaxpr.invars.len() + segment_len + jaxpr.outvars.len() + 8) * 4;
    if slot_count > dense_slot_limit {
        return Ok(false);
    }

    let mut values = vec![0_i64; slot_count];
    let mut levels = vec![0_usize; slot_count];
    let mut present = vec![false; slot_count];
    for (var, value) in env.iter() {
        let Some(idx) = var_slot_index(*var, slot_count) else {
            continue;
        };
        let Value::Scalar(Literal::I64(value)) = value else {
            continue;
        };
        values[idx] = *value;
        present[idx] = true;
    }

    let mut level_counts = vec![0_usize; segment_len + 1];
    let mut observed_max_ready_wave = 0_usize;

    for equation in &jaxpr.equations[segment_start..segment_end] {
        if equation.primitive != Primitive::Add
            || !equation.params.is_empty()
            || !equation.effects.is_empty()
            || !equation.sub_jaxprs.is_empty()
            || equation.inputs.len() != 2
            || equation.outputs.len() != 1
        {
            return Ok(false);
        }

        let Some((left, left_idx)) = i64_atom_slot_value(&equation.inputs[0], &values, &present)
        else {
            return Ok(false);
        };
        let Some((right, right_idx)) = i64_atom_slot_value(&equation.inputs[1], &values, &present)
        else {
            return Ok(false);
        };
        let Some(out_idx) = var_slot_index(equation.outputs[0], slot_count) else {
            return Ok(false);
        };

        let level = levels[left_idx].max(levels[right_idx]) + 1;
        values[out_idx] = left.wrapping_add(right);
        levels[out_idx] = level;
        present[out_idx] = true;
        level_counts[level] += 1;
        observed_max_ready_wave = observed_max_ready_wave.max(level_counts[level]);
    }

    let Some(out_idx) = var_slot_index(jaxpr.outvars[0], slot_count) else {
        return Ok(false);
    };
    if !present[out_idx] {
        return Ok(false);
    }

    env.insert(jaxpr.outvars[0], Value::scalar_i64(values[out_idx]));
    *remaining -= segment_len;
    *max_ready_wave = (*max_ready_wave).max(observed_max_ready_wave);
    Ok(true)
}

fn execute_terminal_f64_tensor_add_dag_segment(
    jaxpr: &Jaxpr,
    segment_start: usize,
    segment_end: usize,
    env: &mut HashMap<VarId, Value>,
    remaining: &mut usize,
    max_ready_wave: &mut usize,
) -> Result<bool, InterpreterError> {
    let segment_len = segment_end - segment_start;
    if segment_len < F64_TENSOR_DAG_MIN_SEGMENT_LEN
        || segment_end != jaxpr.equations.len()
        || jaxpr.outvars.len() != 1
    {
        return Ok(false);
    }

    let Some(max_slot) = max_slot_index_for_add_dag(jaxpr, segment_start, segment_end) else {
        return Ok(false);
    };
    let Some(slot_count) = max_slot.checked_add(1) else {
        return Ok(false);
    };
    let dense_slot_limit = (jaxpr.invars.len() + segment_len + jaxpr.outvars.len() + 8) * 4;
    if slot_count > dense_slot_limit {
        return Ok(false);
    }

    let compiled = {
        let mut source_slots = vec![None; slot_count];
        let mut source_shapes = vec![None; slot_count];
        let mut present = vec![false; slot_count];
        let mut levels = vec![0_usize; slot_count];
        for (var, value) in env.iter() {
            let Some(idx) = var_slot_index(*var, slot_count) else {
                continue;
            };
            let Value::Tensor(tensor) = value else {
                continue;
            };
            let Some(source) = f64_tensor_source(tensor) else {
                continue;
            };
            source_slots[idx] = Some(source);
            source_shapes[idx] = Some(&tensor.shape);
            present[idx] = true;
        }

        let mut steps = Vec::with_capacity(segment_len);
        let mut level_counts = vec![0_usize; segment_len + 1];
        let mut observed_max_ready_wave = 0_usize;
        let mut output_info = F64TensorDagOutputInfo {
            shape: None,
            len: None,
            all_source_tensors_dense: true,
        };

        for equation in &jaxpr.equations[segment_start..segment_end] {
            if equation.primitive != Primitive::Add
                || !equation.params.is_empty()
                || !equation.effects.is_empty()
                || !equation.sub_jaxprs.is_empty()
                || equation.inputs.len() != 2
                || equation.outputs.len() != 1
            {
                return Ok(false);
            }

            let Some((left, left_level)) = f64_tensor_atom_source(
                &equation.inputs[0],
                &source_slots,
                &source_shapes,
                &present,
                &levels,
                &mut output_info,
            ) else {
                return Ok(false);
            };
            let Some((right, right_level)) = f64_tensor_atom_source(
                &equation.inputs[1],
                &source_slots,
                &source_shapes,
                &present,
                &levels,
                &mut output_info,
            ) else {
                return Ok(false);
            };
            let Some(out_idx) = var_slot_index(equation.outputs[0], slot_count) else {
                return Ok(false);
            };
            if output_info
                .len
                .is_some_and(|len| len > F64_TENSOR_DAG_MAX_DIRECT_ELEMENTS)
            {
                return Ok(false);
            }

            let level = left_level.max(right_level) + 1;
            present[out_idx] = true;
            levels[out_idx] = level;
            level_counts[level] += 1;
            observed_max_ready_wave = observed_max_ready_wave.max(level_counts[level]);
            steps.push(F64TensorAddStep {
                left,
                right,
                out_idx,
            });
        }

        let Some(out_idx) = var_slot_index(jaxpr.outvars[0], slot_count) else {
            return Ok(false);
        };
        if !present[out_idx] {
            return Ok(false);
        }
        let Some(out_shape) = output_info.shape else {
            return Ok(false);
        };
        let Some(tensor_len) = output_info.len else {
            return Ok(false);
        };
        if tensor_len > F64_TENSOR_DAG_MAX_DIRECT_ELEMENTS {
            return Ok(false);
        }

        let input_sources = source_slots
            .iter()
            .enumerate()
            .filter_map(|(slot, source)| source.map(|source| (slot, source)))
            .collect::<Vec<_>>();
        let mut slot_values = vec![0.0_f64; slot_count];
        let mut output_values = Vec::with_capacity(tensor_len);
        for elem_idx in 0..tensor_len {
            for (slot, source) in &input_sources {
                let Some(value) = source.value_at(elem_idx) else {
                    return Ok(false);
                };
                slot_values[*slot] = value;
            }
            for step in &steps {
                slot_values[step.out_idx] =
                    step.left.value(&slot_values) + step.right.value(&slot_values);
            }
            output_values.push(slot_values[out_idx]);
        }

        Some((
            out_shape,
            output_values,
            observed_max_ready_wave,
            output_info.all_source_tensors_dense,
        ))
    };

    let Some((out_shape, output_values, observed_max_ready_wave, output_dense)) = compiled else {
        return Ok(false);
    };
    let output = if output_dense {
        TensorValue::new_f64_values(out_shape, output_values)
            .map_err(|err| InterpreterError::Primitive(fj_lax::EvalError::InvalidTensor(err)))?
    } else {
        let elements = output_values
            .into_iter()
            .map(Literal::from_f64)
            .collect::<Vec<_>>();
        TensorValue::new(DType::F64, out_shape, elements)
            .map_err(|err| InterpreterError::Primitive(fj_lax::EvalError::InvalidTensor(err)))?
    };

    env.insert(jaxpr.outvars[0], Value::Tensor(output));
    *remaining -= segment_len;
    *max_ready_wave = (*max_ready_wave).max(observed_max_ready_wave);
    Ok(true)
}

fn add_chain_step(
    equation: &Equation,
    current_var: VarId,
) -> Result<Option<(VarId, i64)>, InterpreterError> {
    if equation.primitive != Primitive::Add
        || !equation.params.is_empty()
        || !equation.effects.is_empty()
        || !equation.sub_jaxprs.is_empty()
        || equation.inputs.len() != 2
    {
        return Ok(None);
    }

    let out_var = single_output_var(equation)?;
    match (&equation.inputs[0], &equation.inputs[1]) {
        (Atom::Var(var), Atom::Lit(Literal::I64(offset)))
        | (Atom::Lit(Literal::I64(offset)), Atom::Var(var))
            if *var == current_var =>
        {
            Ok(Some((out_var, *offset)))
        }
        _ => Ok(None),
    }
}

fn execute_terminal_i64_add_chain_segment(
    jaxpr: &Jaxpr,
    segment_start: usize,
    segment_end: usize,
    env: &mut HashMap<VarId, Value>,
    remaining: &mut usize,
    max_ready_wave: &mut usize,
) -> Result<bool, InterpreterError> {
    if segment_start >= segment_end
        || segment_end != jaxpr.equations.len()
        || jaxpr.outvars.len() != 1
    {
        return Ok(false);
    }

    let last_output = single_output_var(&jaxpr.equations[segment_end - 1])?;
    if jaxpr.outvars[0] != last_output {
        return Ok(false);
    }

    let mut current_var = match jaxpr.equations[segment_start]
        .inputs
        .iter()
        .find_map(|atom| {
            if let Atom::Var(var) = atom {
                Some(*var)
            } else {
                None
            }
        }) {
        Some(var) => var,
        None => return Ok(false),
    };

    let mut current_value = match env.get(&current_var) {
        Some(Value::Scalar(Literal::I64(value))) => *value,
        _ => return Ok(false),
    };

    for equation in &jaxpr.equations[segment_start..segment_end] {
        let Some((out_var, offset)) = add_chain_step(equation, current_var)? else {
            return Ok(false);
        };
        current_value = current_value.wrapping_add(offset);
        current_var = out_var;
    }

    env.insert(last_output, Value::scalar_i64(current_value));
    *remaining -= segment_end - segment_start;
    *max_ready_wave = (*max_ready_wave).max(1);
    Ok(true)
}

fn evaluate_jaxpr_parallel_inner(
    jaxpr: &Jaxpr,
    args: &[Value],
    max_ready_wave: &mut usize,
) -> Result<Vec<Value>, InterpreterError> {
    if !jaxpr.constvars.is_empty() {
        return Err(InterpreterError::ConstArity {
            expected: jaxpr.constvars.len(),
            actual: 0,
        });
    }
    if args.len() != jaxpr.invars.len() {
        return Err(InterpreterError::InputArity {
            expected: jaxpr.invars.len(),
            actual: args.len(),
        });
    }
    validate_scheduler_bindings(jaxpr)?;

    let mut env: HashMap<VarId, Value> = HashMap::with_capacity_and_hasher(
        jaxpr.invars.len() + jaxpr.equations.len(),
        Default::default(),
    );
    for (index, var) in jaxpr.invars.iter().enumerate() {
        env.insert(*var, args[index].clone());
    }

    let mut executed = vec![false; jaxpr.equations.len()];
    let mut remaining = jaxpr.equations.len();
    let mut next_pending = 0_usize;

    while remaining > 0 {
        while next_pending < executed.len() && executed[next_pending] {
            next_pending += 1;
        }
        if next_pending == executed.len() {
            return Err(InterpreterError::InvariantViolation {
                detail: format!(
                    "scheduler reported {remaining} remaining equation(s) with no pending work"
                ),
            });
        }
        let first_pending = next_pending;
        let first_eqn = &jaxpr.equations[first_pending];

        if is_scheduler_barrier(first_eqn) {
            let outputs = evaluate_equation_multi(first_eqn, &env)?;
            for (out_var, out_val) in first_eqn.outputs.iter().zip(outputs) {
                env.insert(*out_var, out_val);
            }
            executed[first_pending] = true;
            remaining -= 1;
            next_pending += 1;
            continue;
        }

        let segment_end = (first_pending..jaxpr.equations.len())
            .find(|idx| !executed[*idx] && is_scheduler_barrier(&jaxpr.equations[*idx]))
            .unwrap_or(jaxpr.equations.len());
        let segment_len = segment_end - first_pending;
        if segment_len < DEPENDENCY_COUNT_MIN_SEGMENT_LEN {
            execute_scan_segment(
                jaxpr,
                first_pending,
                segment_end,
                &mut env,
                &mut executed,
                &mut remaining,
                max_ready_wave,
            )?;
        } else {
            execute_pure_segment(
                jaxpr,
                first_pending,
                segment_end,
                &mut env,
                &mut executed,
                &mut remaining,
                max_ready_wave,
            )?;
        }
        next_pending = segment_end;
    }

    jaxpr
        .outvars
        .iter()
        .map(|var| {
            env.get(var)
                .cloned()
                .ok_or(InterpreterError::MissingVariable(*var))
        })
        .collect()
}

fn evaluate_jaxpr_parallel(jaxpr: &Jaxpr, args: &[Value]) -> Result<Vec<Value>, InterpreterError> {
    let mut ignored_max_ready_wave = 0_usize;
    evaluate_jaxpr_parallel_inner(jaxpr, args, &mut ignored_max_ready_wave)
}

/// CPU backend: interprets Jaxpr programs on the host CPU.
///
/// V1 scope: single CPU device (DeviceId(0)). Execution is synchronous and
/// uses dependency-wave parallel scheduling for independent equations.
pub struct CpuBackend {
    /// Number of logical CPU devices to expose.
    /// V1: always 1.
    device_count: u32,
    /// Version string for cache key inclusion.
    version_string: String,
}

impl CpuBackend {
    /// Create a CPU backend with a single device.
    #[must_use]
    pub fn new() -> Self {
        Self {
            device_count: 1,
            version_string: format!("fj-backend-cpu/{}", env!("CARGO_PKG_VERSION")),
        }
    }

    /// Create a CPU backend exposing multiple logical devices.
    ///
    /// Returns a structured error when the configuration is invalid.
    pub fn try_with_device_count(count: u32) -> Result<Self, BackendError> {
        if count == 0 {
            return Err(BackendError::InvalidConfiguration {
                backend: "cpu".to_owned(),
                detail: "device count must be at least 1".to_owned(),
            });
        }

        Ok(Self {
            device_count: count,
            version_string: format!("fj-backend-cpu/{}", env!("CARGO_PKG_VERSION")),
        })
    }

    /// Create a CPU backend exposing multiple logical devices.
    /// Useful for testing multi-device dispatch without GPU hardware.
    ///
    /// Invalid counts fall back to the single-device baseline to keep the CPU
    /// backend total and panic-free.
    #[must_use]
    pub fn with_device_count(count: u32) -> Self {
        Self::try_with_device_count(count).unwrap_or_default()
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn devices(&self) -> Vec<DeviceInfo> {
        (0..self.device_count)
            .map(|i| DeviceInfo {
                id: DeviceId(i),
                platform: Platform::Cpu,
                host_id: 0,
                process_index: 0,
            })
            .collect()
    }

    fn default_device(&self) -> DeviceId {
        DeviceId(0)
    }

    fn execute(
        &self,
        jaxpr: &Jaxpr,
        args: &[Value],
        device: DeviceId,
    ) -> Result<Vec<Value>, BackendError> {
        if device.0 >= self.device_count {
            return Err(BackendError::ExecutionFailed {
                detail: format!(
                    "device {} not available (have {})",
                    device.0, self.device_count
                ),
            });
        }
        evaluate_jaxpr_parallel(jaxpr, args).map_err(|e| BackendError::ExecutionFailed {
            detail: e.to_string(),
        })
    }

    fn allocate(&self, size_bytes: usize, device: DeviceId) -> Result<Buffer, BackendError> {
        if device.0 >= self.device_count {
            return Err(BackendError::AllocationFailed {
                device,
                detail: format!(
                    "device {} not available (have {})",
                    device.0, self.device_count
                ),
            });
        }
        Ok(Buffer::zeroed(size_bytes, device))
    }

    fn transfer(&self, buffer: &Buffer, target: DeviceId) -> Result<Buffer, BackendError> {
        if target.0 >= self.device_count {
            return Err(BackendError::TransferFailed {
                source: buffer.device(),
                target,
                detail: format!("target device {} not available", target.0),
            });
        }
        // CPU "transfer" is a clone (same memory space).
        Ok(Buffer::new(buffer.as_bytes().to_vec(), target))
    }

    fn version(&self) -> &str {
        &self.version_string
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            supported_dtypes: vec![DType::F64, DType::I64],
            max_tensor_rank: 8,
            memory_limit_bytes: None, // host memory, effectively unlimited
            multi_device: self.device_count > 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fj_core::{
        Atom, DType, Equation, Jaxpr, Literal, Primitive, ProgramSpec, Shape, TensorValue, VarId,
        build_program,
    };
    use std::collections::BTreeMap;

    fn make_parallel_independent_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(5)],
            vec![
                Equation {
                    primitive: Primitive::Neg,
                    inputs: vec![Atom::Var(VarId(1))].into(),
                    outputs: vec![VarId(3)].into(),
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Neg,
                    inputs: vec![Atom::Var(VarId(2))].into(),
                    outputs: vec![VarId(4)].into(),
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: vec![Atom::Var(VarId(3)), Atom::Var(VarId(4))].into(),
                    outputs: vec![VarId(5)].into(),
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
            ],
        )
    }

    fn make_switch_branch_identity_jaxpr() -> Jaxpr {
        Jaxpr::new(vec![VarId(1)], vec![], vec![VarId(1)], vec![])
    }

    fn make_switch_branch_self_binary_jaxpr(primitive: Primitive) -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive,
                inputs: vec![Atom::Var(VarId(1)), Atom::Var(VarId(1))].into(),
                outputs: vec![VarId(2)].into(),
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        )
    }

    fn make_switch_control_flow_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3)],
            vec![Equation {
                primitive: Primitive::Switch,
                inputs: vec![Atom::Var(VarId(1)), Atom::Var(VarId(2))].into(),
                outputs: vec![VarId(3)].into(),
                params: BTreeMap::from([("num_branches".to_owned(), "3".to_owned())]),
                effects: vec![],
                sub_jaxprs: vec![
                    make_switch_branch_identity_jaxpr(),
                    make_switch_branch_self_binary_jaxpr(Primitive::Add),
                    make_switch_branch_self_binary_jaxpr(Primitive::Mul),
                ],
            }],
        )
    }

    fn make_out_of_order_dependency_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Add,
                    inputs: vec![Atom::Var(VarId(2)), Atom::Lit(Literal::I64(1))].into(),
                    outputs: vec![VarId(3)].into(),
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: vec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(1))].into(),
                    outputs: vec![VarId(2)].into(),
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
            ],
        )
    }

    fn make_barrier_order_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(6)],
            vec![
                Equation {
                    primitive: Primitive::Neg,
                    inputs: vec![Atom::Var(VarId(1))].into(),
                    outputs: vec![VarId(3)].into(),
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Switch,
                    inputs: vec![Atom::Var(VarId(2)), Atom::Var(VarId(3))].into(),
                    outputs: vec![VarId(4)].into(),
                    params: BTreeMap::from([("num_branches".to_owned(), "1".to_owned())]),
                    effects: vec![],
                    sub_jaxprs: vec![make_switch_branch_identity_jaxpr()],
                },
                Equation {
                    primitive: Primitive::Neg,
                    inputs: vec![Atom::Var(VarId(1))].into(),
                    outputs: vec![VarId(5)].into(),
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: vec![Atom::Var(VarId(4)), Atom::Var(VarId(5))].into(),
                    outputs: vec![VarId(6)].into(),
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
            ],
        )
    }

    fn make_missing_dependency_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Add,
                inputs: vec![Atom::Var(VarId(99)), Atom::Lit(Literal::I64(1))].into(),
                outputs: vec![VarId(2)].into(),
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        )
    }

    fn make_dependency_chain_jaxpr(length: usize) -> Jaxpr {
        let input = VarId(1);
        let mut current = input;
        let mut equations = Vec::with_capacity(length);

        for next_var in (2_u32..).take(length) {
            let out = VarId(next_var);
            equations.push(Equation {
                primitive: Primitive::Add,
                inputs: vec![Atom::Var(current), Atom::Lit(Literal::I64(1))].into(),
                outputs: vec![out].into(),
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            });
            current = out;
        }

        Jaxpr::new(vec![input], vec![], vec![current], equations)
    }

    fn make_branched_fanin_jaxpr(branches: usize, depth: usize) -> Jaxpr {
        let input = VarId(1);
        let mut next_var = 2_u32;
        let mut equations = Vec::with_capacity(branches * depth + branches - 1);
        let mut active = vec![input; branches];

        for _ in 0..depth {
            for var in &mut active {
                let out = VarId(next_var);
                next_var += 1;
                equations.push(Equation {
                    primitive: Primitive::Add,
                    inputs: vec![Atom::Var(*var), Atom::Lit(Literal::I64(1))].into(),
                    outputs: vec![out].into(),
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                });
                *var = out;
            }
        }

        while active.len() > 1 {
            let mut next_level = Vec::with_capacity(active.len().div_ceil(2));
            for chunk in active.chunks(2) {
                if chunk.len() == 1 {
                    next_level.push(chunk[0]);
                    continue;
                }
                let out = VarId(next_var);
                next_var += 1;
                equations.push(Equation {
                    primitive: Primitive::Add,
                    inputs: vec![Atom::Var(chunk[0]), Atom::Var(chunk[1])].into(),
                    outputs: vec![out].into(),
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                });
                next_level.push(out);
            }
            active = next_level;
        }

        Jaxpr::new(vec![input], vec![], vec![active[0]], equations)
    }

    fn make_f64_tensor_branched_fanin_jaxpr(branches: usize, depth: usize) -> Jaxpr {
        let input = VarId(1);
        let mut next_var = 2_u32;
        let mut equations = Vec::with_capacity(branches * depth + branches - 1);
        let mut active = vec![input; branches];

        for _ in 0..depth {
            for var in &mut active {
                let out = VarId(next_var);
                next_var += 1;
                equations.push(Equation {
                    primitive: Primitive::Add,
                    inputs: vec![Atom::Var(*var), Atom::Lit(Literal::from_f64(1.0))].into(),
                    outputs: vec![out].into(),
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                });
                *var = out;
            }
        }

        while active.len() > 1 {
            let mut next_level = Vec::with_capacity(active.len().div_ceil(2));
            for chunk in active.chunks(2) {
                if chunk.len() == 1 {
                    next_level.push(chunk[0]);
                    continue;
                }
                let out = VarId(next_var);
                next_var += 1;
                equations.push(Equation {
                    primitive: Primitive::Add,
                    inputs: vec![Atom::Var(chunk[0]), Atom::Var(chunk[1])].into(),
                    outputs: vec![out].into(),
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                });
                next_level.push(out);
            }
            active = next_level;
        }

        Jaxpr::new(vec![input], vec![], vec![active[0]], equations)
    }

    fn make_tensor_ready_wave_jaxpr(width: usize) -> Jaxpr {
        let input = VarId(1);
        let mut equations = Vec::with_capacity(width);
        let mut outputs = Vec::with_capacity(width);

        for next_var in (2_u32..).take(width) {
            let out = VarId(next_var);
            equations.push(Equation {
                primitive: Primitive::Add,
                inputs: vec![Atom::Var(input), Atom::Lit(Literal::from_f64(1.0))].into(),
                outputs: vec![out].into(),
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            });
            outputs.push(out);
        }

        Jaxpr::new(vec![input], vec![], outputs, equations)
    }

    fn vector_f64_arg(len: usize) -> Value {
        let elements = (0..len)
            .map(|idx| Literal::from_f64(idx as f64))
            .collect::<Vec<_>>();
        Value::Tensor(
            TensorValue::new(DType::F64, Shape::vector(len as u32), elements)
                .expect("test tensor should be valid"),
        )
    }

    fn tensor_f64_bits(value: &Value) -> Vec<u64> {
        let tensor = value.as_tensor().expect("expected tensor");
        if let Some(values) = tensor.elements.as_f64_slice() {
            return values.iter().map(|value| value.to_bits()).collect();
        }
        tensor
            .elements
            .iter()
            .map(|literal| literal.as_f64().expect("expected f64").to_bits())
            .collect()
    }

    fn make_long_missing_dependency_jaxpr(length: usize) -> Jaxpr {
        let mut equations = Vec::with_capacity(length);
        let mut current = VarId(99);

        for next_var in (2_u32..).take(length) {
            let out = VarId(next_var);
            equations.push(Equation {
                primitive: Primitive::Add,
                inputs: vec![Atom::Var(current), Atom::Lit(Literal::I64(1))].into(),
                outputs: vec![out].into(),
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            });
            current = out;
        }

        Jaxpr::new(vec![VarId(1)], vec![], vec![current], equations)
    }

    fn make_long_duplicate_output_binding_jaxpr(length: usize) -> Jaxpr {
        let mut equations = Vec::with_capacity(length);
        equations.push(Equation {
            primitive: Primitive::Add,
            inputs: vec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(1))].into(),
            outputs: vec![VarId(2)].into(),
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        });
        equations.push(Equation {
            primitive: Primitive::Add,
            inputs: vec![Atom::Var(VarId(2)), Atom::Lit(Literal::I64(1))].into(),
            outputs: vec![VarId(3)].into(),
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        });
        equations.push(Equation {
            primitive: Primitive::Add,
            inputs: vec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(100))].into(),
            outputs: vec![VarId(2)].into(),
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        });

        for next_var in (4_u32..).take(length.saturating_sub(equations.len())) {
            equations.push(Equation {
                primitive: Primitive::Add,
                inputs: vec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(1))].into(),
                outputs: vec![VarId(next_var)].into(),
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            });
        }

        Jaxpr::new(vec![VarId(1)], vec![], vec![VarId(3)], equations)
    }

    #[test]
    fn cpu_backend_name() {
        let backend = CpuBackend::new();
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn cpu_backend_default_device() {
        let backend = CpuBackend::new();
        assert_eq!(backend.default_device(), DeviceId(0));
    }

    #[test]
    fn cpu_backend_single_device_discovery() {
        let backend = CpuBackend::new();
        let devices = backend.devices();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].id, DeviceId(0));
        assert_eq!(devices[0].platform, Platform::Cpu);
        assert_eq!(devices[0].host_id, 0);
        assert_eq!(devices[0].process_index, 0);
    }

    #[test]
    fn cpu_backend_multi_device_discovery() {
        let backend = CpuBackend::with_device_count(4);
        let devices = backend.devices();
        assert_eq!(devices.len(), 4);
        for (i, dev) in devices.iter().enumerate() {
            assert_eq!(dev.id, DeviceId(i as u32));
            assert_eq!(dev.platform, Platform::Cpu);
        }
    }

    #[test]
    fn cpu_backend_try_with_zero_devices_returns_structured_error() {
        let result = CpuBackend::try_with_device_count(0);
        assert!(matches!(
            result,
            Err(BackendError::InvalidConfiguration { .. })
        ));
        if let Err(err) = result {
            assert!(err.to_string().contains("device count must be at least 1"));
        }
    }

    #[test]
    fn cpu_backend_with_zero_devices_falls_back_to_single_device() {
        let backend = CpuBackend::with_device_count(0);
        let devices = backend.devices();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].id, DeviceId(0));
    }

    #[test]
    fn cpu_backend_execute_add2() {
        let backend = CpuBackend::new();
        let jaxpr = build_program(ProgramSpec::Add2);
        let result = backend
            .execute(
                &jaxpr,
                &[Value::scalar_i64(3), Value::scalar_i64(4)],
                DeviceId(0),
            )
            .expect("execution should succeed");
        assert_eq!(result, vec![Value::scalar_i64(7)]);
    }

    #[test]
    fn cpu_backend_execute_square() {
        let backend = CpuBackend::new();
        let jaxpr = build_program(ProgramSpec::Square);
        let result = backend
            .execute(&jaxpr, &[Value::scalar_f64(5.0)], DeviceId(0))
            .expect("execution should succeed");
        let val = result[0].as_f64_scalar().expect("should be f64");
        assert!((val - 25.0).abs() < 1e-10);
    }

    #[test]
    fn cpu_backend_execute_on_secondary_device() {
        let backend = CpuBackend::with_device_count(2);
        let jaxpr = build_program(ProgramSpec::Add2);
        let result = backend
            .execute(
                &jaxpr,
                &[Value::scalar_i64(3), Value::scalar_i64(4)],
                DeviceId(1),
            )
            .expect("configured secondary device should execute");
        assert_eq!(result, vec![Value::scalar_i64(7)]);
    }

    #[test]
    fn cpu_backend_execute_invalid_device() {
        let backend = CpuBackend::new();
        let jaxpr = build_program(ProgramSpec::Add2);
        let err = backend
            .execute(
                &jaxpr,
                &[Value::scalar_i64(3), Value::scalar_i64(4)],
                DeviceId(1),
            )
            .expect_err("invalid device should fail");

        let msg = err.to_string();
        assert!(msg.contains("device 1"), "error should identify device");
        assert!(
            msg.contains("not available"),
            "error should mention availability"
        );
    }

    #[test]
    fn test_parallel_independent_ops() {
        let jaxpr = make_parallel_independent_jaxpr();
        let mut max_ready_wave = 0_usize;
        let result = evaluate_jaxpr_parallel_inner(
            &jaxpr,
            &[Value::scalar_i64(3), Value::scalar_i64(4)],
            &mut max_ready_wave,
        )
        .expect("parallel execution should succeed");

        assert_eq!(result, vec![Value::scalar_i64(-7)]);
        assert!(
            max_ready_wave >= 2,
            "expected at least one parallel-ready wave with width >=2, got {max_ready_wave}"
        );
    }

    #[test]
    fn dependency_scheduler_executes_out_of_order_pure_equations() {
        let jaxpr = make_out_of_order_dependency_jaxpr();
        let mut max_ready_wave = 0_usize;
        let result =
            evaluate_jaxpr_parallel_inner(&jaxpr, &[Value::scalar_i64(40)], &mut max_ready_wave);

        assert_eq!(result, Ok(vec![Value::scalar_i64(42)]));
        assert_eq!(max_ready_wave, 1);
    }

    #[test]
    fn dependency_scheduler_keeps_control_flow_barriers_ordered() {
        let jaxpr = make_barrier_order_jaxpr();
        let mut max_ready_wave = 0_usize;
        let result = evaluate_jaxpr_parallel_inner(
            &jaxpr,
            &[Value::scalar_i64(10), Value::scalar_i64(0)],
            &mut max_ready_wave,
        );

        assert_eq!(result, Ok(vec![Value::scalar_i64(-20)]));
        assert_eq!(
            max_ready_wave, 1,
            "ready equations after the switch barrier must not join the pre-barrier wave"
        );
    }

    #[test]
    fn dependency_scheduler_reports_missing_segment_input() {
        let jaxpr = make_missing_dependency_jaxpr();
        let mut max_ready_wave = 0_usize;
        let err =
            evaluate_jaxpr_parallel_inner(&jaxpr, &[Value::scalar_i64(1)], &mut max_ready_wave);

        assert_eq!(err, Err(InterpreterError::MissingVariable(VarId(99))));
    }

    #[test]
    fn dependency_scheduler_executes_long_chain_with_dependency_counts() {
        let length = DEPENDENCY_COUNT_MIN_SEGMENT_LEN;
        let jaxpr = make_dependency_chain_jaxpr(length);
        let mut max_ready_wave = 0_usize;
        let result =
            evaluate_jaxpr_parallel_inner(&jaxpr, &[Value::scalar_i64(7)], &mut max_ready_wave);

        assert_eq!(result, Ok(vec![Value::scalar_i64(7 + length as i64)]));
        assert_eq!(max_ready_wave, 1);
    }

    #[test]
    fn dependency_scheduler_executes_very_long_topological_chain() {
        let length = 1000_usize;
        let jaxpr = make_dependency_chain_jaxpr(length);
        let mut max_ready_wave = 0_usize;
        let result =
            evaluate_jaxpr_parallel_inner(&jaxpr, &[Value::scalar_i64(7)], &mut max_ready_wave);

        assert_eq!(result, Ok(vec![Value::scalar_i64(7 + length as i64)]));
        assert_eq!(max_ready_wave, 1);
    }

    #[test]
    fn dependency_scheduler_direct_i64_add_chain_preserves_wrapping_order() {
        let jaxpr = make_dependency_chain_jaxpr(2);
        let mut max_ready_wave = 0_usize;
        let result = evaluate_jaxpr_parallel_inner(
            &jaxpr,
            &[Value::scalar_i64(i64::MAX)],
            &mut max_ready_wave,
        );

        assert_eq!(result, Ok(vec![Value::scalar_i64(i64::MIN + 1)]));
        assert_eq!(max_ready_wave, 1);
    }

    #[test]
    fn dependency_scheduler_direct_i64_add_chain_preserves_visible_intermediates() {
        let mut jaxpr = make_dependency_chain_jaxpr(3);
        jaxpr.outvars = vec![VarId(2), VarId(4)];
        let mut max_ready_wave = 0_usize;
        let result =
            evaluate_jaxpr_parallel_inner(&jaxpr, &[Value::scalar_i64(7)], &mut max_ready_wave);

        assert_eq!(
            result,
            Ok(vec![Value::scalar_i64(8), Value::scalar_i64(10)])
        );
        assert_eq!(max_ready_wave, 1);
    }

    #[test]
    fn dependency_scheduler_executes_long_branched_fanin_with_dependency_counts() {
        let branches = 16_usize;
        let depth = 8_usize;
        let jaxpr = make_branched_fanin_jaxpr(branches, depth);
        let mut max_ready_wave = 0_usize;
        let result =
            evaluate_jaxpr_parallel_inner(&jaxpr, &[Value::scalar_i64(7)], &mut max_ready_wave);

        assert_eq!(
            result,
            Ok(vec![Value::scalar_i64(
                branches as i64 * (7 + depth as i64)
            )])
        );
        assert_eq!(jaxpr.equations.len(), branches * depth + branches - 1);
        assert_eq!(max_ready_wave, branches);
    }

    #[test]
    fn dependency_scheduler_direct_i64_add_dag_preserves_wrapping_and_wave() {
        let branches = 4_usize;
        let depth = 2_usize;
        let jaxpr = make_branched_fanin_jaxpr(branches, depth);
        let mut max_ready_wave = 0_usize;
        let result = evaluate_jaxpr_parallel_inner(
            &jaxpr,
            &[Value::scalar_i64(i64::MAX)],
            &mut max_ready_wave,
        );
        let branch_value = i64::MAX.wrapping_add(depth as i64);
        let expected = (0..branches).fold(0_i64, |acc, _| acc.wrapping_add(branch_value));

        assert_eq!(result, Ok(vec![Value::scalar_i64(expected)]));
        assert_eq!(max_ready_wave, branches);
    }

    #[test]
    fn dependency_scheduler_direct_i64_add_dag_preserves_visible_intermediates() {
        let mut jaxpr = make_branched_fanin_jaxpr(4, 2);
        jaxpr.outvars = vec![VarId(2), VarId(12)];
        let mut max_ready_wave = 0_usize;
        let result =
            evaluate_jaxpr_parallel_inner(&jaxpr, &[Value::scalar_i64(7)], &mut max_ready_wave);

        assert_eq!(
            result,
            Ok(vec![Value::scalar_i64(8), Value::scalar_i64(36)])
        );
        assert_eq!(max_ready_wave, 4);
    }

    #[test]
    fn dependency_scheduler_direct_f64_tensor_add_dag_matches_interpreter_bits() {
        let jaxpr = make_f64_tensor_branched_fanin_jaxpr(32, 4);
        let input = vector_f64_arg(4);
        let expected =
            fj_interpreters::eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).expect("interpreter");
        let mut max_ready_wave = 0_usize;
        let result = evaluate_jaxpr_parallel_inner(&jaxpr, &[input], &mut max_ready_wave)
            .expect("backend scheduler");

        assert_eq!(tensor_f64_bits(&result[0]), tensor_f64_bits(&expected[0]));
        assert_eq!(
            tensor_f64_bits(&result[0]),
            vec![128.0, 160.0, 192.0, 224.0]
                .into_iter()
                .map(f64::to_bits)
                .collect::<Vec<_>>()
        );
        assert!(
            result[0]
                .as_tensor()
                .expect("tensor")
                .elements
                .as_f64_slice()
                .is_none()
        );
        assert_eq!(max_ready_wave, 32);
    }

    #[test]
    fn dependency_scheduler_direct_f64_tensor_add_dag_preserves_visible_intermediates() {
        let mut jaxpr = make_f64_tensor_branched_fanin_jaxpr(4, 2);
        jaxpr.outvars = vec![VarId(2), VarId(12)];
        let input = vector_f64_arg(2);
        let expected =
            fj_interpreters::eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).expect("interpreter");
        let mut max_ready_wave = 0_usize;
        let result = evaluate_jaxpr_parallel_inner(&jaxpr, &[input], &mut max_ready_wave)
            .expect("backend scheduler");

        assert_eq!(result.len(), expected.len());
        for (actual, expected) in result.iter().zip(&expected) {
            assert_eq!(tensor_f64_bits(actual), tensor_f64_bits(expected));
        }
        assert_eq!(
            tensor_f64_bits(&result[0]),
            vec![1.0, 2.0]
                .into_iter()
                .map(f64::to_bits)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            tensor_f64_bits(&result[1]),
            vec![8.0, 12.0]
                .into_iter()
                .map(f64::to_bits)
                .collect::<Vec<_>>()
        );
        assert_eq!(max_ready_wave, 4);
    }

    #[test]
    fn ready_wave_tensor_work_counts_one_output_work_unit_per_equation() {
        let jaxpr = make_tensor_ready_wave_jaxpr(4);
        let mut env = HashMap::with_capacity_and_hasher(1, Default::default());
        env.insert(VarId(1), vector_f64_arg(8));

        assert_eq!(
            ready_wave_cost(&jaxpr, &env, &[0, 1, 2, 3]),
            ReadyWaveCost {
                tensor_element_work: 32,
                max_tensor_elements: 8
            }
        );
    }

    #[test]
    fn ready_wave_parallel_cost_gate_requires_enough_work() {
        assert!(!should_parallelize_ready_wave_by_cost(
            1,
            usize::MAX,
            usize::MAX,
            4
        ));
        assert!(!should_parallelize_ready_wave_by_cost(
            SCALAR_PARALLEL_READY_WAVE_MIN_LEN,
            0,
            0,
            1
        ));
        assert!(should_parallelize_ready_wave_by_cost(
            SCALAR_PARALLEL_READY_WAVE_MIN_LEN,
            0,
            0,
            4
        ));
        assert!(!should_parallelize_ready_wave_by_cost(
            16,
            TENSOR_PARALLEL_READY_WAVE_MIN_ELEMENTS - 1,
            TENSOR_PARALLEL_INPUT_MIN_ELEMENTS - 1,
            4
        ));
        assert!(should_parallelize_ready_wave_by_cost(
            16,
            TENSOR_PARALLEL_READY_WAVE_MIN_ELEMENTS,
            TENSOR_PARALLEL_INPUT_MIN_ELEMENTS - 1,
            4
        ));
        assert!(should_parallelize_ready_wave_by_cost(
            2,
            0,
            TENSOR_PARALLEL_INPUT_MIN_ELEMENTS,
            4
        ));
    }

    #[test]
    fn dependency_scheduler_reports_missing_input_in_long_dependency_segment() {
        let jaxpr = make_long_missing_dependency_jaxpr(DEPENDENCY_COUNT_MIN_SEGMENT_LEN);
        let mut max_ready_wave = 0_usize;
        let err =
            evaluate_jaxpr_parallel_inner(&jaxpr, &[Value::scalar_i64(1)], &mut max_ready_wave);

        assert_eq!(err, Err(InterpreterError::MissingVariable(VarId(99))));
    }

    #[test]
    fn dependency_scheduler_rejects_duplicate_output_bindings_before_reordering() {
        let jaxpr = make_long_duplicate_output_binding_jaxpr(DEPENDENCY_COUNT_MIN_SEGMENT_LEN);
        let mut max_ready_wave = 0_usize;
        let err =
            evaluate_jaxpr_parallel_inner(&jaxpr, &[Value::scalar_i64(1)], &mut max_ready_wave)
                .expect_err("duplicate output binding should fail closed before scheduling");

        let detail = match err {
            InterpreterError::InvariantViolation { detail } => detail,
            other => format!("unexpected error variant: {other:?}"),
        };
        assert!(
            detail.contains("duplicate equation output binding v2"),
            "error should identify the duplicate binding, got {detail}"
        );
    }

    #[test]
    fn test_parallel_correctness() {
        let backend = CpuBackend::new();
        let jaxpr = make_parallel_independent_jaxpr();
        let args = vec![Value::scalar_i64(11), Value::scalar_i64(-6)];

        let backend_outputs = backend
            .execute(&jaxpr, &args, DeviceId(0))
            .expect("backend execution should succeed");
        let interpreter_outputs = fj_interpreters::eval_jaxpr(&jaxpr, &args)
            .expect("interpreter execution should succeed");

        assert_eq!(backend_outputs, interpreter_outputs);
    }

    #[test]
    fn cpu_executes_switch_with_sub_jaxprs() {
        let backend = CpuBackend::new();
        let jaxpr = make_switch_control_flow_jaxpr();
        let outputs = backend
            .execute(
                &jaxpr,
                &[Value::scalar_i64(2), Value::scalar_i64(7)],
                DeviceId(0),
            )
            .expect("switch with sub_jaxprs should execute on CPU");
        assert_eq!(outputs, vec![Value::scalar_i64(49)]);
    }

    #[test]
    fn test_parallel_no_data_race() {
        use std::sync::Arc;
        use std::thread;

        let backend = Arc::new(CpuBackend::new());
        let jaxpr = Arc::new(make_parallel_independent_jaxpr());

        let mut workers = Vec::new();
        for worker_id in 0_i64..8 {
            let backend = Arc::clone(&backend);
            let jaxpr = Arc::clone(&jaxpr);
            workers.push(thread::spawn(move || {
                for offset in 0_i64..64 {
                    let a = worker_id * 10 + offset;
                    let b = -offset;
                    let outputs = backend
                        .execute(
                            &jaxpr,
                            &[Value::scalar_i64(a), Value::scalar_i64(b)],
                            DeviceId(0),
                        )
                        .expect("concurrent execution should succeed");
                    assert_eq!(outputs, vec![Value::scalar_i64(-(a + b))]);
                }
            }));
        }

        for worker in workers {
            worker.join().expect("worker thread should complete");
        }
    }

    #[test]
    fn cpu_backend_allocate_and_access() {
        let backend = CpuBackend::new();
        let buf = backend
            .allocate(256, DeviceId(0))
            .expect("alloc should succeed");
        assert_eq!(buf.size(), 256);
        assert_eq!(buf.device(), DeviceId(0));
        assert!(buf.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn cpu_backend_allocate_invalid_device() {
        let backend = CpuBackend::new();
        let err = backend.allocate(256, DeviceId(1)).expect_err("should fail");
        assert!(matches!(err, BackendError::AllocationFailed { .. }));
    }

    #[test]
    fn cpu_backend_transfer_same_device() {
        let backend = CpuBackend::new();
        let buf = Buffer::new(vec![1, 2, 3], DeviceId(0));
        let transferred = backend
            .transfer(&buf, DeviceId(0))
            .expect("transfer should succeed");
        assert_eq!(transferred.as_bytes(), &[1, 2, 3]);
        assert_eq!(transferred.device(), DeviceId(0));
    }

    #[test]
    fn cpu_backend_transfer_cross_device() {
        let backend = CpuBackend::with_device_count(2);
        let buf = Buffer::new(vec![10, 20, 30], DeviceId(0));
        let transferred = backend
            .transfer(&buf, DeviceId(1))
            .expect("cross-device transfer");
        assert_eq!(transferred.as_bytes(), &[10, 20, 30]);
        assert_eq!(transferred.device(), DeviceId(1));
        // Original buffer unchanged
        assert_eq!(buf.device(), DeviceId(0));
    }

    #[test]
    fn cpu_backend_transfer_invalid_target() {
        let backend = CpuBackend::new();
        let buf = Buffer::new(vec![1], DeviceId(0));
        let err = backend
            .transfer(&buf, DeviceId(5))
            .expect_err("should fail");
        assert!(matches!(err, BackendError::TransferFailed { .. }));
    }

    #[test]
    fn cpu_backend_version_string() {
        let backend = CpuBackend::new();
        assert!(backend.version().starts_with("fj-backend-cpu/"));
    }

    #[test]
    fn cpu_backend_buffer_roundtrip_preserves_data() {
        // Contract p2c006.strict.inv003: device_put/device_get round-trip
        let backend = CpuBackend::new();
        let original = vec![0xCA, 0xFE, 0xBA, 0xBE];
        let buf = Buffer::new(original.clone(), DeviceId(0));
        let data = buf.into_bytes();
        assert_eq!(original, data);

        // Through allocate + write
        let mut buf = backend.allocate(4, DeviceId(0)).expect("alloc");
        buf.as_bytes_mut().copy_from_slice(&original);
        assert_eq!(buf.as_bytes(), &original[..]);
    }

    #[test]
    fn cpu_backend_capabilities_supported_dtypes() {
        let backend = CpuBackend::new();
        let caps = backend.capabilities();
        assert!(caps.supported_dtypes.contains(&DType::F64));
        assert!(caps.supported_dtypes.contains(&DType::I64));
    }

    #[test]
    fn cpu_backend_capabilities_rank_limit() {
        let backend = CpuBackend::new();
        let caps = backend.capabilities();
        assert!(caps.max_tensor_rank >= 4);
    }

    #[test]
    fn cpu_backend_capabilities_memory_unlimited() {
        let backend = CpuBackend::new();
        let caps = backend.capabilities();
        assert!(caps.memory_limit_bytes.is_none());
    }

    #[test]
    fn cpu_backend_single_device_not_multi() {
        let backend = CpuBackend::new();
        assert!(!backend.capabilities().multi_device);
    }

    #[test]
    fn cpu_backend_multi_device_caps() {
        let backend = CpuBackend::with_device_count(2);
        assert!(backend.capabilities().multi_device);
    }

    // ── Registry tests ────────────────────────────────────────────

    use fj_runtime::backend::BackendRegistry;
    use fj_runtime::device::DevicePlacement;

    #[test]
    fn registry_get_by_name() {
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
        assert!(registry.get("cpu").is_some());
        assert!(registry.get("gpu").is_none());
    }

    #[test]
    fn registry_default_backend() {
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
        let default = registry.default_backend().expect("should have default");
        assert_eq!(default.name(), "cpu");
    }

    #[test]
    fn registry_available_backends() {
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
        assert_eq!(registry.available_backends(), vec!["cpu"]);
    }

    #[test]
    fn registry_resolve_default_placement() {
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
        let (backend, device) = registry
            .resolve_placement(&DevicePlacement::Default, None)
            .expect("should resolve");
        assert_eq!(backend.name(), "cpu");
        assert_eq!(device, DeviceId(0));
    }

    #[test]
    fn registry_resolve_explicit_backend() {
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
        let (backend, device) = registry
            .resolve_placement(&DevicePlacement::Default, Some("cpu"))
            .expect("should resolve");
        assert_eq!(backend.name(), "cpu");
        assert_eq!(device, DeviceId(0));
    }

    #[test]
    fn registry_resolve_unavailable_backend() {
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
        let result = registry.resolve_placement(&DevicePlacement::Default, Some("gpu"));
        assert!(matches!(
            result,
            Err(BackendError::Unavailable { backend }) if backend == "gpu"
        ));
    }

    #[test]
    fn registry_resolve_with_fallback() {
        // Contract p2c006.hardened.inv008: missing backend → CPU fallback
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
        let (backend, device, fell_back) = registry
            .resolve_with_fallback(&DevicePlacement::Default, Some("gpu"))
            .expect("should fallback to CPU");
        assert_eq!(backend.name(), "cpu");
        assert_eq!(device, DeviceId(0));
        assert!(fell_back, "should report fallback occurred");
    }

    #[test]
    fn registry_fallback_rebinds_invalid_explicit_device_to_cpu_default() {
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
        let (backend, device, fell_back) = registry
            .resolve_with_fallback(&DevicePlacement::Explicit(DeviceId(9)), Some("gpu"))
            .expect("fallback should choose a runnable default device");
        assert_eq!(backend.name(), "cpu");
        assert_eq!(device, DeviceId(0));
        assert!(fell_back, "should report fallback occurred");
    }

    #[test]
    fn registry_resolve_no_fallback_needed() {
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
        let (backend, _, fell_back) = registry
            .resolve_with_fallback(&DevicePlacement::Default, Some("cpu"))
            .expect("should resolve directly");
        assert_eq!(backend.name(), "cpu");
        assert!(!fell_back, "no fallback should be needed");
    }

    #[test]
    fn registry_resolve_explicit_device_id() {
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::with_device_count(4))]);
        let (backend, device) = registry
            .resolve_placement(&DevicePlacement::Explicit(DeviceId(2)), Some("cpu"))
            .expect("should resolve");
        assert_eq!(backend.name(), "cpu");
        assert_eq!(device, DeviceId(2));
    }

    #[test]
    fn registry_rejects_invalid_explicit_device_id() {
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
        let err = registry
            .resolve_placement(&DevicePlacement::Explicit(DeviceId(1)), Some("cpu"))
            .err()
            .expect("invalid explicit device should fail");
        let msg = err.to_string();
        assert!(msg.contains("device:1"), "error should identify device");
        assert!(msg.contains("cpu"), "error should identify backend");
    }

    // ── Category 1: All primitives execute correctly on CPU ───────

    #[test]
    fn cpu_executes_add_one() {
        let backend = CpuBackend::new();
        let jaxpr = build_program(ProgramSpec::AddOne);
        let result = backend
            .execute(&jaxpr, &[Value::scalar_i64(41)], DeviceId(0))
            .expect("should succeed");
        assert_eq!(result, vec![Value::scalar_i64(42)]);
    }

    #[test]
    fn cpu_executes_sin() {
        let backend = CpuBackend::new();
        let jaxpr = build_program(ProgramSpec::SinX);
        let result = backend
            .execute(&jaxpr, &[Value::scalar_f64(0.0)], DeviceId(0))
            .expect("should succeed");
        let val = result[0].as_f64_scalar().expect("f64");
        assert!(val.abs() < 1e-10, "sin(0) should be 0");
    }

    #[test]
    fn cpu_executes_cos() {
        let backend = CpuBackend::new();
        let jaxpr = build_program(ProgramSpec::CosX);
        let result = backend
            .execute(&jaxpr, &[Value::scalar_f64(0.0)], DeviceId(0))
            .expect("should succeed");
        let val = result[0].as_f64_scalar().expect("f64");
        assert!((val - 1.0).abs() < 1e-10, "cos(0) should be 1");
    }

    // ── Category 3: Backend selection correctness ─────────────────

    #[test]
    fn backend_selection_routes_to_named_backend() {
        let registry = BackendRegistry::new(vec![Box::new(CpuBackend::new())]);
        let (backend, _) = registry
            .resolve_placement(&DevicePlacement::Default, Some("cpu"))
            .expect("cpu should resolve");
        assert_eq!(backend.name(), "cpu");
    }

    // ── Category 5: Memory layout ─────────────────────────────────

    #[test]
    fn buffer_data_is_contiguous() {
        let backend = CpuBackend::new();
        let mut buf = backend.allocate(16, DeviceId(0)).expect("alloc");
        // Write pattern and verify contiguity
        for (i, byte) in buf.as_bytes_mut().iter_mut().enumerate() {
            *byte = i as u8;
        }
        let data = buf.as_bytes();
        for (i, &byte) in data.iter().enumerate().take(16) {
            assert_eq!(byte, i as u8);
        }
    }

    // ── Category 6: Multi-backend isolation ───────────────────────

    #[test]
    fn two_cpu_backend_instances_are_independent() {
        let backend_a = CpuBackend::new();
        let backend_b = CpuBackend::with_device_count(2);

        // They should have different device counts
        assert_eq!(backend_a.devices().len(), 1);
        assert_eq!(backend_b.devices().len(), 2);

        // Execution on one doesn't affect the other
        let jaxpr = build_program(ProgramSpec::Add2);
        let result_a = backend_a
            .execute(
                &jaxpr,
                &[Value::scalar_i64(1), Value::scalar_i64(2)],
                DeviceId(0),
            )
            .expect("backend_a");
        let result_b = backend_b
            .execute(
                &jaxpr,
                &[Value::scalar_i64(3), Value::scalar_i64(4)],
                DeviceId(0),
            )
            .expect("backend_b");
        assert_eq!(result_a, vec![Value::scalar_i64(3)]);
        assert_eq!(result_b, vec![Value::scalar_i64(7)]);
    }

    // ── Structured logging contract ───────────────────────────────

    #[test]
    fn test_backend_cpu_test_log_schema_contract() {
        let fixture_id =
            fj_test_utils::fixture_id_from_json(&("backend-cpu", "execute")).expect("digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_backend_cpu_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }

    // ── Property tests ────────────────────────────────────────────

    proptest::proptest! {
        #[test]
        fn prop_cpu_backend_allocation_size_matches(size in 0_usize..4096) {
            let backend = CpuBackend::new();
            let buf = backend.allocate(size, DeviceId(0)).expect("alloc");
            proptest::prop_assert_eq!(buf.size(), size);
        }

        #[test]
        fn prop_cpu_backend_transfer_preserves_data(
            data in proptest::collection::vec(proptest::prelude::any::<u8>(), 0..512)
        ) {
            let backend = CpuBackend::with_device_count(2);
            let buf = Buffer::new(data.clone(), DeviceId(0));
            let transferred = backend.transfer(&buf, DeviceId(1)).expect("transfer");
            proptest::prop_assert_eq!(transferred.as_bytes(), &data[..]);
            proptest::prop_assert_eq!(transferred.device(), DeviceId(1));
        }

        #[test]
        fn prop_cpu_backend_device_count_matches(count in 1_u32..16) {
            let backend = CpuBackend::with_device_count(count);
            proptest::prop_assert_eq!(backend.devices().len(), count as usize);
        }

        #[test]
        fn metamorphic_eval_determinism(a in -100i64..100, b in -100i64..100) {
            let jaxpr = build_program(ProgramSpec::Add2);
            let args = vec![Value::scalar_i64(a), Value::scalar_i64(b)];
            let result1 = evaluate_jaxpr_parallel(&jaxpr, &args).expect("eval1");
            let result2 = evaluate_jaxpr_parallel(&jaxpr, &args).expect("eval2");
            proptest::prop_assert_eq!(result1, result2, "evaluation must be deterministic");
        }

        #[test]
        fn metamorphic_eval_commutativity_add(a in -100i64..100, b in -100i64..100) {
            let jaxpr = build_program(ProgramSpec::Add2);
            let result_ab = evaluate_jaxpr_parallel(&jaxpr, &[Value::scalar_i64(a), Value::scalar_i64(b)]).expect("ab");
            let result_ba = evaluate_jaxpr_parallel(&jaxpr, &[Value::scalar_i64(b), Value::scalar_i64(a)]).expect("ba");
            proptest::prop_assert_eq!(result_ab, result_ba, "add should be commutative");
        }

        #[test]
        fn metamorphic_parallel_vs_interpreter_equivalence(x in -50i64..50) {
            let jaxpr = build_program(ProgramSpec::Square);
            let args = vec![Value::scalar_i64(x)];
            let parallel_result = evaluate_jaxpr_parallel(&jaxpr, &args).expect("parallel");
            let interp_result = fj_interpreters::eval_jaxpr(&jaxpr, &args).expect("interp");
            proptest::prop_assert_eq!(parallel_result, interp_result, "parallel must match interpreter");
        }

        #[test]
        fn metamorphic_wide_dag_determinism(
            vals in proptest::collection::vec(-100i64..100, 4..8)
        ) {
            let jaxpr = make_parallel_independent_jaxpr();
            if vals.len() >= 2 {
                let args = vec![Value::scalar_i64(vals[0]), Value::scalar_i64(vals[1])];
                let r1 = evaluate_jaxpr_parallel(&jaxpr, &args).expect("r1");
                let r2 = evaluate_jaxpr_parallel(&jaxpr, &args).expect("r2");
                proptest::prop_assert_eq!(r1, r2, "wide DAG must be deterministic");
            }
        }
    }
}
