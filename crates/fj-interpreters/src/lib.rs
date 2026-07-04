#![feature(portable_simd)]
#![forbid(unsafe_code)]

pub mod partial_eval;
pub mod staging;

use fj_core::{
    Atom, DType, Equation, Jaxpr, Literal, Primitive, Shape, TensorValue, Value, ValueError, VarId,
};
use fj_lax::{EvalError, eval_primitive, eval_primitive_multi};
use rustc_hash::FxHashMap;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterpreterError {
    InputArity {
        expected: usize,
        actual: usize,
    },
    ConstArity {
        expected: usize,
        actual: usize,
    },
    MissingVariable(VarId),
    UnexpectedOutputArity {
        primitive: fj_core::Primitive,
        expected: usize,
        actual: usize,
    },
    InvariantViolation {
        detail: String,
    },
    Primitive(EvalError),
}

impl std::fmt::Display for InterpreterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputArity { expected, actual } => {
                write!(
                    f,
                    "input arity mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::ConstArity { expected, actual } => {
                write!(
                    f,
                    "const arity mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::MissingVariable(var) => write!(f, "missing variable v{}", var.0),
            Self::UnexpectedOutputArity {
                primitive,
                expected,
                actual,
            } => write!(
                f,
                "primitive {} returned {} outputs for {} bindings",
                primitive.as_str(),
                actual,
                expected
            ),
            Self::InvariantViolation { detail } => {
                write!(f, "interpreter invariant violated: {detail}")
            }
            Self::Primitive(err) => write!(f, "primitive eval failed: {err}"),
        }
    }
}

impl std::error::Error for InterpreterError {}

impl From<EvalError> for InterpreterError {
    fn from(value: EvalError) -> Self {
        Self::Primitive(value)
    }
}

pub fn eval_jaxpr(jaxpr: &Jaxpr, args: &[Value]) -> Result<Vec<Value>, InterpreterError> {
    eval_jaxpr_with_consts(jaxpr, &[], args)
}

fn resolve_equation_inputs(
    equation: &Equation,
    env: &FxHashMap<VarId, Value>,
) -> Result<Vec<Value>, InterpreterError> {
    let mut resolved = Vec::with_capacity(equation.inputs.len());
    for atom in &equation.inputs {
        match atom {
            Atom::Var(var) => {
                let value = env
                    .get(var)
                    .cloned()
                    .ok_or(InterpreterError::MissingVariable(*var))?;
                resolved.push(value);
            }
            Atom::Lit(lit) => resolved.push(Value::Scalar(*lit)),
        }
    }
    Ok(resolved)
}

fn scalar_literal_from_value(primitive: Primitive, value: &Value) -> Result<Literal, EvalError> {
    match value {
        Value::Scalar(literal) => Ok(*literal),
        Value::Tensor(tensor) => {
            if tensor.shape != Shape::scalar() {
                return Err(EvalError::ShapeMismatch {
                    primitive,
                    left: tensor.shape.clone(),
                    right: Shape::scalar(),
                });
            }
            if tensor.elements.len() != 1 {
                return Err(EvalError::InvalidTensor(ValueError::ElementCountMismatch {
                    shape: tensor.shape.clone(),
                    expected_count: 1,
                    actual_count: tensor.elements.len(),
                }));
            }
            Ok(tensor.elements[0])
        }
    }
}

fn predicate_value_to_bool(primitive: Primitive, value: &Value) -> Result<bool, EvalError> {
    match scalar_literal_from_value(primitive, value)? {
        Literal::Bool(value) => Ok(value),
        Literal::I32(value) => Ok(value != 0),
        Literal::I64(value) => Ok(value != 0),
        Literal::U32(value) => Ok(value != 0),
        Literal::U64(value) => Ok(value != 0),
        Literal::BF16Bits(bits) => Ok(Literal::BF16Bits(bits)
            .as_f64()
            .is_some_and(|value| value != 0.0)),
        Literal::F16Bits(bits) => Ok(Literal::F16Bits(bits)
            .as_f64()
            .is_some_and(|value| value != 0.0)),
        Literal::F32Bits(bits) => Ok(f32::from_bits(bits) != 0.0),
        Literal::F64Bits(bits) => Ok(f64::from_bits(bits) != 0.0),
        Literal::Complex64Bits(..) | Literal::Complex128Bits(..) => Err(EvalError::TypeMismatch {
            primitive,
            detail: "predicate must be boolean or numeric",
        }),
    }
}

fn value_shape(value: &Value) -> Shape {
    match value {
        Value::Scalar(_) => Shape::scalar(),
        Value::Tensor(tensor) => tensor.shape.clone(),
    }
}

fn map_sub_jaxpr_error(primitive: Primitive, context: &str, err: InterpreterError) -> EvalError {
    EvalError::Unsupported {
        primitive,
        detail: format!("{context} sub_jaxpr failed: {err}"),
    }
}

fn evaluate_switch_sub_jaxprs(
    equation: &Equation,
    resolved: &[Value],
) -> Result<Vec<Value>, InterpreterError> {
    let index_value = match resolved.first() {
        Some(value) => value,
        None => {
            return Err(InterpreterError::Primitive(EvalError::ArityMismatch {
                primitive: Primitive::Switch,
                expected: 1,
                actual: 0,
            }));
        }
    };
    let index_literal = scalar_literal_from_value(Primitive::Switch, index_value)
        .map_err(InterpreterError::Primitive)?;

    let expected_branches = equation
        .params
        .get("num_branches")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(equation.sub_jaxprs.len());
    if expected_branches != equation.sub_jaxprs.len() {
        return Err(InterpreterError::Primitive(EvalError::Unsupported {
            primitive: Primitive::Switch,
            detail: format!(
                "switch declares {expected_branches} branches but carries {} sub_jaxprs",
                equation.sub_jaxprs.len()
            ),
        }));
    }

    let branch_idx = clamped_switch_index(index_literal, equation.sub_jaxprs.len(), index_value)?;
    let selected_branch = equation.sub_jaxprs.get(branch_idx).ok_or_else(|| {
        InterpreterError::Primitive(EvalError::Unsupported {
            primitive: Primitive::Switch,
            detail: "switch requires at least one branch".to_owned(),
        })
    })?;

    let provided_bindings = &resolved[1..];
    let expected_bindings = selected_branch.constvars.len() + selected_branch.invars.len();
    if provided_bindings.len() != expected_bindings {
        return Err(InterpreterError::InputArity {
            expected: expected_bindings,
            actual: provided_bindings.len(),
        });
    }

    let (const_values, branch_args) = provided_bindings.split_at(selected_branch.constvars.len());
    eval_jaxpr_with_consts(selected_branch, const_values, branch_args)
}

fn clamped_switch_index(
    literal: Literal,
    branch_count: usize,
    original_value: &Value,
) -> Result<usize, InterpreterError> {
    if branch_count == 0 {
        return Err(InterpreterError::Primitive(EvalError::Unsupported {
            primitive: Primitive::Switch,
            detail: "switch requires at least one branch".to_owned(),
        }));
    }

    let last_branch = branch_count - 1;
    match literal {
        Literal::I64(value) => {
            if value <= 0 {
                Ok(0)
            } else {
                Ok((value as u64).min(last_branch as u64) as usize)
            }
        }
        Literal::U32(value) => Ok((value as usize).min(last_branch)),
        Literal::U64(value) => Ok(value.min(last_branch as u64) as usize),
        Literal::Bool(value) => Ok(usize::from(value).min(last_branch)),
        _ => Err(InterpreterError::Primitive(EvalError::Unsupported {
            primitive: Primitive::Switch,
            detail: format!(
                "switch index must be integer, got {:?}",
                original_value.dtype()
            ),
        })),
    }
}

fn evaluate_cond_sub_jaxprs(
    equation: &Equation,
    resolved: &[Value],
) -> Result<Vec<Value>, InterpreterError> {
    let predicate_value = match resolved.first() {
        Some(value) => value,
        None => {
            return Err(InterpreterError::Primitive(EvalError::ArityMismatch {
                primitive: Primitive::Cond,
                expected: 1,
                actual: 0,
            }));
        }
    };
    if equation.sub_jaxprs.len() != 2 {
        return Err(InterpreterError::Primitive(EvalError::Unsupported {
            primitive: Primitive::Cond,
            detail: format!(
                "cond expects exactly 2 sub_jaxprs, got {}",
                equation.sub_jaxprs.len()
            ),
        }));
    }

    let predicate = predicate_value_to_bool(Primitive::Cond, predicate_value)
        .map_err(InterpreterError::Primitive)?;

    let selected_branch = if predicate {
        &equation.sub_jaxprs[0]
    } else {
        &equation.sub_jaxprs[1]
    };
    let provided_bindings = &resolved[1..];
    let expected_bindings = selected_branch.constvars.len() + selected_branch.invars.len();
    if provided_bindings.len() != expected_bindings {
        return Err(InterpreterError::InputArity {
            expected: expected_bindings,
            actual: provided_bindings.len(),
        });
    }

    let (const_values, branch_args) = provided_bindings.split_at(selected_branch.constvars.len());
    eval_jaxpr_with_consts(selected_branch, const_values, branch_args)
}

fn evaluate_while_sub_jaxprs(
    equation: &Equation,
    resolved: &[Value],
) -> Result<Vec<Value>, InterpreterError> {
    if equation.sub_jaxprs.len() != 2 {
        return Err(InterpreterError::Primitive(EvalError::Unsupported {
            primitive: Primitive::While,
            detail: format!(
                "while expects exactly 2 sub_jaxprs, got {}",
                equation.sub_jaxprs.len()
            ),
        }));
    }
    let cond_jaxpr = &equation.sub_jaxprs[0];
    let body_jaxpr = &equation.sub_jaxprs[1];

    let max_iter: usize = match equation.params.get("max_iter") {
        Some(raw) => raw.parse().map_err(|_| {
            InterpreterError::Primitive(EvalError::Unsupported {
                primitive: Primitive::While,
                detail: format!("invalid max_iter value: {raw}"),
            })
        })?,
        None => 1000,
    };

    let const_count = cond_jaxpr.constvars.len() + body_jaxpr.constvars.len();
    if resolved.len() < const_count {
        return Err(InterpreterError::InputArity {
            expected: const_count,
            actual: resolved.len(),
        });
    }
    let (const_bindings, carry_bindings) = resolved.split_at(const_count);
    let (cond_consts, body_consts) = const_bindings.split_at(cond_jaxpr.constvars.len());
    if cond_jaxpr.invars.len() != carry_bindings.len() {
        return Err(InterpreterError::InputArity {
            expected: const_count + cond_jaxpr.invars.len(),
            actual: resolved.len(),
        });
    }
    if body_jaxpr.invars.len() != carry_bindings.len() {
        return Err(InterpreterError::InputArity {
            expected: const_count + body_jaxpr.invars.len(),
            actual: resolved.len(),
        });
    }

    let mut carry = carry_bindings.to_vec();
    let init_shapes: Vec<Shape> = carry.iter().map(value_shape).collect();
    let init_dtypes: Vec<_> = carry.iter().map(Value::dtype).collect();

    // The cond/body sub-jaxprs are re-evaluated once per iteration. Their
    // slot/liveness analysis is value-independent, so derive it ONCE and reuse a
    // single env buffer per sub-jaxpr across all iterations — eliminating the
    // per-iteration re-analysis + env/last_use allocations that otherwise dominate
    // the dispatch cost of a loop with a cheap body. Falls back to the
    // self-contained `eval_jaxpr_with_consts` for any sub-jaxpr that is not
    // dense-eligible (sparse/huge var ids), so behavior is unchanged.
    let cond_plan = build_dense_plan(cond_jaxpr);
    let body_plan = build_dense_plan(body_jaxpr);
    let mut cond_env: Vec<Option<Value>> = vec![None; cond_plan.as_ref().map_or(0, |p| p.slots)];
    let mut body_env: Vec<Option<Value>> = vec![None; body_plan.as_ref().map_or(0, |p| p.slots)];
    // Reusable per-iteration buffers: one input-resolution scratch shared by both
    // sub-jaxprs (they run sequentially, never concurrently), plus output buffers
    // for the cond result and the next carry — so a converged-body loop performs
    // ZERO heap allocations per iteration (the carry is swapped, not reallocated).
    let mut scratch: Vec<Value> = Vec::new();
    let mut scalar_buffers = ScalarPlanBuffers::default();
    let mut cond_outputs: Vec<Value> = Vec::new();
    let mut next_carry: Vec<Value> = Vec::new();

    for _ in 0..max_iter {
        match &cond_plan {
            Some(plan) => run_dense_plan_into(
                cond_jaxpr,
                cond_consts,
                &carry,
                &mut cond_env,
                plan,
                &mut scratch,
                &mut cond_outputs,
                &mut scalar_buffers,
            ),
            None => eval_jaxpr_with_consts(cond_jaxpr, cond_consts, &carry).map(|o| {
                cond_outputs = o;
            }),
        }
        .map_err(|err| {
            InterpreterError::Primitive(map_sub_jaxpr_error(Primitive::While, "while cond", err))
        })?;
        if cond_outputs.len() != 1 {
            return Err(InterpreterError::InvariantViolation {
                detail: format!(
                    "while cond sub_jaxpr returned {} outputs; expected 1",
                    cond_outputs.len()
                ),
            });
        }
        if !predicate_value_to_bool(Primitive::While, &cond_outputs[0])
            .map_err(InterpreterError::Primitive)?
        {
            return Ok(carry);
        }

        match &body_plan {
            Some(plan) => run_dense_plan_into(
                body_jaxpr,
                body_consts,
                &carry,
                &mut body_env,
                plan,
                &mut scratch,
                &mut next_carry,
                &mut scalar_buffers,
            ),
            None => eval_jaxpr_with_consts(body_jaxpr, body_consts, &carry).map(|o| {
                next_carry = o;
            }),
        }
        .map_err(|err| {
            InterpreterError::Primitive(map_sub_jaxpr_error(Primitive::While, "while body", err))
        })?;
        if next_carry.len() != carry.len() {
            return Err(InterpreterError::InvariantViolation {
                detail: format!(
                    "while body sub_jaxpr returned {} carry values; expected {}",
                    next_carry.len(),
                    carry.len()
                ),
            });
        }
        for (idx, value) in next_carry.iter().enumerate() {
            let new_shape = value_shape(value);
            if new_shape != init_shapes[idx] {
                return Err(InterpreterError::Primitive(EvalError::ShapeChanged {
                    primitive: Primitive::While,
                    detail: format!(
                        "carry element {idx} changed shape from {:?} to {:?}",
                        init_shapes[idx].dims, new_shape.dims
                    ),
                }));
            }
            let new_dtype = value.dtype();
            if new_dtype != init_dtypes[idx] {
                return Err(InterpreterError::Primitive(EvalError::TypeMismatch {
                    primitive: Primitive::While,
                    detail: "while body changed carry dtype",
                }));
            }
        }
        // Swap (not move): `carry` adopts this iteration's result while the old
        // `carry` allocation is recycled as next iteration's `next_carry` buffer.
        std::mem::swap(&mut carry, &mut next_carry);
    }

    Err(InterpreterError::Primitive(
        EvalError::MaxIterationsExceeded {
            primitive: Primitive::While,
            max_iterations: max_iter,
        },
    ))
}

fn evaluate_scan_sub_jaxprs(
    equation: &Equation,
    resolved: &[Value],
) -> Result<Vec<Value>, InterpreterError> {
    if equation.sub_jaxprs.len() != 1 {
        return Err(InterpreterError::Primitive(EvalError::Unsupported {
            primitive: Primitive::Scan,
            detail: format!(
                "scan expects exactly 1 body sub_jaxpr, got {}",
                equation.sub_jaxprs.len()
            ),
        }));
    }
    let body_jaxpr = &equation.sub_jaxprs[0];
    let carry_count = body_jaxpr.invars.len().checked_sub(1).ok_or_else(|| {
        InterpreterError::Primitive(EvalError::Unsupported {
            primitive: Primitive::Scan,
            detail: "scan body requires carry inputs plus one xs input".to_owned(),
        })
    })?;
    if body_jaxpr.outvars.len() < carry_count {
        return Err(InterpreterError::InvariantViolation {
            detail: format!(
                "scan body sub_jaxpr returned {} values for {carry_count} carries",
                body_jaxpr.outvars.len()
            ),
        });
    }

    let const_count = body_jaxpr.constvars.len();
    let expected_bindings = const_count + carry_count + 1;
    if resolved.len() != expected_bindings {
        return Err(InterpreterError::InputArity {
            expected: expected_bindings,
            actual: resolved.len(),
        });
    }
    if equation.outputs.len() != body_jaxpr.outvars.len() {
        return Err(InterpreterError::UnexpectedOutputArity {
            primitive: Primitive::Scan,
            expected: equation.outputs.len(),
            actual: body_jaxpr.outvars.len(),
        });
    }

    let (const_values, state_inputs) = resolved.split_at(const_count);
    let (carry_inputs, xs_inputs) = state_inputs.split_at(carry_count);
    let xs = &xs_inputs[0];
    let scan_len = scan_input_len(xs)?;
    let y_count = body_jaxpr.outvars.len() - carry_count;
    if scan_len == 0 && y_count > 0 {
        return Err(InterpreterError::Primitive(EvalError::Unsupported {
            primitive: Primitive::Scan,
            detail: "zero-length functional scan outputs require abstract output shapes".to_owned(),
        }));
    }

    let reverse = equation
        .params
        .get("reverse")
        .is_some_and(|value| value == "true");
    if let Some(result) =
        try_eval_scan_i64_add_emit(equation, body_jaxpr, carry_inputs, xs, scan_len, reverse)
    {
        return result;
    }
    if let Some(result) =
        try_eval_scan_f32_add_emit(equation, body_jaxpr, carry_inputs, xs, scan_len, reverse)
    {
        return result;
    }

    let mut carry = carry_inputs.to_vec();
    let init_shapes: Vec<Shape> = carry.iter().map(value_shape).collect();
    let init_dtypes: Vec<_> = carry.iter().map(Value::dtype).collect();
    let mut per_y = vec![Vec::with_capacity(scan_len); y_count];

    // The body sub-jaxpr is re-evaluated once per scan step. Its slot/liveness
    // analysis is value-independent, so derive it ONCE and reuse a single env
    // buffer + scratch + output buffer across all steps — eliminating the
    // per-step re-analysis and env/last_use/scratch/output allocations that
    // otherwise dominate a scan with a cheap body. Falls back to the
    // self-contained `eval_jaxpr_with_consts` when the body is not dense-eligible.
    let body_plan = build_dense_plan(body_jaxpr);
    let mut body_env: Vec<Option<Value>> = vec![None; body_plan.as_ref().map_or(0, |p| p.slots)];
    let mut scratch: Vec<Value> = Vec::new();
    let mut scalar_buffers = ScalarPlanBuffers::default();
    let mut body_out: Vec<Value> = Vec::new();

    let scan_context = ScanIterationContext {
        body_jaxpr,
        body_plan: &body_plan,
        const_values,
        xs,
        init_shapes: &init_shapes,
        init_dtypes: &init_dtypes,
    };
    let mut body_args = Vec::with_capacity(carry_count + 1);
    if reverse {
        for scan_idx in (0..scan_len).rev() {
            evaluate_scan_iteration(
                &scan_context,
                scan_idx,
                &mut carry,
                &mut per_y,
                &mut body_args,
                &mut body_env,
                &mut scratch,
                &mut scalar_buffers,
                &mut body_out,
            )?;
        }
    } else {
        for scan_idx in 0..scan_len {
            evaluate_scan_iteration(
                &scan_context,
                scan_idx,
                &mut carry,
                &mut per_y,
                &mut body_args,
                &mut body_env,
                &mut scratch,
                &mut scalar_buffers,
                &mut body_out,
            )?;
        }
    }

    if reverse {
        for values in &mut per_y {
            values.reverse();
        }
    }

    let mut outputs = carry;
    for values in per_y {
        let stacked = TensorValue::stack_axis0(&values)
            .map_err(|error| InterpreterError::Primitive(EvalError::InvalidTensor(error)))?;
        outputs.push(Value::Tensor(stacked));
    }
    Ok(outputs)
}

fn try_eval_scan_i64_add_emit(
    equation: &Equation,
    body_jaxpr: &Jaxpr,
    carry_inputs: &[Value],
    xs: &Value,
    scan_len: usize,
    reverse: bool,
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !scan_equation_params_are_i64_add_emit_safe(equation, reverse)
        || !equation.effects.is_empty()
        || !is_i64_add_emit_scan_body(body_jaxpr)
        || carry_inputs.len() != 1
    {
        return None;
    }
    let Value::Scalar(Literal::I64(init_carry)) = carry_inputs.first()? else {
        return None;
    };

    let mut carry = *init_carry;
    let mut ys = vec![0_i64; scan_len];
    let mut y_shape = Shape::vector(1);
    match xs {
        Value::Scalar(Literal::I64(x)) => {
            if scan_len != 1 {
                return None;
            }
            carry = carry.wrapping_add(*x);
            ys[0] = carry;
        }
        Value::Tensor(tensor) => {
            if tensor.dtype != DType::I64 || tensor.shape.rank() != 1 || tensor.len() != scan_len {
                return None;
            }
            if let Some(values) = tensor.elements.as_i64_slice() {
                scan_i64_values_into(values, reverse, &mut carry, &mut ys);
            } else if reverse {
                for idx in (0..scan_len).rev() {
                    let x = tensor.elements.get(idx).copied()?.as_i64()?;
                    carry = carry.wrapping_add(x);
                    ys[idx] = carry;
                }
            } else {
                for (idx, y) in ys.iter_mut().enumerate() {
                    let x = tensor.elements.get(idx).copied()?.as_i64()?;
                    carry = carry.wrapping_add(x);
                    *y = carry;
                }
            }
            y_shape = tensor.shape.clone();
        }
        Value::Scalar(_) => return None,
    }

    Some(
        TensorValue::new_i64_values(y_shape, ys)
            .map(|tensor| vec![Value::scalar_i64(carry), Value::Tensor(tensor)])
            .map_err(|error| InterpreterError::Primitive(EvalError::InvalidTensor(error))),
    )
}

fn scan_equation_params_are_i64_add_emit_safe(equation: &Equation, reverse: bool) -> bool {
    if reverse {
        equation.params.len() == 1
            && equation
                .params
                .get("reverse")
                .is_some_and(|value| value == "true")
    } else {
        equation.params.is_empty()
    }
}

fn scan_i64_values_into(values: &[i64], reverse: bool, carry: &mut i64, ys: &mut [i64]) {
    if reverse {
        for idx in (0..values.len()).rev() {
            *carry = carry.wrapping_add(values[idx]);
            ys[idx] = *carry;
        }
    } else {
        for (x, y) in values.iter().zip(ys.iter_mut()) {
            *carry = carry.wrapping_add(*x);
            *y = *carry;
        }
    }
}

fn is_i64_add_emit_scan_body(jaxpr: &Jaxpr) -> bool {
    if !jaxpr.constvars.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 2
        || jaxpr.outvars.len() != 2
        || jaxpr.equations.len() != 2
    {
        return false;
    }

    let carry_var = jaxpr.invars[0];
    let x_var = jaxpr.invars[1];
    let carry_out = jaxpr.outvars[0];
    let y_out = jaxpr.outvars[1];
    let carry_eqn = &jaxpr.equations[0];
    if carry_eqn.primitive != Primitive::Add
        || !carry_eqn.params.is_empty()
        || !carry_eqn.effects.is_empty()
        || !carry_eqn.sub_jaxprs.is_empty()
        || carry_eqn.outputs.as_slice() != [carry_out]
        || carry_eqn.inputs.as_slice() != [Atom::Var(carry_var), Atom::Var(x_var)]
    {
        return false;
    }

    let emit_eqn = &jaxpr.equations[1];
    emit_eqn.primitive == Primitive::Add
        && emit_eqn.params.is_empty()
        && emit_eqn.effects.is_empty()
        && emit_eqn.sub_jaxprs.is_empty()
        && emit_eqn.outputs.as_slice() == [y_out]
        && emit_eqn.inputs.as_slice() == [Atom::Var(carry_out), Atom::Lit(Literal::I64(0))]
}

fn try_eval_scan_f32_add_emit(
    equation: &Equation,
    body_jaxpr: &Jaxpr,
    carry_inputs: &[Value],
    xs: &Value,
    scan_len: usize,
    reverse: bool,
) -> Option<Result<Vec<Value>, InterpreterError>> {
    let y_bias = f32_add_emit_scan_body_bias(body_jaxpr)?;
    if !scan_equation_params_are_i64_add_emit_safe(equation, reverse)
        || !equation.effects.is_empty()
        || carry_inputs.len() != 1
    {
        return None;
    }
    let Value::Tensor(carry_tensor) = carry_inputs.first()? else {
        return None;
    };
    if carry_tensor.dtype != DType::F32 {
        return None;
    }
    let carry_values = carry_tensor.elements.as_f32_slice()?;
    let Value::Tensor(xs_tensor) = xs else {
        return None;
    };
    if xs_tensor.dtype != DType::F32
        || xs_tensor.rank() != carry_tensor.rank() + 1
        || xs_tensor.shape.dims.first().copied()? as usize != scan_len
        || xs_tensor.shape.dims[1..] != carry_tensor.shape.dims
    {
        return None;
    }
    let xs_values = xs_tensor.elements.as_f32_slice()?;
    let row_len = carry_values.len();
    if xs_values.len() != scan_len.checked_mul(row_len)? {
        return None;
    }

    let mut carry = carry_values.to_vec();
    let mut ys = vec![0.0_f32; xs_values.len()];
    if reverse {
        for scan_idx in (0..scan_len).rev() {
            scan_f32_add_emit_row(&mut carry, xs_values, &mut ys, scan_idx, row_len, y_bias);
        }
    } else {
        for scan_idx in 0..scan_len {
            scan_f32_add_emit_row(&mut carry, xs_values, &mut ys, scan_idx, row_len, y_bias);
        }
    }

    Some(
        TensorValue::new_f32_values(carry_tensor.shape.clone(), carry)
            .and_then(|carry_tensor| {
                TensorValue::new_f32_values(xs_tensor.shape.clone(), ys)
                    .map(|ys_tensor| (carry_tensor, ys_tensor))
            })
            .map(|(carry_tensor, ys_tensor)| {
                vec![Value::Tensor(carry_tensor), Value::Tensor(ys_tensor)]
            })
            .map_err(|error| InterpreterError::Primitive(EvalError::InvalidTensor(error))),
    )
}

fn scan_f32_add_emit_row(
    carry: &mut [f32],
    xs_values: &[f32],
    ys: &mut [f32],
    scan_idx: usize,
    row_len: usize,
    y_bias: f32,
) {
    let row_offset = scan_idx * row_len;
    for lane in 0..row_len {
        let next = apply_scalar_f32_binary(
            ScalarF64BinaryOp::Add,
            carry[lane],
            xs_values[row_offset + lane],
        );
        carry[lane] = next;
        ys[row_offset + lane] = apply_scalar_f32_binary(ScalarF64BinaryOp::Add, next, y_bias);
    }
}

fn f32_add_emit_scan_body_bias(jaxpr: &Jaxpr) -> Option<f32> {
    if !jaxpr.constvars.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 2
        || jaxpr.outvars.len() != 2
        || jaxpr.equations.len() != 2
    {
        return None;
    }

    let carry_var = jaxpr.invars[0];
    let x_var = jaxpr.invars[1];
    let carry_out = jaxpr.outvars[0];
    let y_out = jaxpr.outvars[1];
    let carry_eqn = &jaxpr.equations[0];
    if carry_eqn.primitive != Primitive::Add
        || !carry_eqn.params.is_empty()
        || !carry_eqn.effects.is_empty()
        || !carry_eqn.sub_jaxprs.is_empty()
        || carry_eqn.outputs.as_slice() != [carry_out]
        || carry_eqn.inputs.as_slice() != [Atom::Var(carry_var), Atom::Var(x_var)]
    {
        return None;
    }

    let emit_eqn = &jaxpr.equations[1];
    if emit_eqn.primitive != Primitive::Add
        || !emit_eqn.params.is_empty()
        || !emit_eqn.effects.is_empty()
        || !emit_eqn.sub_jaxprs.is_empty()
        || emit_eqn.outputs.as_slice() != [y_out]
    {
        return None;
    }
    let [Atom::Var(var), Atom::Lit(Literal::F32Bits(bits))] = emit_eqn.inputs.as_slice() else {
        return None;
    };
    (*var == carry_out).then_some(f32::from_bits(*bits))
}

struct ScanIterationContext<'a> {
    body_jaxpr: &'a Jaxpr,
    body_plan: &'a Option<DenseEvalPlan>,
    const_values: &'a [Value],
    xs: &'a Value,
    init_shapes: &'a [Shape],
    init_dtypes: &'a [DType],
}

#[allow(clippy::too_many_arguments)]
fn evaluate_scan_iteration(
    context: &ScanIterationContext<'_>,
    scan_idx: usize,
    carry: &mut [Value],
    per_y: &mut [Vec<Value>],
    body_args: &mut Vec<Value>,
    body_env: &mut [Option<Value>],
    scratch: &mut Vec<Value>,
    scalar_buffers: &mut ScalarPlanBuffers,
    body_out: &mut Vec<Value>,
) -> Result<(), InterpreterError> {
    let x_slice = scan_slice_at(context.xs, scan_idx)?;
    body_args.clear();
    body_args.extend(carry.iter().cloned());
    body_args.push(x_slice);

    let carry_count = carry.len();
    let y_count = per_y.len();
    // Reuse the prepared plan + buffers when the body is dense-eligible (zero
    // per-step analysis/allocation); else the self-contained allocating path.
    match context.body_plan {
        Some(plan) => run_dense_plan_into(
            context.body_jaxpr,
            context.const_values,
            body_args,
            body_env,
            plan,
            scratch,
            body_out,
            scalar_buffers,
        ),
        None => eval_jaxpr_with_consts(context.body_jaxpr, context.const_values, body_args)
            .map(|o| *body_out = o),
    }
    .map_err(|err| {
        InterpreterError::Primitive(map_sub_jaxpr_error(Primitive::Scan, "scan body", err))
    })?;
    if body_out.len() != carry_count + y_count {
        return Err(InterpreterError::InvariantViolation {
            detail: format!(
                "scan body sub_jaxpr returned {} outputs; expected {}",
                body_out.len(),
                carry_count + y_count
            ),
        });
    }

    for (idx, value) in body_out[..carry_count].iter().enumerate() {
        let new_shape = value_shape(value);
        if new_shape != context.init_shapes[idx] {
            return Err(InterpreterError::Primitive(EvalError::ShapeChanged {
                primitive: Primitive::Scan,
                detail: format!(
                    "carry element {idx} changed shape from {:?} to {:?}",
                    context.init_shapes[idx].dims, new_shape.dims
                ),
            }));
        }
        if value.dtype() != context.init_dtypes[idx] {
            return Err(InterpreterError::Primitive(EvalError::TypeMismatch {
                primitive: Primitive::Scan,
                detail: "scan body changed carry dtype",
            }));
        }
    }

    // Drain the reused output buffer (owning the values, leaving it empty for
    // the next step): first `carry_count` update the carry in place, the rest
    // are appended to the per-output `y` stacks.
    let mut outputs = body_out.drain(..);
    for (slot, value) in carry.iter_mut().zip(outputs.by_ref().take(carry_count)) {
        *slot = value;
    }
    for (bucket, y_value) in per_y.iter_mut().zip(outputs) {
        bucket.push(y_value);
    }

    Ok(())
}

fn scan_input_len(xs: &Value) -> Result<usize, InterpreterError> {
    match xs {
        Value::Scalar(_) => Ok(1),
        Value::Tensor(tensor) => tensor.shape.dims.first().map(|dim| *dim as usize).ok_or({
            InterpreterError::Primitive(EvalError::TypeMismatch {
                primitive: Primitive::Scan,
                detail: "scan tensor xs must have a leading axis",
            })
        }),
    }
}

fn scan_slice_at(xs: &Value, index: usize) -> Result<Value, InterpreterError> {
    match xs {
        Value::Scalar(_) => Ok(xs.clone()),
        Value::Tensor(tensor) => tensor
            .slice_axis0(index)
            .map_err(|error| InterpreterError::Primitive(EvalError::InvalidTensor(error))),
    }
}

/// Evaluate a single equation against the current environment.
///
/// This handles equation-level control-flow semantics that require access to
/// `sub_jaxprs` and therefore cannot be expressed via primitive evaluation
/// alone.
pub fn eval_equation_outputs(
    equation: &Equation,
    env: &FxHashMap<VarId, Value>,
) -> Result<Vec<Value>, InterpreterError> {
    let resolved = resolve_equation_inputs(equation, env)?;
    eval_equation_outputs_from_resolved(equation, &resolved)
}

/// Evaluate `equation` from its already-resolved input values.
///
/// This is the part of [`eval_equation_outputs`] that does not touch the
/// variable environment: given the equation's inputs already resolved to
/// `Value`s (in atom order), it dispatches to the multi-output primitive
/// evaluator or the control-flow sub-jaxpr evaluators and validates output
/// arity. Splitting it out lets both the hash-map environment and the flat
/// slot-array environment share identical evaluation/dispatch semantics —
/// they differ only in how inputs are resolved.
fn eval_equation_outputs_from_resolved(
    equation: &Equation,
    resolved: &[Value],
) -> Result<Vec<Value>, InterpreterError> {
    let outputs = if equation.sub_jaxprs.is_empty() {
        eval_primitive_multi(equation.primitive, resolved, &equation.params)?
    } else {
        match equation.primitive {
            Primitive::Cond => evaluate_cond_sub_jaxprs(equation, resolved)?,
            Primitive::Scan => evaluate_scan_sub_jaxprs(equation, resolved)?,
            Primitive::While => evaluate_while_sub_jaxprs(equation, resolved)?,
            Primitive::Switch => evaluate_switch_sub_jaxprs(equation, resolved)?,
            primitive => {
                return Err(InterpreterError::Primitive(EvalError::Unsupported {
                    primitive,
                    detail: format!(
                        "sub_jaxprs are only valid for cond, scan, while, or switch; {} received sub_jaxprs",
                        primitive.as_str()
                    ),
                }));
            }
        }
    };

    if outputs.len() != equation.outputs.len() {
        return Err(InterpreterError::UnexpectedOutputArity {
            primitive: equation.primitive,
            expected: equation.outputs.len(),
            actual: outputs.len(),
        });
    }
    Ok(outputs)
}

/// True for the multi-output primitives whose `eval_primitive` (single-output)
/// result would silently drop later outputs. The interpreter routes these
/// through the multi-output path; everything else can take the single-output
/// fast path. Mirrors the list in [`eval_equation_single`].
#[inline]
fn is_multi_output_primitive(primitive: Primitive) -> bool {
    matches!(
        primitive,
        Primitive::Qr
            | Primitive::Lu
            | Primitive::Svd
            | Primitive::Eigh
            | Primitive::TopK
            | Primitive::Slogdet
            | Primitive::Eig
    )
}

/// Single-output fast path for [`eval_equation_outputs`].
///
/// For an equation with no sub-jaxprs whose primitive is single-output, this
/// resolves inputs and calls [`eval_primitive`] directly, returning the lone
/// `Value` without the intermediate one-element `Vec<Value>` that
/// `eval_primitive_multi` allocates (it returns `vec![eval_primitive(..)]` for
/// every non-multi-output primitive). The result is bit-for-bit identical to
/// `eval_equation_outputs(equation, env)?` followed by taking its single
/// element. Multi-output primitives (Qr/Lu/Svd/Eigh/TopK/Slogdet/Eig) and any
/// equation carrying sub-jaxprs delegate to the multi path so semantics and
/// arity validation are unchanged.
pub fn eval_equation_single(
    equation: &Equation,
    env: &FxHashMap<VarId, Value>,
) -> Result<Value, InterpreterError> {
    let is_multi_output = matches!(
        equation.primitive,
        Primitive::Qr
            | Primitive::Lu
            | Primitive::Svd
            | Primitive::Eigh
            | Primitive::TopK
            | Primitive::Slogdet
            | Primitive::Eig
    );
    if !equation.sub_jaxprs.is_empty() || is_multi_output {
        let outputs = eval_equation_outputs(equation, env)?;
        if outputs.len() != 1 {
            return Err(InterpreterError::UnexpectedOutputArity {
                primitive: equation.primitive,
                expected: 1,
                actual: outputs.len(),
            });
        }
        return Ok(outputs.into_iter().next().expect("len checked == 1"));
    }

    if equation.outputs.len() != 1 {
        return Err(InterpreterError::UnexpectedOutputArity {
            primitive: equation.primitive,
            expected: 1,
            actual: equation.outputs.len(),
        });
    }
    let resolved = resolve_equation_inputs(equation, env)?;
    eval_primitive(equation.primitive, &resolved, &equation.params)
        .map_err(InterpreterError::Primitive)
}

pub fn eval_jaxpr_with_consts(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Result<Vec<Value>, InterpreterError> {
    if const_values.len() != jaxpr.constvars.len() {
        return Err(InterpreterError::ConstArity {
            expected: jaxpr.constvars.len(),
            actual: const_values.len(),
        });
    }

    if args.len() != jaxpr.invars.len() {
        return Err(InterpreterError::InputArity {
            expected: jaxpr.invars.len(),
            actual: args.len(),
        });
    }

    if let Some(result) = try_eval_top_level_scan_i64_add_emit(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_softmax_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_log_softmax_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_layer_norm_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_rms_norm_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_softmax_cross_entropy_2d_f64(jaxpr, const_values, args)
    {
        return result;
    }

    if let Some(result) = try_eval_top_level_logsumexp_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_var_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_std_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_skewness_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_kurtosis_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_covariance_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_l2_norm_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_l2_normalize_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_l1_norm_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_rms_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_zscore_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_minmax_normalize_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_cosine_similarity_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_euclidean_distance_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_mean_squared_error_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_mean_absolute_error_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) =
        try_eval_top_level_root_mean_squared_error_2d_f64(jaxpr, const_values, args)
    {
        return result;
    }

    if let Some(result) = try_eval_top_level_r2_score_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_pearson_correlation_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_kl_divergence_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_entropy_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_manhattan_distance_2d_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_gelu_erf_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_log_sigmoid_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_softplus_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_silu_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_swiglu_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_geglu_f64(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_top_level_scalar_half_arith(jaxpr, const_values, args) {
        return result;
    }

    if let Some(result) = try_eval_scalar_i64_add_chain(jaxpr, const_values, args) {
        return result;
    }

    // The tracer mints variables densely and sequentially (`ensure_dense_var`
    // in fj-trace), so the defined-variable ids are a compact range. When that
    // holds we evaluate against a flat `Vec<Option<Value>>` indexed by
    // `VarId.0` — a direct array index per lookup instead of a hash, and a
    // single allocation instead of an incrementally-grown hash map. Pathological
    // (sparse / very large) id ranges fall back to the hash-map environment so
    // we never over-allocate. Both paths are bit-for-bit identical: same input
    // resolution order, same primitive dispatch, same `MissingVariable` errors.
    let mut max_var: u32 = 0;
    let mut def_count: usize = 0;
    for var in jaxpr.constvars.iter().chain(jaxpr.invars.iter()) {
        max_var = max_var.max(var.0);
        def_count += 1;
    }
    for eqn in &jaxpr.equations {
        for out_var in &eqn.outputs {
            max_var = max_var.max(out_var.0);
            def_count += 1;
        }
    }
    let slots_needed = max_var as usize + 1;
    if slots_needed <= def_count.saturating_mul(8).max(256) {
        eval_jaxpr_dense_env(jaxpr, const_values, args, slots_needed)
    } else {
        eval_jaxpr_hashed_env(jaxpr, const_values, args)
    }
}

fn try_eval_top_level_scan_i64_add_emit(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 2
        || jaxpr.outvars.len() != 2
        || jaxpr.equations.len() != 1
    {
        return None;
    }

    let equation = &jaxpr.equations[0];
    if equation.primitive != Primitive::Scan
        || !equation.effects.is_empty()
        || equation.sub_jaxprs.len() != 1
        || equation.inputs.len() != 2
        || equation.outputs.as_slice() != jaxpr.outvars.as_slice()
    {
        return None;
    }
    let [Atom::Var(carry_var), Atom::Var(xs_var)] = equation.inputs.as_slice() else {
        return None;
    };
    if *carry_var != jaxpr.invars[0] || *xs_var != jaxpr.invars[1] {
        return None;
    }

    let reverse = equation
        .params
        .get("reverse")
        .is_some_and(|value| value == "true");
    let scan_len = match scan_input_len(&args[1]) {
        Ok(scan_len) => scan_len,
        Err(error) => return Some(Err(error)),
    };
    try_eval_scan_i64_add_emit(
        equation,
        &equation.sub_jaxprs[0],
        &args[..1],
        &args[1],
        scan_len,
        reverse,
    )
}

fn try_eval_top_level_scalar_half_arith(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 2
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 6
    {
        return None;
    }

    let x = jaxpr.invars[0];
    let y = jaxpr.invars[1];
    let out = jaxpr.outvars[0];
    let [neg_eq, abs_eq, mul_eq, add_eq, div_eq, max_eq] = jaxpr.equations.as_slice() else {
        return None;
    };

    let neg = single_output_for_primitive(neg_eq, Primitive::Neg)?;
    if neg_eq.inputs.as_slice() != [Atom::Var(x)] {
        return None;
    }
    let abs = single_output_for_primitive(abs_eq, Primitive::Abs)?;
    if abs_eq.inputs.as_slice() != [Atom::Var(neg)] {
        return None;
    }
    let prod = single_output_for_primitive(mul_eq, Primitive::Mul)?;
    if mul_eq.inputs.as_slice() != [Atom::Var(abs), Atom::Var(y)] {
        return None;
    }
    let sum = single_output_for_primitive(add_eq, Primitive::Add)?;
    let [Atom::Var(add_lhs), add_rhs] = add_eq.inputs.as_slice() else {
        return None;
    };
    if *add_lhs != prod {
        return None;
    }
    let (dtype, literal_bits) = scalar_half_literal_bits(add_rhs)?;
    let quot = single_output_for_primitive(div_eq, Primitive::Div)?;
    if div_eq.inputs.as_slice() != [Atom::Var(sum), Atom::Var(y)] {
        return None;
    }
    let max_out = single_output_for_primitive(max_eq, Primitive::Max)?;
    if max_out != out || max_eq.inputs.as_slice() != [Atom::Var(quot), Atom::Var(x)] {
        return None;
    }

    let [x_bits, y_bits] = scalar_half_args_bits(args, dtype)?;
    let neg_bits = apply_scalar_half_op(dtype, ScalarF64BinaryOp::Neg, x_bits, x_bits)?;
    let abs_bits = apply_scalar_half_op(dtype, ScalarF64BinaryOp::Abs, neg_bits, neg_bits)?;
    let prod_bits = apply_scalar_half_op(dtype, ScalarF64BinaryOp::Mul, abs_bits, y_bits)?;
    let sum_bits = apply_scalar_half_op(dtype, ScalarF64BinaryOp::Add, prod_bits, literal_bits)?;
    let quot_bits = apply_scalar_half_op(dtype, ScalarF64BinaryOp::Div, sum_bits, y_bits)?;
    let out_bits = apply_scalar_half_op(dtype, ScalarF64BinaryOp::Max, quot_bits, x_bits)?;

    let literal = if dtype == DType::BF16 {
        Literal::BF16Bits(out_bits)
    } else {
        Literal::F16Bits(out_bits)
    };
    Some(Ok(vec![Value::Scalar(literal)]))
}

fn try_eval_top_level_softmax_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 7
        || args.len() != 1
    {
        return None;
    }

    let x = jaxpr.invars[0];
    let out = jaxpr.outvars[0];
    let [
        max_eq,
        max_bcast_eq,
        sub_eq,
        exp_eq,
        sum_eq,
        sum_bcast_eq,
        div_eq,
    ] = jaxpr.equations.as_slice()
    else {
        return None;
    };

    let max = single_output_for_param_primitive(max_eq, Primitive::ReduceMax)?;
    if max_eq.inputs.as_slice() != [Atom::Var(x)] || !axis1_reduce_params(&max_eq.params) {
        return None;
    }

    let max_bcast = single_output_for_param_primitive(max_bcast_eq, Primitive::BroadcastInDim)?;
    if max_bcast_eq.inputs.as_slice() != [Atom::Var(max)]
        || !rank2_axis0_broadcast_params(&max_bcast_eq.params)
    {
        return None;
    }

    let shifted = single_output_for_primitive(sub_eq, Primitive::Sub)?;
    if sub_eq.inputs.as_slice() != [Atom::Var(x), Atom::Var(max_bcast)] {
        return None;
    }

    let exp = single_output_for_primitive(exp_eq, Primitive::Exp)?;
    if exp_eq.inputs.as_slice() != [Atom::Var(shifted)] {
        return None;
    }

    let sum = single_output_for_param_primitive(sum_eq, Primitive::ReduceSum)?;
    if sum_eq.inputs.as_slice() != [Atom::Var(exp)] || !axis1_reduce_params(&sum_eq.params) {
        return None;
    }

    let sum_bcast = single_output_for_param_primitive(sum_bcast_eq, Primitive::BroadcastInDim)?;
    if sum_bcast_eq.inputs.as_slice() != [Atom::Var(sum)]
        || sum_bcast_eq.params != max_bcast_eq.params
    {
        return None;
    }

    let div = single_output_for_primitive(div_eq, Primitive::Div)?;
    if div != out || div_eq.inputs.as_slice() != [Atom::Var(exp), Atom::Var(sum_bcast)] {
        return None;
    }

    let Value::Tensor(input) = &args[0] else {
        return None;
    };
    if input.dtype != DType::F64 || input.shape.dims.len() != 2 {
        return None;
    }
    let rows = input.shape.dims[0] as usize;
    let cols = input.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }
    let expected_shape = format!("{rows},{cols}");
    if max_bcast_eq.params.get("shape").map(String::as_str) != Some(expected_shape.as_str()) {
        return None;
    }

    let values = input.elements.as_f64_slice()?;
    if !values.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::softmax_2d(values, rows, cols);
    Some(
        TensorValue::new_f64_values(input.shape.clone(), output)
            .map(Value::Tensor)
            .map(|value| vec![value])
            .map_err(EvalError::InvalidTensor)
            .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the canonical finite dense rank-2 f64 log-softmax
/// graph `ReduceMax(axis=1) → BroadcastInDim(axis0) → Sub → Exp → ReduceSum(axis=1) →
/// Log → BroadcastInDim(axis0) → Sub`, i.e. `shifted - log(sum(exp(shifted)))` with
/// `shifted = x - rowmax`. Mirrors [`try_eval_top_level_softmax_2d_f64`] one equation
/// longer (the extra `Log`, and a final `Sub` on `shifted` instead of `Div` on `exp`).
/// Exact-graph/exact-shape only; nonfinite or noncanonical programs fall through to the
/// generic interpreter. The fused `log_softmax_2d_shifted` uses the SAME reduction order
/// and subtraction grouping as this graph, so it is bit-identical to the decomposed path.
fn try_eval_top_level_log_softmax_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 8
        || args.len() != 1
    {
        return None;
    }

    let x = jaxpr.invars[0];
    let out = jaxpr.outvars[0];
    let [
        max_eq,
        max_bcast_eq,
        sub_eq,
        exp_eq,
        sum_eq,
        log_eq,
        log_bcast_eq,
        out_sub_eq,
    ] = jaxpr.equations.as_slice()
    else {
        return None;
    };

    let max = single_output_for_param_primitive(max_eq, Primitive::ReduceMax)?;
    if max_eq.inputs.as_slice() != [Atom::Var(x)] || !axis1_reduce_params(&max_eq.params) {
        return None;
    }

    let max_bcast = single_output_for_param_primitive(max_bcast_eq, Primitive::BroadcastInDim)?;
    if max_bcast_eq.inputs.as_slice() != [Atom::Var(max)]
        || !rank2_axis0_broadcast_params(&max_bcast_eq.params)
    {
        return None;
    }

    let shifted = single_output_for_primitive(sub_eq, Primitive::Sub)?;
    if sub_eq.inputs.as_slice() != [Atom::Var(x), Atom::Var(max_bcast)] {
        return None;
    }

    let exp = single_output_for_primitive(exp_eq, Primitive::Exp)?;
    if exp_eq.inputs.as_slice() != [Atom::Var(shifted)] {
        return None;
    }

    let sum = single_output_for_param_primitive(sum_eq, Primitive::ReduceSum)?;
    if sum_eq.inputs.as_slice() != [Atom::Var(exp)] || !axis1_reduce_params(&sum_eq.params) {
        return None;
    }

    let logv = single_output_for_primitive(log_eq, Primitive::Log)?;
    if log_eq.inputs.as_slice() != [Atom::Var(sum)] {
        return None;
    }

    let log_bcast = single_output_for_param_primitive(log_bcast_eq, Primitive::BroadcastInDim)?;
    if log_bcast_eq.inputs.as_slice() != [Atom::Var(logv)]
        || log_bcast_eq.params != max_bcast_eq.params
    {
        return None;
    }

    let out_sub = single_output_for_primitive(out_sub_eq, Primitive::Sub)?;
    if out_sub != out || out_sub_eq.inputs.as_slice() != [Atom::Var(shifted), Atom::Var(log_bcast)]
    {
        return None;
    }

    let Value::Tensor(input) = &args[0] else {
        return None;
    };
    if input.dtype != DType::F64 || input.shape.dims.len() != 2 {
        return None;
    }
    let rows = input.shape.dims[0] as usize;
    let cols = input.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }
    let expected_shape = format!("{rows},{cols}");
    if max_bcast_eq.params.get("shape").map(String::as_str) != Some(expected_shape.as_str()) {
        return None;
    }

    let values = input.elements.as_f64_slice()?;
    if !values.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::log_softmax_2d_shifted(values, rows, cols);
    Some(
        TensorValue::new_f64_values(input.shape.clone(), output)
            .map(Value::Tensor)
            .map(|value| vec![value])
            .map_err(EvalError::InvalidTensor)
            .map_err(InterpreterError::Primitive),
    )
}

fn try_eval_top_level_layer_norm_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 11
        || args.len() != 1
    {
        return None;
    }

    let x = jaxpr.invars[0];
    let out = jaxpr.outvars[0];
    let [
        sum_eq,
        mean_eq,
        mean_bcast_eq,
        centered_eq,
        squared_eq,
        var_sum_eq,
        variance_eq,
        eps_eq,
        rsqrt_eq,
        inv_bcast_eq,
        mul_eq,
    ] = jaxpr.equations.as_slice()
    else {
        return None;
    };

    let sum = single_output_for_param_primitive(sum_eq, Primitive::ReduceSum)?;
    if sum_eq.inputs.as_slice() != [Atom::Var(x)] || !axis1_reduce_params(&sum_eq.params) {
        return None;
    }

    let mean = single_output_for_primitive(mean_eq, Primitive::Div)?;
    let [Atom::Var(mean_lhs), mean_rhs] = mean_eq.inputs.as_slice() else {
        return None;
    };
    if *mean_lhs != sum {
        return None;
    }
    let mean_divisor = f64_literal(mean_rhs)?;

    let mean_bcast = single_output_for_param_primitive(mean_bcast_eq, Primitive::BroadcastInDim)?;
    if mean_bcast_eq.inputs.as_slice() != [Atom::Var(mean)]
        || !rank2_axis0_broadcast_params(&mean_bcast_eq.params)
    {
        return None;
    }

    let centered = single_output_for_primitive(centered_eq, Primitive::Sub)?;
    if centered_eq.inputs.as_slice() != [Atom::Var(x), Atom::Var(mean_bcast)] {
        return None;
    }

    let squared = single_output_for_primitive(squared_eq, Primitive::Square)?;
    if squared_eq.inputs.as_slice() != [Atom::Var(centered)] {
        return None;
    }

    let var_sum = single_output_for_param_primitive(var_sum_eq, Primitive::ReduceSum)?;
    if var_sum_eq.inputs.as_slice() != [Atom::Var(squared)] || var_sum_eq.params != sum_eq.params {
        return None;
    }

    let variance = single_output_for_primitive(variance_eq, Primitive::Div)?;
    let [Atom::Var(variance_lhs), variance_rhs] = variance_eq.inputs.as_slice() else {
        return None;
    };
    if *variance_lhs != var_sum {
        return None;
    }
    let variance_divisor = f64_literal(variance_rhs)?;

    let var_eps = single_output_for_primitive(eps_eq, Primitive::Add)?;
    let epsilon = match eps_eq.inputs.as_slice() {
        [Atom::Var(var), atom] if *var == variance => f64_literal(atom)?,
        [atom, Atom::Var(var)] if *var == variance => f64_literal(atom)?,
        _ => return None,
    };
    if !epsilon.is_finite() || epsilon < 0.0 {
        return None;
    }

    let inv = single_output_for_primitive(rsqrt_eq, Primitive::Rsqrt)?;
    if rsqrt_eq.inputs.as_slice() != [Atom::Var(var_eps)] {
        return None;
    }

    let inv_bcast = single_output_for_param_primitive(inv_bcast_eq, Primitive::BroadcastInDim)?;
    if inv_bcast_eq.inputs.as_slice() != [Atom::Var(inv)]
        || inv_bcast_eq.params != mean_bcast_eq.params
    {
        return None;
    }

    let mul = single_output_for_primitive(mul_eq, Primitive::Mul)?;
    if mul != out || mul_eq.inputs.as_slice() != [Atom::Var(centered), Atom::Var(inv_bcast)] {
        return None;
    }

    let Value::Tensor(input) = &args[0] else {
        return None;
    };
    if input.dtype != DType::F64 || input.shape.dims.len() != 2 {
        return None;
    }
    let rows = input.shape.dims[0] as usize;
    let cols = input.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }
    let cols_f64 = cols as f64;
    if mean_divisor.to_bits() != cols_f64.to_bits()
        || variance_divisor.to_bits() != cols_f64.to_bits()
    {
        return None;
    }
    let expected_shape = format!("{rows},{cols}");
    if mean_bcast_eq.params.get("shape").map(String::as_str) != Some(expected_shape.as_str()) {
        return None;
    }

    let values = input.elements.as_f64_slice()?;
    if !values.iter().all(|value| value.is_finite()) {
        return None;
    }

    let mut output = vec![0.0; values.len()];
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let row_values = &values[start..end];
        let row_output = &mut output[start..end];

        let mut sum = 0.0;
        for &value in row_values {
            sum += value;
        }
        let mean = sum / cols_f64;

        let mut sum_sq = 0.0;
        for (slot, &value) in row_output.iter_mut().zip(row_values) {
            let centered = value - mean;
            *slot = centered;
            sum_sq += centered * centered;
        }

        let inv_std = 1.0 / ((sum_sq / cols_f64) + epsilon).sqrt();
        for value in row_output {
            *value *= inv_std;
        }
    }

    Some(
        TensorValue::new_f64_values(input.shape.clone(), output)
            .map(Value::Tensor)
            .map(|value| vec![value])
            .map_err(EvalError::InvalidTensor)
            .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the canonical finite dense rank-2 f64 RMS-norm graph
/// `Square → ReduceSum(axis=1) → Div(cols) → Add(eps) → Rsqrt → BroadcastInDim(axis0) →
/// Mul(x, ·)`, i.e. `x * rsqrt(mean(x²) + eps)` (RMSNorm without a learned scale — the
/// modern-transformer normalization, sibling of the layer-norm superinstruction but with NO
/// mean subtraction). Exact-graph / exact-shape / finite-only; anything else falls through to
/// the generic interpreter. Fused inline serial like `try_eval_top_level_layer_norm_2d_f64`,
/// reusing its bit-exact op forms: `Square`=`x*x`, index-order `ReduceSum`, `Rsqrt`=`1/√·`.
fn try_eval_top_level_rms_norm_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 7
        || args.len() != 1
    {
        return None;
    }

    let x = jaxpr.invars[0];
    let out = jaxpr.outvars[0];
    let [
        squared_eq,
        sq_sum_eq,
        mean_eq,
        eps_eq,
        rsqrt_eq,
        inv_bcast_eq,
        mul_eq,
    ] = jaxpr.equations.as_slice()
    else {
        return None;
    };

    let squared = single_output_for_primitive(squared_eq, Primitive::Square)?;
    if squared_eq.inputs.as_slice() != [Atom::Var(x)] {
        return None;
    }

    let sq_sum = single_output_for_param_primitive(sq_sum_eq, Primitive::ReduceSum)?;
    if sq_sum_eq.inputs.as_slice() != [Atom::Var(squared)]
        || !axis1_reduce_params(&sq_sum_eq.params)
    {
        return None;
    }

    let mean = single_output_for_primitive(mean_eq, Primitive::Div)?;
    let [Atom::Var(mean_lhs), mean_rhs] = mean_eq.inputs.as_slice() else {
        return None;
    };
    if *mean_lhs != sq_sum {
        return None;
    }
    let mean_divisor = f64_literal(mean_rhs)?;

    let ms_eps = single_output_for_primitive(eps_eq, Primitive::Add)?;
    let epsilon = match eps_eq.inputs.as_slice() {
        [Atom::Var(var), atom] if *var == mean => f64_literal(atom)?,
        [atom, Atom::Var(var)] if *var == mean => f64_literal(atom)?,
        _ => return None,
    };
    if !epsilon.is_finite() || epsilon < 0.0 {
        return None;
    }

    let inv = single_output_for_primitive(rsqrt_eq, Primitive::Rsqrt)?;
    if rsqrt_eq.inputs.as_slice() != [Atom::Var(ms_eps)] {
        return None;
    }

    let inv_bcast = single_output_for_param_primitive(inv_bcast_eq, Primitive::BroadcastInDim)?;
    if inv_bcast_eq.inputs.as_slice() != [Atom::Var(inv)]
        || !rank2_axis0_broadcast_params(&inv_bcast_eq.params)
    {
        return None;
    }

    let mul = single_output_for_primitive(mul_eq, Primitive::Mul)?;
    if mul != out || mul_eq.inputs.as_slice() != [Atom::Var(x), Atom::Var(inv_bcast)] {
        return None;
    }

    let Value::Tensor(input) = &args[0] else {
        return None;
    };
    if input.dtype != DType::F64 || input.shape.dims.len() != 2 {
        return None;
    }
    let rows = input.shape.dims[0] as usize;
    let cols = input.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }
    let cols_f64 = cols as f64;
    if mean_divisor.to_bits() != cols_f64.to_bits() {
        return None;
    }
    let expected_shape = format!("{rows},{cols}");
    if inv_bcast_eq.params.get("shape").map(String::as_str) != Some(expected_shape.as_str()) {
        return None;
    }

    let values = input.elements.as_f64_slice()?;
    if !values.iter().all(|value| value.is_finite()) {
        return None;
    }

    // `rms_norm_2d` divides by `cols as f64`, matching the checked `mean_divisor`.
    let output = fj_lax::nn::rms_norm_2d(values, rows, cols, epsilon);
    Some(
        TensorValue::new_f64_values(input.shape.clone(), output)
            .map(Value::Tensor)
            .map(|value| vec![value])
            .map_err(EvalError::InvalidTensor)
            .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the canonical finite dense rank-2 f64 softmax
/// cross-entropy loss graph `-Σ labels · log_softmax(logits)` (`optax.softmax_cross_entropy`):
/// the 8-equation log-softmax subgraph on `logits` (`ReduceMax → Bcast → Sub → Exp →
/// ReduceSum → Log → Bcast → Sub`) followed by `Mul(labels, ls) → ReduceSum(axis=1) → Neg`.
/// Two f64 [rows,cols] inputs, one [rows] output. Exact-graph / exact-shape / finite-only;
/// anything else falls through to the generic interpreter. Fused via the row-parallel
/// `softmax_cross_entropy_2d`, which reuses the log-softmax superinstruction's bit-exact op
/// forms (fold-max, index-order exp-sum, `shifted - log` grouping) so it matches the generic
/// decomposed path bit-for-bit.
fn try_eval_top_level_softmax_cross_entropy_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 2
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 11
        || args.len() != 2
    {
        return None;
    }

    let logits = jaxpr.invars[0];
    let labels = jaxpr.invars[1];
    let out = jaxpr.outvars[0];
    let [
        max_eq,
        max_bcast_eq,
        sub_eq,
        exp_eq,
        sum_eq,
        log_eq,
        log_bcast_eq,
        ls_eq,
        mul_eq,
        ce_sum_eq,
        neg_eq,
    ] = jaxpr.equations.as_slice()
    else {
        return None;
    };

    // --- log_softmax(logits) subgraph (identical to the log-softmax superinstruction) ---
    let max = single_output_for_param_primitive(max_eq, Primitive::ReduceMax)?;
    if max_eq.inputs.as_slice() != [Atom::Var(logits)] || !axis1_reduce_params(&max_eq.params) {
        return None;
    }
    let max_bcast = single_output_for_param_primitive(max_bcast_eq, Primitive::BroadcastInDim)?;
    if max_bcast_eq.inputs.as_slice() != [Atom::Var(max)]
        || !rank2_axis0_broadcast_params(&max_bcast_eq.params)
    {
        return None;
    }
    let shifted = single_output_for_primitive(sub_eq, Primitive::Sub)?;
    if sub_eq.inputs.as_slice() != [Atom::Var(logits), Atom::Var(max_bcast)] {
        return None;
    }
    let exp = single_output_for_primitive(exp_eq, Primitive::Exp)?;
    if exp_eq.inputs.as_slice() != [Atom::Var(shifted)] {
        return None;
    }
    let sum = single_output_for_param_primitive(sum_eq, Primitive::ReduceSum)?;
    if sum_eq.inputs.as_slice() != [Atom::Var(exp)] || !axis1_reduce_params(&sum_eq.params) {
        return None;
    }
    let logv = single_output_for_primitive(log_eq, Primitive::Log)?;
    if log_eq.inputs.as_slice() != [Atom::Var(sum)] {
        return None;
    }
    let log_bcast = single_output_for_param_primitive(log_bcast_eq, Primitive::BroadcastInDim)?;
    if log_bcast_eq.inputs.as_slice() != [Atom::Var(logv)]
        || log_bcast_eq.params != max_bcast_eq.params
    {
        return None;
    }
    let ls = single_output_for_primitive(ls_eq, Primitive::Sub)?;
    if ls_eq.inputs.as_slice() != [Atom::Var(shifted), Atom::Var(log_bcast)] {
        return None;
    }

    // --- cross-entropy tail: -sum(labels * log_softmax, axis=1) ---
    let prod = single_output_for_primitive(mul_eq, Primitive::Mul)?;
    if mul_eq.inputs.as_slice() != [Atom::Var(labels), Atom::Var(ls)] {
        return None;
    }
    let ce_sum = single_output_for_param_primitive(ce_sum_eq, Primitive::ReduceSum)?;
    if ce_sum_eq.inputs.as_slice() != [Atom::Var(prod)] || !axis1_reduce_params(&ce_sum_eq.params) {
        return None;
    }
    let neg = single_output_for_primitive(neg_eq, Primitive::Neg)?;
    if neg != out || neg_eq.inputs.as_slice() != [Atom::Var(ce_sum)] {
        return None;
    }

    let Value::Tensor(logits_t) = &args[0] else {
        return None;
    };
    let Value::Tensor(labels_t) = &args[1] else {
        return None;
    };
    if logits_t.dtype != DType::F64
        || labels_t.dtype != DType::F64
        || logits_t.shape.dims.len() != 2
        || labels_t.shape.dims != logits_t.shape.dims
    {
        return None;
    }
    let rows = logits_t.shape.dims[0] as usize;
    let cols = logits_t.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }
    let expected_shape = format!("{rows},{cols}");
    if max_bcast_eq.params.get("shape").map(String::as_str) != Some(expected_shape.as_str()) {
        return None;
    }

    let logits_v = logits_t.elements.as_f64_slice()?;
    let labels_v = labels_t.elements.as_f64_slice()?;
    if !logits_v.iter().all(|value| value.is_finite())
        || !labels_v.iter().all(|value| value.is_finite())
    {
        return None;
    }

    let output = fj_lax::nn::softmax_cross_entropy_2d(logits_v, labels_v, rows, cols);
    Some(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32],
            },
            output,
        )
        .map(Value::Tensor)
        .map(|value| vec![value])
        .map_err(EvalError::InvalidTensor)
        .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the row-wise log-sum-exp reduction
/// `max(x) + log(sum(exp(x - max(x))))` (`jax.scipy.special.logsumexp` / `jax.nn.logsumexp` over the
/// last axis): the 7-equation graph `ReduceMax(axis=1) → BroadcastInDim → Sub → Exp →
/// ReduceSum(axis=1) → Log → Add`. One f64 [rows,cols] input, one [rows] output (rank-reducing).
/// Exact-graph / exact-shape / finite-only; anything else falls through to the generic interpreter.
/// Fused via the row-parallel `logsumexp_2d`, which reuses the same bit-exact op forms as the
/// cross-entropy superinstruction (fold-max reduction, index-order exp-sum, `max + log(sum)`) so it
/// matches the generic decomposed path bit-for-bit. The general fuser cannot fuse it (Exp/Log ∉
/// `cheap_op`, and reductions break the elementwise fuser), so the decomposed path materializes the
/// full [rows,cols] shifted/exp intermediates plus per-reduction Vecs.
fn try_eval_top_level_logsumexp_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 7
        || args.len() != 1
    {
        return None;
    }

    let x = jaxpr.invars[0];
    let out = jaxpr.outvars[0];
    let [max_eq, max_bcast_eq, sub_eq, exp_eq, sum_eq, log_eq, add_eq] = jaxpr.equations.as_slice()
    else {
        return None;
    };

    let max = single_output_for_param_primitive(max_eq, Primitive::ReduceMax)?;
    if max_eq.inputs.as_slice() != [Atom::Var(x)] || !axis1_reduce_params(&max_eq.params) {
        return None;
    }
    let max_bcast = single_output_for_param_primitive(max_bcast_eq, Primitive::BroadcastInDim)?;
    if max_bcast_eq.inputs.as_slice() != [Atom::Var(max)]
        || !rank2_axis0_broadcast_params(&max_bcast_eq.params)
    {
        return None;
    }
    let shifted = single_output_for_primitive(sub_eq, Primitive::Sub)?;
    if sub_eq.inputs.as_slice() != [Atom::Var(x), Atom::Var(max_bcast)] {
        return None;
    }
    let exp = single_output_for_primitive(exp_eq, Primitive::Exp)?;
    if exp_eq.inputs.as_slice() != [Atom::Var(shifted)] {
        return None;
    }
    let sum = single_output_for_param_primitive(sum_eq, Primitive::ReduceSum)?;
    if sum_eq.inputs.as_slice() != [Atom::Var(exp)] || !axis1_reduce_params(&sum_eq.params) {
        return None;
    }
    let logv = single_output_for_primitive(log_eq, Primitive::Log)?;
    if log_eq.inputs.as_slice() != [Atom::Var(sum)] {
        return None;
    }
    // `max` and `logv` are both [rows] — a rank-1 elementwise Add (no broadcast). `Add` is
    // commutative and IEEE-bit-commutative, so accept either operand order.
    let added = single_output_for_primitive(add_eq, Primitive::Add)?;
    let add_ok = matches!(
        add_eq.inputs.as_slice(),
        [Atom::Var(a), Atom::Var(b)]
            if (*a == max && *b == logv) || (*a == logv && *b == max)
    );
    if added != out || !add_ok {
        return None;
    }

    let Value::Tensor(input) = &args[0] else {
        return None;
    };
    if input.dtype != DType::F64 || input.shape.dims.len() != 2 {
        return None;
    }
    let rows = input.shape.dims[0] as usize;
    let cols = input.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }
    let expected_shape = format!("{rows},{cols}");
    if max_bcast_eq.params.get("shape").map(String::as_str) != Some(expected_shape.as_str()) {
        return None;
    }

    let values = input.elements.as_f64_slice()?;
    if !values.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::logsumexp_2d(values, rows, cols);
    Some(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32],
            },
            output,
        )
        .map(Value::Tensor)
        .map(|value| vec![value])
        .map_err(EvalError::InvalidTensor)
        .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the row-wise population standard deviation
/// `sqrt(mean((x - mean(x))²))` (`jax.numpy.std(x, axis=-1)`, `ddof=0`): the variance 7-equation
/// graph followed by a trailing `Sqrt` (8 equations total). One f64 [rows,cols] input, one [rows]
/// output (rank-reducing). NO transcendentals — a pure statistical moment reduction plus a
/// correctly-rounded `Sqrt`, distinct from the exp/log reductions. The general fuser cannot fuse it
/// (the two reductions break the elementwise fuser), so the decomposed path materializes the full
/// [rows,cols] centered+squared intermediates plus the per-reduction Vecs. Fused via the row-parallel
/// `std_2d`, whose index-order sums + `sum/n` mean grouping + `f64::sqrt` match the graph bit-for-bit.
/// Reuses the exact variance shape/divisor guards; finite dense rank-2 f64 only, else falls through.
fn try_eval_top_level_std_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 8
        || args.len() != 1
    {
        return None;
    }

    let x = jaxpr.invars[0];
    let out = jaxpr.outvars[0];
    let [
        sum1_eq,
        mean_eq,
        bcast_eq,
        sub_eq,
        sq_eq,
        sum2_eq,
        var_eq,
        sqrt_eq,
    ] = jaxpr.equations.as_slice()
    else {
        return None;
    };

    // mean = ReduceSum(x, axis=1) / n
    let s1 = single_output_for_param_primitive(sum1_eq, Primitive::ReduceSum)?;
    if sum1_eq.inputs.as_slice() != [Atom::Var(x)] || !axis1_reduce_params(&sum1_eq.params) {
        return None;
    }
    let mean = single_output_for_primitive(mean_eq, Primitive::Div)?;
    let [Atom::Var(mean_num), mean_div] = mean_eq.inputs.as_slice() else {
        return None;
    };
    if *mean_num != s1 {
        return None;
    }
    // broadcast mean back to [rows,cols]
    let mean_b = single_output_for_param_primitive(bcast_eq, Primitive::BroadcastInDim)?;
    if bcast_eq.inputs.as_slice() != [Atom::Var(mean)]
        || !rank2_axis0_broadcast_params(&bcast_eq.params)
    {
        return None;
    }
    // centered = x - mean_b
    let centered = single_output_for_primitive(sub_eq, Primitive::Sub)?;
    if sub_eq.inputs.as_slice() != [Atom::Var(x), Atom::Var(mean_b)] {
        return None;
    }
    // sq = centered * centered (both operands the same var)
    let sq = single_output_for_primitive(sq_eq, Primitive::Mul)?;
    if sq_eq.inputs.as_slice() != [Atom::Var(centered), Atom::Var(centered)] {
        return None;
    }
    // var = ReduceSum(sq, axis=1) / n
    let s2 = single_output_for_param_primitive(sum2_eq, Primitive::ReduceSum)?;
    if sum2_eq.inputs.as_slice() != [Atom::Var(sq)] || !axis1_reduce_params(&sum2_eq.params) {
        return None;
    }
    let var = single_output_for_primitive(var_eq, Primitive::Div)?;
    let [Atom::Var(var_num), var_div] = var_eq.inputs.as_slice() else {
        return None;
    };
    if *var_num != s2 {
        return None;
    }
    // std = Sqrt(var)
    let std = single_output_for_primitive(sqrt_eq, Primitive::Sqrt)?;
    if sqrt_eq.inputs.as_slice() != [Atom::Var(var)] || std != out {
        return None;
    }

    let Value::Tensor(input) = &args[0] else {
        return None;
    };
    if input.dtype != DType::F64 || input.shape.dims.len() != 2 {
        return None;
    }
    let rows = input.shape.dims[0] as usize;
    let cols = input.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }
    // Both divisors must be exactly the row length `n = cols` (as f64), matching `std_row`.
    let n_bits = (cols as f64).to_bits();
    if f64_literal(mean_div)?.to_bits() != n_bits || f64_literal(var_div)?.to_bits() != n_bits {
        return None;
    }
    let expected_shape = format!("{rows},{cols}");
    if bcast_eq.params.get("shape").map(String::as_str) != Some(expected_shape.as_str()) {
        return None;
    }

    let values = input.elements.as_f64_slice()?;
    if !values.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::std_2d(values, rows, cols);
    Some(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32],
            },
            output,
        )
        .map(Value::Tensor)
        .map(|value| vec![value])
        .map_err(EvalError::InvalidTensor)
        .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the row-wise population skewness `m3 / m2^1.5`
/// (`scipy.stats.skew(x, axis=-1)`, Fisher–Pearson, `ddof=0`) with `m2 = mean((x-μ)²)`,
/// `m3 = mean((x-μ)³)`: the 13-equation graph `ReduceSum → Div(mean) → BroadcastInDim → Sub →
/// Mul(d²) → Mul(d³) → ReduceSum → Div(m2) → ReduceSum → Div(m3) → Sqrt → Mul(denom) → Div`, one f64
/// [rows,cols] input, one [rows] output (rank-reducing). NO transcendentals — a pure statistical
/// third-moment reduction, so per the transcendental-free thesis it lands a large multiple (bigger
/// graph than variance/std → bigger win). The general fuser cannot fuse it (the three reductions
/// break the elementwise fuser AND `Sqrt` ∉ `cheap_op`), so the decomposed path materializes the full
/// [rows,cols] centered / squared / cubed intermediates plus per-reduction Vecs. Fused via the
/// row-parallel `skewness_2d`, whose index-order sums, `sum/n` moment grouping, and `m2·sqrt(m2)`
/// denominator match the graph bit-for-bit. All three `Div` divisors must equal `cols` (as f64) and
/// the broadcast must restore exactly `rows,cols`; finite dense rank-2 f64 only, else falls through.
fn try_eval_top_level_skewness_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 13
        || args.len() != 1
    {
        return None;
    }

    let x = jaxpr.invars[0];
    let out = jaxpr.outvars[0];
    let [
        sum1_eq,
        mean_eq,
        bcast_eq,
        sub_eq,
        d2_eq,
        d3_eq,
        sum2_eq,
        m2_eq,
        sum3_eq,
        m3_eq,
        sqrt_eq,
        denom_eq,
        skew_eq,
    ] = jaxpr.equations.as_slice()
    else {
        return None;
    };

    // mean = ReduceSum(x, axis=1) / n
    let s1 = single_output_for_param_primitive(sum1_eq, Primitive::ReduceSum)?;
    if sum1_eq.inputs.as_slice() != [Atom::Var(x)] || !axis1_reduce_params(&sum1_eq.params) {
        return None;
    }
    let mean = single_output_for_primitive(mean_eq, Primitive::Div)?;
    let [Atom::Var(mean_num), mean_div] = mean_eq.inputs.as_slice() else {
        return None;
    };
    if *mean_num != s1 {
        return None;
    }
    // broadcast mean back to [rows,cols]
    let mean_b = single_output_for_param_primitive(bcast_eq, Primitive::BroadcastInDim)?;
    if bcast_eq.inputs.as_slice() != [Atom::Var(mean)]
        || !rank2_axis0_broadcast_params(&bcast_eq.params)
    {
        return None;
    }
    // d = x - mean_b
    let d = single_output_for_primitive(sub_eq, Primitive::Sub)?;
    if sub_eq.inputs.as_slice() != [Atom::Var(x), Atom::Var(mean_b)] {
        return None;
    }
    // d2 = d * d, d3 = d2 * d
    let d2 = single_output_for_primitive(d2_eq, Primitive::Mul)?;
    if d2_eq.inputs.as_slice() != [Atom::Var(d), Atom::Var(d)] {
        return None;
    }
    let d3 = single_output_for_primitive(d3_eq, Primitive::Mul)?;
    if d3_eq.inputs.as_slice() != [Atom::Var(d2), Atom::Var(d)] {
        return None;
    }
    // m2 = ReduceSum(d2, axis=1) / n
    let sum2 = single_output_for_param_primitive(sum2_eq, Primitive::ReduceSum)?;
    if sum2_eq.inputs.as_slice() != [Atom::Var(d2)] || !axis1_reduce_params(&sum2_eq.params) {
        return None;
    }
    let m2 = single_output_for_primitive(m2_eq, Primitive::Div)?;
    let [Atom::Var(m2_num), m2_div] = m2_eq.inputs.as_slice() else {
        return None;
    };
    if *m2_num != sum2 {
        return None;
    }
    // m3 = ReduceSum(d3, axis=1) / n
    let sum3 = single_output_for_param_primitive(sum3_eq, Primitive::ReduceSum)?;
    if sum3_eq.inputs.as_slice() != [Atom::Var(d3)] || !axis1_reduce_params(&sum3_eq.params) {
        return None;
    }
    let m3 = single_output_for_primitive(m3_eq, Primitive::Div)?;
    let [Atom::Var(m3_num), m3_div] = m3_eq.inputs.as_slice() else {
        return None;
    };
    if *m3_num != sum3 {
        return None;
    }
    // s = Sqrt(m2), denom = m2 * s, skew = m3 / denom
    let s = single_output_for_primitive(sqrt_eq, Primitive::Sqrt)?;
    if sqrt_eq.inputs.as_slice() != [Atom::Var(m2)] {
        return None;
    }
    let denom = single_output_for_primitive(denom_eq, Primitive::Mul)?;
    if denom_eq.inputs.as_slice() != [Atom::Var(m2), Atom::Var(s)] {
        return None;
    }
    let skew = single_output_for_primitive(skew_eq, Primitive::Div)?;
    let [Atom::Var(skew_num), Atom::Var(skew_den)] = skew_eq.inputs.as_slice() else {
        return None;
    };
    if skew != out || *skew_num != m3 || *skew_den != denom {
        return None;
    }

    let Value::Tensor(input) = &args[0] else {
        return None;
    };
    if input.dtype != DType::F64 || input.shape.dims.len() != 2 {
        return None;
    }
    let rows = input.shape.dims[0] as usize;
    let cols = input.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }
    // All three divisors must be exactly the row length `n = cols` (as f64), matching `skewness_row`.
    let n_bits = (cols as f64).to_bits();
    if f64_literal(mean_div)?.to_bits() != n_bits
        || f64_literal(m2_div)?.to_bits() != n_bits
        || f64_literal(m3_div)?.to_bits() != n_bits
    {
        return None;
    }
    let expected_shape = format!("{rows},{cols}");
    if bcast_eq.params.get("shape").map(String::as_str) != Some(expected_shape.as_str()) {
        return None;
    }

    let values = input.elements.as_f64_slice()?;
    if !values.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::skewness_2d(values, rows, cols);
    Some(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32],
            },
            output,
        )
        .map(Value::Tensor)
        .map(|value| vec![value])
        .map_err(EvalError::InvalidTensor)
        .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the row-wise population kurtosis `m4 / m2²`
/// (`scipy.stats.kurtosis(x, axis=-1, fisher=False)`, Pearson β₂, `ddof=0`) with `m2 = mean((x-μ)²)`,
/// `m4 = mean((x-μ)⁴)`: the 12-equation graph `ReduceSum → Div(mean) → BroadcastInDim → Sub →
/// Mul(d²) → Mul(d⁴) → ReduceSum → Div(m2) → ReduceSum → Div(m4) → Mul(denom) → Div`, one f64
/// [rows,cols] input, one [rows] output (rank-reducing). NO transcendentals AT ALL — a pure fourth-
/// moment reduction (add/sub/mul/div only), so per the transcendental-free thesis it lands a large
/// multiple. The general fuser cannot fuse it (the three reductions break the elementwise fuser), so
/// the decomposed path materializes the full [rows,cols] centered / squared / fourth-power
/// intermediates plus per-reduction Vecs. Fused via the row-parallel `kurtosis_2d`, whose index-order
/// sums, `sum/n` moment grouping, and `m2·m2` denominator match the graph bit-for-bit. All three
/// `Div` divisors must equal `cols` (as f64) and the broadcast must restore exactly `rows,cols`;
/// finite dense rank-2 f64 only, else falls through.
fn try_eval_top_level_kurtosis_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 12
        || args.len() != 1
    {
        return None;
    }

    let x = jaxpr.invars[0];
    let out = jaxpr.outvars[0];
    let [
        sum1_eq,
        mean_eq,
        bcast_eq,
        sub_eq,
        d2_eq,
        d4_eq,
        sum2_eq,
        m2_eq,
        sum4_eq,
        m4_eq,
        denom_eq,
        kurt_eq,
    ] = jaxpr.equations.as_slice()
    else {
        return None;
    };

    // mean = ReduceSum(x, axis=1) / n
    let s1 = single_output_for_param_primitive(sum1_eq, Primitive::ReduceSum)?;
    if sum1_eq.inputs.as_slice() != [Atom::Var(x)] || !axis1_reduce_params(&sum1_eq.params) {
        return None;
    }
    let mean = single_output_for_primitive(mean_eq, Primitive::Div)?;
    let [Atom::Var(mean_num), mean_div] = mean_eq.inputs.as_slice() else {
        return None;
    };
    if *mean_num != s1 {
        return None;
    }
    // broadcast mean back to [rows,cols]
    let mean_b = single_output_for_param_primitive(bcast_eq, Primitive::BroadcastInDim)?;
    if bcast_eq.inputs.as_slice() != [Atom::Var(mean)]
        || !rank2_axis0_broadcast_params(&bcast_eq.params)
    {
        return None;
    }
    // d = x - mean_b
    let d = single_output_for_primitive(sub_eq, Primitive::Sub)?;
    if sub_eq.inputs.as_slice() != [Atom::Var(x), Atom::Var(mean_b)] {
        return None;
    }
    // d2 = d * d, d4 = d2 * d2
    let d2 = single_output_for_primitive(d2_eq, Primitive::Mul)?;
    if d2_eq.inputs.as_slice() != [Atom::Var(d), Atom::Var(d)] {
        return None;
    }
    let d4 = single_output_for_primitive(d4_eq, Primitive::Mul)?;
    if d4_eq.inputs.as_slice() != [Atom::Var(d2), Atom::Var(d2)] {
        return None;
    }
    // m2 = ReduceSum(d2, axis=1) / n
    let sum2 = single_output_for_param_primitive(sum2_eq, Primitive::ReduceSum)?;
    if sum2_eq.inputs.as_slice() != [Atom::Var(d2)] || !axis1_reduce_params(&sum2_eq.params) {
        return None;
    }
    let m2 = single_output_for_primitive(m2_eq, Primitive::Div)?;
    let [Atom::Var(m2_num), m2_div] = m2_eq.inputs.as_slice() else {
        return None;
    };
    if *m2_num != sum2 {
        return None;
    }
    // m4 = ReduceSum(d4, axis=1) / n
    let sum4 = single_output_for_param_primitive(sum4_eq, Primitive::ReduceSum)?;
    if sum4_eq.inputs.as_slice() != [Atom::Var(d4)] || !axis1_reduce_params(&sum4_eq.params) {
        return None;
    }
    let m4 = single_output_for_primitive(m4_eq, Primitive::Div)?;
    let [Atom::Var(m4_num), m4_div] = m4_eq.inputs.as_slice() else {
        return None;
    };
    if *m4_num != sum4 {
        return None;
    }
    // denom = m2 * m2, kurt = m4 / denom
    let denom = single_output_for_primitive(denom_eq, Primitive::Mul)?;
    if denom_eq.inputs.as_slice() != [Atom::Var(m2), Atom::Var(m2)] {
        return None;
    }
    let kurt = single_output_for_primitive(kurt_eq, Primitive::Div)?;
    let [Atom::Var(kurt_num), Atom::Var(kurt_den)] = kurt_eq.inputs.as_slice() else {
        return None;
    };
    if kurt != out || *kurt_num != m4 || *kurt_den != denom {
        return None;
    }

    let Value::Tensor(input) = &args[0] else {
        return None;
    };
    if input.dtype != DType::F64 || input.shape.dims.len() != 2 {
        return None;
    }
    let rows = input.shape.dims[0] as usize;
    let cols = input.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }
    // All three divisors must be exactly the row length `n = cols` (as f64), matching `kurtosis_row`.
    let n_bits = (cols as f64).to_bits();
    if f64_literal(mean_div)?.to_bits() != n_bits
        || f64_literal(m2_div)?.to_bits() != n_bits
        || f64_literal(m4_div)?.to_bits() != n_bits
    {
        return None;
    }
    let expected_shape = format!("{rows},{cols}");
    if bcast_eq.params.get("shape").map(String::as_str) != Some(expected_shape.as_str()) {
        return None;
    }

    let values = input.elements.as_f64_slice()?;
    if !values.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::kurtosis_2d(values, rows, cols);
    Some(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32],
            },
            output,
        )
        .map(Value::Tensor)
        .map(|value| vec![value])
        .map_err(EvalError::InvalidTensor)
        .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the row-wise population covariance `mean((a-ā)(b-b̄))`
/// (`ddof=0`): the 11-equation, TWO-input graph `ReduceSum(a) → Div(mean_a) → BroadcastInDim →
/// ReduceSum(b) → Div(mean_b) → BroadcastInDim → Sub(ca) → Sub(cb) → Mul → ReduceSum → Div`, two f64
/// [rows,cols] inputs, one [rows] output (rank-reducing). NO transcendentals — a pure second cross-
/// moment reduction (add/sub/mul/div only), distinct from the 1-input variance/std/skew/kurt moments
/// and from the (normalized) Pearson correlation. The general fuser cannot fuse it (the three
/// reductions break the elementwise fuser), so the decomposed path materializes the full [rows,cols]
/// two-centering + product intermediates plus per-reduction Vecs. Fused via the row-parallel
/// `covariance_2d`, whose index-order sums + `sum/n` mean grouping match the graph bit-for-bit. All
/// three `Div` divisors must equal `cols` (as f64) and both broadcasts must restore exactly
/// `rows,cols`; finite dense rank-2 f64 with matching shapes only, else falls through.
fn try_eval_top_level_covariance_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 2
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 11
        || args.len() != 2
    {
        return None;
    }

    let a = jaxpr.invars[0];
    let b = jaxpr.invars[1];
    let out = jaxpr.outvars[0];
    let [
        suma_eq,
        mean_a_eq,
        bcast_a_eq,
        sumb_eq,
        mean_b_eq,
        bcast_b_eq,
        ca_eq,
        cb_eq,
        prod_eq,
        sum_eq,
        cov_eq,
    ] = jaxpr.equations.as_slice()
    else {
        return None;
    };

    // mean_a = ReduceSum(a, axis=1) / n, broadcast to [rows,cols]
    let suma = single_output_for_param_primitive(suma_eq, Primitive::ReduceSum)?;
    if suma_eq.inputs.as_slice() != [Atom::Var(a)] || !axis1_reduce_params(&suma_eq.params) {
        return None;
    }
    let mean_a = single_output_for_primitive(mean_a_eq, Primitive::Div)?;
    let [Atom::Var(mean_a_num), mean_a_div] = mean_a_eq.inputs.as_slice() else {
        return None;
    };
    if *mean_a_num != suma {
        return None;
    }
    let mean_a_b = single_output_for_param_primitive(bcast_a_eq, Primitive::BroadcastInDim)?;
    if bcast_a_eq.inputs.as_slice() != [Atom::Var(mean_a)]
        || !rank2_axis0_broadcast_params(&bcast_a_eq.params)
    {
        return None;
    }

    // mean_b = ReduceSum(b, axis=1) / n, broadcast to [rows,cols]
    let sumb = single_output_for_param_primitive(sumb_eq, Primitive::ReduceSum)?;
    if sumb_eq.inputs.as_slice() != [Atom::Var(b)] || !axis1_reduce_params(&sumb_eq.params) {
        return None;
    }
    let mean_b = single_output_for_primitive(mean_b_eq, Primitive::Div)?;
    let [Atom::Var(mean_b_num), mean_b_div] = mean_b_eq.inputs.as_slice() else {
        return None;
    };
    if *mean_b_num != sumb {
        return None;
    }
    let mean_b_b = single_output_for_param_primitive(bcast_b_eq, Primitive::BroadcastInDim)?;
    if bcast_b_eq.inputs.as_slice() != [Atom::Var(mean_b)]
        || !rank2_axis0_broadcast_params(&bcast_b_eq.params)
    {
        return None;
    }

    // ca = a - mean_a_b, cb = b - mean_b_b
    let ca = single_output_for_primitive(ca_eq, Primitive::Sub)?;
    if ca_eq.inputs.as_slice() != [Atom::Var(a), Atom::Var(mean_a_b)] {
        return None;
    }
    let cb = single_output_for_primitive(cb_eq, Primitive::Sub)?;
    if cb_eq.inputs.as_slice() != [Atom::Var(b), Atom::Var(mean_b_b)] {
        return None;
    }
    // prod = ca * cb (Mul commutative → either operand order)
    let prod = single_output_for_primitive(prod_eq, Primitive::Mul)?;
    let prod_ok = matches!(
        prod_eq.inputs.as_slice(),
        [Atom::Var(x), Atom::Var(y)] if (*x == ca && *y == cb) || (*x == cb && *y == ca)
    );
    if !prod_ok {
        return None;
    }
    // cov = ReduceSum(prod, axis=1) / n
    let s = single_output_for_param_primitive(sum_eq, Primitive::ReduceSum)?;
    if sum_eq.inputs.as_slice() != [Atom::Var(prod)] || !axis1_reduce_params(&sum_eq.params) {
        return None;
    }
    let cov = single_output_for_primitive(cov_eq, Primitive::Div)?;
    let [Atom::Var(cov_num), cov_div] = cov_eq.inputs.as_slice() else {
        return None;
    };
    if cov != out || *cov_num != s {
        return None;
    }

    let Value::Tensor(a_t) = &args[0] else {
        return None;
    };
    let Value::Tensor(b_t) = &args[1] else {
        return None;
    };
    if a_t.dtype != DType::F64
        || b_t.dtype != DType::F64
        || a_t.shape.dims.len() != 2
        || b_t.shape.dims != a_t.shape.dims
    {
        return None;
    }
    let rows = a_t.shape.dims[0] as usize;
    let cols = a_t.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }
    // All three divisors must be exactly the row length `n = cols` (as f64), matching `covariance_row`.
    let n_bits = (cols as f64).to_bits();
    if f64_literal(mean_a_div)?.to_bits() != n_bits
        || f64_literal(mean_b_div)?.to_bits() != n_bits
        || f64_literal(cov_div)?.to_bits() != n_bits
    {
        return None;
    }
    let expected_shape = format!("{rows},{cols}");
    if bcast_a_eq.params.get("shape").map(String::as_str) != Some(expected_shape.as_str())
        || bcast_b_eq.params.get("shape").map(String::as_str) != Some(expected_shape.as_str())
    {
        return None;
    }

    let a_v = a_t.elements.as_f64_slice()?;
    let b_v = b_t.elements.as_f64_slice()?;
    if !a_v.iter().all(|value| value.is_finite()) || !b_v.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::covariance_2d(a_v, b_v, rows, cols);
    Some(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32],
            },
            output,
        )
        .map(Value::Tensor)
        .map(|value| vec![value])
        .map_err(EvalError::InvalidTensor)
        .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the row-wise L2 (Euclidean / Frobenius) norm `sqrt(Σ x²)`
/// (`jax.numpy.linalg.norm(x, axis=-1)`): the 3-equation graph `Mul(x,x) → ReduceSum(axis=1) → Sqrt`,
/// one f64 [rows,cols] input, one [rows] output (rank-reducing). NO transcendentals (a sum-of-squares
/// reduction + a correctly-rounded `Sqrt`), and the simplest graph in the reduction family — no `Div`
/// or `BroadcastInDim`. The general fuser cannot fuse it (the reduction breaks the elementwise fuser),
/// so the decomposed path materializes the full [rows,cols] squared intermediate plus the reduction
/// Vec. Fused via the row-parallel `l2_norm_2d`, whose index-order sum-of-squares + `f64::sqrt` match
/// the graph bit-for-bit. Finite dense rank-2 f64 only, else falls through.
fn try_eval_top_level_l2_norm_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 3
        || args.len() != 1
    {
        return None;
    }

    let x = jaxpr.invars[0];
    let out = jaxpr.outvars[0];
    let [sq_eq, sum_eq, sqrt_eq] = jaxpr.equations.as_slice() else {
        return None;
    };

    // sq = x * x (both operands the same var)
    let sq = single_output_for_primitive(sq_eq, Primitive::Mul)?;
    if sq_eq.inputs.as_slice() != [Atom::Var(x), Atom::Var(x)] {
        return None;
    }
    // s = ReduceSum(sq, axis=1)
    let s = single_output_for_param_primitive(sum_eq, Primitive::ReduceSum)?;
    if sum_eq.inputs.as_slice() != [Atom::Var(sq)] || !axis1_reduce_params(&sum_eq.params) {
        return None;
    }
    // out = Sqrt(s)
    let norm = single_output_for_primitive(sqrt_eq, Primitive::Sqrt)?;
    if sqrt_eq.inputs.as_slice() != [Atom::Var(s)] || norm != out {
        return None;
    }

    let Value::Tensor(input) = &args[0] else {
        return None;
    };
    if input.dtype != DType::F64 || input.shape.dims.len() != 2 {
        return None;
    }
    let rows = input.shape.dims[0] as usize;
    let cols = input.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }

    let values = input.elements.as_f64_slice()?;
    if !values.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::l2_norm_2d(values, rows, cols);
    Some(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32],
            },
            output,
        )
        .map(Value::Tensor)
        .map(|value| vec![value])
        .map_err(EvalError::InvalidTensor)
        .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the row-wise L2-normalize `x / sqrt(Σ x²)`
/// (`jax.nn.normalize` with the default L2 order, no epsilon): the 5-equation graph
/// `Mul(x,x) → ReduceSum(axis=1) → Sqrt → BroadcastInDim → Div`, one f64 [rows,cols] input, one
/// [rows,cols] output (rank-PRESERVING — distinct from the reducing norms/moments). NO transcendentals
/// (sum-of-squares + `Sqrt` + a true divide). The general fuser cannot fuse it (the reduction breaks
/// the elementwise fuser), so the decomposed path materializes the full [rows,cols] squared + broadcast
/// intermediates plus the reduction Vec. Fused via the row-parallel `l2_normalize_2d`, whose
/// index-order sum-of-squares, `f64::sqrt`, and per-element TRUE division match the graph bit-for-bit
/// (the `Div` primitive — NOT reciprocal-multiply). The broadcast must restore exactly `rows,cols`;
/// finite dense rank-2 f64 only, else falls through.
fn try_eval_top_level_l2_normalize_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 5
        || args.len() != 1
    {
        return None;
    }

    let x = jaxpr.invars[0];
    let out = jaxpr.outvars[0];
    let [sq_eq, sum_eq, sqrt_eq, bcast_eq, div_eq] = jaxpr.equations.as_slice() else {
        return None;
    };

    // sq = x * x (both operands the same var)
    let sq = single_output_for_primitive(sq_eq, Primitive::Mul)?;
    if sq_eq.inputs.as_slice() != [Atom::Var(x), Atom::Var(x)] {
        return None;
    }
    // s = ReduceSum(sq, axis=1)
    let s = single_output_for_param_primitive(sum_eq, Primitive::ReduceSum)?;
    if sum_eq.inputs.as_slice() != [Atom::Var(sq)] || !axis1_reduce_params(&sum_eq.params) {
        return None;
    }
    // norm = Sqrt(s)
    let norm = single_output_for_primitive(sqrt_eq, Primitive::Sqrt)?;
    if sqrt_eq.inputs.as_slice() != [Atom::Var(s)] {
        return None;
    }
    // norm_b = BroadcastInDim(norm) to [rows,cols]
    let norm_b = single_output_for_param_primitive(bcast_eq, Primitive::BroadcastInDim)?;
    if bcast_eq.inputs.as_slice() != [Atom::Var(norm)]
        || !rank2_axis0_broadcast_params(&bcast_eq.params)
    {
        return None;
    }
    // out = Div(x, norm_b) (non-commutative → numerator = x)
    let quotient = single_output_for_primitive(div_eq, Primitive::Div)?;
    if quotient != out || div_eq.inputs.as_slice() != [Atom::Var(x), Atom::Var(norm_b)] {
        return None;
    }

    let Value::Tensor(input) = &args[0] else {
        return None;
    };
    if input.dtype != DType::F64 || input.shape.dims.len() != 2 {
        return None;
    }
    let rows = input.shape.dims[0] as usize;
    let cols = input.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }
    let expected_shape = format!("{rows},{cols}");
    if bcast_eq.params.get("shape").map(String::as_str) != Some(expected_shape.as_str()) {
        return None;
    }

    let values = input.elements.as_f64_slice()?;
    if !values.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::l2_normalize_2d(values, rows, cols);
    Some(
        TensorValue::new_f64_values(input.shape.clone(), output)
            .map(Value::Tensor)
            .map(|value| vec![value])
            .map_err(EvalError::InvalidTensor)
            .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the row-wise L1 (Manhattan / taxicab) norm `Σ |x|`
/// (`jax.numpy.linalg.norm(x, ord=1, axis=-1)`): the 2-equation graph `Abs(x) → ReduceSum(axis=1)`,
/// one f64 [rows,cols] input, one [rows] output (rank-reducing). NO transcendentals and the SMALLEST
/// graph in the reduction family — an absolute-value elementwise op + a reduction. The general fuser
/// cannot fuse it (the reduction breaks the elementwise fuser), so the decomposed path materializes
/// the full [rows,cols] absolute-value intermediate plus the reduction Vec. Fused via the row-parallel
/// `l1_norm_2d`, whose index-order sum of `|x[i]|` matches the graph bit-for-bit (`f64::abs` = the
/// `Abs` primitive). One-input `Σ|x|`, distinct from the 2-input Manhattan distance `Σ|a-b|`.
/// Finite dense rank-2 f64 only, else falls through.
fn try_eval_top_level_l1_norm_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 2
        || args.len() != 1
    {
        return None;
    }

    let x = jaxpr.invars[0];
    let out = jaxpr.outvars[0];
    let [abs_eq, sum_eq] = jaxpr.equations.as_slice() else {
        return None;
    };

    // a = Abs(x)
    let a = single_output_for_primitive(abs_eq, Primitive::Abs)?;
    if abs_eq.inputs.as_slice() != [Atom::Var(x)] {
        return None;
    }
    // out = ReduceSum(a, axis=1)
    let s = single_output_for_param_primitive(sum_eq, Primitive::ReduceSum)?;
    if sum_eq.inputs.as_slice() != [Atom::Var(a)] || !axis1_reduce_params(&sum_eq.params) {
        return None;
    }
    if s != out {
        return None;
    }

    let Value::Tensor(input) = &args[0] else {
        return None;
    };
    if input.dtype != DType::F64 || input.shape.dims.len() != 2 {
        return None;
    }
    let rows = input.shape.dims[0] as usize;
    let cols = input.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }

    let values = input.elements.as_f64_slice()?;
    if !values.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::l1_norm_2d(values, rows, cols);
    Some(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32],
            },
            output,
        )
        .map(Value::Tensor)
        .map(|value| vec![value])
        .map_err(EvalError::InvalidTensor)
        .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the row-wise root-mean-square `sqrt(mean(x²))`
/// (`sqrt(jax.numpy.mean(x**2, axis=-1))`): the 4-equation graph `Mul(x,x) → ReduceSum(axis=1) →
/// Div(·, n) → Sqrt`, one f64 [rows,cols] input, one [rows] output (rank-reducing). NO transcendentals
/// (sum-of-squares + a mean + a correctly-rounded `Sqrt`). Distinct from the L2 norm (which omits the
/// `/n`) and population std (which subtracts the mean first). The general fuser cannot fuse it (the
/// reduction breaks the elementwise fuser), so the decomposed path materializes the full [rows,cols]
/// squared intermediate plus the reduction Vec. Fused via the row-parallel `rms_2d`, whose index-order
/// sum-of-squares + `sum/n` + `f64::sqrt` match the graph bit-for-bit. The `Div` divisor must equal
/// `cols` (as f64); finite dense rank-2 f64 only, else falls through.
fn try_eval_top_level_rms_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 4
        || args.len() != 1
    {
        return None;
    }

    let x = jaxpr.invars[0];
    let out = jaxpr.outvars[0];
    let [sq_eq, sum_eq, mean_eq, sqrt_eq] = jaxpr.equations.as_slice() else {
        return None;
    };

    // sq = x * x (both operands the same var)
    let sq = single_output_for_primitive(sq_eq, Primitive::Mul)?;
    if sq_eq.inputs.as_slice() != [Atom::Var(x), Atom::Var(x)] {
        return None;
    }
    // s = ReduceSum(sq, axis=1)
    let s = single_output_for_param_primitive(sum_eq, Primitive::ReduceSum)?;
    if sum_eq.inputs.as_slice() != [Atom::Var(sq)] || !axis1_reduce_params(&sum_eq.params) {
        return None;
    }
    // ms = Div(s, n)
    let ms = single_output_for_primitive(mean_eq, Primitive::Div)?;
    let [Atom::Var(ms_num), ms_div] = mean_eq.inputs.as_slice() else {
        return None;
    };
    if *ms_num != s {
        return None;
    }
    // out = Sqrt(ms)
    let rms = single_output_for_primitive(sqrt_eq, Primitive::Sqrt)?;
    if sqrt_eq.inputs.as_slice() != [Atom::Var(ms)] || rms != out {
        return None;
    }

    let Value::Tensor(input) = &args[0] else {
        return None;
    };
    if input.dtype != DType::F64 || input.shape.dims.len() != 2 {
        return None;
    }
    let rows = input.shape.dims[0] as usize;
    let cols = input.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }
    // The divisor must be exactly the row length `n = cols` (as f64), matching `rms_row`.
    if f64_literal(ms_div)?.to_bits() != (cols as f64).to_bits() {
        return None;
    }

    let values = input.elements.as_f64_slice()?;
    if !values.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::rms_2d(values, rows, cols);
    Some(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32],
            },
            output,
        )
        .map(Value::Tensor)
        .map(|value| vec![value])
        .map_err(EvalError::InvalidTensor)
        .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the row-wise z-score standardize `(x − μ) / σ` (NO epsilon),
/// `σ = sqrt(mean((x-μ)²))` (`scipy.stats.zscore(x, axis=-1, ddof=0)` / per-row StandardScaler): the
/// 10-equation graph `ReduceSum → Div(mean) → BroadcastInDim → Sub → Mul(square) → ReduceSum →
/// Div(var) → Sqrt → BroadcastInDim → Div`, one f64 [rows,cols] input, one [rows,cols] output
/// (rank-PRESERVING). NO transcendentals — the centered-moment analog of `l2_normalize`. The general
/// fuser cannot fuse it (the two reductions break the elementwise fuser), so the decomposed path
/// materializes the full [rows,cols] centered / squared / broadcast intermediates plus the reduction
/// Vecs. Fused via the row-parallel `zscore_2d`, whose index-order mean+variance, `f64::sqrt`, and
/// per-element TRUE division match the graph bit-for-bit (the `Div` primitive — NOT reciprocal-
/// multiply; distinct from `rms_norm`'s `Rsqrt×Mul`). Both `Div(·,n)` divisors must equal `cols` (as
/// f64) and both broadcasts must restore exactly `rows,cols`; finite dense rank-2 f64 only.
fn try_eval_top_level_zscore_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 10
        || args.len() != 1
    {
        return None;
    }

    let x = jaxpr.invars[0];
    let out = jaxpr.outvars[0];
    let [
        sum1_eq,
        mean_eq,
        mean_bcast_eq,
        sub_eq,
        sq_eq,
        sum2_eq,
        var_eq,
        sqrt_eq,
        std_bcast_eq,
        div_eq,
    ] = jaxpr.equations.as_slice()
    else {
        return None;
    };

    // mean = ReduceSum(x, axis=1) / n, broadcast to [rows,cols]
    let s1 = single_output_for_param_primitive(sum1_eq, Primitive::ReduceSum)?;
    if sum1_eq.inputs.as_slice() != [Atom::Var(x)] || !axis1_reduce_params(&sum1_eq.params) {
        return None;
    }
    let mean = single_output_for_primitive(mean_eq, Primitive::Div)?;
    let [Atom::Var(mean_num), mean_div] = mean_eq.inputs.as_slice() else {
        return None;
    };
    if *mean_num != s1 {
        return None;
    }
    let mean_b = single_output_for_param_primitive(mean_bcast_eq, Primitive::BroadcastInDim)?;
    if mean_bcast_eq.inputs.as_slice() != [Atom::Var(mean)]
        || !rank2_axis0_broadcast_params(&mean_bcast_eq.params)
    {
        return None;
    }
    // centered = x - mean_b, sq = centered * centered
    let centered = single_output_for_primitive(sub_eq, Primitive::Sub)?;
    if sub_eq.inputs.as_slice() != [Atom::Var(x), Atom::Var(mean_b)] {
        return None;
    }
    let sq = single_output_for_primitive(sq_eq, Primitive::Mul)?;
    if sq_eq.inputs.as_slice() != [Atom::Var(centered), Atom::Var(centered)] {
        return None;
    }
    // var = ReduceSum(sq, axis=1) / n
    let s2 = single_output_for_param_primitive(sum2_eq, Primitive::ReduceSum)?;
    if sum2_eq.inputs.as_slice() != [Atom::Var(sq)] || !axis1_reduce_params(&sum2_eq.params) {
        return None;
    }
    let var = single_output_for_primitive(var_eq, Primitive::Div)?;
    let [Atom::Var(var_num), var_div] = var_eq.inputs.as_slice() else {
        return None;
    };
    if *var_num != s2 {
        return None;
    }
    // std = Sqrt(var), broadcast to [rows,cols]
    let std = single_output_for_primitive(sqrt_eq, Primitive::Sqrt)?;
    if sqrt_eq.inputs.as_slice() != [Atom::Var(var)] {
        return None;
    }
    let std_b = single_output_for_param_primitive(std_bcast_eq, Primitive::BroadcastInDim)?;
    if std_bcast_eq.inputs.as_slice() != [Atom::Var(std)]
        || !rank2_axis0_broadcast_params(&std_bcast_eq.params)
    {
        return None;
    }
    // out = Div(centered, std_b) (non-commutative → numerator = centered)
    let quotient = single_output_for_primitive(div_eq, Primitive::Div)?;
    if quotient != out || div_eq.inputs.as_slice() != [Atom::Var(centered), Atom::Var(std_b)] {
        return None;
    }

    let Value::Tensor(input) = &args[0] else {
        return None;
    };
    if input.dtype != DType::F64 || input.shape.dims.len() != 2 {
        return None;
    }
    let rows = input.shape.dims[0] as usize;
    let cols = input.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }
    // Both divisors must be exactly the row length `n = cols` (as f64), matching `zscore_row_into`.
    let n_bits = (cols as f64).to_bits();
    if f64_literal(mean_div)?.to_bits() != n_bits || f64_literal(var_div)?.to_bits() != n_bits {
        return None;
    }
    let expected_shape = format!("{rows},{cols}");
    if mean_bcast_eq.params.get("shape").map(String::as_str) != Some(expected_shape.as_str())
        || std_bcast_eq.params.get("shape").map(String::as_str) != Some(expected_shape.as_str())
    {
        return None;
    }

    let values = input.elements.as_f64_slice()?;
    if !values.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::zscore_2d(values, rows, cols);
    Some(
        TensorValue::new_f64_values(input.shape.clone(), output)
            .map(Value::Tensor)
            .map(|value| vec![value])
            .map_err(EvalError::InvalidTensor)
            .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for row-wise min-max (feature-scaling) normalization
/// `(x − x.min(-1)) / (x.max(-1) − x.min(-1))` (per-row sklearn `MinMaxScaler` to `[0,1]`): the
/// 7-equation graph `ReduceMax(axis=1) → BroadcastInDim | ReduceMin(axis=1) → BroadcastInDim |
/// Sub(x, min) | Sub(max, min) | Div`, one f64 [rows,cols] input → one [rows,cols] output
/// (rank-PRESERVING). Computed via the row-parallel [`fj_lax::nn::minmax_normalize_2d`], which fuses the
/// max/min into one pass (bit-identical to the two separate `ReduceMax`/`ReduceMin` folds — see the
/// kernel doc) and divides `(x − min)` by the scalar row range (identical to the elementwise `Div` by
/// the constant broadcast `Sub(max, min)` tensor). DISTINCT from `zscore_2d` (centered/std, uses
/// `ReduceSum`+`Sqrt`) and from the producerless `ptp` (`max − min` alone regresses — this op has the
/// materialized shifted/range intermediates a fused kernel eliminates). Falls through unless the input
/// is dense finite rank-2 f64 (a nonfinite element makes the fused max/min diverge from the folds).
fn try_eval_top_level_minmax_normalize_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 7
        || args.len() != 1
    {
        return None;
    }

    let x = jaxpr.invars[0];
    let out = jaxpr.outvars[0];
    let [
        max_eq,
        max_bcast_eq,
        min_eq,
        min_bcast_eq,
        num_eq,
        den_eq,
        div_eq,
    ] = jaxpr.equations.as_slice()
    else {
        return None;
    };

    // max = ReduceMax(x, axis=1), broadcast to [rows,cols]
    let mx = single_output_for_param_primitive(max_eq, Primitive::ReduceMax)?;
    if max_eq.inputs.as_slice() != [Atom::Var(x)] || !axis1_reduce_params(&max_eq.params) {
        return None;
    }
    let mx_b = single_output_for_param_primitive(max_bcast_eq, Primitive::BroadcastInDim)?;
    if max_bcast_eq.inputs.as_slice() != [Atom::Var(mx)]
        || !rank2_axis0_broadcast_params(&max_bcast_eq.params)
    {
        return None;
    }
    // min = ReduceMin(x, axis=1), broadcast to [rows,cols]
    let mn = single_output_for_param_primitive(min_eq, Primitive::ReduceMin)?;
    if min_eq.inputs.as_slice() != [Atom::Var(x)] || !axis1_reduce_params(&min_eq.params) {
        return None;
    }
    let mn_b = single_output_for_param_primitive(min_bcast_eq, Primitive::BroadcastInDim)?;
    if min_bcast_eq.inputs.as_slice() != [Atom::Var(mn)]
        || !rank2_axis0_broadcast_params(&min_bcast_eq.params)
    {
        return None;
    }
    // num = Sub(x, min_b); den = Sub(max_b, min_b) (both non-commutative → operand order fixed)
    let num = single_output_for_primitive(num_eq, Primitive::Sub)?;
    if num_eq.inputs.as_slice() != [Atom::Var(x), Atom::Var(mn_b)] {
        return None;
    }
    let den = single_output_for_primitive(den_eq, Primitive::Sub)?;
    if den_eq.inputs.as_slice() != [Atom::Var(mx_b), Atom::Var(mn_b)] {
        return None;
    }
    // out = Div(num, den)
    let quotient = single_output_for_primitive(div_eq, Primitive::Div)?;
    if quotient != out || div_eq.inputs.as_slice() != [Atom::Var(num), Atom::Var(den)] {
        return None;
    }

    let Value::Tensor(input) = &args[0] else {
        return None;
    };
    if input.dtype != DType::F64 || input.shape.dims.len() != 2 {
        return None;
    }
    let rows = input.shape.dims[0] as usize;
    let cols = input.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }
    let expected_shape = format!("{rows},{cols}");
    if max_bcast_eq.params.get("shape").map(String::as_str) != Some(expected_shape.as_str())
        || min_bcast_eq.params.get("shape").map(String::as_str) != Some(expected_shape.as_str())
    {
        return None;
    }

    let values = input.elements.as_f64_slice()?;
    if !values.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::minmax_normalize_2d(values, rows, cols);
    Some(
        TensorValue::new_f64_values(input.shape.clone(), output)
            .map(Value::Tensor)
            .map(|value| vec![value])
            .map_err(EvalError::InvalidTensor)
            .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the row-wise population variance `mean((x - mean(x))²)`
/// (`jax.numpy.var(x, axis=-1)`, `ddof=0`): the 7-equation graph `ReduceSum(axis=1) → Div(·, n) →
/// BroadcastInDim → Sub → Mul(square) → ReduceSum(axis=1) → Div(·, n)`. One f64 [rows,cols] input,
/// one [rows] output (rank-reducing). NO transcendentals — a pure statistical moment reduction,
/// distinct from the exp/log reductions (logsumexp/cross-entropy). The general fuser cannot fuse it
/// (the two reductions break the elementwise fuser), so the decomposed path materializes the full
/// [rows,cols] centered+squared intermediates plus per-reduction Vecs. Fused via the row-parallel
/// `variance_2d`, whose index-order sums + `sum/n` mean grouping match the graph bit-for-bit. The
/// two `Div` divisors must both equal `cols` (as f64) and the broadcast must restore exactly
/// `rows,cols`; finite dense rank-2 f64 only, else falls through.
fn try_eval_top_level_var_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 7
        || args.len() != 1
    {
        return None;
    }

    let x = jaxpr.invars[0];
    let out = jaxpr.outvars[0];
    let [sum1_eq, mean_eq, bcast_eq, sub_eq, sq_eq, sum2_eq, var_eq] = jaxpr.equations.as_slice()
    else {
        return None;
    };

    // mean = ReduceSum(x, axis=1) / n
    let s1 = single_output_for_param_primitive(sum1_eq, Primitive::ReduceSum)?;
    if sum1_eq.inputs.as_slice() != [Atom::Var(x)] || !axis1_reduce_params(&sum1_eq.params) {
        return None;
    }
    let mean = single_output_for_primitive(mean_eq, Primitive::Div)?;
    let [Atom::Var(mean_num), mean_div] = mean_eq.inputs.as_slice() else {
        return None;
    };
    if *mean_num != s1 {
        return None;
    }
    // broadcast mean back to [rows,cols]
    let mean_b = single_output_for_param_primitive(bcast_eq, Primitive::BroadcastInDim)?;
    if bcast_eq.inputs.as_slice() != [Atom::Var(mean)]
        || !rank2_axis0_broadcast_params(&bcast_eq.params)
    {
        return None;
    }
    // centered = x - mean_b
    let centered = single_output_for_primitive(sub_eq, Primitive::Sub)?;
    if sub_eq.inputs.as_slice() != [Atom::Var(x), Atom::Var(mean_b)] {
        return None;
    }
    // sq = centered * centered (both operands the same var)
    let sq = single_output_for_primitive(sq_eq, Primitive::Mul)?;
    if sq_eq.inputs.as_slice() != [Atom::Var(centered), Atom::Var(centered)] {
        return None;
    }
    // var = ReduceSum(sq, axis=1) / n
    let s2 = single_output_for_param_primitive(sum2_eq, Primitive::ReduceSum)?;
    if sum2_eq.inputs.as_slice() != [Atom::Var(sq)] || !axis1_reduce_params(&sum2_eq.params) {
        return None;
    }
    let var = single_output_for_primitive(var_eq, Primitive::Div)?;
    let [Atom::Var(var_num), var_div] = var_eq.inputs.as_slice() else {
        return None;
    };
    if var != out || *var_num != s2 {
        return None;
    }

    let Value::Tensor(input) = &args[0] else {
        return None;
    };
    if input.dtype != DType::F64 || input.shape.dims.len() != 2 {
        return None;
    }
    let rows = input.shape.dims[0] as usize;
    let cols = input.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }
    // Both divisors must be exactly the row length `n = cols` (as f64), matching `variance_row`.
    let n_bits = (cols as f64).to_bits();
    if f64_literal(mean_div)?.to_bits() != n_bits || f64_literal(var_div)?.to_bits() != n_bits {
        return None;
    }
    let expected_shape = format!("{rows},{cols}");
    if bcast_eq.params.get("shape").map(String::as_str) != Some(expected_shape.as_str()) {
        return None;
    }

    let values = input.elements.as_f64_slice()?;
    if !values.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::variance_2d(values, rows, cols);
    Some(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32],
            },
            output,
        )
        .map(Value::Tensor)
        .map(|value| vec![value])
        .map_err(EvalError::InvalidTensor)
        .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the row-wise cosine similarity
/// `Σ(a·b) / (√Σa² · √Σb²)` along the last axis (the ubiquitous embedding/retrieval metric): the
/// 10-equation, TWO-input graph `Mul(a,b) → ReduceSum(dot) | Mul(a,a) → ReduceSum → Sqrt(‖a‖) |
/// Mul(b,b) → ReduceSum → Sqrt(‖b‖) | Mul(denom) → Div(dot, denom)`. Two f64 [rows,cols] inputs, one
/// [rows] output (rank-reducing). NO transcendentals (Sqrt is IEEE-exact) — the fused kernel is pure
/// bandwidth + cheap FLOPs, so like `var_2d` this yields a large multiple. The general fuser cannot
/// fuse it (three reductions break the elementwise fuser), so the decomposed path materializes three
/// full [rows,cols] intermediates (a·b, a², b²) plus per-reduction Vecs. Fused via the row-parallel
/// `cosine_similarity_2d`, whose independent index-order dot/‖a‖²/‖b‖² sums match the graph bit-for-
/// bit; finite dense rank-2 f64 only, else falls through.
fn try_eval_top_level_cosine_similarity_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 2
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 10
        || args.len() != 2
    {
        return None;
    }

    let a = jaxpr.invars[0];
    let b = jaxpr.invars[1];
    let out = jaxpr.outvars[0];
    let [
        ab_eq,
        dot_eq,
        a2_eq,
        sa_eq,
        na_eq,
        b2_eq,
        sb_eq,
        nb_eq,
        denom_eq,
        div_eq,
    ] = jaxpr.equations.as_slice()
    else {
        return None;
    };

    // dot = ReduceSum(a * b) (Mul commutative → either operand order)
    let ab = single_output_for_primitive(ab_eq, Primitive::Mul)?;
    let ab_ok = matches!(
        ab_eq.inputs.as_slice(),
        [Atom::Var(x), Atom::Var(y)] if (*x == a && *y == b) || (*x == b && *y == a)
    );
    if !ab_ok {
        return None;
    }
    let dot = single_output_for_param_primitive(dot_eq, Primitive::ReduceSum)?;
    if dot_eq.inputs.as_slice() != [Atom::Var(ab)] || !axis1_reduce_params(&dot_eq.params) {
        return None;
    }

    // ‖a‖ = Sqrt(ReduceSum(a * a))
    let a2 = single_output_for_primitive(a2_eq, Primitive::Mul)?;
    if a2_eq.inputs.as_slice() != [Atom::Var(a), Atom::Var(a)] {
        return None;
    }
    let sa = single_output_for_param_primitive(sa_eq, Primitive::ReduceSum)?;
    if sa_eq.inputs.as_slice() != [Atom::Var(a2)] || !axis1_reduce_params(&sa_eq.params) {
        return None;
    }
    let na = single_output_for_primitive(na_eq, Primitive::Sqrt)?;
    if na_eq.inputs.as_slice() != [Atom::Var(sa)] {
        return None;
    }

    // ‖b‖ = Sqrt(ReduceSum(b * b))
    let b2 = single_output_for_primitive(b2_eq, Primitive::Mul)?;
    if b2_eq.inputs.as_slice() != [Atom::Var(b), Atom::Var(b)] {
        return None;
    }
    let sb = single_output_for_param_primitive(sb_eq, Primitive::ReduceSum)?;
    if sb_eq.inputs.as_slice() != [Atom::Var(b2)] || !axis1_reduce_params(&sb_eq.params) {
        return None;
    }
    let nb = single_output_for_primitive(nb_eq, Primitive::Sqrt)?;
    if nb_eq.inputs.as_slice() != [Atom::Var(sb)] {
        return None;
    }

    // denom = ‖a‖ · ‖b‖ (Mul commutative → either order)
    let denom = single_output_for_primitive(denom_eq, Primitive::Mul)?;
    let denom_ok = matches!(
        denom_eq.inputs.as_slice(),
        [Atom::Var(x), Atom::Var(y)] if (*x == na && *y == nb) || (*x == nb && *y == na)
    );
    if !denom_ok {
        return None;
    }
    // out = dot / denom (Div non-commutative → numerator = dot)
    let quotient = single_output_for_primitive(div_eq, Primitive::Div)?;
    if quotient != out || div_eq.inputs.as_slice() != [Atom::Var(dot), Atom::Var(denom)] {
        return None;
    }

    let Value::Tensor(a_t) = &args[0] else {
        return None;
    };
    let Value::Tensor(b_t) = &args[1] else {
        return None;
    };
    if a_t.dtype != DType::F64
        || b_t.dtype != DType::F64
        || a_t.shape.dims.len() != 2
        || b_t.shape.dims != a_t.shape.dims
    {
        return None;
    }
    let rows = a_t.shape.dims[0] as usize;
    let cols = a_t.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }

    let a_v = a_t.elements.as_f64_slice()?;
    let b_v = b_t.elements.as_f64_slice()?;
    if !a_v.iter().all(|value| value.is_finite()) || !b_v.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::cosine_similarity_2d(a_v, b_v, rows, cols);
    Some(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32],
            },
            output,
        )
        .map(Value::Tensor)
        .map(|value| vec![value])
        .map_err(EvalError::InvalidTensor)
        .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the row-wise Euclidean (L2) distance `√Σ(a - b)²` along the
/// last axis (the ubiquitous kNN / clustering / contrastive-loss metric): the 4-equation, TWO-input
/// graph `Sub(a, b) → Mul(square) → ReduceSum(axis=1) → Sqrt`. Two f64 [rows,cols] inputs, one
/// [rows] output (rank-reducing). NO transcendentals (`Sqrt` = `f64::sqrt`, IEEE-exact). The general
/// fuser cannot fuse it (the reduction breaks the elementwise fuser), so the decomposed path
/// materializes two full [rows,cols] intermediates (`a-b`, `(a-b)²`). Fused via the row-parallel
/// `euclidean_distance_2d`, whose index-order `Σ(a-b)²` + `sqrt` match the graph bit-for-bit;
/// finite dense rank-2 f64 only, else falls through.
fn try_eval_top_level_euclidean_distance_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 2
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 4
        || args.len() != 2
    {
        return None;
    }

    let a = jaxpr.invars[0];
    let b = jaxpr.invars[1];
    let out = jaxpr.outvars[0];
    let [sub_eq, sq_eq, sum_eq, sqrt_eq] = jaxpr.equations.as_slice() else {
        return None;
    };

    // diff = a - b (Sub non-commutative — operand order fixed)
    let diff = single_output_for_primitive(sub_eq, Primitive::Sub)?;
    if sub_eq.inputs.as_slice() != [Atom::Var(a), Atom::Var(b)] {
        return None;
    }
    // sq = diff * diff (both operands the same var)
    let sq = single_output_for_primitive(sq_eq, Primitive::Mul)?;
    if sq_eq.inputs.as_slice() != [Atom::Var(diff), Atom::Var(diff)] {
        return None;
    }
    // s = ReduceSum(sq, axis=1)
    let s = single_output_for_param_primitive(sum_eq, Primitive::ReduceSum)?;
    if sum_eq.inputs.as_slice() != [Atom::Var(sq)] || !axis1_reduce_params(&sum_eq.params) {
        return None;
    }
    // out = Sqrt(s)
    let rooted = single_output_for_primitive(sqrt_eq, Primitive::Sqrt)?;
    if rooted != out || sqrt_eq.inputs.as_slice() != [Atom::Var(s)] {
        return None;
    }

    let Value::Tensor(a_t) = &args[0] else {
        return None;
    };
    let Value::Tensor(b_t) = &args[1] else {
        return None;
    };
    if a_t.dtype != DType::F64
        || b_t.dtype != DType::F64
        || a_t.shape.dims.len() != 2
        || b_t.shape.dims != a_t.shape.dims
    {
        return None;
    }
    let rows = a_t.shape.dims[0] as usize;
    let cols = a_t.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }

    let a_v = a_t.elements.as_f64_slice()?;
    let b_v = b_t.elements.as_f64_slice()?;
    if !a_v.iter().all(|value| value.is_finite()) || !b_v.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::euclidean_distance_2d(a_v, b_v, rows, cols);
    Some(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32],
            },
            output,
        )
        .map(Value::Tensor)
        .map(|value| vec![value])
        .map_err(EvalError::InvalidTensor)
        .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for row-wise mean squared error `Σ(a - b)² / n` along the last
/// axis: the exact 4-equation, TWO-input graph `Sub(a,b) -> Mul(diff,diff) -> ReduceSum(axis=1) ->
/// Div(cols)`. The decomposed path materializes `a-b` and `(a-b)^2` as full [rows,cols] tensors;
/// this streams each row once and writes one [rows] output. Finite dense rank-2 f64 only, else
/// falls through.
fn try_eval_top_level_mean_squared_error_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 2
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 4
        || args.len() != 2
    {
        return None;
    }

    let a = jaxpr.invars[0];
    let b = jaxpr.invars[1];
    let out = jaxpr.outvars[0];
    let [sub_eq, sq_eq, sum_eq, div_eq] = jaxpr.equations.as_slice() else {
        return None;
    };

    let diff = single_output_for_primitive(sub_eq, Primitive::Sub)?;
    if sub_eq.inputs.as_slice() != [Atom::Var(a), Atom::Var(b)] {
        return None;
    }
    let sq = single_output_for_primitive(sq_eq, Primitive::Mul)?;
    if sq_eq.inputs.as_slice() != [Atom::Var(diff), Atom::Var(diff)] {
        return None;
    }
    let sum = single_output_for_param_primitive(sum_eq, Primitive::ReduceSum)?;
    if sum_eq.inputs.as_slice() != [Atom::Var(sq)] || !axis1_reduce_params(&sum_eq.params) {
        return None;
    }
    let quotient = single_output_for_primitive(div_eq, Primitive::Div)?;
    let [Atom::Var(num), denom] = div_eq.inputs.as_slice() else {
        return None;
    };
    if quotient != out || *num != sum {
        return None;
    }

    let Value::Tensor(a_t) = &args[0] else {
        return None;
    };
    let Value::Tensor(b_t) = &args[1] else {
        return None;
    };
    if a_t.dtype != DType::F64
        || b_t.dtype != DType::F64
        || a_t.shape.dims.len() != 2
        || b_t.shape.dims != a_t.shape.dims
    {
        return None;
    }
    let rows = a_t.shape.dims[0] as usize;
    let cols = a_t.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }
    if f64_literal(denom)?.to_bits() != (cols as f64).to_bits() {
        return None;
    }

    let a_v = a_t.elements.as_f64_slice()?;
    let b_v = b_t.elements.as_f64_slice()?;
    if !a_v.iter().all(|value| value.is_finite()) || !b_v.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::mean_squared_error_2d(a_v, b_v, rows, cols);
    Some(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32],
            },
            output,
        )
        .map(Value::Tensor)
        .map(|value| vec![value])
        .map_err(EvalError::InvalidTensor)
        .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for row-wise mean absolute error `Σ|a - b| / n` along the last
/// axis (`sklearn.metrics.mean_absolute_error`): the exact 4-equation, TWO-input graph
/// `Sub(a,b) -> Abs -> ReduceSum(axis=1) -> Div(cols)`. The decomposed path materializes `a-b`
/// and `abs(a-b)` as full [rows,cols] tensors; this streams each row once and writes one [rows]
/// output. Finite dense rank-2 f64 only, else falls through.
fn try_eval_top_level_mean_absolute_error_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 2
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 4
        || args.len() != 2
    {
        return None;
    }

    let a = jaxpr.invars[0];
    let b = jaxpr.invars[1];
    let out = jaxpr.outvars[0];
    let [sub_eq, abs_eq, sum_eq, div_eq] = jaxpr.equations.as_slice() else {
        return None;
    };

    let diff = single_output_for_primitive(sub_eq, Primitive::Sub)?;
    if sub_eq.inputs.as_slice() != [Atom::Var(a), Atom::Var(b)] {
        return None;
    }
    let abs = single_output_for_primitive(abs_eq, Primitive::Abs)?;
    if abs_eq.inputs.as_slice() != [Atom::Var(diff)] {
        return None;
    }
    let summed = single_output_for_param_primitive(sum_eq, Primitive::ReduceSum)?;
    if sum_eq.inputs.as_slice() != [Atom::Var(abs)] || !axis1_reduce_params(&sum_eq.params) {
        return None;
    }
    let quotient = single_output_for_primitive(div_eq, Primitive::Div)?;
    let [Atom::Var(num), denom] = div_eq.inputs.as_slice() else {
        return None;
    };
    if quotient != out || *num != summed {
        return None;
    }

    let Value::Tensor(a_t) = &args[0] else {
        return None;
    };
    let Value::Tensor(b_t) = &args[1] else {
        return None;
    };
    if a_t.dtype != DType::F64
        || b_t.dtype != DType::F64
        || a_t.shape.dims.len() != 2
        || b_t.shape.dims != a_t.shape.dims
    {
        return None;
    }
    let rows = a_t.shape.dims[0] as usize;
    let cols = a_t.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }
    if f64_literal(denom)?.to_bits() != (cols as f64).to_bits() {
        return None;
    }

    let a_v = a_t.elements.as_f64_slice()?;
    let b_v = b_t.elements.as_f64_slice()?;
    if !a_v.iter().all(|value| value.is_finite()) || !b_v.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::mean_absolute_error_2d(a_v, b_v, rows, cols);
    Some(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32],
            },
            output,
        )
        .map(Value::Tensor)
        .map(|value| vec![value])
        .map_err(EvalError::InvalidTensor)
        .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the row-wise root mean squared error `sqrt(mean((a-b)²))`
/// (`sklearn.metrics.root_mean_squared_error` — the RMSE regression metric): the 5-equation, TWO-input
/// graph `Sub(a,b) → Mul(square) → ReduceSum(axis=1) → Div(·, n) → Sqrt`, two f64 [rows,cols] inputs →
/// one [rows] output (rank-reducing). Computed via the row-parallel
/// [`fj_lax::nn::root_mean_squared_error_2d`], which fuses the squared-error accumulation into one
/// index-order pass (bit-identical to `mean_squared_error_2d`) and applies `f64::sqrt` (= the `Sqrt`
/// primitive) to the per-row mean. DISTINCT from `mean_squared_error_2d` (identical 4-eq prefix but NO
/// trailing Sqrt) and from `euclidean_distance_2d` (`sqrt(Σ(a-b)²)` — Sqrt directly on the ReduceSum,
/// NO `Div(·, n)`). Falls through unless both inputs are dense finite rank-2 f64 of equal shape.
fn try_eval_top_level_root_mean_squared_error_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 2
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 5
        || args.len() != 2
    {
        return None;
    }

    let a = jaxpr.invars[0];
    let b = jaxpr.invars[1];
    let out = jaxpr.outvars[0];
    let [sub_eq, sq_eq, sum_eq, div_eq, sqrt_eq] = jaxpr.equations.as_slice() else {
        return None;
    };

    let diff = single_output_for_primitive(sub_eq, Primitive::Sub)?;
    if sub_eq.inputs.as_slice() != [Atom::Var(a), Atom::Var(b)] {
        return None;
    }
    let sq = single_output_for_primitive(sq_eq, Primitive::Mul)?;
    if sq_eq.inputs.as_slice() != [Atom::Var(diff), Atom::Var(diff)] {
        return None;
    }
    let sum = single_output_for_param_primitive(sum_eq, Primitive::ReduceSum)?;
    if sum_eq.inputs.as_slice() != [Atom::Var(sq)] || !axis1_reduce_params(&sum_eq.params) {
        return None;
    }
    let ms = single_output_for_primitive(div_eq, Primitive::Div)?;
    let [Atom::Var(num), denom] = div_eq.inputs.as_slice() else {
        return None;
    };
    if *num != sum {
        return None;
    }
    let root = single_output_for_primitive(sqrt_eq, Primitive::Sqrt)?;
    if root != out || sqrt_eq.inputs.as_slice() != [Atom::Var(ms)] {
        return None;
    }

    let Value::Tensor(a_t) = &args[0] else {
        return None;
    };
    let Value::Tensor(b_t) = &args[1] else {
        return None;
    };
    if a_t.dtype != DType::F64
        || b_t.dtype != DType::F64
        || a_t.shape.dims.len() != 2
        || b_t.shape.dims != a_t.shape.dims
    {
        return None;
    }
    let rows = a_t.shape.dims[0] as usize;
    let cols = a_t.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }
    if f64_literal(denom)?.to_bits() != (cols as f64).to_bits() {
        return None;
    }

    let a_v = a_t.elements.as_f64_slice()?;
    let b_v = b_t.elements.as_f64_slice()?;
    if !a_v.iter().all(|value| value.is_finite()) || !b_v.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::root_mean_squared_error_2d(a_v, b_v, rows, cols);
    Some(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32],
            },
            output,
        )
        .map(Value::Tensor)
        .map(|value| vec![value])
        .map_err(EvalError::InvalidTensor)
        .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the row-wise R² / coefficient of determination
/// `1 − Σ(a-b)² / Σ(a-mean(a))²` (`a`=y_true, `b`=y_pred) along the last axis (`sklearn.metrics.r2_score`
/// — the standard regression score): an 11-equation, TWO-input graph
/// `Sub(a,b)→Mul(sq)→ReduceSum(ss_res) | ReduceSum(a)→Div(n)→Bcast→Sub(centered)→Mul(sq)→ReduceSum(ss_tot)
///  | Div(ratio) | Sub(1, ratio)`, two f64 [rows,cols] inputs → one [rows] output (rank-reducing).
/// Computed via the row-parallel [`fj_lax::nn::r2_score_2d`], which fuses the residual sum-of-squares
/// (the MSE fusion) and the mean-centered total sum-of-squares (the variance fusion) into index-order
/// passes, divides (TRUE division), and subtracts from the scalar-broadcast `1.0`. Transcendental-free.
/// DISTINCT from `mean_squared_error_2d` (just the ss_res half) and from `var_2d` (just the ss_tot half,
/// divided by n). Falls through unless both inputs are dense finite rank-2 f64 of equal shape.
fn try_eval_top_level_r2_score_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 2
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 11
        || args.len() != 2
    {
        return None;
    }

    let a = jaxpr.invars[0];
    let b = jaxpr.invars[1];
    let out = jaxpr.outvars[0];
    let [
        resid_eq,
        resid_sq_eq,
        ss_res_eq,
        sum_eq,
        mean_eq,
        mean_bcast_eq,
        centered_eq,
        centered_sq_eq,
        ss_tot_eq,
        ratio_eq,
        final_eq,
    ] = jaxpr.equations.as_slice()
    else {
        return None;
    };

    // ss_res = ReduceSum((a - b)²)
    let resid = single_output_for_primitive(resid_eq, Primitive::Sub)?;
    if resid_eq.inputs.as_slice() != [Atom::Var(a), Atom::Var(b)] {
        return None;
    }
    let resid_sq = single_output_for_primitive(resid_sq_eq, Primitive::Mul)?;
    if resid_sq_eq.inputs.as_slice() != [Atom::Var(resid), Atom::Var(resid)] {
        return None;
    }
    let ss_res = single_output_for_param_primitive(ss_res_eq, Primitive::ReduceSum)?;
    if ss_res_eq.inputs.as_slice() != [Atom::Var(resid_sq)]
        || !axis1_reduce_params(&ss_res_eq.params)
    {
        return None;
    }
    // mean = ReduceSum(a) / n, broadcast to [rows,cols]
    let s = single_output_for_param_primitive(sum_eq, Primitive::ReduceSum)?;
    if sum_eq.inputs.as_slice() != [Atom::Var(a)] || !axis1_reduce_params(&sum_eq.params) {
        return None;
    }
    let mean = single_output_for_primitive(mean_eq, Primitive::Div)?;
    let [Atom::Var(mean_num), mean_div] = mean_eq.inputs.as_slice() else {
        return None;
    };
    if *mean_num != s {
        return None;
    }
    let mean_b = single_output_for_param_primitive(mean_bcast_eq, Primitive::BroadcastInDim)?;
    if mean_bcast_eq.inputs.as_slice() != [Atom::Var(mean)]
        || !rank2_axis0_broadcast_params(&mean_bcast_eq.params)
    {
        return None;
    }
    // ss_tot = ReduceSum((a - mean_b)²)
    let centered = single_output_for_primitive(centered_eq, Primitive::Sub)?;
    if centered_eq.inputs.as_slice() != [Atom::Var(a), Atom::Var(mean_b)] {
        return None;
    }
    let centered_sq = single_output_for_primitive(centered_sq_eq, Primitive::Mul)?;
    if centered_sq_eq.inputs.as_slice() != [Atom::Var(centered), Atom::Var(centered)] {
        return None;
    }
    let ss_tot = single_output_for_param_primitive(ss_tot_eq, Primitive::ReduceSum)?;
    if ss_tot_eq.inputs.as_slice() != [Atom::Var(centered_sq)]
        || !axis1_reduce_params(&ss_tot_eq.params)
    {
        return None;
    }
    // ratio = ss_res / ss_tot; out = 1 - ratio
    let ratio = single_output_for_primitive(ratio_eq, Primitive::Div)?;
    if ratio_eq.inputs.as_slice() != [Atom::Var(ss_res), Atom::Var(ss_tot)] {
        return None;
    }
    let result_var = single_output_for_primitive(final_eq, Primitive::Sub)?;
    let [one_lit, Atom::Var(ratio_operand)] = final_eq.inputs.as_slice() else {
        return None;
    };
    if result_var != out
        || *ratio_operand != ratio
        || f64_literal(one_lit)?.to_bits() != 1.0f64.to_bits()
    {
        return None;
    }

    let Value::Tensor(a_t) = &args[0] else {
        return None;
    };
    let Value::Tensor(b_t) = &args[1] else {
        return None;
    };
    if a_t.dtype != DType::F64
        || b_t.dtype != DType::F64
        || a_t.shape.dims.len() != 2
        || b_t.shape.dims != a_t.shape.dims
    {
        return None;
    }
    let rows = a_t.shape.dims[0] as usize;
    let cols = a_t.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }
    if f64_literal(mean_div)?.to_bits() != (cols as f64).to_bits() {
        return None;
    }
    let expected_shape = format!("{rows},{cols}");
    if mean_bcast_eq.params.get("shape").map(String::as_str) != Some(expected_shape.as_str()) {
        return None;
    }

    let a_v = a_t.elements.as_f64_slice()?;
    let b_v = b_t.elements.as_f64_slice()?;
    if !a_v.iter().all(|value| value.is_finite()) || !b_v.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::r2_score_2d(a_v, b_v, rows, cols);
    Some(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32],
            },
            output,
        )
        .map(Value::Tensor)
        .map(|value| vec![value])
        .map_err(EvalError::InvalidTensor)
        .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the row-wise Pearson correlation coefficient
/// `Σ((a-ma)(b-mb)) / (√Σ(a-ma)² · √Σ(b-mb)²)` (`ma=mean(a)`, `mb=mean(b)`) along the last axis —
/// the standard linear-association statistic, = cosine similarity of the CENTERED rows. An 18-equation,
/// TWO-input graph: `ReduceSum(a)→Div(n)→Bcast→Sub(ca)`, `ReduceSum(b)→Div(n)→Bcast→Sub(cb)`,
/// `Mul(ca,cb)→ReduceSum(cov)`, `Mul(ca,ca)→ReduceSum→Sqrt`, `Mul(cb,cb)→ReduceSum→Sqrt`,
/// `Mul(denom)→Div`. Two f64 [rows,cols] inputs, one [rows] output (rank-reducing). NO transcendentals
/// (`Sqrt`=`f64::sqrt`, IEEE-exact). The general fuser cannot fuse it (five reductions break the
/// elementwise fuser); the decomposed path materializes ~6 full [rows,cols] intermediates. Fused via
/// the row-parallel `pearson_correlation_2d`, whose index-order means + centered-product sums match
/// the graph bit-for-bit; finite dense rank-2 f64 only, else falls through.
fn try_eval_top_level_pearson_correlation_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 2
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 18
        || args.len() != 2
    {
        return None;
    }

    let a = jaxpr.invars[0];
    let b = jaxpr.invars[1];
    let out = jaxpr.outvars[0];
    let [
        sa_eq,
        ma_eq,
        ma_b_eq,
        ca_eq,
        sb_eq,
        mb_eq,
        mb_b_eq,
        cb_eq,
        cov_mul_eq,
        cov_eq,
        va_mul_eq,
        va_eq,
        na_eq,
        vb_mul_eq,
        vb_eq,
        nb_eq,
        denom_eq,
        div_eq,
    ] = jaxpr.equations.as_slice()
    else {
        return None;
    };

    // ma = ReduceSum(a, axis=1) / n
    let sa = single_output_for_param_primitive(sa_eq, Primitive::ReduceSum)?;
    if sa_eq.inputs.as_slice() != [Atom::Var(a)] || !axis1_reduce_params(&sa_eq.params) {
        return None;
    }
    let ma = single_output_for_primitive(ma_eq, Primitive::Div)?;
    let [Atom::Var(ma_num), ma_div] = ma_eq.inputs.as_slice() else {
        return None;
    };
    if *ma_num != sa {
        return None;
    }
    let ma_b = single_output_for_param_primitive(ma_b_eq, Primitive::BroadcastInDim)?;
    if ma_b_eq.inputs.as_slice() != [Atom::Var(ma)]
        || !rank2_axis0_broadcast_params(&ma_b_eq.params)
    {
        return None;
    }
    let ca = single_output_for_primitive(ca_eq, Primitive::Sub)?;
    if ca_eq.inputs.as_slice() != [Atom::Var(a), Atom::Var(ma_b)] {
        return None;
    }

    // mb = ReduceSum(b, axis=1) / n
    let sb = single_output_for_param_primitive(sb_eq, Primitive::ReduceSum)?;
    if sb_eq.inputs.as_slice() != [Atom::Var(b)] || !axis1_reduce_params(&sb_eq.params) {
        return None;
    }
    let mb = single_output_for_primitive(mb_eq, Primitive::Div)?;
    let [Atom::Var(mb_num), mb_div] = mb_eq.inputs.as_slice() else {
        return None;
    };
    if *mb_num != sb {
        return None;
    }
    let mb_b = single_output_for_param_primitive(mb_b_eq, Primitive::BroadcastInDim)?;
    if mb_b_eq.inputs.as_slice() != [Atom::Var(mb)]
        || !rank2_axis0_broadcast_params(&mb_b_eq.params)
    {
        return None;
    }
    let cb = single_output_for_primitive(cb_eq, Primitive::Sub)?;
    if cb_eq.inputs.as_slice() != [Atom::Var(b), Atom::Var(mb_b)] {
        return None;
    }

    // cov = ReduceSum(ca * cb) (Mul commutative → either operand order)
    let cov_prod = single_output_for_primitive(cov_mul_eq, Primitive::Mul)?;
    let cov_mul_ok = matches!(
        cov_mul_eq.inputs.as_slice(),
        [Atom::Var(x), Atom::Var(y)] if (*x == ca && *y == cb) || (*x == cb && *y == ca)
    );
    if !cov_mul_ok {
        return None;
    }
    let cov = single_output_for_param_primitive(cov_eq, Primitive::ReduceSum)?;
    if cov_eq.inputs.as_slice() != [Atom::Var(cov_prod)] || !axis1_reduce_params(&cov_eq.params) {
        return None;
    }

    // ‖ca‖ = Sqrt(ReduceSum(ca * ca))
    let va_prod = single_output_for_primitive(va_mul_eq, Primitive::Mul)?;
    if va_mul_eq.inputs.as_slice() != [Atom::Var(ca), Atom::Var(ca)] {
        return None;
    }
    let va = single_output_for_param_primitive(va_eq, Primitive::ReduceSum)?;
    if va_eq.inputs.as_slice() != [Atom::Var(va_prod)] || !axis1_reduce_params(&va_eq.params) {
        return None;
    }
    let na = single_output_for_primitive(na_eq, Primitive::Sqrt)?;
    if na_eq.inputs.as_slice() != [Atom::Var(va)] {
        return None;
    }

    // ‖cb‖ = Sqrt(ReduceSum(cb * cb))
    let vb_prod = single_output_for_primitive(vb_mul_eq, Primitive::Mul)?;
    if vb_mul_eq.inputs.as_slice() != [Atom::Var(cb), Atom::Var(cb)] {
        return None;
    }
    let vb = single_output_for_param_primitive(vb_eq, Primitive::ReduceSum)?;
    if vb_eq.inputs.as_slice() != [Atom::Var(vb_prod)] || !axis1_reduce_params(&vb_eq.params) {
        return None;
    }
    let nb = single_output_for_primitive(nb_eq, Primitive::Sqrt)?;
    if nb_eq.inputs.as_slice() != [Atom::Var(vb)] {
        return None;
    }

    // denom = ‖ca‖ · ‖cb‖ (Mul commutative → either order)
    let denom = single_output_for_primitive(denom_eq, Primitive::Mul)?;
    let denom_ok = matches!(
        denom_eq.inputs.as_slice(),
        [Atom::Var(x), Atom::Var(y)] if (*x == na && *y == nb) || (*x == nb && *y == na)
    );
    if !denom_ok {
        return None;
    }
    // out = cov / denom (Div non-commutative → numerator = cov)
    let quotient = single_output_for_primitive(div_eq, Primitive::Div)?;
    if quotient != out || div_eq.inputs.as_slice() != [Atom::Var(cov), Atom::Var(denom)] {
        return None;
    }

    let Value::Tensor(a_t) = &args[0] else {
        return None;
    };
    let Value::Tensor(b_t) = &args[1] else {
        return None;
    };
    if a_t.dtype != DType::F64
        || b_t.dtype != DType::F64
        || a_t.shape.dims.len() != 2
        || b_t.shape.dims != a_t.shape.dims
    {
        return None;
    }
    let rows = a_t.shape.dims[0] as usize;
    let cols = a_t.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }
    // Both means divide by the row length `n = cols` (as f64), matching `pearson_correlation_row`.
    let n_bits = (cols as f64).to_bits();
    if f64_literal(ma_div)?.to_bits() != n_bits || f64_literal(mb_div)?.to_bits() != n_bits {
        return None;
    }
    let expected_shape = format!("{rows},{cols}");
    if ma_b_eq.params.get("shape").map(String::as_str) != Some(expected_shape.as_str())
        || mb_b_eq.params.get("shape").map(String::as_str) != Some(expected_shape.as_str())
    {
        return None;
    }

    let a_v = a_t.elements.as_f64_slice()?;
    let b_v = b_t.elements.as_f64_slice()?;
    if !a_v.iter().all(|value| value.is_finite()) || !b_v.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::pearson_correlation_2d(a_v, b_v, rows, cols);
    Some(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32],
            },
            output,
        )
        .map(Value::Tensor)
        .map(|value| vec![value])
        .map_err(EvalError::InvalidTensor)
        .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the row-wise forward KL divergence `Σ p·log(p/q)` along the
/// last axis (the information-theoretic divergence at the heart of distillation / VAE / RLHF): the
/// 4-equation, TWO-input graph `Div(p, q) → Log → Mul(p, ·) → ReduceSum(axis=1)`. Two f64 [rows,cols]
/// inputs, one [rows] output (rank-reducing). The general fuser cannot fuse it (`Log` ∉ `cheap_op`
/// AND the reduction breaks the elementwise fuser), so the decomposed path materializes three full
/// [rows,cols] intermediates (`p/q`, `log(p/q)`, `p·log(p/q)`). Fused via the row-parallel
/// `kl_divergence_2d`, whose index-order `Σ p·ln(p/q)` matches the graph bit-for-bit (`Log` = the same
/// `.ln()` the `Log` primitive dispatches); finite dense rank-2 f64 only, else falls through.
fn try_eval_top_level_kl_divergence_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 2
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 4
        || args.len() != 2
    {
        return None;
    }

    let p = jaxpr.invars[0];
    let q = jaxpr.invars[1];
    let out = jaxpr.outvars[0];
    let [div_eq, log_eq, mul_eq, sum_eq] = jaxpr.equations.as_slice() else {
        return None;
    };

    // ratio = p / q (Div non-commutative — numerator = p)
    let ratio = single_output_for_primitive(div_eq, Primitive::Div)?;
    if div_eq.inputs.as_slice() != [Atom::Var(p), Atom::Var(q)] {
        return None;
    }
    // log_ratio = Log(ratio)
    let log_ratio = single_output_for_primitive(log_eq, Primitive::Log)?;
    if log_eq.inputs.as_slice() != [Atom::Var(ratio)] {
        return None;
    }
    // weighted = p * log_ratio (Mul commutative → either operand order)
    let weighted = single_output_for_primitive(mul_eq, Primitive::Mul)?;
    let mul_ok = matches!(
        mul_eq.inputs.as_slice(),
        [Atom::Var(x), Atom::Var(y)]
            if (*x == p && *y == log_ratio) || (*x == log_ratio && *y == p)
    );
    if !mul_ok {
        return None;
    }
    // out = ReduceSum(weighted, axis=1)
    let summed = single_output_for_param_primitive(sum_eq, Primitive::ReduceSum)?;
    if summed != out
        || sum_eq.inputs.as_slice() != [Atom::Var(weighted)]
        || !axis1_reduce_params(&sum_eq.params)
    {
        return None;
    }

    let Value::Tensor(p_t) = &args[0] else {
        return None;
    };
    let Value::Tensor(q_t) = &args[1] else {
        return None;
    };
    if p_t.dtype != DType::F64
        || q_t.dtype != DType::F64
        || p_t.shape.dims.len() != 2
        || q_t.shape.dims != p_t.shape.dims
    {
        return None;
    }
    let rows = p_t.shape.dims[0] as usize;
    let cols = p_t.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }

    let p_v = p_t.elements.as_f64_slice()?;
    let q_v = q_t.elements.as_f64_slice()?;
    if !p_v.iter().all(|value| value.is_finite()) || !q_v.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::kl_divergence_2d(p_v, q_v, rows, cols);
    Some(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32],
            },
            output,
        )
        .map(Value::Tensor)
        .map(|value| vec![value])
        .map_err(EvalError::InvalidTensor)
        .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the row-wise Shannon entropy `-Σ p·log(p)` along the last axis
/// (the information-theoretic entropy / negentropy regularizer): the 4-equation, ONE-input graph
/// `Log(p) → Mul(p, ·) → ReduceSum(axis=1) → Neg`. One f64 [rows,cols] input, one [rows] output
/// (rank-reducing). The general fuser cannot fuse it (`Log` ∉ `cheap_op` AND the reduction breaks the
/// elementwise fuser), so the decomposed path materializes two full [rows,cols] intermediates
/// (`log(p)`, `p·log(p)`). Fused via the row-parallel `entropy_2d`, whose index-order `-Σ p·ln(p)`
/// matches the graph bit-for-bit (`Log` = the same `.ln()` the `Log` primitive dispatches); finite
/// dense rank-2 f64 only, else falls through.
fn try_eval_top_level_entropy_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 4
        || args.len() != 1
    {
        return None;
    }

    let p = jaxpr.invars[0];
    let out = jaxpr.outvars[0];
    let [log_eq, mul_eq, sum_eq, neg_eq] = jaxpr.equations.as_slice() else {
        return None;
    };

    // log_p = Log(p)
    let log_p = single_output_for_primitive(log_eq, Primitive::Log)?;
    if log_eq.inputs.as_slice() != [Atom::Var(p)] {
        return None;
    }
    // weighted = p * log_p (Mul commutative → either operand order)
    let weighted = single_output_for_primitive(mul_eq, Primitive::Mul)?;
    let mul_ok = matches!(
        mul_eq.inputs.as_slice(),
        [Atom::Var(x), Atom::Var(y)] if (*x == p && *y == log_p) || (*x == log_p && *y == p)
    );
    if !mul_ok {
        return None;
    }
    // summed = ReduceSum(weighted, axis=1)
    let summed = single_output_for_param_primitive(sum_eq, Primitive::ReduceSum)?;
    if sum_eq.inputs.as_slice() != [Atom::Var(weighted)] || !axis1_reduce_params(&sum_eq.params) {
        return None;
    }
    // out = Neg(summed)
    let negated = single_output_for_primitive(neg_eq, Primitive::Neg)?;
    if negated != out || neg_eq.inputs.as_slice() != [Atom::Var(summed)] {
        return None;
    }

    let Value::Tensor(p_t) = &args[0] else {
        return None;
    };
    if p_t.dtype != DType::F64 || p_t.shape.dims.len() != 2 {
        return None;
    }
    let rows = p_t.shape.dims[0] as usize;
    let cols = p_t.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }

    let p_v = p_t.elements.as_f64_slice()?;
    if !p_v.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::entropy_2d(p_v, rows, cols);
    Some(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32],
            },
            output,
        )
        .map(Value::Tensor)
        .map(|value| vec![value])
        .map_err(EvalError::InvalidTensor)
        .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for row-wise Manhattan (L1) distance `Σ|a-b|` along the last axis:
/// the 3-equation, TWO-input graph `Sub(a,b) -> Abs -> ReduceSum(axis=1)`. The decomposed path
/// materializes `a-b` and `abs(a-b)` as full [rows,cols] tensors; this streams each row once and
/// writes one [rows] output. Finite dense rank-2 f64 only, else falls through.
fn try_eval_top_level_manhattan_distance_2d_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 2
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 3
        || args.len() != 2
    {
        return None;
    }

    let a = jaxpr.invars[0];
    let b = jaxpr.invars[1];
    let out = jaxpr.outvars[0];
    let [sub_eq, abs_eq, sum_eq] = jaxpr.equations.as_slice() else {
        return None;
    };

    let diff = single_output_for_primitive(sub_eq, Primitive::Sub)?;
    if sub_eq.inputs.as_slice() != [Atom::Var(a), Atom::Var(b)] {
        return None;
    }
    let abs = single_output_for_primitive(abs_eq, Primitive::Abs)?;
    if abs_eq.inputs.as_slice() != [Atom::Var(diff)] {
        return None;
    }
    let summed = single_output_for_param_primitive(sum_eq, Primitive::ReduceSum)?;
    if summed != out
        || sum_eq.inputs.as_slice() != [Atom::Var(abs)]
        || !axis1_reduce_params(&sum_eq.params)
    {
        return None;
    }

    let Value::Tensor(a_t) = &args[0] else {
        return None;
    };
    let Value::Tensor(b_t) = &args[1] else {
        return None;
    };
    if a_t.dtype != DType::F64
        || b_t.dtype != DType::F64
        || a_t.shape.dims.len() != 2
        || b_t.shape.dims != a_t.shape.dims
    {
        return None;
    }
    let rows = a_t.shape.dims[0] as usize;
    let cols = a_t.shape.dims[1] as usize;
    if rows == 0 || cols == 0 {
        return None;
    }

    let a_v = a_t.elements.as_f64_slice()?;
    let b_v = b_t.elements.as_f64_slice()?;
    if !a_v.iter().all(|value| value.is_finite()) || !b_v.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::manhattan_distance_2d(a_v, b_v, rows, cols);
    Some(
        TensorValue::new_f64_values(
            Shape {
                dims: vec![rows as u32],
            },
            output,
        )
        .map(Value::Tensor)
        .map(|value| vec![value])
        .map_err(EvalError::InvalidTensor)
        .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the exact GELU activation graph
/// `0.5·x·(1 + erf(x/√2))` (`jax.nn.gelu(approximate=False)`): the 5-equation elementwise chain
/// `Div(x, √2) → Erf → Add(1, ·) → Mul(x, 0.5) → Mul`. Pure elementwise (no broadcast/reduce, so
/// shape-agnostic) — the general fuser CANNOT fuse it because `Erf` is not a `CheapOp`, leaving
/// intermediate materializations that the fused `gelu_erf` pass collapses. Finite dense f64 only;
/// anything else falls through to the generic interpreter. Bit-identical: `gelu_erf` calls the
/// same `erf_approx` the `Erf` primitive uses and matches the graph's `(x·0.5)·(1 + erf)` grouping.
fn try_eval_top_level_gelu_erf_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 5
        || args.len() != 1
    {
        return None;
    }

    let x = jaxpr.invars[0];
    let out = jaxpr.outvars[0];
    let [div_eq, erf_eq, add_eq, half_eq, mul_eq] = jaxpr.equations.as_slice() else {
        return None;
    };

    // Div(x, √2)
    let arg = single_output_for_primitive(div_eq, Primitive::Div)?;
    let [Atom::Var(div_lhs), div_rhs] = div_eq.inputs.as_slice() else {
        return None;
    };
    if *div_lhs != x || f64_literal(div_rhs)?.to_bits() != 2.0_f64.sqrt().to_bits() {
        return None;
    }

    // Erf(arg)
    let e = single_output_for_primitive(erf_eq, Primitive::Erf)?;
    if erf_eq.inputs.as_slice() != [Atom::Var(arg)] {
        return None;
    }

    // Add(1, e) (either operand order)
    let one_plus = single_output_for_primitive(add_eq, Primitive::Add)?;
    let add_const = match add_eq.inputs.as_slice() {
        [Atom::Var(v), atom] if *v == e => f64_literal(atom),
        [atom, Atom::Var(v)] if *v == e => f64_literal(atom),
        _ => None,
    };
    if add_const?.to_bits() != 1.0_f64.to_bits() {
        return None;
    }

    // Mul(x, 0.5) (either operand order)
    let half_x = single_output_for_primitive(half_eq, Primitive::Mul)?;
    let half_const = match half_eq.inputs.as_slice() {
        [Atom::Var(v), atom] if *v == x => f64_literal(atom),
        [atom, Atom::Var(v)] if *v == x => f64_literal(atom),
        _ => None,
    };
    if half_const?.to_bits() != 0.5_f64.to_bits() {
        return None;
    }

    // Mul(half_x, one_plus) → out (either operand order; Mul is commutative → bit-identical)
    let mul = single_output_for_primitive(mul_eq, Primitive::Mul)?;
    if mul != out {
        return None;
    }
    let final_ok = match mul_eq.inputs.as_slice() {
        [Atom::Var(a), Atom::Var(b)] => {
            (*a == half_x && *b == one_plus) || (*a == one_plus && *b == half_x)
        }
        _ => false,
    };
    if !final_ok {
        return None;
    }

    let Value::Tensor(input) = &args[0] else {
        return None;
    };
    if input.dtype != DType::F64 {
        return None;
    }
    let values = input.elements.as_f64_slice()?;
    if !values.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::gelu_erf(values);
    Some(
        TensorValue::new_f64_values(input.shape.clone(), output)
            .map(Value::Tensor)
            .map(|value| vec![value])
            .map_err(EvalError::InvalidTensor)
            .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the log-sigmoid activation graph `-log(1 + exp(-x))`
/// (`jax.nn.log_sigmoid`, direct form): the 5-equation elementwise chain
/// `Neg(x) → Exp → Add(1, ·) → Log → Neg`. Pure elementwise (shape-agnostic); the general fuser
/// cannot fuse it because `Exp`/`Log` are not `CheapOp`s. Finite dense f64 only, else falls
/// through. Bit-identical: `log_sigmoid_direct` uses only `exp`/`ln` (both bit-identical to the
/// `Exp`/`Log` primitives, per the softmax/log-softmax superinstructions) in the graph's grouping.
fn try_eval_top_level_log_sigmoid_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 5
        || args.len() != 1
    {
        return None;
    }

    let x = jaxpr.invars[0];
    let out = jaxpr.outvars[0];
    let [neg_eq, exp_eq, add_eq, log_eq, neg2_eq] = jaxpr.equations.as_slice() else {
        return None;
    };

    let nx = single_output_for_primitive(neg_eq, Primitive::Neg)?;
    if neg_eq.inputs.as_slice() != [Atom::Var(x)] {
        return None;
    }
    let e = single_output_for_primitive(exp_eq, Primitive::Exp)?;
    if exp_eq.inputs.as_slice() != [Atom::Var(nx)] {
        return None;
    }
    let one_plus = single_output_for_primitive(add_eq, Primitive::Add)?;
    let add_const = match add_eq.inputs.as_slice() {
        [Atom::Var(v), atom] if *v == e => f64_literal(atom),
        [atom, Atom::Var(v)] if *v == e => f64_literal(atom),
        _ => None,
    };
    if add_const?.to_bits() != 1.0_f64.to_bits() {
        return None;
    }
    let sp = single_output_for_primitive(log_eq, Primitive::Log)?;
    if log_eq.inputs.as_slice() != [Atom::Var(one_plus)] {
        return None;
    }
    let neg2 = single_output_for_primitive(neg2_eq, Primitive::Neg)?;
    if neg2 != out || neg2_eq.inputs.as_slice() != [Atom::Var(sp)] {
        return None;
    }

    let Value::Tensor(input) = &args[0] else {
        return None;
    };
    if input.dtype != DType::F64 {
        return None;
    }
    let values = input.elements.as_f64_slice()?;
    if !values.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::log_sigmoid_direct(values);
    Some(
        TensorValue::new_f64_values(input.shape.clone(), output)
            .map(Value::Tensor)
            .map(|value| vec![value])
            .map_err(EvalError::InvalidTensor)
            .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the softplus activation graph `log(1 + exp(x))`
/// (`jax.nn.softplus`, direct form): the 3-equation elementwise chain `Exp → Add(1, ·) → Log`.
/// Pure elementwise (shape-agnostic); the general fuser cannot fuse it because `Exp`/`Log` are not
/// `CheapOp`s. Finite dense f64 only, else falls through. Bit-identical: `softplus_direct` uses only
/// `exp`/`ln` (both bit-identical to the `Exp`/`Log` primitives) in the graph grouping
/// `(1.0 + v.exp()).ln()`.
fn try_eval_top_level_softplus_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 3
        || args.len() != 1
    {
        return None;
    }

    let x = jaxpr.invars[0];
    let out = jaxpr.outvars[0];
    let [exp_eq, add_eq, log_eq] = jaxpr.equations.as_slice() else {
        return None;
    };

    let e = single_output_for_primitive(exp_eq, Primitive::Exp)?;
    if exp_eq.inputs.as_slice() != [Atom::Var(x)] {
        return None;
    }
    let one_plus = single_output_for_primitive(add_eq, Primitive::Add)?;
    let add_const = match add_eq.inputs.as_slice() {
        [Atom::Var(v), atom] if *v == e => f64_literal(atom),
        [atom, Atom::Var(v)] if *v == e => f64_literal(atom),
        _ => None,
    };
    if add_const?.to_bits() != 1.0_f64.to_bits() {
        return None;
    }
    let logged = single_output_for_primitive(log_eq, Primitive::Log)?;
    if logged != out || log_eq.inputs.as_slice() != [Atom::Var(one_plus)] {
        return None;
    }

    let Value::Tensor(input) = &args[0] else {
        return None;
    };
    if input.dtype != DType::F64 {
        return None;
    }
    let values = input.elements.as_f64_slice()?;
    if !values.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::softplus_direct(values);
    Some(
        TensorValue::new_f64_values(input.shape.clone(), output)
            .map(Value::Tensor)
            .map(|value| vec![value])
            .map_err(EvalError::InvalidTensor)
            .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the SiLU / swish activation graph `x * sigmoid(x)` =
/// `x / (1 + exp(-x))` (`jax.nn.silu` / `jax.nn.swish`): the 4-equation elementwise chain
/// `Neg(x) → Exp → Add(1, ·) → Div(x, ·)`. Pure elementwise (shape-agnostic); the general fuser
/// cannot fuse it because `Exp` is not a `CheapOp`. Finite dense f64 only, else falls through.
/// Bit-identical: `fj_lax::nn::silu` computes `v / (1.0 + (-v).exp())`, matching the graph's exact
/// `Div(x, Add(1, Exp(Neg(x))))` grouping (`Div` order enforced — non-commutative; `Add` order-
/// agnostic; `exp` bit-identical to the `Exp` primitive per the softmax superinstructions).
fn try_eval_top_level_silu_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 4
        || args.len() != 1
    {
        return None;
    }

    let x = jaxpr.invars[0];
    let out = jaxpr.outvars[0];
    let [neg_eq, exp_eq, add_eq, div_eq] = jaxpr.equations.as_slice() else {
        return None;
    };

    let nx = single_output_for_primitive(neg_eq, Primitive::Neg)?;
    if neg_eq.inputs.as_slice() != [Atom::Var(x)] {
        return None;
    }
    let e = single_output_for_primitive(exp_eq, Primitive::Exp)?;
    if exp_eq.inputs.as_slice() != [Atom::Var(nx)] {
        return None;
    }
    let one_plus = single_output_for_primitive(add_eq, Primitive::Add)?;
    let add_const = match add_eq.inputs.as_slice() {
        [Atom::Var(v), atom] if *v == e => f64_literal(atom),
        [atom, Atom::Var(v)] if *v == e => f64_literal(atom),
        _ => None,
    };
    if add_const?.to_bits() != 1.0_f64.to_bits() {
        return None;
    }
    // `Div` is non-commutative — the numerator must be the input `x`, denominator `1 + exp(-x)`.
    let quotient = single_output_for_primitive(div_eq, Primitive::Div)?;
    if quotient != out || div_eq.inputs.as_slice() != [Atom::Var(x), Atom::Var(one_plus)] {
        return None;
    }

    let Value::Tensor(input) = &args[0] else {
        return None;
    };
    if input.dtype != DType::F64 {
        return None;
    }
    let values = input.elements.as_f64_slice()?;
    if !values.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::silu(values);
    Some(
        TensorValue::new_f64_values(input.shape.clone(), output)
            .map(Value::Tensor)
            .map(|value| vec![value])
            .map_err(EvalError::InvalidTensor)
            .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the SwiGLU gated-MLP unit `silu(a) * b` =
/// `(a / (1 + exp(-a))) * b` (the core gated FFN of LLaMA/PaLM/Mistral): the 5-equation,
/// TWO-input elementwise graph `Neg(a) → Exp → Add(1, ·) → Div(a, ·) → Mul(·, b)`. Pure
/// elementwise over two equal-shape f64 inputs; the general fuser cannot fuse it because `Exp` is
/// not a `CheapOp`. Finite dense f64 only, else falls through. Bit-identical: `fj_lax::nn::swiglu`
/// computes `(av / (1.0 + (-av).exp())) * bv`, reproducing `silu`'s exact `a/(1+exp(-a))` form
/// (`Div` order enforced — non-commutative) then the per-element `Mul` by `b` (`Mul` commutative →
/// operand order agnostic; `Add(1, e)` likewise).
fn try_eval_top_level_swiglu_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 2
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 5
        || args.len() != 2
    {
        return None;
    }

    let a = jaxpr.invars[0];
    let b = jaxpr.invars[1];
    let out = jaxpr.outvars[0];
    let [neg_eq, exp_eq, add_eq, div_eq, mul_eq] = jaxpr.equations.as_slice() else {
        return None;
    };

    let na = single_output_for_primitive(neg_eq, Primitive::Neg)?;
    if neg_eq.inputs.as_slice() != [Atom::Var(a)] {
        return None;
    }
    let e = single_output_for_primitive(exp_eq, Primitive::Exp)?;
    if exp_eq.inputs.as_slice() != [Atom::Var(na)] {
        return None;
    }
    let one_plus = single_output_for_primitive(add_eq, Primitive::Add)?;
    let add_const = match add_eq.inputs.as_slice() {
        [Atom::Var(v), atom] if *v == e => f64_literal(atom),
        [atom, Atom::Var(v)] if *v == e => f64_literal(atom),
        _ => None,
    };
    if add_const?.to_bits() != 1.0_f64.to_bits() {
        return None;
    }
    // `Div` is non-commutative — numerator must be the gate input `a`, denominator `1 + exp(-a)`.
    let silu_a = single_output_for_primitive(div_eq, Primitive::Div)?;
    if div_eq.inputs.as_slice() != [Atom::Var(a), Atom::Var(one_plus)] {
        return None;
    }
    // `Mul` is commutative — gate `silu_a` times the value input `b`, in either operand order.
    let product = single_output_for_primitive(mul_eq, Primitive::Mul)?;
    let mul_ok = matches!(
        mul_eq.inputs.as_slice(),
        [Atom::Var(x), Atom::Var(y)]
            if (*x == silu_a && *y == b) || (*x == b && *y == silu_a)
    );
    if product != out || !mul_ok {
        return None;
    }

    let Value::Tensor(a_t) = &args[0] else {
        return None;
    };
    let Value::Tensor(b_t) = &args[1] else {
        return None;
    };
    if a_t.dtype != DType::F64 || b_t.dtype != DType::F64 || a_t.shape.dims != b_t.shape.dims {
        return None;
    }
    let a_v = a_t.elements.as_f64_slice()?;
    let b_v = b_t.elements.as_f64_slice()?;
    if !a_v.iter().all(|value| value.is_finite()) || !b_v.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::swiglu(a_v, b_v);
    Some(
        TensorValue::new_f64_values(a_t.shape.clone(), output)
            .map(Value::Tensor)
            .map(|value| vec![value])
            .map_err(EvalError::InvalidTensor)
            .map_err(InterpreterError::Primitive),
    )
}

/// Interpreter superinstruction for the GEGLU gated-MLP unit `gelu(a) * b` =
/// `(0.5·a·(1 + erf(a/√2))) · b` (the gated FFN of T5-v1.1 / GLM / PaLM variants): the 6-equation,
/// TWO-input graph `Div(a, √2) → Erf → Add(1, ·) → Mul(a, 0.5) → Mul → Mul(·, b)` — the exact
/// gelu-erf subgraph on `a` followed by a per-element `Mul` by `b`. Pure elementwise over two
/// equal-shape f64 inputs; the general fuser cannot fuse it because `Erf` ∉ `cheap_op`. Finite dense
/// f64 only, else falls through. Bit-identical: `fj_lax::nn::geglu` reuses `gelu_erf`'s exact
/// `(a·0.5)·(1 + erf_approx(a/√2))` grouping (its `erf_approx` is bit-identical to the SIMD
/// `erf_f64x8` that the `Erf` primitive dispatches) then the `Mul` by `b` (commutative → order-
/// agnostic; `Add(1,·)`/`Mul(a,0.5)` likewise; `Div` numerator enforced = `a`).
fn try_eval_top_level_geglu_f64(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || !jaxpr.effects.is_empty()
        || jaxpr.invars.len() != 2
        || jaxpr.outvars.len() != 1
        || jaxpr.equations.len() != 6
        || args.len() != 2
    {
        return None;
    }

    let a = jaxpr.invars[0];
    let b = jaxpr.invars[1];
    let out = jaxpr.outvars[0];
    let [div_eq, erf_eq, add_eq, half_eq, gelu_mul_eq, gate_mul_eq] = jaxpr.equations.as_slice()
    else {
        return None;
    };

    // Div(a, √2)
    let arg = single_output_for_primitive(div_eq, Primitive::Div)?;
    let [Atom::Var(div_lhs), div_rhs] = div_eq.inputs.as_slice() else {
        return None;
    };
    if *div_lhs != a || f64_literal(div_rhs)?.to_bits() != 2.0_f64.sqrt().to_bits() {
        return None;
    }
    // Erf(arg)
    let e = single_output_for_primitive(erf_eq, Primitive::Erf)?;
    if erf_eq.inputs.as_slice() != [Atom::Var(arg)] {
        return None;
    }
    // Add(1, e) (either order)
    let one_plus = single_output_for_primitive(add_eq, Primitive::Add)?;
    let add_const = match add_eq.inputs.as_slice() {
        [Atom::Var(v), atom] if *v == e => f64_literal(atom),
        [atom, Atom::Var(v)] if *v == e => f64_literal(atom),
        _ => None,
    };
    if add_const?.to_bits() != 1.0_f64.to_bits() {
        return None;
    }
    // Mul(a, 0.5) (either order)
    let half_a = single_output_for_primitive(half_eq, Primitive::Mul)?;
    let half_const = match half_eq.inputs.as_slice() {
        [Atom::Var(v), atom] if *v == a => f64_literal(atom),
        [atom, Atom::Var(v)] if *v == a => f64_literal(atom),
        _ => None,
    };
    if half_const?.to_bits() != 0.5_f64.to_bits() {
        return None;
    }
    // Mul(half_a, one_plus) → gelu_a (either order; commutative → bit-identical)
    let gelu_a = single_output_for_primitive(gelu_mul_eq, Primitive::Mul)?;
    let gelu_ok = matches!(
        gelu_mul_eq.inputs.as_slice(),
        [Atom::Var(x), Atom::Var(y)]
            if (*x == half_a && *y == one_plus) || (*x == one_plus && *y == half_a)
    );
    if !gelu_ok {
        return None;
    }
    // Mul(gelu_a, b) → out (either order)
    let product = single_output_for_primitive(gate_mul_eq, Primitive::Mul)?;
    let gate_ok = matches!(
        gate_mul_eq.inputs.as_slice(),
        [Atom::Var(x), Atom::Var(y)]
            if (*x == gelu_a && *y == b) || (*x == b && *y == gelu_a)
    );
    if product != out || !gate_ok {
        return None;
    }

    let Value::Tensor(a_t) = &args[0] else {
        return None;
    };
    let Value::Tensor(b_t) = &args[1] else {
        return None;
    };
    if a_t.dtype != DType::F64 || b_t.dtype != DType::F64 || a_t.shape.dims != b_t.shape.dims {
        return None;
    }
    let a_v = a_t.elements.as_f64_slice()?;
    let b_v = b_t.elements.as_f64_slice()?;
    if !a_v.iter().all(|value| value.is_finite()) || !b_v.iter().all(|value| value.is_finite()) {
        return None;
    }

    let output = fj_lax::nn::geglu(a_v, b_v);
    Some(
        TensorValue::new_f64_values(a_t.shape.clone(), output)
            .map(Value::Tensor)
            .map(|value| vec![value])
            .map_err(EvalError::InvalidTensor)
            .map_err(InterpreterError::Primitive),
    )
}

fn axis1_reduce_params(params: &std::collections::BTreeMap<String, String>) -> bool {
    params.len() == 1 && params.get("axes").map(String::as_str) == Some("1")
}

fn rank2_axis0_broadcast_params(params: &std::collections::BTreeMap<String, String>) -> bool {
    params.len() == 2 && params.get("broadcast_dimensions").map(String::as_str) == Some("0")
}

fn single_output_for_param_primitive(equation: &Equation, primitive: Primitive) -> Option<VarId> {
    if equation.primitive == primitive
        && equation.sub_jaxprs.is_empty()
        && equation.effects.is_empty()
        && equation.outputs.len() == 1
    {
        Some(equation.outputs[0])
    } else {
        None
    }
}

fn single_output_for_primitive(equation: &Equation, primitive: Primitive) -> Option<VarId> {
    if equation.primitive == primitive
        && equation.params.is_empty()
        && equation.sub_jaxprs.is_empty()
        && equation.effects.is_empty()
        && equation.outputs.len() == 1
    {
        Some(equation.outputs[0])
    } else {
        None
    }
}

fn f64_literal(atom: &Atom) -> Option<f64> {
    match atom {
        Atom::Lit(Literal::F64Bits(bits)) => Some(f64::from_bits(*bits)),
        _ => None,
    }
}

fn scalar_half_literal_bits(atom: &Atom) -> Option<(DType, u16)> {
    match atom {
        Atom::Lit(Literal::BF16Bits(bits)) => Some((DType::BF16, *bits)),
        Atom::Lit(Literal::F16Bits(bits)) => Some((DType::F16, *bits)),
        Atom::Var(_) | Atom::Lit(_) => None,
    }
}

fn scalar_half_args_bits(args: &[Value], dtype: DType) -> Option<[u16; 2]> {
    match (dtype, args) {
        (
            DType::BF16,
            [
                Value::Scalar(Literal::BF16Bits(x)),
                Value::Scalar(Literal::BF16Bits(y)),
            ],
        )
        | (
            DType::F16,
            [
                Value::Scalar(Literal::F16Bits(x)),
                Value::Scalar(Literal::F16Bits(y)),
            ],
        ) => Some([*x, *y]),
        (_, _) => None,
    }
}

fn try_eval_scalar_i64_add_chain(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Option<Result<Vec<Value>, InterpreterError>> {
    if !jaxpr.constvars.is_empty()
        || !const_values.is_empty()
        || jaxpr.invars.len() != 1
        || jaxpr.outvars.len() != 1
        || !jaxpr.effects.is_empty()
        || jaxpr.equations.is_empty()
    {
        return None;
    }

    let mut current_var = jaxpr.invars[0];
    let mut accumulator = match args.first()? {
        Value::Scalar(Literal::I64(value)) => *value,
        Value::Scalar(_) | Value::Tensor(_) => return None,
    };

    for equation in &jaxpr.equations {
        if equation.primitive != Primitive::Add
            || !equation.params.is_empty()
            || !equation.sub_jaxprs.is_empty()
            || !equation.effects.is_empty()
            || equation.outputs.len() != 1
        {
            return None;
        }

        match equation.inputs.as_slice() {
            [Atom::Var(var), Atom::Lit(Literal::I64(rhs))] if *var == current_var => {
                accumulator = accumulator.wrapping_add(*rhs);
                current_var = equation.outputs[0];
            }
            _ => return None,
        }
    }

    if jaxpr.outvars[0] != current_var {
        return None;
    }

    Some(Ok(vec![Value::scalar_i64(accumulator)]))
}

/// Flat slot-array interpreter environment (hot path). `slots` must be
/// `max_defined_var_id + 1`. Each variable lookup is an `O(1)` bounds-checked
/// array index; the equation-input scratch buffer is reused across equations so
/// a chain of single-output primitives performs no per-equation allocation.
// ── Elementwise operation fusion (bead frankenjax-a8nbp) ───────────────────
//
// A maximal RUN of consecutive cheap-elementwise equations (each single-output,
// same-shape dense f64/f32, with single-use intermediates) is evaluated in ONE
// chunked pass that materializes only the final output — skipping the N-1
// intermediate tensors the per-equation loop would allocate. Each chain step is a
// MONOMORPHIC tight loop over a cache-resident chunk (the `match` is hoisted out
// of the element loop so it autovectorizes), so it stays compute-cheap while
// cutting memory traffic to one read of each input + one write of the output.
//
// BIT-IDENTICAL to running the ops separately: the fused ops are exactly `a OP b`
// / `-a` applied in the same order per element. f32 mirrors fj-lax's dense f32
// contract: widen each tap to f64, apply the op, then round back to f32.

#[derive(Clone, Copy, PartialEq)]
enum CheapOp {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    // NaN-propagating elementwise max/min (jnp.maximum/minimum, JAX's relu/clamp
    // lowering) and abs. All three are exact (no reassociation, deterministic
    // rounding), so fusing them through a chain is bit-identical to the per-op
    // path — and lets activation/clamp pipelines fuse instead of breaking the run.
    Max,
    Min,
    Abs,
    Floor,
    Ceil,
    Trunc,
    RoundAway,
    RoundNearest,
    Sign,
}

impl CheapOp {
    /// Unary ops read a single operand (the chain value); their `b` slot is unused.
    #[inline]
    fn is_unary(self) -> bool {
        matches!(
            self,
            CheapOp::Neg
                | CheapOp::Abs
                | CheapOp::Floor
                | CheapOp::Ceil
                | CheapOp::Trunc
                | CheapOp::RoundAway
                | CheapOp::RoundNearest
                | CheapOp::Sign
        )
    }
}

/// NaN-propagating max, bit-identical to fj-lax's `jax_max_f64` (the per-op path):
/// any NaN operand yields a canonical NaN, else IEEE `f64::max`.
#[inline]
fn fused_jax_max(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else {
        left.max(right)
    }
}

#[inline]
fn fused_jax_min(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else {
        left.min(right)
    }
}

#[inline]
fn f64_fused_unary(op: CheapOp, value: f64) -> f64 {
    match op {
        CheapOp::Neg => -value,
        CheapOp::Abs => value.abs(),
        CheapOp::Floor => value.floor(),
        CheapOp::Ceil => value.ceil(),
        CheapOp::Trunc => value.trunc(),
        CheapOp::RoundAway => value.round(),
        CheapOp::RoundNearest => value.round_ties_even(),
        CheapOp::Sign => scalar_f64_sign(value),
        CheapOp::Add | CheapOp::Sub | CheapOp::Mul | CheapOp::Div | CheapOp::Max | CheapOp::Min => {
            value
        }
    }
}

fn cheap_op(p: Primitive) -> Option<CheapOp> {
    match p {
        Primitive::Add => Some(CheapOp::Add),
        Primitive::Sub => Some(CheapOp::Sub),
        Primitive::Mul => Some(CheapOp::Mul),
        Primitive::Div => Some(CheapOp::Div),
        Primitive::Neg => Some(CheapOp::Neg),
        Primitive::Max => Some(CheapOp::Max),
        Primitive::Min => Some(CheapOp::Min),
        Primitive::Abs => Some(CheapOp::Abs),
        _ => None,
    }
}

fn cheap_round_op(params: &std::collections::BTreeMap<String, String>) -> Option<CheapOp> {
    match params.get("rounding_method").map(|raw| raw.trim()) {
        None | Some("") | Some("0") | Some("AWAY_FROM_ZERO") | Some("away_from_zero") => {
            Some(CheapOp::RoundAway)
        }
        Some("1") | Some("TO_NEAREST_EVEN") | Some("to_nearest_even") | Some("nearest_even") => {
            Some(CheapOp::RoundNearest)
        }
        Some(_) => None,
    }
}

fn cheap_float_op(
    primitive: Primitive,
    params: &std::collections::BTreeMap<String, String>,
) -> Option<CheapOp> {
    if primitive == Primitive::Round {
        return cheap_round_op(params);
    }
    if !params.is_empty() {
        return None;
    }
    match primitive {
        Primitive::Floor => Some(CheapOp::Floor),
        Primitive::Ceil => Some(CheapOp::Ceil),
        Primitive::Trunc => Some(CheapOp::Trunc),
        Primitive::Sign => Some(CheapOp::Sign),
        _ => cheap_op(primitive),
    }
}

fn cheap_i64_op(
    primitive: Primitive,
    params: &std::collections::BTreeMap<String, String>,
) -> Option<CheapOp> {
    if !params.is_empty() {
        return None;
    }
    match primitive {
        Primitive::Sign => Some(CheapOp::Sign),
        _ => cheap_op(primitive),
    }
}

/// `integer_pow[2]` (the lowering of `x**2`, the common square form) equals
/// `Mul(x, x)` bit-identically for every dtype: eval_integer_pow computes
/// `v.powi(2)` (== v*v) for floats and `v.wrapping_pow(2)` (== v.wrapping_mul(v))
/// for ints — exactly eval_mul. The fusion builders treat it like `Primitive::Square`
/// (one operand duplicated into a Mul step). It carries an `exponent` param, so the
/// builders must let it past the params-empty gate; this matches ONLY exponent==2 so
/// no other param-carrying op slips through.
fn is_integer_pow_2(
    primitive: Primitive,
    params: &std::collections::BTreeMap<String, String>,
) -> bool {
    primitive == Primitive::IntegerPow
        && params.len() == 1
        && params
            .get("exponent")
            .map(|s| s.trim() == "2")
            .unwrap_or(false)
}

/// An operand of a fused step: the running chain value, an external dense-f64
/// tensor (index into the gathered `ext` slices), a row/col broadcast vector, or
/// an f64 scalar constant.
#[derive(Clone, Copy)]
enum FOperand {
    Chain,
    Ext(usize),
    RowBroadcast(usize),
    ColBroadcast { idx: usize, cols: usize },
    Scalar(f64),
}

struct FStep {
    op: CheapOp,
    a: FOperand,
    b: FOperand, // unused for Neg
}

enum FusedValues {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    /// Dense half-float (BF16/F16) bit patterns tagged with the logical dtype, so
    /// the chain materializes back via `new_half_float_values` bit-identically.
    Half {
        dtype: DType,
        values: Vec<u16>,
    },
}

struct FusedRun {
    out_var: VarId,
    values: FusedValues,
    shape: Shape,
    ext_vars: Vec<VarId>,
    run_end: usize,
}

const FUSION_MIN_RUN: usize = 3;
const FUSION_MIN_ELEMS: usize = 1024;
const FUSION_CHUNK: usize = 8192;
/// Reusable compiled runners are not universally faster for f64 linear chains:
/// the retained output/scratch path can lose to the owned compiled plan in the
/// mid-cache band. Keep the escape hatch narrow and measured; 4K still favors
/// the reusable in-place path, while 1M already falls through to chunked fusion.
const RUNNER_F64_OWNED_EVAL_MIN_ELEMS: usize = 1 << 16;
const RUNNER_F64_OWNED_EVAL_MAX_ELEMS: usize = 1 << 18;
/// Upper size for the in-place dense-f64 linear-chain path. Below this, mutating one
/// buffer through the chain (single-stream, no per-step alloc) beats the generic per-op
/// path (measured 1.25-1.45x at L2-resident, win shrinking to neutral by ~1M elems). At
/// L3-overflow / DRAM-bound sizes the win is gone and the (potentially threaded) generic
/// path may own it, so we cede those. 1<<20 ≈ 8 MB of f64 keeps the chain cache-resident.
const INPLACE_CHAIN_MAX_ELEMS: usize = 1 << 20;

/// Minimum element count before the chunked fusion driver fans out across OS
/// threads. A fused elementwise chain is arithmetic-dense — it does several ops
/// per loaded byte and writes only the final output — so it crosses the
/// thread-spawn break-even at far smaller `n` than the single-op cheap-elementwise
/// gate (which lives at ~8.4M because one cheap binop is pure bandwidth and the
/// `thread::scope` spawn dwarfs it below that). Measured: a single f64 8-op chain
/// over 1M elems runs at ~6 GB/s on ONE core, so it scales with cores; XLA already
/// uses all of them. CRITICAL contention finding: at L3-RESIDENT sizes (1M-4M, the
/// chain's ~3-buffer working set still fits this host's 128MB L3) threading is
/// BANDWIDTH-bound and its same-invocation A/B is NOT contention-immune — on an
/// idle worker f64 1M measured 1.22-1.28x, but on a CONTENDED shared rch worker the
/// SAME code measured 0.42x (threaded threads oversubscribe and thrash while the
/// serial arm is robust). Only PAST the L3->DRAM transition (>= 8.4M, where the
/// working set exceeds L3 so threads use independent memory channels) is the win
/// robust across idle AND contended: f64 16M 1.21-1.32x, f32 16M 1.33x, i64 1.07x. So the
/// gate matches the established single-op cheap-elementwise threshold (1<<23). 4M
/// and below stay serial. (See `run_{f64,f32,i64}_thread_ab` in eval_fusion_speed.)
const FUSION_THREAD_MIN_ELEMS: usize = 1 << 23; // 8.4 Mi (robust past L3->DRAM)
/// Work-scaled worker count: roughly one worker per this many elements, clamped to
/// the hardware thread count. Keeps each worker's contiguous run large enough to
/// amortize the spawn while still using every core on big tensors.
const FUSION_ELEMS_PER_THREAD: usize = 1 << 18; // 256 Ki / worker

/// Benchmark-only A/B knob (default 0 = use the production gate). When set to a
/// nonzero `k`, `fusion_thread_count` caps the worker count at `k` — so a bench
/// can run the fusion SERIAL (`k == 1`) and THREADED over identical data in the
/// SAME process, the only worker-variance-immune signal on the contended rch
/// host. Mirrors the `vectorize` flag convention in `run_dense_plan_into_core`.
/// Never set in production; the read is one relaxed load.
pub static FUSION_THREAD_CAP_OVERRIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Benchmark-only A/B knob for the dense-f64 donation path. Default 0 enables
/// production donation; nonzero forces the old clone-then-allocate unary path.
pub static DENSE_F64_DONATION_DISABLE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[inline]
fn fusion_thread_count(n: usize) -> usize {
    if n < FUSION_THREAD_MIN_ELEMS {
        return 1;
    }
    let hw = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    let want = (n / FUSION_ELEMS_PER_THREAD).clamp(1, hw);
    let cap = FUSION_THREAD_CAP_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if cap == 0 { want } else { want.min(cap) }
}

/// Drive a per-chunk fused `apply` over `values` in `FUSION_CHUNK`-sized chunks.
///
/// For large `n` the work fans out across a `std::thread::scope` over disjoint,
/// `FUSION_CHUNK`-aligned contiguous segments. This is **bit-identical** to the
/// serial loop: every element is still produced by the SAME ordered sequence of
/// scalar ops — only *which worker owns which contiguous run* changes, and there
/// is no cross-chunk reduction or reassociation. `apply` receives
/// `(chunk, global_base)`; the global base preserves broadcast row/col indexing
/// across the split (each chunk maps to the same absolute element offsets it had
/// in the serial walk).
#[inline]
fn drive_fusion_chunks<T, F>(values: &mut [T], apply: F)
where
    T: Send,
    F: Fn(&mut [T], usize) + Sync,
{
    let n = values.len();
    let threads = fusion_thread_count(n);
    if threads <= 1 {
        let mut s = 0;
        while s < n {
            let e = (s + FUSION_CHUNK).min(n);
            apply(&mut values[s..e], s);
            s = e;
        }
        return;
    }
    // Contiguous, chunk-aligned segment per worker (ceil so we never drop a tail).
    let seg = n.div_ceil(threads).next_multiple_of(FUSION_CHUNK);
    let apply = &apply;
    std::thread::scope(|scope| {
        let mut base = 0usize;
        let mut rest: &mut [T] = values;
        while !rest.is_empty() {
            let take = seg.min(rest.len());
            let (head, tail) = rest.split_at_mut(take);
            rest = tail;
            scope.spawn(move || {
                let mut s = 0;
                while s < head.len() {
                    let e = (s + FUSION_CHUNK).min(head.len());
                    apply(&mut head[s..e], base + s);
                    s = e;
                }
            });
            base += take;
        }
    });
}

/// Classify one operand atom. Pushes external dense-f64 tensors into `ext`/`ext_vars`
/// and sets/checks the run shape `S`. Returns `None` (bail) for anything not
/// fuse-eligible (non-f64 literal, non-dense or wrong-shape tensor, scalar non-f64).
fn classify_fusion_operand<'e>(
    atom: &Atom,
    chain: Option<VarId>,
    env: &'e [Option<Value>],
    ext: &mut Vec<&'e [f64]>,
    ext_vars: &mut Vec<VarId>,
    shape: &mut Option<Shape>,
) -> Option<FOperand> {
    match atom {
        Atom::Lit(Literal::F64Bits(b)) => Some(FOperand::Scalar(f64::from_bits(*b))),
        Atom::Lit(_) => None,
        Atom::Var(v) => {
            if chain == Some(*v) {
                return Some(FOperand::Chain);
            }
            let value = env.get(v.0 as usize).and_then(|s| s.as_ref())?;
            match value {
                Value::Scalar(Literal::F64Bits(b)) => Some(FOperand::Scalar(f64::from_bits(*b))),
                Value::Scalar(_) => None,
                Value::Tensor(t) => {
                    let slice = t.elements.as_f64_slice()?;
                    match shape {
                        None => *shape = Some(t.shape.clone()),
                        Some(s) if *s == t.shape => {}
                        Some(s) => {
                            if row_broadcast_len(s, &t.shape).is_some() {
                                let idx = ext.len();
                                ext.push(slice);
                                ext_vars.push(*v);
                                return Some(FOperand::RowBroadcast(idx));
                            }
                            if let Some(cols) = col_broadcast_cols(s, &t.shape) {
                                let idx = ext.len();
                                ext.push(slice);
                                ext_vars.push(*v);
                                return Some(FOperand::ColBroadcast { idx, cols });
                            }
                            return None;
                        }
                    }
                    let idx = ext.len();
                    ext.push(slice);
                    ext_vars.push(*v);
                    Some(FOperand::Ext(idx))
                }
            }
        }
    }
}

/// One f64 fused binary op. `chain_left == true` evaluates `out OP other`, matching
/// `apply_fusion_other`'s same-shape `Ext` arms; `false` evaluates `other OP out`.
#[inline]
fn f64_fused_binary(op: CheapOp, left: f64, right: f64) -> f64 {
    match op {
        CheapOp::Add => left + right,
        CheapOp::Sub => left - right,
        CheapOp::Mul => left * right,
        CheapOp::Div => left / right,
        CheapOp::Max => fused_jax_max(left, right),
        CheapOp::Min => fused_jax_min(left, right),
        // Unary ops (handled by the chunk driver's unary arms); never a binary step.
        CheapOp::Neg
        | CheapOp::Abs
        | CheapOp::Floor
        | CheapOp::Ceil
        | CheapOp::Trunc
        | CheapOp::RoundAway
        | CheapOp::RoundNearest
        | CheapOp::Sign => left,
    }
}

/// Seed `out` from a row-broadcast vector: element `(r, c)` of an `[R, C]` chunk
/// takes `row[c]`, with `c` cycling `0..C` from the chunk's flattened `base`.
#[inline]
fn seed_f64_row_broadcast(out: &mut [f64], row: &[f64], base: usize) {
    if out.is_empty() {
        return;
    }
    let row_len = row.len();
    let mut done = 0;
    let mut col = base % row_len;
    while done < out.len() {
        let take = (row_len - col).min(out.len() - done);
        out[done..done + take].copy_from_slice(&row[col..col + take]);
        done += take;
        col = 0;
    }
}

#[inline]
fn apply_f64_row_broadcast_other(
    out: &mut [f64],
    op: CheapOp,
    chain_left: bool,
    row: &[f64],
    base: usize,
) {
    if out.is_empty() {
        return;
    }
    let row_len = row.len();
    let mut done = 0;
    let mut col = base % row_len;
    while done < out.len() {
        let take = (row_len - col).min(out.len() - done);
        let out_part = &mut out[done..done + take];
        let row_part = &row[col..col + take];
        match chain_left {
            true => out_part
                .iter_mut()
                .zip(row_part)
                .for_each(|(o, e)| *o = f64_fused_binary(op, *o, *e)),
            false => out_part
                .iter_mut()
                .zip(row_part)
                .for_each(|(o, e)| *o = f64_fused_binary(op, *e, *o)),
        }
        done += take;
        col = 0;
    }
}

/// Seed `out` from a col-broadcast vector: element `(r, c)` of an `[R, C]` chunk
/// takes `col_values[r]` (one value per row, repeated across the `cols` columns).
#[inline]
fn seed_f64_col_broadcast(out: &mut [f64], col_values: &[f64], cols: usize, base: usize) {
    let mut done = 0;
    let mut linear = base;
    while done < out.len() {
        let row = linear / cols;
        let col = linear % cols;
        let take = (cols - col).min(out.len() - done);
        out[done..done + take].fill(col_values[row]);
        done += take;
        linear += take;
    }
}

#[inline]
fn apply_f64_col_broadcast_other(
    out: &mut [f64],
    op: CheapOp,
    chain_left: bool,
    col_values: &[f64],
    cols: usize,
    base: usize,
) {
    let mut done = 0;
    let mut linear = base;
    while done < out.len() {
        let row = linear / cols;
        let col = linear % cols;
        let take = (cols - col).min(out.len() - done);
        let scalar = col_values[row];
        let out_part = &mut out[done..done + take];
        match chain_left {
            true => out_part
                .iter_mut()
                .for_each(|o| *o = f64_fused_binary(op, *o, scalar)),
            false => out_part
                .iter_mut()
                .for_each(|o| *o = f64_fused_binary(op, scalar, *o)),
        }
        done += take;
        linear += take;
    }
}

/// Apply one chain step's NON-chain operand to the chunk buffer (the chain value
/// lives in `out`). Each arm is a monomorphic, autovectorizable loop.
#[inline]
fn apply_fusion_other(
    out: &mut [f64],
    op: CheapOp,
    chain_left: bool,
    other: FOperand,
    ext: &[&[f64]],
    base: usize,
) {
    match other {
        FOperand::Scalar(s) => match (op, chain_left) {
            (CheapOp::Add, _) => out.iter_mut().for_each(|o| *o += s),
            (CheapOp::Mul, _) => out.iter_mut().for_each(|o| *o *= s),
            (CheapOp::Sub, true) => out.iter_mut().for_each(|o| *o -= s),
            (CheapOp::Sub, false) => out.iter_mut().for_each(|o| *o = s - *o),
            (CheapOp::Div, true) => out.iter_mut().for_each(|o| *o /= s),
            (CheapOp::Div, false) => out.iter_mut().for_each(|o| *o = s / *o),
            // Max/Min are commutative (incl. NaN propagation), so `chain_left`
            // does not affect the value.
            (CheapOp::Max, _) => out.iter_mut().for_each(|o| *o = fused_jax_max(*o, s)),
            (CheapOp::Min, _) => out.iter_mut().for_each(|o| *o = fused_jax_min(*o, s)),
            (
                CheapOp::Neg
                | CheapOp::Abs
                | CheapOp::Floor
                | CheapOp::Ceil
                | CheapOp::Trunc
                | CheapOp::RoundAway
                | CheapOp::RoundNearest
                | CheapOp::Sign,
                _,
            ) => {}
        },
        FOperand::Ext(i) => {
            let sl = &ext[i][base..base + out.len()];
            match (op, chain_left) {
                (CheapOp::Add, _) => out.iter_mut().zip(sl).for_each(|(o, e)| *o += *e),
                (CheapOp::Mul, _) => out.iter_mut().zip(sl).for_each(|(o, e)| *o *= *e),
                (CheapOp::Sub, true) => out.iter_mut().zip(sl).for_each(|(o, e)| *o -= *e),
                (CheapOp::Sub, false) => out.iter_mut().zip(sl).for_each(|(o, e)| *o = *e - *o),
                (CheapOp::Div, true) => out.iter_mut().zip(sl).for_each(|(o, e)| *o /= *e),
                (CheapOp::Div, false) => out.iter_mut().zip(sl).for_each(|(o, e)| *o = *e / *o),
                (CheapOp::Max, _) => out
                    .iter_mut()
                    .zip(sl)
                    .for_each(|(o, e)| *o = fused_jax_max(*o, *e)),
                (CheapOp::Min, _) => out
                    .iter_mut()
                    .zip(sl)
                    .for_each(|(o, e)| *o = fused_jax_min(*o, *e)),
                (
                    CheapOp::Neg
                    | CheapOp::Abs
                    | CheapOp::Floor
                    | CheapOp::Ceil
                    | CheapOp::Trunc
                    | CheapOp::RoundAway
                    | CheapOp::RoundNearest
                    | CheapOp::Sign,
                    _,
                ) => {}
            }
        }
        FOperand::RowBroadcast(i) => {
            apply_f64_row_broadcast_other(out, op, chain_left, ext[i], base);
        }
        FOperand::ColBroadcast { idx, cols } => {
            apply_f64_col_broadcast_other(out, op, chain_left, ext[idx], cols, base);
        }
        FOperand::Chain => {}
    }
}

/// Apply a unary `CheapOp` across `out` with the op-match hoisted ABOVE the element
/// loop, so each arm is a monomorphic, vectorizable pass. Bit-identical to
/// `out.iter_mut().for_each(|o| *o = f64_fused_unary(op, *o))`: same per-element op,
/// same order. The in-loop form leaves the enum branch inside the loop body, which
/// blocks autovectorization of the rounding ops (floor/ceil/trunc/round/sign) — the
/// dominant cost of a fused unary chain (measured ~6-12x slower per pass than the
/// statically-typed kernel). Hoisting recovers the SIMD `roundpd` form per arm.
#[inline]
fn apply_f64_unary_chunk(out: &mut [f64], op: CheapOp) {
    match op {
        CheapOp::Neg => out.iter_mut().for_each(|o| *o = -*o),
        CheapOp::Abs => out.iter_mut().for_each(|o| *o = o.abs()),
        CheapOp::Floor => out.iter_mut().for_each(|o| *o = o.floor()),
        CheapOp::Ceil => out.iter_mut().for_each(|o| *o = o.ceil()),
        CheapOp::Trunc => out.iter_mut().for_each(|o| *o = o.trunc()),
        CheapOp::RoundAway => out.iter_mut().for_each(|o| *o = o.round()),
        CheapOp::RoundNearest => out.iter_mut().for_each(|o| *o = o.round_ties_even()),
        CheapOp::Sign => out.iter_mut().for_each(|o| *o = scalar_f64_sign(*o)),
        // Binary ops never reach the unary path (callers guard on op.is_unary());
        // mirror f64_fused_unary's binary arm (identity no-op) for exhaustiveness.
        CheapOp::Add | CheapOp::Sub | CheapOp::Mul | CheapOp::Div | CheapOp::Max | CheapOp::Min => {
        }
    }
}

#[derive(Clone, Copy)]
enum InplaceF64Unary {
    Cheap(CheapOp),
    Reciprocal,
}

fn inplace_f64_unary(
    primitive: Primitive,
    params: &std::collections::BTreeMap<String, String>,
) -> Option<InplaceF64Unary> {
    if primitive == Primitive::Reciprocal && params.is_empty() {
        return Some(InplaceF64Unary::Reciprocal);
    }
    let op = cheap_float_op(primitive, params)?;
    op.is_unary().then_some(InplaceF64Unary::Cheap(op))
}

#[inline]
fn apply_inplace_f64_unary_chunk(out: &mut [f64], op: InplaceF64Unary) {
    match op {
        InplaceF64Unary::Cheap(op) => apply_f64_unary_chunk(out, op),
        InplaceF64Unary::Reciprocal => out.iter_mut().for_each(|o| *o = 1.0 / *o),
    }
}

/// Evaluate the fused run over one chunk `out` (already sized to the chunk length).
fn apply_fusion_first_seeded(out: &mut [f64], step: &FStep, ext: &[&[f64]], base: usize) {
    match step.op {
        op if op.is_unary() => apply_f64_unary_chunk(out, op),
        _ => apply_fusion_other(out, step.op, true, step.b, ext, base),
    }
}

#[allow(clippy::eq_op)] // Chain/Chain must execute x-x and x/x to preserve NaN behavior.
fn apply_fusion_tail(out: &mut [f64], steps: &[FStep], ext: &[&[f64]], base: usize) {
    for step in steps {
        match step.op {
            op if op.is_unary() => {
                apply_f64_unary_chunk(out, op);
                continue;
            }
            _ => {}
        }
        match (step.a, step.b) {
            (FOperand::Chain, FOperand::Chain) => match step.op {
                CheapOp::Add => out.iter_mut().for_each(|o| *o = *o + *o),
                CheapOp::Sub => out.iter_mut().for_each(|o| *o = *o - *o),
                CheapOp::Mul => out.iter_mut().for_each(|o| *o = *o * *o),
                CheapOp::Div => out.iter_mut().for_each(|o| *o = *o / *o),
                // max(x,x)==x and min(x,x)==x (incl. NaN: x is already NaN).
                CheapOp::Max | CheapOp::Min => {}
                CheapOp::Neg
                | CheapOp::Abs
                | CheapOp::Floor
                | CheapOp::Ceil
                | CheapOp::Trunc
                | CheapOp::RoundAway
                | CheapOp::RoundNearest
                | CheapOp::Sign => {}
            },
            (FOperand::Chain, other) => apply_fusion_other(out, step.op, true, other, ext, base),
            (other, FOperand::Chain) => apply_fusion_other(out, step.op, false, other, ext, base),
            _ => {} // unreachable for a chain step
        }
    }
}

/// Evaluate the fused run over one chunk `out` (already sized to the chunk length).
fn apply_fusion_chunk(out: &mut [f64], tape: &[FStep], ext: &[&[f64]], base: usize) {
    // Step 0 has no chain operand: seed `out` from operand `a`, then apply `b`.
    let s0 = &tape[0];
    match s0.a {
        FOperand::Ext(i) => out.copy_from_slice(&ext[i][base..base + out.len()]),
        FOperand::RowBroadcast(i) => seed_f64_row_broadcast(out, ext[i], base),
        FOperand::ColBroadcast { idx, cols } => seed_f64_col_broadcast(out, ext[idx], cols, base),
        FOperand::Scalar(v) => out.fill(v),
        FOperand::Chain => {}
    }
    apply_fusion_first_seeded(out, s0, ext, base);
    apply_fusion_tail(out, &tape[1..], ext, base);
}

fn seeded_fusion_values(tape: &[FStep], ext: &[&[f64]], n: usize) -> Option<Vec<f64>> {
    match tape.first()?.a {
        FOperand::Ext(i) => Some(ext.get(i)?.get(..n)?.to_vec()),
        FOperand::Scalar(v) => Some(vec![v; n]),
        FOperand::RowBroadcast(_) | FOperand::ColBroadcast { .. } | FOperand::Chain => None,
    }
}

fn apply_seeded_fusion_chunk(out: &mut [f64], tape: &[FStep], ext: &[&[f64]], base: usize) {
    apply_fusion_first_seeded(out, &tape[0], ext, base);
    apply_fusion_tail(out, &tape[1..], ext, base);
}

fn f64_scalar_add_chain_input_and_literals<'a>(
    tape: &[FStep],
    ext: &[&'a [f64]],
) -> Option<(&'a [f64], Vec<f64>)> {
    let first = tape.first()?;
    let (CheapOp::Add, FOperand::Ext(ext_idx), FOperand::Scalar(first_add)) =
        (first.op, first.a, first.b)
    else {
        return None;
    };
    let input = ext.get(ext_idx).copied()?;
    let mut adds = Vec::with_capacity(tape.len());
    adds.push(first_add);
    for step in &tape[1..] {
        let (CheapOp::Add, FOperand::Chain, FOperand::Scalar(add)) = (step.op, step.a, step.b)
        else {
            return None;
        };
        adds.push(add);
    }
    Some((input, adds))
}

#[inline]
fn apply_f64_scalar_adds_simd(values: &mut [f64], adds: &[f64]) {
    use std::simd::Simd;

    const LANES: usize = 8;
    let (chunks, remainder) = values.as_chunks_mut::<LANES>();
    for chunk in chunks {
        let mut lanes = Simd::<f64, LANES>::from_slice(chunk);
        for &add in adds {
            lanes += Simd::splat(add);
        }
        chunk.copy_from_slice(lanes.as_array());
    }
    for value in remainder {
        for &add in adds {
            *value += add;
        }
    }
}

fn run_f64_scalar_add_chain_simd(input: &[f64], adds: &[f64]) -> Vec<f64> {
    let mut values = input.to_vec();
    drive_fusion_chunks(&mut values, |chunk, _base| {
        apply_f64_scalar_adds_simd(chunk, adds);
    });
    values
}

/// Try to fuse a maximal cheap-elementwise run starting at `start`. Returns the
/// owned fused result (no borrow of `env`) or `None` to fall through to the normal
/// per-equation path. Opt-in: any non-matching condition bails with zero effect.
#[allow(clippy::while_let_loop)] // The loop has several ordered bailout checkpoints.
fn try_fuse_elementwise_chain_f64(
    jaxpr: &Jaxpr,
    start: usize,
    env: &[Option<Value>],
    last_use: &[usize],
) -> Option<FusedRun> {
    let eqns = &jaxpr.equations;
    let mut ext: Vec<&[f64]> = Vec::new();
    let mut ext_vars: Vec<VarId> = Vec::new();
    let mut tape: Vec<FStep> = Vec::new();
    let mut shape: Option<Shape> = None;
    let mut chain_var: Option<VarId> = None;
    let mut run_out: Option<VarId> = None;
    let mut run_end = start;

    let mut k = start;
    loop {
        let Some(eqn) = eqns.get(k) else { break };
        if !eqn.sub_jaxprs.is_empty() || eqn.outputs.len() != 1 {
            break;
        }
        let ext_mark = ext.len();
        let vars_mark = ext_vars.len();
        // Square(x) is fused as Mul(x, x): eval_square's float_op is `|x| x*x` and its
        // int_op `wrapping_mul(x, x)` — exactly eval_mul — so duplicating the single
        // operand into a Mul step is bit-identical and reuses the proven Mul machinery
        // (no new CheapOp / apply arm). Lets variance/L2/MSE chains (mean((x-mu)^2))
        // fuse instead of breaking the run on the Square primitive.
        let (op, a, b) = if eqn.primitive == Primitive::Square && eqn.params.is_empty()
            || is_integer_pow_2(eqn.primitive, &eqn.params)
        {
            if eqn.inputs.len() != 1 {
                break;
            }
            let Some(a) = classify_fusion_operand(
                &eqn.inputs[0],
                chain_var,
                env,
                &mut ext,
                &mut ext_vars,
                &mut shape,
            ) else {
                ext.truncate(ext_mark);
                ext_vars.truncate(vars_mark);
                break;
            };
            (CheapOp::Mul, a, a)
        } else if eqn.primitive == Primitive::Reciprocal && eqn.params.is_empty() {
            // Reciprocal(x) == Div(1, x): eval_reciprocal is eval_unary_elementwise(|x| 1.0/x),
            // and the f64 fused Div computes 1.0/x identically — so emit Div(Scalar(1.0), x),
            // reusing the proven Div machinery. The classify guard requires a dense-f64 input,
            // so an i64/other input (whose eval_reciprocal widens to f64) bails to generic.
            if eqn.inputs.len() != 1 {
                break;
            }
            let Some(b) = classify_fusion_operand(
                &eqn.inputs[0],
                chain_var,
                env,
                &mut ext,
                &mut ext_vars,
                &mut shape,
            ) else {
                ext.truncate(ext_mark);
                ext_vars.truncate(vars_mark);
                break;
            };
            (CheapOp::Div, FOperand::Scalar(1.0), b)
        } else {
            let Some(op) = cheap_float_op(eqn.primitive, &eqn.params) else {
                break;
            };
            let needed = if op.is_unary() { 1 } else { 2 };
            if eqn.inputs.len() != needed {
                break;
            }
            let a = classify_fusion_operand(
                &eqn.inputs[0],
                chain_var,
                env,
                &mut ext,
                &mut ext_vars,
                &mut shape,
            );
            let b = if op.is_unary() {
                Some(FOperand::Scalar(0.0))
            } else {
                classify_fusion_operand(
                    &eqn.inputs[1],
                    chain_var,
                    env,
                    &mut ext,
                    &mut ext_vars,
                    &mut shape,
                )
            };
            let (Some(a), Some(b)) = (a, b) else {
                ext.truncate(ext_mark);
                ext_vars.truncate(vars_mark);
                break;
            };
            (op, a, b)
        };
        // Steps after the first MUST thread the chain (one operand == Chain). The
        // last_use==k guarantee ensures eqn k uses chain_var, but stay defensive.
        if chain_var.is_some() && !matches!(a, FOperand::Chain) && !matches!(b, FOperand::Chain) {
            ext.truncate(ext_mark);
            ext_vars.truncate(vars_mark);
            break;
        }
        tape.push(FStep { op, a, b });
        run_out = Some(eqn.outputs[0]);
        run_end = k;
        let out_idx = eqn.outputs[0].0 as usize;
        if k + 1 < eqns.len() && last_use.get(out_idx).copied() == Some(k + 1) {
            chain_var = Some(eqn.outputs[0]);
            k += 1;
        } else {
            break;
        }
    }

    if tape.len() < FUSION_MIN_RUN {
        return None;
    }
    let shape = shape?;
    let n = shape.element_count()? as usize;
    if n < FUSION_MIN_ELEMS {
        return None;
    }
    let out_var = run_out?;

    // The single-pass register kernel wins in the medium band, but loses once
    // the existing chunk driver starts threading, so keep the gate below that.
    if (INPLACE_CHAIN_MAX_ELEMS..FUSION_THREAD_MIN_ELEMS).contains(&n)
        && let Some((input, adds)) = f64_scalar_add_chain_input_and_literals(&tape, &ext)
    {
        return Some(FusedRun {
            out_var,
            values: FusedValues::F64(run_f64_scalar_add_chain_simd(input, &adds)),
            shape,
            ext_vars,
            run_end,
        });
    }

    let values = if n < FUSION_THREAD_MIN_ELEMS
        && let Some(mut seeded) = seeded_fusion_values(&tape, &ext, n)
    {
        drive_fusion_chunks(&mut seeded, |chunk, base| {
            apply_seeded_fusion_chunk(chunk, &tape, &ext, base)
        });
        seeded
    } else {
        let mut values = vec![0.0_f64; n];
        drive_fusion_chunks(&mut values, |chunk, base| {
            apply_fusion_chunk(chunk, &tape, &ext, base)
        });
        values
    };
    Some(FusedRun {
        out_var,
        values: FusedValues::F64(values),
        shape,
        ext_vars,
        run_end,
    })
}

/// An operand of a fused f32 step: the running chain value, an external dense-f32
/// tensor (index into the gathered `ext` slices), or an f32 scalar constant.
#[derive(Clone, Copy)]
enum F32Operand {
    Chain,
    Ext(usize),
    RowBroadcast(usize),
    ColBroadcast { idx: usize, cols: usize },
    Scalar(f32),
}

struct F32Step {
    op: CheapOp,
    a: F32Operand,
    b: F32Operand, // unused for Neg
}

/// Classify one f32 operand atom. This is deliberately separate from the f64
/// classifier so the f64 path remains mechanically unchanged.
fn classify_f32_fusion_operand<'e>(
    atom: &Atom,
    chain: Option<VarId>,
    env: &'e [Option<Value>],
    ext: &mut Vec<&'e [f32]>,
    ext_vars: &mut Vec<VarId>,
    shape: &mut Option<Shape>,
) -> Option<F32Operand> {
    match atom {
        Atom::Lit(Literal::F32Bits(b)) => Some(F32Operand::Scalar(f32::from_bits(*b))),
        Atom::Lit(_) => None,
        Atom::Var(v) => {
            if chain == Some(*v) {
                return Some(F32Operand::Chain);
            }
            let value = env.get(v.0 as usize).and_then(|s| s.as_ref())?;
            match value {
                Value::Scalar(Literal::F32Bits(b)) => Some(F32Operand::Scalar(f32::from_bits(*b))),
                Value::Scalar(_) => None,
                Value::Tensor(t) => {
                    if t.dtype != DType::F32 {
                        return None;
                    }
                    let slice = t.elements.as_f32_slice()?;
                    match shape {
                        None => *shape = Some(t.shape.clone()),
                        Some(s) if *s == t.shape => {}
                        Some(s) => {
                            if row_broadcast_len(s, &t.shape).is_some() {
                                let idx = ext.len();
                                ext.push(slice);
                                ext_vars.push(*v);
                                return Some(F32Operand::RowBroadcast(idx));
                            }
                            if let Some(cols) = col_broadcast_cols(s, &t.shape) {
                                let idx = ext.len();
                                ext.push(slice);
                                ext_vars.push(*v);
                                return Some(F32Operand::ColBroadcast { idx, cols });
                            }
                            return None;
                        }
                    }
                    let idx = ext.len();
                    ext.push(slice);
                    ext_vars.push(*v);
                    Some(F32Operand::Ext(idx))
                }
            }
        }
    }
}

/// Shape-only (dtype-agnostic) broadcast classifiers shared by the f64 and f32
/// fusion scanners.
/// Detect a TRAILING-axis row broadcast: a 1-D `[C]` vector broadcasting against
/// the last axis of any rank-≥2 full tensor `[…, C]`. Element flat-index `i` reads
/// `vec[i % C]` (row-major) — exactly what the flat seed/apply (`base % row_len`)
/// and the `at_rc` col index already compute for ANY rank, so generalizing past the
/// old rank-2-only `[R,C]+[C]` is free and bit-identical to the dense-broadcast
/// kernel (whose `[C]` operand has broadcast strides `[0,…,0,1]`). Covers the
/// ubiquitous `[B,S,D] + [D]` transformer bias/scale add.
fn row_broadcast_len(full: &Shape, candidate: &Shape) -> Option<usize> {
    match candidate.dims.as_slice() {
        [row_cols] if full.dims.len() >= 2 && full.dims.last() == Some(row_cols) => {
            Some(*row_cols as usize)
        }
        _ => None,
    }
}

/// Detect a column broadcast: a `[…, 1]` vector with the SAME leading dims as the
/// full tensor `[…, C]` and trailing dim 1. Element flat-index `i` reads
/// `vec[i / C]` — what the flat seed/apply (`linear / cols`) and `at_rc` row index
/// compute for ANY rank, so generalizing past the old rank-2-only `[R,C]+[R,1]` is
/// free and bit-identical (the `[…,1]` operand broadcasts with trailing stride 0).
fn col_broadcast_cols(full: &Shape, candidate: &Shape) -> Option<usize> {
    let f = full.dims.as_slice();
    let c = candidate.dims.as_slice();
    if f.len() == c.len()
        && f.len() >= 2
        && c.last() == Some(&1)
        && f[..f.len() - 1] == c[..c.len() - 1]
    {
        Some(f[f.len() - 1] as usize)
    } else {
        None
    }
}

#[inline]
fn seed_f32_row_broadcast(out: &mut [f32], row: &[f32], base: usize) {
    if out.is_empty() {
        return;
    }
    let row_len = row.len();
    let mut done = 0;
    let mut col = base % row_len;
    while done < out.len() {
        let take = (row_len - col).min(out.len() - done);
        out[done..done + take].copy_from_slice(&row[col..col + take]);
        done += take;
        col = 0;
    }
}

#[inline]
fn apply_f32_row_broadcast_other(
    out: &mut [f32],
    op: CheapOp,
    chain_left: bool,
    row: &[f32],
    base: usize,
) {
    if out.is_empty() {
        return;
    }
    let row_len = row.len();
    let mut done = 0;
    let mut col = base % row_len;
    while done < out.len() {
        let take = (row_len - col).min(out.len() - done);
        let out_part = &mut out[done..done + take];
        let row_part = &row[col..col + take];
        match (op, chain_left) {
            (CheapOp::Add, _) => out_part
                .iter_mut()
                .zip(row_part)
                .for_each(|(o, e)| *o = (f64::from(*o) + f64::from(*e)) as f32),
            (CheapOp::Mul, _) => out_part
                .iter_mut()
                .zip(row_part)
                .for_each(|(o, e)| *o = (f64::from(*o) * f64::from(*e)) as f32),
            (CheapOp::Sub, true) => out_part
                .iter_mut()
                .zip(row_part)
                .for_each(|(o, e)| *o = (f64::from(*o) - f64::from(*e)) as f32),
            (CheapOp::Sub, false) => out_part
                .iter_mut()
                .zip(row_part)
                .for_each(|(o, e)| *o = (f64::from(*e) - f64::from(*o)) as f32),
            (CheapOp::Div, true) => out_part
                .iter_mut()
                .zip(row_part)
                .for_each(|(o, e)| *o = (f64::from(*o) / f64::from(*e)) as f32),
            (CheapOp::Div, false) => out_part
                .iter_mut()
                .zip(row_part)
                .for_each(|(o, e)| *o = (f64::from(*e) / f64::from(*o)) as f32),
            (CheapOp::Max, _) => out_part
                .iter_mut()
                .zip(row_part)
                .for_each(|(o, e)| *o = fused_jax_max(f64::from(*o), f64::from(*e)) as f32),
            (CheapOp::Min, _) => out_part
                .iter_mut()
                .zip(row_part)
                .for_each(|(o, e)| *o = fused_jax_min(f64::from(*o), f64::from(*e)) as f32),
            (
                CheapOp::Neg
                | CheapOp::Abs
                | CheapOp::Floor
                | CheapOp::Ceil
                | CheapOp::Trunc
                | CheapOp::RoundAway
                | CheapOp::RoundNearest
                | CheapOp::Sign,
                _,
            ) => {}
        }
        done += take;
        col = 0;
    }
}

#[inline]
fn seed_f32_col_broadcast(out: &mut [f32], col_values: &[f32], cols: usize, base: usize) {
    let mut done = 0;
    let mut linear = base;
    while done < out.len() {
        let row = linear / cols;
        let col = linear % cols;
        let take = (cols - col).min(out.len() - done);
        out[done..done + take].fill(col_values[row]);
        done += take;
        linear += take;
    }
}

#[inline]
fn apply_f32_col_broadcast_other(
    out: &mut [f32],
    op: CheapOp,
    chain_left: bool,
    col_values: &[f32],
    cols: usize,
    base: usize,
) {
    let mut done = 0;
    let mut linear = base;
    while done < out.len() {
        let row = linear / cols;
        let col = linear % cols;
        let take = (cols - col).min(out.len() - done);
        let scalar = col_values[row];
        let out_part = &mut out[done..done + take];
        let scalar = f64::from(scalar);
        match (op, chain_left) {
            (CheapOp::Add, _) => out_part
                .iter_mut()
                .for_each(|o| *o = (f64::from(*o) + scalar) as f32),
            (CheapOp::Mul, _) => out_part
                .iter_mut()
                .for_each(|o| *o = (f64::from(*o) * scalar) as f32),
            (CheapOp::Sub, true) => out_part
                .iter_mut()
                .for_each(|o| *o = (f64::from(*o) - scalar) as f32),
            (CheapOp::Sub, false) => out_part
                .iter_mut()
                .for_each(|o| *o = (scalar - f64::from(*o)) as f32),
            (CheapOp::Div, true) => out_part
                .iter_mut()
                .for_each(|o| *o = (f64::from(*o) / scalar) as f32),
            (CheapOp::Div, false) => out_part
                .iter_mut()
                .for_each(|o| *o = (scalar / f64::from(*o)) as f32),
            (CheapOp::Max, _) => out_part
                .iter_mut()
                .for_each(|o| *o = fused_jax_max(f64::from(*o), scalar) as f32),
            (CheapOp::Min, _) => out_part
                .iter_mut()
                .for_each(|o| *o = fused_jax_min(f64::from(*o), scalar) as f32),
            (
                CheapOp::Neg
                | CheapOp::Abs
                | CheapOp::Floor
                | CheapOp::Ceil
                | CheapOp::Trunc
                | CheapOp::RoundAway
                | CheapOp::RoundNearest
                | CheapOp::Sign,
                _,
            ) => {}
        }
        done += take;
        linear += take;
    }
}

/// Apply one f32 chain step's NON-chain operand to the chunk buffer. Each arm
/// mirrors fj-lax's dense f32 arithmetic: f32->f64, op, round to f32.
#[inline]
fn apply_f32_fusion_other(
    out: &mut [f32],
    op: CheapOp,
    chain_left: bool,
    other: F32Operand,
    ext: &[&[f32]],
    base: usize,
) {
    match other {
        F32Operand::Scalar(s) => {
            let s = f64::from(s);
            match (op, chain_left) {
                (CheapOp::Add, _) => out.iter_mut().for_each(|o| *o = (f64::from(*o) + s) as f32),
                (CheapOp::Mul, _) => out.iter_mut().for_each(|o| *o = (f64::from(*o) * s) as f32),
                (CheapOp::Sub, true) => {
                    out.iter_mut().for_each(|o| *o = (f64::from(*o) - s) as f32)
                }
                (CheapOp::Sub, false) => {
                    out.iter_mut().for_each(|o| *o = (s - f64::from(*o)) as f32)
                }
                (CheapOp::Div, true) => {
                    out.iter_mut().for_each(|o| *o = (f64::from(*o) / s) as f32)
                }
                (CheapOp::Div, false) => {
                    out.iter_mut().for_each(|o| *o = (s / f64::from(*o)) as f32)
                }
                (CheapOp::Max, _) => out
                    .iter_mut()
                    .for_each(|o| *o = fused_jax_max(f64::from(*o), s) as f32),
                (CheapOp::Min, _) => out
                    .iter_mut()
                    .for_each(|o| *o = fused_jax_min(f64::from(*o), s) as f32),
                (
                    CheapOp::Neg
                    | CheapOp::Abs
                    | CheapOp::Floor
                    | CheapOp::Ceil
                    | CheapOp::Trunc
                    | CheapOp::RoundAway
                    | CheapOp::RoundNearest
                    | CheapOp::Sign,
                    _,
                ) => {}
            }
        }
        F32Operand::Ext(i) => {
            let sl = &ext[i][base..base + out.len()];
            match (op, chain_left) {
                (CheapOp::Add, _) => out
                    .iter_mut()
                    .zip(sl)
                    .for_each(|(o, e)| *o = (f64::from(*o) + f64::from(*e)) as f32),
                (CheapOp::Mul, _) => out
                    .iter_mut()
                    .zip(sl)
                    .for_each(|(o, e)| *o = (f64::from(*o) * f64::from(*e)) as f32),
                (CheapOp::Sub, true) => out
                    .iter_mut()
                    .zip(sl)
                    .for_each(|(o, e)| *o = (f64::from(*o) - f64::from(*e)) as f32),
                (CheapOp::Sub, false) => out
                    .iter_mut()
                    .zip(sl)
                    .for_each(|(o, e)| *o = (f64::from(*e) - f64::from(*o)) as f32),
                (CheapOp::Div, true) => out
                    .iter_mut()
                    .zip(sl)
                    .for_each(|(o, e)| *o = (f64::from(*o) / f64::from(*e)) as f32),
                (CheapOp::Div, false) => out
                    .iter_mut()
                    .zip(sl)
                    .for_each(|(o, e)| *o = (f64::from(*e) / f64::from(*o)) as f32),
                (CheapOp::Max, _) => out
                    .iter_mut()
                    .zip(sl)
                    .for_each(|(o, e)| *o = fused_jax_max(f64::from(*o), f64::from(*e)) as f32),
                (CheapOp::Min, _) => out
                    .iter_mut()
                    .zip(sl)
                    .for_each(|(o, e)| *o = fused_jax_min(f64::from(*o), f64::from(*e)) as f32),
                (
                    CheapOp::Neg
                    | CheapOp::Abs
                    | CheapOp::Floor
                    | CheapOp::Ceil
                    | CheapOp::Trunc
                    | CheapOp::RoundAway
                    | CheapOp::RoundNearest
                    | CheapOp::Sign,
                    _,
                ) => {}
            }
        }
        F32Operand::RowBroadcast(i) => {
            let row = ext[i];
            apply_f32_row_broadcast_other(out, op, chain_left, row, base);
        }
        F32Operand::ColBroadcast { idx, cols } => {
            let col = ext[idx];
            apply_f32_col_broadcast_other(out, op, chain_left, col, cols, base);
        }
        F32Operand::Chain => {}
    }
}

/// f32 sibling of [`apply_f64_unary_chunk`]: hoists the unary op-match above the
/// element loop. Preserves the production widen-to-f64 contract
/// (`f64_fused_unary(op, f64::from(v)) as f32`), so it is bit-identical to
/// `out.iter_mut().for_each(|o| *o = f32_fused_unary(op, *o))` while letting each arm
/// vectorize (widen → `roundpd` → narrow) instead of branching per element.
#[inline]
fn apply_f32_unary_chunk(out: &mut [f32], op: CheapOp) {
    match op {
        CheapOp::Neg => out.iter_mut().for_each(|o| *o = (-f64::from(*o)) as f32),
        CheapOp::Abs => out.iter_mut().for_each(|o| *o = f64::from(*o).abs() as f32),
        CheapOp::Floor => out
            .iter_mut()
            .for_each(|o| *o = f64::from(*o).floor() as f32),
        CheapOp::Ceil => out
            .iter_mut()
            .for_each(|o| *o = f64::from(*o).ceil() as f32),
        CheapOp::Trunc => out
            .iter_mut()
            .for_each(|o| *o = f64::from(*o).trunc() as f32),
        CheapOp::RoundAway => out
            .iter_mut()
            .for_each(|o| *o = f64::from(*o).round() as f32),
        CheapOp::RoundNearest => out
            .iter_mut()
            .for_each(|o| *o = f64::from(*o).round_ties_even() as f32),
        CheapOp::Sign => out
            .iter_mut()
            .for_each(|o| *o = scalar_f64_sign(f64::from(*o)) as f32),
        // Binary ops never reach the unary path (callers guard on op.is_unary()).
        CheapOp::Add | CheapOp::Sub | CheapOp::Mul | CheapOp::Div | CheapOp::Max | CheapOp::Min => {
        }
    }
}

/// Evaluate the fused f32 run over one chunk `out` (already sized to the chunk length).
#[allow(clippy::eq_op)] // Chain/Chain must execute x-x and x/x to preserve NaN behavior.
fn apply_f32_fusion_chunk(out: &mut [f32], tape: &[F32Step], ext: &[&[f32]], base: usize) {
    // Step 0 has no chain operand: seed `out` from operand `a`, then apply `b`.
    let s0 = &tape[0];
    match s0.a {
        F32Operand::Ext(i) => out.copy_from_slice(&ext[i][base..base + out.len()]),
        F32Operand::RowBroadcast(i) => {
            let row = ext[i];
            seed_f32_row_broadcast(out, row, base);
        }
        F32Operand::ColBroadcast { idx, cols } => {
            let col = ext[idx];
            seed_f32_col_broadcast(out, col, cols, base);
        }
        F32Operand::Scalar(v) => out.fill(v),
        F32Operand::Chain => {}
    }
    match s0.op {
        op if op.is_unary() => apply_f32_unary_chunk(out, op),
        _ => apply_f32_fusion_other(out, s0.op, true, s0.b, ext, base),
    }
    for step in &tape[1..] {
        match step.op {
            op if op.is_unary() => {
                apply_f32_unary_chunk(out, op);
                continue;
            }
            _ => {}
        }
        match (step.a, step.b) {
            (F32Operand::Chain, F32Operand::Chain) => match step.op {
                CheapOp::Add => out
                    .iter_mut()
                    .for_each(|o| *o = (f64::from(*o) + f64::from(*o)) as f32),
                CheapOp::Sub => out
                    .iter_mut()
                    .for_each(|o| *o = (f64::from(*o) - f64::from(*o)) as f32),
                CheapOp::Mul => out
                    .iter_mut()
                    .for_each(|o| *o = (f64::from(*o) * f64::from(*o)) as f32),
                CheapOp::Div => out
                    .iter_mut()
                    .for_each(|o| *o = (f64::from(*o) / f64::from(*o)) as f32),
                CheapOp::Max | CheapOp::Min => {}
                CheapOp::Neg
                | CheapOp::Abs
                | CheapOp::Floor
                | CheapOp::Ceil
                | CheapOp::Trunc
                | CheapOp::RoundAway
                | CheapOp::RoundNearest
                | CheapOp::Sign => {}
            },
            (F32Operand::Chain, other) => {
                apply_f32_fusion_other(out, step.op, true, other, ext, base);
            }
            (other, F32Operand::Chain) => {
                apply_f32_fusion_other(out, step.op, false, other, ext, base);
            }
            _ => {} // unreachable for a chain step
        }
    }
}

#[allow(clippy::while_let_loop)] // Mirrors the f64 fusion scanner's ordered bailouts.
fn try_fuse_elementwise_chain_f32(
    jaxpr: &Jaxpr,
    start: usize,
    env: &[Option<Value>],
    last_use: &[usize],
) -> Option<FusedRun> {
    let eqns = &jaxpr.equations;
    let mut ext: Vec<&[f32]> = Vec::new();
    let mut ext_vars: Vec<VarId> = Vec::new();
    let mut tape: Vec<F32Step> = Vec::new();
    let mut shape: Option<Shape> = None;
    let mut chain_var: Option<VarId> = None;
    let mut run_out: Option<VarId> = None;
    let mut run_end = start;

    let mut k = start;
    loop {
        let Some(eqn) = eqns.get(k) else { break };
        if !eqn.sub_jaxprs.is_empty() || eqn.outputs.len() != 1 {
            break;
        }
        let ext_mark = ext.len();
        let vars_mark = ext_vars.len();
        // Square(x) fused as Mul(x, x) — see the f64 builder for the bit-identity proof.
        let (op, a, b) = if eqn.primitive == Primitive::Square && eqn.params.is_empty()
            || is_integer_pow_2(eqn.primitive, &eqn.params)
        {
            if eqn.inputs.len() != 1 {
                break;
            }
            let Some(a) = classify_f32_fusion_operand(
                &eqn.inputs[0],
                chain_var,
                env,
                &mut ext,
                &mut ext_vars,
                &mut shape,
            ) else {
                ext.truncate(ext_mark);
                ext_vars.truncate(vars_mark);
                break;
            };
            (CheapOp::Mul, a, a)
        } else if eqn.primitive == Primitive::Reciprocal && eqn.params.is_empty() {
            // Reciprocal(x) == Div(1, x); the f32 fused Div uses the f32->f64->f32
            // contract matching eval_reciprocal (eval_unary_elementwise f64-widen). See
            // the f64 builder. Dense-f32 input required (else bails to generic).
            if eqn.inputs.len() != 1 {
                break;
            }
            let Some(b) = classify_f32_fusion_operand(
                &eqn.inputs[0],
                chain_var,
                env,
                &mut ext,
                &mut ext_vars,
                &mut shape,
            ) else {
                ext.truncate(ext_mark);
                ext_vars.truncate(vars_mark);
                break;
            };
            (CheapOp::Div, F32Operand::Scalar(1.0), b)
        } else {
            let Some(op) = cheap_float_op(eqn.primitive, &eqn.params) else {
                break;
            };
            let needed = if op.is_unary() { 1 } else { 2 };
            if eqn.inputs.len() != needed {
                break;
            }
            let a = classify_f32_fusion_operand(
                &eqn.inputs[0],
                chain_var,
                env,
                &mut ext,
                &mut ext_vars,
                &mut shape,
            );
            let b = if op.is_unary() {
                Some(F32Operand::Scalar(0.0))
            } else {
                classify_f32_fusion_operand(
                    &eqn.inputs[1],
                    chain_var,
                    env,
                    &mut ext,
                    &mut ext_vars,
                    &mut shape,
                )
            };
            let (Some(a), Some(b)) = (a, b) else {
                ext.truncate(ext_mark);
                ext_vars.truncate(vars_mark);
                break;
            };
            (op, a, b)
        };
        // Steps after the first MUST thread the chain (one operand == Chain). The
        // last_use==k guarantee ensures eqn k uses chain_var, but stay defensive.
        if chain_var.is_some() && !matches!(a, F32Operand::Chain) && !matches!(b, F32Operand::Chain)
        {
            ext.truncate(ext_mark);
            ext_vars.truncate(vars_mark);
            break;
        }
        tape.push(F32Step { op, a, b });
        run_out = Some(eqn.outputs[0]);
        run_end = k;
        let out_idx = eqn.outputs[0].0 as usize;
        if k + 1 < eqns.len() && last_use.get(out_idx).copied() == Some(k + 1) {
            chain_var = Some(eqn.outputs[0]);
            k += 1;
        } else {
            break;
        }
    }

    if tape.len() < FUSION_MIN_RUN {
        return None;
    }
    let shape = shape?;
    let n = shape.element_count()? as usize;
    if n < FUSION_MIN_ELEMS {
        return None;
    }
    let out_var = run_out?;

    let mut values = vec![0.0_f32; n];
    drive_fusion_chunks(&mut values, |chunk, base| {
        apply_f32_fusion_chunk(chunk, &tape, &ext, base)
    });
    Some(FusedRun {
        out_var,
        values: FusedValues::F32(values),
        shape,
        ext_vars,
        run_end,
    })
}

// ── i64 elementwise fusion ─────────────────────────────────────────────────
//
// The integer sibling of the f64/f32 fusion paths. A maximal run of same-shape
// dense-I64 cheap-elementwise equations (Add/Sub/Mul/Div/Neg) with single-use
// intermediates is evaluated in ONE chunked pass. BIT-IDENTICAL to the unfused
// per-equation path because each fused step applies the EXACT same i64 closure
// fj-lax's dispatcher uses (wrapping_add/sub/mul, checked_div(_).unwrap_or(0),
// wrapping_neg) in the same operand order — and a pure-I64 chain never promotes
// dtype (I64⊗I64 -> I64), so no intermediate narrowing can differ.

/// An operand of a fused i64 step: the running chain value, an external dense-i64
/// tensor (index into the gathered `ext` slices), or an i64 scalar constant.
#[derive(Clone, Copy)]
enum I64Operand {
    Chain,
    Ext(usize),
    RowBroadcast(usize),
    ColBroadcast { idx: usize, cols: usize },
    Scalar(i64),
}

/// One i64 fused binary op — the exact fj-lax dispatcher semantics inlined by the
/// i64 fusion arms (wrapping +/-/*, checked_div→0, total max/min). Shared by the
/// i64 broadcast helpers so they match the Scalar/Ext arms bit-for-bit.
#[inline]
fn i64_fused_binary(op: CheapOp, left: i64, right: i64) -> i64 {
    match op {
        CheapOp::Add => left.wrapping_add(right),
        CheapOp::Sub => left.wrapping_sub(right),
        CheapOp::Mul => left.wrapping_mul(right),
        CheapOp::Div => left.checked_div(right).unwrap_or(0),
        CheapOp::Max => left.max(right),
        CheapOp::Min => left.min(right),
        // Unary ops are handled by the chunk driver; never a binary step.
        CheapOp::Neg
        | CheapOp::Abs
        | CheapOp::Floor
        | CheapOp::Ceil
        | CheapOp::Trunc
        | CheapOp::RoundAway
        | CheapOp::RoundNearest
        | CheapOp::Sign => left,
    }
}

#[inline]
fn i64_fused_unary(op: CheapOp, value: i64) -> i64 {
    match op {
        CheapOp::Neg => value.wrapping_neg(),
        CheapOp::Abs => value.wrapping_abs(),
        CheapOp::Sign => value.signum(),
        CheapOp::Floor
        | CheapOp::Ceil
        | CheapOp::Trunc
        | CheapOp::RoundAway
        | CheapOp::RoundNearest
        | CheapOp::Add
        | CheapOp::Sub
        | CheapOp::Mul
        | CheapOp::Div
        | CheapOp::Max
        | CheapOp::Min => value,
    }
}

#[inline]
fn seed_i64_row_broadcast(out: &mut [i64], row: &[i64], base: usize) {
    if out.is_empty() {
        return;
    }
    let row_len = row.len();
    let mut done = 0;
    let mut col = base % row_len;
    while done < out.len() {
        let take = (row_len - col).min(out.len() - done);
        out[done..done + take].copy_from_slice(&row[col..col + take]);
        done += take;
        col = 0;
    }
}

#[inline]
fn apply_i64_row_broadcast_other(
    out: &mut [i64],
    op: CheapOp,
    chain_left: bool,
    row: &[i64],
    base: usize,
) {
    if out.is_empty() {
        return;
    }
    let row_len = row.len();
    let mut done = 0;
    let mut col = base % row_len;
    while done < out.len() {
        let take = (row_len - col).min(out.len() - done);
        let out_part = &mut out[done..done + take];
        let row_part = &row[col..col + take];
        match chain_left {
            true => out_part
                .iter_mut()
                .zip(row_part)
                .for_each(|(o, e)| *o = i64_fused_binary(op, *o, *e)),
            false => out_part
                .iter_mut()
                .zip(row_part)
                .for_each(|(o, e)| *o = i64_fused_binary(op, *e, *o)),
        }
        done += take;
        col = 0;
    }
}

#[inline]
fn seed_i64_col_broadcast(out: &mut [i64], col_values: &[i64], cols: usize, base: usize) {
    let mut done = 0;
    let mut linear = base;
    while done < out.len() {
        let row = linear / cols;
        let col = linear % cols;
        let take = (cols - col).min(out.len() - done);
        out[done..done + take].fill(col_values[row]);
        done += take;
        linear += take;
    }
}

#[inline]
fn apply_i64_col_broadcast_other(
    out: &mut [i64],
    op: CheapOp,
    chain_left: bool,
    col_values: &[i64],
    cols: usize,
    base: usize,
) {
    let mut done = 0;
    let mut linear = base;
    while done < out.len() {
        let row = linear / cols;
        let col = linear % cols;
        let take = (cols - col).min(out.len() - done);
        let scalar = col_values[row];
        let out_part = &mut out[done..done + take];
        match chain_left {
            true => out_part
                .iter_mut()
                .for_each(|o| *o = i64_fused_binary(op, *o, scalar)),
            false => out_part
                .iter_mut()
                .for_each(|o| *o = i64_fused_binary(op, scalar, *o)),
        }
        done += take;
        linear += take;
    }
}

struct I64Step {
    op: CheapOp,
    a: I64Operand,
    b: I64Operand, // unused for Neg
}

/// Classify one i64 operand atom. Pushes external dense-I64 tensors into
/// `ext`/`ext_vars` and sets/checks the run shape. Returns `None` (bail) for
/// anything not fuse-eligible (non-i64 literal, non-dense / wrong-shape / non-I64
/// tensor, i64 scalar of a different dtype). Same-shape only (no broadcast).
fn classify_i64_fusion_operand<'e>(
    atom: &Atom,
    chain: Option<VarId>,
    env: &'e [Option<Value>],
    ext: &mut Vec<&'e [i64]>,
    ext_vars: &mut Vec<VarId>,
    shape: &mut Option<Shape>,
) -> Option<I64Operand> {
    match atom {
        Atom::Lit(Literal::I64(v)) => Some(I64Operand::Scalar(*v)),
        Atom::Lit(_) => None,
        Atom::Var(v) => {
            if chain == Some(*v) {
                return Some(I64Operand::Chain);
            }
            let value = env.get(v.0 as usize).and_then(|s| s.as_ref())?;
            match value {
                Value::Scalar(Literal::I64(s)) => Some(I64Operand::Scalar(*s)),
                Value::Scalar(_) => None,
                Value::Tensor(t) => {
                    if t.dtype != DType::I64 {
                        return None;
                    }
                    let slice = t.elements.as_i64_slice()?;
                    match shape {
                        None => *shape = Some(t.shape.clone()),
                        Some(s) if *s == t.shape => {}
                        Some(s) => {
                            // [C] row / [...,1] col broadcast (shared detectors); pure
                            // wrapping int arith with broadcast indexing, bit-identical.
                            if row_broadcast_len(s, &t.shape).is_some() {
                                let idx = ext.len();
                                ext.push(slice);
                                ext_vars.push(*v);
                                return Some(I64Operand::RowBroadcast(idx));
                            }
                            if let Some(cols) = col_broadcast_cols(s, &t.shape) {
                                let idx = ext.len();
                                ext.push(slice);
                                ext_vars.push(*v);
                                return Some(I64Operand::ColBroadcast { idx, cols });
                            }
                            return None;
                        }
                    }
                    let idx = ext.len();
                    ext.push(slice);
                    ext_vars.push(*v);
                    Some(I64Operand::Ext(idx))
                }
            }
        }
    }
}

/// Apply one i64 chain step's NON-chain operand to the chunk buffer (the chain
/// value lives in `out`). Each arm is a monomorphic loop; the wrapping add/sub/mul
/// arms autovectorize (checked_div carries a zero/overflow branch and does not).
#[inline]
fn apply_i64_fusion_other(
    out: &mut [i64],
    op: CheapOp,
    chain_left: bool,
    other: I64Operand,
    ext: &[&[i64]],
    base: usize,
) {
    match other {
        I64Operand::Scalar(s) => match (op, chain_left) {
            (CheapOp::Add, _) => out.iter_mut().for_each(|o| *o = o.wrapping_add(s)),
            (CheapOp::Mul, _) => out.iter_mut().for_each(|o| *o = o.wrapping_mul(s)),
            (CheapOp::Sub, true) => out.iter_mut().for_each(|o| *o = o.wrapping_sub(s)),
            (CheapOp::Sub, false) => out.iter_mut().for_each(|o| *o = s.wrapping_sub(*o)),
            (CheapOp::Div, true) => out
                .iter_mut()
                .for_each(|o| *o = o.checked_div(s).unwrap_or(0)),
            (CheapOp::Div, false) => out
                .iter_mut()
                .for_each(|o| *o = s.checked_div(*o).unwrap_or(0)),
            // Integer max/min are commutative and total (no NaN).
            (CheapOp::Max, _) => out.iter_mut().for_each(|o| *o = (*o).max(s)),
            (CheapOp::Min, _) => out.iter_mut().for_each(|o| *o = (*o).min(s)),
            (
                CheapOp::Neg
                | CheapOp::Abs
                | CheapOp::Floor
                | CheapOp::Ceil
                | CheapOp::Trunc
                | CheapOp::RoundAway
                | CheapOp::RoundNearest
                | CheapOp::Sign,
                _,
            ) => {}
        },
        I64Operand::Ext(i) => {
            let sl = &ext[i][base..base + out.len()];
            match (op, chain_left) {
                (CheapOp::Add, _) => out
                    .iter_mut()
                    .zip(sl)
                    .for_each(|(o, e)| *o = o.wrapping_add(*e)),
                (CheapOp::Mul, _) => out
                    .iter_mut()
                    .zip(sl)
                    .for_each(|(o, e)| *o = o.wrapping_mul(*e)),
                (CheapOp::Sub, true) => out
                    .iter_mut()
                    .zip(sl)
                    .for_each(|(o, e)| *o = o.wrapping_sub(*e)),
                (CheapOp::Sub, false) => out
                    .iter_mut()
                    .zip(sl)
                    .for_each(|(o, e)| *o = e.wrapping_sub(*o)),
                (CheapOp::Div, true) => out
                    .iter_mut()
                    .zip(sl)
                    .for_each(|(o, e)| *o = o.checked_div(*e).unwrap_or(0)),
                (CheapOp::Div, false) => out
                    .iter_mut()
                    .zip(sl)
                    .for_each(|(o, e)| *o = e.checked_div(*o).unwrap_or(0)),
                (CheapOp::Max, _) => out.iter_mut().zip(sl).for_each(|(o, e)| *o = (*o).max(*e)),
                (CheapOp::Min, _) => out.iter_mut().zip(sl).for_each(|(o, e)| *o = (*o).min(*e)),
                (
                    CheapOp::Neg
                    | CheapOp::Abs
                    | CheapOp::Floor
                    | CheapOp::Ceil
                    | CheapOp::Trunc
                    | CheapOp::RoundAway
                    | CheapOp::RoundNearest
                    | CheapOp::Sign,
                    _,
                ) => {}
            }
        }
        I64Operand::RowBroadcast(i) => {
            apply_i64_row_broadcast_other(out, op, chain_left, ext[i], base);
        }
        I64Operand::ColBroadcast { idx, cols } => {
            apply_i64_col_broadcast_other(out, op, chain_left, ext[idx], cols, base);
        }
        I64Operand::Chain => {}
    }
}

/// Evaluate the fused i64 run over one chunk `out` (already sized to the chunk len).
#[allow(clippy::eq_op)] // Chain/Chain must execute x-x and checked_div(x,x) faithfully.
fn apply_i64_fusion_chunk(out: &mut [i64], tape: &[I64Step], ext: &[&[i64]], base: usize) {
    // Step 0 has no chain operand: seed `out` from operand `a`, then apply `b`.
    let s0 = &tape[0];
    match s0.a {
        I64Operand::Ext(i) => out.copy_from_slice(&ext[i][base..base + out.len()]),
        I64Operand::RowBroadcast(i) => seed_i64_row_broadcast(out, ext[i], base),
        I64Operand::ColBroadcast { idx, cols } => seed_i64_col_broadcast(out, ext[idx], cols, base),
        I64Operand::Scalar(v) => out.fill(v),
        I64Operand::Chain => {}
    }
    match s0.op {
        op if op.is_unary() => out.iter_mut().for_each(|o| *o = i64_fused_unary(op, *o)),
        _ => apply_i64_fusion_other(out, s0.op, true, s0.b, ext, base),
    }
    for step in &tape[1..] {
        match step.op {
            op if op.is_unary() => {
                out.iter_mut().for_each(|o| *o = i64_fused_unary(op, *o));
                continue;
            }
            _ => {}
        }
        match (step.a, step.b) {
            (I64Operand::Chain, I64Operand::Chain) => match step.op {
                CheapOp::Add => out.iter_mut().for_each(|o| *o = o.wrapping_add(*o)),
                CheapOp::Sub => out.iter_mut().for_each(|o| *o = o.wrapping_sub(*o)),
                CheapOp::Mul => out.iter_mut().for_each(|o| *o = o.wrapping_mul(*o)),
                CheapOp::Div => out
                    .iter_mut()
                    .for_each(|o| *o = o.checked_div(*o).unwrap_or(0)),
                // max(x,x)==x, min(x,x)==x.
                CheapOp::Max | CheapOp::Min => {}
                CheapOp::Neg
                | CheapOp::Abs
                | CheapOp::Floor
                | CheapOp::Ceil
                | CheapOp::Trunc
                | CheapOp::RoundAway
                | CheapOp::RoundNearest
                | CheapOp::Sign => {}
            },
            (I64Operand::Chain, other) => {
                apply_i64_fusion_other(out, step.op, true, other, ext, base)
            }
            (other, I64Operand::Chain) => {
                apply_i64_fusion_other(out, step.op, false, other, ext, base)
            }
            _ => {} // unreachable for a chain step
        }
    }
}

#[allow(clippy::while_let_loop)] // Mirrors the f64 fusion scanner's ordered bailouts.
fn try_fuse_elementwise_chain_i64(
    jaxpr: &Jaxpr,
    start: usize,
    env: &[Option<Value>],
    last_use: &[usize],
) -> Option<FusedRun> {
    let eqns = &jaxpr.equations;
    let mut ext: Vec<&[i64]> = Vec::new();
    let mut ext_vars: Vec<VarId> = Vec::new();
    let mut tape: Vec<I64Step> = Vec::new();
    let mut shape: Option<Shape> = None;
    let mut chain_var: Option<VarId> = None;
    let mut run_out: Option<VarId> = None;
    let mut run_end = start;

    let mut k = start;
    loop {
        let Some(eqn) = eqns.get(k) else { break };
        if !eqn.sub_jaxprs.is_empty() || eqn.outputs.len() != 1 {
            break;
        }
        let ext_mark = ext.len();
        let vars_mark = ext_vars.len();
        // Square(x) fused as Mul(x, x): eval_square's i64 int_op is `wrapping_mul(x, x)`
        // and the i64 fusion Mul uses wrapping_mul, so this is bit-identical (see the
        // f64 builder for the general proof).
        let (op, a, b) = if eqn.primitive == Primitive::Square && eqn.params.is_empty()
            || is_integer_pow_2(eqn.primitive, &eqn.params)
        {
            if eqn.inputs.len() != 1 {
                break;
            }
            let Some(a) = classify_i64_fusion_operand(
                &eqn.inputs[0],
                chain_var,
                env,
                &mut ext,
                &mut ext_vars,
                &mut shape,
            ) else {
                ext.truncate(ext_mark);
                ext_vars.truncate(vars_mark);
                break;
            };
            (CheapOp::Mul, a, a)
        } else {
            let Some(op) = cheap_i64_op(eqn.primitive, &eqn.params) else {
                break;
            };
            let needed = if op.is_unary() { 1 } else { 2 };
            if eqn.inputs.len() != needed {
                break;
            }
            let a = classify_i64_fusion_operand(
                &eqn.inputs[0],
                chain_var,
                env,
                &mut ext,
                &mut ext_vars,
                &mut shape,
            );
            let b = if op.is_unary() {
                Some(I64Operand::Scalar(0))
            } else {
                classify_i64_fusion_operand(
                    &eqn.inputs[1],
                    chain_var,
                    env,
                    &mut ext,
                    &mut ext_vars,
                    &mut shape,
                )
            };
            let (Some(a), Some(b)) = (a, b) else {
                ext.truncate(ext_mark);
                ext_vars.truncate(vars_mark);
                break;
            };
            (op, a, b)
        };
        // Steps after the first MUST thread the chain (one operand == Chain).
        if chain_var.is_some() && !matches!(a, I64Operand::Chain) && !matches!(b, I64Operand::Chain)
        {
            ext.truncate(ext_mark);
            ext_vars.truncate(vars_mark);
            break;
        }
        tape.push(I64Step { op, a, b });
        run_out = Some(eqn.outputs[0]);
        run_end = k;
        let out_idx = eqn.outputs[0].0 as usize;
        if k + 1 < eqns.len() && last_use.get(out_idx).copied() == Some(k + 1) {
            chain_var = Some(eqn.outputs[0]);
            k += 1;
        } else {
            break;
        }
    }

    if tape.len() < FUSION_MIN_RUN {
        return None;
    }
    let shape = shape?;
    let n = shape.element_count()? as usize;
    if n < FUSION_MIN_ELEMS {
        return None;
    }
    let out_var = run_out?;

    let mut values = vec![0_i64; n];
    drive_fusion_chunks(&mut values, |chunk, base| {
        apply_i64_fusion_chunk(chunk, &tape, &ext, base)
    });
    Some(FusedRun {
        out_var,
        values: FusedValues::I64(values),
        shape,
        ext_vars,
        run_end,
    })
}

// ── half-float (BF16/F16) elementwise fusion ───────────────────────────────
//
// The half-precision sibling of the f64/f32/i64 fusion paths. A maximal run of
// SAME-half-dtype F16 same-shape dense cheap-elementwise equations is evaluated
// in ONE chunked pass over a `u16` working buffer, materializing only the final
// output and skipping the N-1 intermediate half tensors. BF16 deliberately
// falls through to the ordinary primitive chain: the current widen/round fused
// tape is profile-proven slower than the materialized path, so BF16 needs a
// conversion-saving primitive rather than this per-step emulation.
//
// BIT-IDENTICAL to running the ops separately: every fused step reproduces the
// EXACT half semantics fj-lax's per-op path uses — widen each operand to f64
// (`Literal::{BF16,F16}Bits::as_f64`), apply the f64 closure (`a OP b`,
// `jax_max/min`, `-x`, `|x|`), then round to half via `Literal::from_{bf16,f16}_f64`
// (round-to-odd f64→f32→half). Each intermediate is a real half value, so widening
// the chain operand reproduces the materialized tensor exactly; same operand order,
// same per-element op, same rounding → identical bits incl. inf/NaN.

/// An operand of a fused half step: the running chain value (half bits in `out`),
/// an external dense-half tensor (index into the gathered `ext` slices), or a
/// half scalar constant (raw `u16` bits — widened once at apply time).
#[derive(Clone, Copy)]
enum HalfOperand {
    Chain,
    Ext(usize),
    Scalar(u16),
}

struct HalfStep {
    op: CheapOp,
    a: HalfOperand,
    b: HalfOperand, // unused for Neg/Abs
}

/// Widen a half (BF16/F16) bit pattern to f64, exactly as `half_binary_apply`
/// in fj-lax does (`Literal::{BF16,F16}Bits.as_f64()`).
#[inline]
fn half_fusion_widen(dt: DType, bits: u16) -> f64 {
    if dt == DType::BF16 {
        Literal::BF16Bits(bits)
    } else {
        Literal::F16Bits(bits)
    }
    .as_f64()
    .unwrap_or(0.0)
}

/// Round an f64 result back to half bits, exactly as fj-lax rounds
/// (`Literal::from_{bf16,f16}_f64` — round-to-odd f64→f32 then →half).
#[inline]
fn half_fusion_round(dt: DType, value: f64) -> u16 {
    let rounded = if dt == DType::BF16 {
        Literal::from_bf16_f64(value)
    } else {
        Literal::from_f16_f64(value)
    };
    match rounded {
        Literal::BF16Bits(b) | Literal::F16Bits(b) => b,
        _ => 0,
    }
}

/// One fused half binary op on already-widened f64 operands, rounding to half.
/// `left`/`right` are pre-widened so the chain operand (read once per element)
/// and the other operand (widened once for a scalar / per-element for a tensor)
/// share the same f64 arithmetic as the per-op path.
#[inline]
fn half_fused_binary(dt: DType, op: CheapOp, left: f64, right: f64) -> u16 {
    let result = match op {
        CheapOp::Add => left + right,
        CheapOp::Sub => left - right,
        CheapOp::Mul => left * right,
        CheapOp::Div => left / right,
        CheapOp::Max => fused_jax_max(left, right),
        CheapOp::Min => fused_jax_min(left, right),
        // Unary ops are handled by the chunk driver; never a binary step.
        CheapOp::Neg
        | CheapOp::Abs
        | CheapOp::Floor
        | CheapOp::Ceil
        | CheapOp::Trunc
        | CheapOp::RoundAway
        | CheapOp::RoundNearest
        | CheapOp::Sign => left,
    };
    half_fusion_round(dt, result)
}

/// Classify one half operand atom. Pushes external dense-half tensors into
/// `ext`/`ext_vars` and sets/checks both the run shape AND the run's half dtype
/// (`half_dt`): the first half operand fixes BF16-vs-F16, and any operand of the
/// other half dtype bails (mixed BF16+F16 promotes to F32 → different path).
/// Returns `None` (bail) for anything not fuse-eligible. Same-shape only (no
/// broadcast — mirrors the i64 path's tight v1 scope).
fn classify_half_fusion_operand<'e>(
    atom: &Atom,
    chain: Option<VarId>,
    env: &'e [Option<Value>],
    ext: &mut Vec<&'e [u16]>,
    ext_vars: &mut Vec<VarId>,
    shape: &mut Option<Shape>,
    half_dt: &mut Option<DType>,
) -> Option<HalfOperand> {
    // Confirm an operand's half dtype matches the run's (or set it). Returns
    // `None` on a half/half mismatch.
    fn match_half_dt(half_dt: &mut Option<DType>, dt: DType) -> Option<()> {
        if dt == DType::BF16 {
            return None;
        }
        match half_dt {
            None => {
                *half_dt = Some(dt);
                Some(())
            }
            Some(existing) if *existing == dt => Some(()),
            Some(_) => None,
        }
    }
    match atom {
        Atom::Lit(Literal::BF16Bits(b)) => {
            match_half_dt(half_dt, DType::BF16)?;
            Some(HalfOperand::Scalar(*b))
        }
        Atom::Lit(Literal::F16Bits(b)) => {
            match_half_dt(half_dt, DType::F16)?;
            Some(HalfOperand::Scalar(*b))
        }
        Atom::Lit(_) => None,
        Atom::Var(v) => {
            if chain == Some(*v) {
                return Some(HalfOperand::Chain);
            }
            let value = env.get(v.0 as usize).and_then(|s| s.as_ref())?;
            match value {
                Value::Scalar(Literal::BF16Bits(b)) => {
                    match_half_dt(half_dt, DType::BF16)?;
                    Some(HalfOperand::Scalar(*b))
                }
                Value::Scalar(Literal::F16Bits(b)) => {
                    match_half_dt(half_dt, DType::F16)?;
                    Some(HalfOperand::Scalar(*b))
                }
                Value::Scalar(_) => None,
                Value::Tensor(t) => {
                    if !matches!(t.dtype, DType::BF16 | DType::F16) {
                        return None;
                    }
                    match_half_dt(half_dt, t.dtype)?;
                    let slice = t.elements.as_half_float_slice()?;
                    match shape {
                        None => *shape = Some(t.shape.clone()),
                        Some(s) if *s == t.shape => {}
                        Some(_) => return None,
                    }
                    let idx = ext.len();
                    ext.push(slice);
                    ext_vars.push(*v);
                    Some(HalfOperand::Ext(idx))
                }
            }
        }
    }
}

/// Apply one half chain step's NON-chain operand to the chunk buffer (the chain's
/// half bits live in `out`). Each arm widens the chain element, applies the op in
/// f64, and rounds back to half — exactly the per-op path.
#[inline]
fn apply_half_fusion_other(
    dt: DType,
    out: &mut [u16],
    op: CheapOp,
    chain_left: bool,
    other: HalfOperand,
    ext: &[&[u16]],
    base: usize,
) {
    match other {
        HalfOperand::Scalar(bits) => {
            let s = half_fusion_widen(dt, bits);
            match chain_left {
                true => out
                    .iter_mut()
                    .for_each(|o| *o = half_fused_binary(dt, op, half_fusion_widen(dt, *o), s)),
                false => out
                    .iter_mut()
                    .for_each(|o| *o = half_fused_binary(dt, op, s, half_fusion_widen(dt, *o))),
            }
        }
        HalfOperand::Ext(i) => {
            let sl = &ext[i][base..base + out.len()];
            match chain_left {
                true => out.iter_mut().zip(sl).for_each(|(o, e)| {
                    *o = half_fused_binary(
                        dt,
                        op,
                        half_fusion_widen(dt, *o),
                        half_fusion_widen(dt, *e),
                    )
                }),
                false => out.iter_mut().zip(sl).for_each(|(o, e)| {
                    *o = half_fused_binary(
                        dt,
                        op,
                        half_fusion_widen(dt, *e),
                        half_fusion_widen(dt, *o),
                    )
                }),
            }
        }
        HalfOperand::Chain => {}
    }
}

/// Apply a half unary op (Neg/Abs) in place: widen → f64 op → round to half.
#[inline]
fn apply_half_fusion_unary(dt: DType, out: &mut [u16], op: CheapOp) {
    match op {
        op if op.is_unary() => out.iter_mut().for_each(|o| {
            *o = half_fusion_round(dt, f64_fused_unary(op, half_fusion_widen(dt, *o)))
        }),
        _ => {}
    }
}

/// Evaluate the fused half run over one chunk `out` (already sized to the chunk len).
#[allow(clippy::eq_op)] // Chain/Chain must execute x-x and x/x to preserve NaN behavior.
fn apply_half_fusion_chunk(
    dt: DType,
    out: &mut [u16],
    tape: &[HalfStep],
    ext: &[&[u16]],
    base: usize,
) {
    // Step 0 has no chain operand: seed `out` from operand `a` (raw half bits),
    // then apply `b`.
    let s0 = &tape[0];
    match s0.a {
        HalfOperand::Ext(i) => out.copy_from_slice(&ext[i][base..base + out.len()]),
        HalfOperand::Scalar(bits) => out.fill(bits),
        HalfOperand::Chain => {}
    }
    match s0.op {
        op if op.is_unary() => apply_half_fusion_unary(dt, out, op),
        _ => apply_half_fusion_other(dt, out, s0.op, true, s0.b, ext, base),
    }
    for step in &tape[1..] {
        if step.op.is_unary() {
            apply_half_fusion_unary(dt, out, step.op);
            continue;
        }
        match (step.a, step.b) {
            (HalfOperand::Chain, HalfOperand::Chain) => out.iter_mut().for_each(|o| {
                let v = half_fusion_widen(dt, *o);
                *o = half_fused_binary(dt, step.op, v, v);
            }),
            (HalfOperand::Chain, other) => {
                apply_half_fusion_other(dt, out, step.op, true, other, ext, base)
            }
            (other, HalfOperand::Chain) => {
                apply_half_fusion_other(dt, out, step.op, false, other, ext, base)
            }
            _ => {} // unreachable for a chain step
        }
    }
}

#[allow(clippy::while_let_loop)] // Mirrors the f64 fusion scanner's ordered bailouts.
fn try_fuse_elementwise_chain_half(
    jaxpr: &Jaxpr,
    start: usize,
    env: &[Option<Value>],
    last_use: &[usize],
) -> Option<FusedRun> {
    let eqns = &jaxpr.equations;
    let mut ext: Vec<&[u16]> = Vec::new();
    let mut ext_vars: Vec<VarId> = Vec::new();
    let mut tape: Vec<HalfStep> = Vec::new();
    let mut shape: Option<Shape> = None;
    let mut half_dt: Option<DType> = None;
    let mut chain_var: Option<VarId> = None;
    let mut run_out: Option<VarId> = None;
    let mut run_end = start;

    let mut k = start;
    loop {
        let Some(eqn) = eqns.get(k) else { break };
        if !eqn.sub_jaxprs.is_empty() || eqn.outputs.len() != 1 {
            break;
        }
        let ext_mark = ext.len();
        let vars_mark = ext_vars.len();
        let dt_mark = half_dt;
        // Square(x) fused as Mul(x, x): eval_square's half arm decodes to f64, computes
        // x*x, and re-encodes — identical to eval_mul's half arm — so duplicating the
        // operand into a Mul step is bit-identical (see the f64 builder for the proof).
        let (op, a, b) = if eqn.primitive == Primitive::Square && eqn.params.is_empty()
            || is_integer_pow_2(eqn.primitive, &eqn.params)
        {
            if eqn.inputs.len() != 1 {
                break;
            }
            let Some(a) = classify_half_fusion_operand(
                &eqn.inputs[0],
                chain_var,
                env,
                &mut ext,
                &mut ext_vars,
                &mut shape,
                &mut half_dt,
            ) else {
                ext.truncate(ext_mark);
                ext_vars.truncate(vars_mark);
                half_dt = dt_mark;
                break;
            };
            (CheapOp::Mul, a, a)
        } else {
            let Some(op) = cheap_float_op(eqn.primitive, &eqn.params) else {
                break;
            };
            let needed = if op.is_unary() { 1 } else { 2 };
            if eqn.inputs.len() != needed {
                break;
            }
            let a = classify_half_fusion_operand(
                &eqn.inputs[0],
                chain_var,
                env,
                &mut ext,
                &mut ext_vars,
                &mut shape,
                &mut half_dt,
            );
            let b = if op.is_unary() {
                // Unary ops carry no second operand; the chunk driver ignores `b`.
                Some(HalfOperand::Chain)
            } else {
                classify_half_fusion_operand(
                    &eqn.inputs[1],
                    chain_var,
                    env,
                    &mut ext,
                    &mut ext_vars,
                    &mut shape,
                    &mut half_dt,
                )
            };
            let (Some(a), Some(b)) = (a, b) else {
                ext.truncate(ext_mark);
                ext_vars.truncate(vars_mark);
                half_dt = dt_mark;
                break;
            };
            (op, a, b)
        };
        // Steps after the first MUST thread the chain (one operand == Chain).
        if chain_var.is_some()
            && !matches!(a, HalfOperand::Chain)
            && !matches!(b, HalfOperand::Chain)
        {
            ext.truncate(ext_mark);
            ext_vars.truncate(vars_mark);
            half_dt = dt_mark;
            break;
        }
        tape.push(HalfStep { op, a, b });
        run_out = Some(eqn.outputs[0]);
        run_end = k;
        let out_idx = eqn.outputs[0].0 as usize;
        if k + 1 < eqns.len() && last_use.get(out_idx).copied() == Some(k + 1) {
            chain_var = Some(eqn.outputs[0]);
            k += 1;
        } else {
            break;
        }
    }

    if tape.len() < FUSION_MIN_RUN {
        return None;
    }
    let shape = shape?;
    let dt = half_dt?;
    let n = shape.element_count()? as usize;
    if n < FUSION_MIN_ELEMS {
        return None;
    }
    let out_var = run_out?;

    // Half stays SERIAL: same-invocation A/B measured bf16 threading at 1.01x (1M)
    // and 0.99x (4M) — neutral/marginal, NOT a win (run_bf16_thread_ab in
    // eval_fusion_speed). bf16's cost is dominated by per-call half construction +
    // per-op decode/encode rather than the chunk apply, so fanning the apply out
    // buys nothing. f64/f32/i64 are bandwidth-bound in the apply and DO win.
    let mut values = vec![0_u16; n];
    let mut s = 0;
    while s < n {
        let e = (s + FUSION_CHUNK).min(n);
        apply_half_fusion_chunk(dt, &mut values[s..e], &tape, &ext, s);
        s = e;
    }
    Some(FusedRun {
        out_var,
        values: FusedValues::Half { dtype: dt, values },
        shape,
        ext_vars,
        run_end,
    })
}

fn try_fuse_elementwise_chain(
    jaxpr: &Jaxpr,
    start: usize,
    env: &[Option<Value>],
    last_use: &[usize],
) -> Option<FusedRun> {
    try_fuse_elementwise_chain_f64(jaxpr, start, env, last_use)
        .or_else(|| try_fuse_elementwise_chain_f32(jaxpr, start, env, last_use))
        .or_else(|| try_fuse_elementwise_chain_i64(jaxpr, start, env, last_use))
        .or_else(|| try_fuse_elementwise_chain_half(jaxpr, start, env, last_use))
}

/// Precompute the per-equation liveness map (`last_use`) for the dense
/// interpreter. `last_use[slot]` = index of the LAST equation that reads this var
/// as an input (`usize::MAX` = never read; outvars are pinned `usize::MAX` so they
/// survive to the return). This is a pure function of the jaxpr SHAPE (not the
/// argument values), so for a sub-jaxpr re-run many times (a `while`/`scan` body)
/// it is computed ONCE and reused across iterations — see [`run_dense_env`].
fn compute_dense_last_use(jaxpr: &Jaxpr, slots: usize) -> Vec<usize> {
    let mut last_use: Vec<usize> = vec![usize::MAX; slots];
    for (i, eqn) in jaxpr.equations.iter().enumerate() {
        for atom in &eqn.inputs {
            if let Atom::Var(var) = atom {
                // Bounds-guard: a malformed jaxpr can reference an id beyond the
                // defined-var range; resolution below reports MissingVariable for it,
                // so just skip it here (mirrors the original `env.get(..)`-safe lookup).
                if let Some(slot) = last_use.get_mut(var.0 as usize) {
                    *slot = i;
                }
            }
        }
    }
    for var in &jaxpr.outvars {
        if let Some(slot) = last_use.get_mut(var.0 as usize) {
            *slot = usize::MAX; // pin: returned, never free
        }
    }
    last_use
}

// Op tag shared by the scalar-arith plans. Max/Min use JAX's
// NaN-propagating semantics for floats (canonical NaN if either operand is NaN;
// see jax_max_f64 / jax_min_f64 in fj-lax) and plain Rust max/min for i64. Neg/Abs
// are unary and carry no rhs operand, avoiding a duplicate slot read in tight
// scalar loops.
#[derive(Clone, Copy)]
enum ScalarF64BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    Neg,
    Abs,
    // Binary transcendentals. For a REAL f64 scalar the generic dispatch is
    // `eval_binary_elementwise(.., f64::powf | f64::atan2)` (fj-lax/src/lib.rs Pow/Atan2
    // arms) → `binary_literal_op` float arm = `Literal::from_f64(f64::FUNC(x, y))`, so
    // the same `f64::FUNC` here is bit-identical. Float-only: kept OUT of the i64 plan
    // (whose Pow/Atan2 use a float-powf/atan2-cast-to-i64) via `scalar_int_binary_op`.
    Pow,
    Atan2,
    // Unary transcendental / rounding ops. For a REAL f64 scalar these all reduce
    // to `Literal::from_f64(f64::FUNC(x))` in the generic interpreter
    // (`eval_unary_elementwise` scalar arm, reached via `eval_exp`/`eval_log`/… →
    // `eval_unary_elementwise_parallel` which is serial for scalars), so computing
    // the same `f64::FUNC` here is BIT-IDENTICAL. This unlocks scalar activation
    // bodies (sigmoid `exp(x)/(1+exp(x))`, softplus, tanh-gelu/silu, …) that
    // otherwise fall back to the tree-walking interpreter.
    Exp,
    Log,
    Log2,
    Exp2,
    Expm1,
    Log1p,
    Sqrt,
    Rsqrt,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Floor,
    Ceil,
    Trunc,
    Deg2Rad,
    Rad2Deg,
    // Power ops. `Square` (jnp.square, no param) is `x*x` in fj-lax's float arm
    // (eval_unary_int_or_float float_op `|x| x*x`). `IntegerPow(e)` (jnp `x**const`,
    // the ubiquitous variance/norm squaring) carries its i32 exponent inline; fj-lax's
    // eval_integer_pow float arm is `real_literal_from_f64(dtype, value.powi(e))`, so
    // `lhs.powi(e)` here is bit-identical. Both float-only (kept out of the i64 plan:
    // i64 Square uses plain `x*x` debug-overflow semantics and i64 IntegerPow uses
    // wrapping_pow — neither modeled here).
    Square,
    IntegerPow(i32),
    // More plain-`f64::FUNC` unary ops (fj-lax float arms): Reciprocal `1/x`,
    // Cbrt `f64::cbrt`, Logistic/sigmoid `1/(1+e^-x)`, and Sign (NaN→NaN, ±0→±0,
    // else `signum`). All float-only; Sign's i64 path (signum) and the others'
    // non-float dtypes stay on the generic interpreter.
    Reciprocal,
    Cbrt,
    Logistic,
    Sign,
}

/// Sign for a real f64 scalar, matching fj-lax `eval_unary_int_or_float`'s float arm
/// (`|x| if x.is_nan() { NaN } else if x == 0.0 { x } else { x.signum() }`): propagates
/// NaN and preserves the ±0 input (`x == 0.0` is true for both +0 and -0).
#[inline]
fn scalar_f64_sign(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x == 0.0 {
        x
    } else {
        x.signum()
    }
}

fn scalar_unary_op(primitive: Primitive) -> Option<ScalarF64BinaryOp> {
    match primitive {
        Primitive::Neg => Some(ScalarF64BinaryOp::Neg),
        Primitive::Abs => Some(ScalarF64BinaryOp::Abs),
        // Bit-identical to the generic scalar path (plain `f64::FUNC`, no params,
        // no complex — the arena is f64-only). Kept in lockstep with the
        // `eval_unary_elementwise_parallel(.., f64::FUNC)` arms in fj-lax/src/lib.rs.
        Primitive::Exp => Some(ScalarF64BinaryOp::Exp),
        Primitive::Log => Some(ScalarF64BinaryOp::Log),
        Primitive::Log2 => Some(ScalarF64BinaryOp::Log2),
        Primitive::Exp2 => Some(ScalarF64BinaryOp::Exp2),
        Primitive::Expm1 => Some(ScalarF64BinaryOp::Expm1),
        Primitive::Log1p => Some(ScalarF64BinaryOp::Log1p),
        Primitive::Sqrt => Some(ScalarF64BinaryOp::Sqrt),
        Primitive::Rsqrt => Some(ScalarF64BinaryOp::Rsqrt),
        Primitive::Sin => Some(ScalarF64BinaryOp::Sin),
        Primitive::Cos => Some(ScalarF64BinaryOp::Cos),
        Primitive::Tan => Some(ScalarF64BinaryOp::Tan),
        Primitive::Asin => Some(ScalarF64BinaryOp::Asin),
        Primitive::Acos => Some(ScalarF64BinaryOp::Acos),
        Primitive::Atan => Some(ScalarF64BinaryOp::Atan),
        Primitive::Sinh => Some(ScalarF64BinaryOp::Sinh),
        Primitive::Cosh => Some(ScalarF64BinaryOp::Cosh),
        Primitive::Tanh => Some(ScalarF64BinaryOp::Tanh),
        Primitive::Asinh => Some(ScalarF64BinaryOp::Asinh),
        Primitive::Acosh => Some(ScalarF64BinaryOp::Acosh),
        Primitive::Atanh => Some(ScalarF64BinaryOp::Atanh),
        Primitive::Floor => Some(ScalarF64BinaryOp::Floor),
        Primitive::Ceil => Some(ScalarF64BinaryOp::Ceil),
        Primitive::Trunc => Some(ScalarF64BinaryOp::Trunc),
        Primitive::Deg2Rad => Some(ScalarF64BinaryOp::Deg2Rad),
        Primitive::Rad2Deg => Some(ScalarF64BinaryOp::Rad2Deg),
        Primitive::Square => Some(ScalarF64BinaryOp::Square),
        Primitive::Reciprocal => Some(ScalarF64BinaryOp::Reciprocal),
        Primitive::Cbrt => Some(ScalarF64BinaryOp::Cbrt),
        Primitive::Logistic => Some(ScalarF64BinaryOp::Logistic),
        Primitive::Sign => Some(ScalarF64BinaryOp::Sign),
        _ => None,
    }
}

/// Float-plan unary resolver that also admits the ONE param-carrying scalar op,
/// `IntegerPow` (exponent param). All other unary ops must have empty params (the
/// generic interpreter ignores none, so a stray param would mean a different op).
/// Reads the SAME `exponent` key as fj-lax `eval_integer_pow`, so present→bit-identical,
/// absent→bail to generic (which also needs it).
fn scalar_f64_unary_op_with_params(
    primitive: Primitive,
    params: &std::collections::BTreeMap<String, String>,
) -> Option<ScalarF64BinaryOp> {
    if primitive == Primitive::IntegerPow {
        let exponent: i32 = params.get("exponent")?.trim().parse().ok()?;
        return Some(ScalarF64BinaryOp::IntegerPow(exponent));
    }
    if !params.is_empty() {
        return None;
    }
    scalar_unary_op(primitive)
}

/// Integer-valid unary ops only. The transcendental/rounding variants of
/// `ScalarF64BinaryOp` are float-only (no integer JAX primitive produces them),
/// so the i64 scalar plan must never admit them — keeping `apply_scalar_i64_binary`
/// reachable only for the integer arms.
fn scalar_int_unary_op(primitive: Primitive) -> Option<ScalarF64BinaryOp> {
    match primitive {
        Primitive::Neg => Some(ScalarF64BinaryOp::Neg),
        Primitive::Abs => Some(ScalarF64BinaryOp::Abs),
        _ => None,
    }
}

#[derive(Clone, Copy)]
enum ScalarF64Operand {
    Slot(usize),
    Literal(f64),
}

struct ScalarF64Step {
    op: ScalarF64BinaryOp,
    lhs: ScalarF64Operand,
    rhs: Option<ScalarF64Operand>,
    out_slot: usize,
}

struct ScalarF64Plan {
    slots: usize,
    const_slots: Vec<usize>,
    input_slots: Vec<usize>,
    out_slots: Vec<usize>,
    steps: Vec<ScalarF64Step>,
}

fn scalar_f64_plan_single_tensor_linear_chain_elems(
    plan: &ScalarF64Plan,
    args: &[Value],
) -> Option<usize> {
    if !plan.const_slots.is_empty()
        || plan.input_slots.len() != args.len()
        || plan.out_slots.len() != 1
        || plan.steps.is_empty()
    {
        return None;
    }

    let mut tensor_slot = None;
    let mut elems = None;
    let mut scalar_slots = Vec::new();
    for (&slot, arg) in plan.input_slots.iter().zip(args) {
        match arg {
            Value::Scalar(Literal::F64Bits(_)) => scalar_slots.push(slot),
            Value::Tensor(tensor) => {
                let values = tensor.elements.as_f64_slice()?;
                if tensor_slot.replace(slot).is_some() {
                    return None;
                }
                elems = Some(values.len());
            }
            _ => return None,
        }
    }

    let mut current_slot = tensor_slot?;
    for step in &plan.steps {
        let lhs_is_current = operand_is_slot(step.lhs, current_slot);
        match step.rhs {
            Some(rhs) => {
                let rhs_is_current = operand_is_slot(rhs, current_slot);
                let other = match (lhs_is_current, rhs_is_current) {
                    (true, true) => None,
                    (true, false) => Some(rhs),
                    (false, true) => Some(step.lhs),
                    (false, false) => return None,
                };
                if let Some(ScalarF64Operand::Slot(slot)) = other
                    && !scalar_slots.contains(&slot)
                {
                    return None;
                }
            }
            None if !lhs_is_current => return None,
            None => {}
        }
        current_slot = step.out_slot;
    }

    (plan.out_slots[0] == current_slot).then_some(elems?)
}

/// The four real full-reduction ops the dense plan covers. Each maps to a
/// (seed, fold) pair that mirrors fj-lax's full-reduce exactly (see
/// [`run_dense_f64_reduce_sum_plan_into`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum DenseReduceOp {
    Sum,
    Prod,
    Max,
    Min,
}

impl DenseReduceOp {
    fn from_primitive(primitive: Primitive) -> Option<Self> {
        match primitive {
            Primitive::ReduceSum => Some(Self::Sum),
            Primitive::ReduceProd => Some(Self::Prod),
            Primitive::ReduceMax => Some(Self::Max),
            Primitive::ReduceMin => Some(Self::Min),
            _ => None,
        }
    }

    fn seed(self) -> f64 {
        match self {
            Self::Sum => 0.0,
            Self::Prod => 1.0,
            Self::Max => f64::NEG_INFINITY,
            Self::Min => f64::INFINITY,
        }
    }

    fn fold(self, acc: f64, value: f64) -> f64 {
        match self {
            Self::Sum => acc + value,
            Self::Prod => acc * value,
            Self::Max => jax_max_f64(acc, value),
            Self::Min => jax_min_f64(acc, value),
        }
    }
}

struct DenseF64ReduceSumPlan {
    const_slots: Vec<usize>,
    input_slots: Vec<usize>,
    input_slot: usize,
    op: DenseReduceOp,
}

struct DenseF64DotPlan {
    const_slots: Vec<usize>,
    input_slots: Vec<usize>,
    lhs_slot: usize,
    rhs_slot: usize,
}

/// Single-equation dense F64 rank-2 transpose. The fj-lax kernel is pure data
/// movement; this plan pre-parses the optional permutation and dispatches the
/// common `[1,0]` / default-rank-2 reversal without re-entering `eval_primitive`.
/// Scalars, non-F64 tensors, non-rank-2 tensors, invalid permutations, and empty
/// tensors fall through to the generic interpreter.
struct DenseF64TransposePlan {
    const_slots: Vec<usize>,
    input_slots: Vec<usize>,
    input_slot: usize,
    permutation: Option<Vec<usize>>,
}

/// Single-equation dense F64 `broadcast_in_dim` for the common bias-style
/// rank-1 -> rank-2 trailing-axis case (`broadcast_dimensions=[1]`, or the
/// default rank-1-to-rank-2 mapping). The plan pre-parses the target shape and
/// emits each output row with `extend_from_slice`, matching fj-lax's row-major
/// replication without per-element odometer setup. Scalars, other dtypes/ranks,
/// incompatible shapes, and non-trailing mappings fall through to generic eval.
struct DenseF64BroadcastInDimPlan {
    const_slots: Vec<usize>,
    input_slots: Vec<usize>,
    input_slot: usize,
    target_shape: Shape,
    target_count: usize,
    broadcast_dimensions: Option<Vec<usize>>,
}

/// Single-equation dense tensor reshape. Reshape is metadata-only in fj-lax for
/// tensor inputs, so this plan pre-parses `new_shape` once and then clones the
/// existing backing buffer with a new shape tag. Scalars, shape mismatches, and
/// malformed params fall through to the generic interpreter for exact error
/// behavior.
struct DenseReshapePlan {
    const_slots: Vec<usize>,
    input_slots: Vec<usize>,
    input_slot: usize,
    target: DenseReshapeTarget,
}

enum DenseReshapeTarget {
    Static {
        target_shape: Shape,
        target_count: u64,
    },
    Inferred {
        shape_spec: Vec<i64>,
        known_product: u64,
        inferred_axis: usize,
    },
}

impl DenseReshapeTarget {
    fn resolve(&self, input_count: u64) -> Option<(Shape, u64)> {
        match self {
            Self::Static {
                target_shape,
                target_count,
            } => Some((target_shape.clone(), *target_count)),
            Self::Inferred {
                shape_spec,
                known_product,
                inferred_axis,
            } => {
                if *known_product == 0 || !input_count.is_multiple_of(*known_product) {
                    return None;
                }
                let inferred = input_count / *known_product;
                let inferred = u32::try_from(inferred).ok()?;
                let mut dims = Vec::with_capacity(shape_spec.len());
                for (idx, dim) in shape_spec.iter().copied().enumerate() {
                    if idx == *inferred_axis {
                        dims.push(inferred);
                    } else {
                        dims.push(u32::try_from(dim).ok()?);
                    }
                }
                let shape = Shape { dims };
                let target_count = shape.element_count()?;
                Some((shape, target_count))
            }
        }
    }
}

/// Single-equation reduction over ONE trailing axis (`axes=<rank-1>`), e.g.
/// `jnp.sum(x, axis=-1)` / `jnp.mean(x, axis=(-2,-1))` (softmax denominators,
/// global/spatial pooling). When the reduced axes are a CONTIGUOUS TRAILING block
/// every output cell folds one contiguous run, so the typed plan reproduces
/// fj-lax's contiguous-block axis-reduce exactly while skipping the per-call param
/// re-parse + eval_primitive dispatch. The recorded sorted `axes` are validated
/// against the runtime rank (must be exactly the last `axes.len()` axes).
struct DenseAxisReducePlan {
    const_slots: Vec<usize>,
    input_slots: Vec<usize>,
    input_slot: usize,
    op: DenseReduceOp,
    axes: Vec<usize>,
}

/// Single-equation row-gather (`table[indices]` — embedding lookup), the common
/// contiguous-full-row case: `slice_sizes = [1, op_dims[1..]]` so each gathered
/// slice is a whole contiguous row. The default `Clip` index mode (no mode param)
/// clamps OOB indices, never drops, so the output is a pure contiguous copy. The
/// `slice_sizes` are parsed once at build time, skipping the per-call param parse
/// + eval_primitive dispatch + boxed index extraction.
struct DenseGatherPlan {
    const_slots: Vec<usize>,
    input_slots: Vec<usize>,
    operand_slot: usize,
    indices_slot: usize,
    slice_sizes: Vec<usize>,
}

/// Single-equation argmax/argmin over ONE trailing axis (`axis=rank-1`, the
/// default), e.g. `jnp.argmax(logits, axis=-1)` — decode predictions / sampling.
/// Each output cell is the index of the extremum within a contiguous row;
/// `find_max` picks argmax vs argmin, `axis` (None = default trailing) is checked
/// against the runtime rank.
struct DenseArgExtremumPlan {
    const_slots: Vec<usize>,
    input_slots: Vec<usize>,
    input_slot: usize,
    find_max: bool,
    axis: Option<usize>,
}

/// fj-lax's `arg_extreme_float` reducer, replicated bit-for-bit: first NaN wins
/// (and stops), else strict `>`/`<` with first-occurrence tie-break (±0 compare
/// equal so the earlier index is kept). The fj-lax SIMD argmax falls back to this
/// on any-NaN and matches it otherwise, so this is the exact reference.
fn plan_arg_extreme_float(n: usize, find_max: bool, get: impl Fn(usize) -> f64) -> usize {
    let mut best_idx = 0usize;
    let mut best = get(0);
    let mut best_nan = best.is_nan();
    let mut i = 1;
    while i < n && !best_nan {
        let v = get(i);
        if v.is_nan() {
            best_idx = i;
            best_nan = true;
        } else if (find_max && v > best) || (!find_max && v < best) {
            best_idx = i;
            best = v;
        }
        i += 1;
    }
    best_idx
}

#[derive(Clone, Copy)]
enum ScalarF64Slot {
    Missing,
    NonF64,
    F64(f64),
}

fn scalar_f64_binary_op(primitive: Primitive) -> Option<ScalarF64BinaryOp> {
    match primitive {
        Primitive::Add => Some(ScalarF64BinaryOp::Add),
        Primitive::Sub => Some(ScalarF64BinaryOp::Sub),
        Primitive::Mul => Some(ScalarF64BinaryOp::Mul),
        Primitive::Div => Some(ScalarF64BinaryOp::Div),
        Primitive::Max => Some(ScalarF64BinaryOp::Max),
        Primitive::Min => Some(ScalarF64BinaryOp::Min),
        Primitive::Pow => Some(ScalarF64BinaryOp::Pow),
        Primitive::Atan2 => Some(ScalarF64BinaryOp::Atan2),
        _ => None,
    }
}

/// Integer-valid binary ops only — the float-only `Pow`/`Atan2` arms are excluded so
/// they never enter the i64 plan (where the generic dispatch casts a float result to
/// i64, which the integer arena does not model). Mirrors `scalar_int_unary_op`.
fn scalar_int_binary_op(primitive: Primitive) -> Option<ScalarF64BinaryOp> {
    match primitive {
        Primitive::Add => Some(ScalarF64BinaryOp::Add),
        Primitive::Sub => Some(ScalarF64BinaryOp::Sub),
        Primitive::Mul => Some(ScalarF64BinaryOp::Mul),
        Primitive::Div => Some(ScalarF64BinaryOp::Div),
        Primitive::Max => Some(ScalarF64BinaryOp::Max),
        Primitive::Min => Some(ScalarF64BinaryOp::Min),
        _ => None,
    }
}

fn scalar_f64_operand(atom: &Atom, slots: usize) -> Option<ScalarF64Operand> {
    match atom {
        Atom::Var(var) => {
            let slot = var.0 as usize;
            (slot < slots).then_some(ScalarF64Operand::Slot(slot))
        }
        Atom::Lit(Literal::F64Bits(bits)) => Some(ScalarF64Operand::Literal(f64::from_bits(*bits))),
        Atom::Lit(_) => None,
    }
}

fn var_slots(vars: &[VarId], slots: usize) -> Option<Vec<usize>> {
    let mut out = Vec::with_capacity(vars.len());
    for var in vars {
        let slot = var.0 as usize;
        if slot >= slots {
            return None;
        }
        out.push(slot);
    }
    Some(out)
}

fn build_dense_f64_reduce_sum_plan(jaxpr: &Jaxpr, slots: usize) -> Option<DenseF64ReduceSumPlan> {
    if !jaxpr.effects.is_empty() || jaxpr.equations.len() != 1 {
        return None;
    }
    let equation = &jaxpr.equations[0];
    let op = DenseReduceOp::from_primitive(equation.primitive)?;
    if !equation.params.is_empty()
        || !equation.sub_jaxprs.is_empty()
        || !equation.effects.is_empty()
        || equation.inputs.len() != 1
        || equation.outputs.len() != 1
        || jaxpr.outvars.as_slice() != equation.outputs.as_slice()
    {
        return None;
    }

    let Atom::Var(input_var) = equation.inputs[0] else {
        return None;
    };
    let input_slot = input_var.0 as usize;
    let out_slot = equation.outputs[0].0 as usize;
    if input_slot >= slots || out_slot >= slots {
        return None;
    }

    Some(DenseF64ReduceSumPlan {
        const_slots: var_slots(&jaxpr.constvars, slots)?,
        input_slots: var_slots(&jaxpr.invars, slots)?,
        input_slot,
        op,
    })
}

fn build_dense_axis_reduce_plan(jaxpr: &Jaxpr, slots: usize) -> Option<DenseAxisReducePlan> {
    if !jaxpr.effects.is_empty() || jaxpr.equations.len() != 1 {
        return None;
    }
    let equation = &jaxpr.equations[0];
    let op = DenseReduceOp::from_primitive(equation.primitive)?;
    if !equation.sub_jaxprs.is_empty()
        || !equation.effects.is_empty()
        || equation.inputs.len() != 1
        || equation.outputs.len() != 1
        || jaxpr.outvars.as_slice() != equation.outputs.as_slice()
    {
        return None;
    }
    // Require EXACTLY one param, "axes", a non-empty comma list of axis indices.
    // Anything else (keep_dims, extra params, unparseable) bails to generic. The
    // axes are sorted+deduped here; the runtime validates they are a contiguous
    // TRAILING block for the actual rank (the only bit-exact contiguous-fold case).
    if equation.params.len() != 1 {
        return None;
    }
    let raw = equation.params.get("axes")?;
    let mut axes: Vec<usize> = Vec::new();
    for part in raw.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        axes.push(part.parse().ok()?);
    }
    if axes.is_empty() {
        return None;
    }
    axes.sort_unstable();
    axes.dedup();

    let Atom::Var(input_var) = equation.inputs[0] else {
        return None;
    };
    let input_slot = input_var.0 as usize;
    let out_slot = equation.outputs[0].0 as usize;
    if input_slot >= slots || out_slot >= slots {
        return None;
    }

    Some(DenseAxisReducePlan {
        const_slots: var_slots(&jaxpr.constvars, slots)?,
        input_slots: var_slots(&jaxpr.invars, slots)?,
        input_slot,
        op,
        axes,
    })
}

fn build_dense_gather_plan(jaxpr: &Jaxpr, slots: usize) -> Option<DenseGatherPlan> {
    if !jaxpr.effects.is_empty() || jaxpr.equations.len() != 1 {
        return None;
    }
    let equation = &jaxpr.equations[0];
    if equation.primitive != Primitive::Gather
        || !equation.sub_jaxprs.is_empty()
        || !equation.effects.is_empty()
        || equation.inputs.len() != 2
        || equation.outputs.len() != 1
        || jaxpr.outvars.as_slice() != equation.outputs.as_slice()
    {
        return None;
    }
    // Require EXACTLY {"slice_sizes": <list>} — no index_mode param, so the eval
    // uses the default Clip mode (clamp OOB, never drop -> pure contiguous copy).
    // Any extra param bails to generic, preserving its behavior.
    if equation.params.len() != 1 {
        return None;
    }
    let raw = equation.params.get("slice_sizes")?;
    let mut slice_sizes: Vec<usize> = Vec::new();
    for part in raw.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        slice_sizes.push(part.parse().ok()?);
    }
    if slice_sizes.is_empty() || slice_sizes[0] != 1 {
        return None;
    }

    let Atom::Var(operand_var) = equation.inputs[0] else {
        return None;
    };
    let Atom::Var(indices_var) = equation.inputs[1] else {
        return None;
    };
    let operand_slot = operand_var.0 as usize;
    let indices_slot = indices_var.0 as usize;
    let out_slot = equation.outputs[0].0 as usize;
    if operand_slot >= slots || indices_slot >= slots || out_slot >= slots {
        return None;
    }

    Some(DenseGatherPlan {
        const_slots: var_slots(&jaxpr.constvars, slots)?,
        input_slots: var_slots(&jaxpr.invars, slots)?,
        operand_slot,
        indices_slot,
        slice_sizes,
    })
}

fn build_dense_arg_extremum_plan(jaxpr: &Jaxpr, slots: usize) -> Option<DenseArgExtremumPlan> {
    if !jaxpr.effects.is_empty() || jaxpr.equations.len() != 1 {
        return None;
    }
    let equation = &jaxpr.equations[0];
    let find_max = match equation.primitive {
        Primitive::Argmax => true,
        Primitive::Argmin => false,
        _ => return None,
    };
    if !equation.sub_jaxprs.is_empty()
        || !equation.effects.is_empty()
        || equation.inputs.len() != 1
        || equation.outputs.len() != 1
        || jaxpr.outvars.as_slice() != equation.outputs.as_slice()
    {
        return None;
    }
    // Only the `axis` param is allowed (absent => default trailing axis). Any
    // other key bails to the generic interpreter.
    if equation.params.keys().any(|k| k != "axis") {
        return None;
    }
    let axis = match equation.params.get("axis") {
        Some(raw) => Some(raw.trim().parse::<usize>().ok()?),
        None => None,
    };

    let Atom::Var(input_var) = equation.inputs[0] else {
        return None;
    };
    let input_slot = input_var.0 as usize;
    let out_slot = equation.outputs[0].0 as usize;
    if input_slot >= slots || out_slot >= slots {
        return None;
    }

    Some(DenseArgExtremumPlan {
        const_slots: var_slots(&jaxpr.constvars, slots)?,
        input_slots: var_slots(&jaxpr.invars, slots)?,
        input_slot,
        find_max,
        axis,
    })
}

fn build_dense_f64_dot_plan(jaxpr: &Jaxpr, slots: usize) -> Option<DenseF64DotPlan> {
    if !jaxpr.effects.is_empty() || jaxpr.equations.len() != 1 {
        return None;
    }
    let equation = &jaxpr.equations[0];
    if equation.primitive != Primitive::Dot
        || !equation.params.is_empty()
        || !equation.sub_jaxprs.is_empty()
        || !equation.effects.is_empty()
        || equation.inputs.len() != 2
        || equation.outputs.len() != 1
        || jaxpr.outvars.as_slice() != equation.outputs.as_slice()
    {
        return None;
    }

    let (Atom::Var(lhs_var), Atom::Var(rhs_var)) = (&equation.inputs[0], &equation.inputs[1])
    else {
        return None;
    };
    let lhs_slot = lhs_var.0 as usize;
    let rhs_slot = rhs_var.0 as usize;
    let out_slot = equation.outputs[0].0 as usize;
    if lhs_slot >= slots || rhs_slot >= slots || out_slot >= slots {
        return None;
    }

    Some(DenseF64DotPlan {
        const_slots: var_slots(&jaxpr.constvars, slots)?,
        input_slots: var_slots(&jaxpr.invars, slots)?,
        lhs_slot,
        rhs_slot,
    })
}

fn build_dense_f64_transpose_plan(jaxpr: &Jaxpr, slots: usize) -> Option<DenseF64TransposePlan> {
    if !jaxpr.effects.is_empty() || jaxpr.equations.len() != 1 {
        return None;
    }
    let equation = &jaxpr.equations[0];
    if equation.primitive != Primitive::Transpose
        || !equation.sub_jaxprs.is_empty()
        || !equation.effects.is_empty()
        || equation.inputs.len() != 1
        || equation.outputs.len() != 1
        || jaxpr.outvars.as_slice() != equation.outputs.as_slice()
        || equation.params.keys().any(|key| key != "permutation")
    {
        return None;
    }

    let permutation = match equation.params.get("permutation") {
        Some(raw) => {
            let mut parsed = Vec::new();
            for part in raw
                .split(',')
                .map(str::trim)
                .filter(|part| !part.is_empty())
            {
                parsed.push(part.parse::<usize>().ok()?);
            }
            Some(parsed)
        }
        None => None,
    };

    let Atom::Var(input_var) = equation.inputs[0] else {
        return None;
    };
    let input_slot = input_var.0 as usize;
    let out_slot = equation.outputs[0].0 as usize;
    if input_slot >= slots || out_slot >= slots {
        return None;
    }

    Some(DenseF64TransposePlan {
        const_slots: var_slots(&jaxpr.constvars, slots)?,
        input_slots: var_slots(&jaxpr.invars, slots)?,
        input_slot,
        permutation,
    })
}

fn build_dense_f64_broadcast_in_dim_plan(
    jaxpr: &Jaxpr,
    slots: usize,
) -> Option<DenseF64BroadcastInDimPlan> {
    if !jaxpr.effects.is_empty() || jaxpr.equations.len() != 1 {
        return None;
    }
    let equation = &jaxpr.equations[0];
    if equation.primitive != Primitive::BroadcastInDim
        || !equation.sub_jaxprs.is_empty()
        || !equation.effects.is_empty()
        || equation.inputs.len() != 1
        || equation.outputs.len() != 1
        || jaxpr.outvars.as_slice() != equation.outputs.as_slice()
        || equation
            .params
            .keys()
            .any(|key| key != "shape" && key != "broadcast_dimensions")
    {
        return None;
    }

    let raw_shape = equation.params.get("shape")?;
    let mut dims = Vec::new();
    for part in raw_shape
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let dim = part.parse::<i64>().ok()?;
        if dim < 0 {
            return None;
        }
        dims.push(u32::try_from(dim).ok()?);
    }
    if dims.len() != 2 {
        return None;
    }
    let target_shape = Shape { dims };
    let target_count = usize::try_from(target_shape.element_count()?).ok()?;

    let broadcast_dimensions = match equation.params.get("broadcast_dimensions") {
        Some(raw) => {
            let mut parsed = Vec::new();
            for part in raw
                .split(',')
                .map(str::trim)
                .filter(|part| !part.is_empty())
            {
                parsed.push(part.parse::<usize>().ok()?);
            }
            Some(parsed)
        }
        None => None,
    };

    let Atom::Var(input_var) = equation.inputs[0] else {
        return None;
    };
    let input_slot = input_var.0 as usize;
    let out_slot = equation.outputs[0].0 as usize;
    if input_slot >= slots || out_slot >= slots {
        return None;
    }

    Some(DenseF64BroadcastInDimPlan {
        const_slots: var_slots(&jaxpr.constvars, slots)?,
        input_slots: var_slots(&jaxpr.invars, slots)?,
        input_slot,
        target_shape,
        target_count,
        broadcast_dimensions,
    })
}

fn build_dense_reshape_plan(jaxpr: &Jaxpr, slots: usize) -> Option<DenseReshapePlan> {
    if !jaxpr.effects.is_empty() || jaxpr.equations.len() != 1 {
        return None;
    }
    let equation = &jaxpr.equations[0];
    if equation.primitive != Primitive::Reshape
        || !equation.sub_jaxprs.is_empty()
        || !equation.effects.is_empty()
        || equation.inputs.len() != 1
        || equation.outputs.len() != 1
        || jaxpr.outvars.as_slice() != equation.outputs.as_slice()
        || equation.params.len() != 1
    {
        return None;
    }
    let raw = equation.params.get("new_shape")?;
    let mut shape_spec = Vec::new();
    let mut static_dims = Vec::new();
    let mut inferred_axis = None;
    let mut known_product = 1_u64;
    if !raw.trim().is_empty() {
        for (axis, part) in raw
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .enumerate()
        {
            let dim = part.parse::<i64>().ok()?;
            if dim == -1 {
                if inferred_axis.is_some() {
                    return None;
                }
                inferred_axis = Some(axis);
                shape_spec.push(dim);
                continue;
            }
            if dim < 0 {
                return None;
            }
            let dim_u32 = u32::try_from(dim).ok()?;
            known_product = known_product.checked_mul(u64::from(dim_u32))?;
            shape_spec.push(dim);
            static_dims.push(dim_u32);
        }
    }
    let target = if let Some(inferred_axis) = inferred_axis {
        DenseReshapeTarget::Inferred {
            shape_spec,
            known_product,
            inferred_axis,
        }
    } else {
        let target_shape = Shape { dims: static_dims };
        let target_count = target_shape.element_count()?;
        DenseReshapeTarget::Static {
            target_shape,
            target_count,
        }
    };

    let Atom::Var(input_var) = equation.inputs[0] else {
        return None;
    };
    let input_slot = input_var.0 as usize;
    let out_slot = equation.outputs[0].0 as usize;
    if input_slot >= slots || out_slot >= slots {
        return None;
    }

    Some(DenseReshapePlan {
        const_slots: var_slots(&jaxpr.constvars, slots)?,
        input_slots: var_slots(&jaxpr.invars, slots)?,
        input_slot,
        target,
    })
}

fn build_scalar_f64_arith_plan(jaxpr: &Jaxpr, slots: usize) -> Option<ScalarF64Plan> {
    if jaxpr.equations.is_empty() || !jaxpr.effects.is_empty() {
        return None;
    }

    let mut steps = Vec::with_capacity(jaxpr.equations.len());
    for equation in &jaxpr.equations {
        if !equation.sub_jaxprs.is_empty()
            || !equation.effects.is_empty()
            || equation.outputs.len() != 1
        {
            return None;
        }
        let out_slot = equation.outputs[0].0 as usize;
        if out_slot >= slots {
            return None;
        }
        // 2-input binary op (no params), or 1-input unary — `IntegerPow` carries an
        // exponent param; every other unary requires empty params.
        let (op, lhs, rhs) = match equation.inputs.as_slice() {
            [a, b] => {
                if !equation.params.is_empty() {
                    return None;
                }
                (
                    scalar_f64_binary_op(equation.primitive)?,
                    scalar_f64_operand(a, slots)?,
                    Some(scalar_f64_operand(b, slots)?),
                )
            }
            [a] => {
                let operand = scalar_f64_operand(a, slots)?;
                (
                    scalar_f64_unary_op_with_params(equation.primitive, &equation.params)?,
                    operand,
                    None,
                )
            }
            _ => return None,
        };
        steps.push(ScalarF64Step {
            op,
            lhs,
            rhs,
            out_slot,
        });
    }

    Some(ScalarF64Plan {
        slots,
        const_slots: var_slots(&jaxpr.constvars, slots)?,
        input_slots: var_slots(&jaxpr.invars, slots)?,
        out_slots: var_slots(&jaxpr.outvars, slots)?,
        steps,
    })
}

fn scalar_f64_slot_from_value(value: &Value) -> ScalarF64Slot {
    match value {
        Value::Scalar(Literal::F64Bits(bits)) => ScalarF64Slot::F64(f64::from_bits(*bits)),
        Value::Scalar(_) | Value::Tensor(_) => ScalarF64Slot::NonF64,
    }
}

fn read_scalar_f64_operand(
    slots: &[ScalarF64Slot],
    operand: ScalarF64Operand,
) -> Result<Option<f64>, InterpreterError> {
    match operand {
        ScalarF64Operand::Literal(value) => Ok(Some(value)),
        ScalarF64Operand::Slot(slot) => match slots[slot] {
            ScalarF64Slot::F64(value) => Ok(Some(value)),
            ScalarF64Slot::NonF64 => Ok(None),
            ScalarF64Slot::Missing => Err(InterpreterError::MissingVariable(VarId(slot as u32))),
        },
    }
}

fn jax_max_f64(lhs: f64, rhs: f64) -> f64 {
    if lhs.is_nan() || rhs.is_nan() {
        f64::NAN
    } else {
        lhs.max(rhs)
    }
}

fn jax_min_f64(lhs: f64, rhs: f64) -> f64 {
    if lhs.is_nan() || rhs.is_nan() {
        f64::NAN
    } else {
        lhs.min(rhs)
    }
}

fn apply_scalar_f64_binary(op: ScalarF64BinaryOp, lhs: f64, rhs: f64) -> f64 {
    match op {
        ScalarF64BinaryOp::Add => lhs + rhs,
        ScalarF64BinaryOp::Sub => lhs - rhs,
        ScalarF64BinaryOp::Mul => lhs * rhs,
        ScalarF64BinaryOp::Div => lhs / rhs,
        ScalarF64BinaryOp::Max => jax_max_f64(lhs, rhs),
        ScalarF64BinaryOp::Min => jax_min_f64(lhs, rhs),
        ScalarF64BinaryOp::Neg => -lhs,
        ScalarF64BinaryOp::Abs => lhs.abs(),
        ScalarF64BinaryOp::Pow => lhs.powf(rhs),
        ScalarF64BinaryOp::Atan2 => lhs.atan2(rhs),
        // Unary math: `rhs` is a copy of `lhs` (set by the runner for no-rhs steps)
        // and ignored. Each matches the corresponding `f64::FUNC` the generic
        // scalar interpreter applies, so the result bits are identical.
        ScalarF64BinaryOp::Exp => lhs.exp(),
        ScalarF64BinaryOp::Log => lhs.ln(),
        ScalarF64BinaryOp::Log2 => lhs.log2(),
        ScalarF64BinaryOp::Exp2 => lhs.exp2(),
        ScalarF64BinaryOp::Expm1 => lhs.exp_m1(),
        ScalarF64BinaryOp::Log1p => lhs.ln_1p(),
        ScalarF64BinaryOp::Sqrt => lhs.sqrt(),
        ScalarF64BinaryOp::Rsqrt => 1.0 / lhs.sqrt(),
        ScalarF64BinaryOp::Sin => lhs.sin(),
        ScalarF64BinaryOp::Cos => lhs.cos(),
        ScalarF64BinaryOp::Tan => lhs.tan(),
        ScalarF64BinaryOp::Asin => lhs.asin(),
        ScalarF64BinaryOp::Acos => lhs.acos(),
        ScalarF64BinaryOp::Atan => lhs.atan(),
        ScalarF64BinaryOp::Sinh => lhs.sinh(),
        ScalarF64BinaryOp::Cosh => lhs.cosh(),
        ScalarF64BinaryOp::Tanh => lhs.tanh(),
        ScalarF64BinaryOp::Asinh => lhs.asinh(),
        ScalarF64BinaryOp::Acosh => lhs.acosh(),
        ScalarF64BinaryOp::Atanh => lhs.atanh(),
        ScalarF64BinaryOp::Floor => lhs.floor(),
        ScalarF64BinaryOp::Ceil => lhs.ceil(),
        ScalarF64BinaryOp::Trunc => lhs.trunc(),
        ScalarF64BinaryOp::Deg2Rad => lhs.to_radians(),
        ScalarF64BinaryOp::Rad2Deg => lhs.to_degrees(),
        ScalarF64BinaryOp::Square => lhs * lhs,
        ScalarF64BinaryOp::IntegerPow(exp) => lhs.powi(exp),
        ScalarF64BinaryOp::Reciprocal => 1.0 / lhs,
        ScalarF64BinaryOp::Cbrt => lhs.cbrt(),
        ScalarF64BinaryOp::Logistic => 1.0 / (1.0 + (-lhs).exp()),
        ScalarF64BinaryOp::Sign => scalar_f64_sign(lhs),
    }
}

fn run_scalar_f64_arith_plan_into(
    plan: &ScalarF64Plan,
    const_values: &[Value],
    args: &[Value],
    out: &mut Vec<Value>,
    slots: &mut Vec<ScalarF64Slot>,
) -> Option<Result<(), InterpreterError>> {
    if const_values.len() != plan.const_slots.len() {
        return Some(Err(InterpreterError::ConstArity {
            expected: plan.const_slots.len(),
            actual: const_values.len(),
        }));
    }
    if args.len() != plan.input_slots.len() {
        return Some(Err(InterpreterError::InputArity {
            expected: plan.input_slots.len(),
            actual: args.len(),
        }));
    }

    slots.clear();
    slots.resize(plan.slots, ScalarF64Slot::Missing);
    for (&slot, value) in plan.const_slots.iter().zip(const_values) {
        slots[slot] = scalar_f64_slot_from_value(value);
    }
    for (&slot, value) in plan.input_slots.iter().zip(args) {
        slots[slot] = scalar_f64_slot_from_value(value);
    }

    for step in &plan.steps {
        let lhs = match read_scalar_f64_operand(slots, step.lhs) {
            Ok(Some(value)) => value,
            Ok(None) => return None,
            Err(error) => return Some(Err(error)),
        };
        let rhs = match step.rhs {
            Some(rhs) => match read_scalar_f64_operand(slots, rhs) {
                Ok(Some(value)) => value,
                Ok(None) => return None,
                Err(error) => return Some(Err(error)),
            },
            None => lhs,
        };
        slots[step.out_slot] = ScalarF64Slot::F64(apply_scalar_f64_binary(step.op, lhs, rhs));
    }

    out.clear();
    out.reserve(plan.out_slots.len());
    for &slot in &plan.out_slots {
        match slots[slot] {
            ScalarF64Slot::F64(value) => out.push(Value::scalar_f64(value)),
            ScalarF64Slot::NonF64 => return None,
            ScalarF64Slot::Missing => {
                return Some(Err(InterpreterError::MissingVariable(VarId(slot as u32))));
            }
        }
    }
    Some(Ok(()))
}

// ── scalar-i64 arena plan (sibling of the scalar-f64 plan) ──────────────────
// i64 scalar Add/Sub/Mul/Div use the EXACT wrapping/checked ops eval_primitive's
// `int_op` applies (wrapping_add/sub/mul, checked_div(_).unwrap_or(0)), and two
// I64 literals fold to `Literal::I64(int_op(..))` — see
// `type_promotion::binary_literal_op` I64 arm — so the result is bit-identical.
// NOTE: i32-dtype scalars are stored as `Literal::I64` (no Literal::I32 exists),
// exactly as the generic scalar path treats them, so this matches the (dtype-less)
// generic i64 behavior; genuine i32 tensors are NonI64 and bail.

#[derive(Clone, Copy)]
enum ScalarI64Operand {
    Slot(usize),
    Literal(i64),
}

struct ScalarI64Step {
    op: ScalarF64BinaryOp,
    lhs: ScalarI64Operand,
    rhs: Option<ScalarI64Operand>,
    out_slot: usize,
}

struct ScalarI64Plan {
    slots: usize,
    const_slots: Vec<usize>,
    input_slots: Vec<usize>,
    out_slots: Vec<usize>,
    steps: Vec<ScalarI64Step>,
}

#[derive(Clone, Copy)]
enum ScalarI64Slot {
    Missing,
    NonI64,
    I64(i64),
}

fn scalar_i64_operand(atom: &Atom, slots: usize) -> Option<ScalarI64Operand> {
    match atom {
        Atom::Var(var) => {
            let slot = var.0 as usize;
            (slot < slots).then_some(ScalarI64Operand::Slot(slot))
        }
        Atom::Lit(Literal::I64(value)) => Some(ScalarI64Operand::Literal(*value)),
        Atom::Lit(_) => None,
    }
}

fn build_scalar_i64_arith_plan(jaxpr: &Jaxpr, slots: usize) -> Option<ScalarI64Plan> {
    if jaxpr.equations.is_empty() || !jaxpr.effects.is_empty() {
        return None;
    }
    let mut steps = Vec::with_capacity(jaxpr.equations.len());
    for equation in &jaxpr.equations {
        if !equation.params.is_empty()
            || !equation.sub_jaxprs.is_empty()
            || !equation.effects.is_empty()
            || equation.outputs.len() != 1
        {
            return None;
        }
        let out_slot = equation.outputs[0].0 as usize;
        if out_slot >= slots {
            return None;
        }
        let (op, lhs, rhs) = match equation.inputs.as_slice() {
            [a, b] => (
                scalar_int_binary_op(equation.primitive)?,
                scalar_i64_operand(a, slots)?,
                Some(scalar_i64_operand(b, slots)?),
            ),
            [a] => {
                let operand = scalar_i64_operand(a, slots)?;
                (scalar_int_unary_op(equation.primitive)?, operand, None)
            }
            _ => return None,
        };
        steps.push(ScalarI64Step {
            op,
            lhs,
            rhs,
            out_slot,
        });
    }
    Some(ScalarI64Plan {
        slots,
        const_slots: var_slots(&jaxpr.constvars, slots)?,
        input_slots: var_slots(&jaxpr.invars, slots)?,
        out_slots: var_slots(&jaxpr.outvars, slots)?,
        steps,
    })
}

fn scalar_i64_slot_from_value(value: &Value) -> ScalarI64Slot {
    match value {
        Value::Scalar(Literal::I64(value)) => ScalarI64Slot::I64(*value),
        Value::Scalar(_) | Value::Tensor(_) => ScalarI64Slot::NonI64,
    }
}

fn read_scalar_i64_operand(
    slots: &[ScalarI64Slot],
    operand: ScalarI64Operand,
) -> Result<Option<i64>, InterpreterError> {
    match operand {
        ScalarI64Operand::Literal(value) => Ok(Some(value)),
        ScalarI64Operand::Slot(slot) => match slots[slot] {
            ScalarI64Slot::I64(value) => Ok(Some(value)),
            ScalarI64Slot::NonI64 => Ok(None),
            ScalarI64Slot::Missing => Err(InterpreterError::MissingVariable(VarId(slot as u32))),
        },
    }
}

fn apply_scalar_i64_binary(op: ScalarF64BinaryOp, lhs: i64, rhs: i64) -> i64 {
    match op {
        ScalarF64BinaryOp::Add => lhs.wrapping_add(rhs),
        ScalarF64BinaryOp::Sub => lhs.wrapping_sub(rhs),
        ScalarF64BinaryOp::Mul => lhs.wrapping_mul(rhs),
        ScalarF64BinaryOp::Div => lhs.checked_div(rhs).unwrap_or(0),
        ScalarF64BinaryOp::Max => lhs.max(rhs),
        ScalarF64BinaryOp::Min => lhs.min(rhs),
        ScalarF64BinaryOp::Neg => lhs.wrapping_neg(),
        ScalarF64BinaryOp::Abs => lhs.wrapping_abs(),
        // Float-only transcendental/rounding ops never enter an i64 plan:
        // `scalar_f64_binary_op` (binary) and `scalar_int_unary_op` (unary, the
        // only mappers feeding `build_scalar_i64_arith_plan`) emit none of them.
        _ => unreachable!("transcendental op in i64 scalar plan"),
    }
}

fn run_scalar_i64_arith_plan_into(
    plan: &ScalarI64Plan,
    const_values: &[Value],
    args: &[Value],
    out: &mut Vec<Value>,
    slots: &mut Vec<ScalarI64Slot>,
) -> Option<Result<(), InterpreterError>> {
    if const_values.len() != plan.const_slots.len() {
        return Some(Err(InterpreterError::ConstArity {
            expected: plan.const_slots.len(),
            actual: const_values.len(),
        }));
    }
    if args.len() != plan.input_slots.len() {
        return Some(Err(InterpreterError::InputArity {
            expected: plan.input_slots.len(),
            actual: args.len(),
        }));
    }

    slots.clear();
    slots.resize(plan.slots, ScalarI64Slot::Missing);
    for (&slot, value) in plan.const_slots.iter().zip(const_values) {
        slots[slot] = scalar_i64_slot_from_value(value);
    }
    for (&slot, value) in plan.input_slots.iter().zip(args) {
        slots[slot] = scalar_i64_slot_from_value(value);
    }

    for step in &plan.steps {
        let lhs = match read_scalar_i64_operand(slots, step.lhs) {
            Ok(Some(value)) => value,
            Ok(None) => return None,
            Err(error) => return Some(Err(error)),
        };
        let rhs = match step.rhs {
            Some(rhs) => match read_scalar_i64_operand(slots, rhs) {
                Ok(Some(value)) => value,
                Ok(None) => return None,
                Err(error) => return Some(Err(error)),
            },
            None => lhs,
        };
        slots[step.out_slot] = ScalarI64Slot::I64(apply_scalar_i64_binary(step.op, lhs, rhs));
    }

    out.clear();
    out.reserve(plan.out_slots.len());
    for &slot in &plan.out_slots {
        match slots[slot] {
            ScalarI64Slot::I64(value) => out.push(Value::scalar_i64(value)),
            ScalarI64Slot::NonI64 => return None,
            ScalarI64Slot::Missing => {
                return Some(Err(InterpreterError::MissingVariable(VarId(slot as u32))));
            }
        }
    }
    Some(Ok(()))
}

// ── scalar-f32 arena plan ───────────────────────────────────────────────────
// f32 scalar Add/Sub/Mul/Div go through eval_primitive's `binary_literal_op`
// `_ =>` arm, which WIDENS each f32 operand to f64, applies the f64 op, then
// narrows to f32 (`literal_from_numeric_f64(F32, ..)`). A NATIVE f32 op would
// diverge on NaN sign (see bh7y5), so the plan replicates the widen→f64→narrow
// contract exactly: `((lhs as f64) OP (rhs as f64)) as f32`.

#[derive(Clone, Copy)]
enum ScalarF32Operand {
    Slot(usize),
    Literal(f32),
}

struct ScalarF32Step {
    op: ScalarF64BinaryOp,
    lhs: ScalarF32Operand,
    rhs: Option<ScalarF32Operand>,
    out_slot: usize,
}

struct ScalarF32Plan {
    slots: usize,
    const_slots: Vec<usize>,
    input_slots: Vec<usize>,
    out_slots: Vec<usize>,
    steps: Vec<ScalarF32Step>,
}

#[derive(Clone, Copy)]
enum ScalarF32Slot {
    Missing,
    NonF32,
    F32(f32),
}

fn scalar_f32_operand(atom: &Atom, slots: usize) -> Option<ScalarF32Operand> {
    match atom {
        Atom::Var(var) => {
            let slot = var.0 as usize;
            (slot < slots).then_some(ScalarF32Operand::Slot(slot))
        }
        Atom::Lit(Literal::F32Bits(bits)) => Some(ScalarF32Operand::Literal(f32::from_bits(*bits))),
        Atom::Lit(_) => None,
    }
}

fn build_scalar_f32_arith_plan(jaxpr: &Jaxpr, slots: usize) -> Option<ScalarF32Plan> {
    if jaxpr.equations.is_empty() || !jaxpr.effects.is_empty() {
        return None;
    }
    let mut steps = Vec::with_capacity(jaxpr.equations.len());
    for equation in &jaxpr.equations {
        if !equation.sub_jaxprs.is_empty()
            || !equation.effects.is_empty()
            || equation.outputs.len() != 1
        {
            return None;
        }
        let out_slot = equation.outputs[0].0 as usize;
        if out_slot >= slots {
            return None;
        }
        let (op, lhs, rhs) = match equation.inputs.as_slice() {
            [a, b] => {
                if !equation.params.is_empty() {
                    return None;
                }
                (
                    scalar_f64_binary_op(equation.primitive)?,
                    scalar_f32_operand(a, slots)?,
                    Some(scalar_f32_operand(b, slots)?),
                )
            }
            [a] => {
                let operand = scalar_f32_operand(a, slots)?;
                (
                    scalar_f64_unary_op_with_params(equation.primitive, &equation.params)?,
                    operand,
                    None,
                )
            }
            _ => return None,
        };
        steps.push(ScalarF32Step {
            op,
            lhs,
            rhs,
            out_slot,
        });
    }
    Some(ScalarF32Plan {
        slots,
        const_slots: var_slots(&jaxpr.constvars, slots)?,
        input_slots: var_slots(&jaxpr.invars, slots)?,
        out_slots: var_slots(&jaxpr.outvars, slots)?,
        steps,
    })
}

fn scalar_f32_slot_from_value(value: &Value) -> ScalarF32Slot {
    match value {
        Value::Scalar(Literal::F32Bits(bits)) => ScalarF32Slot::F32(f32::from_bits(*bits)),
        Value::Scalar(_) | Value::Tensor(_) => ScalarF32Slot::NonF32,
    }
}

fn read_scalar_f32_operand(
    slots: &[ScalarF32Slot],
    operand: ScalarF32Operand,
) -> Result<Option<f32>, InterpreterError> {
    match operand {
        ScalarF32Operand::Literal(value) => Ok(Some(value)),
        ScalarF32Operand::Slot(slot) => match slots[slot] {
            ScalarF32Slot::F32(value) => Ok(Some(value)),
            ScalarF32Slot::NonF32 => Ok(None),
            ScalarF32Slot::Missing => Err(InterpreterError::MissingVariable(VarId(slot as u32))),
        },
    }
}

fn apply_scalar_f32_binary(op: ScalarF64BinaryOp, lhs: f32, rhs: f32) -> f32 {
    // Widen→f64-op→narrow, matching eval_primitive's f32 scalar contract exactly.
    let (lhs, rhs) = (f64::from(lhs), f64::from(rhs));
    let result = match op {
        ScalarF64BinaryOp::Add => lhs + rhs,
        ScalarF64BinaryOp::Sub => lhs - rhs,
        ScalarF64BinaryOp::Mul => lhs * rhs,
        ScalarF64BinaryOp::Div => lhs / rhs,
        ScalarF64BinaryOp::Max => jax_max_f64(lhs, rhs),
        ScalarF64BinaryOp::Min => jax_min_f64(lhs, rhs),
        ScalarF64BinaryOp::Neg => -lhs,
        ScalarF64BinaryOp::Abs => lhs.abs(),
        ScalarF64BinaryOp::Pow => lhs.powf(rhs),
        ScalarF64BinaryOp::Atan2 => lhs.atan2(rhs),
        // Widened operands run the f64 op, then `result as f32` narrows — exactly
        // the generic f32 scalar contract `from_f32(f64::FUNC(f64::from(x)) as f32)`.
        ScalarF64BinaryOp::Exp => lhs.exp(),
        ScalarF64BinaryOp::Log => lhs.ln(),
        ScalarF64BinaryOp::Log2 => lhs.log2(),
        ScalarF64BinaryOp::Exp2 => lhs.exp2(),
        ScalarF64BinaryOp::Expm1 => lhs.exp_m1(),
        ScalarF64BinaryOp::Log1p => lhs.ln_1p(),
        ScalarF64BinaryOp::Sqrt => lhs.sqrt(),
        ScalarF64BinaryOp::Rsqrt => 1.0 / lhs.sqrt(),
        ScalarF64BinaryOp::Sin => lhs.sin(),
        ScalarF64BinaryOp::Cos => lhs.cos(),
        ScalarF64BinaryOp::Tan => lhs.tan(),
        ScalarF64BinaryOp::Asin => lhs.asin(),
        ScalarF64BinaryOp::Acos => lhs.acos(),
        ScalarF64BinaryOp::Atan => lhs.atan(),
        ScalarF64BinaryOp::Sinh => lhs.sinh(),
        ScalarF64BinaryOp::Cosh => lhs.cosh(),
        ScalarF64BinaryOp::Tanh => lhs.tanh(),
        ScalarF64BinaryOp::Asinh => lhs.asinh(),
        ScalarF64BinaryOp::Acosh => lhs.acosh(),
        ScalarF64BinaryOp::Atanh => lhs.atanh(),
        ScalarF64BinaryOp::Floor => lhs.floor(),
        ScalarF64BinaryOp::Ceil => lhs.ceil(),
        ScalarF64BinaryOp::Trunc => lhs.trunc(),
        ScalarF64BinaryOp::Deg2Rad => lhs.to_radians(),
        ScalarF64BinaryOp::Rad2Deg => lhs.to_degrees(),
        ScalarF64BinaryOp::Square => lhs * lhs,
        ScalarF64BinaryOp::IntegerPow(exp) => lhs.powi(exp),
        ScalarF64BinaryOp::Reciprocal => 1.0 / lhs,
        ScalarF64BinaryOp::Cbrt => lhs.cbrt(),
        ScalarF64BinaryOp::Logistic => 1.0 / (1.0 + (-lhs).exp()),
        ScalarF64BinaryOp::Sign => scalar_f64_sign(lhs),
    };
    result as f32
}

fn run_scalar_f32_arith_plan_into(
    plan: &ScalarF32Plan,
    const_values: &[Value],
    args: &[Value],
    out: &mut Vec<Value>,
    slots: &mut Vec<ScalarF32Slot>,
) -> Option<Result<(), InterpreterError>> {
    if const_values.len() != plan.const_slots.len() {
        return Some(Err(InterpreterError::ConstArity {
            expected: plan.const_slots.len(),
            actual: const_values.len(),
        }));
    }
    if args.len() != plan.input_slots.len() {
        return Some(Err(InterpreterError::InputArity {
            expected: plan.input_slots.len(),
            actual: args.len(),
        }));
    }

    slots.clear();
    slots.resize(plan.slots, ScalarF32Slot::Missing);
    for (&slot, value) in plan.const_slots.iter().zip(const_values) {
        slots[slot] = scalar_f32_slot_from_value(value);
    }
    for (&slot, value) in plan.input_slots.iter().zip(args) {
        slots[slot] = scalar_f32_slot_from_value(value);
    }

    for step in &plan.steps {
        let lhs = match read_scalar_f32_operand(slots, step.lhs) {
            Ok(Some(value)) => value,
            Ok(None) => return None,
            Err(error) => return Some(Err(error)),
        };
        let rhs = match step.rhs {
            Some(rhs) => match read_scalar_f32_operand(slots, rhs) {
                Ok(Some(value)) => value,
                Ok(None) => return None,
                Err(error) => return Some(Err(error)),
            },
            None => lhs,
        };
        slots[step.out_slot] = ScalarF32Slot::F32(apply_scalar_f32_binary(step.op, lhs, rhs));
    }

    out.clear();
    out.reserve(plan.out_slots.len());
    for &slot in &plan.out_slots {
        match slots[slot] {
            ScalarF32Slot::F32(value) => out.push(Value::scalar_f32(value)),
            ScalarF32Slot::NonF32 => return None,
            ScalarF32Slot::Missing => {
                return Some(Err(InterpreterError::MissingVariable(VarId(slot as u32))));
            }
        }
    }
    Some(Ok(()))
}

// ── scalar half-float (BF16/F16) arena plan ────────────────────────────────
// BF16/F16 scalar arithmetic has the same per-op contract as the half tensor
// fusion path: widen operands to f64, apply the JAX op, then round the result
// back to the same half dtype. Keeping this as a dtype-parametric scalar arena
// avoids the generic env/scratch/eval_primitive dispatch for half activation
// bodies while preserving every intermediate half rounding point.

#[derive(Clone, Copy)]
enum ScalarHalfOperand {
    Slot(usize),
    Literal(u16),
}

struct ScalarHalfStep {
    op: ScalarF64BinaryOp,
    lhs: ScalarHalfOperand,
    rhs: Option<ScalarHalfOperand>,
    out_slot: usize,
}

struct ScalarHalfPlan {
    dtype: DType,
    slots: usize,
    const_slots: Vec<usize>,
    input_slots: Vec<usize>,
    out_slots: Vec<usize>,
    steps: Vec<ScalarHalfStep>,
}

#[derive(Clone, Copy)]
enum ScalarHalfSlot {
    Missing,
    NonHalf,
    Half(u16),
}

fn scalar_half_binary_op(primitive: Primitive) -> Option<ScalarF64BinaryOp> {
    match primitive {
        Primitive::Add => Some(ScalarF64BinaryOp::Add),
        Primitive::Sub => Some(ScalarF64BinaryOp::Sub),
        Primitive::Mul => Some(ScalarF64BinaryOp::Mul),
        Primitive::Div => Some(ScalarF64BinaryOp::Div),
        Primitive::Max => Some(ScalarF64BinaryOp::Max),
        Primitive::Min => Some(ScalarF64BinaryOp::Min),
        _ => None,
    }
}

fn scalar_half_unary_op(primitive: Primitive) -> Option<ScalarF64BinaryOp> {
    match primitive {
        Primitive::Neg => Some(ScalarF64BinaryOp::Neg),
        Primitive::Abs => Some(ScalarF64BinaryOp::Abs),
        _ => None,
    }
}

fn scalar_half_operand(atom: &Atom, slots: usize, dtype: DType) -> Option<ScalarHalfOperand> {
    match (dtype, atom) {
        (_, Atom::Var(var)) => {
            let slot = var.0 as usize;
            (slot < slots).then_some(ScalarHalfOperand::Slot(slot))
        }
        (DType::BF16, Atom::Lit(Literal::BF16Bits(bits)))
        | (DType::F16, Atom::Lit(Literal::F16Bits(bits))) => {
            Some(ScalarHalfOperand::Literal(*bits))
        }
        _ => None,
    }
}

fn build_scalar_half_arith_plan(
    jaxpr: &Jaxpr,
    slots: usize,
    dtype: DType,
) -> Option<ScalarHalfPlan> {
    if jaxpr.equations.is_empty() || !jaxpr.effects.is_empty() {
        return None;
    }
    let mut steps = Vec::with_capacity(jaxpr.equations.len());
    for equation in &jaxpr.equations {
        if !equation.params.is_empty()
            || !equation.sub_jaxprs.is_empty()
            || !equation.effects.is_empty()
            || equation.outputs.len() != 1
        {
            return None;
        }
        let out_slot = equation.outputs[0].0 as usize;
        if out_slot >= slots {
            return None;
        }
        let (op, lhs, rhs) = match equation.inputs.as_slice() {
            [a, b] => (
                scalar_half_binary_op(equation.primitive)?,
                scalar_half_operand(a, slots, dtype)?,
                Some(scalar_half_operand(b, slots, dtype)?),
            ),
            [a] => {
                let operand = scalar_half_operand(a, slots, dtype)?;
                (scalar_half_unary_op(equation.primitive)?, operand, None)
            }
            _ => return None,
        };
        steps.push(ScalarHalfStep {
            op,
            lhs,
            rhs,
            out_slot,
        });
    }
    Some(ScalarHalfPlan {
        dtype,
        slots,
        const_slots: var_slots(&jaxpr.constvars, slots)?,
        input_slots: var_slots(&jaxpr.invars, slots)?,
        out_slots: var_slots(&jaxpr.outvars, slots)?,
        steps,
    })
}

fn scalar_half_slot_from_value(value: &Value, dtype: DType) -> ScalarHalfSlot {
    match (dtype, value) {
        (DType::BF16, Value::Scalar(Literal::BF16Bits(bits)))
        | (DType::F16, Value::Scalar(Literal::F16Bits(bits))) => ScalarHalfSlot::Half(*bits),
        (_, Value::Scalar(_) | Value::Tensor(_)) => ScalarHalfSlot::NonHalf,
    }
}

fn read_scalar_half_operand(
    slots: &[ScalarHalfSlot],
    operand: ScalarHalfOperand,
) -> Result<Option<u16>, InterpreterError> {
    match operand {
        ScalarHalfOperand::Literal(bits) => Ok(Some(bits)),
        ScalarHalfOperand::Slot(slot) => match slots[slot] {
            ScalarHalfSlot::Half(bits) => Ok(Some(bits)),
            ScalarHalfSlot::NonHalf => Ok(None),
            ScalarHalfSlot::Missing => Err(InterpreterError::MissingVariable(VarId(slot as u32))),
        },
    }
}

fn apply_scalar_half_op(
    dtype: DType,
    op: ScalarF64BinaryOp,
    lhs_bits: u16,
    rhs_bits: u16,
) -> Option<u16> {
    let lhs = half_fusion_widen(dtype, lhs_bits);
    let rhs = half_fusion_widen(dtype, rhs_bits);
    Some(match op {
        ScalarF64BinaryOp::Add => half_fused_binary(dtype, CheapOp::Add, lhs, rhs),
        ScalarF64BinaryOp::Sub => half_fused_binary(dtype, CheapOp::Sub, lhs, rhs),
        ScalarF64BinaryOp::Mul => half_fused_binary(dtype, CheapOp::Mul, lhs, rhs),
        ScalarF64BinaryOp::Div => half_fused_binary(dtype, CheapOp::Div, lhs, rhs),
        ScalarF64BinaryOp::Max => half_fused_binary(dtype, CheapOp::Max, lhs, rhs),
        ScalarF64BinaryOp::Min => half_fused_binary(dtype, CheapOp::Min, lhs, rhs),
        ScalarF64BinaryOp::Neg => half_fusion_round(dtype, -lhs),
        ScalarF64BinaryOp::Abs => half_fusion_round(dtype, lhs.abs()),
        _ => return None,
    })
}

fn run_scalar_half_arith_plan_into(
    plan: &ScalarHalfPlan,
    const_values: &[Value],
    args: &[Value],
    out: &mut Vec<Value>,
    slots: &mut Vec<ScalarHalfSlot>,
) -> Option<Result<(), InterpreterError>> {
    if const_values.len() != plan.const_slots.len() {
        return Some(Err(InterpreterError::ConstArity {
            expected: plan.const_slots.len(),
            actual: const_values.len(),
        }));
    }
    if args.len() != plan.input_slots.len() {
        return Some(Err(InterpreterError::InputArity {
            expected: plan.input_slots.len(),
            actual: args.len(),
        }));
    }

    slots.clear();
    slots.resize(plan.slots, ScalarHalfSlot::Missing);
    for (&slot, value) in plan.const_slots.iter().zip(const_values) {
        slots[slot] = scalar_half_slot_from_value(value, plan.dtype);
    }
    for (&slot, value) in plan.input_slots.iter().zip(args) {
        slots[slot] = scalar_half_slot_from_value(value, plan.dtype);
    }

    for step in &plan.steps {
        let lhs = match read_scalar_half_operand(slots, step.lhs) {
            Ok(Some(value)) => value,
            Ok(None) => return None,
            Err(error) => return Some(Err(error)),
        };
        let rhs = match step.rhs {
            Some(rhs) => match read_scalar_half_operand(slots, rhs) {
                Ok(Some(value)) => value,
                Ok(None) => return None,
                Err(error) => return Some(Err(error)),
            },
            None => lhs,
        };
        let value = apply_scalar_half_op(plan.dtype, step.op, lhs, rhs)?;
        slots[step.out_slot] = ScalarHalfSlot::Half(value);
    }

    out.clear();
    out.reserve(plan.out_slots.len());
    for &slot in &plan.out_slots {
        match slots[slot] {
            ScalarHalfSlot::Half(bits) if plan.dtype == DType::BF16 => {
                out.push(Value::Scalar(Literal::BF16Bits(bits)));
            }
            ScalarHalfSlot::Half(bits) if plan.dtype == DType::F16 => {
                out.push(Value::Scalar(Literal::F16Bits(bits)));
            }
            ScalarHalfSlot::Half(_) | ScalarHalfSlot::NonHalf => return None,
            ScalarHalfSlot::Missing => {
                return Some(Err(InterpreterError::MissingVariable(VarId(slot as u32))));
            }
        }
    }
    Some(Ok(()))
}

// ── scalar single-comparison plan (for while/scan cond predicates) ──────────
// A `while`/`scan` cond sub-jaxpr is RE-EVALUATED every iteration, and the
// canonical predicate is a SINGLE comparison `var CMP (var|literal)` (e.g.
// `i > 0`, `x < n`). With the body on the compiled arena, this 1-op cond on the
// generic path becomes the per-iteration bottleneck. This plan resolves the two
// operands directly (no env / scratch / eval_primitive) and applies the SAME
// semantics as fj-lax `compare_literals`: two I64s compare as integers, any
// float operand compares via f64 (NaN → all-false except Ne), producing a bool.

#[derive(Clone, Copy)]
enum CompareOperand {
    Const(usize),
    Input(usize),
    Lit(Literal),
}

struct ScalarComparePlan {
    op: Primitive,
    lhs: CompareOperand,
    rhs: CompareOperand,
    n_consts: usize,
    n_inputs: usize,
}

#[derive(Clone, Copy)]
enum ScalarBoolLogicOp {
    And,
    Or,
    Xor,
}

enum ScalarCompoundCompareStep {
    Compare {
        op: Primitive,
        lhs: CompareOperand,
        rhs: CompareOperand,
        out_slot: usize,
    },
    Logic {
        op: ScalarBoolLogicOp,
        lhs_slot: usize,
        rhs_slot: usize,
        out_slot: usize,
    },
}

struct ScalarCompoundComparePlan {
    steps: Vec<ScalarCompoundCompareStep>,
    out_slot: usize,
    n_consts: usize,
    n_inputs: usize,
    slots: usize,
}

fn compare_atom_operand(atom: &Atom, jaxpr: &Jaxpr) -> Option<CompareOperand> {
    match atom {
        Atom::Var(var) => {
            if let Some(i) = jaxpr.constvars.iter().position(|c| c == var) {
                Some(CompareOperand::Const(i))
            } else {
                jaxpr
                    .invars
                    .iter()
                    .position(|iv| iv == var)
                    .map(CompareOperand::Input)
                // None means an equation-produced intermediate, impossible for a 1-op cond.
            }
        }
        Atom::Lit(lit) => Some(CompareOperand::Lit(*lit)),
    }
}

fn scalar_compare_primitive(primitive: Primitive) -> Option<Primitive> {
    match primitive {
        Primitive::Eq
        | Primitive::Ne
        | Primitive::Lt
        | Primitive::Le
        | Primitive::Gt
        | Primitive::Ge => Some(primitive),
        _ => None,
    }
}

fn scalar_bool_logic_op(primitive: Primitive) -> Option<ScalarBoolLogicOp> {
    match primitive {
        Primitive::BitwiseAnd => Some(ScalarBoolLogicOp::And),
        Primitive::BitwiseOr => Some(ScalarBoolLogicOp::Or),
        Primitive::BitwiseXor => Some(ScalarBoolLogicOp::Xor),
        _ => None,
    }
}

fn scalar_bool_logic_apply(op: ScalarBoolLogicOp, lhs: bool, rhs: bool) -> bool {
    match op {
        ScalarBoolLogicOp::And => lhs & rhs,
        ScalarBoolLogicOp::Or => lhs | rhs,
        ScalarBoolLogicOp::Xor => lhs ^ rhs,
    }
}

fn build_scalar_compare_plan(jaxpr: &Jaxpr) -> Option<ScalarComparePlan> {
    if jaxpr.equations.len() != 1 || !jaxpr.effects.is_empty() || jaxpr.outvars.len() != 1 {
        return None;
    }
    let eqn = &jaxpr.equations[0];
    if !eqn.params.is_empty()
        || !eqn.sub_jaxprs.is_empty()
        || !eqn.effects.is_empty()
        || eqn.inputs.len() != 2
        || eqn.outputs.len() != 1
        || eqn.outputs[0] != jaxpr.outvars[0]
    {
        return None;
    }
    let op = scalar_compare_primitive(eqn.primitive)?;
    Some(ScalarComparePlan {
        op,
        lhs: compare_atom_operand(&eqn.inputs[0], jaxpr)?,
        rhs: compare_atom_operand(&eqn.inputs[1], jaxpr)?,
        n_consts: jaxpr.constvars.len(),
        n_inputs: jaxpr.invars.len(),
    })
}

fn bool_input_slot(atom: &Atom, produced: &[bool]) -> Option<usize> {
    match atom {
        Atom::Var(var) => {
            let slot = var.0 as usize;
            produced
                .get(slot)
                .copied()
                .is_some_and(|is_bool| is_bool)
                .then_some(slot)
        }
        Atom::Lit(_) => None,
    }
}

fn build_scalar_compound_compare_plan(
    jaxpr: &Jaxpr,
    slots: usize,
) -> Option<ScalarCompoundComparePlan> {
    if jaxpr.equations.len() < 2 || !jaxpr.effects.is_empty() || jaxpr.outvars.len() != 1 {
        return None;
    }

    let mut produced_bool = vec![false; slots];
    let mut steps = Vec::with_capacity(jaxpr.equations.len());
    for equation in &jaxpr.equations {
        if !equation.params.is_empty()
            || !equation.sub_jaxprs.is_empty()
            || !equation.effects.is_empty()
            || equation.inputs.len() != 2
            || equation.outputs.len() != 1
        {
            return None;
        }
        let out_slot = equation.outputs[0].0 as usize;
        if out_slot >= slots {
            return None;
        }

        if let Some(op) = scalar_compare_primitive(equation.primitive) {
            steps.push(ScalarCompoundCompareStep::Compare {
                op,
                lhs: compare_atom_operand(&equation.inputs[0], jaxpr)?,
                rhs: compare_atom_operand(&equation.inputs[1], jaxpr)?,
                out_slot,
            });
            produced_bool[out_slot] = true;
        } else {
            let op = scalar_bool_logic_op(equation.primitive)?;
            let lhs_slot = bool_input_slot(&equation.inputs[0], &produced_bool)?;
            let rhs_slot = bool_input_slot(&equation.inputs[1], &produced_bool)?;
            steps.push(ScalarCompoundCompareStep::Logic {
                op,
                lhs_slot,
                rhs_slot,
                out_slot,
            });
            produced_bool[out_slot] = true;
        }
    }

    let out_slot = jaxpr.outvars[0].0 as usize;
    if out_slot >= slots || !produced_bool[out_slot] {
        return None;
    }

    Some(ScalarCompoundComparePlan {
        steps,
        out_slot,
        n_consts: jaxpr.constvars.len(),
        n_inputs: jaxpr.invars.len(),
        slots,
    })
}

fn resolve_compare_operand(
    operand: CompareOperand,
    const_values: &[Value],
    args: &[Value],
) -> Option<Literal> {
    let value = match operand {
        CompareOperand::Lit(lit) => return Some(lit),
        CompareOperand::Const(i) => const_values.get(i)?,
        CompareOperand::Input(i) => args.get(i)?,
    };
    match value {
        Value::Scalar(lit) => Some(*lit),
        Value::Tensor(_) => None,
    }
}

fn compare_float_operand(lit: Literal) -> Option<f64> {
    match lit {
        Literal::F64Bits(bits) => Some(f64::from_bits(bits)),
        Literal::F32Bits(bits) => Some(f64::from(f32::from_bits(bits))),
        _ => None,
    }
}

fn apply_int_compare(op: Primitive, lhs: i64, rhs: i64) -> bool {
    match op {
        Primitive::Eq => lhs == rhs,
        Primitive::Ne => lhs != rhs,
        Primitive::Lt => lhs < rhs,
        Primitive::Le => lhs <= rhs,
        Primitive::Gt => lhs > rhs,
        _ => lhs >= rhs, // Ge (only comparison ops reach here)
    }
}

fn apply_float_compare(op: Primitive, lhs: f64, rhs: f64) -> bool {
    match op {
        Primitive::Eq => lhs == rhs,
        Primitive::Ne => lhs != rhs,
        Primitive::Lt => lhs < rhs,
        Primitive::Le => lhs <= rhs,
        Primitive::Gt => lhs > rhs,
        _ => lhs >= rhs, // Ge
    }
}

/// Run the single-comparison plan. `Some(b)` = computed; `None` = not applicable
/// (operand not a fast-path scalar dtype, or arity off) → caller falls back to the
/// generic interpreter, which produces the same value/error. Mirrors
/// `compare_literals`: both I64 → integer compare; any float operand → f64 compare.
fn run_scalar_compare_plan(
    plan: &ScalarComparePlan,
    const_values: &[Value],
    args: &[Value],
) -> Option<bool> {
    if const_values.len() != plan.n_consts || args.len() != plan.n_inputs {
        return None;
    }
    let lhs = resolve_compare_operand(plan.lhs, const_values, args)?;
    let rhs = resolve_compare_operand(plan.rhs, const_values, args)?;
    match (lhs, rhs) {
        (Literal::I64(a), Literal::I64(b)) => Some(apply_int_compare(plan.op, a, b)),
        _ => {
            let a = compare_float_operand(lhs)?;
            let b = compare_float_operand(rhs)?;
            Some(apply_float_compare(plan.op, a, b))
        }
    }
}

fn run_scalar_compound_compare_plan(
    plan: &ScalarCompoundComparePlan,
    const_values: &[Value],
    args: &[Value],
    bools: &mut Vec<Option<bool>>,
) -> Option<Result<bool, InterpreterError>> {
    if const_values.len() != plan.n_consts || args.len() != plan.n_inputs {
        return None;
    }

    bools.clear();
    bools.resize(plan.slots, None);
    for step in &plan.steps {
        match step {
            ScalarCompoundCompareStep::Compare {
                op,
                lhs,
                rhs,
                out_slot,
            } => {
                let lhs = resolve_compare_operand(*lhs, const_values, args)?;
                let rhs = resolve_compare_operand(*rhs, const_values, args)?;
                let result = match (lhs, rhs) {
                    (Literal::I64(a), Literal::I64(b)) => apply_int_compare(*op, a, b),
                    _ => {
                        let a = compare_float_operand(lhs)?;
                        let b = compare_float_operand(rhs)?;
                        apply_float_compare(*op, a, b)
                    }
                };
                bools[*out_slot] = Some(result);
            }
            ScalarCompoundCompareStep::Logic {
                op,
                lhs_slot,
                rhs_slot,
                out_slot,
            } => {
                let lhs = match bools.get(*lhs_slot).copied().flatten() {
                    Some(value) => value,
                    None => {
                        return Some(Err(InterpreterError::MissingVariable(VarId(
                            *lhs_slot as u32,
                        ))));
                    }
                };
                let rhs = match bools.get(*rhs_slot).copied().flatten() {
                    Some(value) => value,
                    None => {
                        return Some(Err(InterpreterError::MissingVariable(VarId(
                            *rhs_slot as u32,
                        ))));
                    }
                };
                bools[*out_slot] = Some(scalar_bool_logic_apply(*op, lhs, rhs));
            }
        }
    }

    match bools.get(plan.out_slot).copied().flatten() {
        Some(value) => Some(Ok(value)),
        None => Some(Err(InterpreterError::MissingVariable(VarId(
            plan.out_slot as u32,
        )))),
    }
}

// ── scalar mixed f64/bool SELECT arena ──────────────────────────────────────
// The f64/i64/f32 arena plans are SINGLE-dtype, so a scalar body that mixes f64
// arithmetic with a comparison (-> bool) and a `select` (the where / piecewise /
// leaky-relu / relu6 / hardtanh / clamp-via-select class) falls to the tree-walker.
// This plan carries a per-slot MIXED type (f64 OR bool) so such bodies stay on the
// compile-once fast path. Steps reuse `apply_scalar_f64_binary` (every f64 arith op,
// incl. transcendentals/Square/IntegerPow) and `apply_float_compare` (the proven
// bit-identical comparison), and `select` is `if cond { on_true } else { on_false }`
// — bit-identical to fj-lax `eval_select`'s same-dtype scalar arm (both F64 operands
// promote to F64 with no conversion). At runtime any non-f64/bool slot value bails
// the whole plan to the generic interpreter.

#[derive(Clone, Copy)]
enum MixedSlot {
    Missing,
    NonScalar,
    F64(f64),
    Bool(bool),
}

#[derive(Clone, Copy)]
enum BoolOperand {
    Slot(usize),
    Lit(bool),
}

enum ScalarSelectStep {
    F64 {
        op: ScalarF64BinaryOp,
        lhs: ScalarF64Operand,
        rhs: Option<ScalarF64Operand>,
        out_slot: usize,
    },
    Compare {
        op: Primitive,
        lhs: ScalarF64Operand,
        rhs: ScalarF64Operand,
        out_slot: usize,
    },
    Select {
        cond: BoolOperand,
        on_true: ScalarF64Operand,
        on_false: ScalarF64Operand,
        out_slot: usize,
    },
}

struct ScalarSelectPlan {
    slots: usize,
    const_slots: Vec<usize>,
    input_slots: Vec<usize>,
    out_slots: Vec<usize>,
    steps: Vec<ScalarSelectStep>,
}

fn bool_operand(atom: &Atom, slots: usize) -> Option<BoolOperand> {
    match atom {
        Atom::Var(var) => {
            let slot = var.0 as usize;
            (slot < slots).then_some(BoolOperand::Slot(slot))
        }
        Atom::Lit(Literal::Bool(b)) => Some(BoolOperand::Lit(*b)),
        Atom::Lit(_) => None,
    }
}

fn build_scalar_select_plan(jaxpr: &Jaxpr, slots: usize) -> Option<ScalarSelectPlan> {
    if jaxpr.equations.is_empty() || !jaxpr.effects.is_empty() {
        return None;
    }
    // Only build this (slower, type-tagged) plan when the body actually needs the
    // mixed capability — i.e. it contains a `Select`. Pure-f64 bodies use the faster
    // monomorphic f64 plan, which is tried first in `run_dense_plan_into`.
    if !jaxpr
        .equations
        .iter()
        .any(|e| e.primitive == Primitive::Select)
    {
        return None;
    }

    let mut steps = Vec::with_capacity(jaxpr.equations.len());
    for equation in &jaxpr.equations {
        if !equation.sub_jaxprs.is_empty()
            || !equation.effects.is_empty()
            || equation.outputs.len() != 1
        {
            return None;
        }
        let out_slot = equation.outputs[0].0 as usize;
        if out_slot >= slots {
            return None;
        }

        if equation.primitive == Primitive::Select {
            // lax.select(cond, on_true, on_false): 3 inputs, no params.
            if !equation.params.is_empty() || equation.inputs.len() != 3 {
                return None;
            }
            steps.push(ScalarSelectStep::Select {
                cond: bool_operand(&equation.inputs[0], slots)?,
                on_true: scalar_f64_operand(&equation.inputs[1], slots)?,
                on_false: scalar_f64_operand(&equation.inputs[2], slots)?,
                out_slot,
            });
            continue;
        }

        if let Some(op) = scalar_compare_primitive(equation.primitive) {
            if !equation.params.is_empty() || equation.inputs.len() != 2 {
                return None;
            }
            steps.push(ScalarSelectStep::Compare {
                op,
                lhs: scalar_f64_operand(&equation.inputs[0], slots)?,
                rhs: scalar_f64_operand(&equation.inputs[1], slots)?,
                out_slot,
            });
            continue;
        }

        // Otherwise it must be an f64 arithmetic op (binary, or unary incl.
        // IntegerPow's exponent param), reusing the f64 plan's resolvers.
        let (op, lhs, rhs) = match equation.inputs.as_slice() {
            [a, b] => {
                if !equation.params.is_empty() {
                    return None;
                }
                (
                    scalar_f64_binary_op(equation.primitive)?,
                    scalar_f64_operand(a, slots)?,
                    Some(scalar_f64_operand(b, slots)?),
                )
            }
            [a] => (
                scalar_f64_unary_op_with_params(equation.primitive, &equation.params)?,
                scalar_f64_operand(a, slots)?,
                None,
            ),
            _ => return None,
        };
        steps.push(ScalarSelectStep::F64 {
            op,
            lhs,
            rhs,
            out_slot,
        });
    }

    Some(ScalarSelectPlan {
        slots,
        const_slots: var_slots(&jaxpr.constvars, slots)?,
        input_slots: var_slots(&jaxpr.invars, slots)?,
        out_slots: var_slots(&jaxpr.outvars, slots)?,
        steps,
    })
}

fn mixed_slot_from_value(value: &Value) -> MixedSlot {
    match value {
        Value::Scalar(Literal::F64Bits(bits)) => MixedSlot::F64(f64::from_bits(*bits)),
        Value::Scalar(Literal::Bool(b)) => MixedSlot::Bool(*b),
        Value::Scalar(_) | Value::Tensor(_) => MixedSlot::NonScalar,
    }
}

fn read_mixed_f64(
    slots: &[MixedSlot],
    operand: ScalarF64Operand,
) -> Result<Option<f64>, InterpreterError> {
    match operand {
        ScalarF64Operand::Literal(value) => Ok(Some(value)),
        ScalarF64Operand::Slot(slot) => match slots[slot] {
            MixedSlot::F64(value) => Ok(Some(value)),
            MixedSlot::Bool(_) | MixedSlot::NonScalar => Ok(None),
            MixedSlot::Missing => Err(InterpreterError::MissingVariable(VarId(slot as u32))),
        },
    }
}

fn read_mixed_bool(
    slots: &[MixedSlot],
    operand: BoolOperand,
) -> Result<Option<bool>, InterpreterError> {
    match operand {
        BoolOperand::Lit(value) => Ok(Some(value)),
        BoolOperand::Slot(slot) => match slots[slot] {
            MixedSlot::Bool(value) => Ok(Some(value)),
            MixedSlot::F64(_) | MixedSlot::NonScalar => Ok(None),
            MixedSlot::Missing => Err(InterpreterError::MissingVariable(VarId(slot as u32))),
        },
    }
}

fn run_scalar_select_plan_into(
    plan: &ScalarSelectPlan,
    const_values: &[Value],
    args: &[Value],
    out: &mut Vec<Value>,
    slots: &mut Vec<MixedSlot>,
) -> Option<Result<(), InterpreterError>> {
    if const_values.len() != plan.const_slots.len() {
        return Some(Err(InterpreterError::ConstArity {
            expected: plan.const_slots.len(),
            actual: const_values.len(),
        }));
    }
    if args.len() != plan.input_slots.len() {
        return Some(Err(InterpreterError::InputArity {
            expected: plan.input_slots.len(),
            actual: args.len(),
        }));
    }

    slots.clear();
    slots.resize(plan.slots, MixedSlot::Missing);
    for (&slot, value) in plan.const_slots.iter().zip(const_values) {
        slots[slot] = mixed_slot_from_value(value);
    }
    for (&slot, value) in plan.input_slots.iter().zip(args) {
        slots[slot] = mixed_slot_from_value(value);
    }

    macro_rules! read_f64 {
        ($op:expr) => {
            match read_mixed_f64(slots, $op) {
                Ok(Some(v)) => v,
                Ok(None) => return None,
                Err(e) => return Some(Err(e)),
            }
        };
    }

    for step in &plan.steps {
        match step {
            ScalarSelectStep::F64 {
                op,
                lhs,
                rhs,
                out_slot,
            } => {
                let l = read_f64!(*lhs);
                let r = match rhs {
                    Some(rhs) => read_f64!(*rhs),
                    None => l,
                };
                slots[*out_slot] = MixedSlot::F64(apply_scalar_f64_binary(*op, l, r));
            }
            ScalarSelectStep::Compare {
                op,
                lhs,
                rhs,
                out_slot,
            } => {
                let l = read_f64!(*lhs);
                let r = read_f64!(*rhs);
                slots[*out_slot] = MixedSlot::Bool(apply_float_compare(*op, l, r));
            }
            ScalarSelectStep::Select {
                cond,
                on_true,
                on_false,
                out_slot,
            } => {
                let c = match read_mixed_bool(slots, *cond) {
                    Ok(Some(v)) => v,
                    Ok(None) => return None,
                    Err(e) => return Some(Err(e)),
                };
                // Read BOTH branches (matches the generic interpreter, which has both
                // operands resolved); pick by the condition.
                let t = read_f64!(*on_true);
                let f = read_f64!(*on_false);
                slots[*out_slot] = MixedSlot::F64(if c { t } else { f });
            }
        }
    }

    out.clear();
    out.reserve(plan.out_slots.len());
    for &slot in &plan.out_slots {
        match slots[slot] {
            MixedSlot::F64(value) => out.push(Value::scalar_f64(value)),
            MixedSlot::Bool(value) => out.push(Value::scalar_bool(value)),
            MixedSlot::NonScalar => return None,
            MixedSlot::Missing => {
                return Some(Err(InterpreterError::MissingVariable(VarId(slot as u32))));
            }
        }
    }
    Some(Ok(()))
}

// ── scalar mixed i64/bool SELECT arena (sibling of the f64 select arena) ─────
// Integer-conditional scalar bodies — `select(cond, i+1, i)` masked carry updates,
// integer clamps, index logic — that mix i64 arithmetic + an i64 comparison (→bool)
// + a `select` with i64 branches. These appear in scan/while CARRY bodies (the
// per-iteration regime where dispatch elimination pays the most). Reuses
// `apply_scalar_i64_binary` (wrapping int ops) + `apply_int_compare` (the proven
// bit-identical integer comparison, shared with the cond-predicate plans), and
// `select = if cond { on_true } else { on_false }` — bit-identical to fj-lax
// `eval_select`'s same-dtype I64 scalar arm. Built alongside the f64 select plan and
// tried at runtime; a body whose runtime slots are not i64/bool bails to generic.

#[derive(Clone, Copy)]
enum MixedI64Slot {
    Missing,
    NonI64,
    I64(i64),
    Bool(bool),
}

enum ScalarSelectI64Step {
    I64 {
        op: ScalarF64BinaryOp,
        lhs: ScalarI64Operand,
        rhs: Option<ScalarI64Operand>,
        out_slot: usize,
    },
    Compare {
        op: Primitive,
        lhs: ScalarI64Operand,
        rhs: ScalarI64Operand,
        out_slot: usize,
    },
    Select {
        cond: BoolOperand,
        on_true: ScalarI64Operand,
        on_false: ScalarI64Operand,
        out_slot: usize,
    },
}

struct ScalarSelectI64Plan {
    slots: usize,
    const_slots: Vec<usize>,
    input_slots: Vec<usize>,
    out_slots: Vec<usize>,
    steps: Vec<ScalarSelectI64Step>,
}

fn build_scalar_select_i64_plan(jaxpr: &Jaxpr, slots: usize) -> Option<ScalarSelectI64Plan> {
    if jaxpr.equations.is_empty() || !jaxpr.effects.is_empty() {
        return None;
    }
    if !jaxpr
        .equations
        .iter()
        .any(|e| e.primitive == Primitive::Select)
    {
        return None;
    }

    let mut steps = Vec::with_capacity(jaxpr.equations.len());
    for equation in &jaxpr.equations {
        if !equation.sub_jaxprs.is_empty()
            || !equation.effects.is_empty()
            || equation.outputs.len() != 1
        {
            return None;
        }
        let out_slot = equation.outputs[0].0 as usize;
        if out_slot >= slots {
            return None;
        }

        if equation.primitive == Primitive::Select {
            if !equation.params.is_empty() || equation.inputs.len() != 3 {
                return None;
            }
            steps.push(ScalarSelectI64Step::Select {
                cond: bool_operand(&equation.inputs[0], slots)?,
                on_true: scalar_i64_operand(&equation.inputs[1], slots)?,
                on_false: scalar_i64_operand(&equation.inputs[2], slots)?,
                out_slot,
            });
            continue;
        }

        if let Some(op) = scalar_compare_primitive(equation.primitive) {
            if !equation.params.is_empty() || equation.inputs.len() != 2 {
                return None;
            }
            steps.push(ScalarSelectI64Step::Compare {
                op,
                lhs: scalar_i64_operand(&equation.inputs[0], slots)?,
                rhs: scalar_i64_operand(&equation.inputs[1], slots)?,
                out_slot,
            });
            continue;
        }

        // Integer arithmetic: binary (Add/Sub/Mul/Div/Max/Min) or unary (Neg/Abs),
        // matching the i64-arith plan's resolvers and wrapping semantics. No params.
        if !equation.params.is_empty() {
            return None;
        }
        let (op, lhs, rhs) = match equation.inputs.as_slice() {
            [a, b] => (
                scalar_int_binary_op(equation.primitive)?,
                scalar_i64_operand(a, slots)?,
                Some(scalar_i64_operand(b, slots)?),
            ),
            [a] => (
                scalar_int_unary_op(equation.primitive)?,
                scalar_i64_operand(a, slots)?,
                None,
            ),
            _ => return None,
        };
        steps.push(ScalarSelectI64Step::I64 {
            op,
            lhs,
            rhs,
            out_slot,
        });
    }

    Some(ScalarSelectI64Plan {
        slots,
        const_slots: var_slots(&jaxpr.constvars, slots)?,
        input_slots: var_slots(&jaxpr.invars, slots)?,
        out_slots: var_slots(&jaxpr.outvars, slots)?,
        steps,
    })
}

fn mixed_i64_slot_from_value(value: &Value) -> MixedI64Slot {
    match value {
        Value::Scalar(Literal::I64(v)) => MixedI64Slot::I64(*v),
        Value::Scalar(Literal::Bool(b)) => MixedI64Slot::Bool(*b),
        Value::Scalar(_) | Value::Tensor(_) => MixedI64Slot::NonI64,
    }
}

fn read_mixed_i64(
    slots: &[MixedI64Slot],
    operand: ScalarI64Operand,
) -> Result<Option<i64>, InterpreterError> {
    match operand {
        ScalarI64Operand::Literal(value) => Ok(Some(value)),
        ScalarI64Operand::Slot(slot) => match slots[slot] {
            MixedI64Slot::I64(value) => Ok(Some(value)),
            MixedI64Slot::Bool(_) | MixedI64Slot::NonI64 => Ok(None),
            MixedI64Slot::Missing => Err(InterpreterError::MissingVariable(VarId(slot as u32))),
        },
    }
}

fn read_mixed_i64_bool(
    slots: &[MixedI64Slot],
    operand: BoolOperand,
) -> Result<Option<bool>, InterpreterError> {
    match operand {
        BoolOperand::Lit(value) => Ok(Some(value)),
        BoolOperand::Slot(slot) => match slots[slot] {
            MixedI64Slot::Bool(value) => Ok(Some(value)),
            MixedI64Slot::I64(_) | MixedI64Slot::NonI64 => Ok(None),
            MixedI64Slot::Missing => Err(InterpreterError::MissingVariable(VarId(slot as u32))),
        },
    }
}

fn run_scalar_select_i64_plan_into(
    plan: &ScalarSelectI64Plan,
    const_values: &[Value],
    args: &[Value],
    out: &mut Vec<Value>,
    slots: &mut Vec<MixedI64Slot>,
) -> Option<Result<(), InterpreterError>> {
    if const_values.len() != plan.const_slots.len() {
        return Some(Err(InterpreterError::ConstArity {
            expected: plan.const_slots.len(),
            actual: const_values.len(),
        }));
    }
    if args.len() != plan.input_slots.len() {
        return Some(Err(InterpreterError::InputArity {
            expected: plan.input_slots.len(),
            actual: args.len(),
        }));
    }

    slots.clear();
    slots.resize(plan.slots, MixedI64Slot::Missing);
    for (&slot, value) in plan.const_slots.iter().zip(const_values) {
        slots[slot] = mixed_i64_slot_from_value(value);
    }
    for (&slot, value) in plan.input_slots.iter().zip(args) {
        slots[slot] = mixed_i64_slot_from_value(value);
    }

    macro_rules! read_i64 {
        ($op:expr) => {
            match read_mixed_i64(slots, $op) {
                Ok(Some(v)) => v,
                Ok(None) => return None,
                Err(e) => return Some(Err(e)),
            }
        };
    }

    for step in &plan.steps {
        match step {
            ScalarSelectI64Step::I64 {
                op,
                lhs,
                rhs,
                out_slot,
            } => {
                let l = read_i64!(*lhs);
                let r = match rhs {
                    Some(rhs) => read_i64!(*rhs),
                    None => l,
                };
                slots[*out_slot] = MixedI64Slot::I64(apply_scalar_i64_binary(*op, l, r));
            }
            ScalarSelectI64Step::Compare {
                op,
                lhs,
                rhs,
                out_slot,
            } => {
                let l = read_i64!(*lhs);
                let r = read_i64!(*rhs);
                slots[*out_slot] = MixedI64Slot::Bool(apply_int_compare(*op, l, r));
            }
            ScalarSelectI64Step::Select {
                cond,
                on_true,
                on_false,
                out_slot,
            } => {
                let c = match read_mixed_i64_bool(slots, *cond) {
                    Ok(Some(v)) => v,
                    Ok(None) => return None,
                    Err(e) => return Some(Err(e)),
                };
                let t = read_i64!(*on_true);
                let f = read_i64!(*on_false);
                slots[*out_slot] = MixedI64Slot::I64(if c { t } else { f });
            }
        }
    }

    out.clear();
    out.reserve(plan.out_slots.len());
    for &slot in &plan.out_slots {
        match slots[slot] {
            MixedI64Slot::I64(value) => out.push(Value::scalar_i64(value)),
            MixedI64Slot::Bool(value) => out.push(Value::scalar_bool(value)),
            MixedI64Slot::NonI64 => return None,
            MixedI64Slot::Missing => {
                return Some(Err(InterpreterError::MissingVariable(VarId(slot as u32))));
            }
        }
    }
    Some(Ok(()))
}

// ── polymorphic dtype-tagged scalar core (f64/i64/bool + CONVERT) ───────────
// The monomorphic f64/i64/f32/half plans are single-dtype, so a scalar body with a
// `convert_element_type` (dtype cast — ubiquitous in mixed-precision / int↔float
// code) bails to the tree-walker; and a body that genuinely MIXES dtypes can only
// arise via such a cast. This plan carries a per-slot dtype-TAGGED value, dispatching
// arithmetic / comparison / select / convert on the runtime dtype. It is slower than
// the monomorphic plans (a small dtype match per op) but still skips all the generic
// interpreter's Value boxing, env churn, scratch-collect and param re-parse. Built
// ONLY when the body contains a `ConvertElementType` (the capability the others lack)
// and tried LAST, so the fast monomorphic plans are unaffected. Every op reuses a
// proven bit-identical primitive (apply_scalar_f64_binary / apply_scalar_i64_binary /
// apply_float_compare / apply_int_compare); convert uses Rust `as`-casts, exactly the
// semantics fj-lax `convert_literal` documents (f64->i64 `v as i64` NaN->0/inf-saturate,
// f64->bool `v != 0.0`, ...). Any unsupported dtype / op bails the whole plan to generic.

#[derive(Clone, Copy)]
enum PolyVal {
    F64(f64),
    F32(f32),
    I64(i64),
    Bool(bool),
    /// A bf16/f16 bit pattern (the dtype distinguishes them). Carried only across
    /// `convert` boundaries — the hot bf16↔f32 mixed-precision cast — so half ARITH /
    /// compare bail (the math happens in f32/f64 after the cast).
    Half(DType, u16),
}

#[derive(Clone, Copy)]
enum PolySlot {
    Missing,
    NonScalar,
    Val(PolyVal),
}

#[derive(Clone, Copy)]
enum PolyOperand {
    Slot(usize),
    Lit(PolyVal),
}

#[derive(Clone, Copy)]
enum PolyConvTarget {
    F64,
    F32,
    I64,
    Bool,
    BF16,
    F16,
}

enum PolyStep {
    Arith {
        op: ScalarF64BinaryOp,
        lhs: PolyOperand,
        rhs: Option<PolyOperand>,
        out_slot: usize,
    },
    Compare {
        op: Primitive,
        lhs: PolyOperand,
        rhs: PolyOperand,
        out_slot: usize,
    },
    Select {
        cond: PolyOperand,
        on_true: PolyOperand,
        on_false: PolyOperand,
        out_slot: usize,
    },
    Convert {
        src: PolyOperand,
        to: PolyConvTarget,
        out_slot: usize,
    },
}

struct ScalarPolyPlan {
    slots: usize,
    const_slots: Vec<usize>,
    input_slots: Vec<usize>,
    out_slots: Vec<usize>,
    steps: Vec<PolyStep>,
}

fn poly_operand(atom: &Atom, slots: usize) -> Option<PolyOperand> {
    match atom {
        Atom::Var(var) => {
            let slot = var.0 as usize;
            (slot < slots).then_some(PolyOperand::Slot(slot))
        }
        Atom::Lit(Literal::F64Bits(b)) => Some(PolyOperand::Lit(PolyVal::F64(f64::from_bits(*b)))),
        Atom::Lit(Literal::F32Bits(b)) => Some(PolyOperand::Lit(PolyVal::F32(f32::from_bits(*b)))),
        Atom::Lit(Literal::I64(v)) => Some(PolyOperand::Lit(PolyVal::I64(*v))),
        Atom::Lit(Literal::Bool(b)) => Some(PolyOperand::Lit(PolyVal::Bool(*b))),
        Atom::Lit(Literal::BF16Bits(b)) => Some(PolyOperand::Lit(PolyVal::Half(DType::BF16, *b))),
        Atom::Lit(Literal::F16Bits(b)) => Some(PolyOperand::Lit(PolyVal::Half(DType::F16, *b))),
        Atom::Lit(_) => None,
    }
}

/// The arithmetic ops `apply_scalar_i64_binary` accepts without panicking (its other
/// arms are `unreachable!`). A poly Arith step on i64 operands must be one of these.
fn poly_op_is_int_valid(op: ScalarF64BinaryOp) -> bool {
    matches!(
        op,
        ScalarF64BinaryOp::Add
            | ScalarF64BinaryOp::Sub
            | ScalarF64BinaryOp::Mul
            | ScalarF64BinaryOp::Div
            | ScalarF64BinaryOp::Max
            | ScalarF64BinaryOp::Min
            | ScalarF64BinaryOp::Neg
            | ScalarF64BinaryOp::Abs
    )
}

fn build_scalar_poly_plan(jaxpr: &Jaxpr, slots: usize) -> Option<ScalarPolyPlan> {
    if jaxpr.equations.is_empty() || !jaxpr.effects.is_empty() {
        return None;
    }
    // Only the dtype-mixing `convert` distinguishes this plan from the monomorphic
    // ones; without a convert a body is single-dtype and a faster plan handles it.
    if !jaxpr
        .equations
        .iter()
        .any(|e| e.primitive == Primitive::ConvertElementType)
    {
        return None;
    }

    let mut steps = Vec::with_capacity(jaxpr.equations.len());
    for equation in &jaxpr.equations {
        if !equation.sub_jaxprs.is_empty()
            || !equation.effects.is_empty()
            || equation.outputs.len() != 1
        {
            return None;
        }
        let out_slot = equation.outputs[0].0 as usize;
        if out_slot >= slots {
            return None;
        }

        if equation.primitive == Primitive::ConvertElementType {
            if equation.inputs.len() != 1 {
                return None;
            }
            let to = match equation
                .params
                .get("new_dtype")?
                .trim()
                .to_ascii_lowercase()
                .as_str()
            {
                "f64" | "float64" => PolyConvTarget::F64,
                "f32" | "float32" => PolyConvTarget::F32,
                "i64" => PolyConvTarget::I64,
                "bool" => PolyConvTarget::Bool,
                "bf16" | "bfloat16" => PolyConvTarget::BF16,
                "f16" | "float16" => PolyConvTarget::F16,
                _ => return None,
            };
            steps.push(PolyStep::Convert {
                src: poly_operand(&equation.inputs[0], slots)?,
                to,
                out_slot,
            });
            continue;
        }

        if equation.primitive == Primitive::Select {
            if !equation.params.is_empty() || equation.inputs.len() != 3 {
                return None;
            }
            steps.push(PolyStep::Select {
                cond: poly_operand(&equation.inputs[0], slots)?,
                on_true: poly_operand(&equation.inputs[1], slots)?,
                on_false: poly_operand(&equation.inputs[2], slots)?,
                out_slot,
            });
            continue;
        }

        if let Some(op) = scalar_compare_primitive(equation.primitive) {
            if !equation.params.is_empty() || equation.inputs.len() != 2 {
                return None;
            }
            steps.push(PolyStep::Compare {
                op,
                lhs: poly_operand(&equation.inputs[0], slots)?,
                rhs: poly_operand(&equation.inputs[1], slots)?,
                out_slot,
            });
            continue;
        }

        let (op, lhs, rhs) = match equation.inputs.as_slice() {
            [a, b] => {
                if !equation.params.is_empty() {
                    return None;
                }
                (
                    scalar_f64_binary_op(equation.primitive)?,
                    poly_operand(a, slots)?,
                    Some(poly_operand(b, slots)?),
                )
            }
            [a] => (
                scalar_f64_unary_op_with_params(equation.primitive, &equation.params)?,
                poly_operand(a, slots)?,
                None,
            ),
            _ => return None,
        };
        steps.push(PolyStep::Arith {
            op,
            lhs,
            rhs,
            out_slot,
        });
    }

    Some(ScalarPolyPlan {
        slots,
        const_slots: var_slots(&jaxpr.constvars, slots)?,
        input_slots: var_slots(&jaxpr.invars, slots)?,
        out_slots: var_slots(&jaxpr.outvars, slots)?,
        steps,
    })
}

fn poly_slot_from_value(value: &Value) -> PolySlot {
    match value {
        Value::Scalar(Literal::F64Bits(b)) => PolySlot::Val(PolyVal::F64(f64::from_bits(*b))),
        Value::Scalar(Literal::F32Bits(b)) => PolySlot::Val(PolyVal::F32(f32::from_bits(*b))),
        Value::Scalar(Literal::I64(v)) => PolySlot::Val(PolyVal::I64(*v)),
        Value::Scalar(Literal::Bool(b)) => PolySlot::Val(PolyVal::Bool(*b)),
        Value::Scalar(Literal::BF16Bits(b)) => PolySlot::Val(PolyVal::Half(DType::BF16, *b)),
        Value::Scalar(Literal::F16Bits(b)) => PolySlot::Val(PolyVal::Half(DType::F16, *b)),
        Value::Scalar(_) | Value::Tensor(_) => PolySlot::NonScalar,
    }
}

fn read_poly(
    slots: &[PolySlot],
    operand: PolyOperand,
) -> Result<Option<PolyVal>, InterpreterError> {
    match operand {
        PolyOperand::Lit(value) => Ok(Some(value)),
        PolyOperand::Slot(slot) => match slots[slot] {
            PolySlot::Val(value) => Ok(Some(value)),
            PolySlot::NonScalar => Ok(None),
            PolySlot::Missing => Err(InterpreterError::MissingVariable(VarId(slot as u32))),
        },
    }
}

// Rust `as`-cast convert, matching fj-lax `convert_literal`'s documented scalar
// semantics for f64/i64/bool (verified bit-for-bit by the parity test).
/// The f32 value of a half (bf16/f16) bit pattern, via the SAME `Literal` decode
/// `convert_literal` uses — so I64/Bool targets (which cast an f32 source directly)
/// stay bit-identical for half sources too.
fn poly_half_f32(dt: DType, bits: u16) -> f32 {
    if dt == DType::BF16 {
        Literal::BF16Bits(bits).as_bf16_f32().unwrap_or(0.0)
    } else {
        Literal::F16Bits(bits).as_f16_f32().unwrap_or(0.0)
    }
}

fn poly_convert(src: PolyVal, to: PolyConvTarget) -> PolyVal {
    // `convert_literal` routes float TARGETS through `f64_val()` (so a source widens to
    // f64 first, then `as f32` for F32, or single-rounds to bf16/f16), but the I64 target
    // casts an F32/half source DIRECTLY (`f32 as i64`) and Bool compares it directly
    // (`f32 != 0.0`). Mirrored exactly here; half uses fj-core's own round helpers.
    let f64_of = |v: PolyVal| -> f64 {
        match v {
            PolyVal::F64(x) => x,
            PolyVal::F32(x) => f64::from(x),
            PolyVal::I64(x) => x as f64,
            PolyVal::Bool(b) => {
                if b {
                    1.0
                } else {
                    0.0
                }
            }
            PolyVal::Half(dt, b) => f64::from(poly_half_f32(dt, b)),
        }
    };
    // The f32 representation used by the DIRECT integer/bool casts (matches the source's
    // own width: f32 stays f32, half decodes to f32, others widen through f64).
    let f32_direct = |v: PolyVal| -> f32 {
        match v {
            PolyVal::F32(x) => x,
            PolyVal::Half(dt, b) => poly_half_f32(dt, b),
            other => f64_of(other) as f32,
        }
    };
    let half_bits = |lit: Literal| -> u16 {
        match lit {
            Literal::BF16Bits(b) | Literal::F16Bits(b) => b,
            _ => 0,
        }
    };
    match to {
        PolyConvTarget::F64 => PolyVal::F64(f64_of(src)),
        PolyConvTarget::F32 => PolyVal::F32(f64_of(src) as f32),
        PolyConvTarget::BF16 => {
            PolyVal::Half(DType::BF16, half_bits(Literal::from_bf16_f64(f64_of(src))))
        }
        PolyConvTarget::F16 => {
            PolyVal::Half(DType::F16, half_bits(Literal::from_f16_f64(f64_of(src))))
        }
        PolyConvTarget::I64 => PolyVal::I64(match src {
            PolyVal::F64(v) => v as i64,
            PolyVal::I64(v) => v,
            PolyVal::Bool(b) => i64::from(b),
            // f32 / half cast DIRECTLY from the f32 value (not via f64).
            PolyVal::F32(_) | PolyVal::Half(..) => f32_direct(src) as i64,
        }),
        PolyConvTarget::Bool => PolyVal::Bool(match src {
            PolyVal::F64(v) => v != 0.0,
            PolyVal::I64(v) => v != 0,
            PolyVal::Bool(b) => b,
            PolyVal::F32(_) | PolyVal::Half(..) => f32_direct(src) != 0.0,
        }),
    }
}

fn run_scalar_poly_plan_into(
    plan: &ScalarPolyPlan,
    const_values: &[Value],
    args: &[Value],
    out: &mut Vec<Value>,
    slots: &mut Vec<PolySlot>,
) -> Option<Result<(), InterpreterError>> {
    if const_values.len() != plan.const_slots.len() {
        return Some(Err(InterpreterError::ConstArity {
            expected: plan.const_slots.len(),
            actual: const_values.len(),
        }));
    }
    if args.len() != plan.input_slots.len() {
        return Some(Err(InterpreterError::InputArity {
            expected: plan.input_slots.len(),
            actual: args.len(),
        }));
    }

    slots.clear();
    slots.resize(plan.slots, PolySlot::Missing);
    for (&slot, value) in plan.const_slots.iter().zip(const_values) {
        slots[slot] = poly_slot_from_value(value);
    }
    for (&slot, value) in plan.input_slots.iter().zip(args) {
        slots[slot] = poly_slot_from_value(value);
    }

    macro_rules! read {
        ($op:expr) => {
            match read_poly(slots, $op) {
                Ok(Some(v)) => v,
                Ok(None) => return None,
                Err(e) => return Some(Err(e)),
            }
        };
    }

    for step in &plan.steps {
        match step {
            PolyStep::Arith {
                op,
                lhs,
                rhs,
                out_slot,
            } => {
                let l = read!(*lhs);
                let r = match rhs {
                    Some(rhs) => read!(*rhs),
                    None => l,
                };
                let result = match (l, r) {
                    (PolyVal::F64(a), PolyVal::F64(b)) => {
                        PolyVal::F64(apply_scalar_f64_binary(*op, a, b))
                    }
                    (PolyVal::F32(a), PolyVal::F32(b)) => {
                        PolyVal::F32(apply_scalar_f32_binary(*op, a, b))
                    }
                    (PolyVal::I64(a), PolyVal::I64(b)) if poly_op_is_int_valid(*op) => {
                        PolyVal::I64(apply_scalar_i64_binary(*op, a, b))
                    }
                    _ => return None,
                };
                slots[*out_slot] = PolySlot::Val(result);
            }
            PolyStep::Compare {
                op,
                lhs,
                rhs,
                out_slot,
            } => {
                let l = read!(*lhs);
                let r = read!(*rhs);
                let result = match (l, r) {
                    (PolyVal::I64(a), PolyVal::I64(b)) => apply_int_compare(*op, a, b),
                    (PolyVal::F64(a), PolyVal::F64(b)) => apply_float_compare(*op, a, b),
                    // f32 compares widen to f64 (order-preserving) — bit-identical bool,
                    // matching the generic f32 scalar comparison.
                    (PolyVal::F32(a), PolyVal::F32(b)) => {
                        apply_float_compare(*op, f64::from(a), f64::from(b))
                    }
                    _ => return None,
                };
                slots[*out_slot] = PolySlot::Val(PolyVal::Bool(result));
            }
            PolyStep::Select {
                cond,
                on_true,
                on_false,
                out_slot,
            } => {
                let c = match read!(*cond) {
                    PolyVal::Bool(b) => b,
                    _ => return None,
                };
                let t = read!(*on_true);
                let f = read!(*on_false);
                slots[*out_slot] = PolySlot::Val(if c { t } else { f });
            }
            PolyStep::Convert { src, to, out_slot } => {
                let v = read!(*src);
                slots[*out_slot] = PolySlot::Val(poly_convert(v, *to));
            }
        }
    }

    out.clear();
    out.reserve(plan.out_slots.len());
    for &slot in &plan.out_slots {
        match slots[slot] {
            PolySlot::Val(PolyVal::F64(v)) => out.push(Value::scalar_f64(v)),
            PolySlot::Val(PolyVal::F32(v)) => out.push(Value::scalar_f32(v)),
            PolySlot::Val(PolyVal::I64(v)) => out.push(Value::scalar_i64(v)),
            PolySlot::Val(PolyVal::Bool(v)) => out.push(Value::scalar_bool(v)),
            PolySlot::Val(PolyVal::Half(dt, b)) => out.push(Value::Scalar(if dt == DType::BF16 {
                Literal::BF16Bits(b)
            } else {
                Literal::F16Bits(b)
            })),
            PolySlot::NonScalar => return None,
            PolySlot::Missing => {
                return Some(Err(InterpreterError::MissingVariable(VarId(slot as u32))));
            }
        }
    }
    Some(Ok(()))
}

// ── dense-f64 SMALL-TENSOR elementwise arena ────────────────────────────────
// The elementwise FUSION (try_fuse_elementwise_chain) is gated to tensors with
// >= FUSION_MIN_ELEMS (1024) elements, so a scan/while body whose carry is a SMALL
// dense-f64 tensor ([H] RNN state, optimizer moments, …) pays full per-equation
// dispatch (Value boxing, env churn, scratch-collect, an intermediate alloc per op)
// EVERY iteration. This arena REUSES the already-built `scalar_f64_plan` (same op set,
// incl. transcendentals / Square / IntegerPow that the cheap-op fusion can't do) and
// runs each step ELEMENTWISE over dense-f64 tensor cells with scalar broadcast — no
// dispatch, no boxing. Per element it calls the SAME `apply_scalar_f64_binary` the
// scalar arena uses, so it is bit-identical to the generic per-op path. It only fires
// for a same-shape dense-f64 body BELOW the fusion threshold (>= 1024 bails so the
// chunked/threaded fusion still owns large tensors); any non-dense-f64 slot bails it
// to the generic interpreter.

enum DenseF64Cell {
    Missing,
    NonDense,
    Scalar(f64),
    Tensor(Vec<f64>),
    // rank-2 broadcast operands against a [R, C] body (cols = C). Row: a [C] vector
    // broadcast across rows (index `i % C`). Col: an [R, 1] vector down columns
    // (index `i / C`). Matches the fusion's row/col-broadcast helpers exactly.
    RowBcast(Vec<f64>, usize),
    ColBcast(Vec<f64>, usize),
}

#[derive(Clone, Copy)]
enum DenseRef<'a> {
    Scalar(f64),
    Tensor(&'a [f64]),
    RowBcast(&'a [f64], usize),
    ColBcast(&'a [f64], usize),
}

impl DenseRef<'_> {
    /// Flat index — used only on the no-broadcast fast path, so no div/mod.
    #[inline]
    fn at(&self, i: usize) -> f64 {
        match self {
            DenseRef::Scalar(v) => *v,
            DenseRef::Tensor(t) => t[i],
            DenseRef::RowBcast(t, cols) => t[i % cols],
            DenseRef::ColBcast(t, cols) => t[i / cols],
        }
    }

    /// Row/col-indexed read for the broadcast path — `row`/`col` come from the
    /// row-chunked loop, so broadcast operands avoid a per-element div/mod.
    #[inline]
    fn at_rc(&self, row: usize, col: usize, i: usize) -> f64 {
        match self {
            DenseRef::Scalar(v) => *v,
            DenseRef::Tensor(t) => t[i],
            DenseRef::RowBcast(t, _) => t[col],
            DenseRef::ColBcast(t, _) => t[row],
        }
    }

    #[inline]
    fn is_bcast(&self) -> bool {
        matches!(self, DenseRef::RowBcast(..) | DenseRef::ColBcast(..))
    }
}

fn resolve_dense_operand<'a>(
    cells: &'a [DenseF64Cell],
    operand: ScalarF64Operand,
) -> Option<DenseRef<'a>> {
    match operand {
        ScalarF64Operand::Literal(v) => Some(DenseRef::Scalar(v)),
        ScalarF64Operand::Slot(s) => match &cells[s] {
            DenseF64Cell::Scalar(v) => Some(DenseRef::Scalar(*v)),
            DenseF64Cell::Tensor(t) => Some(DenseRef::Tensor(t)),
            DenseF64Cell::RowBcast(t, c) => Some(DenseRef::RowBcast(t, *c)),
            DenseF64Cell::ColBcast(t, c) => Some(DenseRef::ColBcast(t, *c)),
            DenseF64Cell::Missing | DenseF64Cell::NonDense => None,
        },
    }
}

fn operand_is_slot(operand: ScalarF64Operand, slot: usize) -> bool {
    matches!(operand, ScalarF64Operand::Slot(s) if s == slot)
}

/// Apply one in-place linear-chain step: `v[i] = v[i] OP scalar` (or `scalar OP v[i]`
/// when `scalar_on_left`). The op is matched ONCE so the cheap arithmetic ops lower to a
/// straight in-place loop that auto-vectorizes under `+avx2` (vaddpd/…); other ops keep
/// the generic per-element apply. Bit-identical to `apply_scalar_f64_binary` per element:
/// elementwise, no reassociation, no FMA (`+avx2` only, not `+fma`).
#[inline]
fn apply_dense_f64_chain_step(
    values: &mut [f64],
    op: ScalarF64BinaryOp,
    scalar: f64,
    scalar_on_left: bool,
    vectorize: bool,
) {
    match (op, scalar_on_left) {
        (ScalarF64BinaryOp::Add, _) if vectorize => values.iter_mut().for_each(|v| *v += scalar),
        (ScalarF64BinaryOp::Mul, _) if vectorize => values.iter_mut().for_each(|v| *v *= scalar),
        (ScalarF64BinaryOp::Sub, false) if vectorize => {
            values.iter_mut().for_each(|v| *v -= scalar)
        }
        (ScalarF64BinaryOp::Sub, true) if vectorize => {
            values.iter_mut().for_each(|v| *v = scalar - *v)
        }
        (ScalarF64BinaryOp::Div, false) if vectorize => {
            values.iter_mut().for_each(|v| *v /= scalar)
        }
        (ScalarF64BinaryOp::Div, true) if vectorize => {
            values.iter_mut().for_each(|v| *v = scalar / *v)
        }
        (op, false) => values
            .iter_mut()
            .for_each(|v| *v = apply_scalar_f64_binary(op, *v, scalar)),
        (op, true) => values
            .iter_mut()
            .for_each(|v| *v = apply_scalar_f64_binary(op, scalar, *v)),
    }
}

/// The non-`current` operand of an in-place linear-chain step: a scalar/literal, or a
/// rank-2 broadcast vector ([C] row vector or [R] column vector). A full same-shape
/// tensor never appears here — that would be a second tensor slot, which the chain
/// validator rejects.
enum ChainOperand<'a> {
    Scalar(f64),
    Row(&'a [f64]),
    Col(&'a [f64]),
}

fn dense_f64_chain_operand(
    cells: &[DenseF64Cell],
    operand: ScalarF64Operand,
) -> Option<ChainOperand<'_>> {
    match operand {
        ScalarF64Operand::Literal(v) => Some(ChainOperand::Scalar(v)),
        ScalarF64Operand::Slot(s) => match cells.get(s)? {
            DenseF64Cell::Scalar(v) => Some(ChainOperand::Scalar(*v)),
            DenseF64Cell::RowBcast(t, _) => Some(ChainOperand::Row(t)),
            DenseF64Cell::ColBcast(t, _) => Some(ChainOperand::Col(t)),
            _ => None,
        },
    }
}

/// Resolve a chain step's non-current operand, admitting broadcast vecs only for a
/// rank-2 body (so the in-place row walk is well-defined).
fn resolve_chain_operand(
    cells: &[DenseF64Cell],
    operand: ScalarF64Operand,
    rank2: bool,
) -> Option<ChainOperand<'_>> {
    let op = dense_f64_chain_operand(cells, operand)?;
    match op {
        ChainOperand::Scalar(_) => Some(op),
        ChainOperand::Row(_) | ChainOperand::Col(_) if rank2 => Some(op),
        _ => None,
    }
}

/// In-place `row[c] = f(row[c], vec[c])` for every row of a row-major [R,C] buffer — the
/// row-broadcast step. The op closure is monomorphized, so each row vectorizes.
#[inline]
fn apply_inplace_row_bcast_f64<F: Fn(f64, f64) -> f64 + Copy>(
    values: &mut [f64],
    vec: &[f64],
    cols: usize,
    f: F,
) {
    for row in values.chunks_mut(cols) {
        for (v, &s) in row.iter_mut().zip(vec) {
            *v = f(*v, s);
        }
    }
}

/// Apply one in-place linear-chain step where the other operand is a [C] row-broadcast
/// vector. Hoists op+side once; cheap ops vectorize. Bit-identical per element to the
/// per-element `at_rc` broadcast loop.
fn apply_dense_f64_chain_step_row(
    values: &mut [f64],
    op: ScalarF64BinaryOp,
    vec: &[f64],
    cols: usize,
    scalar_on_left: bool,
    vectorize: bool,
) {
    match (op, scalar_on_left) {
        (ScalarF64BinaryOp::Add, _) if vectorize => {
            apply_inplace_row_bcast_f64(values, vec, cols, |x, s| x + s)
        }
        (ScalarF64BinaryOp::Mul, _) if vectorize => {
            apply_inplace_row_bcast_f64(values, vec, cols, |x, s| x * s)
        }
        (ScalarF64BinaryOp::Sub, false) if vectorize => {
            apply_inplace_row_bcast_f64(values, vec, cols, |x, s| x - s)
        }
        (ScalarF64BinaryOp::Sub, true) if vectorize => {
            apply_inplace_row_bcast_f64(values, vec, cols, |x, s| s - x)
        }
        (ScalarF64BinaryOp::Div, false) if vectorize => {
            apply_inplace_row_bcast_f64(values, vec, cols, |x, s| x / s)
        }
        (ScalarF64BinaryOp::Div, true) if vectorize => {
            apply_inplace_row_bcast_f64(values, vec, cols, |x, s| s / x)
        }
        (op, false) => apply_inplace_row_bcast_f64(values, vec, cols, move |x, s| {
            apply_scalar_f64_binary(op, x, s)
        }),
        (op, true) => apply_inplace_row_bcast_f64(values, vec, cols, move |x, s| {
            apply_scalar_f64_binary(op, s, x)
        }),
    }
}

/// Apply one in-place linear-chain step for a scalar OR rank-2 broadcast operand.
/// `ColBcast` is a constant per row, so it reuses the scalar step per row.
fn apply_dense_f64_chain_step_bcast(
    values: &mut [f64],
    op: ScalarF64BinaryOp,
    operand: ChainOperand,
    cols: usize,
    scalar_on_left: bool,
    vectorize: bool,
) {
    match operand {
        ChainOperand::Scalar(s) => {
            apply_dense_f64_chain_step(values, op, s, scalar_on_left, vectorize)
        }
        ChainOperand::Row(vec) => {
            apply_dense_f64_chain_step_row(values, op, vec, cols, scalar_on_left, vectorize)
        }
        ChainOperand::Col(vec) => {
            for (row, chunk) in values.chunks_mut(cols).enumerate() {
                apply_dense_f64_chain_step(chunk, op, vec[row], scalar_on_left, vectorize);
            }
        }
    }
}

fn run_linear_scalar_f64_tensor_chain_into(
    plan: &ScalarF64Plan,
    shape: &Shape,
    out: &mut Vec<Value>,
    cells: &mut [DenseF64Cell],
    vectorize: bool,
) -> Option<Result<(), InterpreterError>> {
    if plan.out_slots.len() != 1 || plan.steps.is_empty() {
        return None;
    }

    let mut tensor_slot = None;
    for (slot, cell) in cells.iter().enumerate() {
        if matches!(cell, DenseF64Cell::Tensor(_)) {
            if tensor_slot.is_some() {
                return None;
            }
            tensor_slot = Some(slot);
        }
    }

    let tensor_slot = tensor_slot?;
    // Broadcast (Row/Col) operands are only meaningful against a rank-2 [R,C] body; a
    // chain whose other operand is a broadcast vec is accepted only then. `cols` is the
    // trailing dim used to walk rows in the in-place mutation.
    let cols = *shape.dims.last()? as usize;
    let rank2 = shape.dims.len() == 2;
    let mut current_slot = tensor_slot;
    for step in &plan.steps {
        let lhs_is_current = operand_is_slot(step.lhs, current_slot);
        match step.rhs {
            Some(rhs) => {
                let rhs_is_current = operand_is_slot(rhs, current_slot);
                match (lhs_is_current, rhs_is_current) {
                    (true, true) => {}
                    (true, false) => {
                        resolve_chain_operand(cells, rhs, rank2)?;
                    }
                    (false, true) => {
                        resolve_chain_operand(cells, step.lhs, rank2)?;
                    }
                    (false, false) => return None,
                }
            }
            None => {
                if !lhs_is_current {
                    return None;
                }
            }
        }
        current_slot = step.out_slot;
    }

    if plan.out_slots[0] != current_slot {
        return None;
    }

    let DenseF64Cell::Tensor(mut values) =
        std::mem::replace(&mut cells[tensor_slot], DenseF64Cell::Missing)
    else {
        unreachable!("validated single dense-f64 tensor slot")
    };

    let mut current_slot = tensor_slot;
    for step in &plan.steps {
        let lhs_is_current = operand_is_slot(step.lhs, current_slot);
        match step.rhs {
            Some(rhs) => {
                let rhs_is_current = operand_is_slot(rhs, current_slot);
                match (lhs_is_current, rhs_is_current) {
                    (true, true) => {
                        for value in &mut values {
                            *value = apply_scalar_f64_binary(step.op, *value, *value);
                        }
                    }
                    (true, false) => {
                        let rhs_op = resolve_chain_operand(cells, rhs, rank2)?;
                        apply_dense_f64_chain_step_bcast(
                            &mut values,
                            step.op,
                            rhs_op,
                            cols,
                            false,
                            vectorize,
                        );
                    }
                    (false, true) => {
                        let lhs_op = resolve_chain_operand(cells, step.lhs, rank2)?;
                        apply_dense_f64_chain_step_bcast(
                            &mut values,
                            step.op,
                            lhs_op,
                            cols,
                            true,
                            vectorize,
                        );
                    }
                    (false, false) => unreachable!("validated linear tensor chain"),
                }
            }
            None => {
                debug_assert!(lhs_is_current);
                for value in &mut values {
                    *value = apply_scalar_f64_binary(step.op, *value, *value);
                }
            }
        }
        current_slot = step.out_slot;
    }

    let tensor = match TensorValue::new_f64_values(shape.clone(), values) {
        Ok(t) => t,
        Err(e) => {
            return Some(Err(InterpreterError::Primitive(EvalError::InvalidTensor(
                e,
            ))));
        }
    };
    out.clear();
    out.push(Value::Tensor(tensor));
    Some(Ok(()))
}

/// Fill `o[i] = f(a[i], b[i])` for the NO-BROADCAST case (operands are scalar or
/// same-shape dense tensors). The op closure `f` is monomorphized and the operand kind
/// is matched ONCE outside the element loop, so for a pure arithmetic `f` the body is a
/// straight-line `o[i] = a[i] OP b[i]` that auto-vectorizes (vaddpd/… under `+avx2`) —
/// unlike the generic `apply_scalar_f64_binary(step.op, a.at(i), b.at(i))` loop whose
/// per-element op-match (40+ arms) and 4-way `DenseRef::at` match block vectorization.
/// Bit-identical: SIMD f64 add/sub/mul/div are elementwise, no reassociation, and we
/// never introduce FMA (`+avx2` only, not `+fma`).
#[inline]
fn fill_dense_f64_nobcast<F: Fn(f64, f64) -> f64>(a: DenseRef, b: DenseRef, o: &mut [f64], f: F) {
    match (a, b) {
        (DenseRef::Tensor(ta), DenseRef::Scalar(sb)) => {
            for (o, &x) in o.iter_mut().zip(ta) {
                *o = f(x, sb);
            }
        }
        (DenseRef::Scalar(sa), DenseRef::Tensor(tb)) => {
            for (o, &y) in o.iter_mut().zip(tb) {
                *o = f(sa, y);
            }
        }
        (DenseRef::Tensor(ta), DenseRef::Tensor(tb)) => {
            for ((o, &x), &y) in o.iter_mut().zip(ta).zip(tb) {
                *o = f(x, y);
            }
        }
        // (Scalar,Scalar) is handled by the caller; broadcast operands never reach here.
        _ => {
            for (i, o) in o.iter_mut().enumerate() {
                *o = f(a.at(i), b.at(i));
            }
        }
    }
}

/// Vectorize the rank-2 broadcast case by decomposing each row into a NO-broadcast
/// subproblem and reusing [`fill_dense_f64_nobcast`]. A `RowBcast` ([C] vector) is the
/// same `Tensor` slice every row; a `ColBcast` ([R,1] vector) is a per-row `Scalar`; a
/// full tensor is its row slice. The op closure is monomorphized once, so each row's
/// inner loop auto-vectorizes. Bit-identical to the per-element `at_rc` broadcast loop:
/// element `row*cols+col` gets exactly the operand value `at_rc` would read.
#[inline]
fn fill_dense_f64_bcast<F: Fn(f64, f64) -> f64 + Copy>(
    a: DenseRef,
    b: DenseRef,
    o: &mut [f64],
    cols: usize,
    f: F,
) {
    fn row_ref(operand: DenseRef, row: usize, cols: usize) -> DenseRef {
        match operand {
            DenseRef::Scalar(v) => DenseRef::Scalar(v),
            DenseRef::Tensor(t) => DenseRef::Tensor(&t[row * cols..row * cols + cols]),
            // [C] row vector: same slice for every row.
            DenseRef::RowBcast(t, _) => DenseRef::Tensor(&t[..cols]),
            // [R,1] column vector: a constant for this whole row.
            DenseRef::ColBcast(t, _) => DenseRef::Scalar(t[row]),
        }
    }
    for (row, orow) in o.chunks_mut(cols).enumerate() {
        fill_dense_f64_nobcast(row_ref(a, row, cols), row_ref(b, row, cols), orow, f);
    }
}

fn run_scalar_f64_plan_as_tensor_into(
    plan: &ScalarF64Plan,
    const_values: &[Value],
    args: &[Value],
    out: &mut Vec<Value>,
    cells: &mut Vec<DenseF64Cell>,
    vectorize: bool,
) -> Option<Result<(), InterpreterError>> {
    if const_values.len() != plan.const_slots.len() {
        return Some(Err(InterpreterError::ConstArity {
            expected: plan.const_slots.len(),
            actual: const_values.len(),
        }));
    }
    if args.len() != plan.input_slots.len() {
        return Some(Err(InterpreterError::InputArity {
            expected: plan.input_slots.len(),
            actual: args.len(),
        }));
    }

    cells.clear();
    cells.resize_with(plan.slots, || DenseF64Cell::Missing);

    // Load consts + inputs; the body shape comes from the dense-f64 tensors (all must
    // agree). A body with NO tensor input is scalar — let the scalar plan own it.
    let mut shape: Option<Shape> = None;
    let mut n: usize = 0;
    for (&slot, value) in plan
        .const_slots
        .iter()
        .zip(const_values)
        .chain(plan.input_slots.iter().zip(args))
    {
        cells[slot] = match value {
            Value::Scalar(Literal::F64Bits(b)) => DenseF64Cell::Scalar(f64::from_bits(*b)),
            Value::Tensor(t) => match t.elements.as_f64_slice() {
                Some(s) => match &shape {
                    None => {
                        shape = Some(t.shape.clone());
                        n = s.len();
                        DenseF64Cell::Tensor(s.to_vec())
                    }
                    Some(sh) if *sh == t.shape => DenseF64Cell::Tensor(s.to_vec()),
                    // Smaller operand that rank-2 row/col-broadcasts against the body
                    // shape — bit-identical to the generic + fusion broadcast.
                    Some(full) => {
                        if let Some(cols) = row_broadcast_len(full, &t.shape) {
                            DenseF64Cell::RowBcast(s.to_vec(), cols)
                        } else {
                            // col-broadcast or incompatible (-> generic via `?`)
                            let cols = col_broadcast_cols(full, &t.shape)?;
                            DenseF64Cell::ColBcast(s.to_vec(), cols)
                        }
                    }
                },
                None => DenseF64Cell::NonDense,
            },
            _ => DenseF64Cell::NonDense,
        };
    }

    let shape = shape?;
    // The in-place linear-chain path mutates ONE buffer through the chain — no per-step
    // alloc and single-stream (in-place) memory traffic, vs the generic per-op path's N
    // allocations and 2-stream (read+write) traffic per op. For an L3-resident chain
    // (e.g. an Adam-style optimizer body on a large parameter vector) that buffer reuse
    // is the documented compiled-jaxpr lever the per-op path lacks. It handles any size;
    // at large n it is gated behind `vectorize` so a bench can A/B it same-invocation
    // against the generic per-op path (which the per-step arena below bails to for
    // n >= FUSION_MIN_ELEMS). Bit-identical regardless of n.
    if (n < FUSION_MIN_ELEMS || (vectorize && n < INPLACE_CHAIN_MAX_ELEMS))
        && let Some(result) =
            run_linear_scalar_f64_tensor_chain_into(plan, &shape, out, cells, vectorize)
    {
        return Some(result);
    }
    // Large non-linear tensors (and DRAM-bound linear chains) are owned by the
    // chunked/threaded fusion.
    if n >= FUSION_MIN_ELEMS {
        return None;
    }

    for step in &plan.steps {
        let a = resolve_dense_operand(cells, step.lhs)?;
        let b = match step.rhs {
            Some(rhs) => resolve_dense_operand(cells, rhs)?,
            None => a,
        };
        let result = match (a, b) {
            (DenseRef::Scalar(av), DenseRef::Scalar(bv)) => {
                DenseF64Cell::Scalar(apply_scalar_f64_binary(step.op, av, bv))
            }
            _ => {
                let mut o = vec![0.0_f64; n];
                if a.is_bcast() || b.is_bcast() {
                    // Row-chunked so RowBcast/ColBcast index by col/row (no per-elem div).
                    let cols = *shape.dims.last().expect("rank>=1") as usize;
                    // Hoist the op once and vectorize each row (decomposed to a
                    // no-broadcast subproblem); other ops keep the per-element loop.
                    match step.op {
                        ScalarF64BinaryOp::Add if vectorize => {
                            fill_dense_f64_bcast(a, b, &mut o, cols, |x, y| x + y)
                        }
                        ScalarF64BinaryOp::Sub if vectorize => {
                            fill_dense_f64_bcast(a, b, &mut o, cols, |x, y| x - y)
                        }
                        ScalarF64BinaryOp::Mul if vectorize => {
                            fill_dense_f64_bcast(a, b, &mut o, cols, |x, y| x * y)
                        }
                        ScalarF64BinaryOp::Div if vectorize => {
                            fill_dense_f64_bcast(a, b, &mut o, cols, |x, y| x / y)
                        }
                        _ => {
                            let mut i = 0usize;
                            let mut row = 0usize;
                            while i < n {
                                for col in 0..cols {
                                    o[i] = apply_scalar_f64_binary(
                                        step.op,
                                        a.at_rc(row, col, i),
                                        b.at_rc(row, col, i),
                                    );
                                    i += 1;
                                }
                                row += 1;
                            }
                        }
                    }
                } else {
                    // Hoist the op + operand-kind matches out of the element loop so the
                    // common arithmetic ops auto-vectorize; other ops keep the generic
                    // per-element path. `vectorize` is the benchmark A/B control.
                    match step.op {
                        ScalarF64BinaryOp::Add if vectorize => {
                            fill_dense_f64_nobcast(a, b, &mut o, |x, y| x + y)
                        }
                        ScalarF64BinaryOp::Sub if vectorize => {
                            fill_dense_f64_nobcast(a, b, &mut o, |x, y| x - y)
                        }
                        ScalarF64BinaryOp::Mul if vectorize => {
                            fill_dense_f64_nobcast(a, b, &mut o, |x, y| x * y)
                        }
                        ScalarF64BinaryOp::Div if vectorize => {
                            fill_dense_f64_nobcast(a, b, &mut o, |x, y| x / y)
                        }
                        _ => {
                            for (i, slot) in o.iter_mut().enumerate() {
                                *slot = apply_scalar_f64_binary(step.op, a.at(i), b.at(i));
                            }
                        }
                    }
                }
                DenseF64Cell::Tensor(o)
            }
        };
        cells[step.out_slot] = result;
    }

    out.clear();
    out.reserve(plan.out_slots.len());
    for &slot in &plan.out_slots {
        match std::mem::replace(&mut cells[slot], DenseF64Cell::Missing) {
            DenseF64Cell::Tensor(values) => {
                match TensorValue::new_f64_values(shape.clone(), values) {
                    Ok(t) => out.push(Value::Tensor(t)),
                    Err(e) => {
                        return Some(Err(InterpreterError::Primitive(EvalError::InvalidTensor(
                            e,
                        ))));
                    }
                }
            }
            DenseF64Cell::Scalar(v) => out.push(Value::scalar_f64(v)),
            // A broadcast operand passed straight through as an output keeps its own
            // (smaller) shape, which this runner doesn't reconstruct — let generic do it.
            DenseF64Cell::NonDense | DenseF64Cell::RowBcast(..) | DenseF64Cell::ColBcast(..) => {
                return None;
            }
            DenseF64Cell::Missing => {
                return Some(Err(InterpreterError::MissingVariable(VarId(slot as u32))));
            }
        }
    }
    Some(Ok(()))
}

// f32 sibling of the dense-f64 small-tensor arena — f32 is JAX's DEFAULT tensor dtype,
// so a small-f32-tensor scan/while carry (RNN state / optimizer moments, the common ML
// case) is exactly this path. Reuses the already-built `scalar_f32_plan` run elementwise
// via the SAME `apply_scalar_f32_binary` (widen->f64-op->narrow, bit-identical to the
// generic f32 per-element path). Same gating: same-shape dense-f32 body below
// FUSION_MIN_ELEMS; non-dense / mixed-shape / large bails to generic.

enum DenseF32Cell {
    Missing,
    NonDense,
    Scalar(f32),
    Tensor(Vec<f32>),
}

#[derive(Clone, Copy)]
enum DenseF32Ref<'a> {
    Scalar(f32),
    Tensor(&'a [f32]),
}

fn resolve_dense_f32_operand<'a>(
    cells: &'a [DenseF32Cell],
    operand: ScalarF32Operand,
) -> Option<DenseF32Ref<'a>> {
    match operand {
        ScalarF32Operand::Literal(v) => Some(DenseF32Ref::Scalar(v)),
        ScalarF32Operand::Slot(s) => match &cells[s] {
            DenseF32Cell::Scalar(v) => Some(DenseF32Ref::Scalar(*v)),
            DenseF32Cell::Tensor(t) => Some(DenseF32Ref::Tensor(t)),
            DenseF32Cell::Missing | DenseF32Cell::NonDense => None,
        },
    }
}

/// f32 sibling of [`fill_dense_f64_nobcast`]. The op closure is monomorphized and the
/// operand kind matched ONCE, so Add/Sub/Mul/Div become a native-f32 `o[i]=a[i] OP b[i]`
/// loop that auto-vectorizes to `vaddps` (8-wide under `+avx2`). BIT-EXACT vs eager's
/// widen→f64-op→narrow: for a single +/-/*/÷, f64 (53-bit mantissa) carries ≥ 2·24+2 bits
/// so `(f64(a) OP f64(b)) as f32 == a OP b` in native f32 (Figueroa: no double rounding).
#[inline]
fn fill_dense_f32_nobcast<F: Fn(f32, f32) -> f32>(
    a: DenseF32Ref,
    b: DenseF32Ref,
    o: &mut [f32],
    f: F,
) {
    match (a, b) {
        (DenseF32Ref::Tensor(ta), DenseF32Ref::Scalar(sb)) => {
            for (o, &x) in o.iter_mut().zip(ta) {
                *o = f(x, sb);
            }
        }
        (DenseF32Ref::Scalar(sa), DenseF32Ref::Tensor(tb)) => {
            for (o, &y) in o.iter_mut().zip(tb) {
                *o = f(sa, y);
            }
        }
        (DenseF32Ref::Tensor(ta), DenseF32Ref::Tensor(tb)) => {
            for ((o, &x), &y) in o.iter_mut().zip(ta).zip(tb) {
                *o = f(x, y);
            }
        }
        // (Scalar,Scalar) is handled by the caller before this is reached.
        (DenseF32Ref::Scalar(sa), DenseF32Ref::Scalar(sb)) => {
            o.iter_mut().for_each(|o| *o = f(sa, sb))
        }
    }
}

#[inline]
fn operand_is_slot_f32(operand: ScalarF32Operand, slot: usize) -> bool {
    matches!(operand, ScalarF32Operand::Slot(s) if s == slot)
}

fn scalar_dense_f32_operand(cells: &[DenseF32Cell], operand: ScalarF32Operand) -> Option<f32> {
    match operand {
        ScalarF32Operand::Literal(v) => Some(v),
        ScalarF32Operand::Slot(s) => match cells.get(s)? {
            DenseF32Cell::Scalar(v) => Some(*v),
            _ => None,
        },
    }
}

/// In-place f32 linear-chain step: `v[i] = v[i] OP scalar` (or `scalar OP v[i]`). Native
/// f32 +/-/*/÷ vectorize (`vaddps`) and are BIT-EXACT vs eager's widen->f64-op->narrow for
/// a single op (Figueroa). Other ops keep the generic per-element widen path.
#[inline]
fn apply_dense_f32_chain_step(
    values: &mut [f32],
    op: ScalarF64BinaryOp,
    scalar: f32,
    scalar_on_left: bool,
    vectorize: bool,
) {
    match (op, scalar_on_left) {
        (ScalarF64BinaryOp::Add, _) if vectorize => values.iter_mut().for_each(|v| *v += scalar),
        (ScalarF64BinaryOp::Mul, _) if vectorize => values.iter_mut().for_each(|v| *v *= scalar),
        (ScalarF64BinaryOp::Sub, false) if vectorize => {
            values.iter_mut().for_each(|v| *v -= scalar)
        }
        (ScalarF64BinaryOp::Sub, true) if vectorize => {
            values.iter_mut().for_each(|v| *v = scalar - *v)
        }
        (ScalarF64BinaryOp::Div, false) if vectorize => {
            values.iter_mut().for_each(|v| *v /= scalar)
        }
        (ScalarF64BinaryOp::Div, true) if vectorize => {
            values.iter_mut().for_each(|v| *v = scalar / *v)
        }
        (op, false) => values
            .iter_mut()
            .for_each(|v| *v = apply_scalar_f32_binary(op, *v, scalar)),
        (op, true) => values
            .iter_mut()
            .for_each(|v| *v = apply_scalar_f32_binary(op, scalar, *v)),
    }
}

/// f32 sibling of [`run_linear_scalar_f64_tensor_chain_into`] (the L3-resident buffer-reuse
/// lever for JAX's DEFAULT dtype). f32 cells carry no broadcast variants, so the chain's
/// other operand is always scalar/literal. Mutates one buffer in place through the chain;
/// bit-identical to the generic per-op path.
fn run_linear_scalar_f32_tensor_chain_into(
    plan: &ScalarF32Plan,
    shape: &Shape,
    out: &mut Vec<Value>,
    cells: &mut [DenseF32Cell],
    vectorize: bool,
) -> Option<Result<(), InterpreterError>> {
    if plan.out_slots.len() != 1 || plan.steps.is_empty() {
        return None;
    }
    let mut tensor_slot = None;
    for (slot, cell) in cells.iter().enumerate() {
        if matches!(cell, DenseF32Cell::Tensor(_)) {
            if tensor_slot.is_some() {
                return None;
            }
            tensor_slot = Some(slot);
        }
    }
    let tensor_slot = tensor_slot?;
    let mut current_slot = tensor_slot;
    for step in &plan.steps {
        let lhs_is_current = operand_is_slot_f32(step.lhs, current_slot);
        match step.rhs {
            Some(rhs) => {
                let rhs_is_current = operand_is_slot_f32(rhs, current_slot);
                match (lhs_is_current, rhs_is_current) {
                    (true, true) => {}
                    (true, false) => {
                        scalar_dense_f32_operand(cells, rhs)?;
                    }
                    (false, true) => {
                        scalar_dense_f32_operand(cells, step.lhs)?;
                    }
                    (false, false) => return None,
                }
            }
            None => {
                if !lhs_is_current {
                    return None;
                }
            }
        }
        current_slot = step.out_slot;
    }
    if plan.out_slots[0] != current_slot {
        return None;
    }

    let DenseF32Cell::Tensor(mut values) =
        std::mem::replace(&mut cells[tensor_slot], DenseF32Cell::Missing)
    else {
        unreachable!("validated single dense-f32 tensor slot")
    };

    let mut current_slot = tensor_slot;
    for step in &plan.steps {
        let lhs_is_current = operand_is_slot_f32(step.lhs, current_slot);
        match step.rhs {
            Some(rhs) => {
                let rhs_is_current = operand_is_slot_f32(rhs, current_slot);
                match (lhs_is_current, rhs_is_current) {
                    (true, true) => {
                        for v in &mut values {
                            *v = apply_scalar_f32_binary(step.op, *v, *v);
                        }
                    }
                    (true, false) => {
                        let s = scalar_dense_f32_operand(cells, rhs)?;
                        apply_dense_f32_chain_step(&mut values, step.op, s, false, vectorize);
                    }
                    (false, true) => {
                        let s = scalar_dense_f32_operand(cells, step.lhs)?;
                        apply_dense_f32_chain_step(&mut values, step.op, s, true, vectorize);
                    }
                    (false, false) => unreachable!("validated linear tensor chain"),
                }
            }
            None => {
                debug_assert!(lhs_is_current);
                for v in &mut values {
                    *v = apply_scalar_f32_binary(step.op, *v, *v);
                }
            }
        }
        current_slot = step.out_slot;
    }

    let tensor = match TensorValue::new_f32_values(shape.clone(), values) {
        Ok(t) => t,
        Err(e) => {
            return Some(Err(InterpreterError::Primitive(EvalError::InvalidTensor(
                e,
            ))));
        }
    };
    out.clear();
    out.push(Value::Tensor(tensor));
    Some(Ok(()))
}

fn run_scalar_f32_plan_as_tensor_into(
    plan: &ScalarF32Plan,
    const_values: &[Value],
    args: &[Value],
    out: &mut Vec<Value>,
    cells: &mut Vec<DenseF32Cell>,
    vectorize: bool,
) -> Option<Result<(), InterpreterError>> {
    if const_values.len() != plan.const_slots.len() {
        return Some(Err(InterpreterError::ConstArity {
            expected: plan.const_slots.len(),
            actual: const_values.len(),
        }));
    }
    if args.len() != plan.input_slots.len() {
        return Some(Err(InterpreterError::InputArity {
            expected: plan.input_slots.len(),
            actual: args.len(),
        }));
    }

    cells.clear();
    cells.resize_with(plan.slots, || DenseF32Cell::Missing);

    let mut shape: Option<Shape> = None;
    let mut n: usize = 0;
    for (&slot, value) in plan
        .const_slots
        .iter()
        .zip(const_values)
        .chain(plan.input_slots.iter().zip(args))
    {
        cells[slot] = match value {
            Value::Scalar(Literal::F32Bits(b)) => DenseF32Cell::Scalar(f32::from_bits(*b)),
            Value::Tensor(t) => match t.elements.as_f32_slice() {
                Some(s) => {
                    match &shape {
                        None => {
                            shape = Some(t.shape.clone());
                            n = s.len();
                        }
                        Some(sh) if *sh == t.shape => {}
                        Some(_) => return None,
                    }
                    DenseF32Cell::Tensor(s.to_vec())
                }
                None => DenseF32Cell::NonDense,
            },
            _ => DenseF32Cell::NonDense,
        };
    }

    let shape = shape?;
    // In-place f32 linear chain (L3-resident buffer-reuse lever for JAX's default dtype).
    // Same gate/A/B structure as f64: handles any size up to the L3 ceiling when vectorize
    // is on; large non-linear bodies bail to the chunked/threaded fusion.
    if (n < FUSION_MIN_ELEMS || (vectorize && n < INPLACE_CHAIN_MAX_ELEMS))
        && let Some(result) =
            run_linear_scalar_f32_tensor_chain_into(plan, &shape, out, cells, vectorize)
    {
        return Some(result);
    }
    if n >= FUSION_MIN_ELEMS {
        return None;
    }

    for step in &plan.steps {
        let a = resolve_dense_f32_operand(cells, step.lhs)?;
        let b = match step.rhs {
            Some(rhs) => resolve_dense_f32_operand(cells, rhs)?,
            None => a,
        };
        let result = match (a, b) {
            (DenseF32Ref::Scalar(av), DenseF32Ref::Scalar(bv)) => {
                DenseF32Cell::Scalar(apply_scalar_f32_binary(step.op, av, bv))
            }
            _ => {
                let mut o = vec![0.0_f32; n];
                // Hoist the op match so the cheap ops vectorize (native f32 == eager's
                // widen→f64→narrow for +/-/*/÷; see fill_dense_f32_nobcast). `vectorize`
                // is the benchmark A/B control; other ops keep the generic widen path.
                match step.op {
                    ScalarF64BinaryOp::Add if vectorize => {
                        fill_dense_f32_nobcast(a, b, &mut o, |x, y| x + y)
                    }
                    ScalarF64BinaryOp::Sub if vectorize => {
                        fill_dense_f32_nobcast(a, b, &mut o, |x, y| x - y)
                    }
                    ScalarF64BinaryOp::Mul if vectorize => {
                        fill_dense_f32_nobcast(a, b, &mut o, |x, y| x * y)
                    }
                    ScalarF64BinaryOp::Div if vectorize => {
                        fill_dense_f32_nobcast(a, b, &mut o, |x, y| x / y)
                    }
                    _ => {
                        for (i, slot) in o.iter_mut().enumerate() {
                            let av = match a {
                                DenseF32Ref::Scalar(v) => v,
                                DenseF32Ref::Tensor(t) => t[i],
                            };
                            let bv = match b {
                                DenseF32Ref::Scalar(v) => v,
                                DenseF32Ref::Tensor(t) => t[i],
                            };
                            *slot = apply_scalar_f32_binary(step.op, av, bv);
                        }
                    }
                }
                DenseF32Cell::Tensor(o)
            }
        };
        cells[step.out_slot] = result;
    }

    out.clear();
    out.reserve(plan.out_slots.len());
    for &slot in &plan.out_slots {
        match std::mem::replace(&mut cells[slot], DenseF32Cell::Missing) {
            DenseF32Cell::Tensor(values) => {
                match TensorValue::new_f32_values(shape.clone(), values) {
                    Ok(t) => out.push(Value::Tensor(t)),
                    Err(e) => {
                        return Some(Err(InterpreterError::Primitive(EvalError::InvalidTensor(
                            e,
                        ))));
                    }
                }
            }
            DenseF32Cell::Scalar(v) => out.push(Value::scalar_f32(v)),
            DenseF32Cell::NonDense => return None,
            DenseF32Cell::Missing => {
                return Some(Err(InterpreterError::MissingVariable(VarId(slot as u32))));
            }
        }
    }
    Some(Ok(()))
}

// i64 sibling of the dense-float small-tensor arenas — for integer small-tensor scan/
// while carries (index buffers, masks, counters). Reuses the already-built
// `scalar_i64_plan` (whose ops are the integer-valid subset Add/Sub/Mul/Div/Max/Min/
// Neg/Abs) run elementwise via the SAME `apply_scalar_i64_binary` (wrapping arithmetic),
// bit-identical to the generic i64 per-element path. GUARD: only `dtype == I64` tensors
// — an I32 tensor is i64-backed but the generic NARROWS its results (narrow_i32_tensor_
// result), which this arena does not, so I32 must bail to the generic interpreter.

enum DenseI64Cell {
    Missing,
    NonDense,
    Scalar(i64),
    Tensor(Vec<i64>),
}

#[derive(Clone, Copy)]
enum DenseI64Ref<'a> {
    Scalar(i64),
    Tensor(&'a [i64]),
}

fn resolve_dense_i64_operand<'a>(
    cells: &'a [DenseI64Cell],
    operand: ScalarI64Operand,
) -> Option<DenseI64Ref<'a>> {
    match operand {
        ScalarI64Operand::Literal(v) => Some(DenseI64Ref::Scalar(v)),
        ScalarI64Operand::Slot(s) => match &cells[s] {
            DenseI64Cell::Scalar(v) => Some(DenseI64Ref::Scalar(*v)),
            DenseI64Cell::Tensor(t) => Some(DenseI64Ref::Tensor(t)),
            DenseI64Cell::Missing | DenseI64Cell::NonDense => None,
        },
    }
}

/// i64 sibling of [`fill_dense_f64_nobcast`]. The op closure is monomorphized and the
/// operand kind matched ONCE, so the wrapping integer ops Add/Sub/Mul become a native
/// `o[i] = a[i] OP b[i]` loop that auto-vectorizes to `vpaddq`/`vpsubq`/`vpmullq` under
/// `+avx2`. Bit-exact: `wrapping_add/sub/mul` ARE the per-element op the generic path
/// applies. (Div/Max/Min/Neg/Abs keep the generic loop — integer div has no SIMD form.)
#[inline]
fn fill_dense_i64_nobcast<F: Fn(i64, i64) -> i64>(
    a: DenseI64Ref,
    b: DenseI64Ref,
    o: &mut [i64],
    f: F,
) {
    match (a, b) {
        (DenseI64Ref::Tensor(ta), DenseI64Ref::Scalar(sb)) => {
            for (o, &x) in o.iter_mut().zip(ta) {
                *o = f(x, sb);
            }
        }
        (DenseI64Ref::Scalar(sa), DenseI64Ref::Tensor(tb)) => {
            for (o, &y) in o.iter_mut().zip(tb) {
                *o = f(sa, y);
            }
        }
        (DenseI64Ref::Tensor(ta), DenseI64Ref::Tensor(tb)) => {
            for ((o, &x), &y) in o.iter_mut().zip(ta).zip(tb) {
                *o = f(x, y);
            }
        }
        (DenseI64Ref::Scalar(sa), DenseI64Ref::Scalar(sb)) => {
            o.iter_mut().for_each(|o| *o = f(sa, sb))
        }
    }
}

fn run_scalar_i64_plan_as_tensor_into(
    plan: &ScalarI64Plan,
    const_values: &[Value],
    args: &[Value],
    out: &mut Vec<Value>,
    cells: &mut Vec<DenseI64Cell>,
    vectorize: bool,
) -> Option<Result<(), InterpreterError>> {
    if const_values.len() != plan.const_slots.len() {
        return Some(Err(InterpreterError::ConstArity {
            expected: plan.const_slots.len(),
            actual: const_values.len(),
        }));
    }
    if args.len() != plan.input_slots.len() {
        return Some(Err(InterpreterError::InputArity {
            expected: plan.input_slots.len(),
            actual: args.len(),
        }));
    }

    cells.clear();
    cells.resize_with(plan.slots, || DenseI64Cell::Missing);

    let mut shape: Option<Shape> = None;
    let mut n: usize = 0;
    for (&slot, value) in plan
        .const_slots
        .iter()
        .zip(const_values)
        .chain(plan.input_slots.iter().zip(args))
    {
        cells[slot] = match value {
            Value::Scalar(Literal::I64(v)) => DenseI64Cell::Scalar(*v),
            // Only genuine I64 tensors — I32 is i64-backed but generic-narrowed.
            Value::Tensor(t) if t.dtype == DType::I64 => match t.elements.as_i64_slice() {
                Some(s) => {
                    match &shape {
                        None => {
                            shape = Some(t.shape.clone());
                            n = s.len();
                        }
                        Some(sh) if *sh == t.shape => {}
                        Some(_) => return None,
                    }
                    DenseI64Cell::Tensor(s.to_vec())
                }
                None => DenseI64Cell::NonDense,
            },
            _ => DenseI64Cell::NonDense,
        };
    }

    let shape = shape?;
    if n >= FUSION_MIN_ELEMS {
        return None;
    }

    for step in &plan.steps {
        let a = resolve_dense_i64_operand(cells, step.lhs)?;
        let b = match step.rhs {
            Some(rhs) => resolve_dense_i64_operand(cells, rhs)?,
            None => a,
        };
        let result = match (a, b) {
            (DenseI64Ref::Scalar(av), DenseI64Ref::Scalar(bv)) => {
                DenseI64Cell::Scalar(apply_scalar_i64_binary(step.op, av, bv))
            }
            _ => {
                let mut o = vec![0_i64; n];
                // Hoist the op match so the wrapping integer ops vectorize (native ==
                // generic per-element). `vectorize` is the benchmark A/B control; other
                // ops (Div/Max/Min/Neg/Abs) keep the generic per-element loop.
                match step.op {
                    ScalarF64BinaryOp::Add if vectorize => {
                        fill_dense_i64_nobcast(a, b, &mut o, i64::wrapping_add)
                    }
                    ScalarF64BinaryOp::Sub if vectorize => {
                        fill_dense_i64_nobcast(a, b, &mut o, i64::wrapping_sub)
                    }
                    ScalarF64BinaryOp::Mul if vectorize => {
                        fill_dense_i64_nobcast(a, b, &mut o, i64::wrapping_mul)
                    }
                    _ => {
                        for (i, slot) in o.iter_mut().enumerate() {
                            let av = match a {
                                DenseI64Ref::Scalar(v) => v,
                                DenseI64Ref::Tensor(t) => t[i],
                            };
                            let bv = match b {
                                DenseI64Ref::Scalar(v) => v,
                                DenseI64Ref::Tensor(t) => t[i],
                            };
                            *slot = apply_scalar_i64_binary(step.op, av, bv);
                        }
                    }
                }
                DenseI64Cell::Tensor(o)
            }
        };
        cells[step.out_slot] = result;
    }

    out.clear();
    out.reserve(plan.out_slots.len());
    for &slot in &plan.out_slots {
        match std::mem::replace(&mut cells[slot], DenseI64Cell::Missing) {
            DenseI64Cell::Tensor(values) => {
                match TensorValue::new_i64_values(shape.clone(), values) {
                    Ok(t) => out.push(Value::Tensor(t)),
                    Err(e) => {
                        return Some(Err(InterpreterError::Primitive(EvalError::InvalidTensor(
                            e,
                        ))));
                    }
                }
            }
            DenseI64Cell::Scalar(v) => out.push(Value::scalar_i64(v)),
            DenseI64Cell::NonDense => return None,
            DenseI64Cell::Missing => {
                return Some(Err(InterpreterError::MissingVariable(VarId(slot as u32))));
            }
        }
    }
    Some(Ok(()))
}

/// Reusable scratch arenas for monomorphic scalar executors. A loop body builds
/// only the plan(s) whose literals and boolean intermediates match its shape,
/// and the runner tries them in order; each non-matching plan bails on the first
/// unsupported operand read, so the unused buffers stay empty.
#[derive(Default)]
struct ScalarPlanBuffers {
    f64: Vec<ScalarF64Slot>,
    i64: Vec<ScalarI64Slot>,
    f32: Vec<ScalarF32Slot>,
    bf16: Vec<ScalarHalfSlot>,
    f16: Vec<ScalarHalfSlot>,
    bools: Vec<Option<bool>>,
    mixed: Vec<MixedSlot>,
    mixed_i64: Vec<MixedI64Slot>,
    poly: Vec<PolySlot>,
    dense_f64: Vec<DenseF64Cell>,
    dense_f32: Vec<DenseF32Cell>,
    dense_i64: Vec<DenseI64Cell>,
}

/// A reusable dense-evaluation plan for a jaxpr: the slot count and the
/// value-independent liveness map. Build it ONCE with [`build_dense_plan`] and
/// feed it (plus a reusable `env` buffer) to [`run_dense_env`] on every call —
/// this is what lets a `while`/`scan` body skip re-deriving its slot/liveness
/// analysis (≈the bulk of the per-iteration dispatch overhead) on each iteration.
struct DenseEvalPlan {
    slots: usize,
    last_use: Vec<usize>,
    scalar_f64_plan: Option<ScalarF64Plan>,
    scalar_i64_plan: Option<ScalarI64Plan>,
    scalar_f32_plan: Option<ScalarF32Plan>,
    scalar_bf16_plan: Option<ScalarHalfPlan>,
    scalar_f16_plan: Option<ScalarHalfPlan>,
    scalar_compare_plan: Option<ScalarComparePlan>,
    scalar_compound_compare_plan: Option<ScalarCompoundComparePlan>,
    scalar_select_plan: Option<ScalarSelectPlan>,
    scalar_select_i64_plan: Option<ScalarSelectI64Plan>,
    scalar_poly_plan: Option<ScalarPolyPlan>,
    dense_f64_reduce_sum_plan: Option<DenseF64ReduceSumPlan>,
    dense_f64_dot_plan: Option<DenseF64DotPlan>,
    dense_f64_transpose_plan: Option<DenseF64TransposePlan>,
    dense_f64_broadcast_in_dim_plan: Option<DenseF64BroadcastInDimPlan>,
    dense_reshape_plan: Option<DenseReshapePlan>,
    dense_axis_reduce_plan: Option<DenseAxisReducePlan>,
    dense_gather_plan: Option<DenseGatherPlan>,
    dense_arg_extremum_plan: Option<DenseArgExtremumPlan>,
}

/// One-time compiled evaluator for hot repeated calls of the same small Jaxpr.
///
/// The compiled form reuses the existing dense slot plan and scalar step plans;
/// it does not change primitive order, tie-breaking, floating-point evaluation,
/// or RNG/effect behavior. Programs outside the pure scalar/dense subset return
/// `None` from [`compile_jaxpr_for_repeated_eval`] and should use
/// [`eval_jaxpr_with_consts`] or the backend scheduler.
pub struct CompiledJaxpr {
    jaxpr: Jaxpr,
    plan: DenseEvalPlan,
}

impl CompiledJaxpr {
    pub fn eval(&self, args: &[Value]) -> Result<Vec<Value>, InterpreterError> {
        if args.len() != self.jaxpr.invars.len() {
            return Err(InterpreterError::InputArity {
                expected: self.jaxpr.invars.len(),
                actual: args.len(),
            });
        }

        let mut env: Vec<Option<Value>> = vec![None; self.plan.slots];
        run_dense_plan(&self.jaxpr, &[], args, &mut env, &self.plan)
    }

    #[must_use]
    pub fn runner(&self) -> CompiledJaxprRunner<'_> {
        CompiledJaxprRunner {
            compiled: self,
            env: vec![None; self.plan.slots],
            scratch: Vec::new(),
            out: Vec::new(),
            scalar_buffers: ScalarPlanBuffers::default(),
        }
    }
}

/// Reusable arena for hot repeated evaluations of one [`CompiledJaxpr`].
///
/// [`CompiledJaxpr::eval`] preserves the simple owned-`Vec` API, but it must build
/// its slot environment and scratch buffers for each call. A runner keeps those
/// allocations warm across calls and writes each result into its retained output
/// vector. The primitive order and dense plan are exactly the same as
/// [`CompiledJaxpr::eval`]; only allocation ownership changes.
pub struct CompiledJaxprRunner<'a> {
    compiled: &'a CompiledJaxpr,
    env: Vec<Option<Value>>,
    scratch: Vec<Value>,
    out: Vec<Value>,
    scalar_buffers: ScalarPlanBuffers,
}

impl<'a> CompiledJaxprRunner<'a> {
    pub fn eval(&mut self, args: &[Value]) -> Result<&[Value], InterpreterError> {
        if args.len() != self.compiled.jaxpr.invars.len() {
            return Err(InterpreterError::InputArity {
                expected: self.compiled.jaxpr.invars.len(),
                actual: args.len(),
            });
        }

        if let Some(plan) = &self.compiled.plan.scalar_f64_plan
            && let Some(elems) = scalar_f64_plan_single_tensor_linear_chain_elems(plan, args)
            && (RUNNER_F64_OWNED_EVAL_MIN_ELEMS..=RUNNER_F64_OWNED_EVAL_MAX_ELEMS).contains(&elems)
        {
            self.out = self.compiled.eval(args)?;
            return Ok(&self.out);
        }

        run_dense_plan_into(
            &self.compiled.jaxpr,
            &[],
            args,
            &mut self.env,
            &self.compiled.plan,
            &mut self.scratch,
            &mut self.out,
            &mut self.scalar_buffers,
        )?;
        Ok(&self.out)
    }

    pub fn eval_owned(&mut self, args: &[Value]) -> Result<Vec<Value>, InterpreterError> {
        self.eval(args)?;
        Ok(self.out.clone())
    }

    /// Benchmark-only control: evaluate with the dense-f64 inner-loop vectorization
    /// DISABLED (generic per-element `apply_scalar_f64_binary` loop), so a bench can A/B
    /// the vectorized path against it in the same binary. Behaviourally identical to
    /// [`eval`](Self::eval); use [`eval`](Self::eval) in production.
    #[doc(hidden)]
    pub fn eval_scalar_inner(&mut self, args: &[Value]) -> Result<&[Value], InterpreterError> {
        if args.len() != self.compiled.jaxpr.invars.len() {
            return Err(InterpreterError::InputArity {
                expected: self.compiled.jaxpr.invars.len(),
                actual: args.len(),
            });
        }

        run_dense_plan_into_core(
            &self.compiled.jaxpr,
            &[],
            args,
            &mut self.env,
            &self.compiled.plan,
            &mut self.scratch,
            &mut self.out,
            &mut self.scalar_buffers,
            false,
        )?;
        Ok(&self.out)
    }
}

/// Compile a pure dense Jaxpr for repeated evaluation.
///
/// This deliberately accepts only no-const, effect-free, sub-jaxpr-free programs
/// with unique bindings and a buildable dense slot plan. Everything else keeps
/// the normal interpreter/backend route so malformed programs keep their
/// existing behavior.
///
/// The compiled form caches the [`DenseEvalPlan`] (slot-indexed env layout,
/// per-equation last-use liveness, and the pre-scanned scalar step plans) so a
/// hot repeated call pays the plan-build cost once instead of per call. It runs
/// through [`run_dense_plan`], whose generic per-equation fallback
/// ([`run_dense_env_into`]) dispatches the same `eval_primitive` kernels in the
/// same order as [`eval_jaxpr_with_consts`] — so the result is bit-for-bit
/// identical, scalar *or* tensor. Earlier this gated on `has_scalar_fast_path`,
/// which left small tensor/reduction programs paying the full per-call dispatch
/// tax; that gate is gone because the generic dense path is an equally exact
/// drop-in.
#[must_use]
pub fn compile_jaxpr_for_repeated_eval(jaxpr: &Jaxpr) -> Option<CompiledJaxpr> {
    if !jaxpr.constvars.is_empty() || !jaxpr.effects.is_empty() {
        return None;
    }
    if !jaxpr_is_uniquely_bound_effect_free(jaxpr) {
        return None;
    }

    let plan = build_dense_plan(jaxpr)?;

    Some(CompiledJaxpr {
        jaxpr: jaxpr.clone(),
        plan,
    })
}

fn jaxpr_is_uniquely_bound_effect_free(jaxpr: &Jaxpr) -> bool {
    let mut bindings = BTreeSet::new();
    for var in jaxpr.constvars.iter().chain(jaxpr.invars.iter()) {
        if !bindings.insert(*var) {
            return false;
        }
    }
    for equation in &jaxpr.equations {
        if !equation.effects.is_empty() || !equation.sub_jaxprs.is_empty() {
            return false;
        }
        if equation.inputs.iter().any(|atom| match atom {
            Atom::Var(var) => !bindings.contains(var),
            Atom::Lit(_) => false,
        }) {
            return false;
        }
        for out_var in &equation.outputs {
            if !bindings.insert(*out_var) {
                return false;
            }
        }
    }

    let mut seen_outputs = BTreeSet::new();
    jaxpr
        .outvars
        .iter()
        .all(|var| seen_outputs.insert(*var) && bindings.contains(var))
}

/// Build a [`DenseEvalPlan`] iff the jaxpr's variable ids are dense enough for the
/// flat-`Vec` environment (same gate as [`eval_jaxpr_with_consts`]); returns
/// `None` for sparse/huge id ranges so the caller keeps the hash-map fallback.
fn build_dense_plan(jaxpr: &Jaxpr) -> Option<DenseEvalPlan> {
    let mut max_var: u32 = 0;
    let mut def_count: usize = 0;
    for var in jaxpr.constvars.iter().chain(jaxpr.invars.iter()) {
        max_var = max_var.max(var.0);
        def_count += 1;
    }
    for eqn in &jaxpr.equations {
        for out_var in &eqn.outputs {
            max_var = max_var.max(out_var.0);
            def_count += 1;
        }
    }
    let slots = max_var as usize + 1;
    if slots <= def_count.saturating_mul(8).max(256) {
        Some(DenseEvalPlan {
            slots,
            last_use: compute_dense_last_use(jaxpr, slots),
            scalar_f64_plan: build_scalar_f64_arith_plan(jaxpr, slots),
            scalar_i64_plan: build_scalar_i64_arith_plan(jaxpr, slots),
            scalar_f32_plan: build_scalar_f32_arith_plan(jaxpr, slots),
            scalar_bf16_plan: build_scalar_half_arith_plan(jaxpr, slots, DType::BF16),
            scalar_f16_plan: build_scalar_half_arith_plan(jaxpr, slots, DType::F16),
            scalar_compare_plan: build_scalar_compare_plan(jaxpr),
            scalar_compound_compare_plan: build_scalar_compound_compare_plan(jaxpr, slots),
            scalar_select_plan: build_scalar_select_plan(jaxpr, slots),
            scalar_select_i64_plan: build_scalar_select_i64_plan(jaxpr, slots),
            scalar_poly_plan: build_scalar_poly_plan(jaxpr, slots),
            dense_f64_reduce_sum_plan: build_dense_f64_reduce_sum_plan(jaxpr, slots),
            dense_f64_dot_plan: build_dense_f64_dot_plan(jaxpr, slots),
            dense_f64_transpose_plan: build_dense_f64_transpose_plan(jaxpr, slots),
            dense_f64_broadcast_in_dim_plan: build_dense_f64_broadcast_in_dim_plan(jaxpr, slots),
            dense_reshape_plan: build_dense_reshape_plan(jaxpr, slots),
            dense_axis_reduce_plan: build_dense_axis_reduce_plan(jaxpr, slots),
            dense_gather_plan: build_dense_gather_plan(jaxpr, slots),
            dense_arg_extremum_plan: build_dense_arg_extremum_plan(jaxpr, slots),
        })
    } else {
        None
    }
}

fn eval_jaxpr_dense_env(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
    slots: usize,
) -> Result<Vec<Value>, InterpreterError> {
    let plan = DenseEvalPlan {
        slots,
        last_use: compute_dense_last_use(jaxpr, slots),
        scalar_f64_plan: build_scalar_f64_arith_plan(jaxpr, slots),
        scalar_i64_plan: build_scalar_i64_arith_plan(jaxpr, slots),
        scalar_f32_plan: build_scalar_f32_arith_plan(jaxpr, slots),
        scalar_bf16_plan: build_scalar_half_arith_plan(jaxpr, slots, DType::BF16),
        scalar_f16_plan: build_scalar_half_arith_plan(jaxpr, slots, DType::F16),
        scalar_compare_plan: build_scalar_compare_plan(jaxpr),
        scalar_compound_compare_plan: build_scalar_compound_compare_plan(jaxpr, slots),
        scalar_select_plan: build_scalar_select_plan(jaxpr, slots),
        scalar_select_i64_plan: build_scalar_select_i64_plan(jaxpr, slots),
        scalar_poly_plan: build_scalar_poly_plan(jaxpr, slots),
        dense_f64_reduce_sum_plan: build_dense_f64_reduce_sum_plan(jaxpr, slots),
        dense_f64_dot_plan: build_dense_f64_dot_plan(jaxpr, slots),
        dense_f64_transpose_plan: build_dense_f64_transpose_plan(jaxpr, slots),
        dense_f64_broadcast_in_dim_plan: build_dense_f64_broadcast_in_dim_plan(jaxpr, slots),
        dense_reshape_plan: build_dense_reshape_plan(jaxpr, slots),
        dense_axis_reduce_plan: build_dense_axis_reduce_plan(jaxpr, slots),
        dense_gather_plan: build_dense_gather_plan(jaxpr, slots),
        dense_arg_extremum_plan: build_dense_arg_extremum_plan(jaxpr, slots),
    };
    let mut env: Vec<Option<Value>> = vec![None; slots];
    run_dense_plan(jaxpr, const_values, args, &mut env, &plan)
}

fn run_dense_plan(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
    env: &mut [Option<Value>],
    plan: &DenseEvalPlan,
) -> Result<Vec<Value>, InterpreterError> {
    let mut scratch: Vec<Value> = Vec::new();
    let mut out: Vec<Value> = Vec::new();
    let mut scalar_buffers = ScalarPlanBuffers::default();
    run_dense_plan_into(
        jaxpr,
        const_values,
        args,
        env,
        plan,
        &mut scratch,
        &mut out,
        &mut scalar_buffers,
    )?;
    Ok(out)
}

fn dense_f64_reduce_sum_input<'a>(
    plan: &DenseF64ReduceSumPlan,
    const_values: &'a [Value],
    args: &'a [Value],
) -> Result<&'a Value, InterpreterError> {
    for (&slot, value) in plan.const_slots.iter().zip(const_values) {
        if slot == plan.input_slot {
            return Ok(value);
        }
    }
    for (&slot, value) in plan.input_slots.iter().zip(args) {
        if slot == plan.input_slot {
            return Ok(value);
        }
    }
    Err(InterpreterError::MissingVariable(VarId(
        plan.input_slot as u32,
    )))
}

fn run_dense_f64_reduce_sum_plan_into(
    plan: &DenseF64ReduceSumPlan,
    const_values: &[Value],
    args: &[Value],
    out: &mut Vec<Value>,
) -> Option<Result<(), InterpreterError>> {
    if const_values.len() != plan.const_slots.len() {
        return Some(Err(InterpreterError::ConstArity {
            expected: plan.const_slots.len(),
            actual: const_values.len(),
        }));
    }
    if args.len() != plan.input_slots.len() {
        return Some(Err(InterpreterError::InputArity {
            expected: plan.input_slots.len(),
            actual: args.len(),
        }));
    }

    let input = match dense_f64_reduce_sum_input(plan, const_values, args) {
        Ok(value) => value,
        Err(err) => return Some(Err(err)),
    };
    let Value::Tensor(tensor) = input else {
        return None;
    };
    // F64 and F32 dense full reduction (sum/prod/max/min, no axes). Both fold the
    // contiguous backing slice in ascending order with NO reassociation,
    // accumulating in f64 — exactly fj-lax's full-reduce path: the dense-F64 fast
    // path folds the f64 slice (max/min via SIMD that is bit-identical to the
    // scalar jax_max/jax_min fold), and F32 falls to the generic loop, which
    // widens each element to f64 (`Literal::as_f64`), seeds the op init, folds in
    // f64, then rounds back to F32 (`reduce_real_literal(F32, acc) ==
    // from_f32(acc as f32)`). For max/min the result is always one of the inputs,
    // so widen→fold→round round-trips exactly; NaN collapses to canonical NaN on
    // both sides. So the bits match either route. Other dtypes
    // (half/int/complex/non-dense) return None and keep the generic interpreter.
    match tensor.dtype {
        DType::F64 => {
            let values = tensor.elements.as_f64_slice()?;
            let reduced = if tensor.shape.rank() == 0 {
                *values.first()?
            } else {
                let mut acc = plan.op.seed();
                for &value in values {
                    acc = plan.op.fold(acc, value);
                }
                acc
            };
            out.clear();
            out.reserve(1);
            out.push(Value::scalar_f64(reduced));
            Some(Ok(()))
        }
        DType::F32 => {
            let values = tensor.elements.as_f32_slice()?;
            if tensor.shape.rank() == 0 {
                // Scalar input: reduce is the identity; return the element
                // unchanged (fj-lax returns the original literal, same bits).
                let v = *values.first()?;
                out.clear();
                out.reserve(1);
                out.push(Value::Scalar(Literal::from_f32(v)));
                return Some(Ok(()));
            }
            let mut acc = plan.op.seed();
            for &value in values {
                acc = plan.op.fold(acc, f64::from(value));
            }
            out.clear();
            out.reserve(1);
            out.push(Value::Scalar(Literal::from_f32(acc as f32)));
            Some(Ok(()))
        }
        _ => None,
    }
}

fn dense_axis_reduce_input<'a>(
    plan: &DenseAxisReducePlan,
    const_values: &'a [Value],
    args: &'a [Value],
) -> Result<&'a Value, InterpreterError> {
    for (&slot, value) in plan.const_slots.iter().zip(const_values) {
        if slot == plan.input_slot {
            return Ok(value);
        }
    }
    for (&slot, value) in plan.input_slots.iter().zip(args) {
        if slot == plan.input_slot {
            return Ok(value);
        }
    }
    Err(InterpreterError::MissingVariable(VarId(
        plan.input_slot as u32,
    )))
}

fn run_dense_axis_reduce_plan_into(
    plan: &DenseAxisReducePlan,
    const_values: &[Value],
    args: &[Value],
    out: &mut Vec<Value>,
) -> Option<Result<(), InterpreterError>> {
    if const_values.len() != plan.const_slots.len() {
        return Some(Err(InterpreterError::ConstArity {
            expected: plan.const_slots.len(),
            actual: const_values.len(),
        }));
    }
    if args.len() != plan.input_slots.len() {
        return Some(Err(InterpreterError::InputArity {
            expected: plan.input_slots.len(),
            actual: args.len(),
        }));
    }

    let input = match dense_axis_reduce_input(plan, const_values, args) {
        Ok(value) => value,
        Err(err) => return Some(Err(err)),
    };
    let Value::Tensor(tensor) = input else {
        return None;
    };
    let rank = tensor.shape.rank();
    let m = plan.axes.len();
    // The reduced axes must be a CONTIGUOUS TRAILING block [rank-m .. rank-1] and
    // leave at least one kept axis (m < rank — reducing ALL axes is the full
    // reduce, which returns a SCALAR in fj-lax, a different shape). Then each
    // output cell folds one contiguous run of K = product(trailing dims), giving
    // shape dims[..rank-m] (axes removed, no keepdims) — exactly fj-lax's
    // contiguous-block axis-reduce. Any other axis set falls to the generic interp.
    if m == 0 || m >= rank {
        return None;
    }
    if plan
        .axes
        .iter()
        .enumerate()
        .any(|(i, &a)| a != rank - m + i)
    {
        return None;
    }
    let dims = &tensor.shape.dims;
    let mut k = 1usize;
    for &d in &dims[rank - m..] {
        k = k.checked_mul(d as usize)?;
    }
    if k == 0 {
        return None;
    }
    let out_dims: Vec<u32> = dims[..rank - m].to_vec();
    // outer = product of kept dims; equals total / k.
    let mut outer = 1usize;
    for &d in &out_dims {
        outer = outer.checked_mul(d as usize)?;
    }

    // Fold each contiguous K-block in ascending order with the op's (seed, fold),
    // accumulating in f64 — matching fj-lax's contiguous-block axis-reduce exactly
    // (sum/prod serial; max/min jax_max/min == its SIMD axis-reduce). F64 emits an
    // f64 result; F32 rounds each cell's f64 accumulator back to f32, mirroring
    // fj-lax's `reduce_real_literal(F32, acc)` (the result is value-equal to the
    // eager path's F32 output — a dense vs boxed F32 storage difference only).
    let built = match tensor.dtype {
        DType::F64 => {
            let values = tensor.elements.as_f64_slice()?;
            if values.len() != outer * k {
                return None;
            }
            let mut result = Vec::with_capacity(outer);
            for cell in 0..outer {
                let mut acc = plan.op.seed();
                for &v in &values[cell * k..cell * k + k] {
                    acc = plan.op.fold(acc, v);
                }
                result.push(acc);
            }
            TensorValue::new_f64_values(Shape { dims: out_dims }, result)
        }
        DType::F32 => {
            let values = tensor.elements.as_f32_slice()?;
            if values.len() != outer * k {
                return None;
            }
            let mut result = Vec::with_capacity(outer);
            for cell in 0..outer {
                let mut acc = plan.op.seed();
                for &v in &values[cell * k..cell * k + k] {
                    acc = plan.op.fold(acc, f64::from(v));
                }
                result.push(acc as f32);
            }
            TensorValue::new_f32_values(Shape { dims: out_dims }, result)
        }
        _ => return None,
    };
    match built {
        Ok(t) => {
            out.clear();
            out.push(Value::Tensor(t));
            Some(Ok(()))
        }
        Err(e) => Some(Err(InterpreterError::Primitive(EvalError::InvalidTensor(
            e,
        )))),
    }
}

fn run_dense_gather_plan_into(
    plan: &DenseGatherPlan,
    const_values: &[Value],
    args: &[Value],
    out: &mut Vec<Value>,
) -> Option<Result<(), InterpreterError>> {
    if const_values.len() != plan.const_slots.len() {
        return Some(Err(InterpreterError::ConstArity {
            expected: plan.const_slots.len(),
            actual: const_values.len(),
        }));
    }
    if args.len() != plan.input_slots.len() {
        return Some(Err(InterpreterError::InputArity {
            expected: plan.input_slots.len(),
            actual: args.len(),
        }));
    }

    let operand = match dense_f64_dot_input(
        &plan.const_slots,
        &plan.input_slots,
        plan.operand_slot,
        const_values,
        args,
    ) {
        Ok(value) => value,
        Err(err) => return Some(Err(err)),
    };
    let indices = match dense_f64_dot_input(
        &plan.const_slots,
        &plan.input_slots,
        plan.indices_slot,
        const_values,
        args,
    ) {
        Ok(value) => value,
        Err(err) => return Some(Err(err)),
    };
    let (Value::Tensor(operand), Value::Tensor(indices)) = (operand, indices) else {
        return None;
    };

    let rank = operand.shape.rank();
    // slice_sizes must describe a full contiguous row: len == rank, [0]==1 (checked
    // at build), and [1..] == operand dims[1..]. Then each gathered slice is one
    // contiguous run of `slice_elems = product(op_dims[1..])` elements.
    if rank == 0 || plan.slice_sizes.len() != rank {
        return None;
    }
    let op_dims = &operand.shape.dims;
    if plan
        .slice_sizes
        .iter()
        .skip(1)
        .zip(op_dims.iter().skip(1))
        .any(|(&ss, &d)| ss != d as usize)
    {
        return None;
    }
    let dim0 = op_dims[0] as usize;
    if dim0 == 0 {
        return None;
    }
    let mut slice_elems = 1usize;
    for &d in &op_dims[1..] {
        slice_elems = slice_elems.checked_mul(d as usize)?;
    }

    // Indices: dense i64 (covers I64 + I32 dense storage). Resolve each via the
    // default Clip mode (idx<dim0 ? idx : dim0-1). Negative indices ERROR in the
    // eager path (lit_to_usize) — bail so the generic interpreter reports it.
    let idx_slice = indices.elements.as_i64_slice()?;
    let mut resolved: Vec<usize> = Vec::with_capacity(idx_slice.len());
    for &raw in idx_slice {
        if raw < 0 {
            return None;
        }
        let i = raw as usize;
        resolved.push(if i < dim0 { i } else { dim0 - 1 });
    }

    // Output shape = indices.shape ++ slice_sizes[1..] (== op_dims[1..]).
    let mut out_dims: Vec<u32> = indices.shape.dims.clone();
    out_dims.extend_from_slice(&op_dims[1..]);

    let built = match operand.dtype {
        DType::F64 => {
            let src = operand.elements.as_f64_slice()?;
            let mut data = Vec::with_capacity(resolved.len() * slice_elems);
            for &idx in &resolved {
                let base = idx * slice_elems;
                if base + slice_elems > src.len() {
                    return None;
                }
                data.extend_from_slice(&src[base..base + slice_elems]);
            }
            TensorValue::new_f64_values(Shape { dims: out_dims }, data)
        }
        DType::F32 => {
            let src = operand.elements.as_f32_slice()?;
            let mut data = Vec::with_capacity(resolved.len() * slice_elems);
            for &idx in &resolved {
                let base = idx * slice_elems;
                if base + slice_elems > src.len() {
                    return None;
                }
                data.extend_from_slice(&src[base..base + slice_elems]);
            }
            TensorValue::new_f32_values(Shape { dims: out_dims }, data)
        }
        // Gather is a pure contiguous COPY (no arithmetic, no dtype decode), so
        // i64/i32 + bf16/f16 tables copy bit-for-bit too — bf16/f16 here have NO
        // widen/round floor (rows are copied as raw u16), unlike bf16 reductions.
        // i32 shares the i64 dense backing; emit the matching dtype ctor.
        DType::I64 | DType::I32 => {
            let src = operand.elements.as_i64_slice()?;
            let mut data = Vec::with_capacity(resolved.len() * slice_elems);
            for &idx in &resolved {
                let base = idx * slice_elems;
                if base + slice_elems > src.len() {
                    return None;
                }
                data.extend_from_slice(&src[base..base + slice_elems]);
            }
            if operand.dtype == DType::I64 {
                TensorValue::new_i64_values(Shape { dims: out_dims }, data)
            } else {
                TensorValue::new_i32_values(Shape { dims: out_dims }, data)
            }
        }
        DType::BF16 | DType::F16 => {
            let src = operand.elements.as_half_float_slice()?;
            let mut data = Vec::with_capacity(resolved.len() * slice_elems);
            for &idx in &resolved {
                let base = idx * slice_elems;
                if base + slice_elems > src.len() {
                    return None;
                }
                data.extend_from_slice(&src[base..base + slice_elems]);
            }
            TensorValue::new_half_float_values(operand.dtype, Shape { dims: out_dims }, data)
        }
        _ => return None,
    };
    match built {
        Ok(t) => {
            out.clear();
            out.push(Value::Tensor(t));
            Some(Ok(()))
        }
        Err(e) => Some(Err(InterpreterError::Primitive(EvalError::InvalidTensor(
            e,
        )))),
    }
}

// 3-way f64/f32/i64 dtype if-let chain fills `result`; clippy::question_mark
// misfires on the trailing `else { return None }` (a dtype alternative, not a
// `?`-expressible early return).
#[allow(clippy::question_mark)]
fn run_dense_arg_extremum_plan_into(
    plan: &DenseArgExtremumPlan,
    const_values: &[Value],
    args: &[Value],
    out: &mut Vec<Value>,
) -> Option<Result<(), InterpreterError>> {
    if const_values.len() != plan.const_slots.len() {
        return Some(Err(InterpreterError::ConstArity {
            expected: plan.const_slots.len(),
            actual: const_values.len(),
        }));
    }
    if args.len() != plan.input_slots.len() {
        return Some(Err(InterpreterError::InputArity {
            expected: plan.input_slots.len(),
            actual: args.len(),
        }));
    }

    let input = match dense_f64_dot_input(
        &plan.const_slots,
        &plan.input_slots,
        plan.input_slot,
        const_values,
        args,
    ) {
        Ok(v) => v,
        Err(err) => return Some(Err(err)),
    };
    let Value::Tensor(tensor) = input else {
        return None;
    };
    let rank = tensor.shape.rank();
    if rank == 0 {
        return None;
    }
    // Only the TRAILING axis: then each output cell scans one contiguous row of
    // axis_dim = dims[rank-1], emitting I64 indices of shape dims[..rank-1] — the
    // contiguous (axis_stride==1) case fj-lax handles with its SIMD argmax, which
    // matches plan_arg_extreme_float bit-for-bit.
    let eff_axis = plan.axis.unwrap_or(rank - 1);
    if eff_axis != rank - 1 {
        return None;
    }
    let dims = &tensor.shape.dims;
    let axis_dim = dims[rank - 1] as usize;
    if axis_dim == 0 {
        return None;
    }
    let mut outer = 1usize;
    for &d in &dims[..rank - 1] {
        outer = outer.checked_mul(d as usize)?;
    }
    let out_dims: Vec<u32> = dims[..rank - 1].to_vec();
    let find_max = plan.find_max;

    let mut result: Vec<i64> = Vec::with_capacity(outer);
    if let Some(values) = tensor.elements.as_f64_slice() {
        if values.len() != outer * axis_dim {
            return None;
        }
        for cell in 0..outer {
            let base = cell * axis_dim;
            let best = plan_arg_extreme_float(axis_dim, find_max, |i| values[base + i]);
            result.push(best as i64);
        }
    } else if let Some(values) = tensor.elements.as_f32_slice() {
        if values.len() != outer * axis_dim {
            return None;
        }
        for cell in 0..outer {
            let base = cell * axis_dim;
            let best = plan_arg_extreme_float(axis_dim, find_max, |i| f64::from(values[base + i]));
            result.push(best as i64);
        }
    } else if let Some(values) = tensor.elements.as_i64_slice() {
        if values.len() != outer * axis_dim {
            return None;
        }
        // Integers: strict cmp, first-occurrence tie-break (no NaN).
        for cell in 0..outer {
            let base = cell * axis_dim;
            let mut best_idx = 0usize;
            let mut best = values[base];
            for i in 1..axis_dim {
                let v = values[base + i];
                if (find_max && v > best) || (!find_max && v < best) {
                    best_idx = i;
                    best = v;
                }
            }
            result.push(best_idx as i64);
        }
    } else if let Some(values) = tensor.elements.as_half_float_slice() {
        // BF16/F16 argmax/argmin (mixed-precision logits). fj-lax's half argmax is
        // SCALAR (decode each tap half->f32->f64, no SIMD — unlike its f64/f32 SIMD
        // argmax), so this matches its kernel exactly while saving the dispatch.
        // Decode mirrors fj-lax: as_bf16_f32/as_f16_f32 -> f32 -> f64.
        if values.len() != outer * axis_dim {
            return None;
        }
        let is_bf16 = tensor.dtype == DType::BF16;
        let decode = |b: u16| -> f64 {
            let f = if is_bf16 {
                Literal::BF16Bits(b).as_bf16_f32()
            } else {
                Literal::F16Bits(b).as_f16_f32()
            };
            f64::from(f.unwrap_or(f32::NAN))
        };
        for cell in 0..outer {
            let base = cell * axis_dim;
            let best = plan_arg_extreme_float(axis_dim, find_max, |i| decode(values[base + i]));
            result.push(best as i64);
        }
    } else {
        return None;
    }

    match TensorValue::new_i64_values(Shape { dims: out_dims }, result) {
        Ok(t) => {
            out.clear();
            out.push(Value::Tensor(t));
            Some(Ok(()))
        }
        Err(e) => Some(Err(InterpreterError::Primitive(EvalError::InvalidTensor(
            e,
        )))),
    }
}

fn dense_f64_dot_input<'a>(
    const_slots: &[usize],
    input_slots: &[usize],
    slot: usize,
    const_values: &'a [Value],
    args: &'a [Value],
) -> Result<&'a Value, InterpreterError> {
    for (&const_slot, value) in const_slots.iter().zip(const_values) {
        if const_slot == slot {
            return Ok(value);
        }
    }
    for (&input_slot, value) in input_slots.iter().zip(args) {
        if input_slot == slot {
            return Ok(value);
        }
    }
    Err(InterpreterError::MissingVariable(VarId(slot as u32)))
}

fn run_dense_f64_dot_plan_into(
    plan: &DenseF64DotPlan,
    const_values: &[Value],
    args: &[Value],
    out: &mut Vec<Value>,
) -> Option<Result<(), InterpreterError>> {
    if const_values.len() != plan.const_slots.len() {
        return Some(Err(InterpreterError::ConstArity {
            expected: plan.const_slots.len(),
            actual: const_values.len(),
        }));
    }
    if args.len() != plan.input_slots.len() {
        return Some(Err(InterpreterError::InputArity {
            expected: plan.input_slots.len(),
            actual: args.len(),
        }));
    }

    let lhs = match dense_f64_dot_input(
        &plan.const_slots,
        &plan.input_slots,
        plan.lhs_slot,
        const_values,
        args,
    ) {
        Ok(value) => value,
        Err(err) => return Some(Err(err)),
    };
    let rhs = match dense_f64_dot_input(
        &plan.const_slots,
        &plan.input_slots,
        plan.rhs_slot,
        const_values,
        args,
    ) {
        Ok(value) => value,
        Err(err) => return Some(Err(err)),
    };
    let (Value::Tensor(lhs), Value::Tensor(rhs)) = (lhs, rhs) else {
        return None;
    };
    if lhs.dtype != DType::F64 || rhs.dtype != DType::F64 {
        return None;
    }
    if lhs.shape.rank() != 1 || rhs.shape.rank() != 1 || lhs.shape.dims[0] != rhs.shape.dims[0] {
        return None;
    }
    let lhs_values = lhs.elements.as_f64_slice()?;
    let rhs_values = rhs.elements.as_f64_slice()?;
    if lhs_values.len() != rhs_values.len() {
        return None;
    }

    let mut sum = 0.0_f64;
    for (&left, &right) in lhs_values.iter().zip(rhs_values) {
        sum += left * right;
    }

    out.clear();
    out.reserve(1);
    out.push(Value::scalar_f64(sum));
    Some(Ok(()))
}

fn run_dense_f64_transpose_plan_into(
    plan: &DenseF64TransposePlan,
    const_values: &[Value],
    args: &[Value],
    out: &mut Vec<Value>,
) -> Option<Result<(), InterpreterError>> {
    if const_values.len() != plan.const_slots.len() {
        return Some(Err(InterpreterError::ConstArity {
            expected: plan.const_slots.len(),
            actual: const_values.len(),
        }));
    }
    if args.len() != plan.input_slots.len() {
        return Some(Err(InterpreterError::InputArity {
            expected: plan.input_slots.len(),
            actual: args.len(),
        }));
    }

    let input = match dense_f64_dot_input(
        &plan.const_slots,
        &plan.input_slots,
        plan.input_slot,
        const_values,
        args,
    ) {
        Ok(value) => value,
        Err(err) => return Some(Err(err)),
    };
    let Value::Tensor(tensor) = input else {
        return None;
    };
    if tensor.dtype != DType::F64 || tensor.shape.rank() != 2 {
        return None;
    }
    let permutation = plan.permutation.as_deref().unwrap_or(&[1, 0]);
    if permutation != [1, 0] {
        return None;
    }
    let src = tensor.elements.as_f64_slice()?;
    if src.is_empty() {
        return None;
    }
    let rows = tensor.shape.dims[0] as usize;
    let cols = tensor.shape.dims[1] as usize;
    if rows.checked_mul(cols)? != src.len() {
        return None;
    }

    let mut result = vec![src[0]; src.len()];
    const BLOCK: usize = 64;
    let mut bi = 0;
    while bi < rows {
        let i_end = (bi + BLOCK).min(rows);
        let mut bj = 0;
        while bj < cols {
            let j_end = (bj + BLOCK).min(cols);
            for i in bi..i_end {
                let src_row = i * cols;
                for j in bj..j_end {
                    result[j * rows + i] = src[src_row + j];
                }
            }
            bj += BLOCK;
        }
        bi += BLOCK;
    }

    match TensorValue::new_f64_values(
        Shape {
            dims: vec![tensor.shape.dims[1], tensor.shape.dims[0]],
        },
        result,
    ) {
        Ok(tensor) => {
            out.clear();
            out.push(Value::Tensor(tensor));
            Some(Ok(()))
        }
        Err(err) => Some(Err(InterpreterError::Primitive(EvalError::InvalidTensor(
            err,
        )))),
    }
}

fn run_dense_f64_broadcast_in_dim_plan_into(
    plan: &DenseF64BroadcastInDimPlan,
    const_values: &[Value],
    args: &[Value],
    out: &mut Vec<Value>,
) -> Option<Result<(), InterpreterError>> {
    if const_values.len() != plan.const_slots.len() {
        return Some(Err(InterpreterError::ConstArity {
            expected: plan.const_slots.len(),
            actual: const_values.len(),
        }));
    }
    if args.len() != plan.input_slots.len() {
        return Some(Err(InterpreterError::InputArity {
            expected: plan.input_slots.len(),
            actual: args.len(),
        }));
    }

    let input = match dense_f64_dot_input(
        &plan.const_slots,
        &plan.input_slots,
        plan.input_slot,
        const_values,
        args,
    ) {
        Ok(value) => value,
        Err(err) => return Some(Err(err)),
    };
    let Value::Tensor(tensor) = input else {
        return None;
    };
    if tensor.dtype != DType::F64 || tensor.shape.rank() != 1 || plan.target_shape.rank() != 2 {
        return None;
    }
    let broadcast_dimensions = plan.broadcast_dimensions.as_deref().unwrap_or(&[1]);
    if broadcast_dimensions != [1] || plan.target_count == 0 {
        return None;
    }
    let src = tensor.elements.as_f64_slice()?;
    if src.is_empty() {
        return None;
    }
    let input_dim = tensor.shape.dims[0] as usize;
    let rows = plan.target_shape.dims[0] as usize;
    let cols = plan.target_shape.dims[1] as usize;
    if rows.checked_mul(cols)? != plan.target_count
        || src.len() != input_dim
        || (input_dim != 1 && input_dim != cols)
    {
        return None;
    }

    let result = if input_dim == 1 {
        vec![src[0]; plan.target_count]
    } else {
        let mut values = Vec::with_capacity(plan.target_count);
        for _ in 0..rows {
            values.extend_from_slice(src);
        }
        values
    };

    match TensorValue::new_f64_values(plan.target_shape.clone(), result) {
        Ok(tensor) => {
            out.clear();
            out.push(Value::Tensor(tensor));
            Some(Ok(()))
        }
        Err(err) => Some(Err(InterpreterError::Primitive(EvalError::InvalidTensor(
            err,
        )))),
    }
}

fn run_dense_reshape_plan_into(
    plan: &DenseReshapePlan,
    const_values: &[Value],
    args: &[Value],
    out: &mut Vec<Value>,
) -> Option<Result<(), InterpreterError>> {
    if const_values.len() != plan.const_slots.len() {
        return Some(Err(InterpreterError::ConstArity {
            expected: plan.const_slots.len(),
            actual: const_values.len(),
        }));
    }
    if args.len() != plan.input_slots.len() {
        return Some(Err(InterpreterError::InputArity {
            expected: plan.input_slots.len(),
            actual: args.len(),
        }));
    }

    let input = match dense_f64_dot_input(
        &plan.const_slots,
        &plan.input_slots,
        plan.input_slot,
        const_values,
        args,
    ) {
        Ok(value) => value,
        Err(err) => return Some(Err(err)),
    };
    let Value::Tensor(tensor) = input else {
        return None;
    };
    let input_count = u64::try_from(tensor.elements.len()).ok()?;
    let (target_shape, target_count) = plan.target.resolve(input_count)?;
    if input_count != target_count {
        return None;
    }

    match TensorValue::new_with_literal_buffer(tensor.dtype, target_shape, tensor.elements.clone())
    {
        Ok(tensor) => {
            out.clear();
            out.push(Value::Tensor(tensor));
            Some(Ok(()))
        }
        Err(err) => Some(Err(InterpreterError::Primitive(EvalError::InvalidTensor(
            err,
        )))),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_dense_plan_into(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
    env: &mut [Option<Value>],
    plan: &DenseEvalPlan,
    scratch: &mut Vec<Value>,
    out: &mut Vec<Value>,
    scalar_buffers: &mut ScalarPlanBuffers,
) -> Result<(), InterpreterError> {
    run_dense_plan_into_core(
        jaxpr,
        const_values,
        args,
        env,
        plan,
        scratch,
        out,
        scalar_buffers,
        true,
    )
}

/// Core of [`run_dense_plan_into`]. `vectorize` selects the auto-vectorizable
/// hoisted-op inner loop for the dense-f64 small-tensor arena (the production default);
/// `false` keeps the generic per-element loop. It is a benchmark A/B knob (see
/// `CompiledJaxprRunner::eval_scalar_inner`) so the vectorized path can be measured
/// against the per-element path in the SAME binary — the only worker-variance-immune
/// signal on the contended bench host. Both modes are bit-for-bit identical.
#[allow(clippy::too_many_arguments)]
fn run_dense_plan_into_core(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
    env: &mut [Option<Value>],
    plan: &DenseEvalPlan,
    scratch: &mut Vec<Value>,
    out: &mut Vec<Value>,
    scalar_buffers: &mut ScalarPlanBuffers,
    vectorize: bool,
) -> Result<(), InterpreterError> {
    // Try each monomorphic scalar-arith executor; each bails (returns None) on the
    // first operand that is not its dtype at runtime, so for a given call at most
    // one actually runs. A body with a typed literal builds only the matching plan;
    // a literal-free body builds multiple dtype plans and the runtime dtype selects one.
    if let Some(p) = &plan.scalar_f64_plan
        && let Some(result) =
            run_scalar_f64_arith_plan_into(p, const_values, args, out, &mut scalar_buffers.f64)
    {
        return result;
    }
    if let Some(p) = &plan.scalar_i64_plan
        && let Some(result) =
            run_scalar_i64_arith_plan_into(p, const_values, args, out, &mut scalar_buffers.i64)
    {
        return result;
    }
    if let Some(p) = &plan.scalar_f32_plan
        && let Some(result) =
            run_scalar_f32_arith_plan_into(p, const_values, args, out, &mut scalar_buffers.f32)
    {
        return result;
    }
    if let Some(p) = &plan.scalar_bf16_plan
        && let Some(result) =
            run_scalar_half_arith_plan_into(p, const_values, args, out, &mut scalar_buffers.bf16)
    {
        return result;
    }
    if let Some(p) = &plan.scalar_f16_plan
        && let Some(result) =
            run_scalar_half_arith_plan_into(p, const_values, args, out, &mut scalar_buffers.f16)
    {
        return result;
    }
    if let Some(p) = &plan.scalar_compare_plan
        && let Some(result) = run_scalar_compare_plan(p, const_values, args)
    {
        out.clear();
        out.push(Value::scalar_bool(result));
        return Ok(());
    }
    if let Some(p) = &plan.scalar_compound_compare_plan
        && let Some(result) =
            run_scalar_compound_compare_plan(p, const_values, args, &mut scalar_buffers.bools)
    {
        let result = result?;
        out.clear();
        out.push(Value::scalar_bool(result));
        return Ok(());
    }
    if let Some(p) = &plan.scalar_select_plan
        && let Some(result) =
            run_scalar_select_plan_into(p, const_values, args, out, &mut scalar_buffers.mixed)
    {
        return result;
    }
    if let Some(p) = &plan.scalar_select_i64_plan
        && let Some(result) = run_scalar_select_i64_plan_into(
            p,
            const_values,
            args,
            out,
            &mut scalar_buffers.mixed_i64,
        )
    {
        return result;
    }
    if let Some(p) = &plan.scalar_poly_plan
        && let Some(result) =
            run_scalar_poly_plan_into(p, const_values, args, out, &mut scalar_buffers.poly)
    {
        return result;
    }
    // One-equation full reduction (sum/prod/max/min) over dense f64/f32 tensors.
    // This mirrors fj-lax's full-reduce fold exactly: seed the op init,
    // accumulate in f64, and visit the backing slice in ascending order with no
    // reassociation (F32 rounds the final f64 accumulator back to f32, matching
    // reduce_real_literal). Axes, half/int/complex dtypes, or non-dense storage
    // fall through to the generic interpreter.
    if let Some(p) = &plan.dense_f64_reduce_sum_plan
        && let Some(result) = run_dense_f64_reduce_sum_plan_into(p, const_values, args, out)
    {
        return result;
    }
    // One-equation single-trailing-axis reduction (sum/prod/max/min) over a dense
    // f64/f32 tensor: per-output-cell contiguous-block fold, bit-identical to
    // fj-lax's contiguous-block axis-reduce. Non-trailing/multi axes, rank<2,
    // half/int/complex, or non-dense fall through to the generic interpreter.
    if let Some(p) = &plan.dense_axis_reduce_plan
        && let Some(result) = run_dense_axis_reduce_plan_into(p, const_values, args, out)
    {
        return result;
    }
    // One-equation contiguous row-gather (embedding lookup) over a dense f64/f32
    // table with i64 indices and default Clip mode: pure contiguous slice copy,
    // bit-identical to fj-lax's dense contiguous gather. Non-row slices, other
    // index/dtype, OOB-fill mode, or negative indices fall to the generic interp.
    if let Some(p) = &plan.dense_gather_plan
        && let Some(result) = run_dense_gather_plan_into(p, const_values, args, out)
    {
        return result;
    }
    // One-equation argmax/argmin over the trailing axis (jnp.argmax(x, -1)) over a
    // dense f64/f32/i64 tensor: per-row scan emitting I64 indices, bit-identical to
    // fj-lax (same first-NaN/strict-cmp/first-occurrence reducer). Non-trailing
    // axis, other dtypes, or non-dense fall to the generic interpreter.
    if let Some(p) = &plan.dense_arg_extremum_plan
        && let Some(result) = run_dense_arg_extremum_plan_into(p, const_values, args, out)
    {
        return result;
    }
    // One-equation rank-1 dense-f64 dot. This is exactly fj-lax's contiguous
    // dot fold for F64 tensors: seed 0.0 and visit the backing slices in
    // ascending order, with no reassociation or fused multiply-add.
    if let Some(p) = &plan.dense_f64_dot_plan
        && let Some(result) = run_dense_f64_dot_plan_into(p, const_values, args, out)
    {
        return result;
    }
    // One-equation dense F64 rank-2 transpose: pre-parsed permutation plus the
    // same blocked row-major data movement as fj-lax's transpose fast path.
    if let Some(p) = &plan.dense_f64_transpose_plan
        && let Some(result) = run_dense_f64_transpose_plan_into(p, const_values, args, out)
    {
        return result;
    }
    // One-equation dense F64 rank-1 -> rank-2 trailing-axis broadcast: pre-parsed
    // shape/dim mapping and row-copy replication, bit-identical to fj-lax.
    if let Some(p) = &plan.dense_f64_broadcast_in_dim_plan
        && let Some(result) = run_dense_f64_broadcast_in_dim_plan_into(p, const_values, args, out)
    {
        return result;
    }
    // One-equation dense tensor reshape: pre-parsed target shape/spec plus the
    // same metadata-only literal-buffer clone as fj-lax. Scalars and runtime
    // shape mismatches fall through to generic eval_primitive.
    if let Some(p) = &plan.dense_reshape_plan
        && let Some(result) = run_dense_reshape_plan_into(p, const_values, args, out)
    {
        return result;
    }
    // Small dense-f64 same-shape elementwise tensor body: reuse the scalar f64 plan's
    // steps but run them elementwise (no per-op dispatch). Bails for non-dense / large
    // (>= FUSION_MIN_ELEMS) bodies, which the generic path + chunked fusion own.
    if let Some(p) = &plan.scalar_f64_plan
        && let Some(result) = run_scalar_f64_plan_as_tensor_into(
            p,
            const_values,
            args,
            out,
            &mut scalar_buffers.dense_f64,
            vectorize,
        )
    {
        return result;
    }
    // f32 sibling (JAX's default tensor dtype): same small-tensor elementwise arena.
    if let Some(p) = &plan.scalar_f32_plan
        && let Some(result) = run_scalar_f32_plan_as_tensor_into(
            p,
            const_values,
            args,
            out,
            &mut scalar_buffers.dense_f32,
            vectorize,
        )
    {
        return result;
    }
    // i64 sibling: integer small-tensor elementwise carries.
    if let Some(p) = &plan.scalar_i64_plan
        && let Some(result) = run_scalar_i64_plan_as_tensor_into(
            p,
            const_values,
            args,
            out,
            &mut scalar_buffers.dense_i64,
            vectorize,
        )
    {
        return result;
    }
    run_dense_env_into(jaxpr, const_values, args, env, &plan.last_use, scratch, out)
}

fn try_eval_donated_dense_f64_unary(
    eqn: &Equation,
    eqn_index: usize,
    env: &mut [Option<Value>],
    last_use: &[usize],
) -> Option<Result<(), InterpreterError>> {
    if DENSE_F64_DONATION_DISABLE.load(std::sync::atomic::Ordering::Relaxed) != 0 {
        return None;
    }
    if !eqn.sub_jaxprs.is_empty()
        || !eqn.effects.is_empty()
        || eqn.inputs.len() != 1
        || eqn.outputs.len() != 1
        || is_multi_output_primitive(eqn.primitive)
    {
        return None;
    }
    let op = inplace_f64_unary(eqn.primitive, &eqn.params)?;
    let Atom::Var(input_var) = eqn.inputs[0] else {
        return None;
    };
    let input_slot = input_var.0 as usize;
    let out_slot = eqn.outputs[0].0 as usize;
    if input_slot >= env.len() {
        return Some(Err(InterpreterError::MissingVariable(input_var)));
    }
    if out_slot >= env.len() {
        return Some(Err(InterpreterError::MissingVariable(eqn.outputs[0])));
    }
    if last_use.get(input_slot).copied() != Some(eqn_index) {
        return None;
    }

    let Some(value) = env[input_slot].take() else {
        return Some(Err(InterpreterError::MissingVariable(input_var)));
    };
    let Value::Tensor(tensor) = value else {
        env[input_slot] = Some(value);
        return None;
    };
    if tensor.dtype != DType::F64 {
        env[input_slot] = Some(Value::Tensor(tensor));
        return None;
    }

    let TensorValue {
        dtype,
        shape,
        elements,
    } = tensor;
    let mut values = match elements.try_into_f64_values() {
        Ok(values) => values,
        Err(elements) => {
            env[input_slot] = Some(Value::Tensor(TensorValue {
                dtype,
                shape,
                elements,
            }));
            return None;
        }
    };
    drive_fusion_chunks(&mut values, |chunk, _base| {
        apply_inplace_f64_unary_chunk(chunk, op);
    });
    let output = match TensorValue::new_f64_values(shape, values) {
        Ok(tensor) => Value::Tensor(tensor),
        Err(err) => {
            return Some(Err(InterpreterError::Primitive(EvalError::InvalidTensor(
                err,
            ))));
        }
    };
    env[out_slot] = Some(output);
    Some(Ok(()))
}

/// [`run_dense_env`] writing its `outvars` into a caller-owned `out` buffer and
/// resolving inputs through a caller-owned `scratch` buffer, so a loop re-running
/// the same sub-jaxpr reuses BOTH allocations instead of a fresh `Vec` per call.
/// `scratch` and `out` are cleared on entry. Bit-for-bit identical to
/// [`run_dense_env`]; only the output-vector ownership differs.
#[allow(clippy::too_many_arguments)]
fn run_dense_env_into(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
    env: &mut [Option<Value>],
    last_use: &[usize],
    scratch: &mut Vec<Value>,
    out: &mut Vec<Value>,
) -> Result<(), InterpreterError> {
    for slot in env.iter_mut() {
        *slot = None;
    }
    for (idx, var) in jaxpr.constvars.iter().enumerate() {
        env[var.0 as usize] = Some(const_values[idx].clone());
    }
    for (idx, var) in jaxpr.invars.iter().enumerate() {
        env[var.0 as usize] = Some(args[idx].clone());
    }

    let mut i = 0;
    while i < jaxpr.equations.len() {
        // Fast path: fuse a maximal cheap-elementwise run starting here into one
        // chunked pass (materializes only the final output). Bails to the normal
        // path below for anything not fuse-eligible.
        if let Some(run) = try_fuse_elementwise_chain(jaxpr, i, env, last_use) {
            let tensor = match run.values {
                FusedValues::F64(values) => TensorValue::new_f64_values(run.shape, values),
                FusedValues::F32(values) => TensorValue::new_f32_values(run.shape, values),
                FusedValues::I64(values) => TensorValue::new_i64_values(run.shape, values),
                FusedValues::Half { dtype, values } => {
                    TensorValue::new_half_float_values(dtype, run.shape, values)
                }
            }
            .map_err(EvalError::InvalidTensor)
            .map_err(InterpreterError::Primitive)?;
            env[run.out_var.0 as usize] = Some(Value::Tensor(tensor));
            // Free the run's external inputs whose last read was within the run
            // (outvars are pinned to usize::MAX, so they survive).
            for v in &run.ext_vars {
                let s = v.0 as usize;
                if last_use[s] <= run.run_end {
                    env[s] = None;
                }
            }
            i = run.run_end + 1;
            continue;
        }

        let eqn = &jaxpr.equations[i];
        if let Some(result) = try_eval_donated_dense_f64_unary(eqn, i, env, last_use) {
            result?;
            i += 1;
            continue;
        }

        scratch.clear();
        scratch.reserve(eqn.inputs.len());
        for atom in &eqn.inputs {
            match atom {
                Atom::Var(var) => {
                    let value = env
                        .get(var.0 as usize)
                        .and_then(|slot| slot.as_ref())
                        .ok_or(InterpreterError::MissingVariable(*var))?;
                    scratch.push(value.clone());
                }
                Atom::Lit(lit) => scratch.push(Value::Scalar(*lit)),
            }
        }

        if eqn.sub_jaxprs.is_empty()
            && eqn.outputs.len() == 1
            && !is_multi_output_primitive(eqn.primitive)
        {
            let output = eval_primitive(eqn.primitive, scratch, &eqn.params)
                .map_err(InterpreterError::Primitive)?;
            env[eqn.outputs[0].0 as usize] = Some(output);
        } else {
            let outputs = eval_equation_outputs_from_resolved(eqn, scratch)?;
            for (out_var, output) in eqn.outputs.iter().zip(outputs) {
                env[out_var.0 as usize] = Some(output);
            }
        }

        // Free intermediates whose last read was this equation (outvars are pinned
        // to usize::MAX above, so they are never dropped here).
        for atom in &eqn.inputs {
            if let Atom::Var(var) = atom {
                let s = var.0 as usize;
                if last_use[s] == i {
                    env[s] = None;
                }
            }
        }

        i += 1;
    }

    out.clear();
    out.reserve(jaxpr.outvars.len());
    for var in &jaxpr.outvars {
        let value = env
            .get(var.0 as usize)
            .and_then(|slot| slot.clone())
            .ok_or(InterpreterError::MissingVariable(*var))?;
        out.push(value);
    }
    Ok(())
}

/// Hash-map interpreter environment — fallback for jaxprs whose variable ids
/// are sparse or very large. Semantically identical to [`eval_jaxpr_dense_env`].
fn eval_jaxpr_hashed_env(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
) -> Result<Vec<Value>, InterpreterError> {
    let mut env: FxHashMap<VarId, Value> = FxHashMap::with_capacity_and_hasher(
        jaxpr.constvars.len() + jaxpr.invars.len() + jaxpr.equations.len(),
        Default::default(),
    );
    for (idx, var) in jaxpr.constvars.iter().enumerate() {
        env.insert(*var, const_values[idx].clone());
    }

    for (idx, var) in jaxpr.invars.iter().enumerate() {
        env.insert(*var, args[idx].clone());
    }

    for eqn in &jaxpr.equations {
        let outputs = eval_equation_outputs(eqn, &env)?;
        for (out_var, output) in eqn.outputs.iter().zip(outputs) {
            env.insert(*out_var, output);
        }
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

#[cfg(test)]
mod tests {
    use super::{InterpreterError, eval_jaxpr, eval_jaxpr_hashed_env, eval_jaxpr_with_consts};
    use fj_core::{
        Atom, DType, Equation, Jaxpr, Literal, Primitive, ProgramSpec, Shape, TensorValue, Value,
        VarId, build_program,
    };
    use smallvec::smallvec;
    use std::collections::BTreeMap;

    #[test]
    fn eval_simple_add_jaxpr() {
        let jaxpr = build_program(ProgramSpec::Add2);
        let outputs = eval_jaxpr(&jaxpr, &[Value::scalar_i64(4), Value::scalar_i64(5)]);
        assert_eq!(outputs, Ok(vec![Value::scalar_i64(9)]));
    }

    // ── Elementwise fusion (a8nbp) bit-identity ──
    fn f64_tensor(n: usize, f: impl Fn(usize) -> f64) -> Value {
        Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![n as u32],
                },
                (0..n).map(f).collect(),
            )
            .unwrap(),
        )
    }
    fn f64_vec(v: &Value) -> Vec<f64> {
        match v {
            // Robust to both dense-f64 storage (fused output) and boxed Literal
            // storage (the per-equation path's output).
            Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
            _ => panic!("expected tensor"),
        }
    }
    fn lit(x: f64) -> Atom {
        Atom::Lit(Literal::from_f64(x))
    }

    #[test]
    fn eval_donated_broadcast_reciprocal_matches_generic() {
        let x = VarId(0);
        let bcast = VarId(1);
        let out = VarId(2);
        let mut broadcast_params = BTreeMap::new();
        broadcast_params.insert("shape".to_owned(), "4,3".to_owned());
        broadcast_params.insert("broadcast_dimensions".to_owned(), "1".to_owned());
        let jaxpr = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(x)],
                    outputs: smallvec![bcast],
                    params: broadcast_params,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Reciprocal,
                    inputs: smallvec![Atom::Var(bcast)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        );
        let arg = Value::vector_f64(&[1.0, 2.0, 4.0]).expect("vector");
        let planned = eval_jaxpr(&jaxpr, std::slice::from_ref(&arg)).expect("planned");
        let generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], std::slice::from_ref(&arg)).expect("generic");

        assert_eq!(planned, generic);
        assert_eq!(
            f64_vec(&planned[0]),
            vec![
                1.0, 0.5, 0.25, 1.0, 0.5, 0.25, 1.0, 0.5, 0.25, 1.0, 0.5, 0.25,
            ]
        );
        assert_eq!(f64_vec(&arg), vec![1.0, 2.0, 4.0]);
    }

    fn f32_tensor(n: usize, f: impl Fn(usize) -> f32) -> Value {
        Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: vec![n as u32],
                },
                (0..n).map(f).collect(),
            )
            .unwrap(),
        )
    }

    fn f32_tensor_values(dims: Vec<u32>, values: Vec<f32>) -> Value {
        Value::Tensor(TensorValue::new_f32_values(Shape { dims }, values).unwrap())
    }

    fn f32_bits(v: &Value) -> Vec<u32> {
        match v {
            Value::Tensor(t) => {
                if let Some(values) = t.elements.as_f32_slice() {
                    return values.iter().map(|value| value.to_bits()).collect();
                }
                t.elements
                    .iter()
                    .map(|literal| match literal {
                        Literal::F32Bits(bits) => *bits,
                        other => panic!("expected f32 tensor element, got {other:?}"),
                    })
                    .collect()
            }
            _ => panic!("expected tensor"),
        }
    }

    /// Canonicalize NaN lanes to one fixed bit pattern before a bit-exact f32
    /// comparison. A NaN's sign and payload are IEEE-UNSPECIFIED through
    /// arithmetic, so a fused NATIVE-f32 chain and the unfused f32→f64→f32
    /// reference legitimately disagree on the NaN sign after a `Neg`/`Mul`
    /// (both are NaN — observably identical to JAX/XLA, which likewise does not
    /// pin NaN bits). Finite values, ±0, and ±inf stay EXACT, so the proof still
    /// catches any real NaN-vs-finite / inf-sign / ±0 / subnormal divergence.
    fn canon_f32_nan_bits(bits: &[u32]) -> Vec<u32> {
        bits.iter()
            .map(|&b| {
                if f32::from_bits(b).is_nan() {
                    0x7fc0_0000
                } else {
                    b
                }
            })
            .collect()
    }

    fn lit32(x: f32) -> Atom {
        Atom::Lit(Literal::from_f32(x))
    }

    fn bf16_bits_of(x: f64) -> u16 {
        match Literal::from_bf16_f64(x) {
            Literal::BF16Bits(b) => b,
            other => panic!("expected bf16 literal, got {other:?}"),
        }
    }

    fn bf16_tensor(n: usize, f: impl Fn(usize) -> u16) -> Value {
        Value::Tensor(
            TensorValue::new_half_float_values(
                DType::BF16,
                Shape {
                    dims: vec![n as u32],
                },
                (0..n).map(f).collect(),
            )
            .unwrap(),
        )
    }

    fn half_bits(v: &Value) -> Vec<u16> {
        match v {
            Value::Tensor(t) => {
                if let Some(values) = t.elements.as_half_float_slice() {
                    return values.to_vec();
                }
                t.elements
                    .iter()
                    .map(|literal| match literal {
                        Literal::BF16Bits(bits) | Literal::F16Bits(bits) => *bits,
                        other => panic!("expected half-float tensor element, got {other:?}"),
                    })
                    .collect()
            }
            _ => panic!("expected tensor"),
        }
    }

    fn lit_bf16(x: f64) -> Atom {
        Atom::Lit(Literal::from_bf16_f64(x))
    }

    fn f64_tensor_values(dims: Vec<u32>, values: Vec<f64>) -> Value {
        Value::Tensor(TensorValue::new_f64_values(Shape { dims }, values).unwrap())
    }

    fn i64_tensor_values(dims: Vec<u32>, values: Vec<i64>) -> Value {
        Value::Tensor(TensorValue::new_i64_values(Shape { dims }, values).unwrap())
    }

    fn i64_vals(v: &Value) -> Vec<i64> {
        match v {
            Value::Tensor(t) => {
                if let Some(values) = t.elements.as_i64_slice() {
                    return values.to_vec();
                }
                t.elements
                    .iter()
                    .map(|literal| match literal {
                        Literal::I64(x) => *x,
                        other => panic!("expected i64 element, got {other:?}"),
                    })
                    .collect()
            }
            _ => panic!("expected tensor"),
        }
    }

    fn liti(x: i64) -> Atom {
        Atom::Lit(Literal::I64(x))
    }

    fn f64_bits(v: &Value) -> Vec<u64> {
        match v {
            Value::Tensor(t) => {
                if let Some(values) = t.elements.as_f64_slice() {
                    return values.iter().map(|value| value.to_bits()).collect();
                }
                t.elements
                    .iter()
                    .map(|literal| literal.as_f64().unwrap().to_bits())
                    .collect()
            }
            _ => panic!("expected tensor"),
        }
    }

    #[test]
    fn fusion_chain_matches_reference_bit_for_bit() {
        // Build: v1 = mul(x, x); v2 = add(v1, 2.5); v3 = sub(7.0, v2); v4 = div(v3, y);
        //        v5 = neg(v4); out = mul(v5, x)
        // Mixes square (mul(x,x)), scalar (Add/Sub commute & non-commute), external
        // tensor (Div by y, Mul by x), and Neg. Single-use chain, 6 ops -> fuses.
        // out must equal an independent per-element fold, bit-for-bit (incl any inf/NaN).
        let n = 4096usize; // > FUSION_MIN_ELEMS, > one chunk
        let xv: Vec<f64> = (0..n).map(|i| i as f64 * 0.013 - 9.0).collect();
        let yv: Vec<f64> = (0..n).map(|i| (i as f64 * 0.007).sin() + 1.5).collect();
        let x = VarId(0);
        let y = VarId(1);
        let (v1, v2, v3, v4, v5, out) =
            (VarId(2), VarId(3), VarId(4), VarId(5), VarId(6), VarId(7));
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let eqns = vec![
            mk(Primitive::Mul, smallvec![Atom::Var(x), Atom::Var(x)], v1),
            mk(Primitive::Add, smallvec![Atom::Var(v1), lit(2.5)], v2),
            mk(Primitive::Sub, smallvec![lit(7.0), Atom::Var(v2)], v3),
            mk(Primitive::Div, smallvec![Atom::Var(v3), Atom::Var(y)], v4),
            mk(Primitive::Neg, smallvec![Atom::Var(v4)], v5),
            mk(Primitive::Mul, smallvec![Atom::Var(v5), Atom::Var(x)], out),
        ];
        let jaxpr = Jaxpr::new(vec![x, y], vec![], vec![out], eqns);
        let got = f64_vec(
            &eval_jaxpr(
                &jaxpr,
                &[f64_tensor(n, |i| xv[i]), f64_tensor(n, |i| yv[i])],
            )
            .unwrap()[0],
        );

        let want: Vec<f64> = (0..n)
            .map(|i| {
                let v1 = xv[i] * xv[i];
                let v2 = v1 + 2.5;
                let v3 = 7.0 - v2;
                let v4 = v3 / yv[i];
                let v5 = -v4;
                v5 * xv[i]
            })
            .collect();
        assert_eq!(got.len(), n);
        for i in 0..n {
            assert_eq!(
                got[i].to_bits(),
                want[i].to_bits(),
                "fused chain mismatch at {i}: {} vs {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn f64_scalar_add_chain_simd_matches_ordered_scalar_reference() {
        let input = vec![
            -0.0,
            0.0,
            1.0,
            -2.5,
            f64::from_bits(0x7ff8_0000_0000_0001),
            f64::INFINITY,
            f64::NEG_INFINITY,
            8192.25,
            -8192.25,
            f64::MIN_POSITIVE,
        ];
        let adds = [1.0, -0.0, 2.25, -3.5, f64::from_bits(0x3ff0_0000_0000_0001)];
        let mut tape = Vec::with_capacity(adds.len());
        tape.push(super::FStep {
            op: super::CheapOp::Add,
            a: super::FOperand::Ext(0),
            b: super::FOperand::Scalar(adds[0]),
        });
        for &add in &adds[1..] {
            tape.push(super::FStep {
                op: super::CheapOp::Add,
                a: super::FOperand::Chain,
                b: super::FOperand::Scalar(add),
            });
        }
        let ext = vec![input.as_slice()];
        let (detected_input, detected_adds) =
            super::f64_scalar_add_chain_input_and_literals(&tape, &ext)
                .expect("plain tensor + scalar add chain should specialize");
        assert_eq!(detected_input.as_ptr(), input.as_ptr());
        assert_eq!(detected_adds, adds);

        let got = super::run_f64_scalar_add_chain_simd(detected_input, &detected_adds);
        let want: Vec<f64> = input
            .iter()
            .map(|&value| {
                let mut ordered = value;
                for &add in &adds {
                    ordered += add;
                }
                ordered
            })
            .collect();
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        let reversed_first = [super::FStep {
            op: super::CheapOp::Add,
            a: super::FOperand::Scalar(adds[0]),
            b: super::FOperand::Ext(0),
        }];
        assert!(
            super::f64_scalar_add_chain_input_and_literals(&reversed_first, &ext).is_none(),
            "operand-reversed first add stays on generic fusion to preserve NaN-payload order"
        );
    }

    #[test]
    fn fusion_threaded_f64_chain_matches_reference_bit_for_bit() {
        // Force the THREADED chunk driver: n >= FUSION_THREAD_MIN_ELEMS with a
        // non-chunk-aligned tail, so segment partitioning and the per-worker
        // global-base offsets must reproduce the serial walk exactly. Same 6-op
        // chain as `fusion_chain_matches_reference_bit_for_bit`, just larger.
        let n = (1usize << 23) + 1234; // > FUSION_THREAD_MIN_ELEMS, unaligned tail
        let xv: Vec<f64> = (0..n).map(|i| i as f64 * 0.013 - 9.0).collect();
        let yv: Vec<f64> = (0..n).map(|i| (i as f64 * 0.007).sin() + 1.5).collect();
        let x = VarId(0);
        let y = VarId(1);
        let (v1, v2, v3, v4, v5, out) =
            (VarId(2), VarId(3), VarId(4), VarId(5), VarId(6), VarId(7));
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let eqns = vec![
            mk(Primitive::Mul, smallvec![Atom::Var(x), Atom::Var(x)], v1),
            mk(Primitive::Add, smallvec![Atom::Var(v1), lit(2.5)], v2),
            mk(Primitive::Sub, smallvec![lit(7.0), Atom::Var(v2)], v3),
            mk(Primitive::Div, smallvec![Atom::Var(v3), Atom::Var(y)], v4),
            mk(Primitive::Neg, smallvec![Atom::Var(v4)], v5),
            mk(Primitive::Mul, smallvec![Atom::Var(v5), Atom::Var(x)], out),
        ];
        let jaxpr = Jaxpr::new(vec![x, y], vec![], vec![out], eqns);
        let got = f64_vec(
            &eval_jaxpr(
                &jaxpr,
                &[f64_tensor(n, |i| xv[i]), f64_tensor(n, |i| yv[i])],
            )
            .unwrap()[0],
        );
        let want: Vec<f64> = (0..n)
            .map(|i| {
                let v1 = xv[i] * xv[i];
                let v2 = v1 + 2.5;
                let v3 = 7.0 - v2;
                let v4 = v3 / yv[i];
                let v5 = -v4;
                v5 * xv[i]
            })
            .collect();
        assert_eq!(got.len(), n);
        for i in 0..n {
            assert_eq!(
                got[i].to_bits(),
                want[i].to_bits(),
                "threaded fused chain mismatch at {i}: {} vs {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn fusion_threaded_f64_row_broadcast_matches_reference_bit_for_bit() {
        // Threaded driver + row broadcast: cols (257) divides neither FUSION_CHUNK
        // nor the per-worker segment, so chunk AND thread boundaries fall mid-row.
        // Element (r, c) must still map to bias[c] from the absolute base offset.
        let cols = 257usize;
        let rows = 32768usize;
        let n = rows * cols; // 8_421_376 > FUSION_THREAD_MIN_ELEMS
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.003 - 7.0).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.011).cos() + 1.25).collect();
        let bias: Vec<f64> = (0..cols).map(|i| i as f64 * 0.02 - 0.7).collect();
        let xv = VarId(0);
        let bv = VarId(1);
        let yv = VarId(2);
        let v: Vec<VarId> = (3..=10).map(VarId).collect();
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let eqns = vec![
            mk(
                Primitive::Add,
                smallvec![Atom::Var(xv), Atom::Var(bv)],
                v[0],
            ),
            mk(Primitive::Mul, smallvec![Atom::Var(v[0]), lit(1.25)], v[1]),
            mk(
                Primitive::Sub,
                smallvec![Atom::Var(v[1]), Atom::Var(bv)],
                v[2],
            ),
            mk(
                Primitive::Mul,
                smallvec![Atom::Var(v[2]), Atom::Var(yv)],
                v[3],
            ),
            mk(
                Primitive::Add,
                smallvec![Atom::Var(v[3]), Atom::Var(bv)],
                v[4],
            ),
            mk(Primitive::Sub, smallvec![Atom::Var(v[4]), lit(0.5)], v[5]),
            mk(Primitive::Mul, smallvec![Atom::Var(v[5]), lit(2.0)], v[6]),
            mk(
                Primitive::Add,
                smallvec![Atom::Var(v[6]), Atom::Var(bv)],
                v[7],
            ),
        ];
        let jaxpr = Jaxpr::new(vec![xv, bv, yv], vec![], vec![v[7]], eqns);
        let dims = vec![rows as u32, cols as u32];
        let args = [
            f64_tensor_values(dims.clone(), x.clone()),
            f64_tensor_values(vec![cols as u32], bias.clone()),
            f64_tensor_values(dims, y.clone()),
        ];
        let got = f64_vec(&eval_jaxpr(&jaxpr, &args).unwrap()[0]);
        let want: Vec<f64> = (0..n)
            .map(|i| {
                let c = i % cols;
                let v0 = x[i] + bias[c];
                let v1 = v0 * 1.25;
                let v2 = v1 - bias[c];
                let v3 = v2 * y[i];
                let v4 = v3 + bias[c];
                let v5 = v4 - 0.5;
                let v6 = v5 * 2.0;
                v6 + bias[c]
            })
            .collect();
        assert_eq!(got.len(), n);
        for i in 0..n {
            assert_eq!(
                got[i].to_bits(),
                want[i].to_bits(),
                "threaded row-broadcast mismatch at {i}: {} vs {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn fusion_max_min_abs_chain_matches_reference_bit_for_bit() {
        // Build a clamp/relu/abs pipeline (hardtanh-shaped):
        //   v1 = abs(x); v2 = max(v1, 0.5); v3 = min(v2, 6.0);
        //   v4 = max(v3, y); v5 = mul(v4, 2.0); out = sub(v5, x)
        // Exercises Abs (unary), Max/Min with scalar, and Max with an external
        // tensor — 6 single-use ops -> fuses. Inputs include NaN/inf so the
        // fused result must match the per-op jax_max/jax_min NaN propagation bit
        // for bit (any NaN operand => canonical NaN).
        let n = 4096usize;
        let xv: Vec<f64> = (0..n)
            .map(|i| match i % 7 {
                0 => f64::NAN,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                _ => i as f64 * 0.017 - 30.0,
            })
            .collect();
        let yv: Vec<f64> = (0..n)
            .map(|i| {
                if i % 5 == 0 {
                    f64::NAN
                } else {
                    (i as f64 * 0.011).cos() * 4.0
                }
            })
            .collect();
        let jmax = |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        };
        let jmin = |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.min(b)
            }
        };
        let x = VarId(0);
        let y = VarId(1);
        let (v1, v2, v3, v4, v5, out) =
            (VarId(2), VarId(3), VarId(4), VarId(5), VarId(6), VarId(7));
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let eqns = vec![
            mk(Primitive::Abs, smallvec![Atom::Var(x)], v1),
            mk(Primitive::Max, smallvec![Atom::Var(v1), lit(0.5)], v2),
            mk(Primitive::Min, smallvec![Atom::Var(v2), lit(6.0)], v3),
            mk(Primitive::Max, smallvec![Atom::Var(v3), Atom::Var(y)], v4),
            mk(Primitive::Mul, smallvec![Atom::Var(v4), lit(2.0)], v5),
            mk(Primitive::Sub, smallvec![Atom::Var(v5), Atom::Var(x)], out),
        ];
        let jaxpr = Jaxpr::new(vec![x, y], vec![], vec![out], eqns);
        let got = f64_vec(
            &eval_jaxpr(
                &jaxpr,
                &[f64_tensor(n, |i| xv[i]), f64_tensor(n, |i| yv[i])],
            )
            .unwrap()[0],
        );
        let want: Vec<f64> = (0..n)
            .map(|i| {
                let v1 = xv[i].abs();
                let v2 = jmax(v1, 0.5);
                let v3 = jmin(v2, 6.0);
                let v4 = jmax(v3, yv[i]);
                let v5 = v4 * 2.0;
                v5 - xv[i]
            })
            .collect();
        assert_eq!(got.len(), n);
        for i in 0..n {
            assert_eq!(
                got[i].to_bits(),
                want[i].to_bits(),
                "fused max/min/abs chain mismatch at {i}: {} vs {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn fusion_f32_chain_matches_reference_bit_for_bit() {
        // Same proof shape as the f64 test, but the reference is the hash-map
        // interpreter, which never takes the dense-env fusion fast path. It
        // materializes every intermediate through fj-lax, including f32's
        // f32->f64->f32 per-step rounding contract.
        let n = 4096usize;
        let mut xv: Vec<f32> = (0..n).map(|i| i as f32 * 0.013 - 9.0).collect();
        let mut yv: Vec<f32> = (0..n).map(|i| (i as f32 * 0.007).sin() + 1.5).collect();
        xv[0] = 0.0;
        xv[1] = -0.0;
        xv[2] = f32::INFINITY;
        xv[3] = f32::NEG_INFINITY;
        xv[4] = f32::from_bits(0x7fc0_1234);
        xv[5] = f32::from_bits(1); // smallest positive subnormal
        yv[0] = -0.0;
        yv[1] = 0.0;
        yv[2] = f32::from_bits(0x7fc0_5678);
        yv[3] = 2.0;
        let x = VarId(0);
        let y = VarId(1);
        let (v1, v2, v3, v4, v5, out) =
            (VarId(2), VarId(3), VarId(4), VarId(5), VarId(6), VarId(7));
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let eqns = vec![
            mk(Primitive::Mul, smallvec![Atom::Var(x), Atom::Var(x)], v1),
            mk(Primitive::Add, smallvec![Atom::Var(v1), lit32(2.5)], v2),
            mk(Primitive::Sub, smallvec![lit32(7.0), Atom::Var(v2)], v3),
            mk(Primitive::Div, smallvec![Atom::Var(v3), Atom::Var(y)], v4),
            mk(Primitive::Neg, smallvec![Atom::Var(v4)], v5),
            mk(Primitive::Mul, smallvec![Atom::Var(v5), Atom::Var(x)], out),
        ];
        let jaxpr = Jaxpr::new(vec![x, y], vec![], vec![out], eqns);
        let args = [f32_tensor(n, |i| xv[i]), f32_tensor(n, |i| yv[i])];
        let fused_outputs = eval_jaxpr(&jaxpr, &args).unwrap();
        let unfused_outputs =
            eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("unfused reference evaluates");
        let Value::Tensor(out_tensor) = &fused_outputs[0] else {
            panic!("expected tensor output")
        };
        assert_eq!(out_tensor.dtype, DType::F32);
        assert!(
            out_tensor.elements.as_f32_slice().is_some(),
            "fused output should stay dense f32"
        );
        let got_bits = canon_f32_nan_bits(&f32_bits(&fused_outputs[0]));
        let want_bits = canon_f32_nan_bits(&f32_bits(&unfused_outputs[0]));
        assert_eq!(
            got_bits, want_bits,
            "fused f32 chain must match forced unfused path bit-for-bit (NaN canonicalized)"
        );
        let digest = fj_test_utils::fixture_id_from_json(&want_bits)
            .expect("reference output bits should hash");
        assert_eq!(
            digest,
            "49d07f45aa58ec10dfd8469ebd0bd5f5105faa366a242abe7543e4bde7d10be8"
        );
    }

    #[test]
    fn fusion_bf16_chain_matches_reference_bit_for_bit() {
        // bf16 sibling of the f32 fusion proof: the reference is the hash-map
        // interpreter, which never takes the dense-env fusion fast path. It
        // materializes every intermediate through fj-lax, including bf16's
        // widen→f64→op→round-to-bf16 (round-to-odd) per-step contract. The fused
        // path must reproduce those bits exactly, incl. ±0 / inf / NaN / subnormal.
        let n = 4096usize;
        let mut xb: Vec<u16> = (0..n)
            .map(|i| bf16_bits_of(i as f64 * 0.013 - 9.0))
            .collect();
        let mut yb: Vec<u16> = (0..n)
            .map(|i| bf16_bits_of((i as f64 * 0.007).sin() + 1.5))
            .collect();
        xb[0] = bf16_bits_of(0.0);
        xb[1] = bf16_bits_of(-0.0);
        xb[2] = bf16_bits_of(f64::INFINITY);
        xb[3] = bf16_bits_of(f64::NEG_INFINITY);
        xb[4] = 0x7fc1; // a bf16 NaN with payload
        xb[5] = 0x0001; // smallest positive bf16 subnormal
        yb[0] = bf16_bits_of(-0.0);
        yb[1] = bf16_bits_of(0.0);
        yb[2] = 0x7fd2; // another bf16 NaN payload
        yb[3] = bf16_bits_of(2.0);
        let x = VarId(0);
        let y = VarId(1);
        let (v1, v2, v3, v4, v5, out) =
            (VarId(2), VarId(3), VarId(4), VarId(5), VarId(6), VarId(7));
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let eqns = vec![
            mk(Primitive::Mul, smallvec![Atom::Var(x), Atom::Var(x)], v1),
            mk(Primitive::Add, smallvec![Atom::Var(v1), lit_bf16(2.5)], v2),
            mk(Primitive::Sub, smallvec![lit_bf16(7.0), Atom::Var(v2)], v3),
            mk(Primitive::Div, smallvec![Atom::Var(v3), Atom::Var(y)], v4),
            mk(Primitive::Neg, smallvec![Atom::Var(v4)], v5),
            mk(Primitive::Mul, smallvec![Atom::Var(v5), Atom::Var(x)], out),
        ];
        let jaxpr = Jaxpr::new(vec![x, y], vec![], vec![out], eqns);
        let args = [bf16_tensor(n, |i| xb[i]), bf16_tensor(n, |i| yb[i])];
        let fused_outputs = eval_jaxpr(&jaxpr, &args).unwrap();
        let unfused_outputs =
            eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("unfused reference evaluates");
        let Value::Tensor(out_tensor) = &fused_outputs[0] else {
            panic!("expected tensor output")
        };
        assert_eq!(out_tensor.dtype, DType::BF16);
        assert!(
            out_tensor.elements.as_half_float_slice().is_some(),
            "fused output should stay dense half-float"
        );
        let got_bits = half_bits(&fused_outputs[0]);
        let want_bits = half_bits(&unfused_outputs[0]);
        assert_eq!(
            got_bits, want_bits,
            "fused bf16 chain must match forced unfused path bit-for-bit"
        );
        let digest = fj_test_utils::fixture_id_from_json(&want_bits)
            .expect("reference output bits should hash");
        assert_eq!(
            digest,
            "3132f039bc6e3cbc8f2654e641b4297f4163ecb2cb0b729835873079ae9339ff"
        );
    }

    #[test]
    fn fusion_f16_maxmin_abs_chain_matches_reference_bit_for_bit() {
        // F16 variant exercising Max/Min/Abs (the activation-pipeline ops) plus a
        // tensor operand, proving the F16 widen/round contract and the half NaN
        // semantics (jax_max/min) match the per-op path bit-for-bit.
        fn f16_bits_of(x: f64) -> u16 {
            match Literal::from_f16_f64(x) {
                Literal::F16Bits(b) => b,
                other => panic!("expected f16 literal, got {other:?}"),
            }
        }
        let n = 2048usize;
        let mut xb: Vec<u16> = (0..n)
            .map(|i| f16_bits_of(i as f64 * 0.001 - 1.0))
            .collect();
        let yb: Vec<u16> = (0..n)
            .map(|i| f16_bits_of((i as f64 * 0.005).cos()))
            .collect();
        xb[0] = 0x7e01; // f16 NaN payload
        xb[1] = f16_bits_of(-0.0);
        xb[2] = f16_bits_of(f64::INFINITY);
        let x = VarId(0);
        let y = VarId(1);
        let (v1, v2, v3, out) = (VarId(2), VarId(3), VarId(4), VarId(5));
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let lit_f16 = |x: f64| Atom::Lit(Literal::from_f16_f64(x));
        let eqns = vec![
            mk(Primitive::Max, smallvec![Atom::Var(x), lit_f16(0.0)], v1), // relu
            mk(Primitive::Min, smallvec![Atom::Var(v1), Atom::Var(y)], v2),
            mk(Primitive::Abs, smallvec![Atom::Var(v2)], v3),
            mk(Primitive::Add, smallvec![Atom::Var(v3), lit_f16(0.25)], out),
        ];
        let jaxpr = Jaxpr::new(vec![x, y], vec![], vec![out], eqns);
        let f16_tensor = |bits: &[u16]| {
            Value::Tensor(
                TensorValue::new_half_float_values(
                    DType::F16,
                    Shape {
                        dims: vec![n as u32],
                    },
                    bits.to_vec(),
                )
                .unwrap(),
            )
        };
        let args = [f16_tensor(&xb), f16_tensor(&yb)];
        let fused_outputs = eval_jaxpr(&jaxpr, &args).unwrap();
        let unfused_outputs =
            eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("unfused reference evaluates");
        let Value::Tensor(out_tensor) = &fused_outputs[0] else {
            panic!("expected tensor output")
        };
        assert_eq!(out_tensor.dtype, DType::F16);
        let got_bits = half_bits(&fused_outputs[0]);
        let want_bits = half_bits(&unfused_outputs[0]);
        assert_eq!(
            got_bits, want_bits,
            "fused f16 max/min/abs chain must match forced unfused path bit-for-bit"
        );
        let digest = fj_test_utils::fixture_id_from_json(&want_bits)
            .expect("reference output bits should hash");
        assert_eq!(
            digest,
            "50bd04003ca23bfb110a239a785969d8f4f5da9d3c9ab96f6a79a332d41a149c"
        );
    }

    #[test]
    fn fusion_floor_ceil_trunc_round_sign_chains_match_unfused_reference() {
        fn f16_bits_of(x: f64) -> u16 {
            match Literal::from_f16_f64(x) {
                Literal::F16Bits(b) => b,
                other => panic!("expected f16 literal, got {other:?}"),
            }
        }

        fn mk_eqn(p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, out: VarId) -> Equation {
            Equation {
                primitive: p,
                inputs: ins,
                outputs: smallvec![out],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }
        }

        let make_float_chain = |lit_add: Atom, lit_sub: Atom, lit_round: Atom| {
            let x = VarId(0);
            let v: Vec<VarId> = (1..=9).map(VarId).collect();
            let eqns = vec![
                mk_eqn(
                    Primitive::Add,
                    smallvec![Atom::Var(x), lit_add.clone()],
                    v[0],
                ),
                mk_eqn(Primitive::Floor, smallvec![Atom::Var(v[0])], v[1]),
                mk_eqn(
                    Primitive::Sub,
                    smallvec![Atom::Var(v[1]), lit_sub.clone()],
                    v[2],
                ),
                mk_eqn(Primitive::Ceil, smallvec![Atom::Var(v[2])], v[3]),
                mk_eqn(
                    Primitive::Add,
                    smallvec![Atom::Var(v[3]), lit_round.clone()],
                    v[4],
                ),
                mk_eqn(Primitive::Trunc, smallvec![Atom::Var(v[4])], v[5]),
                mk_eqn(Primitive::Add, smallvec![Atom::Var(v[5]), lit_round], v[6]),
                mk_eqn(Primitive::Round, smallvec![Atom::Var(v[6])], v[7]),
                mk_eqn(Primitive::Sign, smallvec![Atom::Var(v[7])], v[8]),
            ];
            Jaxpr::new(vec![x], vec![], vec![v[8]], eqns)
        };

        let n = 4096usize;
        let mut f64_values: Vec<f64> = (0..n).map(|i| i as f64 * 0.007 - 12.0).collect();
        f64_values[0] = -0.0;
        f64_values[1] = 0.0;
        f64_values[2] = f64::INFINITY;
        f64_values[3] = f64::NEG_INFINITY;
        f64_values[4] = f64::from_bits(0x7ff8_0000_0000_0123);
        let f64_jaxpr = make_float_chain(lit(0.625), lit(1.25), lit(0.5));
        let f64_args = [f64_tensor_values(vec![n as u32], f64_values)];
        let f64_fused = eval_jaxpr(&f64_jaxpr, &f64_args).unwrap();
        let f64_unfused =
            eval_jaxpr_hashed_env(&f64_jaxpr, &[], &f64_args).expect("unfused f64 reference");
        assert_eq!(
            f64_bits(&f64_fused[0]),
            f64_bits(&f64_unfused[0]),
            "fused f64 floor/ceil/trunc/round/sign chain must match unfused reference"
        );

        let mut f32_values: Vec<f32> = (0..n).map(|i| i as f32 * 0.007 - 12.0).collect();
        f32_values[0] = -0.0;
        f32_values[1] = 0.0;
        f32_values[2] = f32::INFINITY;
        f32_values[3] = f32::NEG_INFINITY;
        f32_values[4] = f32::from_bits(0x7fc0_0123);
        let f32_jaxpr = make_float_chain(lit32(0.625), lit32(1.25), lit32(0.5));
        let f32_args = [f32_tensor_values(vec![n as u32], f32_values)];
        let f32_fused = eval_jaxpr(&f32_jaxpr, &f32_args).unwrap();
        let f32_unfused =
            eval_jaxpr_hashed_env(&f32_jaxpr, &[], &f32_args).expect("unfused f32 reference");
        assert_eq!(
            canon_f32_nan_bits(&f32_bits(&f32_fused[0])),
            canon_f32_nan_bits(&f32_bits(&f32_unfused[0])),
            "fused f32 floor/ceil/trunc/round/sign chain must match unfused reference"
        );

        let mut f16_values: Vec<u16> = (0..n)
            .map(|i| f16_bits_of(i as f64 * 0.007 - 12.0))
            .collect();
        f16_values[0] = f16_bits_of(-0.0);
        f16_values[1] = f16_bits_of(0.0);
        f16_values[2] = f16_bits_of(f64::INFINITY);
        f16_values[3] = f16_bits_of(f64::NEG_INFINITY);
        f16_values[4] = 0x7e01;
        let f16_jaxpr = make_float_chain(
            Atom::Lit(Literal::from_f16_f64(0.625)),
            Atom::Lit(Literal::from_f16_f64(1.25)),
            Atom::Lit(Literal::from_f16_f64(0.5)),
        );
        let f16_args = [Value::Tensor(
            TensorValue::new_half_float_values(
                DType::F16,
                Shape {
                    dims: vec![n as u32],
                },
                f16_values,
            )
            .unwrap(),
        )];
        let f16_fused = eval_jaxpr(&f16_jaxpr, &f16_args).unwrap();
        let f16_unfused =
            eval_jaxpr_hashed_env(&f16_jaxpr, &[], &f16_args).expect("unfused f16 reference");
        assert_eq!(
            half_bits(&f16_fused[0]),
            half_bits(&f16_unfused[0]),
            "fused f16 floor/ceil/trunc/round/sign chain must match unfused reference"
        );

        let i64_values: Vec<i64> = vec![i64::MIN, -17, -1, 0, 1, 23, i64::MAX]
            .into_iter()
            .cycle()
            .take(n)
            .collect();
        let x = VarId(0);
        let v: Vec<VarId> = (1..=5).map(VarId).collect();
        let i64_eqns = vec![
            mk_eqn(Primitive::Sign, smallvec![Atom::Var(x)], v[0]),
            mk_eqn(Primitive::Mul, smallvec![Atom::Var(v[0]), liti(-3)], v[1]),
            mk_eqn(Primitive::Sign, smallvec![Atom::Var(v[1])], v[2]),
            mk_eqn(Primitive::Sub, smallvec![liti(5), Atom::Var(v[2])], v[3]),
            mk_eqn(Primitive::Abs, smallvec![Atom::Var(v[3])], v[4]),
        ];
        let i64_jaxpr = Jaxpr::new(vec![x], vec![], vec![v[4]], i64_eqns);
        let i64_args = [i64_tensor_values(vec![n as u32], i64_values)];
        let i64_fused = eval_jaxpr(&i64_jaxpr, &i64_args).unwrap();
        let i64_unfused =
            eval_jaxpr_hashed_env(&i64_jaxpr, &[], &i64_args).expect("unfused i64 reference");
        assert_eq!(
            i64_vals(&i64_fused[0]),
            i64_vals(&i64_unfused[0]),
            "fused i64 sign chain must match unfused reference"
        );
    }

    #[test]
    fn fusion_f32_row_broadcast_chain_matches_reference_bit_for_bit() {
        // Bias-style row broadcasts must gather in row-major order and preserve
        // f32's per-step f32->f64->f32 rounding contract.
        let rows = 64usize;
        let cols = 64usize;
        let n = rows * cols;
        let mut x: Vec<f32> = (0..n).map(|i| i as f32 * 0.003 - 7.0).collect();
        let mut y: Vec<f32> = (0..n).map(|i| (i as f32 * 0.011).cos() + 1.25).collect();
        let mut bias: Vec<f32> = (0..cols).map(|i| i as f32 * 0.02 - 0.7).collect();
        x[0] = -0.0;
        x[1] = f32::INFINITY;
        x[2] = f32::from_bits(0x7fc0_1111);
        y[0] = 0.0;
        y[1] = f32::NEG_INFINITY;
        y[2] = 3.0;
        bias[0] = f32::from_bits(0x7fc0_2222);
        bias[1] = -0.0;
        bias[2] = f32::from_bits(1);

        let xv = VarId(0);
        let bv = VarId(1);
        let yv = VarId(2);
        let v: Vec<VarId> = (3..=10).map(VarId).collect();
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let eqns = vec![
            mk(
                Primitive::Add,
                smallvec![Atom::Var(xv), Atom::Var(bv)],
                v[0],
            ),
            mk(
                Primitive::Mul,
                smallvec![Atom::Var(v[0]), lit32(1.25)],
                v[1],
            ),
            mk(
                Primitive::Sub,
                smallvec![Atom::Var(v[1]), Atom::Var(bv)],
                v[2],
            ),
            mk(
                Primitive::Mul,
                smallvec![Atom::Var(v[2]), Atom::Var(yv)],
                v[3],
            ),
            mk(
                Primitive::Add,
                smallvec![Atom::Var(v[3]), Atom::Var(bv)],
                v[4],
            ),
            mk(Primitive::Sub, smallvec![Atom::Var(v[4]), lit32(0.5)], v[5]),
            mk(Primitive::Mul, smallvec![Atom::Var(v[5]), lit32(2.0)], v[6]),
            mk(
                Primitive::Add,
                smallvec![Atom::Var(v[6]), Atom::Var(bv)],
                v[7],
            ),
        ];
        let jaxpr = Jaxpr::new(vec![xv, bv, yv], vec![], vec![v[7]], eqns);
        let dims = vec![rows as u32, cols as u32];
        let args = [
            f32_tensor_values(dims.clone(), x),
            f32_tensor_values(vec![cols as u32], bias),
            f32_tensor_values(dims, y),
        ];

        let fused_outputs = eval_jaxpr(&jaxpr, &args).unwrap();
        let unfused_outputs =
            eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("unfused reference evaluates");
        let Value::Tensor(out_tensor) = &fused_outputs[0] else {
            panic!("expected tensor output")
        };
        assert_eq!(out_tensor.dtype, DType::F32);
        assert_eq!(out_tensor.shape.dims, vec![rows as u32, cols as u32]);
        assert!(
            out_tensor.elements.as_f32_slice().is_some(),
            "fused row-broadcast output should stay dense f32"
        );
        let got_bits = f32_bits(&fused_outputs[0]);
        let want_bits = f32_bits(&unfused_outputs[0]);
        assert_eq!(
            got_bits, want_bits,
            "fused f32 row-broadcast chain must match forced unfused path bit-for-bit"
        );
        let digest = fj_test_utils::fixture_id_from_json(&want_bits)
            .expect("reference output bits should hash");
        assert_eq!(
            digest,
            "1f742aad15797ada82394f8d78c5b2d488ac650c272e8a81330a694621a64494"
        );
    }

    #[test]
    fn fusion_f32_col_broadcast_chain_matches_reference_bit_for_bit() {
        // Column broadcasts must gather from `[rows, 1]` by row-major row index
        // while preserving f32's per-step f32->f64->f32 rounding contract.
        let rows = 64usize;
        let cols = 64usize;
        let n = rows * cols;
        let mut x: Vec<f32> = (0..n).map(|i| i as f32 * 0.002 - 5.0).collect();
        let mut y: Vec<f32> = (0..n).map(|i| (i as f32 * 0.009).sin() + 1.5).collect();
        let mut bias: Vec<f32> = (0..rows).map(|i| i as f32 * 0.015 - 0.4).collect();
        x[0] = -0.0;
        x[1] = f32::INFINITY;
        x[2] = f32::from_bits(0x7fc0_3333);
        y[0] = 0.0;
        y[1] = f32::NEG_INFINITY;
        y[2] = -3.0;
        bias[0] = f32::from_bits(0x7fc0_4444);
        bias[1] = -0.0;
        bias[2] = f32::from_bits(1);

        let xv = VarId(0);
        let bv = VarId(1);
        let yv = VarId(2);
        let v: Vec<VarId> = (3..=10).map(VarId).collect();
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let eqns = vec![
            mk(
                Primitive::Add,
                smallvec![Atom::Var(xv), Atom::Var(bv)],
                v[0],
            ),
            mk(
                Primitive::Mul,
                smallvec![Atom::Var(v[0]), lit32(1.25)],
                v[1],
            ),
            mk(
                Primitive::Sub,
                smallvec![Atom::Var(v[1]), Atom::Var(bv)],
                v[2],
            ),
            mk(
                Primitive::Mul,
                smallvec![Atom::Var(v[2]), Atom::Var(yv)],
                v[3],
            ),
            mk(
                Primitive::Add,
                smallvec![Atom::Var(v[3]), Atom::Var(bv)],
                v[4],
            ),
            mk(Primitive::Sub, smallvec![Atom::Var(v[4]), lit32(0.5)], v[5]),
            mk(Primitive::Mul, smallvec![Atom::Var(v[5]), lit32(2.0)], v[6]),
            mk(
                Primitive::Add,
                smallvec![Atom::Var(v[6]), Atom::Var(bv)],
                v[7],
            ),
        ];
        let jaxpr = Jaxpr::new(vec![xv, bv, yv], vec![], vec![v[7]], eqns);
        let dims = vec![rows as u32, cols as u32];
        let args = [
            f32_tensor_values(dims.clone(), x),
            f32_tensor_values(vec![rows as u32, 1], bias),
            f32_tensor_values(dims, y),
        ];

        let fused_outputs = eval_jaxpr(&jaxpr, &args).unwrap();
        let unfused_outputs =
            eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("unfused reference evaluates");
        let Value::Tensor(out_tensor) = &fused_outputs[0] else {
            panic!("expected tensor output")
        };
        assert_eq!(out_tensor.dtype, DType::F32);
        assert_eq!(out_tensor.shape.dims, vec![rows as u32, cols as u32]);
        assert!(
            out_tensor.elements.as_f32_slice().is_some(),
            "fused col-broadcast output should stay dense f32"
        );
        let got_bits = canon_f32_nan_bits(&f32_bits(&fused_outputs[0]));
        let want_bits = canon_f32_nan_bits(&f32_bits(&unfused_outputs[0]));
        assert_eq!(
            got_bits, want_bits,
            "fused f32 col-broadcast chain must match forced unfused path bit-for-bit (NaN canonicalized)"
        );
        let digest = fj_test_utils::fixture_id_from_json(&want_bits)
            .expect("reference output bits should hash");
        assert_eq!(
            digest,
            "ebbd5cfb379f28217357239c207690880113a017fa908b1a07a878bb4c8bcc0d"
        );
    }

    #[test]
    fn fusion_f64_row_broadcast_chain_matches_reference_bit_for_bit() {
        // f64 layernorm-style row broadcasts (a [cols] vector against an [R, C]
        // tensor) must fuse, gather in row-major order, and match the unfused
        // per-equation reference bit-for-bit — including signed zeros / inf / NaN.
        let rows = 64usize;
        let cols = 64usize;
        let n = rows * cols;
        let mut x: Vec<f64> = (0..n).map(|i| i as f64 * 0.003 - 7.0).collect();
        let mut y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.011).cos() + 1.25).collect();
        let mut bias: Vec<f64> = (0..cols).map(|i| i as f64 * 0.02 - 0.7).collect();
        x[0] = -0.0;
        x[1] = f64::INFINITY;
        x[2] = f64::from_bits(0x7ff8_0000_0000_1111);
        y[0] = 0.0;
        y[1] = f64::NEG_INFINITY;
        y[2] = 3.0;
        bias[0] = f64::from_bits(0x7ff8_0000_0000_2222);
        bias[1] = -0.0;
        bias[2] = f64::from_bits(1);

        let xv = VarId(0);
        let bv = VarId(1);
        let yv = VarId(2);
        let v: Vec<VarId> = (3..=10).map(VarId).collect();
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let eqns = vec![
            mk(
                Primitive::Add,
                smallvec![Atom::Var(xv), Atom::Var(bv)],
                v[0],
            ),
            mk(Primitive::Mul, smallvec![Atom::Var(v[0]), lit(1.25)], v[1]),
            mk(
                Primitive::Sub,
                smallvec![Atom::Var(v[1]), Atom::Var(bv)],
                v[2],
            ),
            mk(
                Primitive::Mul,
                smallvec![Atom::Var(v[2]), Atom::Var(yv)],
                v[3],
            ),
            mk(
                Primitive::Add,
                smallvec![Atom::Var(v[3]), Atom::Var(bv)],
                v[4],
            ),
            mk(Primitive::Sub, smallvec![Atom::Var(v[4]), lit(0.5)], v[5]),
            mk(Primitive::Mul, smallvec![Atom::Var(v[5]), lit(2.0)], v[6]),
            mk(
                Primitive::Add,
                smallvec![Atom::Var(v[6]), Atom::Var(bv)],
                v[7],
            ),
        ];
        let jaxpr = Jaxpr::new(vec![xv, bv, yv], vec![], vec![v[7]], eqns);
        let dims = vec![rows as u32, cols as u32];
        let args = [
            f64_tensor_values(dims.clone(), x),
            f64_tensor_values(vec![cols as u32], bias),
            f64_tensor_values(dims, y),
        ];

        let fused_outputs = eval_jaxpr(&jaxpr, &args).unwrap();
        let unfused_outputs =
            eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("unfused reference evaluates");
        let Value::Tensor(out_tensor) = &fused_outputs[0] else {
            panic!("expected tensor output")
        };
        assert_eq!(out_tensor.dtype, DType::F64);
        assert_eq!(out_tensor.shape.dims, vec![rows as u32, cols as u32]);
        assert!(
            out_tensor.elements.as_f64_slice().is_some(),
            "fused row-broadcast output should stay dense f64"
        );
        assert_eq!(
            f64_bits(&fused_outputs[0]),
            f64_bits(&unfused_outputs[0]),
            "fused f64 row-broadcast chain must match forced unfused path bit-for-bit"
        );
    }

    #[test]
    fn fusion_f64_col_broadcast_chain_matches_reference_bit_for_bit() {
        // f64 column broadcasts (a [rows, 1] vector against [R, C]) must fuse,
        // gather by row-major row index, and match the unfused reference bit-for-bit.
        let rows = 64usize;
        let cols = 64usize;
        let n = rows * cols;
        let mut x: Vec<f64> = (0..n).map(|i| i as f64 * 0.002 - 5.0).collect();
        let mut y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.009).sin() + 1.5).collect();
        let mut bias: Vec<f64> = (0..rows).map(|i| i as f64 * 0.015 - 0.4).collect();
        x[0] = -0.0;
        x[1] = f64::INFINITY;
        x[2] = f64::from_bits(0x7ff8_0000_0000_3333);
        y[0] = 0.0;
        y[1] = f64::NEG_INFINITY;
        y[2] = -3.0;
        bias[0] = f64::from_bits(0x7ff8_0000_0000_4444);
        bias[1] = -0.0;
        bias[2] = f64::from_bits(1);

        let xv = VarId(0);
        let bv = VarId(1);
        let yv = VarId(2);
        let v: Vec<VarId> = (3..=10).map(VarId).collect();
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let eqns = vec![
            mk(
                Primitive::Add,
                smallvec![Atom::Var(xv), Atom::Var(bv)],
                v[0],
            ),
            mk(Primitive::Mul, smallvec![Atom::Var(v[0]), lit(1.25)], v[1]),
            mk(
                Primitive::Sub,
                smallvec![Atom::Var(v[1]), Atom::Var(bv)],
                v[2],
            ),
            mk(
                Primitive::Mul,
                smallvec![Atom::Var(v[2]), Atom::Var(yv)],
                v[3],
            ),
            mk(
                Primitive::Add,
                smallvec![Atom::Var(v[3]), Atom::Var(bv)],
                v[4],
            ),
            mk(Primitive::Sub, smallvec![Atom::Var(v[4]), lit(0.5)], v[5]),
            mk(Primitive::Mul, smallvec![Atom::Var(v[5]), lit(2.0)], v[6]),
            mk(
                Primitive::Add,
                smallvec![Atom::Var(v[6]), Atom::Var(bv)],
                v[7],
            ),
        ];
        let jaxpr = Jaxpr::new(vec![xv, bv, yv], vec![], vec![v[7]], eqns);
        let dims = vec![rows as u32, cols as u32];
        let args = [
            f64_tensor_values(dims.clone(), x),
            f64_tensor_values(vec![rows as u32, 1], bias),
            f64_tensor_values(dims, y),
        ];

        let fused_outputs = eval_jaxpr(&jaxpr, &args).unwrap();
        let unfused_outputs =
            eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("unfused reference evaluates");
        let Value::Tensor(out_tensor) = &fused_outputs[0] else {
            panic!("expected tensor output")
        };
        assert_eq!(out_tensor.dtype, DType::F64);
        assert_eq!(out_tensor.shape.dims, vec![rows as u32, cols as u32]);
        assert!(
            out_tensor.elements.as_f64_slice().is_some(),
            "fused col-broadcast output should stay dense f64"
        );
        assert_eq!(
            f64_bits(&fused_outputs[0]),
            f64_bits(&unfused_outputs[0]),
            "fused f64 col-broadcast chain must match forced unfused path bit-for-bit"
        );
    }

    #[test]
    fn broadcast_matchers_generalize_past_rank2() {
        let s = |d: &[u32]| Shape { dims: d.to_vec() };
        // Row broadcast: a [C] vector against the trailing axis of any rank.
        assert_eq!(super::row_broadcast_len(&s(&[4, 64]), &s(&[64])), Some(64));
        assert_eq!(
            super::row_broadcast_len(&s(&[4, 8, 64]), &s(&[64])),
            Some(64)
        ); // rank-3 [B,S,D]+[D]
        assert_eq!(
            super::row_broadcast_len(&s(&[2, 4, 8, 64]), &s(&[64])),
            Some(64)
        ); // rank-4
        assert_eq!(super::row_broadcast_len(&s(&[4, 8, 64]), &s(&[8])), None); // wrong trailing
        // Col broadcast: a [..,1] vector with matching leading dims, any rank.
        assert_eq!(
            super::col_broadcast_cols(&s(&[4, 64]), &s(&[4, 1])),
            Some(64)
        );
        assert_eq!(
            super::col_broadcast_cols(&s(&[4, 8, 64]), &s(&[4, 8, 1])),
            Some(64)
        ); // rank-3 [B,S,D]+[B,S,1]
        assert_eq!(
            super::col_broadcast_cols(&s(&[4, 8, 64]), &s(&[4, 9, 1])),
            None
        ); // leading mismatch
    }

    #[test]
    fn fusion_rank3_trailing_broadcast_chain_matches_unfused_bit_for_bit() {
        // The hot transformer pattern: rank-3 [B,S,D] activations with a [D] row-
        // broadcast bias/scale AND a [B,S,1] col-broadcast, in one fused elementwise
        // chain. Must match the forced-unfused per-equation path bit-for-bit (incl
        // signed zeros / inf / NaN), proving the rank>2 matcher generalization is
        // sound. The fused output must also stay dense (so the fusion really ran).
        let (b, s_dim, d) = (4usize, 8usize, 64usize);
        let n = b * s_dim * d; // 2048 >= FUSION_MIN_ELEMS
        let mut x: Vec<f64> = (0..n).map(|i| i as f64 * 0.0017 - 5.0).collect();
        let mut scale: Vec<f64> = (0..d).map(|i| (i as f64 * 0.013).cos() + 1.1).collect();
        let mut colv: Vec<f64> = (0..b * s_dim).map(|i| i as f64 * 0.05 - 1.0).collect();
        x[0] = -0.0;
        x[1] = f64::INFINITY;
        x[2] = f64::from_bits(0x7ff8_0000_0000_1234);
        scale[0] = -0.0;
        scale[1] = f64::from_bits(1);
        colv[0] = f64::NEG_INFINITY;

        let xv = VarId(0);
        let sv = VarId(1); // [D] row-broadcast
        let cv = VarId(2); // [B,S,1] col-broadcast
        let v: Vec<VarId> = (3..=8).map(VarId).collect();
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let eqns = vec![
            mk(
                Primitive::Mul,
                smallvec![Atom::Var(xv), Atom::Var(sv)],
                v[0],
            ),
            mk(
                Primitive::Add,
                smallvec![Atom::Var(v[0]), Atom::Var(cv)],
                v[1],
            ),
            mk(
                Primitive::Sub,
                smallvec![Atom::Var(v[1]), Atom::Var(sv)],
                v[2],
            ),
            mk(Primitive::Mul, smallvec![Atom::Var(v[2]), lit(1.5)], v[3]),
            mk(
                Primitive::Add,
                smallvec![Atom::Var(v[3]), Atom::Var(cv)],
                v[4],
            ),
        ];
        let jaxpr = Jaxpr::new(vec![xv, sv, cv], vec![], vec![v[4]], eqns);
        let args = [
            f64_tensor_values(vec![b as u32, s_dim as u32, d as u32], x),
            f64_tensor_values(vec![d as u32], scale),
            f64_tensor_values(vec![b as u32, s_dim as u32, 1], colv),
        ];
        let fused = eval_jaxpr(&jaxpr, &args).unwrap();
        let unfused = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("unfused reference");
        let Value::Tensor(t) = &fused[0] else {
            panic!("tensor")
        };
        assert_eq!(t.shape.dims, vec![b as u32, s_dim as u32, d as u32]);
        assert!(
            t.elements.as_f64_slice().is_some(),
            "fused rank-3 broadcast output should stay dense"
        );
        assert_eq!(
            f64_bits(&fused[0]),
            f64_bits(&unfused[0]),
            "fused rank-3 trailing-broadcast chain must match unfused bit-for-bit"
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_fusion_rank3_broadcast_vs_unfused() {
        use std::time::Instant;
        // Rank-3 [B,S,D] + [D] bias/scale chain: fused (eval_jaxpr) vs forced
        // per-equation unfused path. The fusion does ONE pass; unfused materializes a
        // [B,S,D] intermediate per equation + re-dispatches.
        let (b, s_dim, d) = (64usize, 128usize, 256usize);
        let n = b * s_dim * d;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.0001).sin()).collect();
        let scale: Vec<f64> = (0..d).map(|i| 1.0 + i as f64 * 1e-4).collect();
        let bias: Vec<f64> = (0..d).map(|i| i as f64 * 1e-3).collect();
        let xv = VarId(0);
        let sv = VarId(1);
        let bvv = VarId(2);
        let v: Vec<VarId> = (3..=8).map(VarId).collect();
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let eqns = vec![
            mk(
                Primitive::Mul,
                smallvec![Atom::Var(xv), Atom::Var(sv)],
                v[0],
            ),
            mk(
                Primitive::Add,
                smallvec![Atom::Var(v[0]), Atom::Var(bvv)],
                v[1],
            ),
            mk(
                Primitive::Mul,
                smallvec![Atom::Var(v[1]), Atom::Var(sv)],
                v[2],
            ),
            mk(
                Primitive::Add,
                smallvec![Atom::Var(v[2]), Atom::Var(bvv)],
                v[3],
            ),
            mk(
                Primitive::Sub,
                smallvec![Atom::Var(v[3]), Atom::Var(bvv)],
                v[4],
            ),
        ];
        let jaxpr = Jaxpr::new(vec![xv, sv, bvv], vec![], vec![v[4]], eqns);
        let args = [
            f64_tensor_values(vec![b as u32, s_dim as u32, d as u32], x),
            f64_tensor_values(vec![d as u32], scale),
            f64_tensor_values(vec![d as u32], bias),
        ];
        let best = |f: &dyn Fn() -> u64| {
            let _ = f();
            let mut t = f64::MAX;
            let mut digest = 0u64;
            for _ in 0..5 {
                let s = Instant::now();
                digest = f();
                t = t.min(s.elapsed().as_secs_f64());
            }
            (t, digest)
        };
        let (t_fused, d_fused) = best(&|| {
            f64_bits(&eval_jaxpr(&jaxpr, &args).unwrap()[0])
                .iter()
                .fold(0u64, |a, &x| a ^ x)
        });
        let (t_unfused, d_unfused) = best(&|| {
            f64_bits(&eval_jaxpr_hashed_env(&jaxpr, &[], &args).unwrap()[0])
                .iter()
                .fold(0u64, |a, &x| a ^ x)
        });
        assert_eq!(d_fused, d_unfused, "fused vs unfused digest");
        println!(
            "BENCH rank3 [B{b},S{s_dim},D{d}] bias/scale chain: unfused={:.2}ms fused={:.2}ms speedup={:.2}x digest={d_fused:016x}",
            t_unfused * 1e3,
            t_fused * 1e3,
            t_unfused / t_fused
        );
    }

    #[test]
    fn fusion_i64_chain_matches_reference_bit_for_bit() {
        // Integer chain exercising every fused op, external tensor + scalar operands,
        // Neg, non-commutative Sub/Div order, and the exact wrapping/checked_div
        // semantics fj-lax's dispatcher uses — including i64::MIN/MAX overflow
        // (wrapping_mul/add) and division by zero / MIN/-1 (checked_div -> 0):
        //   v1 = mul(x, x); v2 = add(v1, 7); v3 = sub(1000, v2); v4 = div(v3, y);
        //   v5 = neg(v4); out = mul(v5, x)
        let n = 4096usize; // > FUSION_MIN_ELEMS, > one chunk
        let mut x: Vec<i64> = (0..n).map(|i| i as i64 - 2048).collect();
        let mut y: Vec<i64> = (0..n).map(|i| (i as i64 % 7) - 3).collect(); // includes 0
        x[0] = i64::MAX;
        x[1] = i64::MIN;
        x[2] = -1;
        x[3] = 0;
        y[0] = 0; // div-by-zero -> checked_div -> 0
        y[1] = -1; // pairs with a possible MIN numerator
        y[2] = 1;
        y[3] = i64::MIN;

        let xv = VarId(0);
        let yv = VarId(1);
        let (v1, v2, v3, v4, v5, out) =
            (VarId(2), VarId(3), VarId(4), VarId(5), VarId(6), VarId(7));
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let eqns = vec![
            mk(Primitive::Mul, smallvec![Atom::Var(xv), Atom::Var(xv)], v1),
            mk(Primitive::Add, smallvec![Atom::Var(v1), liti(7)], v2),
            mk(Primitive::Sub, smallvec![liti(1000), Atom::Var(v2)], v3),
            mk(Primitive::Div, smallvec![Atom::Var(v3), Atom::Var(yv)], v4),
            mk(Primitive::Neg, smallvec![Atom::Var(v4)], v5),
            mk(Primitive::Mul, smallvec![Atom::Var(v5), Atom::Var(xv)], out),
        ];
        let jaxpr = Jaxpr::new(vec![xv, yv], vec![], vec![out], eqns);
        let args = [
            i64_tensor_values(vec![n as u32], x.clone()),
            i64_tensor_values(vec![n as u32], y.clone()),
        ];

        let fused_outputs = eval_jaxpr(&jaxpr, &args).unwrap();
        let unfused_outputs =
            eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("unfused reference evaluates");
        let Value::Tensor(out_tensor) = &fused_outputs[0] else {
            panic!("expected tensor output")
        };
        assert_eq!(out_tensor.dtype, DType::I64);
        assert!(
            out_tensor.elements.as_i64_slice().is_some(),
            "fused i64 output should stay dense i64"
        );
        assert_eq!(
            i64_vals(&fused_outputs[0]),
            i64_vals(&unfused_outputs[0]),
            "fused i64 chain must match forced unfused path exactly"
        );

        // Cross-check against an independent manual fold of the documented closures.
        let want: Vec<i64> = (0..n)
            .map(|i| {
                let v1 = x[i].wrapping_mul(x[i]);
                let v2 = v1.wrapping_add(7);
                let v3 = 1000_i64.wrapping_sub(v2);
                let v4 = v3.checked_div(y[i]).unwrap_or(0);
                let v5 = v4.wrapping_neg();
                v5.wrapping_mul(x[i])
            })
            .collect();
        assert_eq!(i64_vals(&fused_outputs[0]), want, "manual fold cross-check");
        let digest =
            fj_test_utils::fixture_id_from_json(&want).expect("reference output should hash");
        assert_eq!(
            digest, "7f7e34d693a2f0e9f63a0d4575b03db01882e16d74746eb5b3978d8a6d25b297",
            "i64 fusion golden output digest must stay fixed"
        );
    }

    #[test]
    fn fusion_respects_multi_use_intermediate_boundary() {
        // v1 = add(x, 1.0); v2 = mul(v1, 2.0); out1 = sub(v2, 3.0); out2 = neg(v1)
        // v1 is used by BOTH v2 and out2 (multi-use) -> the run cannot extend past v1,
        // so v1 MUST be materialized. Both outputs must be correct.
        let n = 2048usize;
        let xv: Vec<f64> = (0..n).map(|i| i as f64 * 0.5 - 100.0).collect();
        let (x, v1, v2, out1, out2) = (VarId(0), VarId(1), VarId(2), VarId(3), VarId(4));
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let eqns = vec![
            mk(Primitive::Add, smallvec![Atom::Var(x), lit(1.0)], v1),
            mk(Primitive::Mul, smallvec![Atom::Var(v1), lit(2.0)], v2),
            mk(Primitive::Sub, smallvec![Atom::Var(v2), lit(3.0)], out1),
            mk(Primitive::Neg, smallvec![Atom::Var(v1)], out2),
        ];
        let jaxpr = Jaxpr::new(vec![x], vec![], vec![out1, out2], eqns);
        let res = eval_jaxpr(&jaxpr, &[f64_tensor(n, |i| xv[i])]).unwrap();
        let got1 = f64_vec(&res[0]);
        let got2 = f64_vec(&res[1]);
        for i in 0..n {
            let v1 = xv[i] + 1.0;
            assert_eq!(got1[i].to_bits(), ((v1 * 2.0) - 3.0).to_bits());
            assert_eq!(got2[i].to_bits(), (-v1).to_bits());
        }
    }

    #[test]
    fn eval_vector_add_one_jaxpr() {
        let jaxpr = build_program(ProgramSpec::AddOne);
        let output = eval_jaxpr(
            &jaxpr,
            &[Value::vector_i64(&[1, 2, 3]).expect("vector value should build")],
        )
        .expect("vector add should succeed");

        assert_eq!(
            output,
            vec![Value::vector_i64(&[2, 3, 4]).expect("vector value should build")]
        );
    }

    #[test]
    fn input_arity_mismatch_is_reported() {
        let jaxpr = build_program(ProgramSpec::Add2);
        let err = eval_jaxpr(&jaxpr, &[Value::scalar_i64(4)]).expect_err("should fail");
        assert_eq!(
            err,
            InterpreterError::InputArity {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn eval_with_constvars_binding_works() {
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![VarId(2)],
            vec![VarId(3)],
            vec![Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let outputs =
            eval_jaxpr_with_consts(&jaxpr, &[Value::scalar_i64(10)], &[Value::scalar_i64(7)])
                .expect("closed-over const path should evaluate");
        assert_eq!(outputs, vec![Value::scalar_i64(17)]);
    }

    // Build a two-equation chain `out = (a + b) * a` with variable ids offset
    // by `base`. `base == 0` yields dense ids (slot-array env); a large `base`
    // yields sparse ids that force the hash-map fallback env.
    fn chain_jaxpr_with_base(base: u32) -> Jaxpr {
        let a = VarId(base + 1);
        let b = VarId(base + 2);
        let sum = VarId(base + 3);
        let out = VarId(base + 4);
        Jaxpr::new(
            vec![a, b],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(a), Atom::Var(b)],
                    outputs: smallvec![sum],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(sum), Atom::Var(a)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn dense_and_hashed_envs_agree() {
        // (a + b) * a with a=4, b=5 -> 9 * 4 = 36.
        let args = [Value::scalar_i64(4), Value::scalar_i64(5)];
        let dense = eval_jaxpr(&chain_jaxpr_with_base(0), &args).expect("dense env evaluates");
        // base 5_000_000 makes slots_needed ~5e6 >> def_count*8, forcing the
        // hash-map fallback path.
        let hashed =
            eval_jaxpr(&chain_jaxpr_with_base(5_000_000), &args).expect("hashed env evaluates");
        assert_eq!(dense, vec![Value::scalar_i64(36)]);
        assert_eq!(dense, hashed);
    }

    fn scalar_i64_add_literal_chain(addends: &[i64]) -> Jaxpr {
        let equations: Vec<Equation> = addends
            .iter()
            .enumerate()
            .map(|(idx, addend)| Equation {
                primitive: Primitive::Add,
                inputs: smallvec![
                    Atom::Var(VarId((idx + 1) as u32)),
                    Atom::Lit(Literal::I64(*addend))
                ],
                outputs: smallvec![VarId((idx + 2) as u32)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            })
            .collect();
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId((addends.len() + 1) as u32)],
            equations,
        )
    }

    #[test]
    fn scalar_i64_add_chain_fast_path_matches_hashed_interpreter() {
        for len in [1_usize, 10, 1000] {
            let addends: Vec<i64> = (0..len)
                .map(|idx| ((idx as i64) * 17).wrapping_sub(31))
                .collect();
            let jaxpr = scalar_i64_add_literal_chain(&addends);
            let args = [Value::scalar_i64(123)];
            let fast = eval_jaxpr(&jaxpr, &args).expect("fast path evaluates");
            let hashed = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("hashed path evaluates");
            assert_eq!(fast, hashed, "chain length {len}");
        }
    }

    #[test]
    fn scalar_i64_add_chain_fast_path_preserves_wrapping() {
        let jaxpr = scalar_i64_add_literal_chain(&[1, 2]);
        let outputs =
            eval_jaxpr(&jaxpr, &[Value::scalar_i64(i64::MAX)]).expect("wrapping chain evaluates");
        assert_eq!(
            outputs,
            vec![Value::scalar_i64(i64::MAX.wrapping_add(1).wrapping_add(2))]
        );
    }

    #[test]
    fn scalar_i64_add_chain_guard_misses_fall_back() {
        let mut literal_left = scalar_i64_add_literal_chain(&[5]);
        literal_left.equations[0].inputs =
            smallvec![Atom::Lit(Literal::I64(5)), Atom::Var(VarId(1))];
        let args = [Value::scalar_i64(7)];
        assert_eq!(
            eval_jaxpr(&literal_left, &args),
            eval_jaxpr_hashed_env(&literal_left, &[], &args),
            "literal-left add should use the generic path"
        );

        let mut f64_literal = scalar_i64_add_literal_chain(&[5]);
        f64_literal.equations[0].inputs[1] = Atom::Lit(Literal::from_f64(5.0));
        assert_eq!(
            eval_jaxpr(&f64_literal, &args),
            eval_jaxpr_hashed_env(&f64_literal, &[], &args),
            "f64 literal add should use the generic path"
        );

        let mut params = scalar_i64_add_literal_chain(&[5]);
        params.equations[0]
            .params
            .insert("unused".into(), "1".into());
        assert_eq!(
            eval_jaxpr(&params, &args),
            eval_jaxpr_hashed_env(&params, &[], &args),
            "non-empty params should use the generic path"
        );

        let mut effects = scalar_i64_add_literal_chain(&[5]);
        effects.equations[0].effects.push("ordered".into());
        assert_eq!(
            eval_jaxpr(&effects, &args),
            eval_jaxpr_hashed_env(&effects, &[], &args),
            "equation effects should use the generic path"
        );

        let mut sub_jaxpr = scalar_i64_add_literal_chain(&[5]);
        sub_jaxpr.equations[0].sub_jaxprs.push(Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(1)],
            vec![],
        ));
        assert_eq!(
            eval_jaxpr(&sub_jaxpr, &args),
            eval_jaxpr_hashed_env(&sub_jaxpr, &[], &args),
            "sub-jaxpr equations should use the generic path"
        );

        let mut wrong_outvar = scalar_i64_add_literal_chain(&[5]);
        wrong_outvar.outvars[0] = VarId(99);
        assert_eq!(
            eval_jaxpr(&wrong_outvar, &args),
            eval_jaxpr_hashed_env(&wrong_outvar, &[], &args),
            "non-final outvars should keep the generic error behavior"
        );
    }

    fn scalar_f64_arith_body_jaxpr() -> Jaxpr {
        let (carry, x, prod, sum, out) = (VarId(0), VarId(1), VarId(2), VarId(3), VarId(4));
        let mk =
            |primitive: Primitive, inputs: smallvec::SmallVec<[Atom; 4]>, output: VarId| Equation {
                primitive,
                inputs,
                outputs: smallvec![output],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            };
        Jaxpr::new(
            vec![carry, x],
            vec![],
            vec![out],
            vec![
                mk(
                    Primitive::Mul,
                    smallvec![Atom::Var(carry), Atom::Var(x)],
                    prod,
                ),
                mk(
                    Primitive::Add,
                    smallvec![Atom::Var(prod), Atom::Var(carry)],
                    sum,
                ),
                mk(Primitive::Sub, smallvec![Atom::Var(sum), Atom::Var(x)], out),
            ],
        )
    }

    #[test]
    fn scalar_f64_arith_plan_matches_generic_bits() {
        let jaxpr = scalar_f64_arith_body_jaxpr();
        let cases = [
            (1.0001, 0.9999),
            (-0.0, 3.0),
            (f64::INFINITY, 0.0),
            (f64::NEG_INFINITY, -2.0),
            (f64::from_bits(0x7ff8_0000_0000_1234), 2.0),
            (f64::MIN_POSITIVE, -2.0),
        ];

        for (carry, x) in cases {
            let args = [Value::scalar_f64(carry), Value::scalar_f64(x)];
            let planned = eval_jaxpr(&jaxpr, &args).expect("planned path evaluates");
            let generic =
                eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic path evaluates");
            assert_eq!(planned, generic, "carry={carry:?} x={x:?}");
        }
    }

    #[test]
    fn scalar_f64_arith_plan_guard_miss_uses_generic_path() {
        let jaxpr = scalar_f64_arith_body_jaxpr();
        let args = [Value::scalar_i64(7), Value::scalar_i64(3)];
        let planned = eval_jaxpr(&jaxpr, &args).expect("guard miss should fall back");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic path evaluates");
        assert_eq!(planned, generic);
    }

    // Single-equation `div(a, b)` body for exercising the i64 checked_div / f64
    // div paths (div-by-zero, i64::MIN / -1 overflow).
    fn scalar_div_body_jaxpr() -> Jaxpr {
        let (a, b, out) = (VarId(0), VarId(1), VarId(2));
        Jaxpr::new(
            vec![a, b],
            vec![],
            vec![out],
            vec![Equation {
                primitive: Primitive::Div,
                inputs: smallvec![Atom::Var(a), Atom::Var(b)],
                outputs: smallvec![out],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        )
    }

    #[test]
    fn scalar_i64_arith_plan_matches_generic() {
        // The literal-free Mul/Add/Sub body builds multiple scalar plans; i64
        // args select the i64 executor at runtime. Cover wrapping at the extremes.
        let jaxpr = scalar_f64_arith_body_jaxpr();
        let cases = [
            (7_i64, 3),
            (-5, 4),
            (i64::MAX, 2),
            (i64::MIN, 3),
            (i64::MAX, i64::MAX),
            (1_000_000_007, 1_000_000_009),
        ];
        for (carry, x) in cases {
            let args = [Value::scalar_i64(carry), Value::scalar_i64(x)];
            let planned = eval_jaxpr(&jaxpr, &args).expect("planned");
            let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic");
            assert_eq!(planned, generic, "carry={carry} x={x}");
        }
    }

    #[test]
    fn scalar_i64_div_matches_generic() {
        let jaxpr = scalar_div_body_jaxpr();
        let cases = [(10_i64, 3), (10, 0), (i64::MIN, -1), (-7, 2), (0, 0)];
        for (a, b) in cases {
            let args = [Value::scalar_i64(a), Value::scalar_i64(b)];
            let planned = eval_jaxpr(&jaxpr, &args).expect("planned");
            let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic");
            assert_eq!(planned, generic, "a={a} b={b}");
        }
    }

    #[test]
    fn scalar_i64_guard_miss_uses_generic_path() {
        let jaxpr = scalar_f64_arith_body_jaxpr();
        let args = [Value::scalar_f64(7.5), Value::scalar_f64(3.0)];
        let planned = eval_jaxpr(&jaxpr, &args).expect("guard miss falls back");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic");
        assert_eq!(planned, generic);
    }

    #[test]
    fn scalar_f32_arith_plan_matches_generic_bits() {
        // f32 must match eval_primitive's widen->f64-op->narrow contract bit-for-bit,
        // including NaN / inf / -0 / subnormal (a native-f32 op would diverge on NaN).
        let jaxpr = scalar_f64_arith_body_jaxpr();
        let cases = [
            (1.0001_f32, 0.9999_f32),
            (-0.0, 3.0),
            (f32::INFINITY, 0.0),
            (f32::NEG_INFINITY, -2.0),
            (f32::from_bits(0x7fc0_1234), 2.0),
            (f32::MIN_POSITIVE, -2.0),
            (f32::from_bits(0x0000_0001), 3.0),
        ];
        for (carry, x) in cases {
            let args = [Value::scalar_f32(carry), Value::scalar_f32(x)];
            let planned = eval_jaxpr(&jaxpr, &args).expect("planned");
            let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic");
            assert_eq!(planned, generic, "carry={carry:?} x={x:?}");
        }
    }

    #[test]
    fn scalar_f32_div_matches_generic_bits() {
        let jaxpr = scalar_div_body_jaxpr();
        let cases = [
            (1.0_f32, 0.0_f32),
            (-1.0, 0.0),
            (0.0, 0.0),
            (3.5, 2.0),
            (f32::from_bits(0x7fc0_5678), 2.0),
        ];
        for (a, b) in cases {
            let args = [Value::scalar_f32(a), Value::scalar_f32(b)];
            let planned = eval_jaxpr(&jaxpr, &args).expect("planned");
            let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic");
            assert_eq!(planned, generic, "a={a:?} b={b:?}");
        }
    }

    // clamp-shaped body `out = min(max(a, b), a)` — exercises Max then Min, the
    // relu/clamp/saturation pattern that scalar recurrences use.
    fn scalar_minmax_body_jaxpr() -> Jaxpr {
        let (a, b, m, out) = (VarId(0), VarId(1), VarId(2), VarId(3));
        let mk =
            |primitive: Primitive, inputs: smallvec::SmallVec<[Atom; 4]>, output: VarId| Equation {
                primitive,
                inputs,
                outputs: smallvec![output],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            };
        Jaxpr::new(
            vec![a, b],
            vec![],
            vec![out],
            vec![
                mk(Primitive::Max, smallvec![Atom::Var(a), Atom::Var(b)], m),
                mk(Primitive::Min, smallvec![Atom::Var(m), Atom::Var(a)], out),
            ],
        )
    }

    #[test]
    fn scalar_f64_minmax_matches_generic_bits() {
        // JAX Max/Min propagate NaN (canonical), unlike Rust f64::max/min — the
        // plan must match eval_primitive's jax_max/jax_min bit-for-bit.
        let jaxpr = scalar_minmax_body_jaxpr();
        let cases = [
            (1.0_f64, 2.0_f64),
            (-0.0, 0.0),
            (f64::INFINITY, 5.0),
            (f64::NEG_INFINITY, 5.0),
            (f64::NAN, 1.0),
            (1.0, f64::NAN),
            (f64::from_bits(0x7ff8_0000_0000_1234), 2.0),
        ];
        for (a, b) in cases {
            let args = [Value::scalar_f64(a), Value::scalar_f64(b)];
            let planned = eval_jaxpr(&jaxpr, &args).expect("planned");
            let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic");
            assert_eq!(planned, generic, "a={a:?} b={b:?}");
        }
    }

    #[test]
    fn scalar_i64_minmax_matches_generic() {
        let jaxpr = scalar_minmax_body_jaxpr();
        let cases = [(1_i64, 2), (-5, 4), (i64::MAX, i64::MIN), (7, 7)];
        for (a, b) in cases {
            let args = [Value::scalar_i64(a), Value::scalar_i64(b)];
            let planned = eval_jaxpr(&jaxpr, &args).expect("planned");
            let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic");
            assert_eq!(planned, generic, "a={a} b={b}");
        }
    }

    #[test]
    fn scalar_f32_minmax_matches_generic_bits() {
        let jaxpr = scalar_minmax_body_jaxpr();
        let cases = [
            (1.0_f32, 2.0_f32),
            (f32::INFINITY, 5.0),
            (f32::NAN, 1.0),
            (1.0, f32::NAN),
            (f32::from_bits(0x7fc0_1234), 2.0),
        ];
        for (a, b) in cases {
            let args = [Value::scalar_f32(a), Value::scalar_f32(b)];
            let planned = eval_jaxpr(&jaxpr, &args).expect("planned");
            let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic");
            assert_eq!(planned, generic, "a={a:?} b={b:?}");
        }
    }

    fn scalar_half_literal(dtype: DType, bits: u16) -> Literal {
        if dtype == DType::BF16 {
            Literal::BF16Bits(bits)
        } else {
            Literal::F16Bits(bits)
        }
    }

    fn scalar_half_from_f64(dtype: DType, value: f64) -> Literal {
        if dtype == DType::BF16 {
            Literal::from_bf16_f64(value)
        } else {
            Literal::from_f16_f64(value)
        }
    }

    fn scalar_half_output_bits(output: &Value, dtype: DType) -> u16 {
        match (dtype, output) {
            (DType::BF16, Value::Scalar(Literal::BF16Bits(bits)))
            | (DType::F16, Value::Scalar(Literal::F16Bits(bits))) => *bits,
            _ => 0,
        }
    }

    fn scalar_half_arith_body_jaxpr(dtype: DType) -> Jaxpr {
        let (x, y, neg, abs, prod, sum, quot, out) = (
            VarId(0),
            VarId(1),
            VarId(2),
            VarId(3),
            VarId(4),
            VarId(5),
            VarId(6),
            VarId(7),
        );
        let mk =
            |primitive: Primitive, inputs: smallvec::SmallVec<[Atom; 4]>, output: VarId| Equation {
                primitive,
                inputs,
                outputs: smallvec![output],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            };
        Jaxpr::new(
            vec![x, y],
            vec![],
            vec![out],
            vec![
                mk(Primitive::Neg, smallvec![Atom::Var(x)], neg),
                mk(Primitive::Abs, smallvec![Atom::Var(neg)], abs),
                mk(
                    Primitive::Mul,
                    smallvec![Atom::Var(abs), Atom::Var(y)],
                    prod,
                ),
                mk(
                    Primitive::Add,
                    smallvec![
                        Atom::Var(prod),
                        Atom::Lit(scalar_half_from_f64(dtype, 0.25))
                    ],
                    sum,
                ),
                mk(
                    Primitive::Div,
                    smallvec![Atom::Var(sum), Atom::Var(y)],
                    quot,
                ),
                mk(
                    Primitive::Max,
                    smallvec![Atom::Var(quot), Atom::Var(x)],
                    out,
                ),
            ],
        )
    }

    #[test]
    fn scalar_half_arith_plan_matches_generic_bits() {
        // Scalar BF16/F16 must preserve the existing half contract exactly:
        // widen each operand to f64, apply the op in equation order, then round
        // every intermediate back to the same half dtype. The forced hash-map
        // path is the reference because it cannot take the dense scalar arena.
        let mut golden_rows: Vec<(&'static str, u16, u16, u16)> = Vec::new();
        let cases = [
            (0x3f80, 0x4000), // 1.0, 2.0 in BF16; harmless finite F16 patterns too
            (0x8000, 0x3f80), // -0.0, finite
            (0x7f80, 0x4000), // +inf, finite (BF16)
            (0xff80, 0x4000), // -inf, finite (BF16)
            (0x7fc1, 0x4000), // NaN payload, finite (BF16)
            (0x0001, 0x4000), // smallest subnormal-ish payload
        ];
        for dtype in [DType::BF16, DType::F16] {
            let jaxpr = scalar_half_arith_body_jaxpr(dtype);
            let plan = super::build_dense_plan(&jaxpr).expect("dense plan should build");
            if dtype == DType::BF16 {
                assert!(plan.scalar_bf16_plan.is_some());
                assert!(plan.scalar_f16_plan.is_none());
            } else {
                assert!(plan.scalar_bf16_plan.is_none());
                assert!(plan.scalar_f16_plan.is_some());
            }
            for (a, b) in cases {
                let args = [
                    Value::Scalar(scalar_half_literal(dtype, a)),
                    Value::Scalar(scalar_half_literal(dtype, b)),
                ];
                let planned = eval_jaxpr(&jaxpr, &args).expect("planned");
                let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic reference");
                assert_eq!(planned, generic, "dtype={dtype:?} a={a:#06x} b={b:#06x}");
                let dtype_name = if dtype == DType::BF16 { "bf16" } else { "f16" };
                golden_rows.push((
                    dtype_name,
                    a,
                    b,
                    scalar_half_output_bits(&planned[0], dtype),
                ));
            }
        }
        let digest = fj_test_utils::fixture_id_from_json(&golden_rows)
            .expect("scalar half golden rows should hash");
        assert_eq!(
            digest,
            "2a61385a28dd56a659b204ce161fb1d9cbcd30bd6a6f8e42e57f328894746d1c"
        );
    }

    #[test]
    fn top_level_scalar_half_arith_fast_path_matches_generic_and_golden() {
        let mut golden_rows: Vec<(&'static str, &'static str, u16, u16, u16)> = Vec::new();
        let cases = [
            (0x3f80, 0x4000),
            (0x8000, 0x3f80),
            (0x7f80, 0x4000),
            (0xff80, 0x4000),
            (0x7fc1, 0x4000),
            (0x0001, 0x4000),
        ];

        for dtype in [DType::BF16, DType::F16] {
            let jaxpr = scalar_half_arith_body_jaxpr(dtype);
            let dtype_name = if dtype == DType::BF16 { "bf16" } else { "f16" };
            for (a, b) in cases {
                let args = [
                    Value::Scalar(scalar_half_literal(dtype, a)),
                    Value::Scalar(scalar_half_literal(dtype, b)),
                ];

                let fast = super::try_eval_top_level_scalar_half_arith(&jaxpr, &[], &args)
                    .expect("top-level half fast path should match")
                    .expect("top-level half fast path should evaluate");
                let planned = eval_jaxpr(&jaxpr, &args).expect("planned eval");
                let generic =
                    eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic hash-map reference");
                assert_eq!(fast, generic, "fast dtype={dtype:?} a={a:#06x} b={b:#06x}");
                assert_eq!(
                    planned, generic,
                    "eval dtype={dtype:?} a={a:#06x} b={b:#06x}"
                );
                golden_rows.push((
                    "frankenjax-m6kqz",
                    dtype_name,
                    a,
                    b,
                    scalar_half_output_bits(&fast[0], dtype),
                ));
            }
        }

        let digest = fj_test_utils::fixture_id_from_json(&golden_rows)
            .expect("top-level scalar half golden rows should hash");
        eprintln!("top-level scalar half arithmetic golden digest: {digest}");
        assert_eq!(
            digest,
            "fdd1466145016f889175119a82d8a56655c636ca3beb53b5229865ab35ccaf1b"
        );
    }

    // Single-comparison cond body `out = cmp(x, lit)` — the canonical while/scan
    // predicate the scalar-compare plan accelerates.
    fn scalar_compare_body_jaxpr(op: Primitive, lit: Literal) -> Jaxpr {
        let (x, out) = (VarId(0), VarId(1));
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![Equation {
                primitive: op,
                inputs: smallvec![Atom::Var(x), Atom::Lit(lit)],
                outputs: smallvec![out],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        )
    }

    // Compound cond body `out = lo < x && x < hi`. This is the common range
    // predicate shape that currently misses the single-comparison scalar plan.
    fn compound_scalar_compare_logic_body_jaxpr(
        lo: Literal,
        hi: Literal,
        logic: Primitive,
    ) -> Jaxpr {
        let (x, lower_ok, upper_ok, out) = (VarId(0), VarId(1), VarId(2), VarId(3));
        let mk =
            |primitive: Primitive, inputs: smallvec::SmallVec<[Atom; 4]>, output: VarId| Equation {
                primitive,
                inputs,
                outputs: smallvec![output],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            };
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                mk(
                    Primitive::Lt,
                    smallvec![Atom::Lit(lo), Atom::Var(x)],
                    lower_ok,
                ),
                mk(
                    Primitive::Lt,
                    smallvec![Atom::Var(x), Atom::Lit(hi)],
                    upper_ok,
                ),
                mk(
                    logic,
                    smallvec![Atom::Var(lower_ok), Atom::Var(upper_ok)],
                    out,
                ),
            ],
        )
    }

    fn compound_scalar_compare_body_jaxpr(lo: Literal, hi: Literal) -> Jaxpr {
        compound_scalar_compare_logic_body_jaxpr(lo, hi, Primitive::BitwiseAnd)
    }

    fn compound_scalar_compare_nonmax_output_jaxpr() -> Jaxpr {
        let (x, out, lower_ok, upper_ok) = (VarId(0), VarId(1), VarId(2), VarId(3));
        let mk =
            |primitive: Primitive, inputs: smallvec::SmallVec<[Atom; 4]>, output: VarId| Equation {
                primitive,
                inputs,
                outputs: smallvec![output],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            };
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                mk(
                    Primitive::Lt,
                    smallvec![Atom::Lit(Literal::I64(0)), Atom::Var(x)],
                    lower_ok,
                ),
                mk(
                    Primitive::Lt,
                    smallvec![Atom::Var(x), Atom::Lit(Literal::I64(10))],
                    upper_ok,
                ),
                mk(
                    Primitive::BitwiseAnd,
                    smallvec![Atom::Var(lower_ok), Atom::Var(upper_ok)],
                    out,
                ),
            ],
        )
    }

    #[test]
    fn scalar_compare_i64_matches_generic() {
        let ops = [
            Primitive::Eq,
            Primitive::Ne,
            Primitive::Lt,
            Primitive::Le,
            Primitive::Gt,
            Primitive::Ge,
        ];
        for op in ops {
            let jaxpr = scalar_compare_body_jaxpr(op, Literal::I64(0));
            for x in [-3_i64, 0, 5, i64::MIN, i64::MAX] {
                let args = [Value::scalar_i64(x)];
                let planned = eval_jaxpr(&jaxpr, &args).expect("planned");
                let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic");
                assert_eq!(planned, generic, "op={op:?} x={x}");
            }
        }
    }

    #[test]
    fn scalar_compare_f64_matches_generic_incl_nan() {
        let ops = [
            Primitive::Eq,
            Primitive::Ne,
            Primitive::Lt,
            Primitive::Le,
            Primitive::Gt,
            Primitive::Ge,
        ];
        for op in ops {
            let jaxpr = scalar_compare_body_jaxpr(op, Literal::from_f64(1.0));
            for x in [
                -2.0_f64,
                1.0,
                3.0,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                -0.0,
            ] {
                let args = [Value::scalar_f64(x)];
                let planned = eval_jaxpr(&jaxpr, &args).expect("planned");
                let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic");
                assert_eq!(planned, generic, "op={op:?} x={x:?}");
            }
        }
    }

    #[test]
    fn scalar_compare_f32_matches_generic_incl_nan() {
        let jaxpr = scalar_compare_body_jaxpr(Primitive::Lt, Literal::from_f32(1.0_f32));
        for x in [-2.0_f32, 1.0, 3.0, f32::NAN, f32::INFINITY] {
            let args = [Value::scalar_f32(x)];
            let planned = eval_jaxpr(&jaxpr, &args).expect("planned");
            let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic");
            assert_eq!(planned, generic, "x={x:?}");
        }
    }

    #[test]
    fn scalar_compare_guard_miss_uses_generic() {
        // Tensor operand → not a fast-path scalar → bails to generic.
        let jaxpr = scalar_compare_body_jaxpr(Primitive::Gt, Literal::I64(0));
        let t = f64_tensor(4, |i| i as f64 - 1.0);
        let planned = eval_jaxpr(&jaxpr, std::slice::from_ref(&t)).expect("planned");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[t]).expect("generic");
        assert_eq!(planned, generic);
    }

    #[test]
    fn compound_scalar_compare_i64_matches_generic() {
        let jaxpr = compound_scalar_compare_body_jaxpr(Literal::I64(0), Literal::I64(10));
        for x in [-1_i64, 0, 1, 5, 10, 11, i64::MIN, i64::MAX] {
            let args = [Value::scalar_i64(x)];
            let planned = eval_jaxpr(&jaxpr, &args).expect("planned");
            let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic");
            assert_eq!(planned, generic, "x={x}");
        }
        let golden = eval_jaxpr(&jaxpr, &[Value::scalar_i64(3)]).expect("golden");
        let sha256 = fj_test_utils::fixture_id_from_json(&("frankenjax-yyue9", &golden))
            .expect("golden digest");
        assert_eq!(
            sha256,
            "b4d2e55ea3321f3774d7b08216eb26e5fab867a5c8b3d5d35a86b7d826c505cb",
        );
    }

    #[test]
    fn compound_scalar_compare_f64_matches_generic_incl_nan() {
        let jaxpr =
            compound_scalar_compare_body_jaxpr(Literal::from_f64(0.0), Literal::from_f64(10.0));
        for x in [
            -0.0_f64,
            0.0,
            1.0,
            10.0,
            11.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            let args = [Value::scalar_f64(x)];
            let planned = eval_jaxpr(&jaxpr, &args).expect("planned");
            let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic");
            assert_eq!(planned, generic, "x={x:?}");
        }
    }

    #[test]
    fn compound_scalar_compare_f32_matches_generic_incl_nan() {
        let jaxpr =
            compound_scalar_compare_body_jaxpr(Literal::from_f32(0.0), Literal::from_f32(10.0));
        for x in [
            -0.0_f32,
            0.0,
            1.0,
            10.0,
            11.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let args = [Value::scalar_f32(x)];
            let planned = eval_jaxpr(&jaxpr, &args).expect("planned");
            let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic");
            assert_eq!(planned, generic, "x={x:?}");
        }
    }

    #[test]
    fn compound_scalar_compare_bool_logic_matches_generic() {
        for logic in [Primitive::BitwiseOr, Primitive::BitwiseXor] {
            let jaxpr =
                compound_scalar_compare_logic_body_jaxpr(Literal::I64(0), Literal::I64(10), logic);
            for x in [-1_i64, 0, 1, 5, 10, 11] {
                let args = [Value::scalar_i64(x)];
                let planned = eval_jaxpr(&jaxpr, &args).expect("planned");
                let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic");
                assert_eq!(planned, generic, "logic={logic:?} x={x}");
            }
        }
    }

    #[test]
    fn compound_scalar_compare_guard_miss_uses_generic() {
        let jaxpr =
            compound_scalar_compare_body_jaxpr(Literal::from_f64(0.0), Literal::from_f64(10.0));
        let t = f64_tensor(4, |i| i as f64 - 1.0);
        let planned = eval_jaxpr(&jaxpr, std::slice::from_ref(&t)).expect("planned");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[t]).expect("generic");
        assert_eq!(planned, generic);
    }

    #[test]
    fn compound_scalar_compare_uses_jaxpr_outvar_slot() {
        let jaxpr = compound_scalar_compare_nonmax_output_jaxpr();
        for x in [-1_i64, 3, 10] {
            let args = [Value::scalar_i64(x)];
            let planned = eval_jaxpr(&jaxpr, &args).expect("planned");
            let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic");
            assert_eq!(planned, generic, "x={x}");
        }
    }

    // Body mixing unary + binary: `out = abs(neg(x) - y)` (Neg, Sub, Abs).
    // Literal-free -> builds multiple dtype plans; runtime dtype selects.
    fn scalar_unary_body_jaxpr() -> Jaxpr {
        let (x, y, n, d, out) = (VarId(0), VarId(1), VarId(2), VarId(3), VarId(4));
        let mk =
            |primitive: Primitive, inputs: smallvec::SmallVec<[Atom; 4]>, output: VarId| Equation {
                primitive,
                inputs,
                outputs: smallvec![output],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            };
        Jaxpr::new(
            vec![x, y],
            vec![],
            vec![out],
            vec![
                mk(Primitive::Neg, smallvec![Atom::Var(x)], n),
                mk(Primitive::Sub, smallvec![Atom::Var(n), Atom::Var(y)], d),
                mk(Primitive::Abs, smallvec![Atom::Var(d)], out),
            ],
        )
    }

    #[test]
    fn scalar_f64_unary_matches_generic_bits() {
        let jaxpr = scalar_unary_body_jaxpr();
        let cases = [
            (1.5_f64, 0.5_f64),
            (-0.0, 0.0),
            (f64::INFINITY, 1.0),
            (f64::NEG_INFINITY, 1.0),
            (f64::NAN, 2.0),
            (f64::from_bits(0x7ff8_0000_0000_1234), 2.0),
        ];
        for (x, y) in cases {
            let args = [Value::scalar_f64(x), Value::scalar_f64(y)];
            let planned = eval_jaxpr(&jaxpr, &args).expect("planned");
            let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic");
            assert_eq!(planned, generic, "x={x:?} y={y:?}");
        }
        let golden =
            eval_jaxpr(&jaxpr, &[Value::scalar_f64(2.5), Value::scalar_f64(-1.0)]).expect("golden");
        let sha256 = fj_test_utils::fixture_id_from_json(&("frankenjax-47a5o", &golden))
            .expect("golden digest");
        assert_eq!(
            sha256, "110a41d0f6848551f112522b6170fb5f5fbcf82f871a46f784b302beead86670",
            "scalar-unary golden output digest must stay fixed"
        );
    }

    #[test]
    fn scalar_i64_unary_matches_generic() {
        // wrapping_neg / wrapping_abs at i64::MIN must match eval_primitive.
        let jaxpr = scalar_unary_body_jaxpr();
        let cases = [
            (5_i64, 3),
            (-7, 2),
            (i64::MIN, 0),
            (i64::MIN, 1),
            (3, i64::MIN),
        ];
        for (x, y) in cases {
            let args = [Value::scalar_i64(x), Value::scalar_i64(y)];
            let planned = eval_jaxpr(&jaxpr, &args).expect("planned");
            let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic");
            assert_eq!(planned, generic, "x={x} y={y}");
        }
    }

    #[test]
    fn scalar_f32_unary_matches_generic_bits() {
        let jaxpr = scalar_unary_body_jaxpr();
        let cases = [
            (1.5_f32, 0.5_f32),
            (-0.0, 0.0),
            (f32::INFINITY, 1.0),
            (f32::NAN, 2.0),
            (f32::from_bits(0x7fc0_1234), 2.0),
        ];
        for (x, y) in cases {
            let args = [Value::scalar_f32(x), Value::scalar_f32(y)];
            let planned = eval_jaxpr(&jaxpr, &args).expect("planned");
            let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &args).expect("generic");
            assert_eq!(planned, generic, "x={x:?} y={y:?}");
        }
    }

    #[test]
    fn const_arity_mismatch_is_reported() {
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![VarId(2)],
            vec![VarId(3)],
            vec![Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let err = eval_jaxpr_with_consts(&jaxpr, &[], &[Value::scalar_i64(7)])
            .expect_err("const arity mismatch should fail");
        assert_eq!(
            err,
            InterpreterError::ConstArity {
                expected: 1,
                actual: 0,
            }
        );
    }

    #[test]
    fn eval_multi_output_qr_jaxpr() {
        let jaxpr = build_program(ProgramSpec::LaxQr);
        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0),
                ],
            )
            .expect("matrix tensor should build"),
        );

        let outputs = eval_jaxpr(&jaxpr, &[input]).expect("qr eval should succeed");
        assert_eq!(outputs.len(), 2);

        let q = outputs[0].as_tensor().expect("q should be tensor");
        let r = outputs[1].as_tensor().expect("r should be tensor");
        assert_eq!(q.shape, Shape { dims: vec![2, 2] });
        assert_eq!(r.shape, Shape { dims: vec![2, 2] });
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
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
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
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::from([("num_branches".to_owned(), "3".to_owned())]),
                sub_jaxprs: vec![
                    make_switch_branch_identity_jaxpr(),
                    make_switch_branch_self_binary_jaxpr(Primitive::Add),
                    make_switch_branch_self_binary_jaxpr(Primitive::Mul),
                ],
                effects: vec![],
            }],
        )
    }

    fn make_cond_control_flow_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3)],
            vec![Equation {
                primitive: Primitive::Cond,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![
                    make_switch_branch_self_binary_jaxpr(Primitive::Add),
                    make_switch_branch_self_binary_jaxpr(Primitive::Mul),
                ],
                effects: vec![],
            }],
        )
    }

    fn make_cond_branch_with_const(primitive: Primitive) -> Jaxpr {
        Jaxpr::new(
            vec![VarId(2)],
            vec![VarId(1)],
            vec![VarId(3)],
            vec![Equation {
                primitive,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        )
    }

    fn make_cond_with_const_binding_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1), VarId(2), VarId(3)],
            vec![],
            vec![VarId(4)],
            vec![Equation {
                primitive: Primitive::Cond,
                inputs: smallvec![
                    Atom::Var(VarId(1)),
                    Atom::Var(VarId(2)),
                    Atom::Var(VarId(3))
                ],
                outputs: smallvec![VarId(4)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![
                    make_cond_branch_with_const(Primitive::Add),
                    make_cond_branch_with_const(Primitive::Mul),
                ],
                effects: vec![],
            }],
        )
    }

    fn make_scan_body_add_emit_carry_jaxpr() -> Jaxpr {
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
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(3)), Atom::Lit(Literal::I64(0))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    fn make_scan_sub_jaxpr_control_flow_jaxpr(reverse: bool) -> Jaxpr {
        let params = if reverse {
            BTreeMap::from([("reverse".to_owned(), "true".to_owned())])
        } else {
            BTreeMap::new()
        };
        Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3), VarId(4)],
            vec![Equation {
                primitive: Primitive::Scan,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3), VarId(4)],
                params,
                sub_jaxprs: vec![make_scan_body_add_emit_carry_jaxpr()],
                effects: vec![],
            }],
        )
    }

    fn make_softmax_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
        let x = VarId(1);
        let max = VarId(2);
        let max_b = VarId(3);
        let shifted = VarId(4);
        let exp = VarId(5);
        let sum = VarId(6);
        let sum_b = VarId(7);
        let out = VarId(8);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        let bcast = BTreeMap::from([
            ("shape".to_owned(), format!("{rows},{cols}")),
            ("broadcast_dimensions".to_owned(), "0".to_owned()),
        ]);
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::ReduceMax,
                    inputs: smallvec![Atom::Var(x)],
                    outputs: smallvec![max],
                    params: reduce_axis1.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(max)],
                    outputs: smallvec![max_b],
                    params: bcast.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(x), Atom::Var(max_b)],
                    outputs: smallvec![shifted],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Exp,
                    inputs: smallvec![Atom::Var(shifted)],
                    outputs: smallvec![exp],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(exp)],
                    outputs: smallvec![sum],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(sum)],
                    outputs: smallvec![sum_b],
                    params: bcast,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(exp), Atom::Var(sum_b)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    fn make_log_softmax_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
        let x = VarId(1);
        let max = VarId(2);
        let max_b = VarId(3);
        let shifted = VarId(4);
        let exp = VarId(5);
        let sum = VarId(6);
        let logv = VarId(7);
        let log_b = VarId(8);
        let out = VarId(9);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        let bcast = BTreeMap::from([
            ("shape".to_owned(), format!("{rows},{cols}")),
            ("broadcast_dimensions".to_owned(), "0".to_owned()),
        ]);
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::ReduceMax,
                    inputs: smallvec![Atom::Var(x)],
                    outputs: smallvec![max],
                    params: reduce_axis1.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(max)],
                    outputs: smallvec![max_b],
                    params: bcast.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(x), Atom::Var(max_b)],
                    outputs: smallvec![shifted],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Exp,
                    inputs: smallvec![Atom::Var(shifted)],
                    outputs: smallvec![exp],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(exp)],
                    outputs: smallvec![sum],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Log,
                    inputs: smallvec![Atom::Var(sum)],
                    outputs: smallvec![logv],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(logv)],
                    outputs: smallvec![log_b],
                    params: bcast,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(shifted), Atom::Var(log_b)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    fn make_layer_norm_2d_jaxpr(rows: usize, cols: usize, epsilon: f64) -> Jaxpr {
        let x = VarId(1);
        let sum = VarId(2);
        let mean = VarId(3);
        let mean_b = VarId(4);
        let centered = VarId(5);
        let squared = VarId(6);
        let var_sum = VarId(7);
        let variance = VarId(8);
        let var_eps = VarId(9);
        let inv = VarId(10);
        let inv_b = VarId(11);
        let out = VarId(12);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        let bcast = BTreeMap::from([
            ("shape".to_owned(), format!("{rows},{cols}")),
            ("broadcast_dimensions".to_owned(), "0".to_owned()),
        ]);
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(x)],
                    outputs: smallvec![sum],
                    params: reduce_axis1.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(sum), Atom::Lit(Literal::from_f64(cols as f64))],
                    outputs: smallvec![mean],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(mean)],
                    outputs: smallvec![mean_b],
                    params: bcast.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(x), Atom::Var(mean_b)],
                    outputs: smallvec![centered],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Square,
                    inputs: smallvec![Atom::Var(centered)],
                    outputs: smallvec![squared],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(squared)],
                    outputs: smallvec![var_sum],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![
                        Atom::Var(var_sum),
                        Atom::Lit(Literal::from_f64(cols as f64))
                    ],
                    outputs: smallvec![variance],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(variance), Atom::Lit(Literal::from_f64(epsilon))],
                    outputs: smallvec![var_eps],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Rsqrt,
                    inputs: smallvec![Atom::Var(var_eps)],
                    outputs: smallvec![inv],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(inv)],
                    outputs: smallvec![inv_b],
                    params: bcast,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(centered), Atom::Var(inv_b)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    fn make_scan_body_f32_add_emit_carry_jaxpr() -> Jaxpr {
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
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(3)), Atom::Lit(Literal::from_f32(0.0))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    fn make_scan_f32_add_emit_control_flow_jaxpr(reverse: bool) -> Jaxpr {
        let params = if reverse {
            BTreeMap::from([("reverse".to_owned(), "true".to_owned())])
        } else {
            BTreeMap::new()
        };
        Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3), VarId(4)],
            vec![Equation {
                primitive: Primitive::Scan,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3), VarId(4)],
                params,
                sub_jaxprs: vec![make_scan_body_f32_add_emit_carry_jaxpr()],
                effects: vec![],
            }],
        )
    }

    fn make_scan_multi_carry_body_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1), VarId(2), VarId(3)],
            vec![],
            vec![VarId(4), VarId(5), VarId(6)],
            vec![
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(5)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(4)), Atom::Var(VarId(5))],
                    outputs: smallvec![VarId(6)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    fn make_scan_multi_carry_control_flow_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1), VarId(2), VarId(3)],
            vec![],
            vec![VarId(4), VarId(5), VarId(6)],
            vec![Equation {
                primitive: Primitive::Scan,
                inputs: smallvec![
                    Atom::Var(VarId(1)),
                    Atom::Var(VarId(2)),
                    Atom::Var(VarId(3))
                ],
                outputs: smallvec![VarId(4), VarId(5), VarId(6)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![make_scan_multi_carry_body_jaxpr()],
                effects: vec![],
            }],
        )
    }

    fn make_while_cond_gt_zero_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Gt,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(0))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        )
    }

    fn make_while_body_sub_step_jaxpr(step: i64) -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Sub,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(step))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        )
    }

    fn make_while_cond_gt_const_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(2)],
            vec![VarId(1)],
            vec![VarId(3)],
            vec![Equation {
                primitive: Primitive::Gt,
                inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        )
    }

    fn make_while_body_sub_const_jaxpr() -> Jaxpr {
        Jaxpr::new(
            vec![VarId(2)],
            vec![VarId(1)],
            vec![VarId(3)],
            vec![Equation {
                primitive: Primitive::Sub,
                inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        )
    }

    fn make_while_control_flow_jaxpr(step: i64, max_iter: usize) -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::While,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::from([("max_iter".to_owned(), max_iter.to_string())]),
                sub_jaxprs: vec![
                    make_while_cond_gt_zero_jaxpr(),
                    make_while_body_sub_step_jaxpr(step),
                ],
                effects: vec![],
            }],
        )
    }

    fn make_while_control_flow_with_const_bindings_jaxpr(max_iter: usize) -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1), VarId(2), VarId(3)],
            vec![],
            vec![VarId(4)],
            vec![Equation {
                primitive: Primitive::While,
                inputs: smallvec![
                    Atom::Var(VarId(1)),
                    Atom::Var(VarId(2)),
                    Atom::Var(VarId(3))
                ],
                outputs: smallvec![VarId(4)],
                params: BTreeMap::from([("max_iter".to_owned(), max_iter.to_string())]),
                sub_jaxprs: vec![
                    make_while_cond_gt_const_jaxpr(),
                    make_while_body_sub_const_jaxpr(),
                ],
                effects: vec![],
            }],
        )
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_while_loop_interpreter_overhead() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        // SAME-INVOCATION A/B: the per-iteration sub-jaxpr eval is the loop body
        // cost. OLD = self-contained `eval_jaxpr_with_consts` (re-derives slot +
        // liveness analysis and allocates env/last_use/scratch every call). NEW =
        // `run_dense_env_buf` over a plan + buffers built ONCE (what the `while`
        // handler now does). Both compute the identical scalar Sub.
        let n: usize = 4_000_000;
        let body = make_while_body_sub_step_jaxpr(1);
        let arg = vec![Value::scalar_i64(7)];

        let t_old = best_time(|| {
            let mut acc = 0i64;
            for _ in 0..n {
                let out = eval_jaxpr_with_consts(&body, &[], &arg).expect("old");
                if let Value::Scalar(Literal::I64(v)) = &out[0] {
                    acc = acc.wrapping_add(*v);
                }
            }
            std::hint::black_box(acc);
        });

        let plan = super::build_dense_plan(&body).expect("dense");
        let t_new = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut out: Vec<Value> = Vec::new();
            let mut acc = 0i64;
            for _ in 0..n {
                super::run_dense_env_into(
                    &body,
                    &[],
                    &arg,
                    &mut env,
                    &plan.last_use,
                    &mut scratch,
                    &mut out,
                )
                .expect("new");
                if let Value::Scalar(Literal::I64(v)) = &out[0] {
                    acc = acc.wrapping_add(*v);
                }
            }
            std::hint::black_box(acc);
        });

        println!(
            "BENCH while-body dispatch {n} evals: OLD {:.1}ns/eval -> NEW {:.1}ns/eval = {:.2}x",
            t_old * 1e9 / n as f64,
            t_new * 1e9 / n as f64,
            t_old / t_new,
        );
    }

    #[test]
    fn scalar_select_arena_bit_identical_to_generic() {
        // The mixed f64/bool SELECT arena must match the generic interpreter bit-for-bit
        // on conditional scalar bodies (leaky-relu / relu6-clamp / where), AND actually
        // be selected (scalar_select_plan.is_some()) — a non-vacuous check.
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let lit = |v: f64| Atom::Lit(Literal::from_f64(v));
        let bits_f64 = |v: &Value| -> u64 {
            match v {
                Value::Scalar(Literal::F64Bits(b)) => *b,
                other => panic!("expected f64 scalar, got {other:?}"),
            }
        };
        let xs = [
            f64::NEG_INFINITY,
            -100.0,
            -6.5,
            -1.0,
            -0.0,
            0.0,
            0.01,
            3.0,
            6.0,
            6.5,
            100.0,
            f64::INFINITY,
            f64::NAN,
        ];

        // leaky_relu(x) = select(x > 0, x, 0.01*x).
        let (x, gt, scaled, out) = (VarId(0), VarId(1), VarId(2), VarId(3));
        let leaky = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                mk(Primitive::Gt, smallvec![Atom::Var(x), lit(0.0)], gt),
                mk(Primitive::Mul, smallvec![Atom::Var(x), lit(0.01)], scaled),
                mk(
                    Primitive::Select,
                    smallvec![Atom::Var(gt), Atom::Var(x), Atom::Var(scaled)],
                    out,
                ),
            ],
        );

        // relu6(x) = clamp to [0,6] via nested select: select(x>6, 6, select(x<0, 0, x)).
        let (x2, gt6, lt0, inner, out2) = (VarId(0), VarId(1), VarId(2), VarId(3), VarId(4));
        let relu6 = Jaxpr::new(
            vec![x2],
            vec![],
            vec![out2],
            vec![
                mk(Primitive::Gt, smallvec![Atom::Var(x2), lit(6.0)], gt6),
                mk(Primitive::Lt, smallvec![Atom::Var(x2), lit(0.0)], lt0),
                mk(
                    Primitive::Select,
                    smallvec![Atom::Var(lt0), lit(0.0), Atom::Var(x2)],
                    inner,
                ),
                mk(
                    Primitive::Select,
                    smallvec![Atom::Var(gt6), lit(6.0), Atom::Var(inner)],
                    out2,
                ),
            ],
        );

        for (label, body) in [("leaky_relu", &leaky), ("relu6", &relu6)] {
            let plan = super::build_dense_plan(body).expect("dense plan");
            assert!(
                plan.scalar_select_plan.is_some(),
                "{label} not routed through the select arena"
            );
            // The pure-f64 plan must NOT swallow a select body (it has no Select op).
            assert!(
                plan.scalar_f64_plan.is_none(),
                "{label} unexpectedly built an f64-only plan"
            );
            for &xv in &xs {
                let args = [Value::scalar_f64(xv)];
                let mut genv: Vec<Option<Value>> = vec![None; plan.slots];
                let mut gscr: Vec<Value> = Vec::new();
                let mut gout: Vec<Value> = Vec::new();
                super::run_dense_env_into(
                    body,
                    &[],
                    &args,
                    &mut genv,
                    &plan.last_use,
                    &mut gscr,
                    &mut gout,
                )
                .expect("generic select");
                let mut cenv: Vec<Option<Value>> = vec![None; plan.slots];
                let mut cscr: Vec<Value> = Vec::new();
                let mut cout: Vec<Value> = Vec::new();
                let mut bufs = super::ScalarPlanBuffers::default();
                super::run_dense_plan_into(
                    body,
                    &[],
                    &args,
                    &mut cenv,
                    &plan,
                    &mut cscr,
                    &mut cout,
                    &mut bufs,
                )
                .expect("compiled select");
                assert_eq!(
                    bits_f64(&cout[0]),
                    bits_f64(&gout[0]),
                    "{label}({xv}) select arena bits differ from generic"
                );
            }
        }

        // select with a bool INPUT predicate: select(pred, a, b).
        let (p, a, b, o) = (VarId(0), VarId(1), VarId(2), VarId(3));
        let where_body = Jaxpr::new(
            vec![p, a, b],
            vec![],
            vec![o],
            vec![mk(
                Primitive::Select,
                smallvec![Atom::Var(p), Atom::Var(a), Atom::Var(b)],
                o,
            )],
        );
        let plan = super::build_dense_plan(&where_body).expect("dense plan");
        assert!(plan.scalar_select_plan.is_some());
        for &pred in &[true, false] {
            for &(av, bv) in &[(1.5_f64, -2.0), (f64::NAN, 0.0), (-0.0, 0.0)] {
                let args = [
                    Value::scalar_bool(pred),
                    Value::scalar_f64(av),
                    Value::scalar_f64(bv),
                ];
                let mut genv: Vec<Option<Value>> = vec![None; plan.slots];
                let mut gscr: Vec<Value> = Vec::new();
                let mut gout: Vec<Value> = Vec::new();
                super::run_dense_env_into(
                    &where_body,
                    &[],
                    &args,
                    &mut genv,
                    &plan.last_use,
                    &mut gscr,
                    &mut gout,
                )
                .expect("generic where");
                let mut cenv: Vec<Option<Value>> = vec![None; plan.slots];
                let mut cscr: Vec<Value> = Vec::new();
                let mut cout: Vec<Value> = Vec::new();
                let mut bufs = super::ScalarPlanBuffers::default();
                super::run_dense_plan_into(
                    &where_body,
                    &[],
                    &args,
                    &mut cenv,
                    &plan,
                    &mut cscr,
                    &mut cout,
                    &mut bufs,
                )
                .expect("compiled where");
                assert_eq!(
                    bits_f64(&cout[0]),
                    bits_f64(&gout[0]),
                    "where({pred},{av},{bv}) select arena bits differ"
                );
            }
        }
    }

    #[test]
    fn dense_i64_tensor_arena_bit_identical_to_generic() {
        // The i64 small-tensor arena must HANDLE small dense-i64 elementwise bodies
        // (Some(Ok)) and match the generic interpreter (incl. wrapping at i64::MIN/MAX),
        // BAIL for large bodies, and BAIL for I32-dtype tensors (generic-narrowed).
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let ilit = |v: i64| Atom::Lit(Literal::I64(v));
        let tensor = |data: &[i64]| -> Value {
            Value::Tensor(
                TensorValue::new_i64_values(
                    Shape {
                        dims: vec![data.len() as u32],
                    },
                    data.to_vec(),
                )
                .unwrap(),
            )
        };
        let vals = |v: &Value| -> Vec<i64> {
            match v {
                Value::Tensor(t) => t.elements.as_i64_slice().unwrap().to_vec(),
                Value::Scalar(Literal::I64(x)) => vec![*x],
                other => panic!("unexpected {other:?}"),
            }
        };

        let data: Vec<i64> = vec![i64::MIN, -1000, -1, 0, 1, 1000, i64::MAX, 42, -42, 7];

        // clamp-ish: max(x*2 + 1, 0)  (wrapping mul/add + i64 max).
        let (x, m, a, out) = (VarId(0), VarId(1), VarId(2), VarId(3));
        let body1 = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                mk(Primitive::Mul, smallvec![Atom::Var(x), ilit(2)], m),
                mk(Primitive::Add, smallvec![Atom::Var(m), ilit(1)], a),
                mk(Primitive::Max, smallvec![Atom::Var(a), ilit(0)], out),
            ],
        );
        // two same-shape tensors: (x + y) - x  (== y, but exercises two-tensor wrapping).
        let (xa, ya, s, o2) = (VarId(0), VarId(1), VarId(2), VarId(3));
        let body2 = Jaxpr::new(
            vec![xa, ya],
            vec![],
            vec![o2],
            vec![
                mk(Primitive::Add, smallvec![Atom::Var(xa), Atom::Var(ya)], s),
                mk(Primitive::Sub, smallvec![Atom::Var(s), Atom::Var(xa)], o2),
            ],
        );

        let cases: Vec<(&str, Jaxpr, Vec<Value>)> = vec![
            ("clamp", body1, vec![tensor(&data)]),
            (
                "two",
                body2,
                vec![
                    tensor(&data),
                    tensor(&[1, -2, 3, i64::MAX, i64::MIN, 9, -1, 2, -3, 25]),
                ],
            ),
        ];

        for (name, body, args) in &cases {
            let plan = super::build_dense_plan(body).expect("dense plan");
            let p = plan
                .scalar_i64_plan
                .as_ref()
                .expect("scalar i64 plan built for elementwise body");

            let mut genv: Vec<Option<Value>> = vec![None; plan.slots];
            let mut gscr: Vec<Value> = Vec::new();
            let mut gout: Vec<Value> = Vec::new();
            super::run_dense_env_into(
                body,
                &[],
                args,
                &mut genv,
                &plan.last_use,
                &mut gscr,
                &mut gout,
            )
            .expect("generic");

            let mut tout: Vec<Value> = Vec::new();
            let mut cells: Vec<super::DenseI64Cell> = Vec::new();
            let handled = super::run_scalar_i64_plan_as_tensor_into(
                p,
                &[],
                args,
                &mut tout,
                &mut cells,
                true,
            )
            .unwrap_or_else(|| panic!("{name}: i64 tensor arena bailed (None)"));
            handled.expect("i64 tensor arena ok");
            assert_eq!(
                vals(&tout[0]),
                vals(&gout[0]),
                "{name} i64 tensor arena vs generic"
            );
        }

        // I32-dtype tensor (i64-backed but generic-narrowed) must BAIL.
        let (xi, mi, oi) = (VarId(0), VarId(1), VarId(2));
        let i32_body = Jaxpr::new(
            vec![xi],
            vec![],
            vec![oi],
            vec![
                mk(Primitive::Mul, smallvec![Atom::Var(xi), ilit(2)], mi),
                mk(Primitive::Add, smallvec![Atom::Var(mi), ilit(1)], oi),
            ],
        );
        let plan = super::build_dense_plan(&i32_body).expect("plan");
        let p = plan.scalar_i64_plan.as_ref().unwrap();
        let i32_tensor = Value::Tensor(
            TensorValue::new(
                DType::I32,
                Shape { dims: vec![4] },
                vec![Literal::I64(5); 4],
            )
            .unwrap(),
        );
        let mut tout: Vec<Value> = Vec::new();
        let mut cells: Vec<super::DenseI64Cell> = Vec::new();
        assert!(
            super::run_scalar_i64_plan_as_tensor_into(
                p,
                &[],
                &[i32_tensor],
                &mut tout,
                &mut cells,
                true
            )
            .is_none(),
            "I32 tensor must bail (generic narrows i32 results)"
        );

        // Large i64 body must BAIL to the fusion path.
        let big = vec![1i64; super::FUSION_MIN_ELEMS];
        let plan = super::build_dense_plan(&i32_body).expect("plan");
        let p = plan.scalar_i64_plan.as_ref().unwrap();
        let mut tout: Vec<Value> = Vec::new();
        let mut cells: Vec<super::DenseI64Cell> = Vec::new();
        assert!(
            super::run_scalar_i64_plan_as_tensor_into(
                p,
                &[],
                &[tensor(&big)],
                &mut tout,
                &mut cells,
                true
            )
            .is_none(),
            "large i64 tensor must bail"
        );
    }

    #[test]
    fn dense_f32_tensor_arena_bit_identical_to_generic() {
        // f32 sibling: the arena must HANDLE small dense-f32 elementwise bodies (Some(Ok),
        // not a None bail) and match the generic interpreter bit-for-bit, incl.
        // transcendentals, NaN-prop max, and scalar broadcast — and BAIL for large bodies.
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let f32lit = |v: f32| Atom::Lit(Literal::F32Bits(v.to_bits()));
        let tensor = |data: &[f32]| -> Value {
            Value::Tensor(
                TensorValue::new_f32_values(
                    Shape {
                        dims: vec![data.len() as u32],
                    },
                    data.to_vec(),
                )
                .unwrap(),
            )
        };
        let bits = |v: &Value| -> Vec<u32> {
            match v {
                Value::Tensor(t) => t
                    .elements
                    .as_f32_slice()
                    .unwrap()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect(),
                Value::Scalar(Literal::F32Bits(b)) => vec![*b],
                other => panic!("unexpected {other:?}"),
            }
        };

        let data: Vec<f32> = vec![
            -3.0,
            -0.5,
            -0.0,
            0.0,
            0.5,
            1.0,
            2.0,
            f32::NAN,
            f32::INFINITY,
            7.5,
        ];

        // relu-ish: max(x*2 + 1, 0).
        let (x, m, a, out) = (VarId(0), VarId(1), VarId(2), VarId(3));
        let relu = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                mk(Primitive::Mul, smallvec![Atom::Var(x), f32lit(2.0)], m),
                mk(Primitive::Add, smallvec![Atom::Var(m), f32lit(1.0)], a),
                mk(Primitive::Max, smallvec![Atom::Var(a), f32lit(0.0)], out),
            ],
        );
        // transcendental: tanh(x * 0.5).
        let (x2, h, o2) = (VarId(0), VarId(1), VarId(2));
        let act = Jaxpr::new(
            vec![x2],
            vec![],
            vec![o2],
            vec![
                mk(Primitive::Mul, smallvec![Atom::Var(x2), f32lit(0.5)], h),
                mk(Primitive::Tanh, smallvec![Atom::Var(h)], o2),
            ],
        );
        // two same-shape tensors: (x + y) * x.
        let (xa, ya, s, o3) = (VarId(0), VarId(1), VarId(2), VarId(3));
        let two = Jaxpr::new(
            vec![xa, ya],
            vec![],
            vec![o3],
            vec![
                mk(Primitive::Add, smallvec![Atom::Var(xa), Atom::Var(ya)], s),
                mk(Primitive::Mul, smallvec![Atom::Var(s), Atom::Var(xa)], o3),
            ],
        );

        let cases: Vec<(&str, Jaxpr, Vec<Value>)> = vec![
            ("relu", relu, vec![tensor(&data)]),
            ("act", act, vec![tensor(&data)]),
            (
                "two",
                two,
                vec![
                    tensor(&data),
                    tensor(&[1.0, -2.0, 3.0, -0.0, 0.0, 9.0, -1.5, 2.5, -3.5, 0.25]),
                ],
            ),
        ];

        for (name, body, args) in &cases {
            let plan = super::build_dense_plan(body).expect("dense plan");
            let p = plan
                .scalar_f32_plan
                .as_ref()
                .expect("scalar f32 plan built for elementwise body");

            let mut genv: Vec<Option<Value>> = vec![None; plan.slots];
            let mut gscr: Vec<Value> = Vec::new();
            let mut gout: Vec<Value> = Vec::new();
            super::run_dense_env_into(
                body,
                &[],
                args,
                &mut genv,
                &plan.last_use,
                &mut gscr,
                &mut gout,
            )
            .expect("generic");

            let mut tout: Vec<Value> = Vec::new();
            let mut cells: Vec<super::DenseF32Cell> = Vec::new();
            let handled = super::run_scalar_f32_plan_as_tensor_into(
                p,
                &[],
                args,
                &mut tout,
                &mut cells,
                true,
            )
            .unwrap_or_else(|| panic!("{name}: f32 tensor arena bailed (None)"));
            handled.expect("f32 tensor arena ok");
            assert_eq!(
                bits(&tout[0]),
                bits(&gout[0]),
                "{name} f32 tensor arena vs generic"
            );
        }

        // Large LINEAR chain (>= FUSION_MIN_ELEMS, below the in-place ceiling) is now
        // HANDLED in place and must stay bit-identical to the generic per-op path.
        let big = vec![1.0f32; super::FUSION_MIN_ELEMS];
        let (xb, mb, ob) = (VarId(0), VarId(1), VarId(2));
        let big_body = Jaxpr::new(
            vec![xb],
            vec![],
            vec![ob],
            vec![
                mk(Primitive::Mul, smallvec![Atom::Var(xb), f32lit(2.0)], mb),
                mk(Primitive::Add, smallvec![Atom::Var(mb), f32lit(1.0)], ob),
            ],
        );
        let plan = super::build_dense_plan(&big_body).expect("plan");
        let p = plan.scalar_f32_plan.as_ref().unwrap();

        let mut genv: Vec<Option<Value>> = vec![None; plan.slots];
        let mut gscr: Vec<Value> = Vec::new();
        let mut gout: Vec<Value> = Vec::new();
        super::run_dense_env_into(
            &big_body,
            &[],
            &[tensor(&big)],
            &mut genv,
            &plan.last_use,
            &mut gscr,
            &mut gout,
        )
        .expect("generic large f32");

        let mut tout: Vec<Value> = Vec::new();
        let mut cells: Vec<super::DenseF32Cell> = Vec::new();
        super::run_scalar_f32_plan_as_tensor_into(
            p,
            &[],
            &[tensor(&big)],
            &mut tout,
            &mut cells,
            true,
        )
        .expect("large f32 linear chain handled in place")
        .expect("ok");
        assert_eq!(
            bits(&tout[0]),
            bits(&gout[0]),
            "large f32 in-place chain vs generic"
        );

        // The non-vectorized control still BAILS so the fusion owns it.
        let mut tout2: Vec<Value> = Vec::new();
        let mut cells2: Vec<super::DenseF32Cell> = Vec::new();
        assert!(
            super::run_scalar_f32_plan_as_tensor_into(
                p,
                &[],
                &[tensor(&big)],
                &mut tout2,
                &mut cells2,
                false,
            )
            .is_none(),
            "large f32 chain bails when in-place lever is disabled"
        );
    }

    #[test]
    fn dense_f64_tensor_arena_broadcast_bit_identical() {
        // rank-2 row ([R,C] op [C]) and col ([R,C] op [R,1]) broadcast must match the
        // generic interpreter bit-for-bit AND actually be HANDLED (Some(Ok)).
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let t2 = |dims: Vec<u32>, data: &[f64]| -> Value {
            Value::Tensor(TensorValue::new_f64_values(Shape { dims }, data.to_vec()).unwrap())
        };
        let bits = |v: &Value| -> Vec<u64> {
            match v {
                Value::Tensor(t) => t
                    .elements
                    .as_f64_slice()
                    .unwrap()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect(),
                other => panic!("unexpected {other:?}"),
            }
        };
        // x[2,3]; data incl NaN/inf/±0.
        let x = t2(vec![2, 3], &[-1.0, 0.0, f64::NAN, 2.5, -0.0, f64::INFINITY]);

        // ROW: out = (x + bias[3]) * 2.0.
        let (xv, bv, s, out) = (VarId(0), VarId(1), VarId(2), VarId(3));
        let row_body = Jaxpr::new(
            vec![xv, bv],
            vec![],
            vec![out],
            vec![
                mk(Primitive::Add, smallvec![Atom::Var(xv), Atom::Var(bv)], s),
                mk(
                    Primitive::Mul,
                    smallvec![Atom::Var(s), Atom::Lit(Literal::from_f64(2.0))],
                    out,
                ),
            ],
        );
        // COL: out = x * scale[2,1].
        let (xc, sc, oc) = (VarId(0), VarId(1), VarId(2));
        let col_body = Jaxpr::new(
            vec![xc, sc],
            vec![],
            vec![oc],
            vec![mk(
                Primitive::Mul,
                smallvec![Atom::Var(xc), Atom::Var(sc)],
                oc,
            )],
        );

        let cases: Vec<(&str, Jaxpr, Vec<Value>)> = vec![
            (
                "row",
                row_body,
                vec![x.clone(), t2(vec![3], &[10.0, -5.0, 0.5])],
            ),
            (
                "col",
                col_body,
                vec![x.clone(), t2(vec![2, 1], &[3.0, -2.0])],
            ),
        ];

        for (name, body, args) in &cases {
            let plan = super::build_dense_plan(body).expect("dense plan");
            let p = plan.scalar_f64_plan.as_ref().expect("f64 plan");

            let mut genv: Vec<Option<Value>> = vec![None; plan.slots];
            let mut gscr: Vec<Value> = Vec::new();
            let mut gout: Vec<Value> = Vec::new();
            super::run_dense_env_into(
                body,
                &[],
                args,
                &mut genv,
                &plan.last_use,
                &mut gscr,
                &mut gout,
            )
            .expect("generic");

            let mut tout: Vec<Value> = Vec::new();
            let mut cells: Vec<super::DenseF64Cell> = Vec::new();
            let handled = super::run_scalar_f64_plan_as_tensor_into(
                p,
                &[],
                args,
                &mut tout,
                &mut cells,
                true,
            )
            .unwrap_or_else(|| panic!("{name}: broadcast body bailed (None)"));
            handled.expect("ok");
            assert_eq!(
                bits(&tout[0]),
                bits(&gout[0]),
                "{name} broadcast arena vs generic"
            );
        }
    }

    #[test]
    fn dense_f64_tensor_arena_bit_identical_to_generic() {
        // The small-tensor dense-f64 elementwise arena must (a) actually HANDLE these
        // bodies (run_..._as_tensor_into returns Some(Ok), not a None bail) and (b)
        // match the generic interpreter bit-for-bit, including transcendentals, NaN-
        // propagating max/min, and scalar broadcast.
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let f64lit = |v: f64| Atom::Lit(Literal::from_f64(v));
        let tensor = |data: &[f64]| -> Value {
            Value::Tensor(
                TensorValue::new_f64_values(
                    Shape {
                        dims: vec![data.len() as u32],
                    },
                    data.to_vec(),
                )
                .unwrap(),
            )
        };
        let bits = |v: &Value| -> Vec<u64> {
            match v {
                Value::Tensor(t) => t
                    .elements
                    .as_f64_slice()
                    .unwrap()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect(),
                Value::Scalar(Literal::F64Bits(b)) => vec![*b],
                other => panic!("unexpected {other:?}"),
            }
        };

        let data: Vec<f64> = vec![
            -3.0,
            -0.5,
            -0.0,
            0.0,
            0.5,
            1.0,
            2.0,
            f64::NAN,
            f64::INFINITY,
            7.5,
        ];

        // relu-ish: max(x*2 + 1, 0)  (cheap ops + scalar broadcast + NaN-prop max).
        let (x, m, a, out) = (VarId(0), VarId(1), VarId(2), VarId(3));
        let relu = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                mk(Primitive::Mul, smallvec![Atom::Var(x), f64lit(2.0)], m),
                mk(Primitive::Add, smallvec![Atom::Var(m), f64lit(1.0)], a),
                mk(Primitive::Max, smallvec![Atom::Var(a), f64lit(0.0)], out),
            ],
        );
        // transcendental: tanh(x * 0.5)  (the arena handles ops the cheap fusion can't).
        let (x2, h, o2) = (VarId(0), VarId(1), VarId(2));
        let act = Jaxpr::new(
            vec![x2],
            vec![],
            vec![o2],
            vec![
                mk(Primitive::Mul, smallvec![Atom::Var(x2), f64lit(0.5)], h),
                mk(Primitive::Tanh, smallvec![Atom::Var(h)], o2),
            ],
        );
        // two tensor operands same shape: (x + y) * x.
        let (xa, ya, s, o3) = (VarId(0), VarId(1), VarId(2), VarId(3));
        let two = Jaxpr::new(
            vec![xa, ya],
            vec![],
            vec![o3],
            vec![
                mk(Primitive::Add, smallvec![Atom::Var(xa), Atom::Var(ya)], s),
                mk(Primitive::Mul, smallvec![Atom::Var(s), Atom::Var(xa)], o3),
            ],
        );

        let cases: Vec<(&str, Jaxpr, Vec<Value>)> = vec![
            ("relu", relu, vec![tensor(&data)]),
            ("act", act, vec![tensor(&data)]),
            (
                "two",
                two,
                vec![
                    tensor(&data),
                    tensor(&[1.0, -2.0, 3.0, -0.0, 0.0, 9.0, -1.5, 2.5, -3.5, 0.25]),
                ],
            ),
        ];

        for (name, body, args) in &cases {
            let plan = super::build_dense_plan(body).expect("dense plan");
            let p = plan
                .scalar_f64_plan
                .as_ref()
                .expect("scalar f64 plan built for elementwise body");

            // Generic reference.
            let mut genv: Vec<Option<Value>> = vec![None; plan.slots];
            let mut gscr: Vec<Value> = Vec::new();
            let mut gout: Vec<Value> = Vec::new();
            super::run_dense_env_into(
                body,
                &[],
                args,
                &mut genv,
                &plan.last_use,
                &mut gscr,
                &mut gout,
            )
            .expect("generic");

            // Tensor arena — must HANDLE it (Some(Ok)), proving non-vacuous.
            let mut tout: Vec<Value> = Vec::new();
            let mut cells: Vec<super::DenseF64Cell> = Vec::new();
            let handled = super::run_scalar_f64_plan_as_tensor_into(
                p,
                &[],
                args,
                &mut tout,
                &mut cells,
                true,
            )
            .unwrap_or_else(|| panic!("{name}: tensor arena bailed (None) — should handle it"));
            handled.expect("tensor arena ok");
            assert_eq!(
                bits(&tout[0]),
                bits(&gout[0]),
                "{name} tensor arena vs generic"
            );
        }

        // A LARGE LINEAR chain (>= FUSION_MIN_ELEMS, below the in-place ceiling) is now
        // HANDLED in place — one buffer mutated through the chain — and must stay bit-
        // identical to the generic per-op path (the L3-resident buffer-reuse lever).
        let big = vec![1.0f64; super::FUSION_MIN_ELEMS];
        let (xb, mb, ob) = (VarId(0), VarId(1), VarId(2));
        let big_body = Jaxpr::new(
            vec![xb],
            vec![],
            vec![ob],
            vec![
                mk(Primitive::Mul, smallvec![Atom::Var(xb), f64lit(2.0)], mb),
                mk(Primitive::Add, smallvec![Atom::Var(mb), f64lit(1.0)], ob),
            ],
        );
        let plan = super::build_dense_plan(&big_body).expect("plan");
        let p = plan.scalar_f64_plan.as_ref().unwrap();

        let mut genv: Vec<Option<Value>> = vec![None; plan.slots];
        let mut gscr: Vec<Value> = Vec::new();
        let mut gout: Vec<Value> = Vec::new();
        super::run_dense_env_into(
            &big_body,
            &[],
            &[tensor(&big)],
            &mut genv,
            &plan.last_use,
            &mut gscr,
            &mut gout,
        )
        .expect("generic large");

        let mut tout: Vec<Value> = Vec::new();
        let mut cells: Vec<super::DenseF64Cell> = Vec::new();
        super::run_scalar_f64_plan_as_tensor_into(
            p,
            &[],
            &[tensor(&big)],
            &mut tout,
            &mut cells,
            true,
        )
        .expect("large linear chain handled in place")
        .expect("ok");
        assert_eq!(
            bits(&tout[0]),
            bits(&gout[0]),
            "large in-place chain vs generic"
        );

        // The non-vectorized control path (and any non-linear large body) still BAILS so
        // the chunked/threaded fusion owns it.
        let mut tout2: Vec<Value> = Vec::new();
        let mut cells2: Vec<super::DenseF64Cell> = Vec::new();
        assert!(
            super::run_scalar_f64_plan_as_tensor_into(
                p,
                &[],
                &[tensor(&big)],
                &mut tout2,
                &mut cells2,
                false,
            )
            .is_none(),
            "large chain bails when vectorization (in-place lever) is disabled"
        );
    }

    #[test]
    fn scalar_poly_arena_bit_identical_to_generic() {
        // The polymorphic dtype-tagged core (f64/i64/bool + convert) must match the
        // generic interpreter bit-for-bit on dtype-MIXING bodies, and be actually
        // selected (scalar_poly_plan.is_some()). The generic path runs the real
        // fj-lax eval_convert_element_type, so any convert-semantics drift fails here.
        let mk = |p: Primitive,
                  ins: smallvec::SmallVec<[Atom; 4]>,
                  params: BTreeMap<String, String>,
                  o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params,
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let conv = |to: &str| {
            let mut p = BTreeMap::new();
            p.insert("new_dtype".to_owned(), to.to_owned());
            p
        };
        let np = BTreeMap::new;
        let f64lit = |v: f64| Atom::Lit(Literal::from_f64(v));
        // value-equality that is NaN-bit-aware for f64.
        let eq_val = |a: &Value, b: &Value, ctx: &str| match (a, b) {
            (Value::Scalar(Literal::F64Bits(x)), Value::Scalar(Literal::F64Bits(y))) => {
                assert_eq!(x, y, "{ctx} f64 bits")
            }
            (Value::Scalar(Literal::F32Bits(x)), Value::Scalar(Literal::F32Bits(y))) => {
                assert_eq!(x, y, "{ctx} f32 bits")
            }
            (Value::Scalar(Literal::BF16Bits(x)), Value::Scalar(Literal::BF16Bits(y))) => {
                assert_eq!(x, y, "{ctx} bf16 bits")
            }
            (Value::Scalar(Literal::F16Bits(x)), Value::Scalar(Literal::F16Bits(y))) => {
                assert_eq!(x, y, "{ctx} f16 bits")
            }
            (Value::Scalar(Literal::I64(x)), Value::Scalar(Literal::I64(y))) => {
                assert_eq!(x, y, "{ctx} i64")
            }
            (Value::Scalar(Literal::Bool(x)), Value::Scalar(Literal::Bool(y))) => {
                assert_eq!(x, y, "{ctx} bool")
            }
            (x, y) => panic!("{ctx}: dtype/shape mismatch {x:?} vs {y:?}"),
        };

        // Body 1 (i64 input): f = convert(i, f64); out = f*0.5 + 1.0  (int→float math).
        let (i, f, h, out) = (VarId(0), VarId(1), VarId(2), VarId(3));
        let to_float = Jaxpr::new(
            vec![i],
            vec![],
            vec![out],
            vec![
                mk(
                    Primitive::ConvertElementType,
                    smallvec![Atom::Var(i)],
                    conv("f64"),
                    f,
                ),
                mk(
                    Primitive::Mul,
                    smallvec![Atom::Var(f), f64lit(0.5)],
                    np(),
                    h,
                ),
                mk(
                    Primitive::Add,
                    smallvec![Atom::Var(h), f64lit(1.0)],
                    np(),
                    out,
                ),
            ],
        );
        // Body 2 (f64 input): out = convert(x, i64)  (truncation: NaN→0, inf saturate).
        let (x2, o2) = (VarId(0), VarId(1));
        let to_int = Jaxpr::new(
            vec![x2],
            vec![],
            vec![o2],
            vec![mk(
                Primitive::ConvertElementType,
                smallvec![Atom::Var(x2)],
                conv("i64"),
                o2,
            )],
        );
        // Body 3 (f64 input): bool count = convert(x > 0, i64); out = convert(x, bool) too.
        let (x3, gt, cnt) = (VarId(0), VarId(1), VarId(2));
        let bool_to_int = Jaxpr::new(
            vec![x3],
            vec![],
            vec![cnt],
            vec![
                mk(
                    Primitive::Gt,
                    smallvec![Atom::Var(x3), f64lit(0.0)],
                    np(),
                    gt,
                ),
                mk(
                    Primitive::ConvertElementType,
                    smallvec![Atom::Var(gt)],
                    conv("i64"),
                    cnt,
                ),
            ],
        );
        // Body 4 (f64 input): convert to bool directly.
        let (x4, o4) = (VarId(0), VarId(1));
        let to_bool = Jaxpr::new(
            vec![x4],
            vec![],
            vec![o4],
            vec![mk(
                Primitive::ConvertElementType,
                smallvec![Atom::Var(x4)],
                conv("bool"),
                o4,
            )],
        );

        let f64_inputs = [
            f64::NEG_INFINITY,
            -1e20,
            -3.7,
            -1.0,
            -0.0,
            0.0,
            0.5,
            1.0,
            3.7,
            1e20,
            f64::INFINITY,
            f64::NAN,
            9.2e18,
            -9.3e18,
        ];
        let i64_inputs = [i64::MIN, -1000, -1, 0, 1, 1000, 1_000_000_007, i64::MAX];

        let run_both = |body: &Jaxpr, arg: Value, ctx: &str| {
            let plan = super::build_dense_plan(body).expect("dense plan");
            assert!(
                plan.scalar_poly_plan.is_some(),
                "{ctx} not routed through poly core"
            );
            let args = [arg];
            let mut genv: Vec<Option<Value>> = vec![None; plan.slots];
            let mut gscr: Vec<Value> = Vec::new();
            let mut gout: Vec<Value> = Vec::new();
            super::run_dense_env_into(
                body,
                &[],
                &args,
                &mut genv,
                &plan.last_use,
                &mut gscr,
                &mut gout,
            )
            .expect("generic");
            let mut cenv: Vec<Option<Value>> = vec![None; plan.slots];
            let mut cscr: Vec<Value> = Vec::new();
            let mut cout: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            super::run_dense_plan_into(
                body,
                &[],
                &args,
                &mut cenv,
                &plan,
                &mut cscr,
                &mut cout,
                &mut bufs,
            )
            .expect("compiled");
            eq_val(&cout[0], &gout[0], ctx);
        };

        for &xv in &f64_inputs {
            run_both(&to_int, Value::scalar_f64(xv), &format!("to_int({xv})"));
            run_both(
                &bool_to_int,
                Value::scalar_f64(xv),
                &format!("bool_to_int({xv})"),
            );
            run_both(&to_bool, Value::scalar_f64(xv), &format!("to_bool({xv})"));
        }
        for &iv in &i64_inputs {
            run_both(&to_float, Value::scalar_i64(iv), &format!("to_float({iv})"));
            // i64 -> f32 goes VIA f64 in convert_literal; verify the round matches.
            let to_f32 = Jaxpr::new(
                vec![VarId(0)],
                vec![],
                vec![VarId(1)],
                vec![mk(
                    Primitive::ConvertElementType,
                    smallvec![Atom::Var(VarId(0))],
                    conv("f32"),
                    VarId(1),
                )],
            );
            run_both(&to_f32, Value::scalar_i64(iv), &format!("i64->f32({iv})"));
        }

        // f32 source bodies (mixed-precision — the hot ML cast path).
        let f32lit = |v: f32| Atom::Lit(Literal::F32Bits(v.to_bits()));
        // f32 -> i64 (DIRECT f32 cast in convert_literal, not via f64).
        let f32_to_i64 = Jaxpr::new(
            vec![VarId(0)],
            vec![],
            vec![VarId(1)],
            vec![mk(
                Primitive::ConvertElementType,
                smallvec![Atom::Var(VarId(0))],
                conv("i64"),
                VarId(1),
            )],
        );
        // mixed-precision: x_f32 -> f64; y = x*pi + 1; -> f32  (widen, f64 math, narrow).
        let (xf, xd, m, s, of) = (VarId(0), VarId(1), VarId(2), VarId(3), VarId(4));
        let mixed_prec = Jaxpr::new(
            vec![xf],
            vec![],
            vec![of],
            vec![
                mk(
                    Primitive::ConvertElementType,
                    smallvec![Atom::Var(xf)],
                    conv("f64"),
                    xd,
                ),
                mk(
                    Primitive::Mul,
                    smallvec![Atom::Var(xd), f64lit(std::f64::consts::PI)],
                    np(),
                    m,
                ),
                mk(
                    Primitive::Add,
                    smallvec![Atom::Var(m), f64lit(1.0)],
                    np(),
                    s,
                ),
                mk(
                    Primitive::ConvertElementType,
                    smallvec![Atom::Var(s)],
                    conv("f32"),
                    of,
                ),
            ],
        );
        // f32 arith + f32 select gated behind a convert so the poly plan owns it:
        //   x_f32 -> f64 (forces poly) -> back f32 ignored; out = select(x>0, x*2, -x) in f32.
        let (xs, _xd2, gtb, dbl, neg, sel) =
            (VarId(0), VarId(1), VarId(2), VarId(3), VarId(4), VarId(5));
        let f32_cond = Jaxpr::new(
            vec![xs],
            vec![],
            vec![sel],
            vec![
                mk(
                    Primitive::ConvertElementType,
                    smallvec![Atom::Var(xs)],
                    conv("f32"),
                    _xd2,
                ),
                mk(
                    Primitive::Gt,
                    smallvec![Atom::Var(_xd2), f32lit(0.0)],
                    np(),
                    gtb,
                ),
                mk(
                    Primitive::Mul,
                    smallvec![Atom::Var(_xd2), f32lit(2.0)],
                    np(),
                    dbl,
                ),
                mk(Primitive::Neg, smallvec![Atom::Var(_xd2)], np(), neg),
                mk(
                    Primitive::Select,
                    smallvec![Atom::Var(gtb), Atom::Var(dbl), Atom::Var(neg)],
                    np(),
                    sel,
                ),
            ],
        );
        let f32_to_bool = Jaxpr::new(
            vec![VarId(0)],
            vec![],
            vec![VarId(1)],
            vec![mk(
                Primitive::ConvertElementType,
                smallvec![Atom::Var(VarId(0))],
                conv("bool"),
                VarId(1),
            )],
        );
        let f32_inputs: [f32; 12] = [
            f32::NEG_INFINITY,
            -1e18,
            -3.7,
            -1.0,
            -0.0,
            0.0,
            0.5,
            1.0,
            3.7,
            1e18,
            f32::INFINITY,
            f32::NAN,
        ];
        for &xv in &f32_inputs {
            run_both(
                &f32_to_i64,
                Value::scalar_f32(xv),
                &format!("f32->i64({xv})"),
            );
            run_both(
                &f32_to_bool,
                Value::scalar_f32(xv),
                &format!("f32->bool({xv})"),
            );
            run_both(
                &mixed_prec,
                Value::scalar_f32(xv),
                &format!("mixed_prec({xv})"),
            );
            run_both(&f32_cond, Value::scalar_f32(xv), &format!("f32_cond({xv})"));
        }

        // bf16/f16 mixed-precision convert (the hot ML training cast: half<->f32).
        for (name, half_dt) in [("bf16", "bf16"), ("f16", "f16")] {
            // half -> f64; y = x*pi + 1; -> half  (the canonical mixed-precision body).
            let (xh, xd, m, s, oh) = (VarId(0), VarId(1), VarId(2), VarId(3), VarId(4));
            let half_mixed = Jaxpr::new(
                vec![xh],
                vec![],
                vec![oh],
                vec![
                    mk(
                        Primitive::ConvertElementType,
                        smallvec![Atom::Var(xh)],
                        conv("f64"),
                        xd,
                    ),
                    mk(
                        Primitive::Mul,
                        smallvec![Atom::Var(xd), f64lit(std::f64::consts::PI)],
                        np(),
                        m,
                    ),
                    mk(
                        Primitive::Add,
                        smallvec![Atom::Var(m), f64lit(1.0)],
                        np(),
                        s,
                    ),
                    mk(
                        Primitive::ConvertElementType,
                        smallvec![Atom::Var(s)],
                        conv(half_dt),
                        oh,
                    ),
                ],
            );
            // half -> i64 (DIRECT f32-of-half cast) and half -> bool.
            let half_to_i64 = Jaxpr::new(
                vec![VarId(0)],
                vec![],
                vec![VarId(1)],
                vec![mk(
                    Primitive::ConvertElementType,
                    smallvec![Atom::Var(VarId(0))],
                    conv("i64"),
                    VarId(1),
                )],
            );
            // f64 -> half (downcast, single-rounded).
            let f64_to_half = Jaxpr::new(
                vec![VarId(0)],
                vec![],
                vec![VarId(1)],
                vec![mk(
                    Primitive::ConvertElementType,
                    smallvec![Atom::Var(VarId(0))],
                    conv(half_dt),
                    VarId(1),
                )],
            );
            let mkhalf = |v: f64| -> Value {
                let lit = if half_dt == "bf16" {
                    Literal::from_bf16_f64(v)
                } else {
                    Literal::from_f16_f64(v)
                };
                Value::Scalar(lit)
            };
            for &xv in &[
                -100.0_f64,
                -3.7,
                -1.0,
                -0.0,
                0.0,
                0.5,
                1.0,
                3.7,
                100.0,
                f64::INFINITY,
                f64::NAN,
            ] {
                run_both(&half_mixed, mkhalf(xv), &format!("{name}_mixed({xv})"));
                run_both(&half_to_i64, mkhalf(xv), &format!("{name}->i64({xv})"));
                run_both(
                    &f64_to_half,
                    Value::scalar_f64(xv),
                    &format!("f64->{name}({xv})"),
                );
            }
        }
    }

    #[test]
    fn scalar_select_i64_arena_bit_identical_to_generic() {
        // The mixed i64/bool SELECT arena must match the generic interpreter on
        // integer-conditional bodies (masked increment / integer clamp / where), and
        // be actually selected (scalar_select_i64_plan.is_some()).
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let ilit = |v: i64| Atom::Lit(Literal::I64(v));
        let val_i64 = |v: &Value| -> i64 {
            match v {
                Value::Scalar(Literal::I64(b)) => *b,
                other => panic!("expected i64 scalar, got {other:?}"),
            }
        };
        let xs = [i64::MIN, -1000, -11, -1, 0, 1, 5, 10, 11, 1000, i64::MAX];

        // masked increment: select(i < n, i+1, i), with n a fixed const.
        let (i, n, lt, inc, out) = (VarId(0), VarId(1), VarId(2), VarId(3), VarId(4));
        let masked_inc = Jaxpr::new(
            vec![i, n],
            vec![],
            vec![out],
            vec![
                mk(Primitive::Lt, smallvec![Atom::Var(i), Atom::Var(n)], lt),
                mk(Primitive::Add, smallvec![Atom::Var(i), ilit(1)], inc),
                mk(
                    Primitive::Select,
                    smallvec![Atom::Var(lt), Atom::Var(inc), Atom::Var(i)],
                    out,
                ),
            ],
        );

        // integer clamp to [0,10]: select(x>10, 10, select(x<0, 0, x)).
        let (x2, gt, lt0, inner, out2) = (VarId(0), VarId(1), VarId(2), VarId(3), VarId(4));
        let clamp = Jaxpr::new(
            vec![x2],
            vec![],
            vec![out2],
            vec![
                mk(Primitive::Gt, smallvec![Atom::Var(x2), ilit(10)], gt),
                mk(Primitive::Lt, smallvec![Atom::Var(x2), ilit(0)], lt0),
                mk(
                    Primitive::Select,
                    smallvec![Atom::Var(lt0), ilit(0), Atom::Var(x2)],
                    inner,
                ),
                mk(
                    Primitive::Select,
                    smallvec![Atom::Var(gt), ilit(10), Atom::Var(inner)],
                    out2,
                ),
            ],
        );

        for (label, body, nargs) in [("masked_inc", &masked_inc, 2usize), ("clamp", &clamp, 1)] {
            let plan = super::build_dense_plan(body).expect("dense plan");
            assert!(
                plan.scalar_select_i64_plan.is_some(),
                "{label} not routed through the i64 select arena"
            );
            assert!(
                plan.scalar_select_plan.is_none(),
                "{label} unexpectedly built an f64 select plan"
            );
            for &xv in &xs {
                let args: Vec<Value> = if nargs == 2 {
                    vec![Value::scalar_i64(xv), Value::scalar_i64(5)]
                } else {
                    vec![Value::scalar_i64(xv)]
                };
                let mut genv: Vec<Option<Value>> = vec![None; plan.slots];
                let mut gscr: Vec<Value> = Vec::new();
                let mut gout: Vec<Value> = Vec::new();
                super::run_dense_env_into(
                    body,
                    &[],
                    &args,
                    &mut genv,
                    &plan.last_use,
                    &mut gscr,
                    &mut gout,
                )
                .expect("generic i64 select");
                let mut cenv: Vec<Option<Value>> = vec![None; plan.slots];
                let mut cscr: Vec<Value> = Vec::new();
                let mut cout: Vec<Value> = Vec::new();
                let mut bufs = super::ScalarPlanBuffers::default();
                super::run_dense_plan_into(
                    body,
                    &[],
                    &args,
                    &mut cenv,
                    &plan,
                    &mut cscr,
                    &mut cout,
                    &mut bufs,
                )
                .expect("compiled i64 select");
                assert_eq!(
                    val_i64(&cout[0]),
                    val_i64(&gout[0]),
                    "{label}({xv}) i64 select arena differs from generic"
                );
            }
        }

        // select with a bool INPUT predicate and i64 branches.
        let (p, a, b, o) = (VarId(0), VarId(1), VarId(2), VarId(3));
        let where_body = Jaxpr::new(
            vec![p, a, b],
            vec![],
            vec![o],
            vec![mk(
                Primitive::Select,
                smallvec![Atom::Var(p), Atom::Var(a), Atom::Var(b)],
                o,
            )],
        );
        let plan = super::build_dense_plan(&where_body).expect("dense plan");
        assert!(plan.scalar_select_i64_plan.is_some());
        for &pred in &[true, false] {
            let args = [
                Value::scalar_bool(pred),
                Value::scalar_i64(i64::MIN),
                Value::scalar_i64(i64::MAX),
            ];
            let mut genv: Vec<Option<Value>> = vec![None; plan.slots];
            let mut gscr: Vec<Value> = Vec::new();
            let mut gout: Vec<Value> = Vec::new();
            super::run_dense_env_into(
                &where_body,
                &[],
                &args,
                &mut genv,
                &plan.last_use,
                &mut gscr,
                &mut gout,
            )
            .expect("generic i64 where");
            let mut cenv: Vec<Option<Value>> = vec![None; plan.slots];
            let mut cscr: Vec<Value> = Vec::new();
            let mut cout: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            super::run_dense_plan_into(
                &where_body,
                &[],
                &args,
                &mut cenv,
                &plan,
                &mut cscr,
                &mut cout,
                &mut bufs,
            )
            .expect("compiled i64 where");
            assert_eq!(
                val_i64(&cout[0]),
                val_i64(&gout[0]),
                "i64 where({pred}) differs"
            );
        }
    }

    #[test]
    fn scalar_arena_transcendentals_bit_identical_to_generic() {
        // Proves the scalar-f64/f32 arena handles unary transcendental/rounding ops
        // BIT-FOR-BIT identically to the generic tree-walking interpreter, and that
        // the arena is ACTUALLY selected (non-vacuous): build_dense_plan must
        // populate scalar_f64_plan/scalar_f32_plan for these bodies.
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        // Every newly-arena'd unary op. Inputs span negatives, zero, sub/super-unit,
        // and large magnitudes — including domain violations (e.g. Log/Sqrt/Acos of a
        // negative -> NaN), which must yield IDENTICAL NaN bits since both paths call
        // the same `f64::FUNC`. Bit comparison (not Value PartialEq) handles NaN.
        let ops = [
            Primitive::Exp,
            Primitive::Log,
            Primitive::Log2,
            Primitive::Exp2,
            Primitive::Expm1,
            Primitive::Log1p,
            Primitive::Sqrt,
            Primitive::Rsqrt,
            Primitive::Sin,
            Primitive::Cos,
            Primitive::Tan,
            Primitive::Asin,
            Primitive::Acos,
            Primitive::Atan,
            Primitive::Sinh,
            Primitive::Cosh,
            Primitive::Tanh,
            Primitive::Asinh,
            Primitive::Acosh,
            Primitive::Atanh,
            Primitive::Floor,
            Primitive::Ceil,
            Primitive::Trunc,
            Primitive::Deg2Rad,
            Primitive::Rad2Deg,
            Primitive::Square,
            Primitive::Reciprocal,
            Primitive::Cbrt,
            Primitive::Logistic,
            Primitive::Sign,
        ];
        let xs = [
            -100.0_f64, -5.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.3, 0.5, 0.9999, 1.0, 1.5, 2.0, 3.7,
            100.0,
        ];

        let bits_f64 = |v: &Value| -> u64 {
            match v {
                Value::Scalar(Literal::F64Bits(b)) => *b,
                other => panic!("expected f64 scalar, got {other:?}"),
            }
        };
        let bits_f32 = |v: &Value| -> u32 {
            match v {
                Value::Scalar(Literal::F32Bits(b)) => *b,
                other => panic!("expected f32 scalar, got {other:?}"),
            }
        };

        for op in ops {
            let (x, out) = (VarId(0), VarId(1));
            let body = Jaxpr::new(
                vec![x],
                vec![],
                vec![out],
                vec![mk(op, smallvec![Atom::Var(x)], out)],
            );
            let plan = super::build_dense_plan(&body).expect("dense plan");
            assert!(
                plan.scalar_f64_plan.is_some(),
                "{op:?} not routed through the scalar f64 arena"
            );
            assert!(
                plan.scalar_f32_plan.is_some(),
                "{op:?} not routed through the scalar f32 arena"
            );

            for &xv in &xs {
                // f64
                let args = [Value::scalar_f64(xv)];
                let mut genv: Vec<Option<Value>> = vec![None; plan.slots];
                let mut gscr: Vec<Value> = Vec::new();
                let mut gout: Vec<Value> = Vec::new();
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut genv,
                    &plan.last_use,
                    &mut gscr,
                    &mut gout,
                )
                .expect("generic f64");

                let mut cenv: Vec<Option<Value>> = vec![None; plan.slots];
                let mut cscr: Vec<Value> = Vec::new();
                let mut cout: Vec<Value> = Vec::new();
                let mut bufs = super::ScalarPlanBuffers::default();
                super::run_dense_plan_into(
                    &body,
                    &[],
                    &args,
                    &mut cenv,
                    &plan,
                    &mut cscr,
                    &mut cout,
                    &mut bufs,
                )
                .expect("compiled f64");
                assert_eq!(
                    bits_f64(&cout[0]),
                    bits_f64(&gout[0]),
                    "{op:?}({xv}) f64 arena bits differ from generic"
                );

                // f32 (JAX default float): widen→f64→narrow contract.
                let args32 = [Value::Scalar(Literal::from_f32(xv as f32))];
                let mut g32: Vec<Option<Value>> = vec![None; plan.slots];
                let mut gs32: Vec<Value> = Vec::new();
                let mut go32: Vec<Value> = Vec::new();
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args32,
                    &mut g32,
                    &plan.last_use,
                    &mut gs32,
                    &mut go32,
                )
                .expect("generic f32");
                let mut c32: Vec<Option<Value>> = vec![None; plan.slots];
                let mut cs32: Vec<Value> = Vec::new();
                let mut co32: Vec<Value> = Vec::new();
                let mut bufs32 = super::ScalarPlanBuffers::default();
                super::run_dense_plan_into(
                    &body,
                    &[],
                    &args32,
                    &mut c32,
                    &plan,
                    &mut cs32,
                    &mut co32,
                    &mut bufs32,
                )
                .expect("compiled f32");
                assert_eq!(
                    bits_f32(&co32[0]),
                    bits_f32(&go32[0]),
                    "{op:?}({xv}) f32 arena bits differ from generic"
                );
            }
        }

        // IntegerPow (param-carrying: `x ** const`). Exponent param must be parsed
        // and applied as `lhs.powi(e)`, bit-identical to fj-lax eval_integer_pow's
        // float arm. Covers negative (1/x^n), zero (->1), and positive exponents.
        for exponent in [-3_i32, -1, 0, 1, 2, 3, 5] {
            let (x, out) = (VarId(0), VarId(1));
            let mut params = BTreeMap::new();
            params.insert("exponent".to_owned(), exponent.to_string());
            let body = Jaxpr::new(
                vec![x],
                vec![],
                vec![out],
                vec![Equation {
                    primitive: Primitive::IntegerPow,
                    inputs: smallvec![Atom::Var(x)],
                    outputs: smallvec![out],
                    params,
                    sub_jaxprs: vec![],
                    effects: vec![],
                }],
            );
            let plan = super::build_dense_plan(&body).expect("dense plan");
            assert!(
                plan.scalar_f64_plan.is_some() && plan.scalar_f32_plan.is_some(),
                "IntegerPow[{exponent}] not routed through the scalar arena"
            );
            for &xv in &xs {
                let args = [Value::scalar_f64(xv)];
                let mut genv: Vec<Option<Value>> = vec![None; plan.slots];
                let mut gscr: Vec<Value> = Vec::new();
                let mut gout: Vec<Value> = Vec::new();
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut genv,
                    &plan.last_use,
                    &mut gscr,
                    &mut gout,
                )
                .expect("generic integer_pow");
                let mut cenv: Vec<Option<Value>> = vec![None; plan.slots];
                let mut cscr: Vec<Value> = Vec::new();
                let mut cout: Vec<Value> = Vec::new();
                let mut bufs = super::ScalarPlanBuffers::default();
                super::run_dense_plan_into(
                    &body,
                    &[],
                    &args,
                    &mut cenv,
                    &plan,
                    &mut cscr,
                    &mut cout,
                    &mut bufs,
                )
                .expect("compiled integer_pow");
                assert_eq!(
                    bits_f64(&cout[0]),
                    bits_f64(&gout[0]),
                    "IntegerPow[{exponent}]({xv}) f64 arena bits differ from generic"
                );
            }
        }

        // Binary transcendentals Pow/Atan2: out = x BINOP y, swept over operand pairs
        // (incl. domain edges: negative base ^ fractional exp -> NaN; atan2(0,0) -> 0).
        for op in [Primitive::Pow, Primitive::Atan2] {
            let (a, b, out) = (VarId(0), VarId(1), VarId(2));
            let body = Jaxpr::new(
                vec![a, b],
                vec![],
                vec![out],
                vec![mk(op, smallvec![Atom::Var(a), Atom::Var(b)], out)],
            );
            let plan = super::build_dense_plan(&body).expect("dense plan");
            assert!(
                plan.scalar_f64_plan.is_some() && plan.scalar_f32_plan.is_some(),
                "{op:?} not routed through the scalar arena"
            );
            for &av in &xs {
                for &bv in &[-2.5_f64, -1.0, -0.0, 0.0, 0.5, 1.0, 2.0, 3.3] {
                    let args = [Value::scalar_f64(av), Value::scalar_f64(bv)];
                    let mut genv: Vec<Option<Value>> = vec![None; plan.slots];
                    let mut gscr: Vec<Value> = Vec::new();
                    let mut gout: Vec<Value> = Vec::new();
                    super::run_dense_env_into(
                        &body,
                        &[],
                        &args,
                        &mut genv,
                        &plan.last_use,
                        &mut gscr,
                        &mut gout,
                    )
                    .expect("generic binary");
                    let mut cenv: Vec<Option<Value>> = vec![None; plan.slots];
                    let mut cscr: Vec<Value> = Vec::new();
                    let mut cout: Vec<Value> = Vec::new();
                    let mut bufs = super::ScalarPlanBuffers::default();
                    super::run_dense_plan_into(
                        &body,
                        &[],
                        &args,
                        &mut cenv,
                        &plan,
                        &mut cscr,
                        &mut cout,
                        &mut bufs,
                    )
                    .expect("compiled binary");
                    assert_eq!(
                        bits_f64(&cout[0]),
                        bits_f64(&gout[0]),
                        "{op:?}({av},{bv}) f64 arena bits differ from generic"
                    );
                }
            }
        }

        // Multi-op activation body (sigmoid·tanh blend) to prove chaining through the
        // arena stays bit-identical: out = (exp(x)/(1+exp(x)))·tanh(x) + sqrt(1+exp(x)) − log(1+exp(x)).
        let (x, e, d, s, t, q, lg, m, a8, out) = (
            VarId(0),
            VarId(1),
            VarId(2),
            VarId(3),
            VarId(4),
            VarId(5),
            VarId(6),
            VarId(7),
            VarId(8),
            VarId(9),
        );
        let one = Atom::Lit(Literal::from_f64(1.0));
        let body = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                mk(Primitive::Exp, smallvec![Atom::Var(x)], e),
                mk(Primitive::Add, smallvec![Atom::Var(e), one], d),
                mk(Primitive::Div, smallvec![Atom::Var(e), Atom::Var(d)], s),
                mk(Primitive::Tanh, smallvec![Atom::Var(x)], t),
                mk(Primitive::Sqrt, smallvec![Atom::Var(d)], q),
                mk(Primitive::Log, smallvec![Atom::Var(d)], lg),
                mk(Primitive::Mul, smallvec![Atom::Var(s), Atom::Var(t)], m),
                mk(Primitive::Add, smallvec![Atom::Var(m), Atom::Var(q)], a8),
                mk(Primitive::Sub, smallvec![Atom::Var(a8), Atom::Var(lg)], out),
            ],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        assert!(plan.scalar_f64_plan.is_some(), "chained body not arena'd");
        for &xv in &xs {
            let args = [Value::scalar_f64(xv)];
            let mut genv: Vec<Option<Value>> = vec![None; plan.slots];
            let mut gscr: Vec<Value> = Vec::new();
            let mut gout: Vec<Value> = Vec::new();
            super::run_dense_env_into(
                &body,
                &[],
                &args,
                &mut genv,
                &plan.last_use,
                &mut gscr,
                &mut gout,
            )
            .expect("generic chain");
            let mut cenv: Vec<Option<Value>> = vec![None; plan.slots];
            let mut cscr: Vec<Value> = Vec::new();
            let mut cout: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            super::run_dense_plan_into(
                &body,
                &[],
                &args,
                &mut cenv,
                &plan,
                &mut cscr,
                &mut cout,
                &mut bufs,
            )
            .expect("compiled chain");
            assert_eq!(
                bits_f64(&cout[0]),
                bits_f64(&gout[0]),
                "chained activation body bits differ at x={xv}"
            );
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_dense_i64_tensor_arena() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let ilit = |v: i64| Atom::Lit(Literal::I64(v));
        // [64] i64 carry-style body: 4 elementwise ops (x*3 + 1; max 0; min 1000).
        let (x, m, a, r, out) = (VarId(0), VarId(1), VarId(2), VarId(3), VarId(4));
        let body = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                mk(Primitive::Mul, smallvec![Atom::Var(x), ilit(3)], m),
                mk(Primitive::Add, smallvec![Atom::Var(m), ilit(1)], a),
                mk(Primitive::Max, smallvec![Atom::Var(a), ilit(0)], r),
                mk(Primitive::Min, smallvec![Atom::Var(r), ilit(1000)], out),
            ],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        let p = plan.scalar_i64_plan.as_ref().expect("i64 plan");
        let data = vec![5i64; 64];
        let arg =
            Value::Tensor(TensorValue::new_i64_values(Shape { dims: vec![64] }, data).unwrap());
        let n: usize = 1_000_000;
        let args = [arg];

        let t_generic = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut acc = 0i64;
            for _ in 0..n {
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan.last_use,
                    &mut scratch,
                    &mut o,
                )
                .expect("generic");
                if let Value::Tensor(t) = &o[0] {
                    acc = acc.wrapping_add(t.elements.as_i64_slice().unwrap()[0]);
                }
            }
            std::hint::black_box(acc);
        });
        let t_arena = best_time(|| {
            let mut o: Vec<Value> = Vec::new();
            let mut cells: Vec<super::DenseI64Cell> = Vec::new();
            let mut acc = 0i64;
            for _ in 0..n {
                super::run_scalar_i64_plan_as_tensor_into(p, &[], &args, &mut o, &mut cells, true)
                    .expect("handled")
                    .expect("ok");
                if let Value::Tensor(t) = &o[0] {
                    acc = acc.wrapping_add(t.elements.as_i64_slice().unwrap()[0]);
                }
            }
            std::hint::black_box(acc);
        });
        println!(
            "BENCH dense-i64 tensor arena [64] {n} evals (4 elementwise ops): GENERIC {:.1}ns/eval -> ARENA {:.1}ns/eval = {:.2}x",
            t_generic * 1e9 / n as f64,
            t_arena * 1e9 / n as f64,
            t_generic / t_arena,
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_dense_f32_tensor_arena() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let f32lit = |v: f32| Atom::Lit(Literal::F32Bits(v.to_bits()));
        // [64] f32 RNN-carry-style body: 4 elementwise ops (x*a + b; max 0; *c).
        let (x, m, a, r, out) = (VarId(0), VarId(1), VarId(2), VarId(3), VarId(4));
        let body = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                mk(Primitive::Mul, smallvec![Atom::Var(x), f32lit(0.9)], m),
                mk(Primitive::Add, smallvec![Atom::Var(m), f32lit(0.1)], a),
                mk(Primitive::Max, smallvec![Atom::Var(a), f32lit(0.0)], r),
                mk(Primitive::Mul, smallvec![Atom::Var(r), f32lit(1.01)], out),
            ],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        let p = plan.scalar_f32_plan.as_ref().expect("f32 plan");
        let data = vec![0.5f32; 64];
        let arg =
            Value::Tensor(TensorValue::new_f32_values(Shape { dims: vec![64] }, data).unwrap());
        let n: usize = 1_000_000;
        let args = [arg];

        let t_generic = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut acc = 0.0f32;
            for _ in 0..n {
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan.last_use,
                    &mut scratch,
                    &mut o,
                )
                .expect("generic");
                if let Value::Tensor(t) = &o[0] {
                    acc += t.elements.as_f32_slice().unwrap()[0];
                }
            }
            std::hint::black_box(acc);
        });
        let t_arena = best_time(|| {
            let mut o: Vec<Value> = Vec::new();
            let mut cells: Vec<super::DenseF32Cell> = Vec::new();
            let mut acc = 0.0f32;
            for _ in 0..n {
                super::run_scalar_f32_plan_as_tensor_into(p, &[], &args, &mut o, &mut cells, true)
                    .expect("handled")
                    .expect("ok");
                if let Value::Tensor(t) = &o[0] {
                    acc += t.elements.as_f32_slice().unwrap()[0];
                }
            }
            std::hint::black_box(acc);
        });
        println!(
            "BENCH dense-f32 tensor arena [64] {n} evals (4 elementwise ops): GENERIC {:.1}ns/eval -> ARENA {:.1}ns/eval = {:.2}x",
            t_generic * 1e9 / n as f64,
            t_arena * 1e9 / n as f64,
            t_generic / t_arena,
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_dense_f64_tensor_arena_broadcast() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        // Realistic body: ONE broadcast (bias add) among same-shape elementwise ops,
        //   out = min(max((x[8,8] + bias[8]) * 2 - 1, 0), 10)   (1 bcast + 4 same-shape).
        let lit = |v: f64| Atom::Lit(Literal::from_f64(v));
        let (x, b, s1, s2, s3, r, out) = (
            VarId(0),
            VarId(1),
            VarId(2),
            VarId(3),
            VarId(4),
            VarId(5),
            VarId(6),
        );
        let body = Jaxpr::new(
            vec![x, b],
            vec![],
            vec![out],
            vec![
                mk(Primitive::Add, smallvec![Atom::Var(x), Atom::Var(b)], s1),
                mk(Primitive::Mul, smallvec![Atom::Var(s1), lit(2.0)], s2),
                mk(Primitive::Sub, smallvec![Atom::Var(s2), lit(1.0)], s3),
                mk(Primitive::Max, smallvec![Atom::Var(s3), lit(0.0)], r),
                mk(Primitive::Min, smallvec![Atom::Var(r), lit(10.0)], out),
            ],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        let p = plan.scalar_f64_plan.as_ref().expect("f64 plan");
        let xv = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: vec![8, 8] }, vec![0.5f64; 64]).unwrap(),
        );
        let bias = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: vec![8] }, vec![0.1; 8]).unwrap(),
        );
        let n: usize = 1_000_000;
        let args = [xv, bias];

        let t_generic = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan.last_use,
                    &mut scratch,
                    &mut o,
                )
                .expect("generic");
                if let Value::Tensor(t) = &o[0] {
                    acc += t.elements.as_f64_slice().unwrap()[0];
                }
            }
            std::hint::black_box(acc);
        });
        let t_arena = best_time(|| {
            let mut o: Vec<Value> = Vec::new();
            let mut cells: Vec<super::DenseF64Cell> = Vec::new();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_scalar_f64_plan_as_tensor_into(p, &[], &args, &mut o, &mut cells, true)
                    .expect("handled")
                    .expect("ok");
                if let Value::Tensor(t) = &o[0] {
                    acc += t.elements.as_f64_slice().unwrap()[0];
                }
            }
            std::hint::black_box(acc);
        });
        println!(
            "BENCH dense-f64 tensor arena BROADCAST [8,8]+bias[8] (1 bcast + 4 same-shape) {n} evals: GENERIC {:.1}ns/eval -> ARENA {:.1}ns/eval = {:.2}x",
            t_generic * 1e9 / n as f64,
            t_arena * 1e9 / n as f64,
            t_generic / t_arena,
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_dense_f64_tensor_arena() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let f64lit = |v: f64| Atom::Lit(Literal::from_f64(v));
        // Small-tensor [64] RNN-carry-style body: 4 elementwise ops (x*a + b; max 0; *c).
        let (x, m, a, r, out) = (VarId(0), VarId(1), VarId(2), VarId(3), VarId(4));
        let body = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                mk(Primitive::Mul, smallvec![Atom::Var(x), f64lit(0.9)], m),
                mk(Primitive::Add, smallvec![Atom::Var(m), f64lit(0.1)], a),
                mk(Primitive::Max, smallvec![Atom::Var(a), f64lit(0.0)], r),
                mk(Primitive::Mul, smallvec![Atom::Var(r), f64lit(1.01)], out),
            ],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        let p = plan.scalar_f64_plan.as_ref().expect("f64 plan");
        let data = vec![0.5f64; 64];
        let arg =
            Value::Tensor(TensorValue::new_f64_values(Shape { dims: vec![64] }, data).unwrap());
        let n: usize = 1_000_000;
        let args = [arg];

        let t_generic = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan.last_use,
                    &mut scratch,
                    &mut o,
                )
                .expect("generic");
                if let Value::Tensor(t) = &o[0] {
                    acc += t.elements.as_f64_slice().unwrap()[0];
                }
            }
            std::hint::black_box(acc);
        });
        let t_arena = best_time(|| {
            let mut o: Vec<Value> = Vec::new();
            let mut cells: Vec<super::DenseF64Cell> = Vec::new();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_scalar_f64_plan_as_tensor_into(p, &[], &args, &mut o, &mut cells, true)
                    .expect("handled")
                    .expect("ok");
                if let Value::Tensor(t) = &o[0] {
                    acc += t.elements.as_f64_slice().unwrap()[0];
                }
            }
            std::hint::black_box(acc);
        });
        println!(
            "BENCH dense-f64 tensor arena [64] {n} evals (4 elementwise ops): GENERIC {:.1}ns/eval -> ARENA {:.1}ns/eval = {:.2}x",
            t_generic * 1e9 / n as f64,
            t_arena * 1e9 / n as f64,
            t_generic / t_arena,
        );
    }

    #[test]
    fn dense_f64_reduce_sum_plan_matches_generic_sha256() {
        let (x, out) = (VarId(0), VarId(1));
        let body = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec![Atom::Var(x)],
                outputs: smallvec![out],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        assert!(
            plan.dense_f64_reduce_sum_plan.is_some(),
            "reduce_sum body should compile to the dense-f64 plan"
        );

        let data: Vec<f64> = (0..64)
            .map(|i| {
                if i == 7 {
                    -0.0
                } else {
                    (i as f64) * 0.125 - 3.5
                }
            })
            .collect();
        let manual = data.iter().fold(0.0_f64, |acc, &value| acc + value);
        let arg =
            Value::Tensor(TensorValue::new_f64_values(Shape { dims: vec![64] }, data).unwrap());
        let args = [arg];

        let mut generic_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut generic_scratch: Vec<Value> = Vec::new();
        let mut generic_out: Vec<Value> = Vec::new();
        super::run_dense_env_into(
            &body,
            &[],
            &args,
            &mut generic_env,
            &plan.last_use,
            &mut generic_scratch,
            &mut generic_out,
        )
        .expect("generic reduce_sum");

        let mut planned_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut planned_scratch: Vec<Value> = Vec::new();
        let mut planned_out: Vec<Value> = Vec::new();
        let mut bufs = super::ScalarPlanBuffers::default();
        super::run_dense_plan_into(
            &body,
            &[],
            &args,
            &mut planned_env,
            &plan,
            &mut planned_scratch,
            &mut planned_out,
            &mut bufs,
        )
        .expect("planned reduce_sum");

        assert_eq!(planned_out, generic_out);
        let Value::Scalar(Literal::F64Bits(bits)) = &planned_out[0] else {
            panic!("dense reduce_sum must return f64 scalar");
        };
        assert_eq!(*bits, manual.to_bits());
        let sha256 =
            fj_test_utils::fixture_id_from_json(&("frankenjax-0x3pu-reduce-sum", &planned_out))
                .expect("golden digest");
        assert_eq!(
            sha256,
            "561117f6fd0383063821dfcf3074491ba1e5f3943a828629e080c4fa7897c8cd"
        );
    }

    #[test]
    fn dense_f32_reduce_sum_plan_matches_generic_sha256() {
        // F32 (JAX's default float dtype) full reduce_sum must take the dense
        // plan and produce BIT-IDENTICAL output to the generic interpreter
        // (which routes eval_primitive(ReduceSum) -> fj-lax, the true oracle):
        // accumulate in f64, round the final accumulator to f32.
        let (x, out) = (VarId(0), VarId(1));
        let body = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec![Atom::Var(x)],
                outputs: smallvec![out],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        assert!(
            plan.dense_f64_reduce_sum_plan.is_some(),
            "reduce_sum body should compile to the dense reduce-sum plan"
        );

        // Mix of magnitudes + a -0.0 so the f64-accumulate/round-to-f32 path is
        // exercised (naive f32 accumulation would diverge from the oracle).
        let data: Vec<f32> = (0..257)
            .map(|i| {
                if i == 9 {
                    -0.0
                } else {
                    (i as f32) * 0.1 - 12.5 + (i as f32) * 1.0e6
                }
            })
            .collect();
        let arg =
            Value::Tensor(TensorValue::new_f32_values(Shape { dims: vec![257] }, data).unwrap());
        let args = [arg];

        let mut generic_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut generic_scratch: Vec<Value> = Vec::new();
        let mut generic_out: Vec<Value> = Vec::new();
        super::run_dense_env_into(
            &body,
            &[],
            &args,
            &mut generic_env,
            &plan.last_use,
            &mut generic_scratch,
            &mut generic_out,
        )
        .expect("generic reduce_sum");

        let mut planned_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut planned_scratch: Vec<Value> = Vec::new();
        let mut planned_out: Vec<Value> = Vec::new();
        let mut bufs = super::ScalarPlanBuffers::default();
        super::run_dense_plan_into(
            &body,
            &[],
            &args,
            &mut planned_env,
            &plan,
            &mut planned_scratch,
            &mut planned_out,
            &mut bufs,
        )
        .expect("planned reduce_sum");

        assert_eq!(
            planned_out, generic_out,
            "f32 plan must match fj-lax oracle"
        );
        assert!(
            matches!(&planned_out[0], Value::Scalar(Literal::F32Bits(_))),
            "f32 reduce_sum must return an f32 scalar, got {:?}",
            planned_out[0]
        );
        let sha256 =
            fj_test_utils::fixture_id_from_json(&("frankenjax-0x3pu-reduce-sum-f32", &planned_out))
                .expect("golden digest");
        assert_eq!(
            sha256, "13863282368fc5ed093b41839f1ce31488de738b630ea96a904d028197023ba7",
            "f32 reduce_sum golden"
        );
    }

    #[test]
    fn dense_reduce_prod_max_min_plans_match_generic() {
        // Prod/Max/Min full reductions over f64 AND f32 must take the dense plan
        // and produce BIT-IDENTICAL output to the generic interpreter (which
        // routes eval_primitive -> fj-lax, the production oracle), including the
        // ±0 / NaN / mixed-magnitude patterns that distinguish a naive fold.
        let (x, out) = (VarId(0), VarId(1));
        let f64_data: Vec<f64> = (0..130)
            .map(|i| match i {
                7 => -0.0,
                40 => f64::NAN,
                _ => (i as f64) * 0.37 - 24.0,
            })
            .collect();
        let f32_data: Vec<f32> = (0..130)
            .map(|i| match i {
                7 => -0.0,
                40 => f32::NAN,
                _ => (i as f32) * 0.37 - 24.0,
            })
            .collect();

        for primitive in [
            Primitive::ReduceProd,
            Primitive::ReduceMax,
            Primitive::ReduceMin,
        ] {
            let body = Jaxpr::new(
                vec![x],
                vec![],
                vec![out],
                vec![Equation {
                    primitive,
                    inputs: smallvec![Atom::Var(x)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                }],
            );
            let plan = super::build_dense_plan(&body).expect("dense plan");
            assert!(
                plan.dense_f64_reduce_sum_plan.is_some(),
                "{primitive:?} should compile to the dense reduce plan"
            );

            for arg in [
                Value::Tensor(
                    TensorValue::new_f64_values(Shape { dims: vec![130] }, f64_data.clone())
                        .unwrap(),
                ),
                Value::Tensor(
                    TensorValue::new_f32_values(Shape { dims: vec![130] }, f32_data.clone())
                        .unwrap(),
                ),
            ] {
                let args = [arg];
                let mut genv: Vec<Option<Value>> = vec![None; plan.slots];
                let mut gscratch: Vec<Value> = Vec::new();
                let mut gout: Vec<Value> = Vec::new();
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut genv,
                    &plan.last_use,
                    &mut gscratch,
                    &mut gout,
                )
                .expect("generic reduce");

                let mut penv: Vec<Option<Value>> = vec![None; plan.slots];
                let mut pscratch: Vec<Value> = Vec::new();
                let mut pout: Vec<Value> = Vec::new();
                let mut bufs = super::ScalarPlanBuffers::default();
                super::run_dense_plan_into(
                    &body,
                    &[],
                    &args,
                    &mut penv,
                    &plan,
                    &mut pscratch,
                    &mut pout,
                    &mut bufs,
                )
                .expect("planned reduce");

                // Compare bit-for-bit (NaN payloads included) via the scalar bits.
                let pbits = scalar_real_bits(&pout[0]);
                let gbits = scalar_real_bits(&gout[0]);
                assert_eq!(
                    pbits, gbits,
                    "{primitive:?} dense plan diverged from generic oracle"
                );
            }
        }
    }

    // Raw bit pattern of a real scalar (f64 or f32), for NaN-payload-exact compare.
    fn scalar_real_bits(value: &Value) -> u64 {
        match value {
            Value::Scalar(Literal::F64Bits(b)) => *b,
            Value::Scalar(Literal::F32Bits(b)) => u64::from(*b),
            other => panic!("expected real scalar, got {other:?}"),
        }
    }

    // Storage-AGNOSTIC (dtype, native-width per-element bits) of a real tensor,
    // NaN-payload-exact. Iterating `.elements` yields Literals regardless of dense
    // vs boxed backing, so a dense-F32 and a boxed-F32 tensor with equal values
    // compare equal (the typed plan returns dense; the eager path may return
    // boxed). Comparing the (dtype, bits) tuple catches any real divergence.
    fn tensor_dtype_bits(value: &Value) -> (DType, Vec<u64>) {
        match value {
            Value::Tensor(t) => {
                let bits = t
                    .elements
                    .iter()
                    .map(|l| match l {
                        Literal::F64Bits(b) => *b,
                        Literal::F32Bits(b) => u64::from(*b),
                        Literal::BF16Bits(b) | Literal::F16Bits(b) => u64::from(*b),
                        Literal::I64(n) => *n as u64,
                        other => panic!("expected real/int element, got {other:?}"),
                    })
                    .collect();
                (t.dtype, bits)
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn dense_axis_reduce_plan_matches_generic_and_golden() {
        // Single trailing-axis reductions (sum/prod/max/min) over dense f64 AND
        // f32 tensors of rank 2 and 3 must take the typed plan and be value-equal
        // (dtype + native-width bits) to the generic interpreter (eval_primitive
        // -> fj-lax, the oracle), including -0.0 and NaN. The plan returns dense
        // tensors while the eager path may return boxed — same logical value, so
        // the comparison is storage-agnostic. A golden freezes the f64 sum case.
        let (x, out) = (VarId(0), VarId(1));
        let mut golden: Option<Vec<u64>> = None;
        for (shape, axes) in [
            (vec![6u32, 8u32], vec![1usize]),
            (vec![3u32, 4u32, 5u32], vec![2usize]),
            (vec![3u32, 4u32, 5u32], vec![1usize, 2usize]),
        ] {
            let n: usize = shape.iter().map(|&d| d as usize).product();
            let f64v: Vec<f64> = (0..n)
                .map(|i| match i {
                    2 => -0.0,
                    11 => f64::NAN,
                    _ => (i as f64) * 0.07 - 1.3,
                })
                .collect();
            let f32v: Vec<f32> = f64v.iter().map(|&v| v as f32).collect();
            for primitive in [
                Primitive::ReduceSum,
                Primitive::ReduceProd,
                Primitive::ReduceMax,
                Primitive::ReduceMin,
            ] {
                let body = Jaxpr::new(
                    vec![x],
                    vec![],
                    vec![out],
                    vec![Equation {
                        primitive,
                        inputs: smallvec![Atom::Var(x)],
                        outputs: smallvec![out],
                        params: BTreeMap::from([(
                            "axes".to_owned(),
                            axes.iter()
                                .map(|a| a.to_string())
                                .collect::<Vec<_>>()
                                .join(","),
                        )]),
                        sub_jaxprs: vec![],
                        effects: vec![],
                    }],
                );
                let plan = super::build_dense_plan(&body).expect("dense plan");
                assert!(
                    plan.dense_axis_reduce_plan.is_some(),
                    "{primitive:?} trailing-axis should compile to the dense axis-reduce plan"
                );

                let f64_arg = Value::Tensor(
                    TensorValue::new_f64_values(
                        Shape {
                            dims: shape.clone(),
                        },
                        f64v.clone(),
                    )
                    .unwrap(),
                );
                let f32_arg = Value::Tensor(
                    TensorValue::new_f32_values(
                        Shape {
                            dims: shape.clone(),
                        },
                        f32v.clone(),
                    )
                    .unwrap(),
                );
                for arg in [f64_arg, f32_arg] {
                    let is_f64 = matches!(&arg, Value::Tensor(t) if t.dtype == DType::F64);
                    let args = [arg];
                    let mut genv: Vec<Option<Value>> = vec![None; plan.slots];
                    let mut gscratch: Vec<Value> = Vec::new();
                    let mut gout: Vec<Value> = Vec::new();
                    super::run_dense_env_into(
                        &body,
                        &[],
                        &args,
                        &mut genv,
                        &plan.last_use,
                        &mut gscratch,
                        &mut gout,
                    )
                    .expect("generic reduce");

                    let mut penv: Vec<Option<Value>> = vec![None; plan.slots];
                    let mut pscratch: Vec<Value> = Vec::new();
                    let mut pout: Vec<Value> = Vec::new();
                    let mut bufs = super::ScalarPlanBuffers::default();
                    super::run_dense_plan_into(
                        &body,
                        &[],
                        &args,
                        &mut penv,
                        &plan,
                        &mut pscratch,
                        &mut pout,
                        &mut bufs,
                    )
                    .expect("planned reduce");

                    assert_eq!(
                        tensor_dtype_bits(&pout[0]),
                        tensor_dtype_bits(&gout[0]),
                        "{primitive:?} axes={axes:?} f64={is_f64} plan diverged from oracle"
                    );
                    if primitive == Primitive::ReduceSum && is_f64 && shape.len() == 2 {
                        golden = Some(tensor_dtype_bits(&pout[0]).1);
                    }
                }
            }
        }
        let sha = fj_test_utils::fixture_id_from_json(&(
            "frankenjax-zmdg5-axis-reduce",
            &golden.expect("golden case ran"),
        ))
        .expect("digest");
        assert_eq!(
            sha,
            "1bcf973d9732ae43f0400652c2559be5b187f4b9560dbf28fb926d295f8647d7"
        );
    }

    #[test]
    fn dense_arg_extremum_plan_matches_generic_and_golden() {
        // Trailing-axis argmax/argmin over dense f64/f32/i64 must take the typed
        // plan and be value-equal (dtype + bits) to the generic interpreter
        // (eval_primitive -> fj-lax). Data includes ties (first-occurrence wins),
        // -0.0, and NaN (first-NaN wins) — the parity-sensitive reducer cases.
        // Tested with default axis (no param) and explicit axis=rank-1.
        let (x, out) = (VarId(0), VarId(1));
        // [4, 5]: row 0 has a tie at the max; another row a NaN.
        let shape = vec![4u32, 5u32];
        let n = 20usize;
        let f64v: Vec<f64> = vec![
            1.0,
            3.0,
            3.0,
            2.0,
            -0.0, // tie at idx 1,2 (max 3.0) -> argmax 1
            -1.0,
            -1.0,
            0.0,
            5.0,
            4.0, // argmax 3
            f64::NAN,
            2.0,
            9.0,
            1.0,
            0.0, // NaN at idx 0 -> argmax 0 (first NaN)
            7.0,
            7.0,
            7.0,
            7.0,
            7.0, // all-tie -> argmax 0 / argmin 0
        ];
        assert_eq!(f64v.len(), n);
        let mut golden: Option<Vec<u64>> = None;
        for find_max in [true, false] {
            let prim = if find_max {
                Primitive::Argmax
            } else {
                Primitive::Argmin
            };
            for axis_param in [None, Some(1usize)] {
                let params = match axis_param {
                    Some(a) => BTreeMap::from([("axis".to_owned(), a.to_string())]),
                    None => BTreeMap::new(),
                };
                let body = Jaxpr::new(
                    vec![x],
                    vec![],
                    vec![out],
                    vec![Equation {
                        primitive: prim,
                        inputs: smallvec![Atom::Var(x)],
                        outputs: smallvec![out],
                        params,
                        sub_jaxprs: vec![],
                        effects: vec![],
                    }],
                );
                let plan = super::build_dense_plan(&body).expect("dense plan");
                assert!(
                    plan.dense_arg_extremum_plan.is_some(),
                    "{prim:?} axis={axis_param:?} should compile to the dense arg-extremum plan"
                );

                let args_list = [
                    Value::Tensor(
                        TensorValue::new_f64_values(
                            Shape {
                                dims: shape.clone(),
                            },
                            f64v.clone(),
                        )
                        .unwrap(),
                    ),
                    Value::Tensor(
                        TensorValue::new_f32_values(
                            Shape {
                                dims: shape.clone(),
                            },
                            f64v.iter().map(|&x| x as f32).collect(),
                        )
                        .unwrap(),
                    ),
                    Value::Tensor(
                        TensorValue::new_i64_values(
                            Shape {
                                dims: shape.clone(),
                            },
                            // distinct-ish ints with a tie per row
                            vec![1, 3, 3, 2, 0, -1, -1, 0, 5, 4, 8, 2, 9, 1, 0, 7, 7, 7, 7, 7],
                        )
                        .unwrap(),
                    ),
                    Value::Tensor(
                        TensorValue::new_half_float_values(
                            DType::BF16,
                            Shape {
                                dims: shape.clone(),
                            },
                            f64v.iter()
                                .map(|&x| match Literal::from_bf16_f64(x) {
                                    Literal::BF16Bits(b) => b,
                                    _ => unreachable!(),
                                })
                                .collect(),
                        )
                        .unwrap(),
                    ),
                    Value::Tensor(
                        TensorValue::new_half_float_values(
                            DType::F16,
                            Shape {
                                dims: shape.clone(),
                            },
                            f64v.iter()
                                .map(|&x| match Literal::from_f16_f64(x) {
                                    Literal::F16Bits(b) => b,
                                    _ => unreachable!(),
                                })
                                .collect(),
                        )
                        .unwrap(),
                    ),
                ];
                for arg in args_list {
                    let is_f64 = matches!(&arg, Value::Tensor(t) if t.dtype == DType::F64);
                    let args = [arg];
                    let mut genv: Vec<Option<Value>> = vec![None; plan.slots];
                    let mut gscratch: Vec<Value> = Vec::new();
                    let mut gout: Vec<Value> = Vec::new();
                    super::run_dense_env_into(
                        &body,
                        &[],
                        &args,
                        &mut genv,
                        &plan.last_use,
                        &mut gscratch,
                        &mut gout,
                    )
                    .expect("generic arg");

                    let mut penv: Vec<Option<Value>> = vec![None; plan.slots];
                    let mut pscratch: Vec<Value> = Vec::new();
                    let mut pout: Vec<Value> = Vec::new();
                    let mut bufs = super::ScalarPlanBuffers::default();
                    super::run_dense_plan_into(
                        &body,
                        &[],
                        &args,
                        &mut penv,
                        &plan,
                        &mut pscratch,
                        &mut pout,
                        &mut bufs,
                    )
                    .expect("planned arg");

                    assert_eq!(
                        tensor_dtype_bits(&pout[0]),
                        tensor_dtype_bits(&gout[0]),
                        "{prim:?} axis={axis_param:?} f64={is_f64} plan diverged from oracle"
                    );
                    if find_max && axis_param.is_none() && is_f64 {
                        golden = Some(tensor_dtype_bits(&pout[0]).1);
                    }
                }
            }
        }
        let sha = fj_test_utils::fixture_id_from_json(&(
            "frankenjax-argmax",
            &golden.expect("golden ran"),
        ))
        .expect("digest");
        assert_eq!(
            sha,
            "dc014a70f31d3199aad026dadbdcba86776afde194471136262869505b7bc876"
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_dense_arg_extremum_plan_overhead() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        let (x, out) = (VarId(0), VarId(1));
        let body = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![Equation {
                primitive: Primitive::Argmax,
                inputs: smallvec![Atom::Var(x)],
                outputs: smallvec![out],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        // [8, 16] f32 logits, argmax over last axis -> [8].
        let data: Vec<f32> = (0..8 * 16).map(|i| ((i * 7) % 16) as f32 * 0.1).collect();
        let args = [Value::Tensor(
            TensorValue::new_f32_values(Shape { dims: vec![8, 16] }, data).unwrap(),
        )];
        let n: usize = 500_000;
        let t_generic = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut acc = 0i64;
            for _ in 0..n {
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan.last_use,
                    &mut scratch,
                    &mut o,
                )
                .expect("generic");
                acc = acc.wrapping_add(tensor_dtype_bits(&o[0]).1[0] as i64);
            }
            std::hint::black_box(acc);
        });
        let t_planned = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            let mut acc = 0i64;
            for _ in 0..n {
                super::run_dense_plan_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan,
                    &mut scratch,
                    &mut o,
                    &mut bufs,
                )
                .expect("planned");
                acc = acc.wrapping_add(tensor_dtype_bits(&o[0]).1[0] as i64);
            }
            std::hint::black_box(acc);
        });
        println!(
            "BENCH dense-f32 argmax [8,16] axis-1 {n} evals: GENERIC {:.1}ns/eval -> PLANNED {:.1}ns/eval = {:.2}x",
            t_generic * 1e9 / n as f64,
            t_planned * 1e9 / n as f64,
            t_generic / t_planned,
        );
    }

    #[test]
    fn dense_gather_plan_matches_generic_and_golden() {
        // Contiguous row-gather (embedding lookup) over a dense f64/f32 [V,D] table
        // with i64 indices (incl OOB, exercising default Clip clamp) must take the
        // typed plan and be value-equal (dtype + native bits) to the generic
        // interpreter (eval_primitive -> fj-lax oracle). Golden freezes f64.
        let (table, idx, out) = (VarId(0), VarId(1), VarId(2));
        let (v, d) = (10usize, 4usize);
        let table_f64: Vec<f64> = (0..v * d).map(|i| (i as f64) * 0.5 - 3.0).collect();
        let table_f32: Vec<f32> = table_f64.iter().map(|&x| x as f32).collect();
        // Mix in-bounds, first, last, and OOB (>=v -> clip to v-1) indices.
        let idx_vals: Vec<i64> = vec![2, 0, 9, 15, 7, 3];
        let nidx = idx_vals.len();
        let indices = Value::Tensor(
            TensorValue::new_i64_values(
                Shape {
                    dims: vec![nidx as u32],
                },
                idx_vals,
            )
            .unwrap(),
        );
        let body = Jaxpr::new(
            vec![table, idx],
            vec![],
            vec![out],
            vec![Equation {
                primitive: Primitive::Gather,
                inputs: smallvec![Atom::Var(table), Atom::Var(idx)],
                outputs: smallvec![out],
                params: BTreeMap::from([("slice_sizes".to_owned(), format!("1,{d}"))]),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        assert!(
            plan.dense_gather_plan.is_some(),
            "row-gather should compile to the dense gather plan"
        );

        let mut golden: Option<Vec<u64>> = None;
        for (is_f64, table_arg) in [
            (
                true,
                Value::Tensor(
                    TensorValue::new_f64_values(
                        Shape {
                            dims: vec![v as u32, d as u32],
                        },
                        table_f64.clone(),
                    )
                    .unwrap(),
                ),
            ),
            (
                false,
                Value::Tensor(
                    TensorValue::new_f32_values(
                        Shape {
                            dims: vec![v as u32, d as u32],
                        },
                        table_f32.clone(),
                    )
                    .unwrap(),
                ),
            ),
            (
                false,
                Value::Tensor(
                    TensorValue::new_i64_values(
                        Shape {
                            dims: vec![v as u32, d as u32],
                        },
                        (0..v * d).map(|i| i as i64 - 5).collect(),
                    )
                    .unwrap(),
                ),
            ),
            (
                false,
                Value::Tensor(
                    TensorValue::new_half_float_values(
                        DType::BF16,
                        Shape {
                            dims: vec![v as u32, d as u32],
                        },
                        (0..v * d)
                            .map(|i| match Literal::from_bf16_f64((i as f64) * 0.5 - 3.0) {
                                Literal::BF16Bits(b) => b,
                                _ => unreachable!(),
                            })
                            .collect(),
                    )
                    .unwrap(),
                ),
            ),
        ] {
            let args = [table_arg, indices.clone()];
            let mut genv: Vec<Option<Value>> = vec![None; plan.slots];
            let mut gscratch: Vec<Value> = Vec::new();
            let mut gout: Vec<Value> = Vec::new();
            super::run_dense_env_into(
                &body,
                &[],
                &args,
                &mut genv,
                &plan.last_use,
                &mut gscratch,
                &mut gout,
            )
            .expect("generic gather");

            let mut penv: Vec<Option<Value>> = vec![None; plan.slots];
            let mut pscratch: Vec<Value> = Vec::new();
            let mut pout: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            super::run_dense_plan_into(
                &body,
                &[],
                &args,
                &mut penv,
                &plan,
                &mut pscratch,
                &mut pout,
                &mut bufs,
            )
            .expect("planned gather");

            assert_eq!(
                tensor_dtype_bits(&pout[0]),
                tensor_dtype_bits(&gout[0]),
                "gather f64={is_f64} plan diverged from oracle"
            );
            if is_f64 {
                golden = Some(tensor_dtype_bits(&pout[0]).1);
            }
        }
        let sha = fj_test_utils::fixture_id_from_json(&(
            "frankenjax-zmdg5-gather",
            &golden.expect("golden ran"),
        ))
        .expect("digest");
        assert_eq!(
            sha,
            "4b618ca529cc6bf02e477c6bdf9d42cee93562ad1711ddb239596de1bdc4198f"
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_dense_gather_plan_overhead() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        let (table, idx, out) = (VarId(0), VarId(1), VarId(2));
        let (v, d) = (64usize, 8usize);
        let table_data: Vec<f32> = (0..v * d).map(|i| i as f32 * 0.01).collect();
        let body = Jaxpr::new(
            vec![table, idx],
            vec![],
            vec![out],
            vec![Equation {
                primitive: Primitive::Gather,
                inputs: smallvec![Atom::Var(table), Atom::Var(idx)],
                outputs: smallvec![out],
                params: BTreeMap::from([("slice_sizes".to_owned(), format!("1,{d}"))]),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        let args = [
            Value::Tensor(
                TensorValue::new_f32_values(
                    Shape {
                        dims: vec![v as u32, d as u32],
                    },
                    table_data,
                )
                .unwrap(),
            ),
            Value::Tensor(
                TensorValue::new_i64_values(Shape { dims: vec![4] }, vec![3i64, 17, 40, 8])
                    .unwrap(),
            ),
        ];
        let n: usize = 500_000;
        let t_generic = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan.last_use,
                    &mut scratch,
                    &mut o,
                )
                .expect("generic");
                acc += tensor_dtype_bits(&o[0]).1[0] as f64;
            }
            std::hint::black_box(acc);
        });
        let t_planned = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_plan_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan,
                    &mut scratch,
                    &mut o,
                    &mut bufs,
                )
                .expect("planned");
                acc += tensor_dtype_bits(&o[0]).1[0] as f64;
            }
            std::hint::black_box(acc);
        });
        println!(
            "BENCH dense-f32 gather [64,8] 4-idx {n} evals: GENERIC {:.1}ns/eval -> PLANNED {:.1}ns/eval = {:.2}x",
            t_generic * 1e9 / n as f64,
            t_planned * 1e9 / n as f64,
            t_generic / t_planned,
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_dense_axis_reduce_plan_overhead() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }

        let (x, out) = (VarId(0), VarId(1));
        let body = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec![Atom::Var(x)],
                outputs: smallvec![out],
                params: BTreeMap::from([("axes".to_owned(), "1".to_owned())]),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        let data: Vec<f64> = (0..16 * 16).map(|i| (i as f64) * 0.013 - 1.0).collect();
        let arg =
            Value::Tensor(TensorValue::new_f64_values(Shape { dims: vec![16, 16] }, data).unwrap());
        let args = [arg];

        let n: usize = 500_000;
        let t_generic = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan.last_use,
                    &mut scratch,
                    &mut o,
                )
                .expect("generic");
                acc += tensor_dtype_bits(&o[0]).1[0] as f64;
            }
            std::hint::black_box(acc);
        });
        let t_planned = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_plan_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan,
                    &mut scratch,
                    &mut o,
                    &mut bufs,
                )
                .expect("planned");
                acc += tensor_dtype_bits(&o[0]).1[0] as f64;
            }
            std::hint::black_box(acc);
        });
        println!(
            "BENCH dense-f64 axis-reduce [16,16] axis1 {n} evals: GENERIC {:.1}ns/eval -> PLANNED {:.1}ns/eval = {:.2}x",
            t_generic * 1e9 / n as f64,
            t_planned * 1e9 / n as f64,
            t_generic / t_planned,
        );
    }

    #[test]
    fn dense_f64_dot_plan_matches_generic_sha256() {
        let (lhs, rhs, out) = (VarId(0), VarId(1), VarId(2));
        let body = Jaxpr::new(
            vec![lhs, rhs],
            vec![],
            vec![out],
            vec![Equation {
                primitive: Primitive::Dot,
                inputs: smallvec![Atom::Var(lhs), Atom::Var(rhs)],
                outputs: smallvec![out],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        assert!(
            plan.dense_f64_dot_plan.is_some(),
            "dot body should compile to the dense-f64 plan"
        );

        let lhs_data: Vec<f64> = (0..64)
            .map(|i| {
                if i == 5 {
                    -0.0
                } else {
                    (i as f64) * 0.03125 - 1.0
                }
            })
            .collect();
        let rhs_data: Vec<f64> = (0..64)
            .map(|i| {
                if i == 11 {
                    f64::NAN
                } else {
                    0.75 - (i as f64) * 0.015625
                }
            })
            .collect();
        let args = [
            Value::Tensor(TensorValue::new_f64_values(Shape { dims: vec![64] }, lhs_data).unwrap()),
            Value::Tensor(TensorValue::new_f64_values(Shape { dims: vec![64] }, rhs_data).unwrap()),
        ];

        let mut generic_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut generic_scratch: Vec<Value> = Vec::new();
        let mut generic_out: Vec<Value> = Vec::new();
        super::run_dense_env_into(
            &body,
            &[],
            &args,
            &mut generic_env,
            &plan.last_use,
            &mut generic_scratch,
            &mut generic_out,
        )
        .expect("generic dot");

        let mut planned_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut planned_scratch: Vec<Value> = Vec::new();
        let mut planned_out: Vec<Value> = Vec::new();
        let mut bufs = super::ScalarPlanBuffers::default();
        super::run_dense_plan_into(
            &body,
            &[],
            &args,
            &mut planned_env,
            &plan,
            &mut planned_scratch,
            &mut planned_out,
            &mut bufs,
        )
        .expect("planned dot");

        assert_eq!(planned_out, generic_out);
        let sha256 = fj_test_utils::fixture_id_from_json(&("frankenjax-1xu5v-dot", &planned_out))
            .expect("golden digest");
        assert_eq!(
            sha256,
            "7006db6112a6270916d28d82faab880018eabc2507e4aaf5e130c9bad845a4fa"
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_dense_f64_reduce_sum_plan_overhead() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }

        let (x, out) = (VarId(0), VarId(1));
        let body = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec![Atom::Var(x)],
                outputs: smallvec![out],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        assert!(
            plan.dense_f64_reduce_sum_plan.is_some(),
            "reduce_sum body should compile to the dense-f64 plan"
        );
        let data: Vec<f64> = (0..64)
            .map(|i| {
                if i == 7 {
                    -0.0
                } else {
                    (i as f64) * 0.125 - 3.5
                }
            })
            .collect();
        let arg =
            Value::Tensor(TensorValue::new_f64_values(Shape { dims: vec![64] }, data).unwrap());
        let args = [arg];
        let mut generic_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut generic_scratch: Vec<Value> = Vec::new();
        let mut generic_out: Vec<Value> = Vec::new();
        super::run_dense_env_into(
            &body,
            &[],
            &args,
            &mut generic_env,
            &plan.last_use,
            &mut generic_scratch,
            &mut generic_out,
        )
        .expect("generic sample");

        let mut planned_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut planned_scratch: Vec<Value> = Vec::new();
        let mut planned_out: Vec<Value> = Vec::new();
        let mut bufs = super::ScalarPlanBuffers::default();
        super::run_dense_plan_into(
            &body,
            &[],
            &args,
            &mut planned_env,
            &plan,
            &mut planned_scratch,
            &mut planned_out,
            &mut bufs,
        )
        .expect("planned sample");
        assert_eq!(planned_out, generic_out);
        let sha256 =
            fj_test_utils::fixture_id_from_json(&("frankenjax-0x3pu-reduce-sum", &planned_out))
                .expect("golden digest");
        assert_eq!(
            sha256,
            "561117f6fd0383063821dfcf3074491ba1e5f3943a828629e080c4fa7897c8cd"
        );

        let n: usize = 1_000_000;
        let t_generic = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan.last_use,
                    &mut scratch,
                    &mut o,
                )
                .expect("generic");
                acc += o[0].as_f64_scalar().expect("scalar reduce_sum");
            }
            std::hint::black_box(acc);
        });
        let t_planned = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_plan_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan,
                    &mut scratch,
                    &mut o,
                    &mut bufs,
                )
                .expect("planned");
                acc += o[0].as_f64_scalar().expect("scalar reduce_sum");
            }
            std::hint::black_box(acc);
        });
        println!(
            "BENCH dense-f64 reduce_sum [64] {n} evals: GENERIC {:.1}ns/eval -> PLANNED {:.1}ns/eval = {:.2}x sha256={}",
            t_generic * 1e9 / n as f64,
            t_planned * 1e9 / n as f64,
            t_generic / t_planned,
            sha256,
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_dense_f32_reduce_sum_plan_overhead() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }

        let (x, out) = (VarId(0), VarId(1));
        let body = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec![Atom::Var(x)],
                outputs: smallvec![out],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        let data: Vec<f32> = (0..64).map(|i| (i as f32) * 0.125 - 3.5).collect();
        let arg =
            Value::Tensor(TensorValue::new_f32_values(Shape { dims: vec![64] }, data).unwrap());
        let args = [arg];

        let n: usize = 1_000_000;
        let t_generic = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan.last_use,
                    &mut scratch,
                    &mut o,
                )
                .expect("generic");
                acc += o[0].as_f64_scalar().expect("scalar reduce_sum");
            }
            std::hint::black_box(acc);
        });
        let t_planned = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_plan_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan,
                    &mut scratch,
                    &mut o,
                    &mut bufs,
                )
                .expect("planned");
                acc += o[0].as_f64_scalar().expect("scalar reduce_sum");
            }
            std::hint::black_box(acc);
        });
        println!(
            "BENCH dense-f32 reduce_sum [64] {n} evals: GENERIC {:.1}ns/eval -> PLANNED {:.1}ns/eval = {:.2}x",
            t_generic * 1e9 / n as f64,
            t_planned * 1e9 / n as f64,
            t_generic / t_planned,
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_dense_f64_dot_plan_overhead() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }

        let (lhs, rhs, out) = (VarId(0), VarId(1), VarId(2));
        let body = Jaxpr::new(
            vec![lhs, rhs],
            vec![],
            vec![out],
            vec![Equation {
                primitive: Primitive::Dot,
                inputs: smallvec![Atom::Var(lhs), Atom::Var(rhs)],
                outputs: smallvec![out],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        assert!(
            plan.dense_f64_dot_plan.is_some(),
            "dot body should compile to the dense-f64 plan"
        );
        let lhs_data: Vec<f64> = (0..64)
            .map(|i| {
                if i == 5 {
                    -0.0
                } else {
                    (i as f64) * 0.03125 - 1.0
                }
            })
            .collect();
        let rhs_data: Vec<f64> = (0..64)
            .map(|i| {
                if i == 11 {
                    f64::NAN
                } else {
                    0.75 - (i as f64) * 0.015625
                }
            })
            .collect();
        let args = [
            Value::Tensor(TensorValue::new_f64_values(Shape { dims: vec![64] }, lhs_data).unwrap()),
            Value::Tensor(TensorValue::new_f64_values(Shape { dims: vec![64] }, rhs_data).unwrap()),
        ];
        let mut generic_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut generic_scratch: Vec<Value> = Vec::new();
        let mut generic_out: Vec<Value> = Vec::new();
        super::run_dense_env_into(
            &body,
            &[],
            &args,
            &mut generic_env,
            &plan.last_use,
            &mut generic_scratch,
            &mut generic_out,
        )
        .expect("generic sample");

        let mut planned_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut planned_scratch: Vec<Value> = Vec::new();
        let mut planned_out: Vec<Value> = Vec::new();
        let mut bufs = super::ScalarPlanBuffers::default();
        super::run_dense_plan_into(
            &body,
            &[],
            &args,
            &mut planned_env,
            &plan,
            &mut planned_scratch,
            &mut planned_out,
            &mut bufs,
        )
        .expect("planned sample");
        assert_eq!(planned_out, generic_out);
        let sha256 = fj_test_utils::fixture_id_from_json(&("frankenjax-1xu5v-dot", &planned_out))
            .expect("golden digest");
        assert_eq!(
            sha256,
            "7006db6112a6270916d28d82faab880018eabc2507e4aaf5e130c9bad845a4fa"
        );

        let n: usize = 1_000_000;
        let t_generic = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan.last_use,
                    &mut scratch,
                    &mut o,
                )
                .expect("generic");
                acc += o[0].as_f64_scalar().expect("scalar dot");
            }
            std::hint::black_box(acc);
        });
        let t_planned = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_plan_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan,
                    &mut scratch,
                    &mut o,
                    &mut bufs,
                )
                .expect("planned");
                acc += o[0].as_f64_scalar().expect("scalar dot");
            }
            std::hint::black_box(acc);
        });
        println!(
            "BENCH dense-f64 dot [64] {n} evals: GENERIC {:.1}ns/eval -> PLANNED {:.1}ns/eval = {:.2}x sha256={}",
            t_generic * 1e9 / n as f64,
            t_planned * 1e9 / n as f64,
            t_generic / t_planned,
            sha256,
        );
    }

    #[test]
    fn dense_reshape_plan_matches_generic_and_golden() {
        let (x, out) = (VarId(0), VarId(1));
        let body = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![Equation {
                primitive: Primitive::Reshape,
                inputs: smallvec![Atom::Var(x)],
                outputs: smallvec![out],
                params: BTreeMap::from([("new_shape".to_owned(), "8,8".to_owned())]),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        assert!(
            plan.dense_reshape_plan.is_some(),
            "reshape body should compile to the dense reshape plan"
        );

        let data: Vec<f64> = (0..64)
            .map(|i| if i == 7 { -0.0 } else { i as f64 * 0.125 })
            .collect();
        let args = [Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: vec![64] }, data).unwrap(),
        )];

        let mut generic_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut generic_scratch: Vec<Value> = Vec::new();
        let mut generic_out: Vec<Value> = Vec::new();
        super::run_dense_env_into(
            &body,
            &[],
            &args,
            &mut generic_env,
            &plan.last_use,
            &mut generic_scratch,
            &mut generic_out,
        )
        .expect("generic reshape");

        let mut planned_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut planned_scratch: Vec<Value> = Vec::new();
        let mut planned_out: Vec<Value> = Vec::new();
        let mut bufs = super::ScalarPlanBuffers::default();
        super::run_dense_plan_into(
            &body,
            &[],
            &args,
            &mut planned_env,
            &plan,
            &mut planned_scratch,
            &mut planned_out,
            &mut bufs,
        )
        .expect("planned reshape");

        assert_eq!(planned_out, generic_out);
        assert!(
            matches!(&planned_out[0], Value::Tensor(tensor) if tensor.shape.dims == vec![8, 8])
        );
        let sha256 =
            fj_test_utils::fixture_id_from_json(&("frankenjax-0x3pu-reshape", &planned_out))
                .expect("golden digest");
        assert_eq!(
            sha256,
            "0fd7ecf0b76e1b8e14e24fb5f64833041302b2fa0170cf203709fc8cc10402ef"
        );
    }

    #[test]
    fn dense_reshape_plan_supports_inferred_dimension() {
        let (x, out) = (VarId(0), VarId(1));
        let body = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![Equation {
                primitive: Primitive::Reshape,
                inputs: smallvec![Atom::Var(x)],
                outputs: smallvec![out],
                params: BTreeMap::from([("new_shape".to_owned(), "8,-1".to_owned())]),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        assert!(
            plan.dense_reshape_plan.is_some(),
            "inferred reshape body should compile to the dense reshape plan"
        );

        let data: Vec<f64> = (0..64)
            .map(|i| if i == 7 { -0.0 } else { i as f64 * 0.125 })
            .collect();
        let args = [Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: vec![64] }, data).unwrap(),
        )];

        let mut generic_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut generic_scratch: Vec<Value> = Vec::new();
        let mut generic_out: Vec<Value> = Vec::new();
        super::run_dense_env_into(
            &body,
            &[],
            &args,
            &mut generic_env,
            &plan.last_use,
            &mut generic_scratch,
            &mut generic_out,
        )
        .expect("generic reshape");

        let mut planned_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut planned_scratch: Vec<Value> = Vec::new();
        let mut planned_out: Vec<Value> = Vec::new();
        let mut bufs = super::ScalarPlanBuffers::default();
        super::run_dense_plan_into(
            &body,
            &[],
            &args,
            &mut planned_env,
            &plan,
            &mut planned_scratch,
            &mut planned_out,
            &mut bufs,
        )
        .expect("planned reshape");

        assert_eq!(planned_out, generic_out);
        assert!(
            matches!(&planned_out[0], Value::Tensor(tensor) if tensor.shape.dims == vec![8, 8])
        );
        let sha256 =
            fj_test_utils::fixture_id_from_json(&("frankenjax-0x3pu-reshape", &planned_out))
                .expect("golden digest");
        assert_eq!(
            sha256,
            "0fd7ecf0b76e1b8e14e24fb5f64833041302b2fa0170cf203709fc8cc10402ef"
        );
    }

    #[test]
    fn dense_f64_transpose_plan_matches_generic_and_golden() {
        let (x, out) = (VarId(0), VarId(1));
        let body = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![Equation {
                primitive: Primitive::Transpose,
                inputs: smallvec![Atom::Var(x)],
                outputs: smallvec![out],
                params: BTreeMap::from([("permutation".to_owned(), "1,0".to_owned())]),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        assert!(
            plan.dense_f64_transpose_plan.is_some(),
            "transpose body should compile to the dense F64 transpose plan"
        );

        let data: Vec<f64> = (0..64)
            .map(|i| if i == 7 { -0.0 } else { i as f64 * 0.125 })
            .collect();
        let args = [Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: vec![8, 8] }, data).unwrap(),
        )];

        let mut generic_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut generic_scratch: Vec<Value> = Vec::new();
        let mut generic_out: Vec<Value> = Vec::new();
        super::run_dense_env_into(
            &body,
            &[],
            &args,
            &mut generic_env,
            &plan.last_use,
            &mut generic_scratch,
            &mut generic_out,
        )
        .expect("generic transpose");

        let mut planned_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut planned_scratch: Vec<Value> = Vec::new();
        let mut planned_out: Vec<Value> = Vec::new();
        let mut bufs = super::ScalarPlanBuffers::default();
        super::run_dense_plan_into(
            &body,
            &[],
            &args,
            &mut planned_env,
            &plan,
            &mut planned_scratch,
            &mut planned_out,
            &mut bufs,
        )
        .expect("planned transpose");

        assert_eq!(planned_out, generic_out);
        let Value::Tensor(tensor) = &planned_out[0] else {
            panic!("transpose should return a tensor");
        };
        assert_eq!(tensor.shape.dims, vec![8, 8]);
        let sha256 =
            fj_test_utils::fixture_id_from_json(&("frankenjax-0x3pu-transpose", &planned_out))
                .expect("golden digest");
        assert_eq!(
            sha256,
            "b2825bcbd11e5a576e0898239bb4e5107300b70b66e475b45ebfa360cb46f308"
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_dense_f64_transpose_plan_overhead() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }

        let (x, out) = (VarId(0), VarId(1));
        let body = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![Equation {
                primitive: Primitive::Transpose,
                inputs: smallvec![Atom::Var(x)],
                outputs: smallvec![out],
                params: BTreeMap::from([("permutation".to_owned(), "1,0".to_owned())]),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        let data: Vec<f64> = (0..64)
            .map(|i| if i == 7 { -0.0 } else { i as f64 * 0.125 })
            .collect();
        let arg =
            Value::Tensor(TensorValue::new_f64_values(Shape { dims: vec![8, 8] }, data).unwrap());
        let args = [arg];

        let mut generic_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut generic_scratch: Vec<Value> = Vec::new();
        let mut generic_out: Vec<Value> = Vec::new();
        super::run_dense_env_into(
            &body,
            &[],
            &args,
            &mut generic_env,
            &plan.last_use,
            &mut generic_scratch,
            &mut generic_out,
        )
        .expect("generic sample");

        let mut planned_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut planned_scratch: Vec<Value> = Vec::new();
        let mut planned_out: Vec<Value> = Vec::new();
        let mut bufs = super::ScalarPlanBuffers::default();
        super::run_dense_plan_into(
            &body,
            &[],
            &args,
            &mut planned_env,
            &plan,
            &mut planned_scratch,
            &mut planned_out,
            &mut bufs,
        )
        .expect("planned sample");
        assert_eq!(planned_out, generic_out);
        let sha256 =
            fj_test_utils::fixture_id_from_json(&("frankenjax-0x3pu-transpose", &planned_out))
                .expect("golden digest");

        let n: usize = 1_000_000;
        let t_generic = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan.last_use,
                    &mut scratch,
                    &mut o,
                )
                .expect("generic");
                acc += tensor_dtype_bits(&o[0]).1[0] as f64;
            }
            std::hint::black_box(acc);
        });
        let t_planned = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_plan_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan,
                    &mut scratch,
                    &mut o,
                    &mut bufs,
                )
                .expect("planned");
                acc += tensor_dtype_bits(&o[0]).1[0] as f64;
            }
            std::hint::black_box(acc);
        });
        println!(
            "BENCH dense-f64 transpose [8,8]->[8,8] {n} evals: GENERIC {:.1}ns/eval -> PLANNED {:.1}ns/eval = {:.2}x sha256={sha256}",
            t_generic * 1e9 / n as f64,
            t_planned * 1e9 / n as f64,
            t_generic / t_planned,
        );
    }

    #[test]
    fn dense_f64_broadcast_in_dim_plan_matches_generic_and_golden() {
        let (x, out) = (VarId(0), VarId(1));
        let body = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![Equation {
                primitive: Primitive::BroadcastInDim,
                inputs: smallvec![Atom::Var(x)],
                outputs: smallvec![out],
                params: BTreeMap::from([
                    ("shape".to_owned(), "8,8".to_owned()),
                    ("broadcast_dimensions".to_owned(), "1".to_owned()),
                ]),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        assert!(
            plan.dense_f64_broadcast_in_dim_plan.is_some(),
            "broadcast body should compile to the dense F64 broadcast plan"
        );

        let data: Vec<f64> = (0..8)
            .map(|i| if i == 7 { -0.0 } else { i as f64 * 0.125 })
            .collect();
        let args = [Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: vec![8] }, data).unwrap(),
        )];

        let mut generic_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut generic_scratch: Vec<Value> = Vec::new();
        let mut generic_out: Vec<Value> = Vec::new();
        super::run_dense_env_into(
            &body,
            &[],
            &args,
            &mut generic_env,
            &plan.last_use,
            &mut generic_scratch,
            &mut generic_out,
        )
        .expect("generic broadcast");

        let mut planned_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut planned_scratch: Vec<Value> = Vec::new();
        let mut planned_out: Vec<Value> = Vec::new();
        let mut bufs = super::ScalarPlanBuffers::default();
        super::run_dense_plan_into(
            &body,
            &[],
            &args,
            &mut planned_env,
            &plan,
            &mut planned_scratch,
            &mut planned_out,
            &mut bufs,
        )
        .expect("planned broadcast");

        assert_eq!(planned_out, generic_out);
        let sha256 =
            fj_test_utils::fixture_id_from_json(&("frankenjax-0x3pu-broadcast", &planned_out))
                .expect("golden digest");
        assert_eq!(
            sha256,
            "d2f5c8ec9a925c62333af4e4713f2dbc106e1dd98de0473eca0199ff91b6fab7"
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_dense_f64_broadcast_in_dim_plan_overhead() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }

        let (x, out) = (VarId(0), VarId(1));
        let body = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![Equation {
                primitive: Primitive::BroadcastInDim,
                inputs: smallvec![Atom::Var(x)],
                outputs: smallvec![out],
                params: BTreeMap::from([
                    ("shape".to_owned(), "8,8".to_owned()),
                    ("broadcast_dimensions".to_owned(), "1".to_owned()),
                ]),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        let data: Vec<f64> = (0..8)
            .map(|i| if i == 7 { -0.0 } else { i as f64 * 0.125 })
            .collect();
        let arg =
            Value::Tensor(TensorValue::new_f64_values(Shape { dims: vec![8] }, data).unwrap());
        let args = [arg];

        let mut generic_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut generic_scratch: Vec<Value> = Vec::new();
        let mut generic_out: Vec<Value> = Vec::new();
        super::run_dense_env_into(
            &body,
            &[],
            &args,
            &mut generic_env,
            &plan.last_use,
            &mut generic_scratch,
            &mut generic_out,
        )
        .expect("generic sample");

        let mut planned_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut planned_scratch: Vec<Value> = Vec::new();
        let mut planned_out: Vec<Value> = Vec::new();
        let mut bufs = super::ScalarPlanBuffers::default();
        super::run_dense_plan_into(
            &body,
            &[],
            &args,
            &mut planned_env,
            &plan,
            &mut planned_scratch,
            &mut planned_out,
            &mut bufs,
        )
        .expect("planned sample");
        assert_eq!(planned_out, generic_out);
        let sha256 =
            fj_test_utils::fixture_id_from_json(&("frankenjax-0x3pu-broadcast", &planned_out))
                .expect("golden digest");

        let n: usize = 1_000_000;
        let t_generic = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan.last_use,
                    &mut scratch,
                    &mut o,
                )
                .expect("generic");
                acc += tensor_dtype_bits(&o[0]).1[0] as f64;
            }
            std::hint::black_box(acc);
        });
        let t_planned = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_plan_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan,
                    &mut scratch,
                    &mut o,
                    &mut bufs,
                )
                .expect("planned");
                acc += tensor_dtype_bits(&o[0]).1[0] as f64;
            }
            std::hint::black_box(acc);
        });
        println!(
            "BENCH dense-f64 broadcast_in_dim [8]->[8,8] {n} evals: GENERIC {:.1}ns/eval -> PLANNED {:.1}ns/eval = {:.2}x sha256={sha256}",
            t_generic * 1e9 / n as f64,
            t_planned * 1e9 / n as f64,
            t_generic / t_planned,
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_dense_f64_reshape_plan_overhead() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }

        let (x, out) = (VarId(0), VarId(1));
        let body = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![Equation {
                primitive: Primitive::Reshape,
                inputs: smallvec![Atom::Var(x)],
                outputs: smallvec![out],
                params: BTreeMap::from([("new_shape".to_owned(), "8,8".to_owned())]),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        let data: Vec<f64> = (0..64)
            .map(|i| if i == 7 { -0.0 } else { i as f64 * 0.125 })
            .collect();
        let arg =
            Value::Tensor(TensorValue::new_f64_values(Shape { dims: vec![64] }, data).unwrap());
        let args = [arg];

        let mut generic_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut generic_scratch: Vec<Value> = Vec::new();
        let mut generic_out: Vec<Value> = Vec::new();
        super::run_dense_env_into(
            &body,
            &[],
            &args,
            &mut generic_env,
            &plan.last_use,
            &mut generic_scratch,
            &mut generic_out,
        )
        .expect("generic sample");

        let mut planned_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut planned_scratch: Vec<Value> = Vec::new();
        let mut planned_out: Vec<Value> = Vec::new();
        let mut bufs = super::ScalarPlanBuffers::default();
        super::run_dense_plan_into(
            &body,
            &[],
            &args,
            &mut planned_env,
            &plan,
            &mut planned_scratch,
            &mut planned_out,
            &mut bufs,
        )
        .expect("planned sample");
        assert_eq!(planned_out, generic_out);
        let sha256 =
            fj_test_utils::fixture_id_from_json(&("frankenjax-0x3pu-reshape", &planned_out))
                .expect("golden digest");

        let n: usize = 1_000_000;
        let t_generic = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan.last_use,
                    &mut scratch,
                    &mut o,
                )
                .expect("generic");
                acc += tensor_dtype_bits(&o[0]).1[0] as f64;
            }
            std::hint::black_box(acc);
        });
        let t_planned = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_plan_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan,
                    &mut scratch,
                    &mut o,
                    &mut bufs,
                )
                .expect("planned");
                acc += tensor_dtype_bits(&o[0]).1[0] as f64;
            }
            std::hint::black_box(acc);
        });
        println!(
            "BENCH dense-f64 reshape [64]->[8,8] {n} evals: GENERIC {:.1}ns/eval -> PLANNED {:.1}ns/eval = {:.2}x sha256={sha256}",
            t_generic * 1e9 / n as f64,
            t_planned * 1e9 / n as f64,
            t_generic / t_planned,
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_scalar_poly_f32_mixed_precision() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        let mk = |p: Primitive,
                  ins: smallvec::SmallVec<[Atom; 4]>,
                  params: BTreeMap<String, String>,
                  o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params,
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let conv = |to: &str| {
            let mut p = BTreeMap::new();
            p.insert("new_dtype".to_owned(), to.to_owned());
            p
        };
        let f64lit = |v: f64| Atom::Lit(Literal::from_f64(v));
        // Mixed-precision body: x_f32 -> f64; y = x*pi + 1; -> f32 (4 ops, 2 converts).
        let (xf, xd, m, s, of) = (VarId(0), VarId(1), VarId(2), VarId(3), VarId(4));
        let body = Jaxpr::new(
            vec![xf],
            vec![],
            vec![of],
            vec![
                mk(
                    Primitive::ConvertElementType,
                    smallvec![Atom::Var(xf)],
                    conv("f64"),
                    xd,
                ),
                mk(
                    Primitive::Mul,
                    smallvec![Atom::Var(xd), f64lit(std::f64::consts::PI)],
                    BTreeMap::new(),
                    m,
                ),
                mk(
                    Primitive::Add,
                    smallvec![Atom::Var(m), f64lit(1.0)],
                    BTreeMap::new(),
                    s,
                ),
                mk(
                    Primitive::ConvertElementType,
                    smallvec![Atom::Var(s)],
                    conv("f32"),
                    of,
                ),
            ],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        assert!(plan.scalar_poly_plan.is_some());
        let n: usize = 2_000_000;
        let args = [Value::scalar_f32(1.5)];
        let run = |use_plan: bool| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            let mut acc = 0.0f32;
            for _ in 0..n {
                if use_plan {
                    super::run_dense_plan_into(
                        &body,
                        &[],
                        &args,
                        &mut env,
                        &plan,
                        &mut scratch,
                        &mut o,
                        &mut bufs,
                    )
                    .expect("compiled");
                } else {
                    super::run_dense_env_into(
                        &body,
                        &[],
                        &args,
                        &mut env,
                        &plan.last_use,
                        &mut scratch,
                        &mut o,
                    )
                    .expect("generic");
                }
                if let Value::Scalar(Literal::F32Bits(b)) = &o[0] {
                    acc += f32::from_bits(*b);
                }
            }
            std::hint::black_box(acc);
        };
        let t_generic = best_time(|| run(false));
        let t_compiled = best_time(|| run(true));
        println!(
            "BENCH poly-f32 mixed-precision {n} evals (f32->f64; *pi+1; ->f32): GENERIC {:.1}ns/eval -> COMPILED {:.1}ns/eval = {:.2}x",
            t_generic * 1e9 / n as f64,
            t_compiled * 1e9 / n as f64,
            t_generic / t_compiled,
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_scalar_poly_arena() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        let mk = |p: Primitive,
                  ins: smallvec::SmallVec<[Atom; 4]>,
                  params: BTreeMap<String, String>,
                  o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params,
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let mut cp = BTreeMap::new();
        cp.insert("new_dtype".to_owned(), "f64".to_owned());
        let f64lit = |v: f64| Atom::Lit(Literal::from_f64(v));
        // int→float mixed body: f = convert(i, f64); out = f*0.5 + 1.0 (convert + 2 arith).
        let (i, f, h, out) = (VarId(0), VarId(1), VarId(2), VarId(3));
        let body = Jaxpr::new(
            vec![i],
            vec![],
            vec![out],
            vec![
                mk(
                    Primitive::ConvertElementType,
                    smallvec![Atom::Var(i)],
                    cp,
                    f,
                ),
                mk(
                    Primitive::Mul,
                    smallvec![Atom::Var(f), f64lit(0.5)],
                    BTreeMap::new(),
                    h,
                ),
                mk(
                    Primitive::Add,
                    smallvec![Atom::Var(h), f64lit(1.0)],
                    BTreeMap::new(),
                    out,
                ),
            ],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        assert!(plan.scalar_poly_plan.is_some());
        let n: usize = 2_000_000;
        let args = [Value::scalar_i64(7)];
        let run = |use_plan: bool| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            let mut acc = 0.0f64;
            for _ in 0..n {
                if use_plan {
                    super::run_dense_plan_into(
                        &body,
                        &[],
                        &args,
                        &mut env,
                        &plan,
                        &mut scratch,
                        &mut o,
                        &mut bufs,
                    )
                    .expect("compiled");
                } else {
                    super::run_dense_env_into(
                        &body,
                        &[],
                        &args,
                        &mut env,
                        &plan.last_use,
                        &mut scratch,
                        &mut o,
                    )
                    .expect("generic");
                }
                if let Value::Scalar(Literal::F64Bits(b)) = &o[0] {
                    acc += f64::from_bits(*b);
                }
            }
            std::hint::black_box(acc);
        };
        let t_generic = best_time(|| run(false));
        let t_compiled = best_time(|| run(true));
        println!(
            "BENCH poly-body dispatch {n} evals (convert i64->f64 + mul + add): GENERIC {:.1}ns/eval -> COMPILED {:.1}ns/eval = {:.2}x",
            t_generic * 1e9 / n as f64,
            t_compiled * 1e9 / n as f64,
            t_generic / t_compiled,
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_scalar_select_i64_arena() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let ilit = |v: i64| Atom::Lit(Literal::I64(v));
        // integer-clamp carry body (2 compares + 2 selects): clamp x to [0,10].
        let (x, gt, lt0, inner, out) = (VarId(0), VarId(1), VarId(2), VarId(3), VarId(4));
        let body = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                mk(Primitive::Gt, smallvec![Atom::Var(x), ilit(10)], gt),
                mk(Primitive::Lt, smallvec![Atom::Var(x), ilit(0)], lt0),
                mk(
                    Primitive::Select,
                    smallvec![Atom::Var(lt0), ilit(0), Atom::Var(x)],
                    inner,
                ),
                mk(
                    Primitive::Select,
                    smallvec![Atom::Var(gt), ilit(10), Atom::Var(inner)],
                    out,
                ),
            ],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        assert!(plan.scalar_select_i64_plan.is_some());
        let n: usize = 2_000_000;
        let args = [Value::scalar_i64(5)];
        let run = |use_plan: bool| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            let mut acc = 0i64;
            for _ in 0..n {
                if use_plan {
                    super::run_dense_plan_into(
                        &body,
                        &[],
                        &args,
                        &mut env,
                        &plan,
                        &mut scratch,
                        &mut o,
                        &mut bufs,
                    )
                    .expect("compiled");
                } else {
                    super::run_dense_env_into(
                        &body,
                        &[],
                        &args,
                        &mut env,
                        &plan.last_use,
                        &mut scratch,
                        &mut o,
                    )
                    .expect("generic");
                }
                if let Value::Scalar(Literal::I64(b)) = &o[0] {
                    acc = acc.wrapping_add(*b);
                }
            }
            std::hint::black_box(acc);
        };
        let t_generic = best_time(|| run(false));
        let t_compiled = best_time(|| run(true));
        println!(
            "BENCH i64-select-body dispatch {n} evals (clamp: 2 cmp + 2 select): GENERIC {:.1}ns/eval -> COMPILED {:.1}ns/eval = {:.2}x",
            t_generic * 1e9 / n as f64,
            t_compiled * 1e9 / n as f64,
            t_generic / t_compiled,
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_scalar_select_arena() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let lit = |v: f64| Atom::Lit(Literal::from_f64(v));
        // relu6-style clamp body (2 compares + 2 selects + arithmetic):
        //   t = x*1.0; select(x>6, 6, select(x<0, 0, t)).
        let (x, gt6, lt0, t, inner, out) =
            (VarId(0), VarId(1), VarId(2), VarId(3), VarId(4), VarId(5));
        let body = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                mk(Primitive::Gt, smallvec![Atom::Var(x), lit(6.0)], gt6),
                mk(Primitive::Lt, smallvec![Atom::Var(x), lit(0.0)], lt0),
                mk(Primitive::Mul, smallvec![Atom::Var(x), lit(1.0)], t),
                mk(
                    Primitive::Select,
                    smallvec![Atom::Var(lt0), lit(0.0), Atom::Var(t)],
                    inner,
                ),
                mk(
                    Primitive::Select,
                    smallvec![Atom::Var(gt6), lit(6.0), Atom::Var(inner)],
                    out,
                ),
            ],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        assert!(plan.scalar_select_plan.is_some());
        let n: usize = 2_000_000;
        let args = [Value::scalar_f64(3.5)];
        let run = |use_plan: bool| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            let mut acc = 0.0f64;
            for _ in 0..n {
                if use_plan {
                    super::run_dense_plan_into(
                        &body,
                        &[],
                        &args,
                        &mut env,
                        &plan,
                        &mut scratch,
                        &mut o,
                        &mut bufs,
                    )
                    .expect("compiled");
                } else {
                    super::run_dense_env_into(
                        &body,
                        &[],
                        &args,
                        &mut env,
                        &plan.last_use,
                        &mut scratch,
                        &mut o,
                    )
                    .expect("generic");
                }
                if let Value::Scalar(Literal::F64Bits(b)) = &o[0] {
                    acc += f64::from_bits(*b);
                }
            }
            std::hint::black_box(acc);
        };
        let t_generic = best_time(|| run(false));
        let t_compiled = best_time(|| run(true));
        println!(
            "BENCH select-body dispatch {n} evals (relu6: 2 cmp + 2 select + mul): GENERIC {:.1}ns/eval -> COMPILED {:.1}ns/eval = {:.2}x",
            t_generic * 1e9 / n as f64,
            t_compiled * 1e9 / n as f64,
            t_generic / t_compiled,
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_scalar_arena_power_body() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        let ipow = |x: VarId, e: i32, o: VarId| {
            let mut params = BTreeMap::new();
            params.insert("exponent".to_owned(), e.to_string());
            Equation {
                primitive: Primitive::IntegerPow,
                inputs: smallvec![Atom::Var(x)],
                outputs: smallvec![o],
                params,
                sub_jaxprs: vec![],
                effects: vec![],
            }
        };
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        // Variance-style power body: out = x**2 + x**3 + square(x) (3 power ops + add).
        let (x, p2, p3, sq, s1, out) = (VarId(0), VarId(1), VarId(2), VarId(3), VarId(4), VarId(5));
        let body = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                ipow(x, 2, p2),
                ipow(x, 3, p3),
                mk(Primitive::Square, smallvec![Atom::Var(x)], sq),
                mk(Primitive::Add, smallvec![Atom::Var(p2), Atom::Var(p3)], s1),
                mk(Primitive::Add, smallvec![Atom::Var(s1), Atom::Var(sq)], out),
            ],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        assert!(plan.scalar_f64_plan.is_some());
        let n: usize = 2_000_000;
        let args = [Value::scalar_f64(1.0001)];
        let run = |use_plan: bool| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            let mut acc = 0.0f64;
            for _ in 0..n {
                if use_plan {
                    super::run_dense_plan_into(
                        &body,
                        &[],
                        &args,
                        &mut env,
                        &plan,
                        &mut scratch,
                        &mut o,
                        &mut bufs,
                    )
                    .expect("compiled");
                } else {
                    super::run_dense_env_into(
                        &body,
                        &[],
                        &args,
                        &mut env,
                        &plan.last_use,
                        &mut scratch,
                        &mut o,
                    )
                    .expect("generic");
                }
                if let Value::Scalar(Literal::F64Bits(b)) = &o[0] {
                    acc += f64::from_bits(*b);
                }
            }
            std::hint::black_box(acc);
        };
        let t_generic = best_time(|| run(false));
        let t_compiled = best_time(|| run(true));
        println!(
            "BENCH power-body dispatch {n} evals (x**2 + x**3 + square(x)): GENERIC {:.1}ns/eval -> COMPILED {:.1}ns/eval = {:.2}x",
            t_generic * 1e9 / n as f64,
            t_compiled * 1e9 / n as f64,
            t_generic / t_compiled,
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_scalar_arena_transcendental_body() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        // sigmoid·tanh activation body (9 ops, 4 transcendental). GENERIC = prepared
        // dense runner through the generic interpreter; COMPILED = scalar f64 arena.
        let (x, e, d, s, t, q, lg, m, a8, out) = (
            VarId(0),
            VarId(1),
            VarId(2),
            VarId(3),
            VarId(4),
            VarId(5),
            VarId(6),
            VarId(7),
            VarId(8),
            VarId(9),
        );
        let one = Atom::Lit(Literal::from_f64(1.0));
        let body = Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                mk(Primitive::Exp, smallvec![Atom::Var(x)], e),
                mk(Primitive::Add, smallvec![Atom::Var(e), one], d),
                mk(Primitive::Div, smallvec![Atom::Var(e), Atom::Var(d)], s),
                mk(Primitive::Tanh, smallvec![Atom::Var(x)], t),
                mk(Primitive::Sqrt, smallvec![Atom::Var(d)], q),
                mk(Primitive::Log, smallvec![Atom::Var(d)], lg),
                mk(Primitive::Mul, smallvec![Atom::Var(s), Atom::Var(t)], m),
                mk(Primitive::Add, smallvec![Atom::Var(m), Atom::Var(q)], a8),
                mk(Primitive::Sub, smallvec![Atom::Var(a8), Atom::Var(lg)], out),
            ],
        );
        let plan = super::build_dense_plan(&body).expect("dense plan");
        assert!(plan.scalar_f64_plan.is_some());
        let n: usize = 2_000_000;
        let args = [Value::scalar_f64(0.7)];

        let t_generic = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan.last_use,
                    &mut scratch,
                    &mut o,
                )
                .expect("generic");
                if let Value::Scalar(Literal::F64Bits(b)) = &o[0] {
                    acc += f64::from_bits(*b);
                }
            }
            std::hint::black_box(acc);
        });
        let t_compiled = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_plan_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan,
                    &mut scratch,
                    &mut o,
                    &mut bufs,
                )
                .expect("compiled");
                if let Value::Scalar(Literal::F64Bits(b)) = &o[0] {
                    acc += f64::from_bits(*b);
                }
            }
            std::hint::black_box(acc);
        });
        println!(
            "BENCH activation-body dispatch {n} evals (9-op, 4 transcendental): GENERIC {:.1}ns/eval -> COMPILED {:.1}ns/eval = {:.2}x",
            t_generic * 1e9 / n as f64,
            t_compiled * 1e9 / n as f64,
            t_generic / t_compiled,
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_scan_body_interpreter_overhead() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        // SAME-INVOCATION A/B for the scan per-step body eval: a realistic
        // multi-equation f64 body (carry,x -> mul,add,sub). GENERIC is the
        // prepared dense runner that still resolves every equation through the
        // generic interpreter machinery; COMPILED is the pre-resolved scalar f64
        // step plan carried by DenseEvalPlan. Both execute the same operations in
        // the same order and must produce bit-identical output.
        let mk = |p: Primitive, ins: smallvec::SmallVec<[Atom; 4]>, o: VarId| Equation {
            primitive: p,
            inputs: ins,
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let (carry, x, v2, v3, out) = (VarId(0), VarId(1), VarId(2), VarId(3), VarId(4));
        let body = Jaxpr::new(
            vec![carry, x],
            vec![],
            vec![out],
            vec![
                mk(
                    Primitive::Mul,
                    smallvec![Atom::Var(carry), Atom::Var(x)],
                    v2,
                ),
                mk(
                    Primitive::Add,
                    smallvec![Atom::Var(v2), Atom::Var(carry)],
                    v3,
                ),
                mk(Primitive::Sub, smallvec![Atom::Var(v3), Atom::Var(x)], out),
            ],
        );
        let n: usize = 4_000_000;
        let args = [Value::scalar_f64(1.0001), Value::scalar_f64(0.9999)];

        let plan = super::build_dense_plan(&body).expect("dense");
        let mut generic_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut generic_scratch: Vec<Value> = Vec::new();
        let mut generic_out: Vec<Value> = Vec::new();
        super::run_dense_env_into(
            &body,
            &[],
            &args,
            &mut generic_env,
            &plan.last_use,
            &mut generic_scratch,
            &mut generic_out,
        )
        .expect("generic sample");

        let mut compiled_env: Vec<Option<Value>> = vec![None; plan.slots];
        let mut compiled_scratch: Vec<Value> = Vec::new();
        let mut compiled_out: Vec<Value> = Vec::new();
        let mut scalar_buffers = super::ScalarPlanBuffers::default();
        super::run_dense_plan_into(
            &body,
            &[],
            &args,
            &mut compiled_env,
            &plan,
            &mut compiled_scratch,
            &mut compiled_out,
            &mut scalar_buffers,
        )
        .expect("compiled sample");
        assert_eq!(compiled_out, generic_out);
        let sha256 = fj_test_utils::fixture_id_from_json(&("frankenjax-z6o97", &compiled_out))
            .expect("golden digest");
        assert_eq!(
            sha256, "14e69331aa9141028f837a867c840db5556926f145329ea1a60f8b318b934a58",
            "compiled scalar f64 plan output digest changed"
        );

        let t_generic = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan.last_use,
                    &mut scratch,
                    &mut o,
                )
                .expect("new");
                if let Value::Scalar(Literal::F64Bits(b)) = &o[0] {
                    acc += f64::from_bits(*b);
                }
            }
            std::hint::black_box(acc);
        });

        let t_compiled = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut scalar_buffers = super::ScalarPlanBuffers::default();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_plan_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan,
                    &mut scratch,
                    &mut o,
                    &mut scalar_buffers,
                )
                .expect("compiled");
                if let Value::Scalar(Literal::F64Bits(b)) = &o[0] {
                    acc += f64::from_bits(*b);
                }
            }
            std::hint::black_box(acc);
        });

        println!(
            "BENCH scan-body dispatch {n} evals (3-op f64): GENERIC {:.1}ns/eval -> COMPILED {:.1}ns/eval = {:.2}x sha256={}",
            t_generic * 1e9 / n as f64,
            t_compiled * 1e9 / n as f64,
            t_generic / t_compiled,
            sha256,
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_scalar_unary_plan_overhead() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        // Unary+binary body `abs(neg(x) - y)` (Neg, Sub, Abs) over f64 scalars.
        let body = scalar_unary_body_jaxpr();
        let n: usize = 4_000_000;
        let args = [Value::scalar_f64(2.5), Value::scalar_f64(-1.0)];
        let plan = super::build_dense_plan(&body).expect("dense");

        let t_generic = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan.last_use,
                    &mut scratch,
                    &mut o,
                )
                .expect("generic");
                if let Value::Scalar(Literal::F64Bits(b)) = &o[0] {
                    acc += f64::from_bits(*b);
                }
            }
            std::hint::black_box(acc);
        });

        let t_compiled = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_plan_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan,
                    &mut scratch,
                    &mut o,
                    &mut bufs,
                )
                .expect("compiled");
                if let Value::Scalar(Literal::F64Bits(b)) = &o[0] {
                    acc += f64::from_bits(*b);
                }
            }
            std::hint::black_box(acc);
        });

        println!(
            "BENCH scalar-f64 abs(neg(x)-y) {n} evals (3-op unary+binary): GENERIC {:.1}ns/eval -> COMPILED {:.1}ns/eval = {:.2}x",
            t_generic * 1e9 / n as f64,
            t_compiled * 1e9 / n as f64,
            t_generic / t_compiled,
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_scalar_compare_cond_overhead() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        // The canonical while/scan predicate `x > 0` (i64). With a compiled body
        // the cond is the per-iteration bottleneck; this A/Bs the cond eval alone.
        let body = scalar_compare_body_jaxpr(Primitive::Gt, Literal::I64(0));
        let n: usize = 4_000_000;
        let args = [Value::scalar_i64(3)];
        let plan = super::build_dense_plan(&body).expect("dense");

        let t_generic = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut acc = 0u64;
            for _ in 0..n {
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan.last_use,
                    &mut scratch,
                    &mut o,
                )
                .expect("generic");
                if let Value::Scalar(Literal::Bool(b)) = &o[0] {
                    acc += u64::from(*b);
                }
            }
            std::hint::black_box(acc);
        });

        let t_compiled = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            let mut acc = 0u64;
            for _ in 0..n {
                super::run_dense_plan_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan,
                    &mut scratch,
                    &mut o,
                    &mut bufs,
                )
                .expect("compiled");
                if let Value::Scalar(Literal::Bool(b)) = &o[0] {
                    acc += u64::from(*b);
                }
            }
            std::hint::black_box(acc);
        });

        println!(
            "BENCH scalar cond `x>0` {n} evals: GENERIC {:.1}ns/eval -> COMPILED {:.1}ns/eval = {:.2}x",
            t_generic * 1e9 / n as f64,
            t_compiled * 1e9 / n as f64,
            t_generic / t_compiled,
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_compound_scalar_compare_cond_overhead() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        // Range predicate `0 < x && x < 10`, A/B'd against the generic runner.
        let body = compound_scalar_compare_body_jaxpr(Literal::I64(0), Literal::I64(10));
        let n: usize = 4_000_000;
        let args = [Value::scalar_i64(3)];
        let plan = super::build_dense_plan(&body).expect("dense");

        let t_generic = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut acc = 0u64;
            for _ in 0..n {
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan.last_use,
                    &mut scratch,
                    &mut o,
                )
                .expect("generic");
                if let Value::Scalar(Literal::Bool(b)) = &o[0] {
                    acc += u64::from(*b);
                }
            }
            std::hint::black_box(acc);
        });

        let t_compiled = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            let mut acc = 0u64;
            for _ in 0..n {
                super::run_dense_plan_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan,
                    &mut scratch,
                    &mut o,
                    &mut bufs,
                )
                .expect("compiled");
                if let Value::Scalar(Literal::Bool(b)) = &o[0] {
                    acc += u64::from(*b);
                }
            }
            std::hint::black_box(acc);
        });

        println!(
            "BENCH compound scalar cond `0<x&&x<10` {n} evals: GENERIC {:.1}ns/eval -> PLANNED {:.1}ns/eval = {:.2}x",
            t_generic * 1e9 / n as f64,
            t_compiled * 1e9 / n as f64,
            t_generic / t_compiled,
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_scalar_f64_minmax_plan_overhead() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        // Max/Min sibling A/B: a clamp body (max then min) over f64 scalars.
        let body = scalar_minmax_body_jaxpr();
        let n: usize = 4_000_000;
        let args = [Value::scalar_f64(2.5), Value::scalar_f64(-1.0)];
        let plan = super::build_dense_plan(&body).expect("dense");

        let t_generic = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan.last_use,
                    &mut scratch,
                    &mut o,
                )
                .expect("generic");
                if let Value::Scalar(Literal::F64Bits(b)) = &o[0] {
                    acc += f64::from_bits(*b);
                }
            }
            std::hint::black_box(acc);
        });

        let t_compiled = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            let mut acc = 0.0f64;
            for _ in 0..n {
                super::run_dense_plan_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan,
                    &mut scratch,
                    &mut o,
                    &mut bufs,
                )
                .expect("compiled");
                if let Value::Scalar(Literal::F64Bits(b)) = &o[0] {
                    acc += f64::from_bits(*b);
                }
            }
            std::hint::black_box(acc);
        });

        println!(
            "BENCH scalar-f64 clamp(max,min) {n} evals (2-op): GENERIC {:.1}ns/eval -> COMPILED {:.1}ns/eval = {:.2}x",
            t_generic * 1e9 / n as f64,
            t_compiled * 1e9 / n as f64,
            t_generic / t_compiled,
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_scalar_i64_plan_overhead() {
        use std::time::Instant;
        fn best_time(mut f: impl FnMut()) -> f64 {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        }
        // i64 sibling of the f64 A/B: GENERIC = run_dense_env_into (full generic
        // machinery), COMPILED = the pre-resolved scalar-i64 step plan.
        let body = scalar_f64_arith_body_jaxpr(); // literal-free Mul/Add/Sub
        let n: usize = 4_000_000;
        let args = [Value::scalar_i64(1_000_003), Value::scalar_i64(99)];
        let plan = super::build_dense_plan(&body).expect("dense");

        let t_generic = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut acc = 0i64;
            for _ in 0..n {
                super::run_dense_env_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan.last_use,
                    &mut scratch,
                    &mut o,
                )
                .expect("generic");
                if let Value::Scalar(Literal::I64(v)) = &o[0] {
                    acc = acc.wrapping_add(*v);
                }
            }
            std::hint::black_box(acc);
        });

        let t_compiled = best_time(|| {
            let mut env: Vec<Option<Value>> = vec![None; plan.slots];
            let mut scratch: Vec<Value> = Vec::new();
            let mut o: Vec<Value> = Vec::new();
            let mut bufs = super::ScalarPlanBuffers::default();
            let mut acc = 0i64;
            for _ in 0..n {
                super::run_dense_plan_into(
                    &body,
                    &[],
                    &args,
                    &mut env,
                    &plan,
                    &mut scratch,
                    &mut o,
                    &mut bufs,
                )
                .expect("compiled");
                if let Value::Scalar(Literal::I64(v)) = &o[0] {
                    acc = acc.wrapping_add(*v);
                }
            }
            std::hint::black_box(acc);
        });

        println!(
            "BENCH scalar-i64 dispatch {n} evals (3-op): GENERIC {:.1}ns/eval -> COMPILED {:.1}ns/eval = {:.2}x",
            t_generic * 1e9 / n as f64,
            t_compiled * 1e9 / n as f64,
            t_generic / t_compiled,
        );
    }

    #[test]
    fn eval_while_with_sub_jaxprs_runs_until_predicate_false() {
        let jaxpr = make_while_control_flow_jaxpr(1, 10);
        let outputs = eval_jaxpr(&jaxpr, &[Value::scalar_i64(3)])
            .expect("while with sub_jaxprs should evaluate");
        assert_eq!(outputs, vec![Value::scalar_i64(0)]);
    }

    #[test]
    fn eval_while_with_sub_jaxprs_splits_cond_and_body_consts_from_carry() {
        let jaxpr = make_while_control_flow_with_const_bindings_jaxpr(10);
        let outputs = eval_jaxpr(
            &jaxpr,
            &[
                Value::scalar_i64(0),
                Value::scalar_i64(2),
                Value::scalar_i64(5),
            ],
        )
        .expect("while with sub_jaxpr const bindings should evaluate");
        assert_eq!(outputs, vec![Value::scalar_i64(-1)]);
    }

    #[test]
    fn eval_while_with_sub_jaxprs_enforces_max_iter() {
        let jaxpr = make_while_control_flow_jaxpr(0, 2);
        let err = eval_jaxpr(&jaxpr, &[Value::scalar_i64(3)])
            .expect_err("non-converging while should hit max_iter");
        let max_iterations = match &err {
            InterpreterError::Primitive(fj_lax::EvalError::MaxIterationsExceeded {
                max_iterations,
                ..
            }) => *max_iterations,
            _ => usize::MAX,
        };
        assert_eq!(max_iterations, 2, "unexpected error: {err:?}");
    }

    #[test]
    fn eval_while_with_sub_jaxprs_rejects_body_arity_change() {
        let mut jaxpr = make_while_control_flow_jaxpr(1, 10);
        jaxpr.equations[0].sub_jaxprs[1] =
            Jaxpr::new(vec![VarId(1)], vec![], vec![VarId(1), VarId(1)], vec![]);
        let err = eval_jaxpr(&jaxpr, &[Value::scalar_i64(3)])
            .expect_err("while body arity change should fail");
        let msg = err.to_string();
        assert!(
            msg.contains("returned 2 carry values; expected 1"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn eval_scan_with_sub_jaxprs_returns_final_carry_and_stacked_ys() {
        let jaxpr = make_scan_sub_jaxpr_control_flow_jaxpr(false);
        let outputs = eval_jaxpr(
            &jaxpr,
            &[
                Value::scalar_i64(0),
                Value::vector_i64(&[1, 2, 3]).expect("xs vector should build"),
            ],
        )
        .expect("scan with body sub_jaxpr should evaluate");

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], Value::scalar_i64(6));
        assert_eq!(
            outputs[1],
            Value::vector_i64(&[1, 3, 6]).expect("ys vector should build")
        );
    }

    #[test]
    fn eval_scan_with_sub_jaxprs_reverse_preserves_input_order_ys() {
        let jaxpr = make_scan_sub_jaxpr_control_flow_jaxpr(true);
        let outputs = eval_jaxpr(
            &jaxpr,
            &[
                Value::scalar_i64(0),
                Value::vector_i64(&[1, 2, 3]).expect("xs vector should build"),
            ],
        )
        .expect("reverse scan with body sub_jaxpr should evaluate");

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], Value::scalar_i64(6));
        assert_eq!(
            outputs[1],
            Value::vector_i64(&[6, 5, 3]).expect("ys vector should build")
        );
    }

    #[test]
    fn eval_scan_with_sub_jaxprs_handles_multi_carry_and_ys() {
        let jaxpr = make_scan_multi_carry_control_flow_jaxpr();
        let outputs = eval_jaxpr(
            &jaxpr,
            &[
                Value::scalar_i64(0),
                Value::scalar_i64(1),
                Value::vector_i64(&[1, 2, 3]).expect("xs vector should build"),
            ],
        )
        .expect("multi-carry scan with body sub_jaxpr should evaluate");

        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0], Value::scalar_i64(6));
        assert_eq!(outputs[1], Value::scalar_i64(6));
        assert_eq!(
            outputs[2],
            Value::vector_i64(&[2, 5, 12]).expect("ys vector should build")
        );
    }

    #[test]
    fn eval_scan_f32_add_emit_fast_path_matches_generic_and_golden() {
        let carry = f32_tensor_values(
            vec![4],
            vec![-0.0, 0.0, f32::from_bits(0x7fc0_1234), f32::NEG_INFINITY],
        );
        let xs = f32_tensor_values(
            vec![3, 4],
            vec![
                0.0,
                -0.0,
                1.25,
                f32::INFINITY,
                -3.0,
                2.0,
                f32::NAN,
                -1.0,
                7.5,
                -9.0,
                0.0,
                f32::INFINITY,
            ],
        );
        let mut golden_rows = Vec::new();
        for reverse in [false, true] {
            let fast = make_scan_f32_add_emit_control_flow_jaxpr(reverse);
            let mut generic = fast.clone();
            generic.equations[0].sub_jaxprs[0].equations[1]
                .params
                .insert("force_generic".to_owned(), "1".to_owned());
            let inputs = [carry.clone(), xs.clone()];

            let fast_outputs = eval_jaxpr(&fast, &inputs).expect("fast f32 scan");
            let generic_outputs = eval_jaxpr(&generic, &inputs).expect("generic f32 scan");
            assert_eq!(fast_outputs, generic_outputs, "reverse={reverse}");
            golden_rows.push(fast_outputs);
        }

        let digest = fj_test_utils::fixture_id_from_json(&("frankenjax-0x3pu", &golden_rows))
            .expect("golden digest");
        eprintln!("f32 scan add-emit golden digest: {digest}");
        assert_eq!(
            digest,
            "62c35b04956901a98a2391c073a95a867e0655327dcb5f8063d1fe9a93c7887c"
        );
    }

    #[test]
    fn eval_scan_i64_add_emit_fast_path_matches_generic_forward_and_reverse() {
        for reverse in [false, true] {
            let fast = make_scan_sub_jaxpr_control_flow_jaxpr(reverse);
            let mut generic = fast.clone();
            generic.equations[0].sub_jaxprs[0].equations[1]
                .params
                .insert("force_generic".to_owned(), "1".to_owned());
            let inputs = [
                Value::scalar_i64(i64::MAX),
                Value::vector_i64(&[1, 2, -5, 9]).expect("xs vector should build"),
            ];

            let fast_outputs = eval_jaxpr(&fast, &inputs).expect("fast scan should evaluate");
            let generic_outputs =
                eval_jaxpr(&generic, &inputs).expect("generic scan should evaluate");
            assert_eq!(fast_outputs, generic_outputs, "reverse={reverse}");
        }
    }

    #[test]
    fn eval_scan_i64_add_emit_fast_path_matches_generic_for_scalar_xs() {
        let fast = make_scan_sub_jaxpr_control_flow_jaxpr(false);
        let mut generic = fast.clone();
        generic.equations[0].sub_jaxprs[0].equations[1]
            .params
            .insert("force_generic".to_owned(), "1".to_owned());
        let inputs = [Value::scalar_i64(5), Value::scalar_i64(-8)];

        let fast_outputs = eval_jaxpr(&fast, &inputs).expect("fast scalar scan should evaluate");
        let generic_outputs =
            eval_jaxpr(&generic, &inputs).expect("generic scalar scan should evaluate");
        assert_eq!(fast_outputs, generic_outputs);
        assert_eq!(fast_outputs[0], Value::scalar_i64(-3));
        assert_eq!(
            fast_outputs[1],
            Value::vector_i64(&[-3]).expect("scalar scan y vector should build")
        );
    }

    #[test]
    fn eval_scan_i64_add_emit_fast_path_guard_miss_matches_generic() {
        let mut guard_miss = make_scan_sub_jaxpr_control_flow_jaxpr(false);
        guard_miss.equations[0]
            .params
            .insert("unrelated".to_owned(), "keeps_generic".to_owned());
        let mut forced_generic = guard_miss.clone();
        forced_generic.equations[0].sub_jaxprs[0].equations[1]
            .params
            .insert("force_generic".to_owned(), "1".to_owned());
        let inputs = [
            Value::scalar_i64(10),
            Value::vector_i64(&[4, -7, 9, 2]).expect("xs vector should build"),
        ];

        let guard_miss_outputs =
            eval_jaxpr(&guard_miss, &inputs).expect("guard-miss scan should evaluate");
        let generic_outputs =
            eval_jaxpr(&forced_generic, &inputs).expect("generic scan should evaluate");
        assert_eq!(guard_miss_outputs, generic_outputs);
    }

    #[test]
    fn eval_top_level_scan_i64_add_emit_fast_path_matches_generic_and_golden() {
        let mut golden_rows = Vec::new();
        for reverse in [false, true] {
            let fast = make_scan_sub_jaxpr_control_flow_jaxpr(reverse);
            let mut generic = fast.clone();
            generic.equations[0].sub_jaxprs[0].equations[1]
                .params
                .insert("force_generic".to_owned(), "1".to_owned());
            let inputs = [
                Value::scalar_i64(i64::MAX - 3),
                Value::vector_i64(&[4, -7, i64::MIN, 11]).expect("xs vector should build"),
            ];

            let fast_outputs =
                eval_jaxpr(&fast, &inputs).expect("top-level fast scan should evaluate");
            let generic_outputs =
                eval_jaxpr(&generic, &inputs).expect("generic scan should evaluate");
            assert_eq!(fast_outputs, generic_outputs, "reverse={reverse}");
            golden_rows.push(fast_outputs);
        }

        let digest = fj_test_utils::fixture_id_from_json(&("frankenjax-0jakq", &golden_rows))
            .expect("golden digest");
        eprintln!("top-level i64 scan add-emit golden digest: {digest}");
        assert_eq!(
            digest,
            "2605a56fa3e47ffab29ebca2c7ce5605d03f513c40995213043a4cb123dbc434"
        );
    }

    #[test]
    fn eval_top_level_softmax_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.37).sin() * 8.0 - ((idx % cols) as f64) * 0.125)
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .expect("dense f64 input"),
        );
        let jaxpr = make_softmax_2d_jaxpr(rows, cols);

        let fast = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).expect("fast softmax");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[input]).expect("generic softmax");
        assert_eq!(fast, generic);

        let mut nonfinite_data: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        nonfinite_data[cols + 1] = f64::NAN;
        let nonfinite = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                nonfinite_data,
            )
            .expect("dense nonfinite f64 input"),
        );
        let through_eval = eval_jaxpr(&jaxpr, std::slice::from_ref(&nonfinite))
            .expect("nonfinite softmax falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[nonfinite]).expect("generic nonfinite softmax");
        assert_eq!(through_eval, through_generic);
    }

    #[test]
    fn eval_top_level_log_softmax_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.37).sin() * 8.0 - ((idx % cols) as f64) * 0.125)
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .expect("dense f64 input"),
        );
        let jaxpr = make_log_softmax_2d_jaxpr(rows, cols);

        let fast = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).expect("fast log_softmax");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[input]).expect("generic log_softmax");
        assert_eq!(fast, generic);

        let mut nonfinite_data: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        nonfinite_data[cols + 1] = f64::NAN;
        let nonfinite = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                nonfinite_data,
            )
            .expect("dense nonfinite f64 input"),
        );
        let through_eval = eval_jaxpr(&jaxpr, std::slice::from_ref(&nonfinite))
            .expect("nonfinite log_softmax falls through");
        let through_generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[nonfinite])
            .expect("generic nonfinite log_softmax");
        assert_eq!(through_eval, through_generic);
    }

    #[test]
    fn eval_top_level_layer_norm_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 8usize;
        let cols = 9usize;
        let epsilon = 1.0e-5;
        let data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.29).cos() * 5.0 + ((idx % cols) as f64) * 0.05)
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .expect("dense f64 input"),
        );
        let jaxpr = make_layer_norm_2d_jaxpr(rows, cols, epsilon);

        let fast = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).expect("fast layer norm");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[input]).expect("generic layer norm");
        assert_eq!(fast, generic);

        let mut nonfinite_data: Vec<f64> = (0..rows * cols)
            .map(|idx| (idx as f64 - 11.0) * 0.125)
            .collect();
        nonfinite_data[cols + 2] = f64::NAN;
        let nonfinite = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                nonfinite_data,
            )
            .expect("dense nonfinite f64 input"),
        );
        let through_eval = eval_jaxpr(&jaxpr, std::slice::from_ref(&nonfinite))
            .expect("nonfinite layer norm falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[nonfinite]).expect("generic nonfinite layer norm");
        assert_eq!(through_eval, through_generic);
    }

    fn make_rms_norm_2d_jaxpr(rows: usize, cols: usize, epsilon: f64) -> Jaxpr {
        let x = VarId(1);
        let squared = VarId(2);
        let sq_sum = VarId(3);
        let ms = VarId(4);
        let ms_eps = VarId(5);
        let inv = VarId(6);
        let inv_b = VarId(7);
        let out = VarId(8);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        let bcast = BTreeMap::from([
            ("shape".to_owned(), format!("{rows},{cols}")),
            ("broadcast_dimensions".to_owned(), "0".to_owned()),
        ]);
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::Square,
                    inputs: smallvec![Atom::Var(x)],
                    outputs: smallvec![squared],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(squared)],
                    outputs: smallvec![sq_sum],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(sq_sum), Atom::Lit(Literal::from_f64(cols as f64))],
                    outputs: smallvec![ms],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(ms), Atom::Lit(Literal::from_f64(epsilon))],
                    outputs: smallvec![ms_eps],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Rsqrt,
                    inputs: smallvec![Atom::Var(ms_eps)],
                    outputs: smallvec![inv],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(inv)],
                    outputs: smallvec![inv_b],
                    params: bcast,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(x), Atom::Var(inv_b)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_rms_norm_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 8usize;
        let cols = 9usize;
        let epsilon = 1.0e-5;
        let data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.29).cos() * 5.0 + ((idx % cols) as f64) * 0.05)
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .expect("dense f64 input"),
        );
        let jaxpr = make_rms_norm_2d_jaxpr(rows, cols, epsilon);

        let fast = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).expect("fast rms norm");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[input]).expect("generic rms norm");
        assert_eq!(fast, generic);

        let mut nonfinite_data: Vec<f64> = (0..rows * cols)
            .map(|idx| (idx as f64 - 11.0) * 0.125)
            .collect();
        nonfinite_data[cols + 2] = f64::NAN;
        let nonfinite = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                nonfinite_data,
            )
            .expect("dense nonfinite f64 input"),
        );
        let through_eval = eval_jaxpr(&jaxpr, std::slice::from_ref(&nonfinite))
            .expect("nonfinite rms norm falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[nonfinite]).expect("generic nonfinite rms norm");
        assert_eq!(through_eval, through_generic);
    }

    fn make_euclidean_distance_2d_jaxpr() -> Jaxpr {
        let a = VarId(1);
        let b = VarId(2);
        let diff = VarId(3);
        let sq = VarId(4);
        let s = VarId(5);
        let out = VarId(6);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        Jaxpr::new(
            vec![a, b],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(a), Atom::Var(b)],
                    outputs: smallvec![diff],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(diff), Atom::Var(diff)],
                    outputs: smallvec![sq],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(sq)],
                    outputs: smallvec![s],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sqrt,
                    inputs: smallvec![Atom::Var(s)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    fn make_mean_squared_error_2d_jaxpr(cols: usize) -> Jaxpr {
        let a = VarId(1);
        let b = VarId(2);
        let diff = VarId(3);
        let sq = VarId(4);
        let s = VarId(5);
        let out = VarId(6);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        Jaxpr::new(
            vec![a, b],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(a), Atom::Var(b)],
                    outputs: smallvec![diff],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(diff), Atom::Var(diff)],
                    outputs: smallvec![sq],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(sq)],
                    outputs: smallvec![s],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(s), Atom::Lit(Literal::from_f64(cols as f64))],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    fn make_mean_absolute_error_2d_jaxpr(cols: usize) -> Jaxpr {
        let a = VarId(1);
        let b = VarId(2);
        let diff = VarId(3);
        let abs = VarId(4);
        let s = VarId(5);
        let out = VarId(6);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        Jaxpr::new(
            vec![a, b],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(a), Atom::Var(b)],
                    outputs: smallvec![diff],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Abs,
                    inputs: smallvec![Atom::Var(diff)],
                    outputs: smallvec![abs],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(abs)],
                    outputs: smallvec![s],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(s), Atom::Lit(Literal::from_f64(cols as f64))],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_euclidean_distance_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let a_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.31).sin() * 4.0 - 1.0)
            .collect();
        let b_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.17).cos() * 3.0 + 0.5)
            .collect();
        let dims = vec![rows as u32, cols as u32];
        let a = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_data).expect("a input"),
        );
        let b = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, b_data).expect("b input"),
        );
        let jaxpr = make_euclidean_distance_2d_jaxpr();

        let fast = eval_jaxpr(&jaxpr, &[a.clone(), b.clone()]).expect("fast euclidean");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[a, b]).expect("generic euclidean");
        assert_eq!(fast, generic);

        let mut a_nonfinite: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        a_nonfinite[cols + 1] = f64::NAN;
        let a_nf = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_nonfinite)
                .expect("nonfinite a"),
        );
        let b_ok = Value::Tensor(
            TensorValue::new_f64_values(
                Shape { dims },
                (0..rows * cols).map(|idx| idx as f64 * 0.02).collect(),
            )
            .expect("finite b"),
        );
        let through_eval = eval_jaxpr(&jaxpr, &[a_nf.clone(), b_ok.clone()])
            .expect("nonfinite euclidean falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[a_nf, b_ok]).expect("generic nonfinite euclidean");
        assert_eq!(through_eval, through_generic);
    }

    #[test]
    fn eval_top_level_mean_squared_error_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let a_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.37).sin() * 3.5 - 1.25)
            .collect();
        let b_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.11).cos() * 2.0 + 0.25)
            .collect();
        let dims = vec![rows as u32, cols as u32];
        let a = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_data).expect("a input"),
        );
        let b = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, b_data).expect("b input"),
        );
        let jaxpr = make_mean_squared_error_2d_jaxpr(cols);

        let fast = eval_jaxpr(&jaxpr, &[a.clone(), b.clone()]).expect("fast mse");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[a, b]).expect("generic mse");
        assert_eq!(fast, generic);

        let mut a_nonfinite: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        a_nonfinite[cols + 1] = f64::NAN;
        let a_nf = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_nonfinite)
                .expect("nonfinite a"),
        );
        let b_ok = Value::Tensor(
            TensorValue::new_f64_values(
                Shape { dims },
                (0..rows * cols).map(|idx| idx as f64 * 0.02).collect(),
            )
            .expect("finite b"),
        );
        let through_eval =
            eval_jaxpr(&jaxpr, &[a_nf.clone(), b_ok.clone()]).expect("nonfinite mse falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[a_nf, b_ok]).expect("generic nonfinite mse");
        assert_eq!(through_eval, through_generic);
    }

    #[test]
    fn eval_top_level_mean_absolute_error_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let a_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.23).sin() * 4.25 - 1.75)
            .collect();
        let b_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.17).cos() * 2.75 + 0.5)
            .collect();
        let dims = vec![rows as u32, cols as u32];
        let a = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_data).expect("a input"),
        );
        let b = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, b_data).expect("b input"),
        );
        let jaxpr = make_mean_absolute_error_2d_jaxpr(cols);

        let fast = eval_jaxpr(&jaxpr, &[a.clone(), b.clone()]).expect("fast mae");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[a, b]).expect("generic mae");
        assert_eq!(fast, generic);

        let mut a_nonfinite: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        a_nonfinite[cols + 1] = f64::NAN;
        let a_nf = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_nonfinite)
                .expect("nonfinite a"),
        );
        let b_ok = Value::Tensor(
            TensorValue::new_f64_values(
                Shape { dims },
                (0..rows * cols).map(|idx| idx as f64 * 0.02).collect(),
            )
            .expect("finite b"),
        );
        let through_eval =
            eval_jaxpr(&jaxpr, &[a_nf.clone(), b_ok.clone()]).expect("nonfinite mae falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[a_nf, b_ok]).expect("generic nonfinite mae");
        assert_eq!(through_eval, through_generic);
    }

    fn make_root_mean_squared_error_2d_jaxpr(cols: usize) -> Jaxpr {
        let a = VarId(1);
        let b = VarId(2);
        let diff = VarId(3);
        let sq = VarId(4);
        let s = VarId(5);
        let ms = VarId(6);
        let out = VarId(7);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        Jaxpr::new(
            vec![a, b],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(a), Atom::Var(b)],
                    outputs: smallvec![diff],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(diff), Atom::Var(diff)],
                    outputs: smallvec![sq],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(sq)],
                    outputs: smallvec![s],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(s), Atom::Lit(Literal::from_f64(cols as f64))],
                    outputs: smallvec![ms],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sqrt,
                    inputs: smallvec![Atom::Var(ms)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_root_mean_squared_error_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let a_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.43).sin() * 3.0 - 0.75)
            .collect();
        let b_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.13).cos() * 2.5 + 0.4)
            .collect();
        let dims = vec![rows as u32, cols as u32];
        let a = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_data).expect("a input"),
        );
        let b = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, b_data).expect("b input"),
        );
        let jaxpr = make_root_mean_squared_error_2d_jaxpr(cols);

        let fast = eval_jaxpr(&jaxpr, &[a.clone(), b.clone()]).expect("fast rmse");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[a, b]).expect("generic rmse");
        assert_eq!(fast, generic);

        let mut a_nonfinite: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        a_nonfinite[cols + 1] = f64::NAN;
        let a_nf = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_nonfinite)
                .expect("nonfinite a"),
        );
        let b_ok = Value::Tensor(
            TensorValue::new_f64_values(
                Shape { dims },
                (0..rows * cols).map(|idx| idx as f64 * 0.02).collect(),
            )
            .expect("finite b"),
        );
        let through_eval = eval_jaxpr(&jaxpr, &[a_nf.clone(), b_ok.clone()])
            .expect("nonfinite rmse falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[a_nf, b_ok]).expect("generic nonfinite rmse");
        assert_eq!(through_eval, through_generic);
    }

    fn make_r2_score_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
        let a = VarId(1);
        let b = VarId(2);
        let resid = VarId(3);
        let resid_sq = VarId(4);
        let ss_res = VarId(5);
        let s = VarId(6);
        let mean = VarId(7);
        let mean_b = VarId(8);
        let centered = VarId(9);
        let centered_sq = VarId(10);
        let ss_tot = VarId(11);
        let ratio = VarId(12);
        let out = VarId(13);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        let bcast = BTreeMap::from([
            ("shape".to_owned(), format!("{rows},{cols}")),
            ("broadcast_dimensions".to_owned(), "0".to_owned()),
        ]);
        let n = Literal::from_f64(cols as f64);
        Jaxpr::new(
            vec![a, b],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(a), Atom::Var(b)],
                    outputs: smallvec![resid],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(resid), Atom::Var(resid)],
                    outputs: smallvec![resid_sq],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(resid_sq)],
                    outputs: smallvec![ss_res],
                    params: reduce_axis1.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(a)],
                    outputs: smallvec![s],
                    params: reduce_axis1.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(s), Atom::Lit(n)],
                    outputs: smallvec![mean],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(mean)],
                    outputs: smallvec![mean_b],
                    params: bcast,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(a), Atom::Var(mean_b)],
                    outputs: smallvec![centered],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(centered), Atom::Var(centered)],
                    outputs: smallvec![centered_sq],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(centered_sq)],
                    outputs: smallvec![ss_tot],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(ss_res), Atom::Var(ss_tot)],
                    outputs: smallvec![ratio],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Lit(Literal::from_f64(1.0)), Atom::Var(ratio)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_r2_score_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        // Per-row-varying y_true (no constant row → ss_tot ≠ 0) and an independent y_pred.
        let a_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.41).sin() * 3.0 - 0.6)
            .collect();
        let b_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.19).cos() * 2.2 + 0.35)
            .collect();
        let dims = vec![rows as u32, cols as u32];
        let a = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_data).expect("a input"),
        );
        let b = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, b_data).expect("b input"),
        );
        let jaxpr = make_r2_score_2d_jaxpr(rows, cols);

        let fast = eval_jaxpr(&jaxpr, &[a.clone(), b.clone()]).expect("fast r2");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[a, b]).expect("generic r2");
        assert_eq!(fast, generic);

        let mut a_nonfinite: Vec<f64> = (0..rows * cols)
            .map(|idx| idx as f64 * 0.03 + 0.1)
            .collect();
        a_nonfinite[cols + 1] = f64::NAN;
        let a_nf = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_nonfinite)
                .expect("nonfinite a"),
        );
        let b_ok = Value::Tensor(
            TensorValue::new_f64_values(
                Shape { dims },
                (0..rows * cols).map(|idx| idx as f64 * 0.02).collect(),
            )
            .expect("finite b"),
        );
        let through_eval =
            eval_jaxpr(&jaxpr, &[a_nf.clone(), b_ok.clone()]).expect("nonfinite r2 falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[a_nf, b_ok]).expect("generic nonfinite r2");
        assert_eq!(through_eval, through_generic);
    }

    fn make_manhattan_distance_2d_jaxpr() -> Jaxpr {
        let a = VarId(1);
        let b = VarId(2);
        let diff = VarId(3);
        let abs = VarId(4);
        let out = VarId(5);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        Jaxpr::new(
            vec![a, b],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(a), Atom::Var(b)],
                    outputs: smallvec![diff],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Abs,
                    inputs: smallvec![Atom::Var(diff)],
                    outputs: smallvec![abs],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(abs)],
                    outputs: smallvec![out],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_manhattan_distance_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let a_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.19).sin() * 5.0 - 2.0)
            .collect();
        let b_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.23).cos() * 2.5 + 0.75)
            .collect();
        let dims = vec![rows as u32, cols as u32];
        let a = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_data).expect("a input"),
        );
        let b = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, b_data).expect("b input"),
        );
        let jaxpr = make_manhattan_distance_2d_jaxpr();

        let fast = eval_jaxpr(&jaxpr, &[a.clone(), b.clone()]).expect("fast manhattan");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[a, b]).expect("generic manhattan");
        assert_eq!(fast, generic);

        let mut a_nonfinite: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        a_nonfinite[cols + 1] = f64::NAN;
        let a_nf = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_nonfinite)
                .expect("nonfinite a"),
        );
        let b_ok = Value::Tensor(
            TensorValue::new_f64_values(
                Shape { dims },
                (0..rows * cols).map(|idx| idx as f64 * 0.02).collect(),
            )
            .expect("finite b"),
        );
        let through_eval = eval_jaxpr(&jaxpr, &[a_nf.clone(), b_ok.clone()])
            .expect("nonfinite manhattan falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[a_nf, b_ok]).expect("generic nonfinite manhattan");
        assert_eq!(through_eval, through_generic);
    }

    fn make_entropy_2d_jaxpr() -> Jaxpr {
        let p = VarId(1);
        let log_p = VarId(2);
        let weighted = VarId(3);
        let summed = VarId(4);
        let out = VarId(5);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        Jaxpr::new(
            vec![p],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::Log,
                    inputs: smallvec![Atom::Var(p)],
                    outputs: smallvec![log_p],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(p), Atom::Var(log_p)],
                    outputs: smallvec![weighted],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(weighted)],
                    outputs: smallvec![summed],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(summed)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_entropy_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        // entropy needs positive p (log p) for finite outputs; use strictly-positive data.
        let p_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.29).sin() * 0.4 + 1.3)
            .collect();
        let p = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                p_data,
            )
            .expect("p input"),
        );
        let jaxpr = make_entropy_2d_jaxpr();

        let fast = eval_jaxpr(&jaxpr, std::slice::from_ref(&p)).expect("fast entropy");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[p]).expect("generic entropy");
        assert_eq!(fast, generic);

        let mut p_nonfinite: Vec<f64> = (0..rows * cols)
            .map(|idx| idx as f64 * 0.01 + 1.0)
            .collect();
        p_nonfinite[cols + 1] = f64::NAN;
        let p_nf = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                p_nonfinite,
            )
            .expect("nonfinite p"),
        );
        let through_eval = eval_jaxpr(&jaxpr, std::slice::from_ref(&p_nf))
            .expect("nonfinite entropy falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[p_nf]).expect("generic nonfinite entropy");
        assert_eq!(through_eval, through_generic);
    }

    fn make_kl_divergence_2d_jaxpr() -> Jaxpr {
        let p = VarId(1);
        let q = VarId(2);
        let ratio = VarId(3);
        let log_ratio = VarId(4);
        let weighted = VarId(5);
        let out = VarId(6);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        Jaxpr::new(
            vec![p, q],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(p), Atom::Var(q)],
                    outputs: smallvec![ratio],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Log,
                    inputs: smallvec![Atom::Var(ratio)],
                    outputs: smallvec![log_ratio],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(p), Atom::Var(log_ratio)],
                    outputs: smallvec![weighted],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(weighted)],
                    outputs: smallvec![out],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_kl_divergence_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        // KL needs positive p, q (p/q > 0) for finite outputs; use strictly-positive data.
        let p_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.31).sin() * 0.4 + 1.5)
            .collect();
        let q_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.17).cos() * 0.3 + 1.2)
            .collect();
        let dims = vec![rows as u32, cols as u32];
        let p = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, p_data).expect("p input"),
        );
        let q = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, q_data).expect("q input"),
        );
        let jaxpr = make_kl_divergence_2d_jaxpr();

        let fast = eval_jaxpr(&jaxpr, &[p.clone(), q.clone()]).expect("fast kl");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[p, q]).expect("generic kl");
        assert_eq!(fast, generic);

        let mut p_nonfinite: Vec<f64> = (0..rows * cols)
            .map(|idx| idx as f64 * 0.01 + 1.0)
            .collect();
        p_nonfinite[cols + 1] = f64::NAN;
        let p_nf = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, p_nonfinite)
                .expect("nonfinite p"),
        );
        let q_ok = Value::Tensor(
            TensorValue::new_f64_values(
                Shape { dims },
                (0..rows * cols)
                    .map(|idx| idx as f64 * 0.02 + 1.0)
                    .collect(),
            )
            .expect("finite q"),
        );
        let through_eval =
            eval_jaxpr(&jaxpr, &[p_nf.clone(), q_ok.clone()]).expect("nonfinite kl falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[p_nf, q_ok]).expect("generic nonfinite kl");
        assert_eq!(through_eval, through_generic);
    }

    fn make_pearson_correlation_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
        let a = VarId(1);
        let b = VarId(2);
        let sa = VarId(3);
        let ma = VarId(4);
        let ma_b = VarId(5);
        let ca = VarId(6);
        let sb = VarId(7);
        let mb = VarId(8);
        let mb_b = VarId(9);
        let cb = VarId(10);
        let cov_prod = VarId(11);
        let cov = VarId(12);
        let va_prod = VarId(13);
        let va = VarId(14);
        let na = VarId(15);
        let vb_prod = VarId(16);
        let vb = VarId(17);
        let nb = VarId(18);
        let denom = VarId(19);
        let out = VarId(20);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        let bcast = BTreeMap::from([
            ("shape".to_owned(), format!("{rows},{cols}")),
            ("broadcast_dimensions".to_owned(), "0".to_owned()),
        ]);
        let n = Literal::from_f64(cols as f64);
        let unary = |prim: Primitive, input: VarId, o: VarId| Equation {
            primitive: prim,
            inputs: smallvec![Atom::Var(input)],
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let binary = |prim: Primitive, l: Atom, r: Atom, o: VarId| Equation {
            primitive: prim,
            inputs: smallvec![l, r],
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let reduce = |input: VarId, o: VarId| Equation {
            primitive: Primitive::ReduceSum,
            inputs: smallvec![Atom::Var(input)],
            outputs: smallvec![o],
            params: reduce_axis1.clone(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let broadcast = |input: VarId, o: VarId| Equation {
            primitive: Primitive::BroadcastInDim,
            inputs: smallvec![Atom::Var(input)],
            outputs: smallvec![o],
            params: bcast.clone(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        Jaxpr::new(
            vec![a, b],
            vec![],
            vec![out],
            vec![
                reduce(a, sa),
                binary(Primitive::Div, Atom::Var(sa), Atom::Lit(n), ma),
                broadcast(ma, ma_b),
                binary(Primitive::Sub, Atom::Var(a), Atom::Var(ma_b), ca),
                reduce(b, sb),
                binary(Primitive::Div, Atom::Var(sb), Atom::Lit(n), mb),
                broadcast(mb, mb_b),
                binary(Primitive::Sub, Atom::Var(b), Atom::Var(mb_b), cb),
                binary(Primitive::Mul, Atom::Var(ca), Atom::Var(cb), cov_prod),
                reduce(cov_prod, cov),
                binary(Primitive::Mul, Atom::Var(ca), Atom::Var(ca), va_prod),
                reduce(va_prod, va),
                unary(Primitive::Sqrt, va, na),
                binary(Primitive::Mul, Atom::Var(cb), Atom::Var(cb), vb_prod),
                reduce(vb_prod, vb),
                unary(Primitive::Sqrt, vb, nb),
                binary(Primitive::Mul, Atom::Var(na), Atom::Var(nb), denom),
                binary(Primitive::Div, Atom::Var(cov), Atom::Var(denom), out),
            ],
        )
    }

    #[test]
    fn eval_top_level_pearson_correlation_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let a_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.29).sin() * 4.0 - 1.0)
            .collect();
        let b_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.13).cos() * 2.5 + 0.75)
            .collect();
        let dims = vec![rows as u32, cols as u32];
        let a = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_data).expect("a input"),
        );
        let b = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, b_data).expect("b input"),
        );
        let jaxpr = make_pearson_correlation_2d_jaxpr(rows, cols);

        let fast = eval_jaxpr(&jaxpr, &[a.clone(), b.clone()]).expect("fast pearson");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[a, b]).expect("generic pearson");
        assert_eq!(fast, generic);

        let mut a_nonfinite: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        a_nonfinite[cols + 1] = f64::NAN;
        let a_nf = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_nonfinite)
                .expect("nonfinite a"),
        );
        let b_ok = Value::Tensor(
            TensorValue::new_f64_values(
                Shape { dims },
                (0..rows * cols)
                    .map(|idx| idx as f64 * 0.02 + 0.1)
                    .collect(),
            )
            .expect("finite b"),
        );
        let through_eval = eval_jaxpr(&jaxpr, &[a_nf.clone(), b_ok.clone()])
            .expect("nonfinite pearson falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[a_nf, b_ok]).expect("generic nonfinite pearson");
        assert_eq!(through_eval, through_generic);
    }

    fn make_cosine_similarity_2d_jaxpr() -> Jaxpr {
        let a = VarId(1);
        let b = VarId(2);
        let ab = VarId(3);
        let dot = VarId(4);
        let a2 = VarId(5);
        let sa = VarId(6);
        let na = VarId(7);
        let b2 = VarId(8);
        let sb = VarId(9);
        let nb = VarId(10);
        let denom = VarId(11);
        let out = VarId(12);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        let mul = |lhs: VarId, rhs: VarId, o: VarId| Equation {
            primitive: Primitive::Mul,
            inputs: smallvec![Atom::Var(lhs), Atom::Var(rhs)],
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let reduce = |input: VarId, o: VarId| Equation {
            primitive: Primitive::ReduceSum,
            inputs: smallvec![Atom::Var(input)],
            outputs: smallvec![o],
            params: reduce_axis1.clone(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let sqrt = |input: VarId, o: VarId| Equation {
            primitive: Primitive::Sqrt,
            inputs: smallvec![Atom::Var(input)],
            outputs: smallvec![o],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        Jaxpr::new(
            vec![a, b],
            vec![],
            vec![out],
            vec![
                mul(a, b, ab),
                reduce(ab, dot),
                mul(a, a, a2),
                reduce(a2, sa),
                sqrt(sa, na),
                mul(b, b, b2),
                reduce(b2, sb),
                sqrt(sb, nb),
                mul(na, nb, denom),
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(dot), Atom::Var(denom)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_cosine_similarity_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let a_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.37).sin() * 3.0 + 0.5)
            .collect();
        let b_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.21).cos() * 2.0 - 0.75)
            .collect();
        let dims = vec![rows as u32, cols as u32];
        let a = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_data).expect("a input"),
        );
        let b = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, b_data).expect("b input"),
        );
        let jaxpr = make_cosine_similarity_2d_jaxpr();

        let fast = eval_jaxpr(&jaxpr, &[a.clone(), b.clone()]).expect("fast cosine");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[a, b]).expect("generic cosine");
        assert_eq!(fast, generic);

        let mut a_nonfinite: Vec<f64> = (0..rows * cols)
            .map(|idx| idx as f64 * 0.01 + 0.1)
            .collect();
        a_nonfinite[cols + 1] = f64::NAN;
        let a_nf = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_nonfinite)
                .expect("nonfinite a"),
        );
        let b_ok = Value::Tensor(
            TensorValue::new_f64_values(
                Shape { dims },
                (0..rows * cols)
                    .map(|idx| idx as f64 * 0.02 + 0.3)
                    .collect(),
            )
            .expect("finite b"),
        );
        let through_eval = eval_jaxpr(&jaxpr, &[a_nf.clone(), b_ok.clone()])
            .expect("nonfinite cosine falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[a_nf, b_ok]).expect("generic nonfinite cosine");
        assert_eq!(through_eval, through_generic);
    }

    fn make_l2_normalize_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
        let x = VarId(1);
        let sq = VarId(2);
        let s = VarId(3);
        let norm = VarId(4);
        let norm_b = VarId(5);
        let out = VarId(6);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        let bcast = BTreeMap::from([
            ("shape".to_owned(), format!("{rows},{cols}")),
            ("broadcast_dimensions".to_owned(), "0".to_owned()),
        ]);
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(x), Atom::Var(x)],
                    outputs: smallvec![sq],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(sq)],
                    outputs: smallvec![s],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sqrt,
                    inputs: smallvec![Atom::Var(s)],
                    outputs: smallvec![norm],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(norm)],
                    outputs: smallvec![norm_b],
                    params: bcast,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(x), Atom::Var(norm_b)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_l2_normalize_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.31).cos() * 3.0 - ((idx % cols) as f64) * 0.45 + 0.2)
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .expect("dense f64 input"),
        );
        let jaxpr = make_l2_normalize_2d_jaxpr(rows, cols);

        let fast = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).expect("fast l2_normalize");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[input]).expect("generic l2_normalize");
        assert_eq!(fast, generic);

        let mut nonfinite_data: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        nonfinite_data[cols + 1] = f64::NAN;
        let nonfinite = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                nonfinite_data,
            )
            .expect("dense nonfinite f64 input"),
        );
        let through_eval = eval_jaxpr(&jaxpr, std::slice::from_ref(&nonfinite))
            .expect("nonfinite l2_normalize falls through");
        let through_generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[nonfinite])
            .expect("generic nonfinite l2_normalize");
        assert_eq!(through_eval, through_generic);
    }

    fn make_zscore_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
        let x = VarId(1);
        let s1 = VarId(2);
        let mean = VarId(3);
        let mean_b = VarId(4);
        let centered = VarId(5);
        let sq = VarId(6);
        let s2 = VarId(7);
        let var = VarId(8);
        let std = VarId(9);
        let std_b = VarId(10);
        let out = VarId(11);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        let bcast = BTreeMap::from([
            ("shape".to_owned(), format!("{rows},{cols}")),
            ("broadcast_dimensions".to_owned(), "0".to_owned()),
        ]);
        let n = Literal::from_f64(cols as f64);
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(x)],
                    outputs: smallvec![s1],
                    params: reduce_axis1.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(s1), Atom::Lit(n)],
                    outputs: smallvec![mean],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(mean)],
                    outputs: smallvec![mean_b],
                    params: bcast.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(x), Atom::Var(mean_b)],
                    outputs: smallvec![centered],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(centered), Atom::Var(centered)],
                    outputs: smallvec![sq],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(sq)],
                    outputs: smallvec![s2],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(s2), Atom::Lit(n)],
                    outputs: smallvec![var],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sqrt,
                    inputs: smallvec![Atom::Var(var)],
                    outputs: smallvec![std],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(std)],
                    outputs: smallvec![std_b],
                    params: bcast,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(centered), Atom::Var(std_b)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_zscore_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.39).sin() * 3.5 - ((idx % cols) as f64) * 0.5 + 0.3)
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .expect("dense f64 input"),
        );
        let jaxpr = make_zscore_2d_jaxpr(rows, cols);

        let fast = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).expect("fast zscore");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[input]).expect("generic zscore");
        assert_eq!(fast, generic);

        let mut nonfinite_data: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        nonfinite_data[cols + 1] = f64::NAN;
        let nonfinite = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                nonfinite_data,
            )
            .expect("dense nonfinite f64 input"),
        );
        let through_eval = eval_jaxpr(&jaxpr, std::slice::from_ref(&nonfinite))
            .expect("nonfinite zscore falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[nonfinite]).expect("generic nonfinite zscore");
        assert_eq!(through_eval, through_generic);
    }

    fn make_minmax_normalize_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
        let x = VarId(1);
        let mx = VarId(2);
        let mx_b = VarId(3);
        let mn = VarId(4);
        let mn_b = VarId(5);
        let num = VarId(6);
        let den = VarId(7);
        let out = VarId(8);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        let bcast = BTreeMap::from([
            ("shape".to_owned(), format!("{rows},{cols}")),
            ("broadcast_dimensions".to_owned(), "0".to_owned()),
        ]);
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::ReduceMax,
                    inputs: smallvec![Atom::Var(x)],
                    outputs: smallvec![mx],
                    params: reduce_axis1.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(mx)],
                    outputs: smallvec![mx_b],
                    params: bcast.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceMin,
                    inputs: smallvec![Atom::Var(x)],
                    outputs: smallvec![mn],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(mn)],
                    outputs: smallvec![mn_b],
                    params: bcast,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(x), Atom::Var(mn_b)],
                    outputs: smallvec![num],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(mx_b), Atom::Var(mn_b)],
                    outputs: smallvec![den],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(num), Atom::Var(den)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_minmax_normalize_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        // Mixed-sign, per-row-varying data (no constant row → no 0/0), plus ±0 in one row to exercise
        // the max(+0,−0)=+0 / min(+0,−0)=−0 conventions shared by the fused pass and the folds.
        let mut data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.47).cos() * 2.5 - ((idx % cols) as f64) * 0.4 + 0.15)
            .collect();
        data[0] = 0.0;
        data[1] = -0.0;
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .expect("dense f64 input"),
        );
        let jaxpr = make_minmax_normalize_2d_jaxpr(rows, cols);

        let fast = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).expect("fast minmax");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[input]).expect("generic minmax");
        assert_eq!(fast, generic);

        let mut nonfinite_data: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        nonfinite_data[cols + 1] = f64::NAN;
        let nonfinite = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                nonfinite_data,
            )
            .expect("dense nonfinite f64 input"),
        );
        let through_eval = eval_jaxpr(&jaxpr, std::slice::from_ref(&nonfinite))
            .expect("nonfinite minmax falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[nonfinite]).expect("generic nonfinite minmax");
        assert_eq!(through_eval, through_generic);
    }

    fn make_rms_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
        let x = VarId(1);
        let sq = VarId(2);
        let s = VarId(3);
        let ms = VarId(4);
        let out = VarId(5);
        let _ = rows;
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        let n = Literal::from_f64(cols as f64);
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(x), Atom::Var(x)],
                    outputs: smallvec![sq],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(sq)],
                    outputs: smallvec![s],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(s), Atom::Lit(n)],
                    outputs: smallvec![ms],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sqrt,
                    inputs: smallvec![Atom::Var(ms)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_rms_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.47).cos() * 3.0 - ((idx % cols) as f64) * 0.35)
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .expect("dense f64 input"),
        );
        let jaxpr = make_rms_2d_jaxpr(rows, cols);

        let fast = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).expect("fast rms");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[input]).expect("generic rms");
        assert_eq!(fast, generic);

        let mut nonfinite_data: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        nonfinite_data[cols + 1] = f64::NAN;
        let nonfinite = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                nonfinite_data,
            )
            .expect("dense nonfinite f64 input"),
        );
        let through_eval = eval_jaxpr(&jaxpr, std::slice::from_ref(&nonfinite))
            .expect("nonfinite rms falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[nonfinite]).expect("generic nonfinite rms");
        assert_eq!(through_eval, through_generic);
    }

    fn make_l1_norm_2d_jaxpr() -> Jaxpr {
        // Abs(x) -> ReduceSum(axis=1); no shape-bearing params, shape lives on the input.
        let x = VarId(1);
        let a = VarId(2);
        let out = VarId(3);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::Abs,
                    inputs: smallvec![Atom::Var(x)],
                    outputs: smallvec![a],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(a)],
                    outputs: smallvec![out],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_l1_norm_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.41).sin() * 3.5 - ((idx % cols) as f64) * 0.6)
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .expect("dense f64 input"),
        );
        let jaxpr = make_l1_norm_2d_jaxpr();

        let fast = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).expect("fast l1_norm");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[input]).expect("generic l1_norm");
        assert_eq!(fast, generic);

        let mut nonfinite_data: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        nonfinite_data[cols + 1] = f64::NAN;
        let nonfinite = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                nonfinite_data,
            )
            .expect("dense nonfinite f64 input"),
        );
        let through_eval = eval_jaxpr(&jaxpr, std::slice::from_ref(&nonfinite))
            .expect("nonfinite l1_norm falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[nonfinite]).expect("generic nonfinite l1_norm");
        assert_eq!(through_eval, through_generic);
    }

    fn make_l2_norm_2d_jaxpr() -> Jaxpr {
        // The L2-norm graph carries no shape-bearing Div/Broadcast params; shape lives on the input.
        let x = VarId(1);
        let sq = VarId(2);
        let s = VarId(3);
        let out = VarId(4);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(x), Atom::Var(x)],
                    outputs: smallvec![sq],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(sq)],
                    outputs: smallvec![s],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sqrt,
                    inputs: smallvec![Atom::Var(s)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_l2_norm_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.29).sin() * 4.0 - ((idx % cols) as f64) * 0.5)
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .expect("dense f64 input"),
        );
        let jaxpr = make_l2_norm_2d_jaxpr();

        let fast = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).expect("fast l2_norm");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[input]).expect("generic l2_norm");
        assert_eq!(fast, generic);

        let mut nonfinite_data: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        nonfinite_data[cols + 1] = f64::NAN;
        let nonfinite = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                nonfinite_data,
            )
            .expect("dense nonfinite f64 input"),
        );
        let through_eval = eval_jaxpr(&jaxpr, std::slice::from_ref(&nonfinite))
            .expect("nonfinite l2_norm falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[nonfinite]).expect("generic nonfinite l2_norm");
        assert_eq!(through_eval, through_generic);
    }

    fn make_covariance_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
        let a = VarId(1);
        let b = VarId(2);
        let suma = VarId(3);
        let mean_a = VarId(4);
        let mean_a_b = VarId(5);
        let sumb = VarId(6);
        let mean_b = VarId(7);
        let mean_b_b = VarId(8);
        let ca = VarId(9);
        let cb = VarId(10);
        let prod = VarId(11);
        let s = VarId(12);
        let out = VarId(13);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        let bcast = BTreeMap::from([
            ("shape".to_owned(), format!("{rows},{cols}")),
            ("broadcast_dimensions".to_owned(), "0".to_owned()),
        ]);
        let n = Literal::from_f64(cols as f64);
        Jaxpr::new(
            vec![a, b],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(a)],
                    outputs: smallvec![suma],
                    params: reduce_axis1.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(suma), Atom::Lit(n)],
                    outputs: smallvec![mean_a],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(mean_a)],
                    outputs: smallvec![mean_a_b],
                    params: bcast.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(b)],
                    outputs: smallvec![sumb],
                    params: reduce_axis1.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(sumb), Atom::Lit(n)],
                    outputs: smallvec![mean_b],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(mean_b)],
                    outputs: smallvec![mean_b_b],
                    params: bcast,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(a), Atom::Var(mean_a_b)],
                    outputs: smallvec![ca],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(b), Atom::Var(mean_b_b)],
                    outputs: smallvec![cb],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(ca), Atom::Var(cb)],
                    outputs: smallvec![prod],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(prod)],
                    outputs: smallvec![s],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(s), Atom::Lit(n)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_covariance_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let a_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.37).sin() * 3.0 + 0.5)
            .collect();
        let b_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.21).cos() * 2.0 - 0.75)
            .collect();
        let dims = vec![rows as u32, cols as u32];
        let a = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_data).expect("a input"),
        );
        let b = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, b_data).expect("b input"),
        );
        let jaxpr = make_covariance_2d_jaxpr(rows, cols);

        let fast = eval_jaxpr(&jaxpr, &[a.clone(), b.clone()]).expect("fast covariance");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[a, b]).expect("generic covariance");
        assert_eq!(fast, generic);

        let mut a_nonfinite: Vec<f64> = (0..rows * cols)
            .map(|idx| idx as f64 * 0.01 + 0.1)
            .collect();
        a_nonfinite[cols + 1] = f64::NAN;
        let a_nf = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_nonfinite)
                .expect("nonfinite a"),
        );
        let b_ok = Value::Tensor(
            TensorValue::new_f64_values(
                Shape { dims },
                (0..rows * cols)
                    .map(|idx| idx as f64 * 0.02 + 0.3)
                    .collect(),
            )
            .expect("finite b"),
        );
        let through_eval = eval_jaxpr(&jaxpr, &[a_nf.clone(), b_ok.clone()])
            .expect("nonfinite covariance falls through");
        let through_generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[a_nf, b_ok])
            .expect("generic nonfinite covariance");
        assert_eq!(through_eval, through_generic);
    }

    fn make_var_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
        let x = VarId(1);
        let s1 = VarId(2);
        let mean = VarId(3);
        let mean_b = VarId(4);
        let centered = VarId(5);
        let sq = VarId(6);
        let s2 = VarId(7);
        let out = VarId(8);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        let bcast = BTreeMap::from([
            ("shape".to_owned(), format!("{rows},{cols}")),
            ("broadcast_dimensions".to_owned(), "0".to_owned()),
        ]);
        let n = Literal::from_f64(cols as f64);
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(x)],
                    outputs: smallvec![s1],
                    params: reduce_axis1.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(s1), Atom::Lit(n)],
                    outputs: smallvec![mean],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(mean)],
                    outputs: smallvec![mean_b],
                    params: bcast,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(x), Atom::Var(mean_b)],
                    outputs: smallvec![centered],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(centered), Atom::Var(centered)],
                    outputs: smallvec![sq],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(sq)],
                    outputs: smallvec![s2],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(s2), Atom::Lit(n)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_var_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.43).sin() * 5.0 - ((idx % cols) as f64) * 0.3)
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .expect("dense f64 input"),
        );
        let jaxpr = make_var_2d_jaxpr(rows, cols);

        let fast = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).expect("fast var");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[input]).expect("generic var");
        assert_eq!(fast, generic);

        let mut nonfinite_data: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        nonfinite_data[cols + 1] = f64::NAN;
        let nonfinite = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                nonfinite_data,
            )
            .expect("dense nonfinite f64 input"),
        );
        let through_eval = eval_jaxpr(&jaxpr, std::slice::from_ref(&nonfinite))
            .expect("nonfinite var falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[nonfinite]).expect("generic nonfinite var");
        assert_eq!(through_eval, through_generic);
    }

    fn make_std_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
        // The variance graph, but its final Div lands in `var` and a trailing Sqrt yields the output.
        let x = VarId(1);
        let s1 = VarId(2);
        let mean = VarId(3);
        let mean_b = VarId(4);
        let centered = VarId(5);
        let sq = VarId(6);
        let s2 = VarId(7);
        let var = VarId(8);
        let out = VarId(9);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        let bcast = BTreeMap::from([
            ("shape".to_owned(), format!("{rows},{cols}")),
            ("broadcast_dimensions".to_owned(), "0".to_owned()),
        ]);
        let n = Literal::from_f64(cols as f64);
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(x)],
                    outputs: smallvec![s1],
                    params: reduce_axis1.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(s1), Atom::Lit(n)],
                    outputs: smallvec![mean],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(mean)],
                    outputs: smallvec![mean_b],
                    params: bcast,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(x), Atom::Var(mean_b)],
                    outputs: smallvec![centered],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(centered), Atom::Var(centered)],
                    outputs: smallvec![sq],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(sq)],
                    outputs: smallvec![s2],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(s2), Atom::Lit(n)],
                    outputs: smallvec![var],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sqrt,
                    inputs: smallvec![Atom::Var(var)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_std_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.43).sin() * 5.0 - ((idx % cols) as f64) * 0.3)
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .expect("dense f64 input"),
        );
        let jaxpr = make_std_2d_jaxpr(rows, cols);

        let fast = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).expect("fast std");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[input]).expect("generic std");
        assert_eq!(fast, generic);

        let mut nonfinite_data: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        nonfinite_data[cols + 1] = f64::NAN;
        let nonfinite = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                nonfinite_data,
            )
            .expect("dense nonfinite f64 input"),
        );
        let through_eval = eval_jaxpr(&jaxpr, std::slice::from_ref(&nonfinite))
            .expect("nonfinite std falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[nonfinite]).expect("generic nonfinite std");
        assert_eq!(through_eval, through_generic);
    }

    fn make_skewness_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
        let x = VarId(1);
        let s1 = VarId(2);
        let mean = VarId(3);
        let mean_b = VarId(4);
        let d = VarId(5);
        let d2 = VarId(6);
        let d3 = VarId(7);
        let sum2 = VarId(8);
        let m2 = VarId(9);
        let sum3 = VarId(10);
        let m3 = VarId(11);
        let s = VarId(12);
        let denom = VarId(13);
        let out = VarId(14);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        let bcast = BTreeMap::from([
            ("shape".to_owned(), format!("{rows},{cols}")),
            ("broadcast_dimensions".to_owned(), "0".to_owned()),
        ]);
        let n = Literal::from_f64(cols as f64);
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(x)],
                    outputs: smallvec![s1],
                    params: reduce_axis1.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(s1), Atom::Lit(n)],
                    outputs: smallvec![mean],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(mean)],
                    outputs: smallvec![mean_b],
                    params: bcast,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(x), Atom::Var(mean_b)],
                    outputs: smallvec![d],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(d), Atom::Var(d)],
                    outputs: smallvec![d2],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(d2), Atom::Var(d)],
                    outputs: smallvec![d3],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(d2)],
                    outputs: smallvec![sum2],
                    params: reduce_axis1.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(sum2), Atom::Lit(n)],
                    outputs: smallvec![m2],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(d3)],
                    outputs: smallvec![sum3],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(sum3), Atom::Lit(n)],
                    outputs: smallvec![m3],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sqrt,
                    inputs: smallvec![Atom::Var(m2)],
                    outputs: smallvec![s],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(m2), Atom::Var(s)],
                    outputs: smallvec![denom],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(m3), Atom::Var(denom)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_skewness_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.61).sin() * 4.0 + ((idx % cols) as f64) * 0.7)
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .expect("dense f64 input"),
        );
        let jaxpr = make_skewness_2d_jaxpr(rows, cols);

        let fast = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).expect("fast skewness");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[input]).expect("generic skewness");
        assert_eq!(fast, generic);

        let mut nonfinite_data: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        nonfinite_data[cols + 1] = f64::NAN;
        let nonfinite = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                nonfinite_data,
            )
            .expect("dense nonfinite f64 input"),
        );
        let through_eval = eval_jaxpr(&jaxpr, std::slice::from_ref(&nonfinite))
            .expect("nonfinite skewness falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[nonfinite]).expect("generic nonfinite skewness");
        assert_eq!(through_eval, through_generic);
    }

    fn make_kurtosis_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
        let x = VarId(1);
        let s1 = VarId(2);
        let mean = VarId(3);
        let mean_b = VarId(4);
        let d = VarId(5);
        let d2 = VarId(6);
        let d4 = VarId(7);
        let sum2 = VarId(8);
        let m2 = VarId(9);
        let sum4 = VarId(10);
        let m4 = VarId(11);
        let denom = VarId(12);
        let out = VarId(13);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        let bcast = BTreeMap::from([
            ("shape".to_owned(), format!("{rows},{cols}")),
            ("broadcast_dimensions".to_owned(), "0".to_owned()),
        ]);
        let n = Literal::from_f64(cols as f64);
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(x)],
                    outputs: smallvec![s1],
                    params: reduce_axis1.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(s1), Atom::Lit(n)],
                    outputs: smallvec![mean],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(mean)],
                    outputs: smallvec![mean_b],
                    params: bcast,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(x), Atom::Var(mean_b)],
                    outputs: smallvec![d],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(d), Atom::Var(d)],
                    outputs: smallvec![d2],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(d2), Atom::Var(d2)],
                    outputs: smallvec![d4],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(d2)],
                    outputs: smallvec![sum2],
                    params: reduce_axis1.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(sum2), Atom::Lit(n)],
                    outputs: smallvec![m2],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(d4)],
                    outputs: smallvec![sum4],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(sum4), Atom::Lit(n)],
                    outputs: smallvec![m4],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(m2), Atom::Var(m2)],
                    outputs: smallvec![denom],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(m4), Atom::Var(denom)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_kurtosis_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.53).cos() * 3.5 - ((idx % cols) as f64) * 0.4)
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .expect("dense f64 input"),
        );
        let jaxpr = make_kurtosis_2d_jaxpr(rows, cols);

        let fast = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).expect("fast kurtosis");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[input]).expect("generic kurtosis");
        assert_eq!(fast, generic);

        let mut nonfinite_data: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        nonfinite_data[cols + 1] = f64::NAN;
        let nonfinite = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                nonfinite_data,
            )
            .expect("dense nonfinite f64 input"),
        );
        let through_eval = eval_jaxpr(&jaxpr, std::slice::from_ref(&nonfinite))
            .expect("nonfinite kurtosis falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[nonfinite]).expect("generic nonfinite kurtosis");
        assert_eq!(through_eval, through_generic);
    }

    fn make_logsumexp_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
        let x = VarId(1);
        let max = VarId(2);
        let max_b = VarId(3);
        let shifted = VarId(4);
        let exp = VarId(5);
        let sum = VarId(6);
        let logv = VarId(7);
        let out = VarId(8);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        let bcast = BTreeMap::from([
            ("shape".to_owned(), format!("{rows},{cols}")),
            ("broadcast_dimensions".to_owned(), "0".to_owned()),
        ]);
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::ReduceMax,
                    inputs: smallvec![Atom::Var(x)],
                    outputs: smallvec![max],
                    params: reduce_axis1.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(max)],
                    outputs: smallvec![max_b],
                    params: bcast,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(x), Atom::Var(max_b)],
                    outputs: smallvec![shifted],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Exp,
                    inputs: smallvec![Atom::Var(shifted)],
                    outputs: smallvec![exp],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(exp)],
                    outputs: smallvec![sum],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Log,
                    inputs: smallvec![Atom::Var(sum)],
                    outputs: smallvec![logv],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(max), Atom::Var(logv)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_logsumexp_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.33).sin() * 6.0 - ((idx % cols) as f64) * 0.25)
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .expect("dense f64 input"),
        );
        let jaxpr = make_logsumexp_2d_jaxpr(rows, cols);

        let fast = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).expect("fast logsumexp");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[input]).expect("generic logsumexp");
        assert_eq!(fast, generic);

        let mut nonfinite_data: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        nonfinite_data[cols + 1] = f64::NAN;
        let nonfinite = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                nonfinite_data,
            )
            .expect("dense nonfinite f64 input"),
        );
        let through_eval = eval_jaxpr(&jaxpr, std::slice::from_ref(&nonfinite))
            .expect("nonfinite logsumexp falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[nonfinite]).expect("generic nonfinite logsumexp");
        assert_eq!(through_eval, through_generic);
    }

    fn make_softmax_cross_entropy_2d_jaxpr(rows: usize, cols: usize) -> Jaxpr {
        let logits = VarId(1);
        let labels = VarId(2);
        let max = VarId(3);
        let max_b = VarId(4);
        let shifted = VarId(5);
        let exp = VarId(6);
        let sum = VarId(7);
        let logv = VarId(8);
        let log_b = VarId(9);
        let ls = VarId(10);
        let prod = VarId(11);
        let ce_sum = VarId(12);
        let out = VarId(13);
        let reduce_axis1 = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        let bcast = BTreeMap::from([
            ("shape".to_owned(), format!("{rows},{cols}")),
            ("broadcast_dimensions".to_owned(), "0".to_owned()),
        ]);
        Jaxpr::new(
            vec![logits, labels],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::ReduceMax,
                    inputs: smallvec![Atom::Var(logits)],
                    outputs: smallvec![max],
                    params: reduce_axis1.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(max)],
                    outputs: smallvec![max_b],
                    params: bcast.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(logits), Atom::Var(max_b)],
                    outputs: smallvec![shifted],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Exp,
                    inputs: smallvec![Atom::Var(shifted)],
                    outputs: smallvec![exp],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(exp)],
                    outputs: smallvec![sum],
                    params: reduce_axis1.clone(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Log,
                    inputs: smallvec![Atom::Var(sum)],
                    outputs: smallvec![logv],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(logv)],
                    outputs: smallvec![log_b],
                    params: bcast,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Sub,
                    inputs: smallvec![Atom::Var(shifted), Atom::Var(log_b)],
                    outputs: smallvec![ls],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(labels), Atom::Var(ls)],
                    outputs: smallvec![prod],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(prod)],
                    outputs: smallvec![ce_sum],
                    params: reduce_axis1,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(ce_sum)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_softmax_cross_entropy_2d_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let logits_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.37).sin() * 8.0 - ((idx % cols) as f64) * 0.125)
            .collect();
        let labels_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.19).cos().abs() * 0.3)
            .collect();
        let dims = vec![rows as u32, cols as u32];
        let logits = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, logits_data)
                .expect("logits input"),
        );
        let labels = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, labels_data)
                .expect("labels input"),
        );
        let jaxpr = make_softmax_cross_entropy_2d_jaxpr(rows, cols);

        let fast =
            eval_jaxpr(&jaxpr, &[logits.clone(), labels.clone()]).expect("fast cross entropy");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[logits.clone(), labels.clone()])
            .expect("generic cross entropy");
        assert_eq!(fast, generic);

        let mut nonfinite_data: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        nonfinite_data[cols + 1] = f64::NAN;
        let logits_nf = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims }, nonfinite_data)
                .expect("nonfinite logits input"),
        );
        let through_eval = eval_jaxpr(&jaxpr, &[logits_nf.clone(), labels.clone()])
            .expect("nonfinite cross entropy falls through");
        let through_generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[logits_nf, labels])
            .expect("generic nonfinite cross entropy");
        assert_eq!(through_eval, through_generic);
    }

    fn make_gelu_erf_jaxpr() -> Jaxpr {
        let x = VarId(1);
        let arg = VarId(2);
        let e = VarId(3);
        let one_plus = VarId(4);
        let half_x = VarId(5);
        let out = VarId(6);
        let sqrt2 = Literal::from_f64(2.0_f64.sqrt());
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(x), Atom::Lit(sqrt2)],
                    outputs: smallvec![arg],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Erf,
                    inputs: smallvec![Atom::Var(arg)],
                    outputs: smallvec![e],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Lit(Literal::from_f64(1.0)), Atom::Var(e)],
                    outputs: smallvec![one_plus],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(x), Atom::Lit(Literal::from_f64(0.5))],
                    outputs: smallvec![half_x],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(half_x), Atom::Var(one_plus)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_gelu_erf_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.37).sin() * 3.0 - 1.0)
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .expect("dense f64 input"),
        );
        let jaxpr = make_gelu_erf_jaxpr();

        let fast = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).expect("fast gelu");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[input]).expect("generic gelu");
        assert_eq!(fast, generic);

        let mut nonfinite_data: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        nonfinite_data[cols + 1] = f64::NAN;
        let nonfinite = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                nonfinite_data,
            )
            .expect("dense nonfinite f64 input"),
        );
        let through_eval = eval_jaxpr(&jaxpr, std::slice::from_ref(&nonfinite))
            .expect("nonfinite gelu falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[nonfinite]).expect("generic nonfinite gelu");
        assert_eq!(through_eval, through_generic);
    }

    fn make_log_sigmoid_jaxpr() -> Jaxpr {
        let x = VarId(1);
        let nx = VarId(2);
        let e = VarId(3);
        let one_plus = VarId(4);
        let sp = VarId(5);
        let out = VarId(6);
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(x)],
                    outputs: smallvec![nx],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Exp,
                    inputs: smallvec![Atom::Var(nx)],
                    outputs: smallvec![e],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Lit(Literal::from_f64(1.0)), Atom::Var(e)],
                    outputs: smallvec![one_plus],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Log,
                    inputs: smallvec![Atom::Var(one_plus)],
                    outputs: smallvec![sp],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(sp)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_log_sigmoid_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.41).cos() * 4.0 - 1.5)
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .expect("dense f64 input"),
        );
        let jaxpr = make_log_sigmoid_jaxpr();

        let fast = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).expect("fast log_sigmoid");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[input]).expect("generic log_sigmoid");
        assert_eq!(fast, generic);

        let mut nonfinite_data: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        nonfinite_data[cols + 1] = f64::NAN;
        let nonfinite = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                nonfinite_data,
            )
            .expect("dense nonfinite f64 input"),
        );
        let through_eval = eval_jaxpr(&jaxpr, std::slice::from_ref(&nonfinite))
            .expect("nonfinite log_sigmoid falls through");
        let through_generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[nonfinite])
            .expect("generic nonfinite log_sigmoid");
        assert_eq!(through_eval, through_generic);
    }

    fn make_softplus_jaxpr() -> Jaxpr {
        let x = VarId(1);
        let e = VarId(2);
        let one_plus = VarId(3);
        let out = VarId(4);
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::Exp,
                    inputs: smallvec![Atom::Var(x)],
                    outputs: smallvec![e],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Lit(Literal::from_f64(1.0)), Atom::Var(e)],
                    outputs: smallvec![one_plus],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Log,
                    inputs: smallvec![Atom::Var(one_plus)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_softplus_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.37).sin() * 4.0 - 1.0)
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .expect("dense f64 input"),
        );
        let jaxpr = make_softplus_jaxpr();

        let fast = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).expect("fast softplus");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[input]).expect("generic softplus");
        assert_eq!(fast, generic);

        let mut nonfinite_data: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        nonfinite_data[cols + 1] = f64::NAN;
        let nonfinite = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                nonfinite_data,
            )
            .expect("dense nonfinite f64 input"),
        );
        let through_eval = eval_jaxpr(&jaxpr, std::slice::from_ref(&nonfinite))
            .expect("nonfinite softplus falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[nonfinite]).expect("generic nonfinite softplus");
        assert_eq!(through_eval, through_generic);
    }

    fn make_silu_jaxpr() -> Jaxpr {
        let x = VarId(1);
        let nx = VarId(2);
        let e = VarId(3);
        let one_plus = VarId(4);
        let out = VarId(5);
        Jaxpr::new(
            vec![x],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(x)],
                    outputs: smallvec![nx],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Exp,
                    inputs: smallvec![Atom::Var(nx)],
                    outputs: smallvec![e],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Lit(Literal::from_f64(1.0)), Atom::Var(e)],
                    outputs: smallvec![one_plus],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(x), Atom::Var(one_plus)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_silu_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.29).sin() * 5.0 - 2.0)
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .expect("dense f64 input"),
        );
        let jaxpr = make_silu_jaxpr();

        let fast = eval_jaxpr(&jaxpr, std::slice::from_ref(&input)).expect("fast silu");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[input]).expect("generic silu");
        assert_eq!(fast, generic);

        let mut nonfinite_data: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        nonfinite_data[cols + 1] = f64::NAN;
        let nonfinite = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                nonfinite_data,
            )
            .expect("dense nonfinite f64 input"),
        );
        let through_eval = eval_jaxpr(&jaxpr, std::slice::from_ref(&nonfinite))
            .expect("nonfinite silu falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[nonfinite]).expect("generic nonfinite silu");
        assert_eq!(through_eval, through_generic);
    }

    fn make_swiglu_jaxpr() -> Jaxpr {
        let a = VarId(1);
        let b = VarId(2);
        let na = VarId(3);
        let e = VarId(4);
        let one_plus = VarId(5);
        let silu_a = VarId(6);
        let out = VarId(7);
        Jaxpr::new(
            vec![a, b],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::Neg,
                    inputs: smallvec![Atom::Var(a)],
                    outputs: smallvec![na],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Exp,
                    inputs: smallvec![Atom::Var(na)],
                    outputs: smallvec![e],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Lit(Literal::from_f64(1.0)), Atom::Var(e)],
                    outputs: smallvec![one_plus],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(a), Atom::Var(one_plus)],
                    outputs: smallvec![silu_a],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(silu_a), Atom::Var(b)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_swiglu_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let a_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.31).sin() * 5.0 - 2.0)
            .collect();
        let b_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.23).cos() * 3.0 + 0.5)
            .collect();
        let dims = vec![rows as u32, cols as u32];
        let a = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_data).expect("gate input"),
        );
        let b = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, b_data).expect("value input"),
        );
        let jaxpr = make_swiglu_jaxpr();

        let fast = eval_jaxpr(&jaxpr, &[a.clone(), b.clone()]).expect("fast swiglu");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[a, b]).expect("generic swiglu");
        assert_eq!(fast, generic);

        let mut a_nonfinite: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        a_nonfinite[cols + 1] = f64::NAN;
        let a_nf = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_nonfinite)
                .expect("nonfinite gate"),
        );
        let b_ok = Value::Tensor(
            TensorValue::new_f64_values(
                Shape { dims },
                (0..rows * cols).map(|idx| idx as f64 * 0.02).collect(),
            )
            .expect("finite value"),
        );
        let through_eval = eval_jaxpr(&jaxpr, &[a_nf.clone(), b_ok.clone()])
            .expect("nonfinite swiglu falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[a_nf, b_ok]).expect("generic nonfinite swiglu");
        assert_eq!(through_eval, through_generic);
    }

    fn make_geglu_jaxpr() -> Jaxpr {
        let a = VarId(1);
        let b = VarId(2);
        let arg = VarId(3);
        let e = VarId(4);
        let one_plus = VarId(5);
        let half_a = VarId(6);
        let gelu_a = VarId(7);
        let out = VarId(8);
        let sqrt2 = Literal::from_f64(2.0_f64.sqrt());
        Jaxpr::new(
            vec![a, b],
            vec![],
            vec![out],
            vec![
                Equation {
                    primitive: Primitive::Div,
                    inputs: smallvec![Atom::Var(a), Atom::Lit(sqrt2)],
                    outputs: smallvec![arg],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Erf,
                    inputs: smallvec![Atom::Var(arg)],
                    outputs: smallvec![e],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Lit(Literal::from_f64(1.0)), Atom::Var(e)],
                    outputs: smallvec![one_plus],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(a), Atom::Lit(Literal::from_f64(0.5))],
                    outputs: smallvec![half_a],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(half_a), Atom::Var(one_plus)],
                    outputs: smallvec![gelu_a],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(gelu_a), Atom::Var(b)],
                    outputs: smallvec![out],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        )
    }

    #[test]
    fn eval_top_level_geglu_f64_matches_generic_and_preserves_edges() {
        let rows = 9usize;
        let cols = 7usize;
        let a_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.27).sin() * 4.0 - 1.5)
            .collect();
        let b_data: Vec<f64> = (0..rows * cols)
            .map(|idx| ((idx as f64) * 0.19).cos() * 2.0 + 0.25)
            .collect();
        let dims = vec![rows as u32, cols as u32];
        let a = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_data).expect("gate input"),
        );
        let b = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, b_data).expect("value input"),
        );
        let jaxpr = make_geglu_jaxpr();

        let fast = eval_jaxpr(&jaxpr, &[a.clone(), b.clone()]).expect("fast geglu");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[a, b]).expect("generic geglu");
        assert_eq!(fast, generic);

        let mut a_nonfinite: Vec<f64> = (0..rows * cols).map(|idx| idx as f64 * 0.01).collect();
        a_nonfinite[cols + 1] = f64::NAN;
        let a_nf = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, a_nonfinite)
                .expect("nonfinite gate"),
        );
        let b_ok = Value::Tensor(
            TensorValue::new_f64_values(
                Shape { dims },
                (0..rows * cols).map(|idx| idx as f64 * 0.02).collect(),
            )
            .expect("finite value"),
        );
        let through_eval = eval_jaxpr(&jaxpr, &[a_nf.clone(), b_ok.clone()])
            .expect("nonfinite geglu falls through");
        let through_generic =
            eval_jaxpr_hashed_env(&jaxpr, &[], &[a_nf, b_ok]).expect("generic nonfinite geglu");
        assert_eq!(through_eval, through_generic);
    }

    #[test]
    fn eval_scan_i64_add_emit_fast_path_accepts_literal_backed_i64_xs() {
        let jaxpr = make_scan_sub_jaxpr_control_flow_jaxpr(false);
        let literal_xs = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape::vector(4),
                vec![
                    Literal::I64(4),
                    Literal::I64(-7),
                    Literal::I64(9),
                    Literal::I64(2),
                ],
            )
            .expect("literal-backed xs should build"),
        );
        let outputs =
            eval_jaxpr(&jaxpr, &[Value::scalar_i64(10), literal_xs]).expect("scan should evaluate");

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0], Value::scalar_i64(18));
        assert_eq!(
            outputs[1],
            Value::vector_i64(&[14, 7, 16, 18]).expect("ys vector should build")
        );
    }

    #[test]
    fn eval_cond_with_sub_jaxprs_selects_true_and_false_branches() {
        let jaxpr = make_cond_control_flow_jaxpr();
        let cases = [(true, 5_i64, 10_i64), (false, 5, 25)];
        for (predicate, operand, expected) in cases {
            let outputs = eval_jaxpr(
                &jaxpr,
                &[Value::scalar_bool(predicate), Value::scalar_i64(operand)],
            )
            .expect("cond with sub_jaxprs should evaluate");
            assert_eq!(outputs, vec![Value::scalar_i64(expected)]);
        }
    }

    #[test]
    fn eval_cond_with_sub_jaxprs_splits_branch_consts_from_args() {
        let jaxpr = make_cond_with_const_binding_jaxpr();
        let outputs = eval_jaxpr(
            &jaxpr,
            &[
                Value::scalar_i64(1),
                Value::scalar_i64(10),
                Value::scalar_i64(7),
            ],
        )
        .expect("truthy cond should use true branch");
        assert_eq!(outputs, vec![Value::scalar_i64(17)]);

        let outputs = eval_jaxpr(
            &jaxpr,
            &[
                Value::scalar_i64(0),
                Value::scalar_i64(10),
                Value::scalar_i64(7),
            ],
        )
        .expect("falsey cond should use false branch");
        assert_eq!(outputs, vec![Value::scalar_i64(70)]);
    }

    #[test]
    fn eval_cond_with_sub_jaxprs_rejects_missing_branch_operand() {
        let mut jaxpr = make_cond_control_flow_jaxpr();
        jaxpr.equations[0].inputs = smallvec![Atom::Var(VarId(1))];
        let err = eval_jaxpr(&jaxpr, &[Value::scalar_bool(true), Value::scalar_i64(5)])
            .expect_err("missing branch operand should fail");
        assert_eq!(
            err,
            InterpreterError::InputArity {
                expected: 1,
                actual: 0,
            }
        );
    }

    #[test]
    fn eval_cond_with_complex_predicate_rejects_predicate_dtype() {
        let jaxpr = make_cond_control_flow_jaxpr();
        let err = eval_jaxpr(
            &jaxpr,
            &[
                Value::Scalar(Literal::Complex128Bits(
                    1.0_f64.to_bits(),
                    0.0_f64.to_bits(),
                )),
                Value::scalar_i64(5),
            ],
        )
        .expect_err("complex cond predicate should fail");
        let msg = err.to_string();
        assert!(
            msg.contains("predicate must be boolean or numeric"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn eval_switch_with_sub_jaxprs_selects_the_requested_branch() {
        let jaxpr = make_switch_control_flow_jaxpr();
        let cases = [(0_i64, 5_i64, 5_i64), (1, 5, 10), (2, 5, 25)];
        for (branch_idx, operand, expected) in cases {
            let outputs = eval_jaxpr(
                &jaxpr,
                &[Value::scalar_i64(branch_idx), Value::scalar_i64(operand)],
            )
            .expect("switch with sub_jaxprs should evaluate");
            assert_eq!(outputs, vec![Value::scalar_i64(expected)]);
        }
    }

    #[test]
    fn eval_switch_with_tensor_scalar_index_selects_branch() {
        let jaxpr = make_switch_control_flow_jaxpr();
        let index = Value::Tensor(
            TensorValue::new(DType::I64, Shape::scalar(), vec![Literal::I64(2)]).unwrap(),
        );
        let outputs =
            eval_jaxpr(&jaxpr, &[index, Value::scalar_i64(5)]).expect("switch should evaluate");
        assert_eq!(outputs, vec![Value::scalar_i64(25)]);
    }

    #[test]
    fn eval_switch_with_sub_jaxprs_clamps_out_of_bounds_indices() {
        let jaxpr = make_switch_control_flow_jaxpr();

        let high_outputs = eval_jaxpr(&jaxpr, &[Value::scalar_i64(3), Value::scalar_i64(5)])
            .expect("high switch index should clamp to the last branch");
        assert_eq!(high_outputs, vec![Value::scalar_i64(25)]);

        let low_outputs = eval_jaxpr(&jaxpr, &[Value::scalar_i64(-1), Value::scalar_i64(5)])
            .expect("negative switch index should clamp to the first branch");
        assert_eq!(low_outputs, vec![Value::scalar_i64(5)]);
    }

    #[test]
    fn test_interpreters_test_log_schema_contract() {
        let fixture_id =
            fj_test_utils::fixture_id_from_json(&("interp", "add2")).expect("fixture digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_interpreters_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }

    // ── Broader primitive coverage through interpreter ──────

    fn make_unary_jaxpr(prim: Primitive) -> Jaxpr {
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
    }

    fn make_binary_jaxpr(prim: Primitive) -> Jaxpr {
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
    }

    #[test]
    fn compiled_jaxpr_eval_matches_eager_eval_jaxpr() {
        // The jit compiled fast path (CompiledJaxpr::eval -> run_dense_plan) must produce
        // results IDENTICAL to eager eval_jaxpr, which ADDITIONALLY applies special
        // pre-plan fast paths (scalar i64 add-chain, scalar half arith) that the compiled
        // path skips. This guards that those fast paths agree with the dense plan — a
        // divergence would make jit(f) silently != f. Pure internal-consistency invariant,
        // no JAX oracle needed.
        let check = |label: &str, jaxpr: &Jaxpr, args: &[Value]| {
            let eager = eval_jaxpr(jaxpr, args).expect("eager eval");
            let compiled_jaxpr =
                super::compile_jaxpr_for_repeated_eval(jaxpr).expect("program should compile");
            let compiled = compiled_jaxpr.eval(args).expect("compiled eval");
            assert_eq!(compiled, eager, "{label}: compiled jit != eager eval_jaxpr");
            let runner = compiled_jaxpr
                .runner()
                .eval_owned(args)
                .expect("compiled runner eval");
            assert_eq!(
                runner, eager,
                "{label}: compiled runner != eager eval_jaxpr"
            );
            // The vectorized dense-f64 inner loop must be bit-identical to the generic
            // per-element loop it replaces — assert directly, not only via eager.
            let mut r = compiled_jaxpr.runner();
            let vectorized = r.eval(args).expect("vectorized eval").to_vec();
            let scalar_inner = r
                .eval_scalar_inner(args)
                .expect("scalar-inner eval")
                .to_vec();
            assert_eq!(
                vectorized, scalar_inner,
                "{label}: vectorized != scalar-inner dense-f64 loop"
            );
        };

        // i64 add chain f(x) = (x + x) + x — triggers eager's scalar_i64_add_chain fast
        // path; the compiled path runs the dense plan instead, so this checks they agree.
        let add = |a: VarId, b: VarId, out: VarId| Equation {
            primitive: Primitive::Add,
            inputs: smallvec![Atom::Var(a), Atom::Var(b)],
            outputs: smallvec![out],
            params: BTreeMap::new(),
            sub_jaxprs: vec![],
            effects: vec![],
        };
        let chain = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                add(VarId(1), VarId(1), VarId(2)),
                add(VarId(2), VarId(1), VarId(3)),
            ],
        );
        check("i64_add_chain", &chain, &[Value::scalar_i64(7)]);
        check("i64_add_chain_neg", &chain, &[Value::scalar_i64(-3)]);
        check(
            "i64_add_chain_overflowy",
            &chain,
            &[Value::scalar_i64(i64::MAX - 1)],
        );

        // scalar f64 / i64 single ops.
        check(
            "f64_mul",
            &make_binary_jaxpr(Primitive::Mul),
            &[Value::scalar_f64(3.0), Value::scalar_f64(7.0)],
        );
        check(
            "i64_sub",
            &make_binary_jaxpr(Primitive::Sub),
            &[Value::scalar_i64(10), Value::scalar_i64(3)],
        );

        // tensor elementwise add (dense plan path, no special fast path).
        let t = |v: Vec<f64>| {
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape {
                        dims: vec![v.len() as u32],
                    },
                    v.into_iter().map(Literal::from_f64).collect(),
                )
                .unwrap(),
            )
        };
        check(
            "tensor_add",
            &make_binary_jaxpr(Primitive::Add),
            &[t(vec![1.0, 2.0, 3.0]), t(vec![10.0, 20.0, 30.0])],
        );

        // Scalar transcendentals (unary): the compiled dense plan must match eager for
        // each, not just for add/mul/sub. A divergence here would make jit(exp) != exp.
        check(
            "f64_exp",
            &make_unary_jaxpr(Primitive::Exp),
            &[Value::scalar_f64(1.5)],
        );
        check(
            "f64_log",
            &make_unary_jaxpr(Primitive::Log),
            &[Value::scalar_f64(2.0)],
        );
        check(
            "f64_sqrt",
            &make_unary_jaxpr(Primitive::Sqrt),
            &[Value::scalar_f64(4.0)],
        );
        check(
            "f64_sin",
            &make_unary_jaxpr(Primitive::Sin),
            &[Value::scalar_f64(0.7)],
        );

        // More tensor binops (f64).
        check(
            "tensor_mul",
            &make_binary_jaxpr(Primitive::Mul),
            &[t(vec![1.5, -2.0, 3.0]), t(vec![4.0, 5.0, -6.0])],
        );
        check(
            "tensor_sub",
            &make_binary_jaxpr(Primitive::Sub),
            &[t(vec![10.0, 20.0, 30.0]), t(vec![1.0, 2.0, 3.0])],
        );

        // Non-f64 tensor dtypes through the compiled plan (f32 is JAX's default; bf16
        // is the dominant training dtype) — exercise dtype-specific dense plan paths.
        let tf32 = |v: Vec<f32>| {
            Value::Tensor(
                TensorValue::new(
                    DType::F32,
                    Shape {
                        dims: vec![v.len() as u32],
                    },
                    v.into_iter().map(Literal::from_f32).collect(),
                )
                .unwrap(),
            )
        };
        check(
            "tensor_f32_add",
            &make_binary_jaxpr(Primitive::Add),
            &[tf32(vec![1.0, 2.0, 3.0]), tf32(vec![0.5, 0.25, -1.0])],
        );
        let tbf16 = |v: Vec<f32>| {
            Value::Tensor(
                TensorValue::new(
                    DType::BF16,
                    Shape {
                        dims: vec![v.len() as u32],
                    },
                    v.into_iter().map(Literal::from_bf16_f32).collect(),
                )
                .unwrap(),
            )
        };
        check(
            "tensor_bf16_mul",
            &make_binary_jaxpr(Primitive::Mul),
            &[tbf16(vec![1.5, 2.0, -0.5]), tbf16(vec![2.0, 0.5, 4.0])],
        );
    }

    #[test]
    fn eval_neg_scalar() {
        let jaxpr = make_unary_jaxpr(Primitive::Neg);
        let out = eval_jaxpr(&jaxpr, &[Value::scalar_f64(5.0)]).unwrap();
        assert!((out[0].as_f64_scalar().unwrap() - (-5.0)).abs() < 1e-12);
    }

    #[test]
    fn eval_abs_negative() {
        let jaxpr = make_unary_jaxpr(Primitive::Abs);
        let out = eval_jaxpr(&jaxpr, &[Value::scalar_f64(-7.0)]).unwrap();
        assert!((out[0].as_f64_scalar().unwrap() - 7.0).abs() < 1e-12);
    }

    #[test]
    fn eval_equation_single_matches_eval_jaxpr() {
        // The two public eval entry points must agree on a single-output equation:
        // eval_equation_single(eqn, env) == eval_jaxpr(single-eqn jaxpr, args)[0].
        // Cross-validates the per-equation path against the full interpreter — a
        // divergence would make callers of eval_equation_single silently differ.
        let check = |label: &str, jaxpr: &Jaxpr, args: &[Value]| {
            let via_jaxpr = eval_jaxpr(jaxpr, args).expect("eval_jaxpr");
            assert_eq!(via_jaxpr.len(), 1, "{label}: single output expected");
            let mut env: rustc_hash::FxHashMap<VarId, Value> = rustc_hash::FxHashMap::default();
            for (var, val) in jaxpr.invars.iter().zip(args) {
                env.insert(*var, val.clone());
            }
            let via_single = crate::eval_equation_single(&jaxpr.equations[0], &env)
                .expect("eval_equation_single");
            assert_eq!(
                via_single, via_jaxpr[0],
                "{label}: eval_equation_single != eval_jaxpr"
            );
        };
        check(
            "f64_add",
            &make_binary_jaxpr(Primitive::Add),
            &[Value::scalar_f64(2.0), Value::scalar_f64(3.0)],
        );
        check(
            "f64_mul",
            &make_binary_jaxpr(Primitive::Mul),
            &[Value::scalar_f64(2.5), Value::scalar_f64(4.0)],
        );
        check(
            "i64_sub",
            &make_binary_jaxpr(Primitive::Sub),
            &[Value::scalar_i64(10), Value::scalar_i64(3)],
        );
        check(
            "f64_exp",
            &make_unary_jaxpr(Primitive::Exp),
            &[Value::scalar_f64(1.0)],
        );
        check(
            "f64_neg",
            &make_unary_jaxpr(Primitive::Neg),
            &[Value::scalar_f64(5.0)],
        );
        check(
            "f64_abs",
            &make_unary_jaxpr(Primitive::Abs),
            &[Value::scalar_f64(-7.0)],
        );
    }

    #[test]
    fn eval_exp_scalar() {
        let jaxpr = make_unary_jaxpr(Primitive::Exp);
        let out = eval_jaxpr(&jaxpr, &[Value::scalar_f64(1.0)]).unwrap();
        assert!((out[0].as_f64_scalar().unwrap() - std::f64::consts::E).abs() < 1e-10);
    }

    #[test]
    fn eval_log_scalar() {
        let jaxpr = make_unary_jaxpr(Primitive::Log);
        let out = eval_jaxpr(&jaxpr, &[Value::scalar_f64(std::f64::consts::E)]).unwrap();
        assert!((out[0].as_f64_scalar().unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn eval_sin_cos_identity() {
        // sin^2(x) + cos^2(x) = 1
        let x = 1.5;
        let sin_jaxpr = make_unary_jaxpr(Primitive::Sin);
        let cos_jaxpr = make_unary_jaxpr(Primitive::Cos);
        let sin_val = eval_jaxpr(&sin_jaxpr, &[Value::scalar_f64(x)]).unwrap()[0]
            .as_f64_scalar()
            .unwrap();
        let cos_val = eval_jaxpr(&cos_jaxpr, &[Value::scalar_f64(x)]).unwrap()[0]
            .as_f64_scalar()
            .unwrap();
        assert!((sin_val * sin_val + cos_val * cos_val - 1.0).abs() < 1e-12);
    }

    #[test]
    fn eval_sqrt_scalar() {
        let jaxpr = make_unary_jaxpr(Primitive::Sqrt);
        let out = eval_jaxpr(&jaxpr, &[Value::scalar_f64(25.0)]).unwrap();
        assert!((out[0].as_f64_scalar().unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn eval_mul_scalar() {
        let jaxpr = make_binary_jaxpr(Primitive::Mul);
        let out = eval_jaxpr(&jaxpr, &[Value::scalar_f64(3.0), Value::scalar_f64(7.0)]).unwrap();
        assert!((out[0].as_f64_scalar().unwrap() - 21.0).abs() < 1e-12);
    }

    #[test]
    fn eval_sub_scalar() {
        let jaxpr = make_binary_jaxpr(Primitive::Sub);
        let out = eval_jaxpr(&jaxpr, &[Value::scalar_i64(10), Value::scalar_i64(3)]).unwrap();
        assert_eq!(out[0].as_i64_scalar().unwrap(), 7);
    }

    #[test]
    fn eval_max_min_scalar() {
        let max_jaxpr = make_binary_jaxpr(Primitive::Max);
        let min_jaxpr = make_binary_jaxpr(Primitive::Min);
        let a = Value::scalar_f64(3.0);
        let b = Value::scalar_f64(7.0);
        let max_out = eval_jaxpr(&max_jaxpr, &[a.clone(), b.clone()]).unwrap();
        let min_out = eval_jaxpr(&min_jaxpr, &[a, b]).unwrap();
        assert!((max_out[0].as_f64_scalar().unwrap() - 7.0).abs() < 1e-12);
        assert!((min_out[0].as_f64_scalar().unwrap() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn eval_chain_neg_exp() {
        // f(x) = exp(neg(x)) = exp(-x)
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
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Exp,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        );
        let out = eval_jaxpr(&jaxpr, &[Value::scalar_f64(2.0)]).unwrap();
        let expected = (-2.0_f64).exp();
        assert!((out[0].as_f64_scalar().unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn eval_literal_input_equation() {
        // f(x) = x + 10 where 10 is a literal
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(10))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let out = eval_jaxpr(&jaxpr, &[Value::scalar_i64(5)]).unwrap();
        assert_eq!(out[0].as_i64_scalar().unwrap(), 15);
    }

    #[test]
    fn eval_vector_neg() {
        let jaxpr = make_unary_jaxpr(Primitive::Neg);
        let input = Value::vector_f64(&[1.0, -2.0, 3.0]).unwrap();
        let out = eval_jaxpr(&jaxpr, &[input]).unwrap();
        let t = out[0].as_tensor().unwrap();
        let vals = t.to_f64_vec().unwrap();
        assert_eq!(vals, vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn eval_cholesky_through_interpreter() {
        let jaxpr = build_program(ProgramSpec::LaxCholesky);
        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::from_f64(4.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                ],
            )
            .unwrap(),
        );
        let outputs = eval_jaxpr(&jaxpr, &[input]).unwrap();
        assert!(!outputs.is_empty());
        let l = outputs[0].as_tensor().unwrap();
        assert_eq!(l.shape, Shape { dims: vec![2, 2] });
    }

    #[test]
    fn eval_error_display() {
        let err = InterpreterError::InputArity {
            expected: 2,
            actual: 1,
        };
        assert!(err.to_string().contains("input arity mismatch"));

        let err = InterpreterError::MissingVariable(VarId(42));
        assert!(err.to_string().contains("v42"));

        let err = InterpreterError::UnexpectedOutputArity {
            primitive: Primitive::Add,
            expected: 1,
            actual: 2,
        };
        assert!(err.to_string().contains("add"));

        let err = InterpreterError::InvariantViolation {
            detail: "scheduler stalled".to_owned(),
        };
        assert!(err.to_string().contains("scheduler stalled"));
    }

    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(proptest::test_runner::Config::with_cases(
                fj_test_utils::property_test_case_count()
            ))]
            #[test]
            fn prop_interpreters_add_commutative(
                a in -1_000_000i64..1_000_000,
                b in -1_000_000i64..1_000_000
            ) {
                let _seed = fj_test_utils::capture_proptest_seed();
                let jaxpr = build_program(ProgramSpec::Add2);
                let out_ab = eval_jaxpr(&jaxpr, &[Value::scalar_i64(a), Value::scalar_i64(b)])
                    .expect("add should succeed");
                let out_ba = eval_jaxpr(&jaxpr, &[Value::scalar_i64(b), Value::scalar_i64(a)])
                    .expect("add should succeed");
                prop_assert_eq!(out_ab, out_ba);
            }

            #[test]
            fn prop_interpreters_add_one_total(a in -1_000_000i64..1_000_000) {
                let jaxpr = build_program(ProgramSpec::AddOne);
                let result = eval_jaxpr(&jaxpr, &[Value::scalar_i64(a)]);
                prop_assert!(result.is_ok());
            }

            #[test]
            fn prop_scalar_i64_add_chain_fast_path_matches_hashed(
                start in any::<i64>(),
                addends in prop::collection::vec(any::<i64>(), 1..32)
            ) {
                let jaxpr = scalar_i64_add_literal_chain(&addends);
                let args = [Value::scalar_i64(start)];
                let fast = eval_jaxpr(&jaxpr, &args).expect("fast path should evaluate");
                let hashed = eval_jaxpr_hashed_env(&jaxpr, &[], &args)
                    .expect("hashed path should evaluate");
                let expected = addends
                    .iter()
                    .fold(start, |acc, addend| acc.wrapping_add(*addend));
                prop_assert_eq!(&fast, &hashed);
                prop_assert_eq!(&fast, &vec![Value::scalar_i64(expected)]);
            }

            #[test]
            fn prop_interpreters_reduce_sum_scalar_identity(x in prop::num::f64::NORMAL) {
                use fj_core::{Atom, Equation, Jaxpr, Primitive, VarId};
                use smallvec::smallvec;
                use std::collections::BTreeMap;
                let jaxpr = Jaxpr::new(
                    vec![VarId(1)],
                    vec![],
                    vec![VarId(2)],
                    vec![Equation {
                        primitive: Primitive::ReduceSum,
                        inputs: smallvec![Atom::Var(VarId(1))],
                        outputs: smallvec![VarId(2)],
                        params: BTreeMap::new(),
                        sub_jaxprs: vec![],
                        effects: vec![],
                    }],
                );
                let out = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x)])
                    .expect("reduce_sum of scalar should succeed");
                let out_val = out[0].as_f64_scalar().expect("should be scalar");
                prop_assert!((out_val - x).abs() < 1e-10);
            }

            #[test]
            fn metamorphic_eval_deterministic(x in -1000.0f64..1000.0) {
                prop_assume!(x.is_finite());
                let jaxpr = build_program(ProgramSpec::Square);
                let out1 = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x)]).expect("eval 1");
                let out2 = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x)]).expect("eval 2");
                prop_assert_eq!(out1, out2, "eval not deterministic at x={}", x);
            }

            #[test]
            fn metamorphic_eval_square_equals_mul(x in -100.0f64..100.0) {
                prop_assume!(x.is_finite());
                let square_jaxpr = build_program(ProgramSpec::Square);
                let square_out = eval_jaxpr(&square_jaxpr, &[Value::scalar_f64(x)])
                    .expect("square eval");
                let expected = x * x;
                let actual = square_out[0].as_f64_scalar().unwrap();
                prop_assert!((actual - expected).abs() < 1e-10, "square(x) != x*x: {} vs {}", actual, expected);
            }

            #[test]
            fn metamorphic_eval_add_zero_identity(x in -1000.0f64..1000.0) {
                prop_assume!(x.is_finite());
                let jaxpr = build_program(ProgramSpec::Add2);
                let out = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x), Value::scalar_f64(0.0)])
                    .expect("add zero");
                let actual = out[0].as_f64_scalar().unwrap();
                prop_assert!((actual - x).abs() < 1e-14, "x + 0 != x: {} + 0 = {}", x, actual);
            }
        }
    }
}
