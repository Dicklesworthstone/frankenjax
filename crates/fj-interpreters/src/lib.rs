#![forbid(unsafe_code)]

pub mod partial_eval;
pub mod staging;

use fj_core::{
    Atom, DType, Equation, Jaxpr, Literal, Primitive, Shape, TensorValue, Value, ValueError, VarId,
};
use fj_lax::{EvalError, eval_primitive, eval_primitive_multi};
use rustc_hash::FxHashMap;

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
}

impl CheapOp {
    /// Unary ops read a single operand (the chain value); their `b` slot is unused.
    #[inline]
    fn is_unary(self) -> bool {
        matches!(self, CheapOp::Neg | CheapOp::Abs)
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
        CheapOp::Neg | CheapOp::Abs => left,
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
            (CheapOp::Neg | CheapOp::Abs, _) => {}
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
                (CheapOp::Neg | CheapOp::Abs, _) => {}
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

/// Evaluate the fused run over one chunk `out` (already sized to the chunk length).
#[allow(clippy::eq_op)] // Chain/Chain must execute x-x and x/x to preserve NaN behavior.
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
    match s0.op {
        CheapOp::Neg => out.iter_mut().for_each(|o| *o = -*o),
        CheapOp::Abs => out.iter_mut().for_each(|o| *o = o.abs()),
        _ => apply_fusion_other(out, s0.op, true, s0.b, ext, base),
    }
    for step in &tape[1..] {
        match step.op {
            CheapOp::Neg => {
                out.iter_mut().for_each(|o| *o = -*o);
                continue;
            }
            CheapOp::Abs => {
                out.iter_mut().for_each(|o| *o = o.abs());
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
                CheapOp::Neg | CheapOp::Abs => {}
            },
            (FOperand::Chain, other) => apply_fusion_other(out, step.op, true, other, ext, base),
            (other, FOperand::Chain) => apply_fusion_other(out, step.op, false, other, ext, base),
            _ => {} // unreachable for a chain step
        }
    }
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
        if !eqn.params.is_empty() || !eqn.sub_jaxprs.is_empty() || eqn.outputs.len() != 1 {
            break;
        }
        let Some(op) = cheap_op(eqn.primitive) else {
            break;
        };
        let needed = if op.is_unary() { 1 } else { 2 };
        if eqn.inputs.len() != needed {
            break;
        }
        let ext_mark = ext.len();
        let vars_mark = ext_vars.len();
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

    let mut values = vec![0.0_f64; n];
    let mut s = 0;
    while s < n {
        let e = (s + FUSION_CHUNK).min(n);
        apply_fusion_chunk(&mut values[s..e], &tape, &ext, s);
        s = e;
    }
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
fn row_broadcast_len(full: &Shape, candidate: &Shape) -> Option<usize> {
    match (full.dims.as_slice(), candidate.dims.as_slice()) {
        ([_, cols], [row_cols]) if cols == row_cols => Some(*cols as usize),
        _ => None,
    }
}

fn col_broadcast_cols(full: &Shape, candidate: &Shape) -> Option<usize> {
    match (full.dims.as_slice(), candidate.dims.as_slice()) {
        ([rows, cols], [candidate_rows, 1]) if rows == candidate_rows => Some(*cols as usize),
        _ => None,
    }
}

#[inline]
fn f32_fused_neg(value: f32) -> f32 {
    (-f64::from(value)) as f32
}

#[inline]
fn f32_fused_abs(value: f32) -> f32 {
    // Mirrors fj-lax dense f32 abs: widen, f64::abs, round to f32 (exact).
    f64::from(value).abs() as f32
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
            (CheapOp::Neg | CheapOp::Abs, _) => {}
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
            (CheapOp::Neg | CheapOp::Abs, _) => {}
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
                (CheapOp::Neg | CheapOp::Abs, _) => {}
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
                (CheapOp::Neg | CheapOp::Abs, _) => {}
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
        CheapOp::Neg => out.iter_mut().for_each(|o| *o = f32_fused_neg(*o)),
        CheapOp::Abs => out.iter_mut().for_each(|o| *o = f32_fused_abs(*o)),
        _ => apply_f32_fusion_other(out, s0.op, true, s0.b, ext, base),
    }
    for step in &tape[1..] {
        match step.op {
            CheapOp::Neg => {
                out.iter_mut().for_each(|o| *o = f32_fused_neg(*o));
                continue;
            }
            CheapOp::Abs => {
                out.iter_mut().for_each(|o| *o = f32_fused_abs(*o));
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
                CheapOp::Neg | CheapOp::Abs => {}
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
        if !eqn.params.is_empty() || !eqn.sub_jaxprs.is_empty() || eqn.outputs.len() != 1 {
            break;
        }
        let Some(op) = cheap_op(eqn.primitive) else {
            break;
        };
        let needed = if op.is_unary() { 1 } else { 2 };
        if eqn.inputs.len() != needed {
            break;
        }
        let ext_mark = ext.len();
        let vars_mark = ext_vars.len();
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
    let mut s = 0;
    while s < n {
        let e = (s + FUSION_CHUNK).min(n);
        apply_f32_fusion_chunk(&mut values[s..e], &tape, &ext, s);
        s = e;
    }
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
    Scalar(i64),
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
                        Some(_) => return None,
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
            (CheapOp::Neg | CheapOp::Abs, _) => {}
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
                (CheapOp::Neg | CheapOp::Abs, _) => {}
            }
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
        I64Operand::Scalar(v) => out.fill(v),
        I64Operand::Chain => {}
    }
    match s0.op {
        CheapOp::Neg => out.iter_mut().for_each(|o| *o = o.wrapping_neg()),
        CheapOp::Abs => out.iter_mut().for_each(|o| *o = o.wrapping_abs()),
        _ => apply_i64_fusion_other(out, s0.op, true, s0.b, ext, base),
    }
    for step in &tape[1..] {
        match step.op {
            CheapOp::Neg => {
                out.iter_mut().for_each(|o| *o = o.wrapping_neg());
                continue;
            }
            CheapOp::Abs => {
                out.iter_mut().for_each(|o| *o = o.wrapping_abs());
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
                CheapOp::Neg | CheapOp::Abs => {}
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
        if !eqn.params.is_empty() || !eqn.sub_jaxprs.is_empty() || eqn.outputs.len() != 1 {
            break;
        }
        let Some(op) = cheap_op(eqn.primitive) else {
            break;
        };
        let needed = if op.is_unary() { 1 } else { 2 };
        if eqn.inputs.len() != needed {
            break;
        }
        let ext_mark = ext.len();
        let vars_mark = ext_vars.len();
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
    let mut s = 0;
    while s < n {
        let e = (s + FUSION_CHUNK).min(n);
        apply_i64_fusion_chunk(&mut values[s..e], &tape, &ext, s);
        s = e;
    }
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
// The half-precision sibling of the f64/f32/i64 fusion paths — bf16 is the
// dominant ML activation dtype, so chained bf16 elementwise ops are common.
// A maximal run of SAME-half-dtype (all BF16 or all F16; mixed promotes to F32
// and takes a different path) same-shape dense cheap-elementwise equations is
// evaluated in ONE chunked pass over a `u16` working buffer, materializing only
// the final output and skipping the N-1 intermediate half tensors.
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
        CheapOp::Neg | CheapOp::Abs => left,
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
        CheapOp::Neg => out
            .iter_mut()
            .for_each(|o| *o = half_fusion_round(dt, -half_fusion_widen(dt, *o))),
        CheapOp::Abs => out
            .iter_mut()
            .for_each(|o| *o = half_fusion_round(dt, half_fusion_widen(dt, *o).abs())),
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
        CheapOp::Neg | CheapOp::Abs => apply_half_fusion_unary(dt, out, s0.op),
        _ => apply_half_fusion_other(dt, out, s0.op, true, s0.b, ext, base),
    }
    for step in &tape[1..] {
        if matches!(step.op, CheapOp::Neg | CheapOp::Abs) {
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
        if !eqn.params.is_empty() || !eqn.sub_jaxprs.is_empty() || eqn.outputs.len() != 1 {
            break;
        }
        let Some(op) = cheap_op(eqn.primitive) else {
            break;
        };
        let needed = if op.is_unary() { 1 } else { 2 };
        if eqn.inputs.len() != needed {
            break;
        }
        let ext_mark = ext.len();
        let vars_mark = ext_vars.len();
        let dt_mark = half_dt;
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

// Binary op tag shared by all three scalar-arith plans (f64/i64/f32). Max/Min use
// JAX's NaN-propagating semantics for floats (canonical NaN if either operand is
// NaN; see jax_max_f64 / jax_min_f64 in fj-lax) and plain Rust max/min for i64.
#[derive(Clone, Copy)]
enum ScalarF64BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

#[derive(Clone, Copy)]
enum ScalarF64Operand {
    Slot(usize),
    Literal(f64),
}

struct ScalarF64Step {
    op: ScalarF64BinaryOp,
    lhs: ScalarF64Operand,
    rhs: ScalarF64Operand,
    out_slot: usize,
}

struct ScalarF64Plan {
    slots: usize,
    const_slots: Vec<usize>,
    input_slots: Vec<usize>,
    out_slots: Vec<usize>,
    steps: Vec<ScalarF64Step>,
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

fn build_scalar_f64_arith_plan(jaxpr: &Jaxpr, slots: usize) -> Option<ScalarF64Plan> {
    if jaxpr.equations.is_empty() || !jaxpr.effects.is_empty() {
        return None;
    }

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

        let op = scalar_f64_binary_op(equation.primitive)?;
        let out_slot = equation.outputs[0].0 as usize;
        if out_slot >= slots {
            return None;
        }
        steps.push(ScalarF64Step {
            op,
            lhs: scalar_f64_operand(&equation.inputs[0], slots)?,
            rhs: scalar_f64_operand(&equation.inputs[1], slots)?,
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
        let rhs = match read_scalar_f64_operand(slots, step.rhs) {
            Ok(Some(value)) => value,
            Ok(None) => return None,
            Err(error) => return Some(Err(error)),
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
    rhs: ScalarI64Operand,
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
            || equation.inputs.len() != 2
            || equation.outputs.len() != 1
        {
            return None;
        }
        let op = scalar_f64_binary_op(equation.primitive)?;
        let out_slot = equation.outputs[0].0 as usize;
        if out_slot >= slots {
            return None;
        }
        steps.push(ScalarI64Step {
            op,
            lhs: scalar_i64_operand(&equation.inputs[0], slots)?,
            rhs: scalar_i64_operand(&equation.inputs[1], slots)?,
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
        let rhs = match read_scalar_i64_operand(slots, step.rhs) {
            Ok(Some(value)) => value,
            Ok(None) => return None,
            Err(error) => return Some(Err(error)),
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
    rhs: ScalarF32Operand,
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
        if !equation.params.is_empty()
            || !equation.sub_jaxprs.is_empty()
            || !equation.effects.is_empty()
            || equation.inputs.len() != 2
            || equation.outputs.len() != 1
        {
            return None;
        }
        let op = scalar_f64_binary_op(equation.primitive)?;
        let out_slot = equation.outputs[0].0 as usize;
        if out_slot >= slots {
            return None;
        }
        steps.push(ScalarF32Step {
            op,
            lhs: scalar_f32_operand(&equation.inputs[0], slots)?,
            rhs: scalar_f32_operand(&equation.inputs[1], slots)?,
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
        let rhs = match read_scalar_f32_operand(slots, step.rhs) {
            Ok(Some(value)) => value,
            Ok(None) => return None,
            Err(error) => return Some(Err(error)),
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

fn compare_atom_operand(atom: &Atom, jaxpr: &Jaxpr) -> Option<CompareOperand> {
    match atom {
        Atom::Var(var) => {
            if let Some(i) = jaxpr.constvars.iter().position(|c| c == var) {
                Some(CompareOperand::Const(i))
            } else if let Some(i) = jaxpr.invars.iter().position(|iv| iv == var) {
                Some(CompareOperand::Input(i))
            } else {
                None // an equation-produced intermediate — impossible for a 1-op cond
            }
        }
        Atom::Lit(lit) => Some(CompareOperand::Lit(*lit)),
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
    let op = match eqn.primitive {
        Primitive::Eq
        | Primitive::Ne
        | Primitive::Lt
        | Primitive::Le
        | Primitive::Gt
        | Primitive::Ge => eqn.primitive,
        _ => return None,
    };
    Some(ScalarComparePlan {
        op,
        lhs: compare_atom_operand(&eqn.inputs[0], jaxpr)?,
        rhs: compare_atom_operand(&eqn.inputs[1], jaxpr)?,
        n_consts: jaxpr.constvars.len(),
        n_inputs: jaxpr.invars.len(),
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

/// Reusable scratch arenas for the three monomorphic scalar-arith executors —
/// one allocation each, reused across every loop iteration. A loop body builds
/// only the plan(s) whose literals match its dtype (or all, if it has no
/// literals), and the runner tries them in order; each non-matching plan bails
/// on the first operand read, so the unused buffers stay empty.
#[derive(Default)]
struct ScalarPlanBuffers {
    f64: Vec<ScalarF64Slot>,
    i64: Vec<ScalarI64Slot>,
    f32: Vec<ScalarF32Slot>,
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
    scalar_compare_plan: Option<ScalarComparePlan>,
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
            scalar_compare_plan: build_scalar_compare_plan(jaxpr),
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
        scalar_compare_plan: build_scalar_compare_plan(jaxpr),
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
    // Try each monomorphic scalar-arith executor; each bails (returns None) on the
    // first operand that is not its dtype at runtime, so for a given call at most
    // one actually runs. A body with a typed literal builds only the matching plan;
    // a literal-free body builds all three and the runtime dtype selects one.
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
    if let Some(p) = &plan.scalar_compare_plan
        && let Some(result) = run_scalar_compare_plan(p, const_values, args)
    {
        out.clear();
        out.push(Value::scalar_bool(result));
        return Ok(());
    }
    run_dense_env_into(jaxpr, const_values, args, env, &plan.last_use, scratch, out)
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
        // The literal-free Mul/Add/Sub body builds all three scalar plans; i64
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
        let planned = eval_jaxpr(&jaxpr, &[t.clone()]).expect("planned");
        let generic = eval_jaxpr_hashed_env(&jaxpr, &[], &[t]).expect("generic");
        assert_eq!(planned, generic);
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
