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

    for _ in 0..max_iter {
        let cond_outputs =
            eval_jaxpr_with_consts(cond_jaxpr, cond_consts, &carry).map_err(|err| {
                InterpreterError::Primitive(map_sub_jaxpr_error(
                    Primitive::While,
                    "while cond",
                    err,
                ))
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

        let next_carry =
            eval_jaxpr_with_consts(body_jaxpr, body_consts, &carry).map_err(|err| {
                InterpreterError::Primitive(map_sub_jaxpr_error(
                    Primitive::While,
                    "while body",
                    err,
                ))
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
        carry = next_carry;
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

    let scan_context = ScanIterationContext {
        body_jaxpr,
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
    const_values: &'a [Value],
    xs: &'a Value,
    init_shapes: &'a [Shape],
    init_dtypes: &'a [DType],
}

fn evaluate_scan_iteration(
    context: &ScanIterationContext<'_>,
    scan_idx: usize,
    carry: &mut [Value],
    per_y: &mut [Vec<Value>],
    body_args: &mut Vec<Value>,
) -> Result<(), InterpreterError> {
    let x_slice = scan_slice_at(context.xs, scan_idx)?;
    body_args.clear();
    body_args.extend(carry.iter().cloned());
    body_args.push(x_slice);

    let carry_count = carry.len();
    let y_count = per_y.len();
    let body_outputs = eval_jaxpr_with_consts(context.body_jaxpr, context.const_values, body_args)
        .map_err(|err| {
            InterpreterError::Primitive(map_sub_jaxpr_error(Primitive::Scan, "scan body", err))
        })?;
    if body_outputs.len() != carry_count + y_count {
        return Err(InterpreterError::InvariantViolation {
            detail: format!(
                "scan body sub_jaxpr returned {} outputs; expected {}",
                body_outputs.len(),
                carry_count + y_count
            ),
        });
    }

    for (idx, value) in body_outputs[..carry_count].iter().enumerate() {
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

    let mut outputs = body_outputs.into_iter();
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
}

fn cheap_op(p: Primitive) -> Option<CheapOp> {
    match p {
        Primitive::Add => Some(CheapOp::Add),
        Primitive::Sub => Some(CheapOp::Sub),
        Primitive::Mul => Some(CheapOp::Mul),
        Primitive::Div => Some(CheapOp::Div),
        Primitive::Neg => Some(CheapOp::Neg),
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
        CheapOp::Neg => left,
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
            (CheapOp::Neg, _) => {}
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
                (CheapOp::Neg, _) => {}
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
    if s0.op == CheapOp::Neg {
        out.iter_mut().for_each(|o| *o = -*o);
    } else {
        apply_fusion_other(out, s0.op, true, s0.b, ext, base);
    }
    for step in &tape[1..] {
        if step.op == CheapOp::Neg {
            out.iter_mut().for_each(|o| *o = -*o);
            continue;
        }
        match (step.a, step.b) {
            (FOperand::Chain, FOperand::Chain) => match step.op {
                CheapOp::Add => out.iter_mut().for_each(|o| *o = *o + *o),
                CheapOp::Sub => out.iter_mut().for_each(|o| *o = *o - *o),
                CheapOp::Mul => out.iter_mut().for_each(|o| *o = *o * *o),
                CheapOp::Div => out.iter_mut().for_each(|o| *o = *o / *o),
                CheapOp::Neg => {}
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
        let needed = if op == CheapOp::Neg { 1 } else { 2 };
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
        let b = if op == CheapOp::Neg {
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
fn f32_fused_binary(op: CheapOp, left: f32, right: f32) -> f32 {
    let left = f64::from(left);
    let right = f64::from(right);
    match op {
        CheapOp::Add => (left + right) as f32,
        CheapOp::Sub => (left - right) as f32,
        CheapOp::Mul => (left * right) as f32,
        CheapOp::Div => (left / right) as f32,
        CheapOp::Neg => left as f32,
    }
}

#[inline]
fn f32_fused_neg(value: f32) -> f32 {
    (-f64::from(value)) as f32
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
        match chain_left {
            true => out_part
                .iter_mut()
                .zip(row_part)
                .for_each(|(o, e)| *o = f32_fused_binary(op, *o, *e)),
            false => out_part
                .iter_mut()
                .zip(row_part)
                .for_each(|(o, e)| *o = f32_fused_binary(op, *e, *o)),
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
        match chain_left {
            true => out_part
                .iter_mut()
                .for_each(|o| *o = f32_fused_binary(op, *o, scalar)),
            false => out_part
                .iter_mut()
                .for_each(|o| *o = f32_fused_binary(op, scalar, *o)),
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
        F32Operand::Scalar(s) => match chain_left {
            true => out
                .iter_mut()
                .for_each(|o| *o = f32_fused_binary(op, *o, s)),
            false => out
                .iter_mut()
                .for_each(|o| *o = f32_fused_binary(op, s, *o)),
        },
        F32Operand::Ext(i) => {
            let sl = &ext[i][base..base + out.len()];
            match chain_left {
                true => out
                    .iter_mut()
                    .zip(sl)
                    .for_each(|(o, e)| *o = f32_fused_binary(op, *o, *e)),
                false => out
                    .iter_mut()
                    .zip(sl)
                    .for_each(|(o, e)| *o = f32_fused_binary(op, *e, *o)),
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
    if s0.op == CheapOp::Neg {
        out.iter_mut().for_each(|o| *o = f32_fused_neg(*o));
    } else {
        apply_f32_fusion_other(out, s0.op, true, s0.b, ext, base);
    }
    for step in &tape[1..] {
        if step.op == CheapOp::Neg {
            out.iter_mut().for_each(|o| *o = f32_fused_neg(*o));
            continue;
        }
        match (step.a, step.b) {
            (F32Operand::Chain, F32Operand::Chain) => out
                .iter_mut()
                .for_each(|o| *o = f32_fused_binary(step.op, *o, *o)),
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
        let needed = if op == CheapOp::Neg { 1 } else { 2 };
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
        let b = if op == CheapOp::Neg {
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
            (CheapOp::Neg, _) => {}
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
                (CheapOp::Neg, _) => {}
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
    if s0.op == CheapOp::Neg {
        out.iter_mut().for_each(|o| *o = o.wrapping_neg());
    } else {
        apply_i64_fusion_other(out, s0.op, true, s0.b, ext, base);
    }
    for step in &tape[1..] {
        if step.op == CheapOp::Neg {
            out.iter_mut().for_each(|o| *o = o.wrapping_neg());
            continue;
        }
        match (step.a, step.b) {
            (I64Operand::Chain, I64Operand::Chain) => match step.op {
                CheapOp::Add => out.iter_mut().for_each(|o| *o = o.wrapping_add(*o)),
                CheapOp::Sub => out.iter_mut().for_each(|o| *o = o.wrapping_sub(*o)),
                CheapOp::Mul => out.iter_mut().for_each(|o| *o = o.wrapping_mul(*o)),
                CheapOp::Div => out
                    .iter_mut()
                    .for_each(|o| *o = o.checked_div(*o).unwrap_or(0)),
                CheapOp::Neg => {}
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
        let needed = if op == CheapOp::Neg { 1 } else { 2 };
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
        let b = if op == CheapOp::Neg {
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

fn try_fuse_elementwise_chain(
    jaxpr: &Jaxpr,
    start: usize,
    env: &[Option<Value>],
    last_use: &[usize],
) -> Option<FusedRun> {
    try_fuse_elementwise_chain_f64(jaxpr, start, env, last_use)
        .or_else(|| try_fuse_elementwise_chain_f32(jaxpr, start, env, last_use))
        .or_else(|| try_fuse_elementwise_chain_i64(jaxpr, start, env, last_use))
}

fn eval_jaxpr_dense_env(
    jaxpr: &Jaxpr,
    const_values: &[Value],
    args: &[Value],
    slots: usize,
) -> Result<Vec<Value>, InterpreterError> {
    let mut env: Vec<Option<Value>> = vec![None; slots];
    for (idx, var) in jaxpr.constvars.iter().enumerate() {
        env[var.0 as usize] = Some(const_values[idx].clone());
    }
    for (idx, var) in jaxpr.invars.iter().enumerate() {
        env[var.0 as usize] = Some(args[idx].clone());
    }

    // Liveness: `last_use[slot]` = index of the LAST equation that reads this var as
    // an input (`usize::MAX` = never read; outvars are pinned `usize::MAX` so they
    // survive to the return). After an equation runs we drop every input slot whose
    // last use was this equation, so an N-equation elementwise CHAIN holds only its
    // live working set (~2 tensors) instead of all N intermediates at once — a large
    // peak-memory reduction for deep jaxprs. Bit-identical: a slot is freed strictly
    // after its final read, outvars are never freed, so outputs are unchanged.
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

    let mut scratch: Vec<Value> = Vec::new();
    let mut i = 0;
    while i < jaxpr.equations.len() {
        // Fast path: fuse a maximal cheap-elementwise run starting here into one
        // chunked pass (materializes only the final output). Bails to the normal
        // path below for anything not fuse-eligible.
        if let Some(run) = try_fuse_elementwise_chain(jaxpr, i, &env, &last_use) {
            let tensor = match run.values {
                FusedValues::F64(values) => TensorValue::new_f64_values(run.shape, values),
                FusedValues::F32(values) => TensorValue::new_f32_values(run.shape, values),
                FusedValues::I64(values) => TensorValue::new_i64_values(run.shape, values),
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
            let output = eval_primitive(eqn.primitive, &scratch, &eqn.params)
                .map_err(InterpreterError::Primitive)?;
            env[eqn.outputs[0].0 as usize] = Some(output);
        } else {
            let outputs = eval_equation_outputs_from_resolved(eqn, &scratch)?;
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

    jaxpr
        .outvars
        .iter()
        .map(|var| {
            env.get(var.0 as usize)
                .and_then(|slot| slot.clone())
                .ok_or(InterpreterError::MissingVariable(*var))
        })
        .collect()
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

    fn lit32(x: f32) -> Atom {
        Atom::Lit(Literal::from_f32(x))
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
        let got_bits = f32_bits(&fused_outputs[0]);
        let want_bits = f32_bits(&unfused_outputs[0]);
        assert_eq!(
            got_bits, want_bits,
            "fused f32 chain must match forced unfused path bit-for-bit"
        );
        let digest = fj_test_utils::fixture_id_from_json(&want_bits)
            .expect("reference output bits should hash");
        assert_eq!(
            digest,
            "fef28624a52e5647abc35f0d388072b443cf081e5941243c6c58a8bd91f40a84"
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
        let got_bits = f32_bits(&fused_outputs[0]);
        let want_bits = f32_bits(&unfused_outputs[0]);
        assert_eq!(
            got_bits, want_bits,
            "fused f32 col-broadcast chain must match forced unfused path bit-for-bit"
        );
        let digest = fj_test_utils::fixture_id_from_json(&want_bits)
            .expect("reference output bits should hash");
        assert_eq!(
            digest,
            "5762f3ec4614f491d21407cbb09c5cd92915840f65d145070f8d8b5e8c7c5e3a"
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
            digest,
            "TO_FILL",
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
