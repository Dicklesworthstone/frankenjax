#![forbid(unsafe_code)]

pub mod partial_eval;
pub mod staging;

use fj_core::{
    Atom, Equation, Jaxpr, Literal, Primitive, Shape, TensorValue, Value, ValueError, VarId,
};
use fj_lax::{EvalError, eval_primitive_multi};
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

    let mut carry = carry_inputs.to_vec();
    let init_shapes: Vec<Shape> = carry.iter().map(value_shape).collect();
    let init_dtypes: Vec<_> = carry.iter().map(Value::dtype).collect();
    let mut per_y = vec![Vec::with_capacity(scan_len); y_count];
    let reverse = equation
        .params
        .get("reverse")
        .is_some_and(|value| value == "true");
    let scan_indices: Box<dyn Iterator<Item = usize>> = if reverse {
        Box::new((0..scan_len).rev())
    } else {
        Box::new(0..scan_len)
    };

    for scan_idx in scan_indices {
        let x_slice = scan_slice_at(xs, scan_idx)?;
        let mut body_args = carry.clone();
        body_args.push(x_slice);
        let body_outputs =
            eval_jaxpr_with_consts(body_jaxpr, const_values, &body_args).map_err(|err| {
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
            if new_shape != init_shapes[idx] {
                return Err(InterpreterError::Primitive(EvalError::ShapeChanged {
                    primitive: Primitive::Scan,
                    detail: format!(
                        "carry element {idx} changed shape from {:?} to {:?}",
                        init_shapes[idx].dims, new_shape.dims
                    ),
                }));
            }
            if value.dtype() != init_dtypes[idx] {
                return Err(InterpreterError::Primitive(EvalError::TypeMismatch {
                    primitive: Primitive::Scan,
                    detail: "scan body changed carry dtype",
                }));
            }
        }

        carry = body_outputs[..carry_count].to_vec();
        for (bucket, y_value) in per_y.iter_mut().zip(body_outputs[carry_count..].iter()) {
            bucket.push(y_value.clone());
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
    let outputs = if equation.sub_jaxprs.is_empty() {
        let resolved = resolve_equation_inputs(equation, env)?;
        eval_primitive_multi(equation.primitive, &resolved, &equation.params)?
    } else {
        match equation.primitive {
            Primitive::Cond => {
                let resolved = resolve_equation_inputs(equation, env)?;
                evaluate_cond_sub_jaxprs(equation, &resolved)?
            }
            Primitive::Scan => {
                let resolved = resolve_equation_inputs(equation, env)?;
                evaluate_scan_sub_jaxprs(equation, &resolved)?
            }
            Primitive::While => {
                let resolved = resolve_equation_inputs(equation, env)?;
                evaluate_while_sub_jaxprs(equation, &resolved)?
            }
            Primitive::Switch => {
                let resolved = resolve_equation_inputs(equation, env)?;
                evaluate_switch_sub_jaxprs(equation, &resolved)?
            }
            primitive => {
                return Err(InterpreterError::Primitive(EvalError::Unsupported {
                    primitive,
                    detail: "sub_jaxpr execution is not implemented for this primitive".to_owned(),
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
    use super::{InterpreterError, eval_jaxpr, eval_jaxpr_with_consts};
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
        }
    }
}
