#![forbid(unsafe_code)]

pub mod partial_eval;
pub mod staging;

use fj_core::{Atom, Equation, Jaxpr, Literal, Primitive, Shape, Value, ValueError, VarId};
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
    let index_literal = match index_value {
        Value::Scalar(literal) => *literal,
        Value::Tensor(tensor) => {
            if tensor.shape != Shape::scalar() {
                return Err(InterpreterError::Primitive(EvalError::ShapeMismatch {
                    primitive: Primitive::Switch,
                    left: tensor.shape.clone(),
                    right: Shape::scalar(),
                }));
            }
            if tensor.elements.len() != 1 {
                return Err(InterpreterError::Primitive(EvalError::InvalidTensor(
                    ValueError::ElementCountMismatch {
                        shape: tensor.shape.clone(),
                        expected_count: 1,
                        actual_count: tensor.elements.len(),
                    },
                )));
            }
            tensor.elements[0]
        }
    };
    let index = match index_literal {
        Literal::I64(value) => value,
        Literal::U32(value) => i64::from(value),
        Literal::U64(value) => i64::try_from(value).map_err(|_| {
            InterpreterError::Primitive(EvalError::Unsupported {
                primitive: Primitive::Switch,
                detail: format!("switch index {value} does not fit in i64"),
            })
        })?,
        Literal::Bool(value) => i64::from(value),
        _ => {
            return Err(InterpreterError::Primitive(EvalError::Unsupported {
                primitive: Primitive::Switch,
                detail: format!(
                    "switch index must be integer, got {:?}",
                    index_value.dtype()
                ),
            }));
        }
    };

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
    if index < 0 || index as usize >= equation.sub_jaxprs.len() {
        return Err(InterpreterError::Primitive(EvalError::Unsupported {
            primitive: Primitive::Switch,
            detail: format!(
                "switch index {index} out of bounds for {} branches",
                equation.sub_jaxprs.len()
            ),
        }));
    }

    let selected_branch = &equation.sub_jaxprs[index as usize];
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
    fn eval_switch_with_sub_jaxprs_rejects_out_of_bounds_index() {
        let jaxpr = make_switch_control_flow_jaxpr();
        let err = eval_jaxpr(&jaxpr, &[Value::scalar_i64(3), Value::scalar_i64(5)])
            .expect_err("out-of-bounds switch index should fail");
        let msg = err.to_string();
        assert!(msg.contains("out of bounds"), "unexpected error: {msg}");
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
