#![forbid(unsafe_code)]

pub mod partial_eval;
pub mod staging;

use fj_core::{Atom, Jaxpr, Value, VarId};
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
        let mut resolved = Vec::with_capacity(eqn.inputs.len());
        for atom in &eqn.inputs {
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

        let outputs = eval_primitive_multi(eqn.primitive, &resolved, &eqn.params)?;
        if outputs.len() != eqn.outputs.len() {
            return Err(InterpreterError::UnexpectedOutputArity {
                primitive: eqn.primitive,
                expected: eqn.outputs.len(),
                actual: outputs.len(),
            });
        }

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
