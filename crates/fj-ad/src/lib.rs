#![forbid(unsafe_code)]

use fj_core::{Atom, Jaxpr, Primitive, Value, VarId};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum AdError {
    UnsupportedPrimitive(Primitive),
    NonScalarGradientOutput,
    EvalFailed(String),
    MissingVariable(VarId),
    InputArity { expected: usize, actual: usize },
}

impl std::fmt::Display for AdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedPrimitive(p) => {
                write!(f, "AD not implemented for primitive: {}", p.as_str())
            }
            Self::NonScalarGradientOutput => write!(f, "grad requires scalar output"),
            Self::EvalFailed(detail) => write!(f, "forward eval failed: {detail}"),
            Self::MissingVariable(var) => write!(f, "missing variable v{}", var.0),
            Self::InputArity { expected, actual } => {
                write!(f, "input arity mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for AdError {}

#[derive(Debug, Clone)]
struct TapeEntry {
    primitive: Primitive,
    inputs: Vec<VarId>,
    output: VarId,
    input_values: Vec<Value>,
}

#[derive(Debug, Clone)]
struct Tape {
    entries: Vec<TapeEntry>,
}

type ForwardResult = (Vec<Value>, Tape, BTreeMap<VarId, Value>);

fn forward_with_tape(jaxpr: &Jaxpr, args: &[Value]) -> Result<ForwardResult, AdError> {
    if args.len() != jaxpr.invars.len() {
        return Err(AdError::InputArity {
            expected: jaxpr.invars.len(),
            actual: args.len(),
        });
    }

    let mut env: BTreeMap<VarId, Value> = BTreeMap::new();
    for (idx, var) in jaxpr.invars.iter().enumerate() {
        env.insert(*var, args[idx].clone());
    }

    let mut tape = Tape {
        entries: Vec::with_capacity(jaxpr.equations.len()),
    };

    for eqn in &jaxpr.equations {
        let mut resolved = Vec::with_capacity(eqn.inputs.len());
        let mut input_var_ids = Vec::with_capacity(eqn.inputs.len());

        for atom in eqn.inputs.iter() {
            match atom {
                Atom::Var(var) => {
                    let value = env
                        .get(var)
                        .cloned()
                        .ok_or(AdError::MissingVariable(*var))?;
                    resolved.push(value);
                    input_var_ids.push(*var);
                }
                Atom::Lit(lit) => {
                    resolved.push(Value::Scalar(*lit));
                    // Literals get a sentinel VarId; their cotangent is discarded
                    input_var_ids.push(VarId(u32::MAX));
                }
            }
        }

        let output = eval_primitive(eqn.primitive, &resolved, &eqn.params)
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;

        let out_var = eqn.outputs[0];
        env.insert(out_var, output);

        tape.entries.push(TapeEntry {
            primitive: eqn.primitive,
            inputs: input_var_ids,
            output: out_var,
            input_values: resolved,
        });
    }

    let outputs = jaxpr
        .outvars
        .iter()
        .map(|var| env.get(var).cloned().ok_or(AdError::MissingVariable(*var)))
        .collect::<Result<Vec<_>, _>>()?;

    Ok((outputs, tape, env))
}

fn backward(
    tape: &Tape,
    output_var: VarId,
    output_cotangent: f64,
    jaxpr: &Jaxpr,
) -> Result<Vec<f64>, AdError> {
    let mut adjoints: BTreeMap<VarId, f64> = BTreeMap::new();
    adjoints.insert(output_var, output_cotangent);

    for entry in tape.entries.iter().rev() {
        let g = adjoints.get(&entry.output).copied().unwrap_or(0.0);
        if g == 0.0 {
            continue;
        }

        let cotangents = vjp(entry.primitive, &entry.input_values, g)?;

        for (var_id, cot) in entry.inputs.iter().zip(cotangents.iter()) {
            if var_id.0 == u32::MAX {
                continue; // literal sentinel
            }
            *adjoints.entry(*var_id).or_insert(0.0) += cot;
        }
    }

    let grads = jaxpr
        .invars
        .iter()
        .map(|var| adjoints.get(var).copied().unwrap_or(0.0))
        .collect();

    Ok(grads)
}

fn vjp(primitive: Primitive, inputs: &[Value], g: f64) -> Result<Vec<f64>, AdError> {
    match primitive {
        Primitive::Add => Ok(vec![g, g]),
        Primitive::Sub => Ok(vec![g, -g]),
        Primitive::Mul => {
            let a = to_f64(&inputs[0])?;
            let b = to_f64(&inputs[1])?;
            Ok(vec![g * b, g * a])
        }
        Primitive::Neg => Ok(vec![-g]),
        Primitive::Abs => {
            let x = to_f64(&inputs[0])?;
            Ok(vec![if x >= 0.0 { g } else { -g }])
        }
        Primitive::Max => {
            let a = to_f64(&inputs[0])?;
            let b = to_f64(&inputs[1])?;
            if a >= b {
                Ok(vec![g, 0.0])
            } else {
                Ok(vec![0.0, g])
            }
        }
        Primitive::Min => {
            let a = to_f64(&inputs[0])?;
            let b = to_f64(&inputs[1])?;
            if a <= b {
                Ok(vec![g, 0.0])
            } else {
                Ok(vec![0.0, g])
            }
        }
        Primitive::Pow => {
            let a = to_f64(&inputs[0])?;
            let b = to_f64(&inputs[1])?;
            // d/da(a^b) = b * a^(b-1), d/db(a^b) = a^b * ln(a)
            Ok(vec![g * b * a.powf(b - 1.0), g * a.powf(b) * a.ln()])
        }
        Primitive::Exp => {
            let x = to_f64(&inputs[0])?;
            Ok(vec![g * x.exp()])
        }
        Primitive::Log => {
            let x = to_f64(&inputs[0])?;
            Ok(vec![g / x])
        }
        Primitive::Sqrt => {
            let x = to_f64(&inputs[0])?;
            Ok(vec![g / (2.0 * x.sqrt())])
        }
        Primitive::Rsqrt => {
            let x = to_f64(&inputs[0])?;
            Ok(vec![-0.5 * g * x.powf(-1.5)])
        }
        Primitive::Floor | Primitive::Ceil | Primitive::Round => {
            // Gradient is zero almost everywhere (piecewise constant)
            Ok(vec![0.0])
        }
        Primitive::Sin => {
            let x = to_f64(&inputs[0])?;
            Ok(vec![g * x.cos()])
        }
        Primitive::Cos => {
            let x = to_f64(&inputs[0])?;
            Ok(vec![g * (-x.sin())])
        }
        Primitive::ReduceSum => {
            // For scalar input, cotangent is just g
            Ok(vec![g])
        }
        Primitive::ReduceMax | Primitive::ReduceMin => {
            // Subgradient: pass gradient to the argmax/argmin element
            // For scalar, just pass through
            Ok(vec![g])
        }
        Primitive::ReduceProd => {
            // For scalar input, gradient is just g
            Ok(vec![g])
        }
        Primitive::Dot => {
            // rank-1 dot(a,b) = sum(a*b), cotangents: (g*b, g*a)
            let a = to_f64(&inputs[0])?;
            let b = to_f64(&inputs[1])?;
            Ok(vec![g * b, g * a])
        }
        // Comparison ops have zero gradient (non-differentiable)
        Primitive::Eq
        | Primitive::Ne
        | Primitive::Lt
        | Primitive::Le
        | Primitive::Gt
        | Primitive::Ge => Ok(vec![0.0, 0.0]),
        Primitive::Reshape
        | Primitive::Slice
        | Primitive::Gather
        | Primitive::Scatter
        | Primitive::Transpose
        | Primitive::BroadcastInDim
        | Primitive::Concatenate => Err(AdError::UnsupportedPrimitive(primitive)),
    }
}

fn to_f64(value: &Value) -> Result<f64, AdError> {
    value
        .as_f64_scalar()
        .ok_or(AdError::NonScalarGradientOutput)
}

// ── Forward-mode JVP ───────────────────────────────────────────────

/// Result of forward-mode JVP computation.
#[derive(Debug, Clone, PartialEq)]
pub struct JvpResult {
    pub primals: Vec<Value>,
    pub tangents: Vec<f64>,
}

/// Compute JVP (forward-mode AD) for a Jaxpr.
///
/// Given primals `x` and tangent vector `dx`, computes `(f(x), df/dx · dx)`.
pub fn jvp(jaxpr: &Jaxpr, primals: &[Value], tangents: &[f64]) -> Result<JvpResult, AdError> {
    if primals.len() != jaxpr.invars.len() {
        return Err(AdError::InputArity {
            expected: jaxpr.invars.len(),
            actual: primals.len(),
        });
    }
    if tangents.len() != primals.len() {
        return Err(AdError::InputArity {
            expected: primals.len(),
            actual: tangents.len(),
        });
    }

    let mut primal_env: BTreeMap<VarId, Value> = BTreeMap::new();
    let mut tangent_env: BTreeMap<VarId, f64> = BTreeMap::new();

    for (idx, var) in jaxpr.invars.iter().enumerate() {
        primal_env.insert(*var, primals[idx].clone());
        tangent_env.insert(*var, tangents[idx]);
    }

    for eqn in &jaxpr.equations {
        let mut resolved_primals = Vec::with_capacity(eqn.inputs.len());
        let mut resolved_tangents = Vec::with_capacity(eqn.inputs.len());

        for atom in eqn.inputs.iter() {
            match atom {
                Atom::Var(var) => {
                    let pval = primal_env
                        .get(var)
                        .cloned()
                        .ok_or(AdError::MissingVariable(*var))?;
                    let tval = tangent_env.get(var).copied().unwrap_or(0.0);
                    resolved_primals.push(pval);
                    resolved_tangents.push(tval);
                }
                Atom::Lit(lit) => {
                    resolved_primals.push(Value::Scalar(*lit));
                    resolved_tangents.push(0.0); // literals have zero tangent
                }
            }
        }

        let primal_out = eval_primitive(eqn.primitive, &resolved_primals, &eqn.params)
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
        let tangent_out = jvp_rule(eqn.primitive, &resolved_primals, &resolved_tangents)?;

        let out_var = eqn.outputs[0];
        primal_env.insert(out_var, primal_out);
        tangent_env.insert(out_var, tangent_out);
    }

    let out_primals = jaxpr
        .outvars
        .iter()
        .map(|var| {
            primal_env
                .get(var)
                .cloned()
                .ok_or(AdError::MissingVariable(*var))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let out_tangents = jaxpr
        .outvars
        .iter()
        .map(|var| tangent_env.get(var).copied().unwrap_or(0.0))
        .collect();

    Ok(JvpResult {
        primals: out_primals,
        tangents: out_tangents,
    })
}

fn jvp_rule(primitive: Primitive, primals: &[Value], tangents: &[f64]) -> Result<f64, AdError> {
    match primitive {
        // d(a+b) = da + db
        Primitive::Add => Ok(tangents[0] + tangents[1]),
        // d(a-b) = da - db
        Primitive::Sub => Ok(tangents[0] - tangents[1]),
        // d(a*b) = da*b + a*db
        Primitive::Mul => {
            let a = to_f64(&primals[0])?;
            let b = to_f64(&primals[1])?;
            Ok(tangents[0] * b + a * tangents[1])
        }
        // d(-x) = -dx
        Primitive::Neg => Ok(-tangents[0]),
        // d(|x|) = sign(x) * dx
        Primitive::Abs => {
            let x = to_f64(&primals[0])?;
            Ok(if x >= 0.0 { tangents[0] } else { -tangents[0] })
        }
        // d(max(a,b)) = da if a>=b, db otherwise
        Primitive::Max => {
            let a = to_f64(&primals[0])?;
            let b = to_f64(&primals[1])?;
            Ok(if a >= b { tangents[0] } else { tangents[1] })
        }
        // d(min(a,b)) = da if a<=b, db otherwise
        Primitive::Min => {
            let a = to_f64(&primals[0])?;
            let b = to_f64(&primals[1])?;
            Ok(if a <= b { tangents[0] } else { tangents[1] })
        }
        // d(a^b) = b*a^(b-1)*da + a^b*ln(a)*db
        Primitive::Pow => {
            let a = to_f64(&primals[0])?;
            let b = to_f64(&primals[1])?;
            Ok(b * a.powf(b - 1.0) * tangents[0] + a.powf(b) * a.ln() * tangents[1])
        }
        // d(exp(x)) = exp(x)*dx
        Primitive::Exp => {
            let x = to_f64(&primals[0])?;
            Ok(x.exp() * tangents[0])
        }
        // d(log(x)) = dx/x
        Primitive::Log => {
            let x = to_f64(&primals[0])?;
            Ok(tangents[0] / x)
        }
        // d(sqrt(x)) = dx/(2*sqrt(x))
        Primitive::Sqrt => {
            let x = to_f64(&primals[0])?;
            Ok(tangents[0] / (2.0 * x.sqrt()))
        }
        // d(rsqrt(x)) = -0.5 * x^(-1.5) * dx
        Primitive::Rsqrt => {
            let x = to_f64(&primals[0])?;
            Ok(-0.5 * x.powf(-1.5) * tangents[0])
        }
        // Piecewise constant: zero derivative
        Primitive::Floor | Primitive::Ceil | Primitive::Round => Ok(0.0),
        // d(sin(x)) = cos(x)*dx
        Primitive::Sin => {
            let x = to_f64(&primals[0])?;
            Ok(x.cos() * tangents[0])
        }
        // d(cos(x)) = -sin(x)*dx
        Primitive::Cos => {
            let x = to_f64(&primals[0])?;
            Ok(-x.sin() * tangents[0])
        }
        // d(reduce_sum(x)) = dx (scalar input)
        Primitive::ReduceSum => Ok(tangents[0]),
        // Subgradient for reduce_max/min
        Primitive::ReduceMax | Primitive::ReduceMin => Ok(tangents[0]),
        // d(reduce_prod(x)) = dx (scalar input)
        Primitive::ReduceProd => Ok(tangents[0]),
        // d(dot(a,b)) = dot(da,b) + dot(a,db) — scalar case: da*b + a*db
        Primitive::Dot => {
            let a = to_f64(&primals[0])?;
            let b = to_f64(&primals[1])?;
            Ok(tangents[0] * b + a * tangents[1])
        }
        // Comparison ops: zero derivative
        Primitive::Eq
        | Primitive::Ne
        | Primitive::Lt
        | Primitive::Le
        | Primitive::Gt
        | Primitive::Ge => Ok(0.0),
        Primitive::Reshape
        | Primitive::Slice
        | Primitive::Gather
        | Primitive::Scatter
        | Primitive::Transpose
        | Primitive::BroadcastInDim
        | Primitive::Concatenate => Err(AdError::UnsupportedPrimitive(primitive)),
    }
}

// ── Forward-mode gradient (convenience) ────────────────────────────

/// Compute gradient via forward-mode JVP by evaluating with unit tangent.
/// Equivalent to grad_first but using forward-mode instead of reverse-mode.
pub fn jvp_grad_first(jaxpr: &Jaxpr, args: &[Value]) -> Result<f64, AdError> {
    let mut tangents = vec![0.0; args.len()];
    tangents[0] = 1.0; // unit tangent for first argument
    let result = jvp(jaxpr, args, &tangents)?;
    result
        .tangents
        .first()
        .copied()
        .ok_or(AdError::NonScalarGradientOutput)
}

/// Compute gradients of a Jaxpr with respect to all inputs.
///
/// The Jaxpr must produce a single scalar output.
pub fn grad_jaxpr(jaxpr: &Jaxpr, args: &[Value]) -> Result<Vec<f64>, AdError> {
    let (outputs, tape, _env) = forward_with_tape(jaxpr, args)?;

    let output_val = outputs
        .first()
        .and_then(Value::as_f64_scalar)
        .ok_or(AdError::NonScalarGradientOutput)?;
    let _ = output_val; // just verify it's scalar

    let output_var = jaxpr.outvars[0];
    backward(&tape, output_var, 1.0, jaxpr)
}

/// Compute gradient with respect to the first input only (convenience wrapper
/// matching JAX's default `grad` behavior).
pub fn grad_first(jaxpr: &Jaxpr, args: &[Value]) -> Result<f64, AdError> {
    let grads = grad_jaxpr(jaxpr, args)?;
    Ok(grads[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use fj_core::{ProgramSpec, build_program};

    #[test]
    fn grad_x_squared_at_3() {
        let jaxpr = build_program(ProgramSpec::Square);
        let grads = grad_jaxpr(&jaxpr, &[Value::scalar_f64(3.0)]).expect("grad should succeed");
        assert!(
            (grads[0] - 6.0).abs() < 1e-10,
            "d/dx(x²) at x=3 = 6, got {}",
            grads[0]
        );
    }

    #[test]
    fn grad_x_squared_plus_2x_at_3() {
        let jaxpr = build_program(ProgramSpec::SquarePlusLinear);
        let grads = grad_jaxpr(&jaxpr, &[Value::scalar_f64(3.0)]).expect("grad should succeed");
        assert!(
            (grads[0] - 8.0).abs() < 1e-10,
            "d/dx(x²+2x) at x=3 = 8, got {}",
            grads[0]
        );
    }

    #[test]
    fn grad_sin_at_zero() {
        // Build sin(x) program: input x, output sin(x)
        use fj_core::{Equation, Jaxpr, VarId};
        use smallvec::smallvec;
        use std::collections::BTreeMap;

        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Sin,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        );
        let grads = grad_jaxpr(&jaxpr, &[Value::scalar_f64(0.0)]).expect("grad should succeed");
        assert!(
            (grads[0] - 1.0).abs() < 1e-10,
            "d/dx(sin(x)) at x=0 = cos(0) = 1, got {}",
            grads[0]
        );
    }

    #[test]
    fn grad_cos_at_zero() {
        use fj_core::{Equation, Jaxpr, VarId};
        use smallvec::smallvec;
        use std::collections::BTreeMap;

        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Cos,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        );
        let grads = grad_jaxpr(&jaxpr, &[Value::scalar_f64(0.0)]).expect("grad should succeed");
        assert!(
            grads[0].abs() < 1e-10,
            "d/dx(cos(x)) at x=0 = -sin(0) = 0, got {}",
            grads[0]
        );
    }

    #[test]
    fn symbolic_matches_numerical() {
        let jaxpr = build_program(ProgramSpec::Square);
        let x = 3.0;
        let symbolic = grad_first(&jaxpr, &[Value::scalar_f64(x)]).expect("symbolic grad");

        // Numerical (finite-diff)
        let eps = 1e-6;
        let no_p = BTreeMap::new();
        let plus = fj_lax::eval_primitive(
            Primitive::Mul,
            &[Value::scalar_f64(x + eps), Value::scalar_f64(x + eps)],
            &no_p,
        )
        .unwrap()
        .as_f64_scalar()
        .unwrap();
        let minus = fj_lax::eval_primitive(
            Primitive::Mul,
            &[Value::scalar_f64(x - eps), Value::scalar_f64(x - eps)],
            &no_p,
        )
        .unwrap()
        .as_f64_scalar()
        .unwrap();
        let numerical = (plus - minus) / (2.0 * eps);

        assert!(
            (symbolic - numerical).abs() < 1e-4,
            "symbolic={symbolic}, numerical={numerical}"
        );
    }

    #[test]
    fn test_ad_test_log_schema_contract() {
        let fixture_id =
            fj_test_utils::fixture_id_from_json(&(3_u32, 9_u32)).expect("fixture digest");
        let log = fj_test_utils::TestLogV1::unit(
            fj_test_utils::test_id(module_path!(), "test_ad_test_log_schema_contract"),
            fixture_id,
            fj_test_utils::TestMode::Strict,
            fj_test_utils::TestResult::Pass,
        );
        assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
    }

    // ── Forward-mode JVP tests ──────────────────────────────────

    #[test]
    fn jvp_x_squared_at_3() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = jvp(&jaxpr, &[Value::scalar_f64(3.0)], &[1.0]).expect("jvp should succeed");
        // f(3) = 9, f'(3) = 6
        let primal = result.primals[0].as_f64_scalar().unwrap();
        assert!((primal - 9.0).abs() < 1e-10, "primal should be 9, got {primal}");
        assert!(
            (result.tangents[0] - 6.0).abs() < 1e-10,
            "tangent should be 6, got {}",
            result.tangents[0]
        );
    }

    #[test]
    fn jvp_matches_reverse_mode() {
        let jaxpr = build_program(ProgramSpec::Square);
        let x = 5.0;
        let fwd = jvp_grad_first(&jaxpr, &[Value::scalar_f64(x)]).expect("jvp grad");
        let rev = grad_first(&jaxpr, &[Value::scalar_f64(x)]).expect("vjp grad");
        assert!(
            (fwd - rev).abs() < 1e-10,
            "forward {fwd} should match reverse {rev}"
        );
    }

    #[test]
    fn jvp_square_plus_linear() {
        let jaxpr = build_program(ProgramSpec::SquarePlusLinear);
        let result = jvp(&jaxpr, &[Value::scalar_f64(3.0)], &[1.0]).expect("jvp should succeed");
        // f(x) = x² + 2x, f(3) = 15, f'(3) = 8
        let primal = result.primals[0].as_f64_scalar().unwrap();
        assert!((primal - 15.0).abs() < 1e-10, "primal should be 15, got {primal}");
        assert!(
            (result.tangents[0] - 8.0).abs() < 1e-10,
            "tangent should be 8, got {}",
            result.tangents[0]
        );
    }

    #[test]
    fn jvp_sin_at_zero() {
        use fj_core::{Equation, Jaxpr, VarId};
        use smallvec::smallvec;
        use std::collections::BTreeMap;

        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Sin,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        );
        let result = jvp(&jaxpr, &[Value::scalar_f64(0.0)], &[1.0]).expect("jvp should succeed");
        // sin(0) = 0, cos(0) = 1
        assert!(result.primals[0].as_f64_scalar().unwrap().abs() < 1e-10);
        assert!((result.tangents[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn jvp_scaled_tangent() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = jvp(&jaxpr, &[Value::scalar_f64(3.0)], &[2.0]).expect("jvp should succeed");
        // With tangent=2.0: df = 2 * f'(3) = 2 * 6 = 12
        assert!(
            (result.tangents[0] - 12.0).abs() < 1e-10,
            "scaled tangent should be 12, got {}",
            result.tangents[0]
        );
    }

    mod proptest_tests {
        use super::*;
        use fj_interpreters::eval_jaxpr;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(proptest::test_runner::Config::with_cases(
                fj_test_utils::property_test_case_count()
            ))]
            #[test]
            fn prop_ad_grad_square_matches_two_x(x in prop::num::f64::NORMAL.prop_filter(
                "finite and not too large",
                |x| x.is_finite() && x.abs() < 1e6
            )) {
                let _seed = fj_test_utils::capture_proptest_seed();
                let jaxpr = build_program(ProgramSpec::Square);
                let symbolic = grad_first(&jaxpr, &[Value::scalar_f64(x)])
                    .expect("symbolic grad");
                let expected = 2.0 * x;
                prop_assert!(
                    (symbolic - expected).abs() < 1e-8,
                    "grad(x²) at x={x}: symbolic={symbolic}, expected={expected}"
                );
            }

            #[test]
            fn prop_ad_symbolic_grad_matches_numerical(x in prop::num::f64::NORMAL.prop_filter(
                "finite and moderate",
                |x| x.is_finite() && x.abs() < 1e3
            )) {
                let jaxpr = build_program(ProgramSpec::Square);
                let symbolic = grad_first(&jaxpr, &[Value::scalar_f64(x)])
                    .expect("symbolic grad");

                // Numerical finite-diff
                let eps = 1e-6;
                let plus = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x + eps)])
                    .unwrap()[0].as_f64_scalar().unwrap();
                let minus = eval_jaxpr(&jaxpr, &[Value::scalar_f64(x - eps)])
                    .unwrap()[0].as_f64_scalar().unwrap();
                let numerical = (plus - minus) / (2.0 * eps);

                prop_assert!(
                    (symbolic - numerical).abs() < 1e-4,
                    "x={x}: symbolic={symbolic}, numerical={numerical}"
                );
            }

            #[test]
            fn prop_jvp_matches_vjp(x in prop::num::f64::NORMAL.prop_filter(
                "finite and not too large",
                |x| x.is_finite() && x.abs() < 1e6
            )) {
                let _seed = fj_test_utils::capture_proptest_seed();
                let jaxpr = build_program(ProgramSpec::Square);
                let fwd = jvp_grad_first(&jaxpr, &[Value::scalar_f64(x)])
                    .expect("jvp grad");
                let rev = grad_first(&jaxpr, &[Value::scalar_f64(x)])
                    .expect("vjp grad");
                prop_assert!(
                    (fwd - rev).abs() < 1e-8,
                    "x={x}: jvp={fwd}, vjp={rev}"
                );
            }
        }
    }
}
