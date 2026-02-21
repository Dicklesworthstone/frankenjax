#![forbid(unsafe_code)]

use fj_core::{Atom, DType, Jaxpr, Literal, Primitive, Shape, TensorValue, Value, VarId};
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
    params: BTreeMap<String, String>,
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
            params: eqn.params.clone(),
        });
    }

    let outputs = jaxpr
        .outvars
        .iter()
        .map(|var| env.get(var).cloned().ok_or(AdError::MissingVariable(*var)))
        .collect::<Result<Vec<_>, _>>()?;

    Ok((outputs, tape, env))
}

// ── Tensor-aware value arithmetic helpers ──────────────────────────

fn value_add(a: &Value, b: &Value) -> Result<Value, AdError> {
    eval_primitive(Primitive::Add, &[a.clone(), b.clone()], &BTreeMap::new())
        .map_err(|e| AdError::EvalFailed(e.to_string()))
}

fn zeros_like(v: &Value) -> Value {
    match v {
        Value::Scalar(lit) => match lit {
            Literal::I64(_) => Value::scalar_i64(0),
            Literal::Bool(_) => Value::scalar_f64(0.0),
            Literal::F64Bits(_) => Value::scalar_f64(0.0),
        },
        Value::Tensor(t) => {
            let (zero_lit, out_dtype) = match t.dtype {
                DType::I64 | DType::I32 => (Literal::I64(0), DType::I64),
                DType::Bool => (Literal::from_f64(0.0), DType::F64),
                DType::F64 | DType::F32 => (Literal::from_f64(0.0), DType::F64),
            };
            let elements = vec![zero_lit; t.elements.len()];
            Value::Tensor(
                TensorValue::new(out_dtype, t.shape.clone(), elements)
                    .expect("zeros_like should never fail for valid tensor shape"),
            )
        }
    }
}

fn ones_like(v: &Value) -> Value {
    match v {
        Value::Scalar(lit) => match lit {
            Literal::I64(_) => Value::scalar_f64(1.0),
            Literal::Bool(_) => Value::scalar_f64(1.0),
            Literal::F64Bits(_) => Value::scalar_f64(1.0),
        },
        Value::Tensor(t) => {
            let elements = vec![Literal::from_f64(1.0); t.elements.len()];
            Value::Tensor(
                TensorValue::new(DType::F64, t.shape.clone(), elements)
                    .expect("ones_like should never fail for valid tensor shape"),
            )
        }
    }
}

fn scalar_value(x: f64) -> Value {
    Value::scalar_f64(x)
}

fn value_mul(a: &Value, b: &Value) -> Result<Value, AdError> {
    eval_primitive(Primitive::Mul, &[a.clone(), b.clone()], &BTreeMap::new())
        .map_err(|e| AdError::EvalFailed(e.to_string()))
}

fn value_neg(a: &Value) -> Result<Value, AdError> {
    eval_primitive(Primitive::Neg, std::slice::from_ref(a), &BTreeMap::new())
        .map_err(|e| AdError::EvalFailed(e.to_string()))
}

fn value_div(a: &Value, b: &Value) -> Result<Value, AdError> {
    eval_primitive(Primitive::Div, &[a.clone(), b.clone()], &BTreeMap::new())
        .map_err(|e| AdError::EvalFailed(e.to_string()))
}

fn value_sub(a: &Value, b: &Value) -> Result<Value, AdError> {
    eval_primitive(Primitive::Sub, &[a.clone(), b.clone()], &BTreeMap::new())
        .map_err(|e| AdError::EvalFailed(e.to_string()))
}

fn is_zero_value(v: &Value) -> bool {
    match v {
        Value::Scalar(Literal::F64Bits(bits)) => f64::from_bits(*bits) == 0.0,
        Value::Scalar(Literal::I64(val)) => *val == 0,
        Value::Scalar(Literal::Bool(val)) => !val,
        Value::Tensor(t) => t.elements.iter().all(|e| match e {
            Literal::F64Bits(bits) => f64::from_bits(*bits) == 0.0,
            Literal::I64(v) => *v == 0,
            Literal::Bool(v) => !v,
        }),
    }
}

// ── Backward pass (tensor-aware) ───────────────────────────────────

fn backward(
    tape: &Tape,
    output_var: VarId,
    output_cotangent: Value,
    jaxpr: &Jaxpr,
    env: &BTreeMap<VarId, Value>,
) -> Result<Vec<Value>, AdError> {
    let mut adjoints: BTreeMap<VarId, Value> = BTreeMap::new();
    adjoints.insert(output_var, output_cotangent);

    for entry in tape.entries.iter().rev() {
        let g = match adjoints.get(&entry.output) {
            Some(v) => v.clone(),
            None => continue,
        };
        if is_zero_value(&g) {
            continue;
        }

        let cotangents = vjp(entry.primitive, &entry.input_values, &g, &entry.params)?;

        for (var_id, cot) in entry.inputs.iter().zip(cotangents.into_iter()) {
            if var_id.0 == u32::MAX {
                continue; // literal sentinel
            }
            let entry = adjoints.entry(*var_id);
            match entry {
                std::collections::btree_map::Entry::Occupied(mut e) => {
                    let new_val = value_add(e.get(), &cot)?;
                    e.insert(new_val);
                }
                std::collections::btree_map::Entry::Vacant(e) => {
                    e.insert(cot);
                }
            }
        }
    }

    let grads = jaxpr
        .invars
        .iter()
        .map(|var| {
            adjoints
                .remove(var)
                .unwrap_or_else(|| zeros_like(env.get(var).unwrap_or(&Value::scalar_f64(0.0))))
        })
        .collect();

    Ok(grads)
}

// ── VJP rules (tensor-aware) ──────────────────────────────────────

fn vjp(
    primitive: Primitive,
    inputs: &[Value],
    g: &Value,
    params: &BTreeMap<String, String>,
) -> Result<Vec<Value>, AdError> {
    match primitive {
        Primitive::Add => Ok(vec![g.clone(), g.clone()]),
        Primitive::Sub => Ok(vec![g.clone(), value_neg(g)?]),
        Primitive::Mul => {
            let a = &inputs[0];
            let b = &inputs[1];
            Ok(vec![value_mul(g, b)?, value_mul(g, a)?])
        }
        Primitive::Neg => Ok(vec![value_neg(g)?]),
        Primitive::Abs => {
            let x = &inputs[0];
            let sign = eval_primitive(Primitive::Sign, std::slice::from_ref(x), &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![value_mul(g, &sign)?])
        }
        Primitive::Max => {
            let a = &inputs[0];
            let b = &inputs[1];
            let cond = eval_primitive(Primitive::Ge, &[a.clone(), b.clone()], &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let ga = eval_primitive(
                Primitive::Select,
                &[cond.clone(), g.clone(), zeros_like(g)],
                &BTreeMap::new(),
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let gb = eval_primitive(
                Primitive::Select,
                &[cond, zeros_like(g), g.clone()],
                &BTreeMap::new(),
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![ga, gb])
        }
        Primitive::Min => {
            let a = &inputs[0];
            let b = &inputs[1];
            let cond = eval_primitive(Primitive::Le, &[a.clone(), b.clone()], &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let ga = eval_primitive(
                Primitive::Select,
                &[cond.clone(), g.clone(), zeros_like(g)],
                &BTreeMap::new(),
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let gb = eval_primitive(
                Primitive::Select,
                &[cond, zeros_like(g), g.clone()],
                &BTreeMap::new(),
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![ga, gb])
        }
        Primitive::Pow => {
            let a = &inputs[0];
            let b = &inputs[1];
            // d/da(a^b) = b * a^(b-1), d/db(a^b) = a^b * ln(a)
            let b_minus_1 = value_sub(b, &ones_like(b))?;
            let a_pow_bm1 =
                eval_primitive(Primitive::Pow, &[a.clone(), b_minus_1], &BTreeMap::new())
                    .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let da = value_mul(g, &value_mul(b, &a_pow_bm1)?)?;

            let a_pow_b = eval_primitive(Primitive::Pow, &[a.clone(), b.clone()], &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let ln_a = eval_primitive(Primitive::Log, std::slice::from_ref(a), &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let db = value_mul(g, &value_mul(&a_pow_b, &ln_a)?)?;
            Ok(vec![da, db])
        }
        Primitive::Exp => {
            let x = &inputs[0];
            let exp_x = eval_primitive(Primitive::Exp, std::slice::from_ref(x), &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![value_mul(g, &exp_x)?])
        }
        Primitive::Log => {
            let x = &inputs[0];
            Ok(vec![value_div(g, x)?])
        }
        Primitive::Sqrt => {
            let x = &inputs[0];
            let sqrt_x = eval_primitive(Primitive::Sqrt, std::slice::from_ref(x), &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let two_sqrt = value_mul(&scalar_value(2.0), &sqrt_x)?;
            Ok(vec![value_div(g, &two_sqrt)?])
        }
        Primitive::Rsqrt => {
            let x = &inputs[0];
            // d/dx(x^(-1/2)) = -0.5 * x^(-3/2)
            let x_pow_neg1p5 = eval_primitive(
                Primitive::Pow,
                &[x.clone(), scalar_value(-1.5)],
                &BTreeMap::new(),
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![value_mul(
                g,
                &value_mul(&scalar_value(-0.5), &x_pow_neg1p5)?,
            )?])
        }
        Primitive::Floor | Primitive::Ceil | Primitive::Round => Ok(vec![zeros_like(&inputs[0])]),
        Primitive::Sin => {
            let x = &inputs[0];
            let cos_x = eval_primitive(Primitive::Cos, std::slice::from_ref(x), &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![value_mul(g, &cos_x)?])
        }
        Primitive::Cos => {
            let x = &inputs[0];
            let sin_x = eval_primitive(Primitive::Sin, std::slice::from_ref(x), &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![value_neg(&value_mul(g, &sin_x)?)?])
        }
        Primitive::Tan => {
            // d/dx tan(x) = 1/cos²(x) = 1 + tan²(x)
            let x = &inputs[0];
            let cos_x = eval_primitive(Primitive::Cos, std::slice::from_ref(x), &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let cos2 = value_mul(&cos_x, &cos_x)?;
            Ok(vec![value_div(g, &cos2)?])
        }
        Primitive::Asin => {
            // d/dx asin(x) = 1/sqrt(1-x²)
            let x = &inputs[0];
            let x2 = value_mul(x, x)?;
            let one_minus_x2 = value_sub(&ones_like(x), &x2)?;
            let denom = eval_primitive(Primitive::Sqrt, &[one_minus_x2], &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![value_div(g, &denom)?])
        }
        Primitive::Acos => {
            // d/dx acos(x) = -1/sqrt(1-x²)
            let x = &inputs[0];
            let x2 = value_mul(x, x)?;
            let one_minus_x2 = value_sub(&ones_like(x), &x2)?;
            let denom = eval_primitive(Primitive::Sqrt, &[one_minus_x2], &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![value_neg(&value_div(g, &denom)?)?])
        }
        Primitive::Atan => {
            // d/dx atan(x) = 1/(1+x²)
            let x = &inputs[0];
            let x2 = value_mul(x, x)?;
            let denom = value_add(&ones_like(x), &x2)?;
            Ok(vec![value_div(g, &denom)?])
        }
        Primitive::Sinh => {
            // d/dx sinh(x) = cosh(x)
            let x = &inputs[0];
            let cosh_x = eval_primitive(Primitive::Cosh, std::slice::from_ref(x), &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![value_mul(g, &cosh_x)?])
        }
        Primitive::Cosh => {
            // d/dx cosh(x) = sinh(x)
            let x = &inputs[0];
            let sinh_x = eval_primitive(Primitive::Sinh, std::slice::from_ref(x), &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![value_mul(g, &sinh_x)?])
        }
        Primitive::Tanh => {
            // d/dx tanh(x) = 1 - tanh²(x)
            let x = &inputs[0];
            let tanh_x = eval_primitive(Primitive::Tanh, std::slice::from_ref(x), &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let tanh2 = value_mul(&tanh_x, &tanh_x)?;
            let one_minus = value_sub(&ones_like(x), &tanh2)?;
            Ok(vec![value_mul(g, &one_minus)?])
        }
        Primitive::Expm1 => {
            // d/dx expm1(x) = exp(x)
            let x = &inputs[0];
            let exp_x = eval_primitive(Primitive::Exp, std::slice::from_ref(x), &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![value_mul(g, &exp_x)?])
        }
        Primitive::Log1p => {
            // d/dx log1p(x) = 1/(1+x)
            let x = &inputs[0];
            let one_plus = value_add(&ones_like(x), x)?;
            Ok(vec![value_div(g, &one_plus)?])
        }
        Primitive::Sign => {
            // Sign is piecewise constant, gradient is 0
            Ok(vec![zeros_like(&inputs[0])])
        }
        Primitive::Square => {
            // d/dx x² = 2x
            let x = &inputs[0];
            let two_x = value_mul(&scalar_value(2.0), x)?;
            Ok(vec![value_mul(g, &two_x)?])
        }
        Primitive::Reciprocal => {
            // d/dx (1/x) = -1/x²
            let x = &inputs[0];
            let x2 = value_mul(x, x)?;
            let neg_inv = value_neg(&value_div(&ones_like(x), &x2)?)?;
            Ok(vec![value_mul(g, &neg_inv)?])
        }
        Primitive::Logistic => {
            // d/dx σ(x) = σ(x)(1-σ(x))
            let x = &inputs[0];
            let sig = eval_primitive(
                Primitive::Logistic,
                std::slice::from_ref(x),
                &BTreeMap::new(),
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let one_minus_sig = value_sub(&ones_like(x), &sig)?;
            let factor = value_mul(&sig, &one_minus_sig)?;
            Ok(vec![value_mul(g, &factor)?])
        }
        Primitive::Erf => {
            // d/dx erf(x) = 2/sqrt(π) * exp(-x²)
            let x = &inputs[0];
            let x2 = value_mul(x, x)?;
            let neg_x2 = value_neg(&x2)?;
            let exp_neg_x2 = eval_primitive(Primitive::Exp, &[neg_x2], &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let coeff = scalar_value(2.0 / std::f64::consts::PI.sqrt());
            let factor = value_mul(&coeff, &exp_neg_x2)?;
            Ok(vec![value_mul(g, &factor)?])
        }
        Primitive::Erfc => {
            // d/dx erfc(x) = -2/sqrt(π) * exp(-x²)
            let x = &inputs[0];
            let x2 = value_mul(x, x)?;
            let neg_x2 = value_neg(&x2)?;
            let exp_neg_x2 = eval_primitive(Primitive::Exp, &[neg_x2], &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let coeff = scalar_value(-2.0 / std::f64::consts::PI.sqrt());
            let factor = value_mul(&coeff, &exp_neg_x2)?;
            Ok(vec![value_mul(g, &factor)?])
        }
        Primitive::Div => {
            // d/da (a/b) = 1/b, d/db (a/b) = -a/b²
            let a = &inputs[0];
            let b = &inputs[1];
            let da = value_div(g, b)?;
            let b2 = value_mul(b, b)?;
            let db = value_neg(&value_mul(g, &value_div(a, &b2)?)?)?;
            Ok(vec![da, db])
        }
        Primitive::Rem => {
            // d/da (a%b) = 1, d/db (a%b) = -floor(a/b)
            let a = &inputs[0];
            let b = &inputs[1];
            let ratio = value_div(a, b)?;
            let floor_ratio = eval_primitive(Primitive::Floor, &[ratio], &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let db = value_neg(&value_mul(g, &floor_ratio)?)?;
            Ok(vec![g.clone(), db])
        }
        Primitive::Atan2 => {
            // d/da atan2(a,b) = b/(a²+b²), d/db = -a/(a²+b²)
            let a = &inputs[0];
            let b = &inputs[1];
            let a2 = value_mul(a, a)?;
            let b2 = value_mul(b, b)?;
            let denom = value_add(&a2, &b2)?;
            let da = value_mul(g, &value_div(b, &denom)?)?;
            let db = value_neg(&value_mul(g, &value_div(a, &denom)?)?)?;
            Ok(vec![da, db])
        }
        Primitive::Select => {
            // select(cond, on_true, on_false): grad to on_true where cond, else to on_false
            let cond = &inputs[0];
            let g_true = eval_primitive(
                Primitive::Select,
                &[cond.clone(), g.clone(), zeros_like(g)],
                &BTreeMap::new(),
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let g_false = eval_primitive(
                Primitive::Select,
                &[cond.clone(), zeros_like(g), g.clone()],
                &BTreeMap::new(),
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![zeros_like(cond), g_true, g_false])
        }
        Primitive::ReduceSum => {
            // VJP of reduce_sum: broadcast g back to input shape
            let input = &inputs[0];
            match input {
                Value::Scalar(_) => Ok(vec![g.clone()]),
                Value::Tensor(t) => {
                    let g_scalar = match g.as_f64_scalar() {
                        Some(v) => v,
                        None => {
                            let kept_axes = if let Some(axes_str) = params.get("axes") {
                                let reduced_axes: Vec<usize> = axes_str
                                    .split(',')
                                    .filter_map(|s| s.trim().parse().ok())
                                    .collect();
                                let rank = t.shape.rank();
                                (0..rank)
                                    .filter(|a| !reduced_axes.contains(a))
                                    .collect::<Vec<_>>()
                            } else {
                                vec![]
                            };
                            return Ok(vec![broadcast_g_to_shape_with_axes(
                                g, &t.shape, &kept_axes,
                            )?]);
                        }
                    };
                    let elements = vec![Literal::from_f64(g_scalar); t.elements.len()];
                    Ok(vec![Value::Tensor(
                        TensorValue::new(DType::F64, t.shape.clone(), elements)
                            .map_err(|e| AdError::EvalFailed(e.to_string()))?,
                    )])
                }
            }
        }
        Primitive::ReduceMax | Primitive::ReduceMin => {
            // Subgradient: pass gradient only to argmax/argmin element(s)
            let input = &inputs[0];
            match input {
                Value::Scalar(_) => Ok(vec![g.clone()]),
                Value::Tensor(t) => {
                    let output_val = eval_primitive(primitive, inputs, params)
                        .map_err(|e| AdError::EvalFailed(e.to_string()))?;

                    let kept_axes = if let Some(axes_str) = params.get("axes") {
                        let reduced_axes: Vec<usize> = axes_str
                            .split(',')
                            .filter_map(|s| s.trim().parse().ok())
                            .collect();
                        let rank = t.shape.rank();
                        (0..rank)
                            .filter(|a| !reduced_axes.contains(a))
                            .collect::<Vec<_>>()
                    } else {
                        vec![]
                    };

                    let out_bcast =
                        broadcast_g_to_shape_with_axes(&output_val, &t.shape, &kept_axes)?;
                    let g_bcast = broadcast_g_to_shape_with_axes(g, &t.shape, &kept_axes)?;

                    // indicator mask: t == out_bcast
                    let mask = eval_primitive(
                        Primitive::Eq,
                        &[Value::Tensor(t.clone()), out_bcast],
                        &BTreeMap::new(),
                    )
                    .map_err(|e| AdError::EvalFailed(e.to_string()))?;

                    // convert mask (bool) to f64 mask: select(mask, 1.0, 0.0)
                    // eval_select supports tensor cond + scalar values via broadcasting
                    let f64_mask = eval_primitive(
                        Primitive::Select,
                        &[mask, scalar_value(1.0), scalar_value(0.0)],
                        &BTreeMap::new(),
                    )
                    .map_err(|e| AdError::EvalFailed(e.to_string()))?;

                    // count ties per reduction window: reduce_sum(f64_mask, axes=...)
                    let tie_counts = eval_primitive(
                        Primitive::ReduceSum,
                        std::slice::from_ref(&f64_mask),
                        params,
                    )
                    .map_err(|e| AdError::EvalFailed(e.to_string()))?;

                    // broadcast tie_counts back
                    let tie_counts_bcast =
                        broadcast_g_to_shape_with_axes(&tie_counts, &t.shape, &kept_axes)?;

                    // share = g_bcast / tie_counts_bcast
                    let share = eval_primitive(
                        Primitive::Div,
                        &[g_bcast, tie_counts_bcast],
                        &BTreeMap::new(),
                    )
                    .map_err(|e| AdError::EvalFailed(e.to_string()))?;

                    // final gradient = f64_mask * share
                    let grad = eval_primitive(Primitive::Mul, &[f64_mask, share], &BTreeMap::new())
                        .map_err(|e| AdError::EvalFailed(e.to_string()))?;

                    Ok(vec![grad])
                }
            }
        }
        Primitive::ReduceProd => {
            // VJP of prod(x) wrt x_i = prod(x) / x_i * g
            let input = &inputs[0];
            match input {
                Value::Scalar(_) => Ok(vec![g.clone()]),
                Value::Tensor(t) => {
                    if params.contains_key("axes") {
                        return Err(AdError::EvalFailed(
                            "partial reduce_prod VJP is unsupported in V1".to_owned(),
                        ));
                    }
                    let output_val = eval_primitive(primitive, inputs, params)
                        .map_err(|e| AdError::EvalFailed(e.to_string()))?;
                    let prod_val = output_val
                        .as_f64_scalar()
                        .ok_or_else(|| AdError::EvalFailed("non-scalar reduce output".into()))?;
                    let g_scalar = g
                        .as_f64_scalar()
                        .ok_or_else(|| AdError::EvalFailed("non-scalar gradient".into()))?;

                    let elements: Vec<Literal> = t
                        .elements
                        .iter()
                        .enumerate()
                        .map(|(idx, lit)| {
                            let v = lit.as_f64().unwrap_or(1.0);
                            if v.abs() < f64::EPSILON {
                                // Handle zero: recompute product excluding this element
                                let partial_prod: f64 = t
                                    .elements
                                    .iter()
                                    .enumerate()
                                    .filter(|(j, _)| *j != idx)
                                    .map(|(_, l)| l.as_f64().unwrap_or(1.0))
                                    .product();
                                Literal::from_f64(g_scalar * partial_prod)
                            } else {
                                Literal::from_f64(g_scalar * prod_val / v)
                            }
                        })
                        .collect();

                    Ok(vec![Value::Tensor(
                        TensorValue::new(DType::F64, t.shape.clone(), elements)
                            .map_err(|e| AdError::EvalFailed(e.to_string()))?,
                    )])
                }
            }
        }
        Primitive::Dot => {
            let a = &inputs[0];
            let b = &inputs[1];
            // Scalar case: dot(a,b) = a*b
            // Vector case: dot(a,b) = sum(a*b), cotangents = (g*b, g*a)
            Ok(vec![value_mul(g, b)?, value_mul(g, a)?])
        }
        // Comparison ops have zero gradient
        Primitive::Eq
        | Primitive::Ne
        | Primitive::Lt
        | Primitive::Le
        | Primitive::Gt
        | Primitive::Ge => Ok(vec![zeros_like(&inputs[0]), zeros_like(&inputs[1])]),
        // Shape manipulation VJP rules
        Primitive::Reshape => {
            // VJP of reshape: reshape g back to original shape
            let original_shape = match &inputs[0] {
                Value::Scalar(_) => Shape::scalar(),
                Value::Tensor(t) => t.shape.clone(),
            };
            let mut reshape_params = BTreeMap::new();
            reshape_params.insert(
                "new_shape".to_owned(),
                original_shape
                    .dims
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            );
            let reshaped_g =
                eval_primitive(Primitive::Reshape, std::slice::from_ref(g), &reshape_params)
                    .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![reshaped_g])
        }
        Primitive::Transpose => {
            // VJP of transpose: transpose g with inverse permutation
            let rank = match &inputs[0] {
                Value::Scalar(_) => 0,
                Value::Tensor(t) => t.shape.rank(),
            };
            let perm = if let Some(raw) = params.get("permutation") {
                raw.split(',')
                    .map(|s| {
                        s.trim().parse::<usize>().map_err(|_| {
                            AdError::EvalFailed(format!("invalid permutation index: {s:?}"))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                (0..rank).rev().collect()
            };
            // Compute inverse permutation (validate bounds)
            let mut inv_perm = vec![0_usize; perm.len()];
            for (i, &p) in perm.iter().enumerate() {
                if p >= inv_perm.len() {
                    return Err(AdError::EvalFailed(format!(
                        "transpose permutation index {p} out of bounds for rank {}",
                        inv_perm.len()
                    )));
                }
                inv_perm[p] = i;
            }
            let mut tp = BTreeMap::new();
            tp.insert(
                "permutation".to_owned(),
                inv_perm
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            );
            let transposed_g = eval_primitive(Primitive::Transpose, std::slice::from_ref(g), &tp)
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![transposed_g])
        }
        Primitive::BroadcastInDim => {
            // VJP of broadcast_in_dim: reduce_sum g along broadcast axes,
            // then reshape to the original input shape.
            let input = &inputs[0];
            let input_shape = match input {
                Value::Scalar(_) => Shape::scalar(),
                Value::Tensor(t) => t.shape.clone(),
            };
            // If input was scalar, sum everything
            if input_shape.rank() == 0 {
                let reduced = eval_primitive(
                    Primitive::ReduceSum,
                    std::slice::from_ref(g),
                    &BTreeMap::new(),
                )
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
                return Ok(vec![reduced]);
            }

            // Parse broadcast_dimensions: which output axes correspond to input axes
            let out_rank = match g {
                Value::Scalar(_) => 0,
                Value::Tensor(t) => t.shape.rank(),
            };

            let broadcast_dims: Vec<usize> =
                if let Some(bd_str) = params.get("broadcast_dimensions") {
                    bd_str
                        .split(',')
                        .filter_map(|v| v.trim().parse().ok())
                        .collect()
                } else {
                    // Default: input axes map to trailing output axes
                    let in_rank = input_shape.rank();
                    (out_rank - in_rank..out_rank).collect()
                };

            // Determine which output axes are "broadcast axes" that need reduction:
            // 1. Output axes not mapped from any input axis
            // 2. Output axes mapped from a size-1 input dim (implicit broadcast)
            let mut reduce_axes: Vec<usize> = Vec::new();
            for out_axis in 0..out_rank {
                if let Some(pos) = broadcast_dims.iter().position(|&d| d == out_axis) {
                    // This output axis maps to input axis `pos`
                    if input_shape.dims[pos] == 1 {
                        // Size-1 input dim was broadcast
                        reduce_axes.push(out_axis);
                    }
                } else {
                    // This output axis has no input correspondence — was broadcast
                    reduce_axes.push(out_axis);
                }
            }

            // If no axes need reduction, g passes through directly
            if reduce_axes.is_empty() {
                return Ok(vec![g.clone()]);
            }

            // Reduce g along broadcast axes (one axis at a time, from highest to lowest
            // to keep axis indices stable)
            let mut current = g.clone();
            for &axis in reduce_axes.iter().rev() {
                let mut reduce_params = BTreeMap::new();
                reduce_params.insert("axes".into(), axis.to_string());
                current = eval_primitive(
                    Primitive::ReduceSum,
                    std::slice::from_ref(&current),
                    &reduce_params,
                )
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            }

            // Reshape to original input shape if needed
            let current_shape = match &current {
                Value::Scalar(_) => Shape::scalar(),
                Value::Tensor(t) => t.shape.clone(),
            };
            if current_shape != input_shape {
                let mut reshape_params = BTreeMap::new();
                reshape_params.insert(
                    "shape".into(),
                    input_shape
                        .dims
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                );
                current = eval_primitive(
                    Primitive::Reshape,
                    std::slice::from_ref(&current),
                    &reshape_params,
                )
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            }

            Ok(vec![current])
        }
        Primitive::Slice => {
            // VJP of slice: embed g into a zero tensor at the slice offsets
            let input = &inputs[0];
            match input {
                Value::Scalar(_) => Ok(vec![g.clone()]),
                Value::Tensor(t) => {
                    // Parse start indices from params
                    let starts: Vec<usize> = params
                        .get("start_indices")
                        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
                        .unwrap_or_default();

                    // Create zero tensor of original shape
                    let total = t.elements.len();
                    let mut result_elements = vec![Literal::from_f64(0.0); total];

                    // Get the gradient tensor elements
                    let g_elements: Vec<f64> = match g {
                        Value::Scalar(lit) => vec![lit.as_f64().unwrap_or(0.0)],
                        Value::Tensor(gt) => gt
                            .elements
                            .iter()
                            .map(|l| l.as_f64().unwrap_or(0.0))
                            .collect(),
                    };

                    let rank = t.shape.rank();
                    let mut in_strides = vec![1_usize; rank];
                    for i in (0..rank.saturating_sub(1)).rev() {
                        in_strides[i] = in_strides[i + 1] * t.shape.dims[i + 1] as usize;
                    }

                    let g_shape = match g {
                        Value::Scalar(_) => Shape::scalar(),
                        Value::Tensor(gt) => gt.shape.clone(),
                    };
                    let g_dims = &g_shape.dims;
                    let mut out_coords = vec![0_usize; rank];

                    for &gval in g_elements.iter() {
                        let mut in_flat = 0_usize;
                        for ax in 0..rank {
                            let start = *starts.get(ax).unwrap_or(&0);
                            in_flat += (out_coords[ax] + start) * in_strides[ax];
                        }
                        if in_flat < total {
                            result_elements[in_flat] = Literal::from_f64(gval);
                        }

                        if rank > 0 {
                            for ax in (0..rank).rev() {
                                out_coords[ax] += 1;
                                if out_coords[ax] < g_dims[ax] as usize {
                                    break;
                                }
                                out_coords[ax] = 0;
                            }
                        }
                    }

                    Ok(vec![Value::Tensor(
                        TensorValue::new(DType::F64, t.shape.clone(), result_elements)
                            .map_err(|e| AdError::EvalFailed(e.to_string()))?,
                    )])
                }
            }
        }
        Primitive::Concatenate => {
            // VJP of concatenate: split g into slices for each input along the concat dimension
            let axis: usize = params
                .get("dimension")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);

            let g_rank = match g {
                Value::Tensor(t) => t.shape.rank(),
                Value::Scalar(_) => 0,
            };

            if g_rank == 0 {
                let mut grad_inputs = Vec::with_capacity(inputs.len());
                for _ in inputs {
                    grad_inputs.push(g.clone());
                }
                return Ok(grad_inputs);
            }

            let g_dims: Vec<usize> = match g {
                Value::Tensor(t) => t.shape.dims.iter().map(|&d| d as usize).collect(),
                Value::Scalar(_) => vec![],
            };

            let mut grad_inputs = Vec::with_capacity(inputs.len());
            let mut current_offset = 0;

            for inp in inputs {
                let inp_shape = match inp {
                    Value::Scalar(_) => Shape::scalar(),
                    Value::Tensor(t) => t.shape.clone(),
                };

                let slice_size = inp_shape.dims.get(axis).copied().unwrap_or(1) as usize;

                let mut start_indices = vec![0; g_rank];
                let mut limit_indices = g_dims.clone();

                start_indices[axis] = current_offset;
                limit_indices[axis] = current_offset + slice_size;

                let mut slice_params = BTreeMap::new();
                slice_params.insert(
                    "start_indices".to_owned(),
                    start_indices
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                );
                slice_params.insert(
                    "limit_indices".to_owned(),
                    limit_indices
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                );

                let slice_g =
                    eval_primitive(Primitive::Slice, std::slice::from_ref(g), &slice_params)
                        .map_err(|e| AdError::EvalFailed(e.to_string()))?;

                grad_inputs.push(slice_g);
                current_offset += slice_size;
            }
            Ok(grad_inputs)
        }
        Primitive::Gather => {
            // VJP of gather: scatter the gradient back into a zero tensor
            // of the original operand shape.
            let operand = &inputs[0];
            let indices = &inputs[1];
            let operand_shape = match operand {
                Value::Scalar(_) => Shape::scalar(),
                Value::Tensor(t) => t.shape.clone(),
            };
            // Create zero tensor of original operand shape
            let total: usize = operand_shape.dims.iter().map(|d| *d as usize).product();
            let zero_elements = vec![Literal::from_f64(0.0); total];
            let zero_operand = Value::Tensor(
                TensorValue::new(DType::F64, operand_shape, zero_elements)
                    .map_err(|e| AdError::EvalFailed(e.to_string()))?,
            );
            let mut scatter_params = BTreeMap::new();
            scatter_params.insert("mode".to_owned(), "add".to_owned());
            // Scatter gradient into the zero tensor at the gathered indices
            let scattered = eval_primitive(
                Primitive::Scatter,
                &[zero_operand, indices.clone(), g.clone()],
                &scatter_params,
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            // Gradient w.r.t. operand is the scattered result; indices have no gradient
            Ok(vec![scattered])
        }
        Primitive::Scatter => {
            // VJP of scatter: gradient of operand = g with scattered positions zeroed out,
            // gradient of updates = gather g at the scattered indices.
            let indices = &inputs[1];
            // Build slice_sizes for gather: [1, g.dims[1], g.dims[2], ...]
            let g_shape = match g {
                Value::Tensor(t) => &t.shape,
                Value::Scalar(_) => {
                    return Err(AdError::EvalFailed(
                        "scatter VJP requires tensor gradient".into(),
                    ));
                }
            };
            let mut gather_params = BTreeMap::new();
            let slice_sizes_str = std::iter::once("1".to_owned())
                .chain(g_shape.dims[1..].iter().map(|d| d.to_string()))
                .collect::<Vec<_>>()
                .join(",");
            gather_params.insert("slice_sizes".to_owned(), slice_sizes_str);
            // Grad w.r.t. updates: gather from g at the scatter indices
            let grad_updates = eval_primitive(
                Primitive::Gather,
                &[g.clone(), indices.clone()],
                &gather_params,
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;

            // Grad w.r.t. operand depends on the scatter mode.
            let mode = params
                .get("mode")
                .map(|s| s.as_str())
                .unwrap_or("overwrite");
            let grad_operand = if mode == "add" {
                // If it was an add, the operand passes gradient 1.0 everywhere.
                g.clone()
            } else {
                // If overwrite, operand gradient at scattered positions is zero.
                zero_scattered_positions(g, indices)?
            };

            Ok(vec![grad_operand, grad_updates])
        }
    }
}

/// Zero out the slices in `g` at the positions indicated by `indices`.
/// Used in Scatter VJP: the operand gradient must be zero at positions
/// where scatter overwrote the operand's values.
fn zero_scattered_positions(g: &Value, indices: &Value) -> Result<Value, AdError> {
    let g_tensor = match g {
        Value::Tensor(t) => t,
        Value::Scalar(_) => return Ok(Value::Scalar(Literal::from_f64(0.0))),
    };

    let index_vals: Vec<usize> = match indices {
        Value::Scalar(lit) => {
            let v = lit
                .as_f64()
                .ok_or_else(|| AdError::EvalFailed("non-numeric index".into()))?;
            if !v.is_finite() || v < 0.0 {
                return Err(AdError::EvalFailed(format!(
                    "scatter index must be a non-negative finite number, got {v}"
                )));
            }
            vec![v as usize]
        }
        Value::Tensor(t) => t
            .elements
            .iter()
            .map(|lit| {
                let v = lit
                    .as_f64()
                    .ok_or_else(|| AdError::EvalFailed("non-numeric index".into()))?;
                if !v.is_finite() || v < 0.0 {
                    return Err(AdError::EvalFailed(format!(
                        "scatter index must be a non-negative finite number, got {v}"
                    )));
                }
                Ok(v as usize)
            })
            .collect::<Result<_, _>>()?,
    };

    let rank = g_tensor.shape.rank();
    if rank == 0 {
        return Ok(Value::Scalar(Literal::from_f64(0.0)));
    }

    let dims = &g_tensor.shape.dims;
    // Elements per axis-0 slice: product of dims[1..]
    let slice_elems: usize = dims[1..]
        .iter()
        .map(|d| *d as usize)
        .product::<usize>()
        .max(1);

    let mut elements = g_tensor.elements.clone();
    for &idx in &index_vals {
        if idx < dims[0] as usize {
            let base = idx * slice_elems;
            for j in 0..slice_elems {
                if base + j < elements.len() {
                    elements[base + j] = Literal::from_f64(0.0);
                }
            }
        }
    }

    Ok(Value::Tensor(
        TensorValue::new(g_tensor.dtype, g_tensor.shape.clone(), elements)
            .map_err(|e| AdError::EvalFailed(e.to_string()))?,
    ))
}

#[allow(dead_code)]
fn broadcast_g_to_shape(g: &Value, target_shape: &Shape) -> Result<Value, AdError> {
    let out_rank = target_shape.rank();
    let in_rank = match g {
        Value::Scalar(_) => 0,
        Value::Tensor(t) => t.shape.rank(),
    };
    let kept_axes: Vec<usize> = if in_rank > out_rank {
        vec![]
    } else {
        (out_rank - in_rank..out_rank).collect()
    };
    broadcast_g_to_shape_with_axes(g, target_shape, &kept_axes)
}

fn broadcast_g_to_shape_with_axes(
    g: &Value,
    target_shape: &Shape,
    broadcast_dimensions: &[usize],
) -> Result<Value, AdError> {
    let g_scalar = g.as_f64_scalar();
    if let Some(v) = g_scalar {
        let count = target_shape.element_count().unwrap_or(1) as usize;
        let elements = vec![Literal::from_f64(v); count];
        return Ok(Value::Tensor(
            TensorValue::new(DType::F64, target_shape.clone(), elements)
                .map_err(|e| AdError::EvalFailed(e.to_string()))?,
        ));
    }
    // If g is already a tensor, attempt broadcast via eval_primitive
    let mut params = BTreeMap::new();
    params.insert(
        "shape".to_owned(),
        target_shape
            .dims
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(","),
    );
    if !broadcast_dimensions.is_empty() {
        params.insert(
            "broadcast_dimensions".to_owned(),
            broadcast_dimensions
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    eval_primitive(Primitive::BroadcastInDim, std::slice::from_ref(g), &params)
        .map_err(|e| AdError::EvalFailed(e.to_string()))
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
        Primitive::Add => Ok(tangents[0] + tangents[1]),
        Primitive::Sub => Ok(tangents[0] - tangents[1]),
        Primitive::Mul => {
            let a = to_f64(&primals[0])?;
            let b = to_f64(&primals[1])?;
            Ok(tangents[0] * b + a * tangents[1])
        }
        Primitive::Neg => Ok(-tangents[0]),
        Primitive::Abs => {
            let x = to_f64(&primals[0])?;
            Ok(if x >= 0.0 { tangents[0] } else { -tangents[0] })
        }
        Primitive::Max => {
            let a = to_f64(&primals[0])?;
            let b = to_f64(&primals[1])?;
            Ok(if a >= b { tangents[0] } else { tangents[1] })
        }
        Primitive::Min => {
            let a = to_f64(&primals[0])?;
            let b = to_f64(&primals[1])?;
            Ok(if a <= b { tangents[0] } else { tangents[1] })
        }
        Primitive::Pow => {
            let a = to_f64(&primals[0])?;
            let b = to_f64(&primals[1])?;
            let da = if tangents[0] == 0.0 {
                0.0
            } else {
                b * a.powf(b - 1.0) * tangents[0]
            };
            let db = if tangents[1] == 0.0 {
                0.0
            } else {
                a.powf(b) * a.ln() * tangents[1]
            };
            Ok(da + db)
        }
        Primitive::Exp => {
            let x = to_f64(&primals[0])?;
            Ok(x.exp() * tangents[0])
        }
        Primitive::Log => {
            let x = to_f64(&primals[0])?;
            Ok(tangents[0] / x)
        }
        Primitive::Sqrt => {
            let x = to_f64(&primals[0])?;
            Ok(tangents[0] / (2.0 * x.sqrt()))
        }
        Primitive::Rsqrt => {
            let x = to_f64(&primals[0])?;
            Ok(-0.5 * x.powf(-1.5) * tangents[0])
        }
        Primitive::Floor | Primitive::Ceil | Primitive::Round => Ok(0.0),
        Primitive::Sin => {
            let x = to_f64(&primals[0])?;
            Ok(x.cos() * tangents[0])
        }
        Primitive::Cos => {
            let x = to_f64(&primals[0])?;
            Ok(-x.sin() * tangents[0])
        }
        Primitive::Tan => {
            let x = to_f64(&primals[0])?;
            let cos_x = x.cos();
            Ok(tangents[0] / (cos_x * cos_x))
        }
        Primitive::Asin => {
            let x = to_f64(&primals[0])?;
            Ok(tangents[0] / (1.0 - x * x).sqrt())
        }
        Primitive::Acos => {
            let x = to_f64(&primals[0])?;
            Ok(-tangents[0] / (1.0 - x * x).sqrt())
        }
        Primitive::Atan => {
            let x = to_f64(&primals[0])?;
            Ok(tangents[0] / (1.0 + x * x))
        }
        Primitive::Sinh => {
            let x = to_f64(&primals[0])?;
            Ok(x.cosh() * tangents[0])
        }
        Primitive::Cosh => {
            let x = to_f64(&primals[0])?;
            Ok(x.sinh() * tangents[0])
        }
        Primitive::Tanh => {
            let x = to_f64(&primals[0])?;
            let th = x.tanh();
            Ok((1.0 - th * th) * tangents[0])
        }
        Primitive::Expm1 => {
            let x = to_f64(&primals[0])?;
            Ok(x.exp() * tangents[0])
        }
        Primitive::Log1p => {
            let x = to_f64(&primals[0])?;
            Ok(tangents[0] / (1.0 + x))
        }
        Primitive::Sign => Ok(0.0),
        Primitive::Square => {
            let x = to_f64(&primals[0])?;
            Ok(2.0 * x * tangents[0])
        }
        Primitive::Reciprocal => {
            let x = to_f64(&primals[0])?;
            Ok(-tangents[0] / (x * x))
        }
        Primitive::Logistic => {
            let x = to_f64(&primals[0])?;
            let sig = 1.0 / (1.0 + (-x).exp());
            Ok(sig * (1.0 - sig) * tangents[0])
        }
        Primitive::Erf => {
            let x = to_f64(&primals[0])?;
            let coeff = 2.0 / std::f64::consts::PI.sqrt();
            Ok(coeff * (-x * x).exp() * tangents[0])
        }
        Primitive::Erfc => {
            let x = to_f64(&primals[0])?;
            let coeff = -2.0 / std::f64::consts::PI.sqrt();
            Ok(coeff * (-x * x).exp() * tangents[0])
        }
        Primitive::Div => {
            let a = to_f64(&primals[0])?;
            let b = to_f64(&primals[1])?;
            Ok(tangents[0] / b - a * tangents[1] / (b * b))
        }
        Primitive::Rem => {
            let a = to_f64(&primals[0])?;
            let b = to_f64(&primals[1])?;
            Ok(tangents[0] - (a / b).floor() * tangents[1])
        }
        Primitive::Atan2 => {
            let a = to_f64(&primals[0])?;
            let b = to_f64(&primals[1])?;
            let denom = a * a + b * b;
            Ok(b * tangents[0] / denom - a * tangents[1] / denom)
        }
        Primitive::Select => {
            // JVP: select(cond, tangent_true, tangent_false) based on primal cond
            let cond = to_f64(&primals[0])?;
            Ok(if cond != 0.0 {
                tangents[1]
            } else {
                tangents[2]
            })
        }
        Primitive::ReduceSum => Ok(tangents[0]),
        Primitive::ReduceMax | Primitive::ReduceMin => Ok(tangents[0]),
        Primitive::ReduceProd => Ok(tangents[0]),
        Primitive::Dot => {
            let a = to_f64(&primals[0])?;
            let b = to_f64(&primals[1])?;
            Ok(tangents[0] * b + a * tangents[1])
        }
        Primitive::Eq
        | Primitive::Ne
        | Primitive::Lt
        | Primitive::Le
        | Primitive::Gt
        | Primitive::Ge => Ok(0.0),
        // Shape ops: pass tangent through reshape/transpose
        Primitive::Reshape | Primitive::Transpose | Primitive::BroadcastInDim => Ok(tangents[0]),
        Primitive::Slice => Ok(tangents[0]),
        Primitive::Concatenate => Ok(tangents.iter().sum()),
        // Gather/Scatter: tangent passes through the same indexing
        Primitive::Gather => Ok(tangents[0]),
        Primitive::Scatter => Ok(tangents.iter().sum()),
    }
}

// ── Forward-mode gradient (convenience) ────────────────────────────

/// Compute gradient via forward-mode JVP by evaluating with unit tangent.
pub fn jvp_grad_first(jaxpr: &Jaxpr, args: &[Value]) -> Result<f64, AdError> {
    let mut tangents = vec![0.0; args.len()];
    tangents[0] = 1.0;
    let result = jvp(jaxpr, args, &tangents)?;
    result
        .tangents
        .first()
        .copied()
        .ok_or(AdError::NonScalarGradientOutput)
}

/// Compute gradients of a Jaxpr with respect to all inputs (tensor-aware).
pub fn grad_jaxpr(jaxpr: &Jaxpr, args: &[Value]) -> Result<Vec<Value>, AdError> {
    let (outputs, tape, env) = forward_with_tape(jaxpr, args)?;

    let output_val = outputs.first().ok_or(AdError::NonScalarGradientOutput)?;

    // JAX semantics: grad requires scalar-valued function output
    if !matches!(output_val, Value::Scalar(_)) {
        return Err(AdError::NonScalarGradientOutput);
    }

    let seed = Value::scalar_f64(1.0);
    let output_var = jaxpr.outvars[0];
    backward(&tape, output_var, seed, jaxpr, &env)
}

/// Compute gradient with respect to the first input only (convenience wrapper).
/// Returns scalar f64 for backward compatibility.
pub fn grad_first(jaxpr: &Jaxpr, args: &[Value]) -> Result<f64, AdError> {
    let grads = grad_jaxpr(jaxpr, args)?;
    to_f64(&grads[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use fj_core::{ProgramSpec, build_program};

    #[test]
    fn grad_x_squared_at_3() {
        let jaxpr = build_program(ProgramSpec::Square);
        let grads = grad_jaxpr(&jaxpr, &[Value::scalar_f64(3.0)]).expect("grad should succeed");
        let g = to_f64(&grads[0]).unwrap();
        assert!((g - 6.0).abs() < 1e-10, "d/dx(x²) at x=3 = 6, got {}", g);
    }

    #[test]
    fn grad_x_squared_plus_2x_at_3() {
        let jaxpr = build_program(ProgramSpec::SquarePlusLinear);
        let grads = grad_jaxpr(&jaxpr, &[Value::scalar_f64(3.0)]).expect("grad should succeed");
        let g = to_f64(&grads[0]).unwrap();
        assert!((g - 8.0).abs() < 1e-10, "d/dx(x²+2x) at x=3 = 8, got {}", g);
    }

    #[test]
    fn grad_sin_at_zero() {
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
        let g = to_f64(&grads[0]).unwrap();
        assert!(
            (g - 1.0).abs() < 1e-10,
            "d/dx(sin(x)) at x=0 = cos(0) = 1, got {}",
            g
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
        let g = to_f64(&grads[0]).unwrap();
        assert!(
            g.abs() < 1e-10,
            "d/dx(cos(x)) at x=0 = -sin(0) = 0, got {}",
            g
        );
    }

    #[test]
    fn symbolic_matches_numerical() {
        let jaxpr = build_program(ProgramSpec::Square);
        let x = 3.0;
        let symbolic = grad_first(&jaxpr, &[Value::scalar_f64(x)]).expect("symbolic grad");

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

    // ── New primitive AD tests ─────────────────────────────────

    #[test]
    fn grad_tan_at_zero() {
        let jaxpr = make_unary_jaxpr(Primitive::Tan);
        let g = grad_first(&jaxpr, &[Value::scalar_f64(0.0)]).unwrap();
        // d/dx tan(x) at x=0 = 1/cos²(0) = 1
        assert!((g - 1.0).abs() < 1e-10, "got {g}");
    }

    #[test]
    fn grad_tanh_at_zero() {
        let jaxpr = make_unary_jaxpr(Primitive::Tanh);
        let g = grad_first(&jaxpr, &[Value::scalar_f64(0.0)]).unwrap();
        // d/dx tanh(x) at x=0 = 1 - tanh²(0) = 1
        assert!((g - 1.0).abs() < 1e-10, "got {g}");
    }

    #[test]
    fn grad_logistic_at_zero() {
        let jaxpr = make_unary_jaxpr(Primitive::Logistic);
        let g = grad_first(&jaxpr, &[Value::scalar_f64(0.0)]).unwrap();
        // σ(0) = 0.5, σ'(0) = 0.5 * 0.5 = 0.25
        assert!((g - 0.25).abs() < 1e-10, "got {g}");
    }

    #[test]
    fn grad_square_at_3() {
        let jaxpr = make_unary_jaxpr(Primitive::Square);
        let g = grad_first(&jaxpr, &[Value::scalar_f64(3.0)]).unwrap();
        // d/dx x² = 2x, at x=3 -> 6
        assert!((g - 6.0).abs() < 1e-10, "got {g}");
    }

    #[test]
    fn grad_reciprocal_at_2() {
        let jaxpr = make_unary_jaxpr(Primitive::Reciprocal);
        let g = grad_first(&jaxpr, &[Value::scalar_f64(2.0)]).unwrap();
        // d/dx 1/x = -1/x², at x=2 -> -0.25
        assert!((g - (-0.25)).abs() < 1e-10, "got {g}");
    }

    #[test]
    fn grad_div_at_6_3() {
        let jaxpr = make_binary_jaxpr(Primitive::Div);
        let grads = grad_jaxpr(&jaxpr, &[Value::scalar_f64(6.0), Value::scalar_f64(3.0)]).unwrap();
        let da = to_f64(&grads[0]).unwrap();
        let db = to_f64(&grads[1]).unwrap();
        // d/da (a/b) = 1/b = 1/3
        assert!((da - 1.0 / 3.0).abs() < 1e-10, "da = {da}");
        // d/db (a/b) = -a/b² = -6/9 = -2/3
        assert!((db - (-2.0 / 3.0)).abs() < 1e-10, "db = {db}");
    }

    fn make_unary_jaxpr(prim: Primitive) -> Jaxpr {
        use fj_core::{Equation, VarId};
        use smallvec::smallvec;
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: prim,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        )
    }

    fn make_binary_jaxpr(prim: Primitive) -> Jaxpr {
        use fj_core::{Equation, VarId};
        use smallvec::smallvec;
        Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3)],
            vec![Equation {
                primitive: prim,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
            }],
        )
    }

    // ── Tensor VJP tests ─────────────────────────────────────────

    #[test]
    fn vjp_reduce_max_selects_argmax() {
        // reduce_max([1,5,3]) = 5, gradient should go to index 1
        use fj_core::{DType, Literal, Shape, TensorValue};

        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape::vector(3),
                vec![Literal::I64(1), Literal::I64(5), Literal::I64(3)],
            )
            .unwrap(),
        );
        let g = Value::scalar_f64(1.0);
        let params = BTreeMap::new();
        let grads = vjp(Primitive::ReduceMax, &[input], &g, &params).unwrap();
        if let Value::Tensor(t) = &grads[0] {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert!((vals[0] - 0.0).abs() < 1e-10, "non-max should be 0");
            assert!((vals[1] - 1.0).abs() < 1e-10, "max should get gradient");
            assert!((vals[2] - 0.0).abs() < 1e-10, "non-max should be 0");
        } else {
            panic!("expected tensor gradient");
        }
    }

    #[test]
    fn vjp_reduce_prod_correct() {
        // reduce_prod([2,3,4]) = 24
        // grad wrt x_0 = 24/2 = 12, x_1 = 24/3 = 8, x_2 = 24/4 = 6
        use fj_core::{DType, Literal, Shape, TensorValue};

        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape::vector(3),
                vec![Literal::I64(2), Literal::I64(3), Literal::I64(4)],
            )
            .unwrap(),
        );
        let g = Value::scalar_f64(1.0);
        let params = BTreeMap::new();
        let grads = vjp(Primitive::ReduceProd, &[input], &g, &params).unwrap();
        if let Value::Tensor(t) = &grads[0] {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert!((vals[0] - 12.0).abs() < 1e-10, "got {}", vals[0]);
            assert!((vals[1] - 8.0).abs() < 1e-10, "got {}", vals[1]);
            assert!((vals[2] - 6.0).abs() < 1e-10, "got {}", vals[2]);
        } else {
            panic!("expected tensor gradient");
        }
    }

    #[test]
    fn vjp_reduce_min_selects_argmin() {
        use fj_core::{DType, Literal, Shape, TensorValue};

        let input = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape::vector(3),
                vec![Literal::I64(5), Literal::I64(1), Literal::I64(3)],
            )
            .unwrap(),
        );
        let g = Value::scalar_f64(2.0);
        let params = BTreeMap::new();
        let grads = vjp(Primitive::ReduceMin, &[input], &g, &params).unwrap();
        if let Value::Tensor(t) = &grads[0] {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert!((vals[0] - 0.0).abs() < 1e-10);
            assert!((vals[1] - 2.0).abs() < 1e-10, "min should get gradient");
            assert!((vals[2] - 0.0).abs() < 1e-10);
        } else {
            panic!("expected tensor gradient");
        }
    }

    // ── BroadcastInDim VJP tests ───────────────────────────────

    #[test]
    fn vjp_broadcast_scalar_to_vector() {
        // BroadcastInDim: scalar -> [3] sums all gradient elements
        use fj_core::{DType, Literal, Shape, TensorValue};

        let input = Value::scalar_f64(2.0);
        let mut params = BTreeMap::new();
        params.insert("shape".into(), "3".into());

        // Gradient is a [3] tensor
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape::vector(3),
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                ],
            )
            .unwrap(),
        );

        let grads = vjp(Primitive::BroadcastInDim, &[input], &g, &params).unwrap();
        let grad_val = grads[0].as_f64_scalar().unwrap();
        assert!(
            (grad_val - 6.0).abs() < 1e-10,
            "scalar broadcast VJP should sum all: got {grad_val}"
        );
    }

    #[test]
    fn vjp_broadcast_vector_to_matrix() {
        // BroadcastInDim: [3] -> [2,3] with broadcast_dimensions=1
        // Output axes: 0 (broadcast), 1 (maps to input axis 0)
        // VJP should sum along axis 0
        use fj_core::{DType, Literal, Shape, TensorValue};

        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape::vector(3),
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                ],
            )
            .unwrap(),
        );

        let mut params = BTreeMap::new();
        params.insert("shape".into(), "2,3".into());
        params.insert("broadcast_dimensions".into(), "1".into());

        // Gradient is a [2,3] tensor
        // Row 0: [1,2,3], Row 1: [4,5,6]
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0),
                    Literal::from_f64(5.0),
                    Literal::from_f64(6.0),
                ],
            )
            .unwrap(),
        );

        let grads = vjp(Primitive::BroadcastInDim, &[input], &g, &params).unwrap();
        // Sum along axis 0: [1+4, 2+5, 3+6] = [5, 7, 9]
        if let Value::Tensor(t) = &grads[0] {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(t.shape.dims, vec![3], "shape should be [3]");
            assert!((vals[0] - 5.0).abs() < 1e-10, "got {}", vals[0]);
            assert!((vals[1] - 7.0).abs() < 1e-10, "got {}", vals[1]);
            assert!((vals[2] - 9.0).abs() < 1e-10, "got {}", vals[2]);
        } else {
            panic!("expected tensor gradient");
        }
    }

    #[test]
    fn vjp_broadcast_no_broadcast_passthrough() {
        // BroadcastInDim where input shape matches output shape (no actual broadcast)
        // Gradient should pass through unchanged
        use fj_core::{DType, Literal, Shape, TensorValue};

        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape::vector(3),
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                ],
            )
            .unwrap(),
        );

        let mut params = BTreeMap::new();
        params.insert("shape".into(), "3".into());
        params.insert("broadcast_dimensions".into(), "0".into());

        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape::vector(3),
                vec![
                    Literal::from_f64(10.0),
                    Literal::from_f64(20.0),
                    Literal::from_f64(30.0),
                ],
            )
            .unwrap(),
        );

        let grads = vjp(Primitive::BroadcastInDim, &[input], &g, &params).unwrap();
        if let Value::Tensor(t) = &grads[0] {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert!((vals[0] - 10.0).abs() < 1e-10);
            assert!((vals[1] - 20.0).abs() < 1e-10);
            assert!((vals[2] - 30.0).abs() < 1e-10);
        } else {
            panic!("expected tensor gradient");
        }
    }

    #[test]
    fn vjp_scatter_zeros_overwritten_positions() {
        // Scatter overwrites operand[1,:] with updates[0,:].
        // The operand gradient should be zero at the overwritten positions.
        use fj_core::{DType, Literal, Shape, TensorValue};

        // operand: [[1,2],[3,4],[5,6]] shape [3,2]
        let operand = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3, 2] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0),
                    Literal::from_f64(5.0),
                    Literal::from_f64(6.0),
                ],
            )
            .unwrap(),
        );
        // indices: [1] — scatter at index 1
        let indices = Value::Tensor(
            TensorValue::new(DType::I64, Shape::vector(1), vec![Literal::I64(1)]).unwrap(),
        );
        // updates: [[10,20]] shape [1,2]
        let updates = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![1, 2] },
                vec![Literal::from_f64(10.0), Literal::from_f64(20.0)],
            )
            .unwrap(),
        );

        // g = [[1,1],[1,1],[1,1]] (uniform gradient)
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3, 2] },
                vec![Literal::from_f64(1.0); 6],
            )
            .unwrap(),
        );

        let params = BTreeMap::new();
        let grads = vjp(
            Primitive::Scatter,
            &[operand, indices, updates],
            &g,
            &params,
        )
        .unwrap();

        // grad_operand should be [[1,1],[0,0],[1,1]] — zeroed at index 1
        if let Value::Tensor(t) = &grads[0] {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals.len(), 6);
            assert!((vals[0] - 1.0).abs() < 1e-10, "row 0 should pass through");
            assert!((vals[1] - 1.0).abs() < 1e-10, "row 0 should pass through");
            assert!(vals[2].abs() < 1e-10, "row 1 should be zeroed");
            assert!(vals[3].abs() < 1e-10, "row 1 should be zeroed");
            assert!((vals[4] - 1.0).abs() < 1e-10, "row 2 should pass through");
            assert!((vals[5] - 1.0).abs() < 1e-10, "row 2 should pass through");
        } else {
            panic!("expected tensor gradient for operand");
        }

        // grad_updates should be [[1,1]] — gathered from g at index 1
        if let Value::Tensor(t) = &grads[1] {
            let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
            assert_eq!(vals.len(), 2);
            assert!((vals[0] - 1.0).abs() < 1e-10);
            assert!((vals[1] - 1.0).abs() < 1e-10);
        } else {
            panic!("expected tensor gradient for updates");
        }
    }

    // ── Forward-mode JVP tests ──────────────────────────────────

    #[test]
    fn jvp_x_squared_at_3() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = jvp(&jaxpr, &[Value::scalar_f64(3.0)], &[1.0]).expect("jvp should succeed");
        let primal = result.primals[0].as_f64_scalar().unwrap();
        assert!(
            (primal - 9.0).abs() < 1e-10,
            "primal should be 9, got {primal}"
        );
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
        let primal = result.primals[0].as_f64_scalar().unwrap();
        assert!(
            (primal - 15.0).abs() < 1e-10,
            "primal should be 15, got {primal}"
        );
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
        assert!(result.primals[0].as_f64_scalar().unwrap().abs() < 1e-10);
        assert!((result.tangents[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn jvp_scaled_tangent() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = jvp(&jaxpr, &[Value::scalar_f64(3.0)], &[2.0]).expect("jvp should succeed");
        assert!(
            (result.tangents[0] - 12.0).abs() < 1e-10,
            "scaled tangent should be 12, got {}",
            result.tangents[0]
        );
    }

    // ── Systematic numerical gradient verification ──────────────
    // Compares VJP-computed gradients against central finite differences
    // for every differentiable unary primitive at a representative point.

    fn numerical_grad_unary(prim: Primitive, x: f64) -> f64 {
        let eps = 1e-6;
        let no_p = BTreeMap::new();
        let f_plus = eval_primitive(prim, &[Value::scalar_f64(x + eps)], &no_p)
            .unwrap()
            .as_f64_scalar()
            .unwrap();
        let f_minus = eval_primitive(prim, &[Value::scalar_f64(x - eps)], &no_p)
            .unwrap()
            .as_f64_scalar()
            .unwrap();
        (f_plus - f_minus) / (2.0 * eps)
    }

    fn symbolic_grad_unary(prim: Primitive, x: f64) -> f64 {
        let jaxpr = make_unary_jaxpr(prim);
        grad_first(&jaxpr, &[Value::scalar_f64(x)]).unwrap()
    }

    fn assert_grad_matches(prim: Primitive, x: f64, tol: f64) {
        let sym = symbolic_grad_unary(prim, x);
        let num = numerical_grad_unary(prim, x);
        assert!(
            (sym - num).abs() < tol,
            "grad({:?}) at x={x}: symbolic={sym}, numerical={num}, diff={}",
            prim,
            (sym - num).abs()
        );
    }

    #[test]
    fn vjp_vs_numerical_exp() {
        assert_grad_matches(Primitive::Exp, 1.0, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_log() {
        assert_grad_matches(Primitive::Log, 2.0, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_sqrt() {
        assert_grad_matches(Primitive::Sqrt, 4.0, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_rsqrt() {
        assert_grad_matches(Primitive::Rsqrt, 4.0, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_sin() {
        assert_grad_matches(Primitive::Sin, 1.0, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_cos() {
        assert_grad_matches(Primitive::Cos, 1.0, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_tan() {
        assert_grad_matches(Primitive::Tan, 0.5, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_asin() {
        assert_grad_matches(Primitive::Asin, 0.5, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_acos() {
        assert_grad_matches(Primitive::Acos, 0.5, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_atan() {
        assert_grad_matches(Primitive::Atan, 1.0, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_sinh() {
        assert_grad_matches(Primitive::Sinh, 1.0, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_cosh() {
        assert_grad_matches(Primitive::Cosh, 1.0, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_tanh() {
        assert_grad_matches(Primitive::Tanh, 1.0, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_expm1() {
        assert_grad_matches(Primitive::Expm1, 0.5, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_log1p() {
        assert_grad_matches(Primitive::Log1p, 1.0, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_square() {
        assert_grad_matches(Primitive::Square, 3.0, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_reciprocal() {
        assert_grad_matches(Primitive::Reciprocal, 2.0, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_logistic() {
        assert_grad_matches(Primitive::Logistic, 1.0, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_erf() {
        assert_grad_matches(Primitive::Erf, 0.5, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_erfc() {
        assert_grad_matches(Primitive::Erfc, 0.5, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_neg() {
        assert_grad_matches(Primitive::Neg, 3.0, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_abs() {
        // Test away from zero where abs is differentiable
        assert_grad_matches(Primitive::Abs, 2.0, 1e-4);
        assert_grad_matches(Primitive::Abs, -2.0, 1e-4);
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
