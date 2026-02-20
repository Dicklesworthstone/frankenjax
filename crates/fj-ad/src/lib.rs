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
            let zero_lit = match t.dtype {
                DType::I64 | DType::I32 => Literal::I64(0),
                DType::Bool => Literal::from_f64(0.0),
                DType::F64 | DType::F32 => Literal::from_f64(0.0),
            };
            let elements = vec![zero_lit; t.elements.len()];
            Value::Tensor(
                TensorValue::new(t.dtype, t.shape.clone(), elements)
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
    eval_primitive(Primitive::Neg, &[a.clone()], &BTreeMap::new())
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
            let sign = eval_primitive(Primitive::Sign, &[x.clone()], &BTreeMap::new())
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
            let a_pow_bm1 = eval_primitive(
                Primitive::Pow,
                &[a.clone(), b_minus_1],
                &BTreeMap::new(),
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let da = value_mul(g, &value_mul(b, &a_pow_bm1)?)?;

            let a_pow_b = eval_primitive(
                Primitive::Pow,
                &[a.clone(), b.clone()],
                &BTreeMap::new(),
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let ln_a = eval_primitive(Primitive::Log, &[a.clone()], &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let db = value_mul(g, &value_mul(&a_pow_b, &ln_a)?)?;
            Ok(vec![da, db])
        }
        Primitive::Exp => {
            let x = &inputs[0];
            let exp_x = eval_primitive(Primitive::Exp, &[x.clone()], &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![value_mul(g, &exp_x)?])
        }
        Primitive::Log => {
            let x = &inputs[0];
            Ok(vec![value_div(g, x)?])
        }
        Primitive::Sqrt => {
            let x = &inputs[0];
            let sqrt_x = eval_primitive(Primitive::Sqrt, &[x.clone()], &BTreeMap::new())
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
            Ok(vec![value_mul(g, &value_mul(&scalar_value(-0.5), &x_pow_neg1p5)?)?])
        }
        Primitive::Floor | Primitive::Ceil | Primitive::Round => {
            Ok(vec![zeros_like(&inputs[0])])
        }
        Primitive::Sin => {
            let x = &inputs[0];
            let cos_x = eval_primitive(Primitive::Cos, &[x.clone()], &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![value_mul(g, &cos_x)?])
        }
        Primitive::Cos => {
            let x = &inputs[0];
            let sin_x = eval_primitive(Primitive::Sin, &[x.clone()], &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![value_neg(&value_mul(g, &sin_x)?)?])
        }
        Primitive::Tan => {
            // d/dx tan(x) = 1/cos²(x) = 1 + tan²(x)
            let x = &inputs[0];
            let cos_x = eval_primitive(Primitive::Cos, &[x.clone()], &BTreeMap::new())
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
            let cosh_x = eval_primitive(Primitive::Cosh, &[x.clone()], &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![value_mul(g, &cosh_x)?])
        }
        Primitive::Cosh => {
            // d/dx cosh(x) = sinh(x)
            let x = &inputs[0];
            let sinh_x = eval_primitive(Primitive::Sinh, &[x.clone()], &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![value_mul(g, &sinh_x)?])
        }
        Primitive::Tanh => {
            // d/dx tanh(x) = 1 - tanh²(x)
            let x = &inputs[0];
            let tanh_x = eval_primitive(Primitive::Tanh, &[x.clone()], &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let tanh2 = value_mul(&tanh_x, &tanh_x)?;
            let one_minus = value_sub(&ones_like(x), &tanh2)?;
            Ok(vec![value_mul(g, &one_minus)?])
        }
        Primitive::Expm1 => {
            // d/dx expm1(x) = exp(x)
            let x = &inputs[0];
            let exp_x = eval_primitive(Primitive::Exp, &[x.clone()], &BTreeMap::new())
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
            let sig = eval_primitive(Primitive::Logistic, &[x.clone()], &BTreeMap::new())
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
                    // If g is scalar (full reduction), broadcast to input shape
                    let g_scalar = match g.as_f64_scalar() {
                        Some(v) => v,
                        None => {
                            // g is already a tensor from partial reduction
                            // Just broadcast g to input shape
                            return Ok(vec![broadcast_g_to_shape(g, &t.shape)?]);
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
            // Subgradient: pass gradient to argmax/argmin element(s)
            let input = &inputs[0];
            match input {
                Value::Scalar(_) => Ok(vec![g.clone()]),
                Value::Tensor(_) => {
                    // For simplicity, pass g through (same as ReduceSum for scalar case)
                    Ok(vec![g.clone()])
                }
            }
        }
        Primitive::ReduceProd => {
            let input = &inputs[0];
            match input {
                Value::Scalar(_) => Ok(vec![g.clone()]),
                Value::Tensor(_) => Ok(vec![g.clone()]),
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
                eval_primitive(Primitive::Reshape, &[g.clone()], &reshape_params)
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
                    .map(|s| s.trim().parse::<usize>().unwrap_or(0))
                    .collect::<Vec<_>>()
            } else {
                (0..rank).rev().collect()
            };
            // Compute inverse permutation
            let mut inv_perm = vec![0_usize; perm.len()];
            for (i, &p) in perm.iter().enumerate() {
                if p < inv_perm.len() {
                    inv_perm[p] = i;
                }
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
            let transposed_g = eval_primitive(Primitive::Transpose, &[g.clone()], &tp)
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![transposed_g])
        }
        Primitive::BroadcastInDim => {
            // VJP of broadcast_in_dim: reduce_sum g along broadcast axes
            let input = &inputs[0];
            let input_shape = match input {
                Value::Scalar(_) => Shape::scalar(),
                Value::Tensor(t) => t.shape.clone(),
            };
            // If input was scalar, sum everything
            if input_shape.rank() == 0 {
                let reduced = eval_primitive(Primitive::ReduceSum, &[g.clone()], &BTreeMap::new())
                    .map_err(|e| AdError::EvalFailed(e.to_string()))?;
                return Ok(vec![reduced]);
            }
            // Otherwise, just pass g through (simplified)
            Ok(vec![g.clone()])
        }
        Primitive::Slice => {
            // VJP of slice: scatter-add g into zeros at slice start
            // Simplified: pass g through
            Ok(vec![g.clone()])
        }
        Primitive::Concatenate => {
            // VJP of concatenate: split g via slices
            // Simplified: return g for each input
            Ok(inputs.iter().map(|_| g.clone()).collect())
        }
        Primitive::Gather | Primitive::Scatter => {
            Err(AdError::UnsupportedPrimitive(primitive))
        }
    }
}

fn broadcast_g_to_shape(g: &Value, target_shape: &Shape) -> Result<Value, AdError> {
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
    eval_primitive(Primitive::BroadcastInDim, &[g.clone()], &params)
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
            Ok(b * a.powf(b - 1.0) * tangents[0] + a.powf(b) * a.ln() * tangents[1])
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
        Primitive::Gather | Primitive::Scatter => {
            Err(AdError::UnsupportedPrimitive(primitive))
        }
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

    let output_val = outputs
        .first()
        .ok_or(AdError::NonScalarGradientOutput)?;

    // For scalar output, seed with 1.0; for tensor output, seed with ones
    let seed = ones_like(output_val);
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
        assert!(
            (g - 6.0).abs() < 1e-10,
            "d/dx(x²) at x=3 = 6, got {}",
            g
        );
    }

    #[test]
    fn grad_x_squared_plus_2x_at_3() {
        let jaxpr = build_program(ProgramSpec::SquarePlusLinear);
        let grads = grad_jaxpr(&jaxpr, &[Value::scalar_f64(3.0)]).expect("grad should succeed");
        let g = to_f64(&grads[0]).unwrap();
        assert!(
            (g - 8.0).abs() < 1e-10,
            "d/dx(x²+2x) at x=3 = 8, got {}",
            g
        );
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

    // ── Forward-mode JVP tests ──────────────────────────────────

    #[test]
    fn jvp_x_squared_at_3() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = jvp(&jaxpr, &[Value::scalar_f64(3.0)], &[1.0]).expect("jvp should succeed");
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
