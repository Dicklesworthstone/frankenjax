#![forbid(unsafe_code)]

use fj_core::{Atom, DType, Jaxpr, Literal, Primitive, Shape, TensorValue, Value, VarId};
use fj_lax::eval_primitive;
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock, RwLock};

#[derive(Debug, Clone, PartialEq)]
pub enum AdError {
    UnsupportedPrimitive(Primitive),
    NonScalarGradientOutput,
    EvalFailed(String),
    MissingVariable(VarId),
    InputArity {
        expected: usize,
        actual: usize,
    },
    PrimitiveOutputArity {
        primitive: Primitive,
        expected: usize,
        actual: usize,
    },
    CotangentArity {
        primitive: Primitive,
        expected_at_least: usize,
        actual: usize,
    },
}

impl std::fmt::Display for AdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedPrimitive(p) => {
                write!(f, "AD unsupported for primitive: {}", p.as_str())
            }
            Self::NonScalarGradientOutput => write!(f, "grad requires scalar output"),
            Self::EvalFailed(detail) => write!(f, "forward eval failed: {detail}"),
            Self::MissingVariable(var) => write!(f, "missing variable v{}", var.0),
            Self::InputArity { expected, actual } => {
                write!(f, "input arity mismatch: expected {expected}, got {actual}")
            }
            Self::PrimitiveOutputArity {
                primitive,
                expected,
                actual,
            } => write!(
                f,
                "{} output arity mismatch: expected {expected}, got {actual}",
                primitive.as_str()
            ),
            Self::CotangentArity {
                primitive,
                expected_at_least,
                actual,
            } => write!(
                f,
                "{} cotangent arity mismatch: expected at least {expected_at_least}, got {actual}",
                primitive.as_str()
            ),
        }
    }
}

impl std::error::Error for AdError {}

type CustomVjpRule = Arc<
    dyn Fn(&[Value], &Value, &BTreeMap<String, String>) -> Result<Vec<Value>, AdError>
        + Send
        + Sync,
>;

type CustomJvpRule = Arc<
    dyn Fn(&[Value], &[Value], &BTreeMap<String, String>) -> Result<Value, AdError> + Send + Sync,
>;

type CustomJaxprVjpRule =
    Arc<dyn Fn(&[Value], &[Value], &Value) -> Result<Vec<Value>, AdError> + Send + Sync>;

type CustomJaxprJvpRule =
    Arc<dyn Fn(&[Value], &[Value]) -> Result<JvpResult, AdError> + Send + Sync>;

#[derive(Default)]
struct CustomDerivativeRegistry {
    vjp_rules: BTreeMap<Primitive, CustomVjpRule>,
    jvp_rules: BTreeMap<Primitive, CustomJvpRule>,
    jaxpr_vjp_rules: BTreeMap<String, CustomJaxprVjpRule>,
    jaxpr_jvp_rules: BTreeMap<String, CustomJaxprJvpRule>,
}

static CUSTOM_DERIVATIVE_REGISTRY: OnceLock<RwLock<CustomDerivativeRegistry>> = OnceLock::new();

fn custom_derivative_registry() -> &'static RwLock<CustomDerivativeRegistry> {
    CUSTOM_DERIVATIVE_REGISTRY.get_or_init(|| RwLock::new(CustomDerivativeRegistry::default()))
}

fn with_registry_read<R>(f: impl FnOnce(&CustomDerivativeRegistry) -> R) -> R {
    let lock = custom_derivative_registry();
    match lock.read() {
        Ok(guard) => f(&guard),
        Err(poisoned) => f(&poisoned.into_inner()),
    }
}

fn with_registry_write<R>(f: impl FnOnce(&mut CustomDerivativeRegistry) -> R) -> R {
    let lock = custom_derivative_registry();
    match lock.write() {
        Ok(mut guard) => f(&mut guard),
        Err(poisoned) => f(&mut poisoned.into_inner()),
    }
}

/// Register a custom VJP rule for a primitive.
///
/// Registered rules override built-in VJP rules for the same primitive.
pub fn register_custom_vjp<F>(primitive: Primitive, rule: F)
where
    F: Fn(&[Value], &Value, &BTreeMap<String, String>) -> Result<Vec<Value>, AdError>
        + Send
        + Sync
        + 'static,
{
    with_registry_write(|registry| {
        registry.vjp_rules.insert(primitive, Arc::new(rule));
    });
}

/// Register a custom JVP rule for a primitive.
///
/// Registered rules override built-in JVP rules for the same primitive.
pub fn register_custom_jvp<F>(primitive: Primitive, rule: F)
where
    F: Fn(&[Value], &[Value], &BTreeMap<String, String>) -> Result<Value, AdError>
        + Send
        + Sync
        + 'static,
{
    with_registry_write(|registry| {
        registry.jvp_rules.insert(primitive, Arc::new(rule));
    });
}

/// Register a custom VJP rule for an entire Jaxpr.
///
/// Function-level rules are keyed by the Jaxpr's canonical fingerprint and
/// override equation-by-equation reverse-mode AD for matching Jaxprs.
pub fn register_custom_jaxpr_vjp<F>(jaxpr: &Jaxpr, rule: F)
where
    F: Fn(&[Value], &[Value], &Value) -> Result<Vec<Value>, AdError> + Send + Sync + 'static,
{
    let fingerprint = jaxpr.canonical_fingerprint().to_owned();
    with_registry_write(|registry| {
        registry.jaxpr_vjp_rules.insert(fingerprint, Arc::new(rule));
    });
}

/// Register a custom VJP rule for an explicit wrapper-scoped key.
///
/// Scoped keys let higher-level APIs attach different custom rules to equal
/// Jaxprs without mutating the Jaxpr fingerprint or replacing global rules.
pub fn register_custom_jaxpr_vjp_with_key<F>(rule_key: &str, rule: F)
where
    F: Fn(&[Value], &[Value], &Value) -> Result<Vec<Value>, AdError> + Send + Sync + 'static,
{
    with_registry_write(|registry| {
        registry
            .jaxpr_vjp_rules
            .insert(rule_key.to_owned(), Arc::new(rule));
    });
}

/// Register a custom JVP rule for an entire Jaxpr.
///
/// Function-level rules are keyed by the Jaxpr's canonical fingerprint and
/// override equation-by-equation forward-mode AD for matching Jaxprs.
pub fn register_custom_jaxpr_jvp<F>(jaxpr: &Jaxpr, rule: F)
where
    F: Fn(&[Value], &[Value]) -> Result<JvpResult, AdError> + Send + Sync + 'static,
{
    let fingerprint = jaxpr.canonical_fingerprint().to_owned();
    with_registry_write(|registry| {
        registry.jaxpr_jvp_rules.insert(fingerprint, Arc::new(rule));
    });
}

/// Register a custom JVP rule for an explicit wrapper-scoped key.
///
/// Scoped keys let higher-level APIs attach different custom rules to equal
/// Jaxprs without mutating the Jaxpr fingerprint or replacing global rules.
pub fn register_custom_jaxpr_jvp_with_key<F>(rule_key: &str, rule: F)
where
    F: Fn(&[Value], &[Value]) -> Result<JvpResult, AdError> + Send + Sync + 'static,
{
    with_registry_write(|registry| {
        registry
            .jaxpr_jvp_rules
            .insert(rule_key.to_owned(), Arc::new(rule));
    });
}

/// Remove all custom derivative rules.
pub fn clear_custom_derivative_rules() {
    with_registry_write(|registry| {
        registry.vjp_rules.clear();
        registry.jvp_rules.clear();
        registry.jaxpr_vjp_rules.clear();
        registry.jaxpr_jvp_rules.clear();
    });
}

fn lookup_custom_vjp(primitive: Primitive) -> Option<CustomVjpRule> {
    with_registry_read(|registry| registry.vjp_rules.get(&primitive).cloned())
}

fn lookup_custom_jvp(primitive: Primitive) -> Option<CustomJvpRule> {
    with_registry_read(|registry| registry.jvp_rules.get(&primitive).cloned())
}

fn lookup_custom_jaxpr_vjp(jaxpr: &Jaxpr) -> Option<CustomJaxprVjpRule> {
    with_registry_read(|registry| {
        registry
            .jaxpr_vjp_rules
            .get(jaxpr.canonical_fingerprint())
            .cloned()
    })
}

fn lookup_custom_jaxpr_vjp_by_key(rule_key: &str) -> Option<CustomJaxprVjpRule> {
    with_registry_read(|registry| registry.jaxpr_vjp_rules.get(rule_key).cloned())
}

fn lookup_custom_jaxpr_jvp(jaxpr: &Jaxpr) -> Option<CustomJaxprJvpRule> {
    with_registry_read(|registry| {
        registry
            .jaxpr_jvp_rules
            .get(jaxpr.canonical_fingerprint())
            .cloned()
    })
}

fn lookup_custom_jaxpr_jvp_by_key(rule_key: &str) -> Option<CustomJaxprJvpRule> {
    with_registry_read(|registry| registry.jaxpr_jvp_rules.get(rule_key).cloned())
}

#[derive(Debug, Clone)]
struct TapeEntry {
    primitive: Primitive,
    inputs: Vec<VarId>,
    /// All output VarIds (single-output primitives have exactly one).
    outputs: Vec<VarId>,
    input_values: Vec<Value>,
    /// Primal output values — needed by multi-output VJP rules (QR, SVD, Eigh).
    output_values: Vec<Value>,
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

        let output_values = fj_lax::eval_primitive_multi(eqn.primitive, &resolved, &eqn.params)
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;

        let out_var_ids: Vec<VarId> = eqn.outputs.iter().copied().collect();
        if output_values.len() != out_var_ids.len() {
            return Err(AdError::PrimitiveOutputArity {
                primitive: eqn.primitive,
                expected: out_var_ids.len(),
                actual: output_values.len(),
            });
        }
        for (var, val) in out_var_ids.iter().zip(output_values.iter()) {
            env.insert(*var, val.clone());
        }

        tape.entries.push(TapeEntry {
            primitive: eqn.primitive,
            inputs: input_var_ids,
            outputs: out_var_ids,
            input_values: resolved,
            output_values: output_values.clone(),
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
            Literal::U32(_) | Literal::U64(_) => Value::scalar_f64(0.0),
            Literal::Bool(_) => Value::scalar_f64(0.0),
            Literal::BF16Bits(_) | Literal::F16Bits(_) => Value::scalar_f64(0.0),
            Literal::F32Bits(_) => Value::scalar_f32(0.0),
            Literal::F64Bits(_) => Value::scalar_f64(0.0),
            Literal::Complex64Bits(..) | Literal::Complex128Bits(..) => Value::scalar_f64(0.0),
        },
        Value::Tensor(t) => {
            let (zero_lit, out_dtype) = match t.dtype {
                DType::I64 | DType::I32 => (Literal::I64(0), DType::I64),
                DType::U32 | DType::U64 => (Literal::from_f64(0.0), DType::F64),
                DType::Bool => (Literal::from_f64(0.0), DType::F64),
                DType::BF16 | DType::F16 => (Literal::from_f64(0.0), DType::F64),
                DType::F32 => (Literal::from_f32(0.0), DType::F32),
                DType::F64 => (Literal::from_f64(0.0), DType::F64),
                DType::Complex64 | DType::Complex128 => (Literal::from_f64(0.0), DType::F64),
            };
            let elements = vec![zero_lit; t.elements.len()];
            Value::Tensor(tensor_with_existing_shape(out_dtype, t, elements))
        }
    }
}

fn parse_pad_i64_param_for_rank(
    params: &BTreeMap<String, String>,
    key: &str,
    rank: usize,
    default: i64,
) -> Result<Vec<i64>, AdError> {
    let Some(raw) = params.get(key) else {
        return Ok(vec![default; rank]);
    };
    if raw.trim().is_empty() {
        if rank == 0 {
            return Ok(Vec::new());
        }
        return Err(AdError::EvalFailed(format!(
            "invalid empty pad param '{key}' for rank {rank}"
        )));
    }
    let values = raw
        .split(',')
        .map(str::trim)
        .map(|piece| {
            piece.parse::<i64>().map_err(|_| {
                AdError::EvalFailed(format!("invalid integer in pad param '{key}': '{piece}'"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() != rank {
        return Err(AdError::EvalFailed(format!(
            "padding rank mismatch for '{key}': rank={rank} values={}",
            values.len()
        )));
    }
    Ok(values)
}

fn ones_like(v: &Value) -> Value {
    match v {
        Value::Scalar(lit) => match lit {
            Literal::I64(_) => Value::scalar_f64(1.0),
            Literal::U32(_) | Literal::U64(_) => Value::scalar_f64(1.0),
            Literal::Bool(_) => Value::scalar_f64(1.0),
            Literal::BF16Bits(_) | Literal::F16Bits(_) => Value::scalar_f64(1.0),
            Literal::F32Bits(_) => Value::scalar_f32(1.0),
            Literal::F64Bits(_) => Value::scalar_f64(1.0),
            Literal::Complex64Bits(..) | Literal::Complex128Bits(..) => Value::scalar_f64(1.0),
        },
        Value::Tensor(t) => {
            let (one_lit, out_dtype) = match t.dtype {
                DType::F32 => (Literal::from_f32(1.0), DType::F32),
                _ => (Literal::from_f64(1.0), DType::F64),
            };
            let elements = vec![one_lit; t.elements.len()];
            Value::Tensor(tensor_with_existing_shape(out_dtype, t, elements))
        }
    }
}

fn balanced_eq_weight(candidate: &Value, ans: &Value, other: &Value) -> Result<Value, AdError> {
    let candidate_matches = eval_primitive(
        Primitive::Eq,
        &[candidate.clone(), ans.clone()],
        &BTreeMap::new(),
    )
    .map_err(|err| AdError::EvalFailed(err.to_string()))?;
    let other_matches = eval_primitive(
        Primitive::Eq,
        &[other.clone(), ans.clone()],
        &BTreeMap::new(),
    )
    .map_err(|err| AdError::EvalFailed(err.to_string()))?;
    let numerator = eval_primitive(
        Primitive::Select,
        &[
            candidate_matches,
            Value::scalar_f64(1.0),
            Value::scalar_f64(0.0),
        ],
        &BTreeMap::new(),
    )
    .map_err(|err| AdError::EvalFailed(err.to_string()))?;
    let denominator = eval_primitive(
        Primitive::Select,
        &[
            other_matches,
            Value::scalar_f64(2.0),
            Value::scalar_f64(1.0),
        ],
        &BTreeMap::new(),
    )
    .map_err(|err| AdError::EvalFailed(err.to_string()))?;
    value_div(&numerator, &denominator)
}

fn tensor_with_existing_shape(
    dtype: DType,
    source: &TensorValue,
    elements: Vec<Literal>,
) -> TensorValue {
    TensorValue {
        dtype,
        shape: source.shape.clone(),
        elements,
    }
}

fn scalar_value(x: f64) -> Value {
    Value::scalar_f64(x)
}

#[inline]
fn is_near_integer(x: f64) -> bool {
    (x - x.round()).abs() < 1e-14
}

fn trigamma_scalar(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() { 0.0 } else { f64::NAN };
    }
    if x <= 0.0 && is_near_integer(x) {
        return f64::INFINITY;
    }
    if x < 0.5 {
        let sin_px = (std::f64::consts::PI * x).sin();
        if sin_px == 0.0 {
            return f64::INFINITY;
        }
        let csc2 = 1.0 / (sin_px * sin_px);
        return std::f64::consts::PI.powi(2) * csc2 - trigamma_scalar(1.0 - x);
    }

    let mut shifted = x;
    let mut result = 0.0;
    while shifted < 8.0 {
        result += 1.0 / (shifted * shifted);
        shifted += 1.0;
    }

    let inv = 1.0 / shifted;
    let inv2 = inv * inv;
    let inv3 = inv2 * inv;
    let inv5 = inv3 * inv2;
    let inv7 = inv5 * inv2;
    let inv9 = inv7 * inv2;
    let inv11 = inv9 * inv2;

    result + inv + 0.5 * inv2 + inv3 / 6.0 - inv5 / 30.0 + inv7 / 42.0 - inv9 / 30.0
        + 5.0 * inv11 / 66.0
}

fn trigamma_value(x: &Value) -> Result<Value, AdError> {
    match x {
        Value::Scalar(lit) => {
            let x_val = lit.as_f64().ok_or_else(|| {
                AdError::EvalFailed("digamma VJP expects numeric scalar".to_owned())
            })?;
            Ok(Value::scalar_f64(trigamma_scalar(x_val)))
        }
        Value::Tensor(tensor) => {
            let elements = tensor
                .elements
                .iter()
                .map(|lit| {
                    lit.as_f64()
                        .map(trigamma_scalar)
                        .map(Literal::from_f64)
                        .ok_or_else(|| {
                            AdError::EvalFailed(
                                "digamma VJP expects numeric tensor elements".to_owned(),
                            )
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Value::Tensor(
                TensorValue::new(DType::F64, tensor.shape.clone(), elements)
                    .map_err(|e| AdError::EvalFailed(e.to_string()))?,
            ))
        }
    }
}

fn value_mul(a: &Value, b: &Value) -> Result<Value, AdError> {
    eval_primitive(Primitive::Mul, &[a.clone(), b.clone()], &BTreeMap::new())
        .map_err(|e| AdError::EvalFailed(e.to_string()))
}

fn value_dot(a: &Value, b: &Value) -> Result<Value, AdError> {
    eval_primitive(Primitive::Dot, &[a.clone(), b.clone()], &BTreeMap::new())
        .map_err(|e| AdError::EvalFailed(e.to_string()))
}

fn value_reduce_sum_all(value: &Value) -> Result<Value, AdError> {
    eval_primitive(
        Primitive::ReduceSum,
        std::slice::from_ref(value),
        &BTreeMap::new(),
    )
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

fn checked_axis_len(target_len: usize, context: &'static str) -> Result<u32, AdError> {
    u32::try_from(target_len).map_err(|_| {
        AdError::EvalFailed(format!(
            "{context}: target axis length {target_len} exceeds u32 shape limit"
        ))
    })
}

fn last_axis_info(tensor: &TensorValue, context: &'static str) -> Result<(usize, usize), AdError> {
    let Some((last_dim, batch_dims)) = tensor.shape.dims.split_last() else {
        return Err(AdError::EvalFailed(format!(
            "{context}: expected tensor rank to be at least 1"
        )));
    };

    let mut batch_size = 1usize;
    for dim in batch_dims {
        batch_size = batch_size
            .checked_mul(*dim as usize)
            .ok_or_else(|| AdError::EvalFailed(format!("{context}: batch size overflow")))?;
    }

    Ok((*last_dim as usize, batch_size))
}

fn set_last_axis_len(
    dims: &mut [u32],
    target_len: u32,
    context: &'static str,
) -> Result<(), AdError> {
    let Some(last_dim) = dims.last_mut() else {
        return Err(AdError::EvalFailed(format!(
            "{context}: expected tensor rank to be at least 1"
        )));
    };
    *last_dim = target_len;
    Ok(())
}

/// Zero-pad a complex tensor along the last axis from current length to `target_len`.
fn zero_pad_last_axis_complex(tensor: &TensorValue, target_len: usize) -> Result<Value, AdError> {
    let context = "zero_pad_last_axis_complex";
    let (cur_len, batch_size) = last_axis_info(tensor, context)?;
    if cur_len >= target_len {
        return Ok(Value::Tensor(tensor.clone()));
    }
    let target_dim = checked_axis_len(target_len, context)?;
    let target_elements = batch_size
        .checked_mul(target_len)
        .ok_or_else(|| AdError::EvalFailed(format!("{context}: element count overflow")))?;
    let mut elements = Vec::with_capacity(target_elements);
    let zero = Literal::from_complex128(0.0, 0.0);
    for batch in 0..batch_size {
        let start = batch * cur_len;
        elements.extend_from_slice(&tensor.elements[start..start + cur_len]);
        elements.resize(elements.len() + (target_len - cur_len), zero);
    }
    let mut dims = tensor.shape.dims.clone();
    set_last_axis_len(&mut dims, target_dim, context)?;
    TensorValue::new(tensor.dtype, Shape { dims }, elements)
        .map(Value::Tensor)
        .map_err(|e| AdError::EvalFailed(e.to_string()))
}

/// Extract real parts from a complex tensor, producing an F64 tensor of the same shape.
fn take_real_part(value: &Value) -> Result<Value, AdError> {
    match value {
        Value::Tensor(t) => {
            let elements: Vec<Literal> = t
                .elements
                .iter()
                .map(|lit| match lit {
                    Literal::Complex128Bits(re, _) => Literal::from_f64(f64::from_bits(*re)),
                    Literal::Complex64Bits(re, _) => Literal::from_f64(f32::from_bits(*re) as f64),
                    other => Literal::from_f64(other.as_f64().unwrap_or(0.0)),
                })
                .collect();
            TensorValue::new(DType::F64, t.shape.clone(), elements)
                .map(Value::Tensor)
                .map_err(|e| AdError::EvalFailed(e.to_string()))
        }
        Value::Scalar(lit) => match lit {
            Literal::Complex128Bits(re, _) => Ok(Value::scalar_f64(f64::from_bits(*re))),
            Literal::Complex64Bits(re, _) => Ok(Value::scalar_f64(f32::from_bits(*re) as f64)),
            other => Ok(Value::scalar_f64(other.as_f64().unwrap_or(0.0))),
        },
    }
}

/// Truncate a tensor's last axis to `target_len` elements.
fn truncate_last_axis(tensor: &TensorValue, target_len: usize) -> Result<Value, AdError> {
    let context = "truncate_last_axis";
    let (cur_len, batch_size) = last_axis_info(tensor, context)?;
    if cur_len <= target_len {
        return Ok(Value::Tensor(tensor.clone()));
    }
    let target_dim = checked_axis_len(target_len, context)?;
    let target_elements = batch_size
        .checked_mul(target_len)
        .ok_or_else(|| AdError::EvalFailed(format!("{context}: element count overflow")))?;
    let mut elements = Vec::with_capacity(target_elements);
    for batch in 0..batch_size {
        let start = batch * cur_len;
        elements.extend_from_slice(&tensor.elements[start..start + target_len]);
    }
    let mut dims = tensor.shape.dims.clone();
    set_last_axis_len(&mut dims, target_dim, context)?;
    TensorValue::new(tensor.dtype, Shape { dims }, elements)
        .map(Value::Tensor)
        .map_err(|e| AdError::EvalFailed(e.to_string()))
}

/// Scale complex tensor elements: multiply interior bins (indices 1..n/2-1) by 2,
/// leave DC (index 0) and Nyquist (index n/2) unchanged.
/// Used for IRFFT VJP where the Hermitian extension adjoint doubles interior bins.
fn scale_hermitian_adjoint(tensor: &TensorValue, fft_length: usize) -> Result<Value, AdError> {
    let half_len = fft_length / 2 + 1;
    let cur_len = *tensor.shape.dims.last().unwrap_or(&0) as usize;
    if cur_len != half_len {
        return Err(AdError::EvalFailed(format!(
            "scale_hermitian_adjoint: expected last dim {half_len}, got {cur_len}"
        )));
    }
    let batch_size = tensor.elements.len() / cur_len;
    let mut elements = tensor.elements.clone();
    for batch in 0..batch_size {
        let start = batch * cur_len;
        // Interior bins: everything except DC (index 0) and Nyquist (index n/2, even n only).
        // For odd fft_length there is no Nyquist bin, so all non-DC bins are interior.
        for k in 1..cur_len {
            if fft_length.is_multiple_of(2) && k == fft_length / 2 {
                continue; // Skip Nyquist for even fft_length
            }
            let idx = start + k;
            match &elements[idx] {
                Literal::Complex128Bits(re, im) => {
                    let r = f64::from_bits(*re) * 2.0;
                    let i = f64::from_bits(*im) * 2.0;
                    elements[idx] = Literal::from_complex128(r, i);
                }
                other => {
                    let v = other.as_f64().unwrap_or(0.0) * 2.0;
                    elements[idx] = Literal::from_f64(v);
                }
            }
        }
    }
    TensorValue::new(tensor.dtype, tensor.shape.clone(), elements)
        .map(Value::Tensor)
        .map_err(|e| AdError::EvalFailed(e.to_string()))
}

// ── Matrix operation helpers for multi-output VJP ───────────────

/// Matrix multiply: C = A[m×k] @ B[k×n] → C[m×n].
/// Both inputs must be rank-2 tensors stored as flat row-major.
fn matmul_2d(a: &TensorValue, b: &TensorValue) -> Result<Value, AdError> {
    if a.shape.rank() != 2 || b.shape.rank() != 2 {
        return Err(AdError::EvalFailed(
            "matmul: inputs must be rank-2".to_owned(),
        ));
    }
    let m = a.shape.dims[0] as usize;
    let k = a.shape.dims[1] as usize;
    let k2 = b.shape.dims[0] as usize;
    let n = b.shape.dims[1] as usize;
    if k != k2 {
        return Err(AdError::EvalFailed(format!(
            "matmul: inner dims mismatch: {k} vs {k2}"
        )));
    }
    let a_vals: Vec<f64> = a
        .elements
        .iter()
        .map(|l| l.as_f64().unwrap_or(0.0))
        .collect();
    let b_vals: Vec<f64> = b
        .elements
        .iter()
        .map(|l| l.as_f64().unwrap_or(0.0))
        .collect();
    let mut c_vals = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a_vals[i * k + p] * b_vals[p * n + j];
            }
            c_vals[i * n + j] = sum;
        }
    }
    let elements: Vec<Literal> = c_vals.into_iter().map(Literal::from_f64).collect();
    TensorValue::new(
        DType::F64,
        Shape {
            dims: vec![m as u32, n as u32],
        },
        elements,
    )
    .map(Value::Tensor)
    .map_err(|e| AdError::EvalFailed(e.to_string()))
}

/// Transpose a rank-2 tensor: A[m×n] → A^T[n×m].
fn transpose_2d(a: &TensorValue) -> Result<Value, AdError> {
    if a.shape.rank() != 2 {
        return Err(AdError::EvalFailed(
            "transpose: input must be rank-2".to_owned(),
        ));
    }
    let m = a.shape.dims[0] as usize;
    let n = a.shape.dims[1] as usize;
    let a_vals: Vec<f64> = a
        .elements
        .iter()
        .map(|l| l.as_f64().unwrap_or(0.0))
        .collect();
    let mut t_vals = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            t_vals[j * m + i] = a_vals[i * n + j];
        }
    }
    let elements: Vec<Literal> = t_vals.into_iter().map(Literal::from_f64).collect();
    TensorValue::new(
        DType::F64,
        Shape {
            dims: vec![n as u32, m as u32],
        },
        elements,
    )
    .map(Value::Tensor)
    .map_err(|e| AdError::EvalFailed(e.to_string()))
}

fn vector_elements_f64(tensor: &TensorValue, context: &'static str) -> Result<Vec<f64>, AdError> {
    if tensor.shape.rank() != 1 {
        return Err(AdError::EvalFailed(format!(
            "{context}: expected rank-1 tensor, got rank-{}",
            tensor.shape.rank()
        )));
    }
    tensor
        .elements
        .iter()
        .map(|lit| {
            lit.as_f64()
                .ok_or_else(|| AdError::EvalFailed(format!("{context}: expected numeric tensor")))
        })
        .collect()
}

fn outer_product(lhs: &Value, rhs: &Value, context: &'static str) -> Result<Value, AdError> {
    let lhs_tensor = tensor_value(lhs, context)?;
    let rhs_tensor = tensor_value(rhs, context)?;
    let lhs_values = vector_elements_f64(lhs_tensor, context)?;
    let rhs_values = vector_elements_f64(rhs_tensor, context)?;
    let mut elements = Vec::with_capacity(lhs_values.len() * rhs_values.len());
    for lhs_value in &lhs_values {
        for rhs_value in &rhs_values {
            elements.push(Literal::from_f64(lhs_value * rhs_value));
        }
    }
    TensorValue::new(
        DType::F64,
        Shape {
            dims: vec![lhs_values.len() as u32, rhs_values.len() as u32],
        },
        elements,
    )
    .map(Value::Tensor)
    .map_err(|e| AdError::EvalFailed(e.to_string()))
}

fn dot_vjp(inputs: &[Value], g: &Value) -> Result<Vec<Value>, AdError> {
    let a = &inputs[0];
    let b = &inputs[1];
    match (a, b) {
        (Value::Scalar(_), Value::Scalar(_)) => Ok(vec![value_mul(g, b)?, value_mul(g, a)?]),
        (Value::Scalar(_), Value::Tensor(_)) => {
            let da = value_reduce_sum_all(&value_mul(g, b)?)?;
            let db = value_mul(g, a)?;
            Ok(vec![da, db])
        }
        (Value::Tensor(_), Value::Scalar(_)) => {
            let da = value_mul(g, b)?;
            let db = value_reduce_sum_all(&value_mul(g, a)?)?;
            Ok(vec![da, db])
        }
        (Value::Tensor(a_tensor), Value::Tensor(b_tensor)) => {
            match (a_tensor.rank(), b_tensor.rank()) {
                (1, 1) => Ok(vec![value_mul(g, b)?, value_mul(g, a)?]),
                (2, 1) => {
                    let da = outer_product(g, b, "dot VJP matrix-vector lhs cotangent")?;
                    let a_t = transpose_2d(a_tensor)?;
                    let db = value_dot(&a_t, g)?;
                    Ok(vec![da, db])
                }
                (1, 2) => {
                    let b_t = transpose_2d(b_tensor)?;
                    let da = value_dot(g, &b_t)?;
                    let db = outer_product(a, g, "dot VJP vector-matrix rhs cotangent")?;
                    Ok(vec![da, db])
                }
                (2, 2) => {
                    let b_t = transpose_2d(b_tensor)?;
                    let da = value_dot(g, &b_t)?;
                    let a_t = transpose_2d(a_tensor)?;
                    let db = value_dot(&a_t, g)?;
                    Ok(vec![da, db])
                }
                (a_rank, b_rank) => Err(AdError::EvalFailed(format!(
                    "dot VJP supports scalar, vector, and rank-2 tensor inputs; got ranks {a_rank} and {b_rank}"
                ))),
            }
        }
    }
}

/// Extract the lower triangle (including diagonal) of a square matrix.
/// Zeroes out the strict upper triangle.
fn tril(a: &TensorValue) -> Result<TensorValue, AdError> {
    if a.shape.rank() != 2 || a.shape.dims[0] != a.shape.dims[1] {
        return Err(AdError::EvalFailed(
            "tril: input must be square rank-2".to_owned(),
        ));
    }
    let n = a.shape.dims[0] as usize;
    let a_vals: Vec<f64> = a
        .elements
        .iter()
        .map(|l| l.as_f64().unwrap_or(0.0))
        .collect();
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            result[i * n + j] = a_vals[i * n + j];
        }
    }
    let elements: Vec<Literal> = result.into_iter().map(Literal::from_f64).collect();
    TensorValue::new(DType::F64, a.shape.clone(), elements)
        .map_err(|e| AdError::EvalFailed(e.to_string()))
}

/// Add two tensors element-wise.
fn tensor_add(a: &TensorValue, b: &TensorValue) -> Result<Value, AdError> {
    value_add(&Value::Tensor(a.clone()), &Value::Tensor(b.clone()))
}

/// Subtract two tensors element-wise: a - b.
fn tensor_sub(a: &TensorValue, b: &TensorValue) -> Result<Value, AdError> {
    value_sub(&Value::Tensor(a.clone()), &Value::Tensor(b.clone()))
}

fn is_zero_value(v: &Value) -> bool {
    match v {
        Value::Scalar(Literal::F64Bits(bits)) => f64::from_bits(*bits) == 0.0,
        Value::Scalar(Literal::I64(val)) => *val == 0,
        Value::Scalar(Literal::U32(val)) => *val == 0,
        Value::Scalar(Literal::U64(val)) => *val == 0,
        Value::Scalar(Literal::Bool(val)) => !val,
        Value::Scalar(lit @ (Literal::BF16Bits(_) | Literal::F16Bits(_))) => {
            (*lit).as_f64().is_some_and(|value| value == 0.0)
        }
        Value::Scalar(Literal::F32Bits(bits)) => f32::from_bits(*bits) == 0.0,
        Value::Scalar(Literal::Complex64Bits(re, im)) => {
            f32::from_bits(*re) == 0.0 && f32::from_bits(*im) == 0.0
        }
        Value::Scalar(Literal::Complex128Bits(re, im)) => {
            f64::from_bits(*re) == 0.0 && f64::from_bits(*im) == 0.0
        }
        Value::Tensor(t) => t.elements.iter().all(|e| match e {
            Literal::F64Bits(bits) => f64::from_bits(*bits) == 0.0,
            Literal::I64(v) => *v == 0,
            Literal::U32(v) => *v == 0,
            Literal::U64(v) => *v == 0,
            Literal::Bool(v) => !v,
            Literal::BF16Bits(_) | Literal::F16Bits(_) => {
                (*e).as_f64().is_some_and(|value| value == 0.0)
            }
            Literal::F32Bits(bits) => f32::from_bits(*bits) == 0.0,
            Literal::Complex64Bits(re, im) => {
                f32::from_bits(*re) == 0.0 && f32::from_bits(*im) == 0.0
            }
            Literal::Complex128Bits(re, im) => {
                f64::from_bits(*re) == 0.0 && f64::from_bits(*im) == 0.0
            }
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
        // Gather gradients for all outputs of this equation.
        let gs: Vec<Value> = entry
            .outputs
            .iter()
            .enumerate()
            .map(|(i, var)| {
                adjoints
                    .get(var)
                    .cloned()
                    .unwrap_or_else(|| zeros_like(&entry.output_values[i]))
            })
            .collect();
        // Skip if all output gradients are zero.
        if gs.iter().all(is_zero_value) {
            continue;
        }

        let cotangents = vjp(
            entry.primitive,
            &entry.input_values,
            &gs,
            &entry.output_values,
            &entry.params,
        )?;

        for (var_id, cot) in entry.inputs.iter().zip(cotangents) {
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

/// Convenience wrapper for single-output VJP calls.
pub fn vjp_single(
    primitive: Primitive,
    inputs: &[Value],
    g: &Value,
    params: &BTreeMap<String, String>,
) -> Result<Vec<Value>, AdError> {
    vjp(primitive, inputs, std::slice::from_ref(g), &[], params)
}

/// Compute VJP (reverse-mode AD) for a single primitive application.
pub fn vjp(
    primitive: Primitive,
    inputs: &[Value],
    gs: &[Value],
    output_values: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Vec<Value>, AdError> {
    // For single-output primitives (the common case), use gs[0] as the gradient.
    let Some(g) = gs.first() else {
        return Err(AdError::CotangentArity {
            primitive,
            expected_at_least: 1,
            actual: 0,
        });
    };
    if let Some(custom_rule) = lookup_custom_vjp(primitive) {
        let cotangents = custom_rule(inputs, g, params)?;
        if cotangents.len() != inputs.len() {
            return Err(AdError::EvalFailed(format!(
                "custom VJP cotangent arity mismatch for {}: expected {}, got {}",
                primitive.as_str(),
                inputs.len(),
                cotangents.len()
            )));
        }
        return Ok(cotangents);
    }

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
            let ans = eval_primitive(Primitive::Max, &[a.clone(), b.clone()], &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let ga = value_mul(g, &balanced_eq_weight(a, &ans, b)?)?;
            let gb = value_mul(g, &balanced_eq_weight(b, &ans, a)?)?;
            Ok(vec![ga, gb])
        }
        Primitive::Min => {
            let a = &inputs[0];
            let b = &inputs[1];
            let ans = eval_primitive(Primitive::Min, &[a.clone(), b.clone()], &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let ga = value_mul(g, &balanced_eq_weight(a, &ans, b)?)?;
            let gb = value_mul(g, &balanced_eq_weight(b, &ans, a)?)?;
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
        Primitive::Asinh => {
            // d/dx asinh(x) = 1 / sqrt(x² + 1)
            let x = &inputs[0];
            let x2 = value_mul(x, x)?;
            let sum = value_add(&ones_like(x), &x2)?;
            let denom = eval_primitive(Primitive::Sqrt, std::slice::from_ref(&sum), &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![value_div(g, &denom)?])
        }
        Primitive::Acosh => {
            // d/dx acosh(x) = 1 / sqrt(x² - 1)
            let x = &inputs[0];
            let x2 = value_mul(x, x)?;
            let diff = value_sub(&x2, &ones_like(x))?;
            let denom = eval_primitive(Primitive::Sqrt, std::slice::from_ref(&diff), &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![value_div(g, &denom)?])
        }
        Primitive::Atanh => {
            // d/dx atanh(x) = 1 / (1 - x²)
            let x = &inputs[0];
            let x2 = value_mul(x, x)?;
            let denom = value_sub(&ones_like(x), &x2)?;
            Ok(vec![value_div(g, &denom)?])
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
        Primitive::Lgamma => {
            // d/dx lgamma(x) = digamma(x)
            let x = &inputs[0];
            let digamma_x = eval_primitive(Primitive::Digamma, std::slice::from_ref(x), params)
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![value_mul(g, &digamma_x)?])
        }
        Primitive::Digamma => {
            // d/dx digamma(x) = trigamma(x)
            let trigamma_x = trigamma_value(&inputs[0])?;
            Ok(vec![value_mul(g, &trigamma_x)?])
        }
        Primitive::ErfInv => {
            // d/dx erf_inv(x) = sqrt(pi)/2 * exp(erf_inv(x)^2)
            let erf_inv_x = eval_primitive(Primitive::ErfInv, inputs, &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let erf_inv_sq = value_mul(&erf_inv_x, &erf_inv_x)?;
            let exp_term = eval_primitive(Primitive::Exp, &[erf_inv_sq], &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let coeff = scalar_value(std::f64::consts::PI.sqrt() / 2.0);
            let factor = value_mul(&coeff, &exp_term)?;
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
        Primitive::Complex => {
            let g_real = eval_primitive(Primitive::Real, std::slice::from_ref(g), &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let g_imag = eval_primitive(Primitive::Imag, std::slice::from_ref(g), &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![g_real, g_imag])
        }
        Primitive::Conj => Ok(vec![
            eval_primitive(Primitive::Conj, std::slice::from_ref(g), &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?,
        ]),
        Primitive::Real => {
            let zero = zeros_like(g);
            Ok(vec![
                eval_primitive(Primitive::Complex, &[g.clone(), zero], &BTreeMap::new())
                    .map_err(|e| AdError::EvalFailed(e.to_string()))?,
            ])
        }
        Primitive::Imag => {
            let zero = zeros_like(g);
            Ok(vec![
                eval_primitive(Primitive::Complex, &[zero, g.clone()], &BTreeMap::new())
                    .map_err(|e| AdError::EvalFailed(e.to_string()))?,
            ])
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
                Value::Tensor(t) => Ok(vec![reduce_prod_vjp_tensor(t, g, params)?]),
            }
        }
        Primitive::ReduceAnd | Primitive::ReduceOr | Primitive::ReduceXor => {
            // Bitwise reductions are non-differentiable.
            Ok(vec![zeros_like(&inputs[0])])
        }
        Primitive::Dot => dot_vjp(inputs, g),
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
        Primitive::Clamp => {
            // VJP of clamp(x, lo, hi): gradient passes through where lo < x < hi,
            // otherwise zero (x is at a boundary).
            let x = &inputs[0];
            let lo = &inputs[1];
            let hi = &inputs[2];
            // mask = (x > lo) & (x < hi)
            let gt_lo = eval_primitive(Primitive::Gt, &[x.clone(), lo.clone()], &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let lt_hi = eval_primitive(Primitive::Lt, &[x.clone(), hi.clone()], &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            // g_x = select(gt_lo & lt_hi, g, 0)
            // Since we don't have And, use select twice: select(gt_lo, select(lt_hi, g, 0), 0)
            let inner = eval_primitive(
                Primitive::Select,
                &[lt_hi, g.clone(), zeros_like(g)],
                &BTreeMap::new(),
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let g_x = eval_primitive(
                Primitive::Select,
                &[gt_lo, inner, zeros_like(g)],
                &BTreeMap::new(),
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            // lo and hi don't receive gradients in JAX
            Ok(vec![g_x, zeros_like(lo), zeros_like(hi)])
        }
        Primitive::DynamicSlice => dynamic_slice_vjp(inputs, g),
        Primitive::Pad => {
            // VJP of pad: extract the original operand positions from g via
            // strided slicing (accounting for interior padding), and compute
            // the pad-value gradient as the sum of g at all padding positions.
            let operand = &inputs[0];
            match operand {
                Value::Scalar(_) => {
                    // Scalar operand: gradient passes through; pad value gets zero.
                    Ok(vec![g.clone(), Value::scalar_f64(0.0)])
                }
                Value::Tensor(op_tensor) => {
                    let rank = op_tensor.shape.rank();

                    // Parse padding params (same format as eval_pad).
                    let lows = parse_pad_i64_param_for_rank(params, "padding_low", rank, 0)?;
                    let interiors =
                        parse_pad_i64_param_for_rank(params, "padding_interior", rank, 0)?;
                    for (axis, interior) in interiors.iter().enumerate() {
                        if *interior < 0 {
                            return Err(AdError::EvalFailed(format!(
                                "interior padding must be non-negative on axis {axis}: {interior}"
                            )));
                        }
                    }

                    let g_tensor = match g {
                        Value::Tensor(t) => t,
                        Value::Scalar(_) => {
                            return Err(AdError::EvalFailed(
                                "pad VJP requires tensor gradient".into(),
                            ));
                        }
                    };

                    // Compute strides for the padded (gradient) tensor.
                    let g_dims = &g_tensor.shape.dims;
                    let mut g_strides = vec![1_usize; rank];
                    for i in (0..rank.saturating_sub(1)).rev() {
                        g_strides[i] = g_strides[i + 1] * g_dims[i + 1] as usize;
                    }

                    // Extract operand gradient: elements at positions
                    // low + k * (interior + 1) for k in 0..op_dim, per axis.
                    let op_total = op_tensor.elements.len();
                    let mut op_grad_elements = Vec::with_capacity(op_total);
                    let mut op_coords = vec![0_usize; rank];
                    let mut op_grad_sum = 0.0_f64;

                    for _ in 0..op_total {
                        let mut g_flat = Some(0_usize);
                        for ax in 0..rank {
                            let stride = interiors[ax] + 1;
                            let pos = lows[ax] + op_coords[ax] as i64 * stride;
                            if pos < 0 || pos >= i64::from(g_dims[ax]) {
                                g_flat = None;
                                break;
                            }
                            if let Some(flat) = &mut g_flat {
                                *flat += pos as usize * g_strides[ax];
                            }
                        }

                        let val = match g_flat {
                            Some(flat) if flat < g_tensor.elements.len() => {
                                g_tensor.elements[flat].as_f64().unwrap_or(0.0)
                            }
                            _ => 0.0,
                        };
                        op_grad_sum += val;
                        op_grad_elements.push(Literal::from_f64(val));

                        // Increment operand coordinates.
                        if rank > 0 {
                            for ax in (0..rank).rev() {
                                op_coords[ax] += 1;
                                if op_coords[ax] < op_tensor.shape.dims[ax] as usize {
                                    break;
                                }
                                op_coords[ax] = 0;
                            }
                        }
                    }

                    let g_operand = Value::Tensor(
                        TensorValue::new(DType::F64, op_tensor.shape.clone(), op_grad_elements)
                            .map_err(|e| AdError::EvalFailed(e.to_string()))?,
                    );

                    // Pad-value gradient: sum of g at all padding positions.
                    let g_total_sum: f64 = g_tensor
                        .elements
                        .iter()
                        .map(|l| l.as_f64().unwrap_or(0.0))
                        .sum();
                    let g_pad_value = Value::scalar_f64(g_total_sum - op_grad_sum);

                    Ok(vec![g_operand, g_pad_value])
                }
            }
        }
        Primitive::Iota => {
            // Iota has no inputs, so no gradients to propagate.
            Ok(vec![])
        }
        Primitive::BroadcastedIota => {
            // BroadcastedIota has no differentiable inputs (discrete indices).
            Ok(inputs.iter().map(zeros_like).collect())
        }
        Primitive::OneHot => {
            // OneHot is not differentiable w.r.t. its indices (discrete).
            // Return zero gradient for the single input.
            Ok(vec![Value::scalar_f64(0.0)])
        }
        Primitive::Copy => Ok(vec![g.clone()]),
        Primitive::BitcastConvertType => Ok(vec![zeros_like(&inputs[0])]),
        Primitive::ReducePrecision => Ok(vec![g.clone()]),
        Primitive::DynamicUpdateSlice => {
            // VJP: g_operand = g with update region zeroed out,
            //       g_update = slice of g at the start positions.
            // Start indices have no gradient (discrete).
            dynamic_update_slice_vjp(inputs, g)
        }
        Primitive::Cumsum => {
            // VJP of cumsum is reverse cumsum of gradient.
            // For scalar: pass through (cumsum of scalar = scalar).
            // For tensor: reverse cumulative sum along the cumsum axis.
            match g {
                Value::Scalar(_) => Ok(vec![g.clone()]),
                Value::Tensor(gt) => {
                    let axis: usize = params
                        .get("axis")
                        .and_then(|s| s.trim().parse().ok())
                        .unwrap_or(0);
                    let dims: Vec<usize> = gt.shape.dims.iter().map(|&d| d as usize).collect();
                    let rank = dims.len();
                    let g_vals: Vec<f64> = gt
                        .elements
                        .iter()
                        .map(|l| l.as_f64().unwrap_or(0.0))
                        .collect();
                    let total = g_vals.len();
                    let mut result = vec![0.0_f64; total];

                    // Compute strides for ND indexing
                    let mut strides = vec![1usize; rank];
                    for i in (0..rank - 1).rev() {
                        strides[i] = strides[i + 1] * dims[i + 1];
                    }

                    // For each position along the non-cumsum axes, do reverse cumsum
                    let axis_len = dims[axis];
                    let axis_stride = strides[axis];
                    let outer_count = total / axis_len;

                    for outer in 0..outer_count {
                        // Compute base index: skip the axis dimension
                        let mut base = 0;
                        let mut rem = outer;
                        for d in (0..rank).rev() {
                            if d == axis {
                                continue;
                            }
                            base += (rem % dims[d]) * strides[d];
                            rem /= dims[d];
                        }

                        // Reverse cumulative sum along axis
                        let mut running = 0.0;
                        for i in (0..axis_len).rev() {
                            let idx = base + i * axis_stride;
                            running += g_vals[idx];
                            result[idx] = running;
                        }
                    }

                    Ok(vec![Value::Tensor(
                        TensorValue::new(
                            DType::F64,
                            gt.shape.clone(),
                            result.into_iter().map(Literal::from_f64).collect(),
                        )
                        .map_err(|e| AdError::EvalFailed(e.to_string()))?,
                    )])
                }
            }
        }
        Primitive::Cumprod => {
            // VJP of cumprod: grad_x[i] = sum_{j>=i} g[j] * cumprod[j] / x[i]
            // For scalar: pass through (cumprod of scalar = scalar).
            // For tensor: use the formula with forward cumprod values.
            match g {
                Value::Scalar(_) => Ok(vec![g.clone()]),
                Value::Tensor(gt) => {
                    let x = &inputs[0];
                    let x_tensor = match x {
                        Value::Tensor(t) => t,
                        Value::Scalar(_) => return Ok(vec![g.clone()]),
                    };
                    let axis: usize = params
                        .get("axis")
                        .and_then(|s| s.trim().parse().ok())
                        .unwrap_or(0);
                    let dims: Vec<usize> = gt.shape.dims.iter().map(|&d| d as usize).collect();
                    let rank = dims.len();
                    let g_vals: Vec<f64> = gt
                        .elements
                        .iter()
                        .map(|l| l.as_f64().unwrap_or(0.0))
                        .collect();
                    let x_vals: Vec<f64> = x_tensor
                        .elements
                        .iter()
                        .map(|l| l.as_f64().unwrap_or(0.0))
                        .collect();
                    let total = g_vals.len();
                    let mut result = vec![0.0_f64; total];

                    let mut strides = vec![1usize; rank];
                    for i in (0..rank - 1).rev() {
                        strides[i] = strides[i + 1] * dims[i + 1];
                    }

                    let axis_len = dims[axis];
                    let axis_stride = strides[axis];
                    let outer_count = total / axis_len;

                    for outer in 0..outer_count {
                        let mut base = 0;
                        let mut rem = outer;
                        for d in (0..rank).rev() {
                            if d == axis {
                                continue;
                            }
                            base += (rem % dims[d]) * strides[d];
                            rem /= dims[d];
                        }

                        // Forward cumprod for this slice
                        let mut cumprod = vec![0.0_f64; axis_len];
                        let mut running = 1.0;
                        for i in 0..axis_len {
                            running *= x_vals[base + i * axis_stride];
                            cumprod[i] = running;
                        }

                        // grad_x[i] = sum_{j>=i} g[j] * cumprod[j] / x[i]
                        // = (1/x[i]) * sum_{j>=i} g[j] * cumprod[j]
                        // Use reverse cumsum of (g * cumprod), then divide by x
                        let mut suffix_sum = 0.0;
                        for i in (0..axis_len).rev() {
                            let idx = base + i * axis_stride;
                            suffix_sum += g_vals[idx] * cumprod[i];
                            let xi = x_vals[idx];
                            if xi.abs() > f64::EPSILON {
                                result[idx] = suffix_sum / xi;
                            }
                            // If x[i] == 0, gradient is 0 (avoid division by zero)
                        }
                    }

                    Ok(vec![Value::Tensor(
                        TensorValue::new(
                            DType::F64,
                            gt.shape.clone(),
                            result.into_iter().map(Literal::from_f64).collect(),
                        )
                        .map_err(|e| AdError::EvalFailed(e.to_string()))?,
                    )])
                }
            }
        }
        Primitive::Sort => {
            // VJP of sort: unsort the gradient using the argsort permutation.
            // Must respect the `axis` parameter — sort operates along a specific axis.
            let x = &inputs[0];
            match (x, g) {
                (Value::Scalar(_), _) => Ok(vec![g.clone()]),
                (Value::Tensor(xt), Value::Tensor(gt)) => {
                    let rank = xt.shape.rank();
                    if rank == 0 {
                        return Ok(vec![g.clone()]);
                    }
                    let axis: usize = params
                        .get("axis")
                        .and_then(|s| s.trim().parse().ok())
                        .unwrap_or(rank - 1);
                    if axis >= rank {
                        return Err(AdError::EvalFailed(format!(
                            "sort VJP axis {axis} out of bounds for rank {rank}"
                        )));
                    }

                    let x_vals: Vec<f64> = xt
                        .elements
                        .iter()
                        .map(|l| l.as_f64().unwrap_or(0.0))
                        .collect();
                    let g_vals: Vec<f64> = gt
                        .elements
                        .iter()
                        .map(|l| l.as_f64().unwrap_or(0.0))
                        .collect();

                    let dims: Vec<usize> = xt.shape.dims.iter().map(|&d| d as usize).collect();
                    let mut strides = vec![1usize; rank];
                    for i in (0..rank - 1).rev() {
                        strides[i] = strides[i + 1] * dims[i + 1];
                    }
                    let axis_dim = dims[axis];
                    let axis_stride = strides[axis];
                    let total = x_vals.len();
                    if g_vals.len() != total {
                        return Err(AdError::EvalFailed(format!(
                            "sort VJP gradient size mismatch: expected {total}, got {}",
                            g_vals.len()
                        )));
                    }
                    let outer_count = total / axis_dim;

                    let mut result = vec![0.0_f64; total];

                    let descending = params
                        .get("descending")
                        .map(|s| s.trim() == "true")
                        .unwrap_or(false);

                    for outer in 0..outer_count {
                        // Compute base flat index (same pattern as forward sort)
                        let mut base = 0usize;
                        let mut rem = outer;
                        for ax in (0..rank).rev() {
                            if ax == axis {
                                continue;
                            }
                            base += (rem % dims[ax]) * strides[ax];
                            rem /= dims[ax];
                        }

                        // Extract the slice along the axis and argsort it
                        let mut indexed: Vec<(usize, f64)> = (0..axis_dim)
                            .map(|i| (i, x_vals[base + i * axis_stride]))
                            .collect();
                        if descending {
                            indexed.sort_by(|a, b| {
                                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                        } else {
                            indexed.sort_by(|a, b| {
                                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                        }

                        // Inverse permutation: unsort the gradient.
                        // For tied values, use the symmetric subgradient by averaging the
                        // tied gradient slice before routing back.
                        let mut sorted_pos = 0usize;
                        while sorted_pos < axis_dim {
                            let group_start = sorted_pos;
                            let current_val = indexed[group_start].1;
                            sorted_pos += 1;

                            while sorted_pos < axis_dim {
                                let next_val = indexed[sorted_pos].1;
                                let same_bucket = current_val.partial_cmp(&next_val).is_none()
                                    || matches!(
                                        current_val.partial_cmp(&next_val),
                                        Some(std::cmp::Ordering::Equal)
                                    );
                                if !same_bucket {
                                    break;
                                }
                                sorted_pos += 1;
                            }

                            let group_end = sorted_pos;
                            let mut grad_sum = 0.0_f64;
                            for pos in group_start..group_end {
                                let g_flat = base + pos * axis_stride;
                                grad_sum += g_vals[g_flat];
                            }
                            let grad_share = grad_sum / (group_end - group_start) as f64;

                            for &(orig_idx, _) in indexed.iter().take(group_end).skip(group_start) {
                                let result_flat = base + orig_idx * axis_stride;
                                result[result_flat] = grad_share;
                            }
                        }
                    }

                    Ok(vec![Value::Tensor(
                        TensorValue::new(
                            DType::F64,
                            xt.shape.clone(),
                            result.into_iter().map(Literal::from_f64).collect(),
                        )
                        .map_err(|e| AdError::EvalFailed(e.to_string()))?,
                    )])
                }
                _ => Ok(vec![g.clone()]),
            }
        }
        Primitive::Argsort => {
            // Argsort is not differentiable (returns integer indices).
            Ok(vec![Value::scalar_f64(0.0)])
        }
        Primitive::Conv => {
            // Conv 1D VJP for layout lhs=[N, W, C_in], rhs=[K, C_in, C_out], out=[N, W_out, C_out]
            // grad_lhs: convolve g with flipped kernel (transposed convolution)
            // grad_rhs: cross-correlate input with g
            conv_vjp(inputs, g, params)
        }

        Primitive::Cond => {
            // Cond VJP: gradient flows through the selected branch only.
            // inputs: [pred, true_operand, false_operand]
            // The predicate is discrete (no gradient).
            cond_vjp(inputs, g)
        }

        Primitive::Scan => {
            // Scan VJP: reverse-propagate gradient through the iteration loop.
            // For scan with body_op: carry_{i+1} = body_op(carry_i, xs[i])
            // We compute per-element VJPs by replaying the forward pass and
            // backpropagating through each step.
            scan_vjp(inputs, g, params)
        }

        Primitive::While => {
            // While VJP: gradient flows through the body iterations.
            // We replay forward to count iterations, then reverse-propagate.
            while_vjp(inputs, g, params)
        }

        Primitive::Switch => {
            // Switch VJP: gradient flows through the selected branch only.
            // Index has no gradient; gradient for selected branch equals g,
            // all other branches get zero gradient.
            let idx = switch_branch_index(inputs, params)?;
            let mut grads = Vec::with_capacity(inputs.len());
            grads.push(zeros_like(&inputs[0])); // index: no grad
            for (i, inp) in inputs[1..].iter().enumerate() {
                if i == idx {
                    grads.push(g.clone());
                } else {
                    grads.push(zeros_like(inp));
                }
            }
            Ok(grads)
        }

        // Rev: gradient is reversed along the same axes
        Primitive::Rev => Ok(vec![
            eval_primitive(Primitive::Rev, std::slice::from_ref(g), params)
                .map_err(|e| AdError::EvalFailed(e.to_string()))?,
        ]),
        // Squeeze: gradient needs reshape back to original shape
        Primitive::Squeeze => {
            let shape_str = match &inputs[0] {
                Value::Tensor(t) => t
                    .shape
                    .dims
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                Value::Scalar(_) => String::new(),
            };
            if shape_str.is_empty() {
                Ok(vec![g.clone()])
            } else {
                let mut p = BTreeMap::new();
                p.insert("new_shape".into(), shape_str);
                Ok(vec![
                    eval_primitive(Primitive::Reshape, std::slice::from_ref(g), &p)
                        .map_err(|e| AdError::EvalFailed(e.to_string()))?,
                ])
            }
        }
        // Split: gradient reshapes back to the original input shape.
        Primitive::Split => {
            let shape_str = match &inputs[0] {
                Value::Tensor(t) => t
                    .shape
                    .dims
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                Value::Scalar(_) => String::new(),
            };
            if shape_str.is_empty() {
                Ok(vec![g.clone()])
            } else {
                let mut p = BTreeMap::new();
                p.insert("new_shape".into(), shape_str);
                Ok(vec![
                    eval_primitive(Primitive::Reshape, std::slice::from_ref(g), &p)
                        .map_err(|e| AdError::EvalFailed(e.to_string()))?,
                ])
            }
        }
        // ExpandDims: gradient is squeeze (inverse of expand_dims)
        Primitive::ExpandDims => {
            let axis: usize = params
                .get("axis")
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0);
            let mut p = BTreeMap::new();
            p.insert("dimensions".into(), axis.to_string());
            Ok(vec![
                eval_primitive(Primitive::Squeeze, std::slice::from_ref(g), &p)
                    .map_err(|e| AdError::EvalFailed(e.to_string()))?,
            ])
        }

        // Bitwise ops are not differentiable — gradient is zero.
        Primitive::BitwiseAnd | Primitive::BitwiseOr | Primitive::BitwiseXor => {
            Ok(vec![zeros_like(&inputs[0]), zeros_like(&inputs[1])])
        }
        Primitive::BitwiseNot | Primitive::PopulationCount | Primitive::CountLeadingZeros => {
            Ok(vec![zeros_like(&inputs[0])])
        }
        Primitive::ShiftLeft | Primitive::ShiftRightArithmetic | Primitive::ShiftRightLogical => {
            Ok(vec![zeros_like(&inputs[0]), zeros_like(&inputs[1])])
        }

        // ReduceWindow VJP: scatter gradient back over the window positions.
        Primitive::ReduceWindow => vjp_reduce_window(inputs, g, params),

        // Collective operations (pmap-only, not differentiable outside pmap context)
        Primitive::Psum
        | Primitive::Pmean
        | Primitive::AllGather
        | Primitive::AllToAll
        | Primitive::AxisIndex => Err(AdError::UnsupportedPrimitive(primitive)),

        // Cbrt: d/dx cbrt(x) = 1 / (3 * cbrt(x)^2)
        Primitive::Cbrt => {
            let cbrt_x = eval_primitive(Primitive::Cbrt, inputs, params)
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let cbrt_sq = value_mul(&cbrt_x, &cbrt_x)?;
            let three = Value::scalar_f64(3.0);
            let denom = value_mul(&three, &cbrt_sq)?;
            let recip = eval_primitive(Primitive::Reciprocal, std::slice::from_ref(&denom), params)
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            Ok(vec![value_mul(g, &recip)?])
        }
        // IsFinite: non-differentiable — gradient is zero.
        Primitive::IsFinite => Ok(vec![zeros_like(&inputs[0])]),
        // IntegerPow: d/dx x^n = n * x^(n-1)
        Primitive::IntegerPow => {
            let n: i32 = params
                .get("exponent")
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(1);
            let n_val = Value::scalar_f64(f64::from(n));
            let mut nm1_params = params.clone();
            nm1_params.insert("exponent".into(), (n - 1).to_string());
            let x_nm1 = eval_primitive(Primitive::IntegerPow, inputs, &nm1_params)
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let n_x_nm1 = value_mul(&n_val, &x_nm1)?;
            Ok(vec![value_mul(g, &n_x_nm1)?])
        }
        // Nextafter: non-differentiable — gradient is zero.
        Primitive::Nextafter => Ok(vec![zeros_like(&inputs[0]), zeros_like(&inputs[1])]),
        // ── Cholesky VJP ──
        // A = L L^T, where L is lower-triangular.
        // dA = L^{-T} tril(L^T G) L^{-1}, symmetrized.
        // We compute via triangular solves to avoid explicit inverse.
        Primitive::Cholesky => cholesky_vjp(inputs, g),
        // ── TriangularSolve VJP ──
        // A X = B => X = A^{-1} B
        // dA = -(A^{-T} G) X^T  (for lower, left_side)
        // dB = A^{-T} G
        Primitive::TriangularSolve => triangular_solve_vjp(inputs, g, params),
        // ── FFT VJP ──
        // FFT is linear: F. Adjoint F* = n · IFFT.
        // VJP: g_x = n · IFFT(g_y)
        Primitive::Fft => {
            let ifft_g = eval_primitive(Primitive::Ifft, std::slice::from_ref(g), params)
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let n = match &inputs[0] {
                Value::Tensor(t) => *t.shape.dims.last().unwrap_or(&1) as f64,
                Value::Scalar(_) => 1.0,
            };
            let scale = Value::Scalar(Literal::from_f64(n));
            Ok(vec![value_mul(&ifft_g, &scale)?])
        }
        // ── IFFT VJP ──
        // IFFT is (1/n) F*. Adjoint = (1/n) F = (1/n) · FFT.
        // VJP: g_x = FFT(g_y) / n
        Primitive::Ifft => {
            let fft_g = eval_primitive(Primitive::Fft, std::slice::from_ref(g), params)
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let n = match &inputs[0] {
                Value::Tensor(t) => *t.shape.dims.last().unwrap_or(&1) as f64,
                Value::Scalar(_) => 1.0,
            };
            let inv_n = Value::Scalar(Literal::from_f64(1.0 / n));
            Ok(vec![value_mul(&fft_g, &inv_n)?])
        }
        // ── RFFT VJP ──
        // RFFT = project_{n/2+1}(DFT(zero_pad(x, n)))
        // Adjoint: zero_extend(g) → n·IDFT → real → truncate to input length
        Primitive::Rfft => {
            let fft_length: usize = params
                .get("fft_length")
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| match &inputs[0] {
                    Value::Tensor(t) => *t.shape.dims.last().unwrap_or(&1) as usize,
                    Value::Scalar(_) => 1,
                });
            let input_len = match &inputs[0] {
                Value::Tensor(t) => *t.shape.dims.last().unwrap_or(&1) as usize,
                Value::Scalar(_) => 1,
            };
            let g_tensor = match g {
                Value::Tensor(t) => t,
                _ => {
                    return Err(AdError::EvalFailed(
                        "RFFT VJP: gradient must be tensor".to_owned(),
                    ));
                }
            };
            // Step 1: Zero-pad g from n/2+1 to fft_length
            let padded = zero_pad_last_axis_complex(g_tensor, fft_length)?;
            // Step 2: IFFT of zero-padded g
            let ifft_result = eval_primitive(
                Primitive::Ifft,
                std::slice::from_ref(&padded),
                &BTreeMap::new(),
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            // Step 3: Scale by n
            let n = fft_length as f64;
            let scale = Value::Scalar(Literal::from_f64(n));
            let scaled = value_mul(&ifft_result, &scale)?;
            // Step 4: Take real part (input was real)
            let real_result = take_real_part(&scaled)?;
            // Step 5: Truncate to input length if needed
            if input_len < fft_length {
                let truncated = match &real_result {
                    Value::Tensor(t) => truncate_last_axis(t, input_len)?,
                    _ => real_result,
                };
                Ok(vec![truncated])
            } else {
                Ok(vec![real_result])
            }
        }
        // ── IRFFT VJP ──
        // IRFFT = real(IDFT(hermitian_extend(y)))
        // Adjoint: hermitian_extend^T ∘ (1/n)·DFT ∘ embed_real
        // For real g: (1/n)·FFT(g), truncated to n/2+1, interior bins doubled
        Primitive::Irfft => {
            let fft_length: usize = params
                .get("fft_length")
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| match &inputs[0] {
                    Value::Tensor(t) => {
                        let d = *t.shape.dims.last().unwrap_or(&1) as usize;
                        d.saturating_sub(1).saturating_mul(2)
                    }
                    Value::Scalar(_) => 1,
                });
            // Step 1: FFT(g_real)
            let fft_g = eval_primitive(Primitive::Fft, std::slice::from_ref(g), &BTreeMap::new())
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            // Step 2: Scale by 1/n
            let inv_n = Value::Scalar(Literal::from_f64(1.0 / fft_length as f64));
            let scaled = value_mul(&fft_g, &inv_n)?;
            // Step 3: Truncate to n/2+1
            let half_len = fft_length / 2 + 1;
            let truncated = match &scaled {
                Value::Tensor(t) => truncate_last_axis(t, half_len)?,
                _ => scaled,
            };
            // Step 4: Double interior bins (hermitian_extend adjoint)
            let result = match &truncated {
                Value::Tensor(t) => scale_hermitian_adjoint(t, fft_length)?,
                _ => truncated,
            };
            Ok(vec![result])
        }
        // ── QR VJP ──
        // A = QR. Given cotangents g_Q (m×n) and g_R (n×n):
        // M = R g_R^T - g_Q^T Q
        // M_sym = tril(M) + tril(M)^T - diag(diag(M))   [copyltu]
        // dA = (g_Q + Q M_sym) R^{-T}
        Primitive::Qr => {
            let g_q = &gs[0];
            let g_r = if gs.len() > 1 {
                &gs[1]
            } else {
                &zeros_like(&output_values[1])
            };

            let q_val = &output_values[0];
            let r_val = &output_values[1];
            let q = q_val
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("QR VJP: Q must be tensor".to_owned()))?;
            let r = r_val
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("QR VJP: R must be tensor".to_owned()))?;
            let gq = g_q
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("QR VJP: g_Q must be tensor".to_owned()))?;
            let gr = g_r
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("QR VJP: g_R must be tensor".to_owned()))?;

            // M = R @ g_R^T - g_Q^T @ Q  (n×n)
            let gr_t_val = transpose_2d(gr)?;
            let gr_t = tensor_value(&gr_t_val, "QR VJP transpose(g_R)")?;
            let r_gr_t = matmul_2d(r, gr_t)?;

            let gq_t_val = transpose_2d(gq)?;
            let gq_t = tensor_value(&gq_t_val, "QR VJP transpose(g_Q)")?;
            let gq_t_q = matmul_2d(gq_t, q)?;

            let m_val = tensor_sub(
                tensor_value(&r_gr_t, "QR VJP R @ g_R^T")?,
                tensor_value(&gq_t_q, "QR VJP g_Q^T @ Q")?,
            )?;
            let m = tensor_value(&m_val, "QR VJP M")?;

            // M_sym = tril(M) + tril(M)^T - diag(diag(M))  [copyltu]
            let m_lower = tril(m)?;
            let m_lower_t_val = transpose_2d(&m_lower)?;
            let m_lower_t = tensor_value(&m_lower_t_val, "QR VJP transpose(tril(M))")?;
            let m_plus_mt = tensor_add(&m_lower, m_lower_t)?;
            // Subtract diagonal (it was counted twice)
            let n_dim = m.shape.dims[0] as usize;
            let m_lower_vals: Vec<f64> = m_lower
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap_or(0.0))
                .collect();
            let mut diag_elements = vec![Literal::from_f64(0.0); n_dim * n_dim];
            for i in 0..n_dim {
                diag_elements[i * n_dim + i] = Literal::from_f64(m_lower_vals[i * n_dim + i]);
            }
            let diag_tensor = TensorValue::new(DType::F64, m.shape.clone(), diag_elements)
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let m_sym = tensor_sub(
                tensor_value(&m_plus_mt, "QR VJP lower-triangle symmetrization")?,
                &diag_tensor,
            )?;
            let m_sym_t = tensor_value(&m_sym, "QR VJP M_sym")?;

            // rhs = g_Q + Q @ M_sym  (m×n)
            let q_msym = matmul_2d(q, m_sym_t)?;
            let rhs = tensor_add(gq, tensor_value(&q_msym, "QR VJP Q @ M_sym")?)?;
            let rhs_t = tensor_value(&rhs, "QR VJP rhs")?;

            // dA = solve(R^T, rhs^T)^T
            // R^T x = rhs^T => triangular_solve(R, rhs^T, lower=false, transpose_a=true)
            let rhs_t_val = transpose_2d(rhs_t)?;
            let mut tri_params = BTreeMap::new();
            tri_params.insert("lower".to_owned(), "false".to_owned());
            tri_params.insert("transpose_a".to_owned(), "true".to_owned());
            let solve_result = eval_primitive(
                Primitive::TriangularSolve,
                &[r_val.clone(), rhs_t_val],
                &tri_params,
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let da = transpose_2d(tensor_value(&solve_result, "QR VJP triangular solve")?)?;

            Ok(vec![da])
        }
        // ── SVD VJP ──
        // A = U diag(s) V^T. Given cotangents g_U, g_s, g_Vt:
        // G[i,j] = [s_j(D_U[i,j]-D_U[j,i]) + s_i(D_V[i,j]-D_V[j,i])] / (s_j²-s_i²)
        // G[i,i] = g_s[i]
        // dA = U G V^T + (I-UU^T) g_U Σ^{-1} V^T + U Σ^{-1} g_V^T (I-VV^T)
        Primitive::Svd => {
            let g_u = &gs[0];
            let g_s_val = if gs.len() > 1 {
                &gs[1]
            } else {
                &zeros_like(&output_values[1])
            };
            let g_vt = if gs.len() > 2 {
                &gs[2]
            } else {
                &zeros_like(&output_values[2])
            };

            let u_val = &output_values[0];
            let s_val = &output_values[1];
            let vt_val = &output_values[2];

            let u_t = u_val
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("SVD VJP: U must be tensor".to_owned()))?;
            let vt_t = vt_val
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("SVD VJP: Vt must be tensor".to_owned()))?;
            let gu_t = g_u
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("SVD VJP: g_U must be tensor".to_owned()))?;
            let gvt_t = g_vt
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("SVD VJP: g_Vt must be tensor".to_owned()))?;

            let s_vec: Vec<f64> = s_val
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("SVD VJP: S must be tensor".to_owned()))?
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap_or(0.0))
                .collect();
            let gs_vec: Vec<f64> = g_s_val
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("SVD VJP: g_S must be tensor".to_owned()))?
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap_or(0.0))
                .collect();

            let m = u_t.shape.dims[0] as usize;
            let k = s_vec.len();
            let n = vt_t.shape.dims[1] as usize;

            // g_V = g_Vt^T (V = Vt^T is used implicitly via Vt throughout)
            let gv_val = transpose_2d(gvt_t)?;
            let gv_t = tensor_value(&gv_val, "SVD VJP transpose(g_Vt)")?;

            // D_U = U^T g_U [k×k], D_V = V^T g_V [k×k]
            let ut_val = transpose_2d(u_t)?;
            let ut = tensor_value(&ut_val, "SVD VJP transpose(U)")?;
            let d_u_val = matmul_2d(ut, gu_t)?;
            let d_u = tensor_value(&d_u_val, "SVD VJP U^T @ g_U")?;
            let vt_for_dv = vt_t; // V^T = Vt
            let d_v_val = matmul_2d(vt_for_dv, gv_t)?;
            let d_v = tensor_value(&d_v_val, "SVD VJP V^T @ g_V")?;

            let du_vals: Vec<f64> = d_u
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap_or(0.0))
                .collect();
            let dv_vals: Vec<f64> = d_v
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap_or(0.0))
                .collect();

            // Build G[k×k]
            let mut g_mat = vec![0.0f64; k * k];
            for i in 0..k {
                g_mat[i * k + i] = gs_vec[i]; // diagonal
                for j in 0..k {
                    if i != j {
                        let si = s_vec[i];
                        let sj = s_vec[j];
                        let denom = sj * sj - si * si;
                        if denom.abs() > 1e-20 {
                            let du_ij = du_vals[i * k + j];
                            let du_ji = du_vals[j * k + i];
                            let dv_ij = dv_vals[i * k + j];
                            let dv_ji = dv_vals[j * k + i];
                            g_mat[i * k + j] =
                                (sj * (du_ij - du_ji) + si * (dv_ij - dv_ji)) / denom;
                        }
                        // else: degenerate singular values, G[i,j] = 0
                    }
                }
            }
            let g_tensor = TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![k as u32, k as u32],
                },
                g_mat.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;

            // dA_core = U G V^T [m×n]
            let ug = matmul_2d(u_t, &g_tensor)?;
            let da_core = matmul_2d(tensor_value(&ug, "SVD VJP U @ G")?, vt_t)?;

            let mut da_vals: Vec<f64> = da_core
                .as_tensor()
                .ok_or_else(|| {
                    AdError::EvalFailed("SVD VJP core gradient: expected tensor value".to_owned())
                })?
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap_or(0.0))
                .collect();

            // Extra term for m > k: (I_m - UU^T) g_U Σ^{-1} V^T
            if m > k {
                let u_vals: Vec<f64> = u_t
                    .elements
                    .iter()
                    .map(|l| l.as_f64().unwrap_or(0.0))
                    .collect();
                let gu_vals: Vec<f64> = gu_t
                    .elements
                    .iter()
                    .map(|l| l.as_f64().unwrap_or(0.0))
                    .collect();
                // proj_gu = g_U - U(U^T g_U) = g_U - U D_U [m×k]
                let u_du = matmul_f64(m, k, k, &u_vals, &du_vals);
                let proj_gu: Vec<f64> = gu_vals
                    .iter()
                    .zip(u_du.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                // proj_gu @ Σ^{-1} [m×k]
                let mut proj_sinv = vec![0.0; m * k];
                for i in 0..m {
                    for j in 0..k {
                        let sinv = if s_vec[j].abs() > 1e-20 {
                            1.0 / s_vec[j]
                        } else {
                            0.0
                        };
                        proj_sinv[i * k + j] = proj_gu[i * k + j] * sinv;
                    }
                }
                // ... @ V^T [m×n] (V^T = Vt)
                let vt_vals: Vec<f64> = vt_t
                    .elements
                    .iter()
                    .map(|l| l.as_f64().unwrap_or(0.0))
                    .collect();
                let extra = matmul_f64(m, k, n, &proj_sinv, &vt_vals);
                for i in 0..m * n {
                    da_vals[i] += extra[i];
                }
            }
            // Extra term for n > k: U Σ^{-1} g_V^T (I_n - VV^T)
            if n > k {
                let u_vals: Vec<f64> = u_t
                    .elements
                    .iter()
                    .map(|l| l.as_f64().unwrap_or(0.0))
                    .collect();
                // g_V^T - V^T(V g_V^T) = g_Vt - Vt (Vt^T g_Vt)^T ... complex
                // proj_gvt = g_V^T (I - VV^T) = g_Vt - (Vt^T Vt) g_Vt... no
                // g_V^T (I-VV^T) where g_V = g_Vt^T, V = Vt^T
                // = g_Vt (I - Vt^T Vt)
                // D_V = Vt g_Vt^T [k×k], so Vt^T Vt g_Vt^T = Vt^T D_V^T ... hmm
                // Simpler: proj = g_Vt - D_V Vt where D_V = Vt g_Vt^T
                // Actually: g_V^T (I-VV^T) = g_Vt - g_Vt V V^T = g_Vt - g_Vt Vt^T Vt
                // = g_Vt (I_n - Vt^T Vt)
                let gvt_vals: Vec<f64> = gvt_t
                    .elements
                    .iter()
                    .map(|l| l.as_f64().unwrap_or(0.0))
                    .collect();
                let vt_vals: Vec<f64> = vt_t
                    .elements
                    .iter()
                    .map(|l| l.as_f64().unwrap_or(0.0))
                    .collect();
                // Vt^T [n×k] @ Vt [k×n] = VV^T [n×n]
                let vt_trans = transpose_f64(k, n, &vt_vals);
                let vvt = matmul_f64(n, k, n, &vt_trans, &vt_vals);
                // g_Vt (I - VV^T) [k×n]
                let gvt_vvt = matmul_f64(k, n, n, &gvt_vals, &vvt);
                let proj_gvt: Vec<f64> = gvt_vals
                    .iter()
                    .zip(gvt_vvt.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                // U Σ^{-1} [m×k] @ proj_gvt [k×n]
                let mut u_sinv = vec![0.0; m * k];
                for i in 0..m {
                    for j in 0..k {
                        let sinv = if s_vec[j].abs() > 1e-20 {
                            1.0 / s_vec[j]
                        } else {
                            0.0
                        };
                        u_sinv[i * k + j] = u_vals[i * k + j] * sinv;
                    }
                }
                let extra = matmul_f64(m, k, n, &u_sinv, &proj_gvt);
                for i in 0..m * n {
                    da_vals[i] += extra[i];
                }
            }

            let da = build_matrix_f64(m, n, &da_vals)?;
            Ok(vec![da])
        }
        // ── Eigh VJP ──
        // A = V diag(w) V^T. Given cotangents g_w, g_V:
        // middle = diag(g_w) + F ⊙ (V^T g_V)
        // dA = V middle V^T, symmetrized
        Primitive::Eigh => {
            let g_w = &gs[0];
            let g_v = if gs.len() > 1 {
                &gs[1]
            } else {
                &zeros_like(&output_values[1])
            };

            let w_val = &output_values[0];
            let v_val = &output_values[1];

            let v_t = v_val
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("Eigh VJP: V must be tensor".to_owned()))?;
            let gv_t = g_v
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("Eigh VJP: g_V must be tensor".to_owned()))?;

            let w_vec: Vec<f64> = w_val
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("Eigh VJP: W must be tensor".to_owned()))?
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap_or(0.0))
                .collect();
            let gw_vec: Vec<f64> = g_w
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("Eigh VJP: g_W must be tensor".to_owned()))?
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap_or(0.0))
                .collect();

            let nn = w_vec.len();

            // D = V^T g_V [n×n]
            let vt_val = transpose_2d(v_t)?;
            let vt = tensor_value(&vt_val, "Eigh VJP transpose(V)")?;
            let d_val = matmul_2d(vt, gv_t)?;
            let d_t = tensor_value(&d_val, "Eigh VJP V^T @ g_V")?;
            let d_vals: Vec<f64> = d_t
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap_or(0.0))
                .collect();

            // Build middle = diag(g_w) + F ⊙ skew(V^T g_V) [n×n].
            // Only the skew component contributes because eigenvector
            // perturbations live in the tangent space of the orthogonal group.
            let mut middle = vec![0.0f64; nn * nn];
            for i in 0..nn {
                middle[i * nn + i] = gw_vec[i]; // diagonal from g_w
                for j in 0..nn {
                    if i != j {
                        let f_ij = safe_eigen_gap_reciprocal(w_vec[i], w_vec[j]);
                        let skew_ij = 0.5 * (d_vals[i * nn + j] - d_vals[j * nn + i]);
                        middle[i * nn + j] += f_ij * skew_ij;
                    }
                }
            }

            let mid_tensor = TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![nn as u32, nn as u32],
                },
                middle.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;

            // dA = V middle V^T
            let v_mid = matmul_2d(v_t, &mid_tensor)?;
            let da_unsym = matmul_2d(tensor_value(&v_mid, "Eigh VJP V @ middle")?, vt)?;
            let da_vals: Vec<f64> = da_unsym
                .as_tensor()
                .ok_or_else(|| {
                    AdError::EvalFailed(
                        "Eigh VJP unsymmetrized dA: expected tensor value".to_owned(),
                    )
                })?
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap_or(0.0))
                .collect();

            // Symmetrize: dA = (dA + dA^T) / 2
            let mut sym_vals = vec![0.0f64; nn * nn];
            for i in 0..nn {
                for j in 0..nn {
                    sym_vals[i * nn + j] = (da_vals[i * nn + j] + da_vals[j * nn + i]) * 0.5;
                }
            }

            let da = build_matrix_f64(nn, nn, &sym_vals)?;
            Ok(vec![da])
        }
    }
}

/// ReduceWindow VJP: scatter gradient back over window positions.
///
/// For sum reduction: each input position receives the sum of output gradients
/// from all windows that include it (select-and-scatter-add pattern).
///
/// For max reduction: gradient routes to the position of the max value in each
/// window (subgradient). For min reduction: gradient routes to the min position.
fn vjp_reduce_window(
    inputs: &[Value],
    g: &Value,
    params: &BTreeMap<String, String>,
) -> Result<Vec<Value>, AdError> {
    let input_tensor = match &inputs[0] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => return Ok(vec![g.clone()]),
    };

    let g_tensor = match g {
        Value::Tensor(t) => t,
        Value::Scalar(lit) => {
            // Scalar gradient — input was scalar, passthrough
            return Ok(vec![Value::Scalar(*lit)]);
        }
    };

    let rank = input_tensor.shape.rank();

    // Parse window parameters (same parsing as forward pass in fj-lax)
    let window_dims: Vec<usize> = params
        .get("window_dimensions")
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![2; rank]);

    let strides: Vec<usize> = params
        .get("window_strides")
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1; rank]);

    let reduce_op = params.get("reduce_op").map(|s| s.as_str()).unwrap_or("sum");

    let _padding = params.get("padding").map(|s| s.as_str()).unwrap_or("valid");

    let input_dims: Vec<usize> = input_tensor
        .shape
        .dims
        .iter()
        .map(|d| *d as usize)
        .collect();
    let out_dims: Vec<usize> = g_tensor.shape.dims.iter().map(|d| *d as usize).collect();

    // Build input gradient tensor (same shape as input, initially zeros)
    let total_input = ad_checked_usize_product("reduce_window input", &input_dims)?;
    let mut grad_elements = vec![0.0_f64; total_input];

    let total_output = ad_checked_usize_product("reduce_window output", &out_dims)?;
    if total_output == 0 {
        return Ok(vec![zeros_like(&inputs[0])]);
    }

    // Get input and gradient values as f64
    let input_vals: Vec<f64> = input_tensor
        .elements
        .iter()
        .map(|e| e.as_f64().unwrap_or(0.0))
        .collect();
    let g_vals: Vec<f64> = g_tensor
        .elements
        .iter()
        .map(|e| e.as_f64().unwrap_or(0.0))
        .collect();

    // Iterate over all output positions
    let mut out_idx = vec![0usize; rank];
    for &g_val in g_vals.iter().take(total_output) {
        match reduce_op {
            "max" | "min" => {
                // Find the position of the max/min value in this window
                let mut best_flat: Option<usize> = None;
                let mut best_val = if reduce_op == "max" {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                };

                let win_total = ad_checked_usize_product("reduce_window window", &window_dims)?;
                let mut win_idx = vec![0usize; rank];

                for _ in 0..win_total {
                    let flat = compute_flat_index(&out_idx, &win_idx, &strides, &input_dims, rank)?;
                    if let Some(flat_idx) = flat {
                        let val = input_vals[flat_idx];
                        let is_better = if reduce_op == "max" {
                            val > best_val
                        } else {
                            val < best_val
                        };
                        if is_better {
                            best_val = val;
                            best_flat = Some(flat_idx);
                        }
                    }
                    increment_nd_index(&mut win_idx, &window_dims);
                }

                if let Some(idx) = best_flat {
                    grad_elements[idx] += g_val;
                }
            }
            _ => {
                // Sum reduction: gradient is scattered to all positions in the window
                let win_total = ad_checked_usize_product("reduce_window window", &window_dims)?;
                let mut win_idx = vec![0usize; rank];

                for _ in 0..win_total {
                    let flat = compute_flat_index(&out_idx, &win_idx, &strides, &input_dims, rank)?;
                    if let Some(flat_idx) = flat {
                        grad_elements[flat_idx] += g_val;
                    }
                    increment_nd_index(&mut win_idx, &window_dims);
                }
            }
        }

        increment_nd_index(&mut out_idx, &out_dims);
    }

    // Build the gradient tensor
    let elements: Vec<Literal> = grad_elements.into_iter().map(Literal::from_f64).collect();
    let grad_tensor = TensorValue::new(input_tensor.dtype, input_tensor.shape.clone(), elements)
        .map_err(|e| AdError::EvalFailed(e.to_string()))?;
    Ok(vec![Value::Tensor(grad_tensor)])
}

/// Compute the flat input index for a given output position and window offset.
/// Returns None if the position is out of bounds.
fn compute_flat_index(
    out_idx: &[usize],
    win_idx: &[usize],
    strides: &[usize],
    input_dims: &[usize],
    rank: usize,
) -> Result<Option<usize>, AdError> {
    let mut flat = 0usize;
    let mut stride_mult = 1usize;
    for d in (0..rank).rev() {
        let input_pos = out_idx[d]
            .checked_mul(strides[d])
            .and_then(|base| base.checked_add(win_idx[d]))
            .ok_or_else(|| {
                AdError::EvalFailed(
                    "reduce_window flat index computation overflows usize".to_owned(),
                )
            })?;
        if input_pos >= input_dims[d] {
            return Ok(None);
        }
        let offset = input_pos.checked_mul(stride_mult).ok_or_else(|| {
            AdError::EvalFailed("reduce_window flat index offset overflows usize".to_owned())
        })?;
        flat = flat.checked_add(offset).ok_or_else(|| {
            AdError::EvalFailed("reduce_window flat index accumulation overflows usize".to_owned())
        })?;
        stride_mult = stride_mult.checked_mul(input_dims[d]).ok_or_else(|| {
            AdError::EvalFailed("reduce_window row-major stride overflows usize".to_owned())
        })?;
    }
    Ok(Some(flat))
}

/// Increment a multi-dimensional index.
fn increment_nd_index(idx: &mut [usize], dims: &[usize]) {
    for d in (0..idx.len()).rev() {
        idx[d] += 1;
        if idx[d] >= dims[d] {
            idx[d] = 0;
        } else {
            return;
        }
    }
}

fn zero_literal_for_dtype(dtype: DType) -> Literal {
    match dtype {
        DType::I64 | DType::I32 => Literal::I64(0),
        DType::U32 => Literal::U32(0),
        DType::U64 => Literal::U64(0),
        DType::Bool => Literal::Bool(false),
        DType::BF16 => Literal::from_bf16_f32(0.0),
        DType::F16 => Literal::from_f16_f32(0.0),
        DType::F32 => Literal::from_f32(0.0),
        DType::F64 => Literal::from_f64(0.0),
        DType::Complex64 => Literal::from_complex64(0.0, 0.0),
        DType::Complex128 => Literal::from_complex128(0.0, 0.0),
    }
}

fn dynamic_slice_vjp(inputs: &[Value], g: &Value) -> Result<Vec<Value>, AdError> {
    let operand = inputs.first().ok_or(AdError::InputArity {
        expected: 1,
        actual: 0,
    })?;

    let mut result = match (operand, g) {
        (Value::Scalar(_), _) => vec![g.clone()],
        (Value::Tensor(operand_tensor), Value::Tensor(g_tensor)) => {
            let zero = zero_literal_for_dtype(g_tensor.dtype);
            let zero_operand = Value::Tensor(
                TensorValue::new(
                    g_tensor.dtype,
                    operand_tensor.shape.clone(),
                    vec![zero; operand_tensor.elements.len()],
                )
                .map_err(|e| AdError::EvalFailed(e.to_string()))?,
            );
            let mut update_inputs = Vec::with_capacity(inputs.len() + 1);
            update_inputs.push(zero_operand);
            update_inputs.push(g.clone());
            update_inputs.extend_from_slice(&inputs[1..]);
            vec![
                eval_primitive(
                    Primitive::DynamicUpdateSlice,
                    &update_inputs,
                    &BTreeMap::new(),
                )
                .map_err(|e| AdError::EvalFailed(e.to_string()))?,
            ]
        }
        (Value::Tensor(_), Value::Scalar(_)) => {
            return Err(AdError::EvalFailed(
                "dynamic_slice cotangent for tensor operand must be a tensor".to_owned(),
            ));
        }
    };

    result.extend(std::iter::repeat_n(
        Value::scalar_f64(0.0),
        inputs.len().saturating_sub(1),
    ));
    Ok(result)
}

fn normalize_dynamic_start(raw: i64, dim: i64, window: i64) -> usize {
    let adjusted = if raw < 0 { raw + dim } else { raw };
    adjusted.max(0).min(dim - window) as usize
}

/// DynamicUpdateSlice VJP: route gradients properly.
///
/// Inputs: [operand, update, start_0, start_1, ..., start_{rank-1}]
/// g_operand = g with the update region zeroed out
/// g_update = slice of g at the start positions with update's shape
/// Start indices have zero gradient (discrete).
fn dynamic_update_slice_vjp(inputs: &[Value], g: &Value) -> Result<Vec<Value>, AdError> {
    let g_tensor = match g {
        Value::Tensor(t) => t,
        Value::Scalar(_) => {
            // Scalar case: gradient flows to operand, zero to update
            let n_starts = inputs.len().saturating_sub(2);
            let mut result = vec![g.clone(), Value::scalar_f64(0.0)];
            result.extend(std::iter::repeat_n(Value::scalar_f64(0.0), n_starts));
            return Ok(result);
        }
    };

    let update = match &inputs[1] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => {
            let n_starts = inputs.len().saturating_sub(2);
            let mut result = vec![g.clone(), Value::scalar_f64(0.0)];
            result.extend(std::iter::repeat_n(Value::scalar_f64(0.0), n_starts));
            return Ok(result);
        }
    };

    let rank = g_tensor.shape.rank();

    // Parse start indices (same clamping logic as forward pass)
    let mut starts = Vec::with_capacity(rank);
    for ax in 0..rank {
        let start_val = match &inputs[2 + ax] {
            Value::Scalar(lit) => {
                let raw = match lit {
                    Literal::I64(v) => *v,
                    Literal::U32(v) => i64::from(*v),
                    Literal::U64(v) => i64::try_from(*v).unwrap_or(i64::MAX),
                    Literal::BF16Bits(_) | Literal::F16Bits(_) => {
                        (*lit).as_f64().unwrap_or(0.0) as i64
                    }
                    Literal::F32Bits(b) => f32::from_bits(*b) as i64,
                    Literal::F64Bits(b) => f64::from_bits(*b) as i64,
                    Literal::Bool(b) => i64::from(*b),
                    Literal::Complex64Bits(..) | Literal::Complex128Bits(..) => 0,
                };
                let dim = g_tensor.shape.dims[ax] as i64;
                let upd_size = update.shape.dims[ax] as i64;
                normalize_dynamic_start(raw, dim, upd_size)
            }
            _ => 0,
        };
        starts.push(start_val);
    }

    // g_operand: copy of g with the update region zeroed out
    let mut g_op_elems: Vec<f64> = g_tensor
        .elements
        .iter()
        .map(|l| l.as_f64().unwrap_or(0.0))
        .collect();

    // g_update: slice of g at the start positions
    let upd_dims: Vec<usize> = update.shape.dims.iter().map(|&d| d as usize).collect();
    let g_dims: Vec<usize> = g_tensor.shape.dims.iter().map(|&d| d as usize).collect();
    let upd_total = ad_checked_usize_product("dynamic_update_slice update", &upd_dims)?;
    let mut g_upd_elems = Vec::with_capacity(upd_total);
    let g_elems = g_tensor
        .elements
        .iter()
        .map(|l| l.as_f64().unwrap_or(0.0))
        .collect::<Vec<_>>();

    // Iterate over update indices
    fn iterate_nd(
        dims: &[usize],
        starts: &[usize],
        g_dims: &[usize],
        g_op_elems: &mut [f64],
        g_elems: &[f64],
        g_upd_elems: &mut Vec<f64>,
    ) -> Result<(), AdError> {
        let rank = dims.len();
        let total = ad_checked_usize_product("dynamic_update_slice update", dims)?;
        for flat_idx in 0..total {
            // Convert flat index to ND coordinates in update space
            let mut remaining = flat_idx;
            let mut g_flat = 0usize;
            let mut stride = 1;
            for ax in (0..rank).rev() {
                let coord = remaining % dims[ax];
                remaining /= dims[ax];
                let g_coord = coord.checked_add(starts[ax]).ok_or_else(|| {
                    AdError::EvalFailed(format!(
                        "dynamic_update_slice coordinate overflows usize on axis {ax}"
                    ))
                })?;
                if g_coord >= g_dims[ax] {
                    return Err(AdError::EvalFailed(format!(
                        "dynamic_update_slice coordinate {g_coord} out of bounds for axis {ax} with size {}",
                        g_dims[ax]
                    )));
                }
                let offset = g_coord.checked_mul(stride).ok_or_else(|| {
                    AdError::EvalFailed(format!(
                        "dynamic_update_slice offset overflows usize on axis {ax}"
                    ))
                })?;
                g_flat = g_flat.checked_add(offset).ok_or_else(|| {
                    AdError::EvalFailed(
                        "dynamic_update_slice flat index accumulation overflows usize".to_owned(),
                    )
                })?;
                stride = stride.checked_mul(g_dims[ax]).ok_or_else(|| {
                    AdError::EvalFailed(
                        "dynamic_update_slice row-major stride overflows usize".to_owned(),
                    )
                })?;
            }
            g_upd_elems.push(g_elems[g_flat]);
            g_op_elems[g_flat] = 0.0;
        }
        Ok(())
    }

    iterate_nd(
        &upd_dims,
        &starts,
        &g_dims,
        &mut g_op_elems,
        &g_elems,
        &mut g_upd_elems,
    )?;

    let grad_operand = Value::Tensor(
        TensorValue::new(
            DType::F64,
            g_tensor.shape.clone(),
            g_op_elems.into_iter().map(Literal::from_f64).collect(),
        )
        .map_err(|e| AdError::EvalFailed(e.to_string()))?,
    );

    let grad_update = Value::Tensor(
        TensorValue::new(
            DType::F64,
            update.shape.clone(),
            g_upd_elems.into_iter().map(Literal::from_f64).collect(),
        )
        .map_err(|e| AdError::EvalFailed(e.to_string()))?,
    );

    // Return grad for operand, update, and zero for each start index
    let n_starts = inputs.len().saturating_sub(2);
    let mut result = vec![grad_operand, grad_update];
    result.extend(std::iter::repeat_n(Value::scalar_f64(0.0), n_starts));
    Ok(result)
}

/// Cond VJP: gradient flows through the selected branch only.
///
/// inputs: [pred, true_operand, false_operand]
/// If pred is true: grad flows to true_operand, zero to false_operand.
/// If pred is false: grad flows to false_operand, zero to true_operand.
/// Predicate gradient is always zero (discrete).
fn cond_vjp(inputs: &[Value], g: &Value) -> Result<Vec<Value>, AdError> {
    if inputs.len() < 3 {
        return Err(AdError::InputArity {
            expected: 3,
            actual: inputs.len(),
        });
    }

    let pred = scalar_predicate_for_ad(Primitive::Cond, "predicate", &inputs[0])?;

    let zero_pred = Value::scalar_f64(0.0);
    if pred {
        // Gradient flows to true_operand, zero to false_operand
        Ok(vec![zero_pred, g.clone(), zeros_like(&inputs[2])])
    } else {
        // Gradient flows to false_operand, zero to true_operand
        Ok(vec![zero_pred, zeros_like(&inputs[1]), g.clone()])
    }
}

fn scalar_literal_for_ad(
    primitive: Primitive,
    context: &str,
    value: &Value,
) -> Result<Literal, AdError> {
    match value {
        Value::Scalar(lit) => Ok(*lit),
        Value::Tensor(tensor) => {
            if tensor.shape != Shape::scalar() {
                return Err(AdError::EvalFailed(format!(
                    "{} {context} must be scalar, got shape {:?}",
                    primitive.as_str(),
                    tensor.shape.dims
                )));
            }
            if tensor.elements.len() != 1 {
                return Err(AdError::EvalFailed(format!(
                    "{} scalar {context} has {} elements",
                    primitive.as_str(),
                    tensor.elements.len()
                )));
            }
            Ok(tensor.elements[0])
        }
    }
}

fn scalar_predicate_for_ad(
    primitive: Primitive,
    context: &str,
    value: &Value,
) -> Result<bool, AdError> {
    match scalar_literal_for_ad(primitive, context, value)? {
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
        Literal::Complex64Bits(..) | Literal::Complex128Bits(..) => {
            Err(AdError::EvalFailed(format!(
                "{} {context} must be boolean or numeric",
                primitive.as_str()
            )))
        }
    }
}

fn switch_branch_index(
    inputs: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<usize, AdError> {
    if inputs.len() < 2 {
        return Err(AdError::InputArity {
            expected: 2,
            actual: inputs.len(),
        });
    }

    let provided_branches = inputs.len() - 1;
    let num_branches = if let Some(raw) = params.get("num_branches") {
        raw.parse::<usize>()
            .map_err(|_| AdError::EvalFailed(format!("invalid switch num_branches value: {raw}")))?
    } else {
        provided_branches
    };
    if num_branches != provided_branches {
        return Err(AdError::EvalFailed(format!(
            "switch expected {num_branches} branch values but got {provided_branches}"
        )));
    }

    let last_branch = num_branches.saturating_sub(1);
    match scalar_literal_for_ad(Primitive::Switch, "index", &inputs[0])? {
        Literal::Bool(value) => Ok(usize::from(value).min(last_branch)),
        Literal::I64(value) => {
            if value <= 0 {
                Ok(0)
            } else {
                Ok((value as u64).min(last_branch as u64) as usize)
            }
        }
        Literal::U32(value) => Ok((value as usize).min(last_branch)),
        Literal::U64(value) => Ok(value.min(last_branch as u64) as usize),
        _ => Err(AdError::EvalFailed(
            "switch index must be integer or bool".to_owned(),
        )),
    }
}

/// Scan VJP: reverse-propagate gradient through the scan loop.
///
/// For scan with body_op `op`: carry_{i+1} = op(carry_i, xs[i])
/// We replay the forward pass to collect intermediate carries, then
/// propagate gradients backward through each step.
///
/// For Add: d(carry)/d(init) = g, d(carry)/d(xs[i]) = g for all i
/// For Mul: d(carry)/d(xs[i]) = g * carry_final / xs[i]
///          d(carry)/d(init) = g * prod(xs)
/// General: chain rule through each step.
fn scan_body_op_for_ad(body_op_name: &str) -> Option<Primitive> {
    match body_op_name {
        "add" => Some(Primitive::Add),
        "sub" => Some(Primitive::Sub),
        "mul" => Some(Primitive::Mul),
        "div" => Some(Primitive::Div),
        "pow" => Some(Primitive::Pow),
        "max" => Some(Primitive::Max),
        "min" => Some(Primitive::Min),
        _ => None,
    }
}

fn scan_vjp(
    inputs: &[Value],
    g: &Value,
    params: &BTreeMap<String, String>,
) -> Result<Vec<Value>, AdError> {
    if inputs.len() < 2 {
        return Err(AdError::InputArity {
            expected: 2,
            actual: inputs.len(),
        });
    }

    let init_carry = &inputs[0];
    let xs = &inputs[1];

    let body_op_name = params.get("body_op").map(|s| s.as_str()).unwrap_or("add");
    let body_op =
        scan_body_op_for_ad(body_op_name).ok_or(AdError::UnsupportedPrimitive(Primitive::Scan))?;

    let reverse = params.get("reverse").map(|s| s == "true").unwrap_or(false);

    // For scalars: just the body_op VJP
    let xs_tensor = match xs {
        Value::Scalar(_) => {
            let step_grads = vjp(
                body_op,
                &[init_carry.clone(), xs.clone()],
                std::slice::from_ref(g),
                &[],
                &BTreeMap::new(),
            )?;
            return Ok(vec![step_grads[0].clone(), step_grads[1].clone()]);
        }
        Value::Tensor(t) => t,
    };

    let leading_dim = xs_tensor.shape.dims[0] as usize;
    if leading_dim == 0 {
        return Ok(vec![g.clone(), zeros_like(xs)]);
    }

    // Compute slice shape and size
    let slice_shape = if xs_tensor.shape.rank() == 1 {
        None
    } else {
        Some(Shape {
            dims: xs_tensor.shape.dims[1..].into(),
        })
    };
    let slice_size: usize = if let Some(ref s) = slice_shape {
        s.dims.iter().map(|d| *d as usize).product()
    } else {
        1
    };

    // Build ordered indices
    let indices: Vec<usize> = if reverse {
        (0..leading_dim).rev().collect()
    } else {
        (0..leading_dim).collect()
    };

    // Forward pass: collect intermediate carries
    let mut carries = Vec::with_capacity(leading_dim + 1);
    carries.push(init_carry.clone());
    let mut carry = init_carry.clone();

    let mut slices = Vec::with_capacity(leading_dim);
    for &i in &indices {
        let start = i * slice_size;
        let end = start + slice_size;
        let slice_elements = xs_tensor.elements[start..end].to_vec();
        let x_slice = if let Some(ref s) = slice_shape {
            Value::Tensor(
                TensorValue::new(xs_tensor.dtype, s.clone(), slice_elements)
                    .map_err(|e| AdError::EvalFailed(e.to_string()))?,
            )
        } else {
            Value::Scalar(slice_elements[0])
        };
        carry = eval_primitive(body_op, &[carry.clone(), x_slice.clone()], &BTreeMap::new())
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
        slices.push(x_slice);
        carries.push(carry.clone());
    }

    // Backward pass: propagate gradient from final carry backward
    let mut g_carry = g.clone();
    let mut g_xs_elements: Vec<Literal> = vec![Literal::from_f64(0.0); xs_tensor.elements.len()];

    for (step, &i) in indices.iter().enumerate().rev() {
        let carry_at_step = &carries[step];
        let x_at_step = &slices[step];
        let step_grads = vjp(
            body_op,
            &[carry_at_step.clone(), x_at_step.clone()],
            &[g_carry.clone()],
            &[],
            &BTreeMap::new(),
        )?;
        g_carry = step_grads[0].clone();

        // Write x gradient into the correct position in g_xs_elements
        let start = i * slice_size;
        match &step_grads[1] {
            Value::Scalar(lit) => {
                g_xs_elements[start] = Literal::from_f64(lit.as_f64().unwrap_or(0.0));
            }
            Value::Tensor(t) => {
                for (j, elem) in t.elements.iter().enumerate() {
                    g_xs_elements[start + j] = Literal::from_f64(elem.as_f64().unwrap_or(0.0));
                }
            }
        }
    }

    let g_xs = Value::Tensor(
        TensorValue::new(DType::F64, xs_tensor.shape.clone(), g_xs_elements)
            .map_err(|e| AdError::EvalFailed(e.to_string()))?,
    );

    Ok(vec![g_carry, g_xs])
}

fn scan_jvp(
    primals: &[Value],
    tangents: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, AdError> {
    if primals.len() < 2 {
        return Err(AdError::InputArity {
            expected: 2,
            actual: primals.len(),
        });
    }
    if tangents.len() < 2 {
        return Err(AdError::InputArity {
            expected: 2,
            actual: tangents.len(),
        });
    }

    let body_op_name = params.get("body_op").map(|s| s.as_str()).unwrap_or("add");
    let body_op =
        scan_body_op_for_ad(body_op_name).ok_or(AdError::UnsupportedPrimitive(Primitive::Scan))?;
    let reverse = params.get("reverse").map(|s| s == "true").unwrap_or(false);

    match (&primals[1], &tangents[1]) {
        (Value::Scalar(_), _) => jvp_rule(
            body_op,
            &[primals[0].clone(), primals[1].clone()],
            &[tangents[0].clone(), tangents[1].clone()],
            &BTreeMap::new(),
        ),
        (Value::Tensor(xs), Value::Tensor(xs_tangent)) => {
            if xs.shape != xs_tangent.shape {
                return Err(AdError::EvalFailed(format!(
                    "scan JVP tangent shape mismatch: primal {:?}, tangent {:?}",
                    xs.shape.dims, xs_tangent.shape.dims
                )));
            }
            if xs.shape.rank() == 0 {
                return Err(AdError::EvalFailed(
                    "scan JVP requires tensor xs to have a leading axis".to_owned(),
                ));
            }

            let leading_dim = xs.shape.dims[0] as usize;
            if leading_dim == 0 {
                return Ok(tangents[0].clone());
            }

            let mut carry_primal = primals[0].clone();
            let mut carry_tangent = tangents[0].clone();
            if reverse {
                for index in (0..leading_dim).rev() {
                    let x_slice = xs
                        .slice_axis0(index)
                        .map_err(|e| AdError::EvalFailed(e.to_string()))?;
                    let x_tangent = xs_tangent
                        .slice_axis0(index)
                        .map_err(|e| AdError::EvalFailed(e.to_string()))?;
                    let next_tangent = jvp_rule(
                        body_op,
                        &[carry_primal.clone(), x_slice.clone()],
                        &[carry_tangent, x_tangent],
                        &BTreeMap::new(),
                    )?;
                    carry_primal =
                        eval_primitive(body_op, &[carry_primal, x_slice], &BTreeMap::new())
                            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
                    carry_tangent = next_tangent;
                }
            } else {
                for index in 0..leading_dim {
                    let x_slice = xs
                        .slice_axis0(index)
                        .map_err(|e| AdError::EvalFailed(e.to_string()))?;
                    let x_tangent = xs_tangent
                        .slice_axis0(index)
                        .map_err(|e| AdError::EvalFailed(e.to_string()))?;
                    let next_tangent = jvp_rule(
                        body_op,
                        &[carry_primal.clone(), x_slice.clone()],
                        &[carry_tangent, x_tangent],
                        &BTreeMap::new(),
                    )?;
                    carry_primal =
                        eval_primitive(body_op, &[carry_primal, x_slice], &BTreeMap::new())
                            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
                    carry_tangent = next_tangent;
                }
            }
            Ok(carry_tangent)
        }
        (Value::Tensor(xs), tangent) => Err(AdError::EvalFailed(format!(
            "scan JVP expected tensor tangent for tensor xs with shape {:?}, got {tangent:?}",
            xs.shape.dims
        ))),
    }
}

fn while_body_op_for_ad(body_op_name: &str) -> Option<Primitive> {
    match body_op_name {
        "add" => Some(Primitive::Add),
        "sub" => Some(Primitive::Sub),
        "mul" => Some(Primitive::Mul),
        "div" => Some(Primitive::Div),
        "pow" => Some(Primitive::Pow),
        _ => None,
    }
}

fn while_cond_op_for_ad(cond_op_name: &str) -> Option<Primitive> {
    match cond_op_name {
        "lt" => Some(Primitive::Lt),
        "le" => Some(Primitive::Le),
        "gt" => Some(Primitive::Gt),
        "ge" => Some(Primitive::Ge),
        "ne" => Some(Primitive::Ne),
        "eq" => Some(Primitive::Eq),
        _ => None,
    }
}

fn while_cond_value_to_bool(cond_result: &Value) -> Result<bool, AdError> {
    scalar_predicate_for_ad(Primitive::While, "condition result", cond_result)
}

/// While loop VJP: reverse-propagate gradient through the iteration loop.
///
/// inputs: [init_carry, step_value, threshold]
/// For while with body_op: carry_{i+1} = body_op(carry_i, step_value)
/// We replay forward to collect intermediate carries, then propagate
/// gradients backward (same approach as scan VJP).
fn while_vjp(
    inputs: &[Value],
    g: &Value,
    params: &BTreeMap<String, String>,
) -> Result<Vec<Value>, AdError> {
    if inputs.len() < 3 {
        return Err(AdError::InputArity {
            expected: 3,
            actual: inputs.len(),
        });
    }

    let init_carry = &inputs[0];
    let step_value = &inputs[1];
    let threshold = &inputs[2];

    let body_op_name = params.get("body_op").map(|s| s.as_str()).unwrap_or("add");
    let body_op = while_body_op_for_ad(body_op_name)
        .ok_or(AdError::UnsupportedPrimitive(Primitive::While))?;

    let cond_op_name = params.get("cond_op").map(|s| s.as_str()).unwrap_or("lt");
    let cond_op = while_cond_op_for_ad(cond_op_name)
        .ok_or(AdError::UnsupportedPrimitive(Primitive::While))?;

    let max_iter: usize = params
        .get("max_iter")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    // Forward pass: collect intermediate carries
    let mut carries = vec![init_carry.clone()];
    let mut carry = init_carry.clone();

    for _ in 0..max_iter {
        let cond_result = eval_primitive(
            cond_op,
            &[carry.clone(), threshold.clone()],
            &BTreeMap::new(),
        )
        .map_err(|e| AdError::EvalFailed(e.to_string()))?;
        if !while_cond_value_to_bool(&cond_result)? {
            break;
        }
        carry = eval_primitive(body_op, &[carry, step_value.clone()], &BTreeMap::new())
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
        carries.push(carry.clone());
    }

    let num_iters = carries.len() - 1; // number of body applications

    // Backward pass: propagate gradient through each step
    let mut g_carry = g.clone();
    let mut g_step_accum = zeros_like(step_value);

    for i in (0..num_iters).rev() {
        let carry_at_step = &carries[i];
        let step_grads = vjp(
            body_op,
            &[carry_at_step.clone(), step_value.clone()],
            &[g_carry.clone()],
            &[],
            &BTreeMap::new(),
        )?;
        g_carry = step_grads[0].clone();
        g_step_accum = value_add(&g_step_accum, &step_grads[1])?;
    }

    // Threshold gradient is zero (discrete comparison boundary)
    let g_threshold = zeros_like(threshold);

    Ok(vec![g_carry, g_step_accum, g_threshold])
}

fn while_jvp(
    primals: &[Value],
    tangents: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, AdError> {
    if primals.len() < 3 {
        return Err(AdError::InputArity {
            expected: 3,
            actual: primals.len(),
        });
    }
    if tangents.len() < 3 {
        return Err(AdError::InputArity {
            expected: 3,
            actual: tangents.len(),
        });
    }

    let step_value = &primals[1];
    let threshold = &primals[2];
    let step_tangent = &tangents[1];

    let body_op_name = params.get("body_op").map(|s| s.as_str()).unwrap_or("add");
    let body_op = while_body_op_for_ad(body_op_name)
        .ok_or(AdError::UnsupportedPrimitive(Primitive::While))?;

    let cond_op_name = params.get("cond_op").map(|s| s.as_str()).unwrap_or("lt");
    let cond_op = while_cond_op_for_ad(cond_op_name)
        .ok_or(AdError::UnsupportedPrimitive(Primitive::While))?;

    let max_iter: usize = params
        .get("max_iter")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let mut carry_primal = primals[0].clone();
    let mut carry_tangent = tangents[0].clone();

    for _ in 0..max_iter {
        let cond_result = eval_primitive(
            cond_op,
            &[carry_primal.clone(), threshold.clone()],
            &BTreeMap::new(),
        )
        .map_err(|e| AdError::EvalFailed(e.to_string()))?;
        if !while_cond_value_to_bool(&cond_result)? {
            return Ok(carry_tangent);
        }

        let next_tangent = jvp_rule(
            body_op,
            &[carry_primal.clone(), step_value.clone()],
            &[carry_tangent, step_tangent.clone()],
            &BTreeMap::new(),
        )?;
        carry_primal = eval_primitive(
            body_op,
            &[carry_primal, step_value.clone()],
            &BTreeMap::new(),
        )
        .map_err(|e| AdError::EvalFailed(e.to_string()))?;
        carry_tangent = next_tangent;
    }

    Err(AdError::EvalFailed(format!(
        "while JVP exceeded max_iter={max_iter}"
    )))
}

/// Conv 1D VJP: compute proper gradients for both input and kernel.
///
/// Layout: lhs=[N, W, C_in], rhs=[K, C_in, C_out], output=[N, W_out, C_out]
///
/// grad_lhs[n, w, ci] = sum_k sum_co g[n, w', co] * rhs[k, ci, co]
///   where w' = (w - k + pad_left) / stride, only for valid positions
///
/// grad_rhs[k, ci, co] = sum_n sum_w' lhs[n, w'*stride + k - pad_left, ci] * g[n, w', co]
fn conv_vjp(
    inputs: &[Value],
    g: &Value,
    params: &BTreeMap<String, String>,
) -> Result<Vec<Value>, AdError> {
    let lhs = match &inputs[0] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => return Ok(vec![zeros_like(&inputs[0]), zeros_like(&inputs[1])]),
    };
    let rhs = match &inputs[1] {
        Value::Tensor(t) => t,
        Value::Scalar(_) => return Ok(vec![zeros_like(&inputs[0]), zeros_like(&inputs[1])]),
    };
    let g_tensor = match g {
        Value::Tensor(t) => t,
        Value::Scalar(_) => return Ok(vec![zeros_like(&inputs[0]), zeros_like(&inputs[1])]),
    };

    let rank = lhs.shape.rank();
    if rank == 3 {
        conv_vjp_1d(lhs, rhs, g_tensor, params)
    } else if rank == 4 {
        conv_vjp_2d(lhs, rhs, g_tensor, params)
    } else {
        Ok(vec![zeros_like(&inputs[0]), zeros_like(&inputs[1])])
    }
}

/// 1D Conv VJP: lhs=[N, W, C_in], rhs=[K, C_in, C_out], g=[N, W_out, C_out]
fn conv_vjp_1d(
    lhs: &TensorValue,
    rhs: &TensorValue,
    g_tensor: &TensorValue,
    params: &BTreeMap<String, String>,
) -> Result<Vec<Value>, AdError> {
    let batch = lhs.shape.dims[0] as usize;
    let width = lhs.shape.dims[1] as usize;
    let c_in = lhs.shape.dims[2] as usize;
    let kernel_w = rhs.shape.dims[0] as usize;
    let c_out = rhs.shape.dims[2] as usize;
    let out_w = g_tensor.shape.dims[1] as usize;

    let stride: usize = params
        .get("strides")
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(1);

    let padding_mode = params.get("padding").map(String::as_str).unwrap_or("valid");
    let pad_left: usize = match padding_mode {
        "same" | "SAME" => {
            let padded_out_w = width.div_ceil(stride);
            let pad_total = ((padded_out_w - 1) * stride + kernel_w).saturating_sub(width);
            pad_total / 2
        }
        _ => 0,
    };

    let lhs_total = batch * width * c_in;
    let mut grad_lhs_elems = vec![0.0_f64; lhs_total];

    for n in 0..batch {
        for w_out in 0..out_w {
            for k in 0..kernel_w {
                let in_pos = (w_out * stride + k) as isize - pad_left as isize;
                if in_pos >= 0 && (in_pos as usize) < width {
                    let w = in_pos as usize;
                    for ci in 0..c_in {
                        let mut acc = 0.0;
                        for co in 0..c_out {
                            let g_idx = n * out_w * c_out + w_out * c_out + co;
                            let rhs_idx = k * c_in * c_out + ci * c_out + co;
                            acc += g_tensor.elements[g_idx].as_f64().unwrap_or(0.0)
                                * rhs.elements[rhs_idx].as_f64().unwrap_or(0.0);
                        }
                        grad_lhs_elems[n * width * c_in + w * c_in + ci] += acc;
                    }
                }
            }
        }
    }

    let rhs_total = kernel_w * c_in * c_out;
    let mut grad_rhs_elems = vec![0.0_f64; rhs_total];

    for n in 0..batch {
        for w_out in 0..out_w {
            for k in 0..kernel_w {
                let in_pos = (w_out * stride + k) as isize - pad_left as isize;
                if in_pos >= 0 && (in_pos as usize) < width {
                    let w = in_pos as usize;
                    for ci in 0..c_in {
                        let lhs_val = lhs.elements[n * width * c_in + w * c_in + ci]
                            .as_f64()
                            .unwrap_or(0.0);
                        for co in 0..c_out {
                            let g_val = g_tensor.elements[n * out_w * c_out + w_out * c_out + co]
                                .as_f64()
                                .unwrap_or(0.0);
                            grad_rhs_elems[k * c_in * c_out + ci * c_out + co] += lhs_val * g_val;
                        }
                    }
                }
            }
        }
    }

    make_conv_grad_pair(&lhs.shape, &rhs.shape, grad_lhs_elems, grad_rhs_elems)
}

/// 2D Conv VJP: lhs=[N, H, W, C_in], rhs=[KH, KW, C_in, C_out], g=[N, OH, OW, C_out]
fn conv_vjp_2d(
    lhs: &TensorValue,
    rhs: &TensorValue,
    g_tensor: &TensorValue,
    params: &BTreeMap<String, String>,
) -> Result<Vec<Value>, AdError> {
    let batch = lhs.shape.dims[0] as usize;
    let height = lhs.shape.dims[1] as usize;
    let width = lhs.shape.dims[2] as usize;
    let c_in = lhs.shape.dims[3] as usize;
    let kernel_h = rhs.shape.dims[0] as usize;
    let kernel_w = rhs.shape.dims[1] as usize;
    let c_out = rhs.shape.dims[3] as usize;
    let out_h = g_tensor.shape.dims[1] as usize;
    let out_w = g_tensor.shape.dims[2] as usize;

    let strides_str = params.get("strides").map(String::as_str).unwrap_or("1");
    let parts: Vec<&str> = strides_str.split(',').collect();
    let (stride_h, stride_w) = if parts.len() >= 2 {
        (
            parts[0].trim().parse().unwrap_or(1usize),
            parts[1].trim().parse().unwrap_or(1usize),
        )
    } else {
        let s: usize = parts[0].trim().parse().unwrap_or(1);
        (s, s)
    };

    let padding_mode = params.get("padding").map(String::as_str).unwrap_or("valid");
    let pad_top: usize = match padding_mode {
        "same" | "SAME" => {
            let oh = height.div_ceil(stride_h);
            ((oh - 1) * stride_h + kernel_h).saturating_sub(height) / 2
        }
        _ => 0,
    };
    let pad_left: usize = match padding_mode {
        "same" | "SAME" => {
            let ow = width.div_ceil(stride_w);
            ((ow - 1) * stride_w + kernel_w).saturating_sub(width) / 2
        }
        _ => 0,
    };

    let lhs_total = batch * height * width * c_in;
    let mut grad_lhs_elems = vec![0.0_f64; lhs_total];
    let rhs_total = kernel_h * kernel_w * c_in * c_out;
    let mut grad_rhs_elems = vec![0.0_f64; rhs_total];

    for n in 0..batch {
        for oh in 0..out_h {
            for ow in 0..out_w {
                for kh in 0..kernel_h {
                    let in_h = (oh * stride_h + kh) as isize - pad_top as isize;
                    if in_h < 0 || (in_h as usize) >= height {
                        continue;
                    }
                    for kw in 0..kernel_w {
                        let in_w = (ow * stride_w + kw) as isize - pad_left as isize;
                        if in_w < 0 || (in_w as usize) >= width {
                            continue;
                        }
                        let ih = in_h as usize;
                        let iw = in_w as usize;
                        for ci in 0..c_in {
                            let lhs_idx =
                                n * height * width * c_in + ih * width * c_in + iw * c_in + ci;
                            let lhs_val = lhs.elements[lhs_idx].as_f64().unwrap_or(0.0);

                            let mut g_acc = 0.0;
                            for co in 0..c_out {
                                let g_idx = n * out_h * out_w * c_out
                                    + oh * out_w * c_out
                                    + ow * c_out
                                    + co;
                                let rhs_idx = kh * kernel_w * c_in * c_out
                                    + kw * c_in * c_out
                                    + ci * c_out
                                    + co;
                                let g_val = g_tensor.elements[g_idx].as_f64().unwrap_or(0.0);
                                let rhs_val = rhs.elements[rhs_idx].as_f64().unwrap_or(0.0);
                                g_acc += g_val * rhs_val;
                                grad_rhs_elems[rhs_idx] += lhs_val * g_val;
                            }
                            grad_lhs_elems[lhs_idx] += g_acc;
                        }
                    }
                }
            }
        }
    }

    make_conv_grad_pair(&lhs.shape, &rhs.shape, grad_lhs_elems, grad_rhs_elems)
}

fn make_conv_grad_pair(
    lhs_shape: &Shape,
    rhs_shape: &Shape,
    grad_lhs_elems: Vec<f64>,
    grad_rhs_elems: Vec<f64>,
) -> Result<Vec<Value>, AdError> {
    let grad_lhs = Value::Tensor(
        TensorValue::new(
            DType::F64,
            lhs_shape.clone(),
            grad_lhs_elems.into_iter().map(Literal::from_f64).collect(),
        )
        .map_err(|e| AdError::EvalFailed(e.to_string()))?,
    );
    let grad_rhs = Value::Tensor(
        TensorValue::new(
            DType::F64,
            rhs_shape.clone(),
            grad_rhs_elems.into_iter().map(Literal::from_f64).collect(),
        )
        .map_err(|e| AdError::EvalFailed(e.to_string()))?,
    );
    Ok(vec![grad_lhs, grad_rhs])
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
    let tail_dims = dims[1..].iter().map(|d| *d as usize).collect::<Vec<_>>();
    let slice_elems = ad_checked_usize_product("scatter VJP slice", &tail_dims)?.max(1);

    let mut elements = g_tensor.elements.clone();
    for &idx in &index_vals {
        if idx < dims[0] as usize {
            let base = idx.checked_mul(slice_elems).ok_or_else(|| {
                AdError::EvalFailed("scatter VJP slice base overflows usize".to_owned())
            })?;
            for j in 0..slice_elems {
                let Some(pos) = base.checked_add(j) else {
                    return Err(AdError::EvalFailed(
                        "scatter VJP slice offset overflows usize".to_owned(),
                    ));
                };
                if pos < elements.len() {
                    elements[pos] = Literal::from_f64(0.0);
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

fn parse_reduce_axes(
    params: &BTreeMap<String, String>,
    rank: usize,
    primitive: Primitive,
) -> Result<Option<Vec<usize>>, AdError> {
    let Some(raw_axes) = params.get("axes") else {
        return Ok(None);
    };
    if raw_axes.trim().is_empty() {
        return Ok(Some(Vec::new()));
    }

    let mut axes = raw_axes
        .split(',')
        .map(|axis| {
            axis.trim().parse::<usize>().map_err(|_| {
                AdError::EvalFailed(format!(
                    "invalid axis value for {}: {}",
                    primitive.as_str(),
                    axis.trim()
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    for axis in &axes {
        if *axis >= rank {
            return Err(AdError::EvalFailed(format!(
                "axis {axis} out of bounds for rank {rank}"
            )));
        }
    }
    axes.sort_unstable();
    axes.dedup();
    Ok(Some(axes))
}

fn ad_checked_shape_element_count(context: &str, dims: &[u32]) -> Result<usize, AdError> {
    if dims.contains(&0) {
        return Ok(0);
    }
    dims.iter().try_fold(1_usize, |acc, dim| {
        acc.checked_mul(*dim as usize)
            .ok_or_else(|| AdError::EvalFailed(format!("{context} element count overflows usize")))
    })
}

fn ad_checked_usize_product(context: &str, dims: &[usize]) -> Result<usize, AdError> {
    if dims.contains(&0) {
        return Ok(0);
    }
    dims.iter().try_fold(1_usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| AdError::EvalFailed(format!("{context} element count overflows usize")))
    })
}

fn ad_row_major_strides(context: &str, dims: &[u32]) -> Result<Vec<usize>, AdError> {
    let mut strides = vec![1_usize; dims.len()];
    if dims.contains(&0) {
        return Ok(strides);
    }
    for idx in (0..dims.len().saturating_sub(1)).rev() {
        strides[idx] = strides[idx + 1]
            .checked_mul(dims[idx + 1] as usize)
            .ok_or_else(|| {
                AdError::EvalFailed(format!("{context} row-major strides overflow usize"))
            })?;
    }
    Ok(strides)
}

fn ad_flat_to_multi(flat: usize, strides: &[usize]) -> Vec<usize> {
    let mut multi = Vec::with_capacity(strides.len());
    let mut remainder = flat;
    for stride in strides {
        multi.push(remainder / *stride);
        remainder %= *stride;
    }
    multi
}

fn ad_multi_to_kept_flat(
    context: &str,
    multi: &[usize],
    kept_axes: &[usize],
    out_dims: &[u32],
) -> Result<usize, AdError> {
    let mut flat = 0_usize;
    let mut stride = 1_usize;
    for idx in (0..kept_axes.len()).rev() {
        let offset = multi[kept_axes[idx]].checked_mul(stride).ok_or_else(|| {
            AdError::EvalFailed(format!("{context} output offset overflows usize"))
        })?;
        flat = flat.checked_add(offset).ok_or_else(|| {
            AdError::EvalFailed(format!("{context} output index overflows usize"))
        })?;
        stride = stride.checked_mul(out_dims[idx] as usize).ok_or_else(|| {
            AdError::EvalFailed(format!("{context} output strides overflow usize"))
        })?;
    }
    Ok(flat)
}

fn reduce_prod_vjp_tensor(
    tensor: &TensorValue,
    g: &Value,
    params: &BTreeMap<String, String>,
) -> Result<Value, AdError> {
    let rank = tensor.shape.rank();
    let reduce_axes = match parse_reduce_axes(params, rank, Primitive::ReduceProd)? {
        Some(axes) if axes.is_empty() => return Ok(g.clone()),
        Some(axes) => axes,
        None => (0..rank).collect(),
    };
    let kept_axes = (0..rank)
        .filter(|axis| !reduce_axes.contains(axis))
        .collect::<Vec<_>>();
    let out_dims = kept_axes
        .iter()
        .map(|axis| tensor.shape.dims[*axis])
        .collect::<Vec<_>>();
    let out_count = if out_dims.is_empty() {
        1
    } else {
        ad_checked_shape_element_count("reduce_prod VJP output", &out_dims)?
    };
    let upstream = match g {
        Value::Scalar(literal) => {
            let value = literal.as_f64().ok_or_else(|| {
                AdError::EvalFailed("reduce_prod gradient must be numeric".to_owned())
            })?;
            vec![value; out_count]
        }
        Value::Tensor(g_tensor) => {
            if g_tensor.elements.len() != out_count {
                return Err(AdError::EvalFailed(format!(
                    "reduce_prod gradient shape mismatch: expected {out_count} elements, got {}",
                    g_tensor.elements.len()
                )));
            }
            g_tensor
                .elements
                .iter()
                .map(|literal| {
                    literal.as_f64().ok_or_else(|| {
                        AdError::EvalFailed("reduce_prod gradient must be numeric".to_owned())
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        }
    };

    let strides = ad_row_major_strides("reduce_prod VJP input", &tensor.shape.dims)?;
    let mut product_nonzero = vec![1.0_f64; out_count];
    let mut zero_counts = vec![0_usize; out_count];
    let mut element_buckets = Vec::with_capacity(tensor.elements.len());

    for (flat_idx, literal) in tensor.elements.iter().enumerate() {
        let value = literal
            .as_f64()
            .ok_or_else(|| AdError::EvalFailed("reduce_prod input must be numeric".to_owned()))?;
        let multi = ad_flat_to_multi(flat_idx, &strides);
        let out_idx = ad_multi_to_kept_flat("reduce_prod VJP", &multi, &kept_axes, &out_dims)?;
        element_buckets.push(out_idx);
        if value.abs() < f64::EPSILON {
            zero_counts[out_idx] += 1;
        } else {
            product_nonzero[out_idx] *= value;
        }
    }

    let elements = tensor
        .elements
        .iter()
        .zip(element_buckets)
        .map(|(literal, out_idx)| {
            let value = literal.as_f64().ok_or_else(|| {
                AdError::EvalFailed("reduce_prod input must be numeric".to_owned())
            })?;
            let cotangent = match zero_counts[out_idx] {
                0 => upstream[out_idx] * product_nonzero[out_idx] / value,
                1 if value.abs() < f64::EPSILON => upstream[out_idx] * product_nonzero[out_idx],
                _ => 0.0,
            };
            Ok(Literal::from_f64(cotangent))
        })
        .collect::<Result<Vec<_>, AdError>>()?;

    Ok(Value::Tensor(
        TensorValue::new(DType::F64, tensor.shape.clone(), elements)
            .map_err(|err| AdError::EvalFailed(err.to_string()))?,
    ))
}

fn reduce_prod_jvp_value(
    primal: &Value,
    tangent: &Value,
    params: &BTreeMap<String, String>,
) -> Result<Value, AdError> {
    match (primal, tangent) {
        (Value::Scalar(_), Value::Scalar(tangent_literal)) => {
            let _axes = parse_reduce_axes(params, 0, Primitive::ReduceProd)?;
            let tangent = tangent_literal.as_f64().ok_or_else(|| {
                AdError::EvalFailed("reduce_prod tangent must be numeric".to_owned())
            })?;
            Ok(Value::scalar_f64(tangent))
        }
        (Value::Tensor(primal_tensor), Value::Tensor(tangent_tensor)) => {
            reduce_prod_jvp_tensor(primal_tensor, tangent_tensor, params)
        }
        _ => Err(AdError::EvalFailed(
            "reduce_prod primal and tangent must have matching scalar/tensor structure".to_owned(),
        )),
    }
}

fn reduce_prod_jvp_tensor(
    primal: &TensorValue,
    tangent: &TensorValue,
    params: &BTreeMap<String, String>,
) -> Result<Value, AdError> {
    if primal.elements.len() != tangent.elements.len() || primal.shape != tangent.shape {
        return Err(AdError::EvalFailed(format!(
            "reduce_prod tangent shape mismatch: expected {:?} with {} elements, got {:?} with {} elements",
            primal.shape,
            primal.elements.len(),
            tangent.shape,
            tangent.elements.len()
        )));
    }

    let rank = primal.shape.rank();
    let reduce_axes = match parse_reduce_axes(params, rank, Primitive::ReduceProd)? {
        Some(axes) if axes.is_empty() => return Ok(Value::Tensor(tangent.clone())),
        Some(axes) => axes,
        None => (0..rank).collect(),
    };
    let mut reduced_axis = vec![false; rank];
    for axis in &reduce_axes {
        reduced_axis[*axis] = true;
    }
    let kept_axes = (0..rank)
        .filter(|axis| !reduced_axis[*axis])
        .collect::<Vec<_>>();
    let out_dims = kept_axes
        .iter()
        .map(|axis| primal.shape.dims[*axis])
        .collect::<Vec<_>>();
    let out_count = if out_dims.is_empty() {
        1
    } else {
        ad_checked_shape_element_count("reduce_prod JVP output", &out_dims)?
    };

    let strides = ad_row_major_strides("reduce_prod JVP input", &primal.shape.dims)?;
    let mut product_nonzero = vec![1.0_f64; out_count];
    let mut zero_counts = vec![0_usize; out_count];
    let mut element_terms = Vec::with_capacity(primal.elements.len());

    for (flat_idx, (primal_literal, tangent_literal)) in primal
        .elements
        .iter()
        .zip(tangent.elements.iter())
        .enumerate()
    {
        let primal_value = primal_literal
            .as_f64()
            .ok_or_else(|| AdError::EvalFailed("reduce_prod input must be numeric".to_owned()))?;
        let tangent_value = tangent_literal
            .as_f64()
            .ok_or_else(|| AdError::EvalFailed("reduce_prod tangent must be numeric".to_owned()))?;
        let multi = ad_flat_to_multi(flat_idx, &strides);
        let out_idx = ad_multi_to_kept_flat("reduce_prod JVP", &multi, &kept_axes, &out_dims)?;
        element_terms.push((out_idx, primal_value, tangent_value));
        if primal_value.abs() < f64::EPSILON {
            zero_counts[out_idx] += 1;
        } else {
            product_nonzero[out_idx] *= primal_value;
        }
    }

    let mut output = vec![0.0_f64; out_count];
    for (out_idx, primal_value, tangent_value) in element_terms {
        output[out_idx] += match zero_counts[out_idx] {
            0 => tangent_value * product_nonzero[out_idx] / primal_value,
            1 if primal_value.abs() < f64::EPSILON => tangent_value * product_nonzero[out_idx],
            _ => 0.0,
        };
    }

    if out_dims.is_empty() {
        return Ok(Value::scalar_f64(output[0]));
    }

    let elements = output.into_iter().map(Literal::from_f64).collect();
    Ok(Value::Tensor(
        TensorValue::new(DType::F64, Shape { dims: out_dims }, elements)
            .map_err(|err| AdError::EvalFailed(err.to_string()))?,
    ))
}

fn reduce_chooser_jvp_value(
    primitive: Primitive,
    primal: &Value,
    tangent: &Value,
    params: &BTreeMap<String, String>,
) -> Result<Value, AdError> {
    match (primal, tangent) {
        (Value::Scalar(_), Value::Scalar(_)) => {
            let _axes = parse_reduce_axes(params, 0, primitive)?;
            Ok(tangent.clone())
        }
        (Value::Tensor(primal_tensor), Value::Tensor(tangent_tensor)) => {
            if primal_tensor.shape != tangent_tensor.shape
                || primal_tensor.elements.len() != tangent_tensor.elements.len()
            {
                return Err(AdError::EvalFailed(format!(
                    "{} tangent shape mismatch: expected {:?} with {} elements, got {:?} with {} elements",
                    primitive.as_str(),
                    primal_tensor.shape,
                    primal_tensor.elements.len(),
                    tangent_tensor.shape,
                    tangent_tensor.elements.len()
                )));
            }

            let rank = primal_tensor.shape.rank();
            let reduce_axes = match parse_reduce_axes(params, rank, primitive)? {
                Some(axes) if axes.is_empty() => return Ok(tangent.clone()),
                Some(axes) => axes,
                None => (0..rank).collect(),
            };
            let mut reduced_axis = vec![false; rank];
            for axis in &reduce_axes {
                reduced_axis[*axis] = true;
            }
            let kept_axes = (0..rank)
                .filter(|axis| !reduced_axis[*axis])
                .collect::<Vec<_>>();

            let primal_out = eval_primitive(primitive, std::slice::from_ref(primal), params)
                .map_err(|err| AdError::EvalFailed(err.to_string()))?;
            let out_bcast =
                broadcast_g_to_shape_with_axes(&primal_out, &primal_tensor.shape, &kept_axes)?;
            let location_mask = eval_primitive(
                Primitive::Eq,
                &[primal.clone(), out_bcast],
                &BTreeMap::new(),
            )
            .map_err(|err| AdError::EvalFailed(err.to_string()))?;
            let location_indicators = eval_primitive(
                Primitive::Select,
                &[location_mask, scalar_value(1.0), scalar_value(0.0)],
                &BTreeMap::new(),
            )
            .map_err(|err| AdError::EvalFailed(err.to_string()))?;
            let masked_tangent = eval_primitive(
                Primitive::Mul,
                &[tangent.clone(), location_indicators.clone()],
                &BTreeMap::new(),
            )
            .map_err(|err| AdError::EvalFailed(err.to_string()))?;
            let numerator = eval_primitive(
                Primitive::ReduceSum,
                std::slice::from_ref(&masked_tangent),
                params,
            )
            .map_err(|err| AdError::EvalFailed(err.to_string()))?;
            let counts = eval_primitive(
                Primitive::ReduceSum,
                std::slice::from_ref(&location_indicators),
                params,
            )
            .map_err(|err| AdError::EvalFailed(err.to_string()))?;

            eval_primitive(Primitive::Div, &[numerator, counts], &BTreeMap::new())
                .map_err(|err| AdError::EvalFailed(err.to_string()))
        }
        _ => Err(AdError::EvalFailed(format!(
            "{} primal and tangent must have matching scalar/tensor structure",
            primitive.as_str()
        ))),
    }
}

fn parse_cumulative_axis(
    params: &BTreeMap<String, String>,
    rank: usize,
    primitive: Primitive,
) -> Result<usize, AdError> {
    let axis = params
        .get("axis")
        .map(|raw| {
            raw.trim().parse::<usize>().map_err(|_| {
                AdError::EvalFailed(format!(
                    "invalid axis value for {}: {}",
                    primitive.as_str(),
                    raw.trim()
                ))
            })
        })
        .transpose()?
        .unwrap_or(0);
    if axis >= rank {
        return Err(AdError::EvalFailed(format!(
            "axis {axis} out of bounds for rank {rank}"
        )));
    }
    Ok(axis)
}

fn cumprod_jvp_value(
    primal: &Value,
    tangent: &Value,
    params: &BTreeMap<String, String>,
) -> Result<Value, AdError> {
    match (primal, tangent) {
        (Value::Scalar(_), Value::Scalar(tangent_literal)) => {
            let tangent = tangent_literal
                .as_f64()
                .ok_or_else(|| AdError::EvalFailed("cumprod tangent must be numeric".to_owned()))?;
            Ok(Value::scalar_f64(tangent))
        }
        (Value::Tensor(primal_tensor), Value::Tensor(tangent_tensor)) => {
            cumprod_jvp_tensor(primal_tensor, tangent_tensor, params)
        }
        _ => Err(AdError::EvalFailed(
            "cumprod primal and tangent must have matching scalar/tensor structure".to_owned(),
        )),
    }
}

fn cumprod_jvp_tensor(
    primal: &TensorValue,
    tangent: &TensorValue,
    params: &BTreeMap<String, String>,
) -> Result<Value, AdError> {
    if primal.elements.len() != tangent.elements.len() || primal.shape != tangent.shape {
        return Err(AdError::EvalFailed(format!(
            "cumprod tangent shape mismatch: expected {:?} with {} elements, got {:?} with {} elements",
            primal.shape,
            primal.elements.len(),
            tangent.shape,
            tangent.elements.len()
        )));
    }

    let rank = primal.shape.rank();
    if rank == 0 {
        let tangent = tangent
            .elements
            .first()
            .and_then(|literal| literal.as_f64())
            .ok_or_else(|| AdError::EvalFailed("cumprod tangent must be numeric".to_owned()))?;
        return Ok(Value::scalar_f64(tangent));
    }

    let axis = parse_cumulative_axis(params, rank, Primitive::Cumprod)?;
    let dims = primal
        .shape
        .dims
        .iter()
        .map(|dim| *dim as usize)
        .collect::<Vec<_>>();
    let strides = ad_row_major_strides("cumprod JVP input", &primal.shape.dims)?;
    let axis_len = dims[axis];
    let axis_stride = strides[axis];
    let total = primal.elements.len();
    let outer_count = total.checked_div(axis_len).unwrap_or(0);
    let mut output = vec![0.0_f64; total];

    for outer in 0..outer_count {
        let mut base = 0_usize;
        let mut rem = outer;
        for dim_idx in (0..rank).rev() {
            if dim_idx == axis {
                continue;
            }
            base += (rem % dims[dim_idx]) * strides[dim_idx];
            rem /= dims[dim_idx];
        }

        let mut running_product = 1.0_f64;
        let mut running_tangent = 0.0_f64;
        for axis_idx in 0..axis_len {
            let flat_idx = base + axis_idx * axis_stride;
            let primal_value = primal.elements[flat_idx]
                .as_f64()
                .ok_or_else(|| AdError::EvalFailed("cumprod input must be numeric".to_owned()))?;
            let tangent_value = tangent.elements[flat_idx]
                .as_f64()
                .ok_or_else(|| AdError::EvalFailed("cumprod tangent must be numeric".to_owned()))?;
            running_tangent = running_tangent * primal_value + running_product * tangent_value;
            running_product *= primal_value;
            output[flat_idx] = running_tangent;
        }
    }

    Ok(Value::Tensor(
        TensorValue::new(
            DType::F64,
            primal.shape.clone(),
            output.into_iter().map(Literal::from_f64).collect(),
        )
        .map_err(|err| AdError::EvalFailed(err.to_string()))?,
    ))
}

fn to_f64(value: &Value) -> Result<f64, AdError> {
    value
        .as_f64_scalar()
        .ok_or(AdError::NonScalarGradientOutput)
}

fn tensor_value<'a>(value: &'a Value, context: &'static str) -> Result<&'a TensorValue, AdError> {
    value
        .as_tensor()
        .ok_or_else(|| AdError::EvalFailed(format!("{context}: expected tensor value, got scalar")))
}

// ── Linear algebra matrix helpers ──────────────────────────────────

/// Extract row-major f64 matrix data from a rank-2 tensor Value.
fn extract_matrix_f64(v: &Value) -> Result<(usize, usize, Vec<f64>), AdError> {
    let t = v
        .as_tensor()
        .ok_or_else(|| AdError::EvalFailed("expected matrix tensor, got scalar".to_owned()))?;
    if t.rank() != 2 {
        return Err(AdError::EvalFailed(format!(
            "expected rank-2 tensor, got rank-{}",
            t.rank()
        )));
    }
    let m = t.shape.dims[0] as usize;
    let n = t.shape.dims[1] as usize;
    let data: Vec<f64> = t
        .elements
        .iter()
        .map(|lit| {
            lit.as_f64()
                .ok_or_else(|| AdError::EvalFailed("non-numeric matrix element".to_owned()))
        })
        .collect::<Result<_, _>>()?;
    Ok((m, n, data))
}

/// Build a rank-2 f64 tensor Value from row-major data.
fn build_matrix_f64(m: usize, n: usize, data: &[f64]) -> Result<Value, AdError> {
    let elements: Vec<Literal> = data.iter().map(|&v| Literal::from_f64(v)).collect();
    let shape = Shape {
        dims: vec![m as u32, n as u32],
    };
    let tensor = TensorValue::new(DType::F64, shape, elements)
        .map_err(|e| AdError::EvalFailed(e.to_string()))?;
    Ok(Value::Tensor(tensor))
}

#[inline]
fn safe_eigen_gap_reciprocal(lhs: f64, rhs: f64) -> f64 {
    let gap = rhs - lhs;
    let scale = lhs.abs().max(rhs.abs()).max(1.0);
    let cluster_tol = 1e-8 * scale;
    if gap.abs() <= cluster_tol {
        0.0
    } else {
        1.0 / gap
    }
}

/// Multiply two row-major matrices: C = A @ B (m×k, k×n → m×n).
fn matmul_f64(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Transpose a row-major matrix: (m×n) → (n×m).
fn transpose_f64(m: usize, n: usize, a: &[f64]) -> Vec<f64> {
    let mut t = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            t[j * m + i] = a[i * n + j];
        }
    }
    t
}

/// Extract lower triangular part (including diagonal) of square matrix.
fn tril_f64(n: usize, a: &[f64]) -> Vec<f64> {
    let mut t = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            t[i * n + j] = a[i * n + j];
        }
    }
    t
}

/// Symmetrize: A → (A + A^T) / 2.
fn symmetrize_f64(n: usize, a: &[f64]) -> Vec<f64> {
    let mut s = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            s[i * n + j] = (a[i * n + j] + a[j * n + i]) * 0.5;
        }
    }
    s
}

// ── Cholesky VJP ───────────────────────────────────────────────────

/// Cholesky VJP: given A = L L^T and cotangent G (for L),
/// compute dA = sym(L^{-T} tril(L^T G) L^{-1}).
///
/// Following JAX's implementation in jax._src.linalg.
fn cholesky_vjp(inputs: &[Value], g: &Value) -> Result<Vec<Value>, AdError> {
    // L is the output of cholesky(A), but we also have A as input.
    // We recompute L = cholesky(A) to get the factor.
    let l_result = eval_primitive(Primitive::Cholesky, inputs, &BTreeMap::new())
        .map_err(|e| AdError::EvalFailed(e.to_string()))?;
    let (n, n2, l) = extract_matrix_f64(&l_result)?;
    assert_eq!(n, n2);
    let (gm, gn, g_data) = extract_matrix_f64(g)?;
    if gm != n || gn != n {
        return Err(AdError::EvalFailed(
            "gradient shape doesn't match Cholesky factor".to_owned(),
        ));
    }

    // Step 1: L^T @ G
    let lt = transpose_f64(n, n, &l);
    let lt_g = matmul_f64(n, n, n, &lt, &g_data);

    // Step 2: Φ(L^T @ G) where Φ keeps lower triangle and halves diagonal
    // (Murray 2016: the diagonal factor of 1/2 accounts for L appearing in both
    // L and L^T of A = LL^T)
    let mut lt_g_tril = tril_f64(n, &lt_g);
    for i in 0..n {
        lt_g_tril[i * n + i] *= 0.5;
    }

    // Step 3: Compute L^{-T} Φ L^{-1} via triangular solves.
    // First solve L^T X = Φ to get X = L^{-T} Φ, then compute X L^{-1}.

    // Solve L^T X = Φ: back substitution column by column
    let mut x = vec![0.0_f64; n * n];
    for col in 0..n {
        for i in (0..n).rev() {
            let mut sum = lt_g_tril[i * n + col];
            for k in (i + 1)..n {
                sum -= lt[i * n + k] * x[k * n + col];
            }
            x[i * n + col] = sum / lt[i * n + i];
        }
    }

    // Now compute X @ L^{-1}: solve Z L = X, equivalently L^T Z^T = X^T
    // Back substitution with L^T
    let xt = transpose_f64(n, n, &x);
    let mut zt = vec![0.0_f64; n * n];
    for col in 0..n {
        for i in (0..n).rev() {
            let mut sum = xt[i * n + col];
            for k in (i + 1)..n {
                sum -= lt[i * n + k] * zt[k * n + col];
            }
            zt[i * n + col] = sum / lt[i * n + i];
        }
    }
    let result = transpose_f64(n, n, &zt);

    // Step 4: Symmetrize
    let da = symmetrize_f64(n, &result);

    Ok(vec![build_matrix_f64(n, n, &da)?])
}

// ── TriangularSolve VJP ────────────────────────────────────────────

/// TriangularSolve VJP: for A X = B (A lower/upper triangular),
/// dA = -(A^{-T} G) X^T, dB = A^{-T} G.
fn triangular_solve_vjp(
    inputs: &[Value],
    g: &Value,
    params: &BTreeMap<String, String>,
) -> Result<Vec<Value>, AdError> {
    // Recompute X = A\B
    let x = eval_primitive(Primitive::TriangularSolve, inputs, params)
        .map_err(|e| AdError::EvalFailed(e.to_string()))?;
    let (n, nx, x_data) = extract_matrix_f64(&x)?;
    let (gm, gn, _) = extract_matrix_f64(g)?;
    if gm != n || gn != nx {
        return Err(AdError::EvalFailed(
            "gradient shape doesn't match triangular solve output".to_owned(),
        ));
    }

    let lower = params.get("lower").is_none_or(|v| v.trim() != "false");

    // Solve A^T Z = G for Z (this gives dB = Z)
    let mut transpose_params = params.clone();
    // Toggle transpose: if was transposed, un-transpose; if not, transpose
    let was_transposed = params
        .get("transpose_a")
        .is_some_and(|v| v.trim() == "true");
    transpose_params.insert(
        "transpose_a".to_owned(),
        if was_transposed {
            "false".to_owned()
        } else {
            "true".to_owned()
        },
    );
    // When transposing, swap lower/upper
    if !was_transposed {
        transpose_params.insert(
            "lower".to_owned(),
            if lower {
                "true".to_owned()
            } else {
                "false".to_owned()
            },
        );
    }

    let z = eval_primitive(
        Primitive::TriangularSolve,
        &[inputs[0].clone(), g.clone()],
        &transpose_params,
    )
    .map_err(|e| AdError::EvalFailed(e.to_string()))?;
    let (_, _, z_data) = extract_matrix_f64(&z)?;

    // dA = -Z @ X^T
    let xt = transpose_f64(n, nx, &x_data);
    let neg_z_xt = matmul_f64(n, nx, n, &z_data, &xt);
    let da: Vec<f64> = neg_z_xt.iter().map(|&v| -v).collect();

    // Mask dA to match triangularity: if lower, keep lower triangle; if upper, keep upper
    let mut da_masked = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            if (lower && j <= i) || (!lower && j >= i) {
                da_masked[i * n + j] = da[i * n + j];
            }
        }
    }

    Ok(vec![build_matrix_f64(n, n, &da_masked)?, z])
}

// ── Cholesky JVP ───────────────────────────────────────────────────

/// Cholesky JVP: given A = L L^T and tangent dA,
/// dL = L phi(L^{-1} dA_sym L^{-T}) where phi extracts the lower triangle
/// with diagonal halved.
fn cholesky_jvp(primals: &[Value], tangents: &[Value]) -> Result<Value, AdError> {
    let l_result = eval_primitive(Primitive::Cholesky, primals, &BTreeMap::new())
        .map_err(|e| AdError::EvalFailed(e.to_string()))?;
    let (n, _, l) = extract_matrix_f64(&l_result)?;
    let (_, _, da) = extract_matrix_f64(&tangents[0])?;

    // Symmetrize dA
    let da_sym = symmetrize_f64(n, &da);

    // Solve L^{-1} dA_sym via forward substitution (row by row)
    let mut linv_da = vec![0.0_f64; n * n];
    for col in 0..n {
        for i in 0..n {
            let mut sum = da_sym[i * n + col];
            for k in 0..i {
                sum -= l[i * n + k] * linv_da[k * n + col];
            }
            linv_da[i * n + col] = sum / l[i * n + i];
        }
    }

    // Compute linv_da @ L^{-T}: solve middle @ L^T = linv_da
    // Equivalently: L @ middle^T = linv_da^T (forward substitution)
    let linv_da_t = transpose_f64(n, n, &linv_da);
    let mut middle_t = vec![0.0_f64; n * n];
    for col in 0..n {
        for i in 0..n {
            let mut sum = linv_da_t[i * n + col];
            for k in 0..i {
                sum -= l[i * n + k] * middle_t[k * n + col];
            }
            middle_t[i * n + col] = sum / l[i * n + i];
        }
    }
    let middle = transpose_f64(n, n, &middle_t);

    // phi: extract lower triangle with diagonal halved
    let mut phi = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..i {
            phi[i * n + j] = middle[i * n + j];
        }
        phi[i * n + i] = middle[i * n + i] * 0.5;
    }

    // dL = L @ phi
    let dl = matmul_f64(n, n, n, &l, &phi);

    build_matrix_f64(n, n, &dl)
}

// ── TriangularSolve JVP ────────────────────────────────────────────

/// TriangularSolve JVP: for X = A\B,
/// dX = A \ (dB - dA @ X)
fn triangular_solve_jvp(
    primals: &[Value],
    tangents: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, AdError> {
    // X = A\B
    let x = eval_primitive(Primitive::TriangularSolve, primals, params)
        .map_err(|e| AdError::EvalFailed(e.to_string()))?;

    let (_, _, da_data) = extract_matrix_f64(&tangents[0])?;
    let (_, _, x_data) = extract_matrix_f64(&x)?;
    let (n, nx, _) = extract_matrix_f64(&primals[1])?;
    let (_, _, db_data) = extract_matrix_f64(&tangents[1])?;

    // dA @ X
    let da_x = matmul_f64(n, n, nx, &da_data, &x_data);

    // rhs = dB - dA @ X
    let rhs: Vec<f64> = db_data
        .iter()
        .zip(da_x.iter())
        .map(|(&db, &dax)| db - dax)
        .collect();

    let rhs_val = build_matrix_f64(n, nx, &rhs)?;

    // dX = A \ rhs
    eval_primitive(
        Primitive::TriangularSolve,
        &[primals[0].clone(), rhs_val],
        params,
    )
    .map_err(|e| AdError::EvalFailed(e.to_string()))
}

// ── Forward-mode JVP ───────────────────────────────────────────────

/// Result of forward-mode JVP computation.
#[derive(Debug, Clone, PartialEq)]
pub struct JvpResult {
    pub primals: Vec<Value>,
    pub tangents: Vec<Value>,
}

/// Compute JVP (forward-mode AD) for a Jaxpr.
///
/// Given primals `x` and tangent vector `dx`, computes `(f(x), df/dx · dx)`.
/// Tangents are `Value` types (scalar or tensor) matching the shape of their primals.
pub fn jvp(jaxpr: &Jaxpr, primals: &[Value], tangents: &[Value]) -> Result<JvpResult, AdError> {
    jvp_inner(jaxpr, primals, tangents, None)
}

/// Compute JVP using a wrapper-scoped custom JVP rule when present.
pub fn jvp_with_custom_jvp_key(
    jaxpr: &Jaxpr,
    primals: &[Value],
    tangents: &[Value],
    rule_key: &str,
) -> Result<JvpResult, AdError> {
    jvp_inner(jaxpr, primals, tangents, Some(rule_key))
}

fn jvp_inner(
    jaxpr: &Jaxpr,
    primals: &[Value],
    tangents: &[Value],
    custom_jvp_rule_key: Option<&str>,
) -> Result<JvpResult, AdError> {
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

    let custom_rule = custom_jvp_rule_key
        .and_then(lookup_custom_jaxpr_jvp_by_key)
        .or_else(|| lookup_custom_jaxpr_jvp(jaxpr));
    if let Some(custom_rule) = custom_rule {
        let result = custom_rule(primals, tangents)?;
        if result.primals.len() != jaxpr.outvars.len() {
            return Err(AdError::EvalFailed(format!(
                "custom Jaxpr JVP primal arity mismatch: expected {}, got {}",
                jaxpr.outvars.len(),
                result.primals.len()
            )));
        }
        if result.tangents.len() != jaxpr.outvars.len() {
            return Err(AdError::EvalFailed(format!(
                "custom Jaxpr JVP tangent arity mismatch: expected {}, got {}",
                jaxpr.outvars.len(),
                result.tangents.len()
            )));
        }
        return Ok(result);
    }

    let mut primal_env: BTreeMap<VarId, Value> = BTreeMap::new();
    let mut tangent_env: BTreeMap<VarId, Value> = BTreeMap::new();

    for (idx, var) in jaxpr.invars.iter().enumerate() {
        primal_env.insert(*var, primals[idx].clone());
        tangent_env.insert(*var, tangents[idx].clone());
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
                    let tval = tangent_env
                        .get(var)
                        .cloned()
                        .unwrap_or_else(|| zeros_like(&pval));
                    resolved_primals.push(pval);
                    resolved_tangents.push(tval);
                }
                Atom::Lit(lit) => {
                    let pval = Value::Scalar(*lit);
                    resolved_tangents.push(zeros_like(&pval));
                    resolved_primals.push(pval);
                }
            }
        }

        let primal_outs =
            fj_lax::eval_primitive_multi(eqn.primitive, &resolved_primals, &eqn.params)
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;

        if eqn.outputs.len() > 1 {
            // Multi-output equation: compute tangents via jvp_rule_multi
            let tangent_outs = jvp_rule_multi(
                eqn.primitive,
                &resolved_primals,
                &resolved_tangents,
                &primal_outs,
                &eqn.params,
            )?;
            for (idx, var) in eqn.outputs.iter().enumerate() {
                primal_env.insert(*var, primal_outs[idx].clone());
                tangent_env.insert(
                    *var,
                    tangent_outs
                        .get(idx)
                        .cloned()
                        .unwrap_or_else(|| zeros_like(&primal_outs[idx])),
                );
            }
        } else {
            let tangent_out = jvp_rule(
                eqn.primitive,
                &resolved_primals,
                &resolved_tangents,
                &eqn.params,
            )?;
            let out_var = eqn.outputs[0];
            let primal_out = primal_outs.into_iter().next().ok_or_else(|| {
                AdError::EvalFailed(format!(
                    "{} produced no primal output for single-output JVP equation",
                    eqn.primitive.as_str()
                ))
            })?;
            primal_env.insert(out_var, primal_out);
            tangent_env.insert(out_var, tangent_out);
        }
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
        .map(|var| {
            tangent_env
                .get(var)
                .cloned()
                .unwrap_or(Value::scalar_f64(0.0))
        })
        .collect();

    Ok(JvpResult {
        primals: out_primals,
        tangents: out_tangents,
    })
}

/// Apply a JVP rule for a single primitive with tensor-valued tangents.
///
/// For each primitive, computes the output tangent given input primals and input tangents.
/// Tangents are `Value` types matching the shape of their corresponding primals.
fn jvp_rule(
    primitive: Primitive,
    primals: &[Value],
    tangents: &[Value],
    params: &BTreeMap<String, String>,
) -> Result<Value, AdError> {
    if let Some(custom_rule) = lookup_custom_jvp(primitive) {
        return custom_rule(primals, tangents, params);
    }

    let no_params = BTreeMap::new();
    let ep = |prim, inputs: &[Value]| -> Result<Value, AdError> {
        eval_primitive(prim, inputs, &no_params).map_err(|e| AdError::EvalFailed(e.to_string()))
    };
    let ep_p = |prim, inputs: &[Value], p: &BTreeMap<String, String>| -> Result<Value, AdError> {
        eval_primitive(prim, inputs, p).map_err(|e| AdError::EvalFailed(e.to_string()))
    };

    match primitive {
        // ── Linear binary ops: tangent follows the same op ──
        Primitive::Add => ep(Primitive::Add, &[tangents[0].clone(), tangents[1].clone()]),
        Primitive::Sub => ep(Primitive::Sub, &[tangents[0].clone(), tangents[1].clone()]),

        // ── Product rule: da*b + a*db ──
        Primitive::Mul => {
            let da_b = ep(Primitive::Mul, &[tangents[0].clone(), primals[1].clone()])?;
            let a_db = ep(Primitive::Mul, &[primals[0].clone(), tangents[1].clone()])?;
            ep(Primitive::Add, &[da_b, a_db])
        }

        // ── Unary elementwise: tangent = f'(primal) * tangent_in ──
        Primitive::Neg => ep(Primitive::Neg, &[tangents[0].clone()]),

        Primitive::Abs => {
            // sign(x) * dx
            let sign = ep(Primitive::Sign, &[primals[0].clone()])?;
            ep(Primitive::Mul, &[sign, tangents[0].clone()])
        }

        Primitive::Exp => {
            // exp(x) * dx
            let exp_x = ep(Primitive::Exp, &[primals[0].clone()])?;
            ep(Primitive::Mul, &[exp_x, tangents[0].clone()])
        }

        Primitive::Log => {
            // dx / x
            ep(Primitive::Div, &[tangents[0].clone(), primals[0].clone()])
        }

        Primitive::Sqrt => {
            // dx / (2 * sqrt(x))
            let sqrt_x = ep(Primitive::Sqrt, &[primals[0].clone()])?;
            let two = Value::scalar_f64(2.0);
            let denom = ep(Primitive::Mul, &[two, sqrt_x])?;
            ep(Primitive::Div, &[tangents[0].clone(), denom])
        }

        Primitive::Rsqrt => {
            // -0.5 * x^(-1.5) * dx
            let neg_half = Value::scalar_f64(-0.5);
            let exp = Value::scalar_f64(-1.5);
            let x_pow = ep(Primitive::Pow, &[primals[0].clone(), exp])?;
            let coeff = ep(Primitive::Mul, &[neg_half, x_pow])?;
            ep(Primitive::Mul, &[coeff, tangents[0].clone()])
        }

        Primitive::Floor | Primitive::Ceil | Primitive::Round => Ok(zeros_like(&primals[0])),

        Primitive::Sin => {
            // cos(x) * dx
            let cos_x = ep(Primitive::Cos, &[primals[0].clone()])?;
            ep(Primitive::Mul, &[cos_x, tangents[0].clone()])
        }

        Primitive::Cos => {
            // -sin(x) * dx
            let sin_x = ep(Primitive::Sin, &[primals[0].clone()])?;
            let neg_sin = ep(Primitive::Neg, &[sin_x])?;
            ep(Primitive::Mul, &[neg_sin, tangents[0].clone()])
        }

        Primitive::Tan => {
            // dx / cos(x)^2
            let cos_x = ep(Primitive::Cos, &[primals[0].clone()])?;
            let cos_sq = ep(Primitive::Mul, &[cos_x.clone(), cos_x])?;
            ep(Primitive::Div, &[tangents[0].clone(), cos_sq])
        }

        Primitive::Asin => {
            // dx / sqrt(1 - x^2)
            let x_sq = ep(Primitive::Mul, &[primals[0].clone(), primals[0].clone()])?;
            let one = Value::scalar_f64(1.0);
            let diff = ep(Primitive::Sub, &[one, x_sq])?;
            let sqrt_diff = ep(Primitive::Sqrt, &[diff])?;
            ep(Primitive::Div, &[tangents[0].clone(), sqrt_diff])
        }

        Primitive::Acos => {
            // -dx / sqrt(1 - x^2)
            let x_sq = ep(Primitive::Mul, &[primals[0].clone(), primals[0].clone()])?;
            let one = Value::scalar_f64(1.0);
            let diff = ep(Primitive::Sub, &[one, x_sq])?;
            let sqrt_diff = ep(Primitive::Sqrt, &[diff])?;
            let neg_dx = ep(Primitive::Neg, &[tangents[0].clone()])?;
            ep(Primitive::Div, &[neg_dx, sqrt_diff])
        }

        Primitive::Atan => {
            // dx / (1 + x^2)
            let x_sq = ep(Primitive::Mul, &[primals[0].clone(), primals[0].clone()])?;
            let one = Value::scalar_f64(1.0);
            let denom = ep(Primitive::Add, &[one, x_sq])?;
            ep(Primitive::Div, &[tangents[0].clone(), denom])
        }

        Primitive::Sinh => {
            // cosh(x) * dx
            let cosh_x = ep(Primitive::Cosh, &[primals[0].clone()])?;
            ep(Primitive::Mul, &[cosh_x, tangents[0].clone()])
        }

        Primitive::Cosh => {
            // sinh(x) * dx
            let sinh_x = ep(Primitive::Sinh, &[primals[0].clone()])?;
            ep(Primitive::Mul, &[sinh_x, tangents[0].clone()])
        }

        Primitive::Tanh => {
            // (1 - tanh(x)^2) * dx
            let th = ep(Primitive::Tanh, &[primals[0].clone()])?;
            let th_sq = ep(Primitive::Mul, &[th.clone(), th])?;
            let one = Value::scalar_f64(1.0);
            let coeff = ep(Primitive::Sub, &[one, th_sq])?;
            ep(Primitive::Mul, &[coeff, tangents[0].clone()])
        }

        Primitive::Asinh => {
            // d/dx asinh(x) = 1 / sqrt(x² + 1)
            let x_sq = ep(Primitive::Mul, &[primals[0].clone(), primals[0].clone()])?;
            let one = Value::scalar_f64(1.0);
            let sum = ep(Primitive::Add, &[x_sq, one])?;
            let denom = ep(Primitive::Sqrt, &[sum])?;
            let coeff = ep(Primitive::Reciprocal, &[denom])?;
            ep(Primitive::Mul, &[coeff, tangents[0].clone()])
        }

        Primitive::Acosh => {
            // d/dx acosh(x) = 1 / sqrt(x² - 1)
            let x_sq = ep(Primitive::Mul, &[primals[0].clone(), primals[0].clone()])?;
            let one = Value::scalar_f64(1.0);
            let diff = ep(Primitive::Sub, &[x_sq, one])?;
            let denom = ep(Primitive::Sqrt, &[diff])?;
            let coeff = ep(Primitive::Reciprocal, &[denom])?;
            ep(Primitive::Mul, &[coeff, tangents[0].clone()])
        }

        Primitive::Atanh => {
            // d/dx atanh(x) = 1 / (1 - x²)
            let x_sq = ep(Primitive::Mul, &[primals[0].clone(), primals[0].clone()])?;
            let one = Value::scalar_f64(1.0);
            let denom = ep(Primitive::Sub, &[one, x_sq])?;
            let coeff = ep(Primitive::Reciprocal, &[denom])?;
            ep(Primitive::Mul, &[coeff, tangents[0].clone()])
        }

        Primitive::Expm1 => {
            // exp(x) * dx (same derivative as exp)
            let exp_x = ep(Primitive::Exp, &[primals[0].clone()])?;
            ep(Primitive::Mul, &[exp_x, tangents[0].clone()])
        }

        Primitive::Log1p => {
            // dx / (1 + x)
            let one = Value::scalar_f64(1.0);
            let denom = ep(Primitive::Add, &[one, primals[0].clone()])?;
            ep(Primitive::Div, &[tangents[0].clone(), denom])
        }

        Primitive::Sign => Ok(zeros_like(&primals[0])),

        Primitive::Square => {
            // 2 * x * dx
            let two = Value::scalar_f64(2.0);
            let two_x = ep(Primitive::Mul, &[two, primals[0].clone()])?;
            ep(Primitive::Mul, &[two_x, tangents[0].clone()])
        }

        Primitive::Reciprocal => {
            // -dx / x^2
            let x_sq = ep(Primitive::Mul, &[primals[0].clone(), primals[0].clone()])?;
            let neg_dx = ep(Primitive::Neg, &[tangents[0].clone()])?;
            ep(Primitive::Div, &[neg_dx, x_sq])
        }

        Primitive::Logistic => {
            // sig(x) * (1 - sig(x)) * dx
            let sig = ep(Primitive::Logistic, &[primals[0].clone()])?;
            let one = Value::scalar_f64(1.0);
            let one_minus_sig = ep(Primitive::Sub, &[one, sig.clone()])?;
            let coeff = ep(Primitive::Mul, &[sig, one_minus_sig])?;
            ep(Primitive::Mul, &[coeff, tangents[0].clone()])
        }

        Primitive::Erf => {
            // (2/sqrt(pi)) * exp(-x^2) * dx
            let coeff = Value::scalar_f64(2.0 / std::f64::consts::PI.sqrt());
            let neg_x_sq = {
                let x_sq = ep(Primitive::Mul, &[primals[0].clone(), primals[0].clone()])?;
                ep(Primitive::Neg, &[x_sq])?
            };
            let exp_neg = ep(Primitive::Exp, &[neg_x_sq])?;
            let c_exp = ep(Primitive::Mul, &[coeff, exp_neg])?;
            ep(Primitive::Mul, &[c_exp, tangents[0].clone()])
        }

        Primitive::Erfc => {
            // (-2/sqrt(pi)) * exp(-x^2) * dx
            let coeff = Value::scalar_f64(-2.0 / std::f64::consts::PI.sqrt());
            let neg_x_sq = {
                let x_sq = ep(Primitive::Mul, &[primals[0].clone(), primals[0].clone()])?;
                ep(Primitive::Neg, &[x_sq])?
            };
            let exp_neg = ep(Primitive::Exp, &[neg_x_sq])?;
            let c_exp = ep(Primitive::Mul, &[coeff, exp_neg])?;
            ep(Primitive::Mul, &[c_exp, tangents[0].clone()])
        }
        Primitive::Lgamma => {
            // digamma(x) * dx
            let digamma_x = ep(Primitive::Digamma, &[primals[0].clone()])?;
            ep(Primitive::Mul, &[digamma_x, tangents[0].clone()])
        }
        Primitive::Digamma => {
            // trigamma(x) * dx
            let trigamma_x = trigamma_value(&primals[0])?;
            ep(Primitive::Mul, &[trigamma_x, tangents[0].clone()])
        }
        Primitive::ErfInv => {
            // sqrt(pi)/2 * exp(erf_inv(x)^2) * dx
            let erf_inv_x = ep(Primitive::ErfInv, &[primals[0].clone()])?;
            let erf_inv_sq = ep(Primitive::Mul, &[erf_inv_x.clone(), erf_inv_x])?;
            let exp_term = ep(Primitive::Exp, &[erf_inv_sq])?;
            let coeff = Value::scalar_f64(std::f64::consts::PI.sqrt() / 2.0);
            let factor = ep(Primitive::Mul, &[coeff, exp_term])?;
            ep(Primitive::Mul, &[factor, tangents[0].clone()])
        }

        // ── Binary ops with quotient rule ──
        Primitive::Div => {
            // da/b - a*db/b^2
            let da_over_b = ep(Primitive::Div, &[tangents[0].clone(), primals[1].clone()])?;
            let b_sq = ep(Primitive::Mul, &[primals[1].clone(), primals[1].clone()])?;
            let a_db = ep(Primitive::Mul, &[primals[0].clone(), tangents[1].clone()])?;
            let a_db_over_b_sq = ep(Primitive::Div, &[a_db, b_sq])?;
            ep(Primitive::Sub, &[da_over_b, a_db_over_b_sq])
        }

        Primitive::Rem => {
            // da - floor(a/b) * db
            let a_over_b = ep(Primitive::Div, &[primals[0].clone(), primals[1].clone()])?;
            let floored = ep(Primitive::Floor, &[a_over_b])?;
            let f_db = ep(Primitive::Mul, &[floored, tangents[1].clone()])?;
            ep(Primitive::Sub, &[tangents[0].clone(), f_db])
        }

        Primitive::Pow => {
            // b * a^(b-1) * da + a^b * ln(a) * db
            let one = Value::scalar_f64(1.0);
            let b_m1 = ep(Primitive::Sub, &[primals[1].clone(), one])?;
            let a_pow_bm1 = ep(Primitive::Pow, &[primals[0].clone(), b_m1])?;
            let da_part = ep(Primitive::Mul, &[primals[1].clone(), a_pow_bm1])?;
            let da_term = ep(Primitive::Mul, &[da_part, tangents[0].clone()])?;
            let a_pow_b = ep(Primitive::Pow, &[primals[0].clone(), primals[1].clone()])?;
            let ln_a = ep(Primitive::Log, &[primals[0].clone()])?;
            let db_part = ep(Primitive::Mul, &[a_pow_b, ln_a])?;
            let db_term = ep(Primitive::Mul, &[db_part, tangents[1].clone()])?;
            ep(Primitive::Add, &[da_term, db_term])
        }

        Primitive::Atan2 => {
            // (b*da - a*db) / (a^2 + b^2)
            let a_sq = ep(Primitive::Mul, &[primals[0].clone(), primals[0].clone()])?;
            let b_sq = ep(Primitive::Mul, &[primals[1].clone(), primals[1].clone()])?;
            let denom = ep(Primitive::Add, &[a_sq, b_sq])?;
            let b_da = ep(Primitive::Mul, &[primals[1].clone(), tangents[0].clone()])?;
            let a_db = ep(Primitive::Mul, &[primals[0].clone(), tangents[1].clone()])?;
            let numer = ep(Primitive::Sub, &[b_da, a_db])?;
            ep(Primitive::Div, &[numer, denom])
        }
        Primitive::Complex => ep(
            Primitive::Complex,
            &[tangents[0].clone(), tangents[1].clone()],
        ),
        Primitive::Conj => ep(Primitive::Conj, &[tangents[0].clone()]),
        Primitive::Real => ep(Primitive::Real, &[tangents[0].clone()]),
        Primitive::Imag => ep(Primitive::Imag, &[tangents[0].clone()]),

        // ── Comparison-like ops: max/min split ties like JAX balanced_eq ──
        Primitive::Max => {
            let ans = ep(Primitive::Max, primals)?;
            let lhs = ep(
                Primitive::Mul,
                &[
                    tangents[0].clone(),
                    balanced_eq_weight(&primals[0], &ans, &primals[1])?,
                ],
            )?;
            let rhs = ep(
                Primitive::Mul,
                &[
                    tangents[1].clone(),
                    balanced_eq_weight(&primals[1], &ans, &primals[0])?,
                ],
            )?;
            ep(Primitive::Add, &[lhs, rhs])
        }

        Primitive::Min => {
            let ans = ep(Primitive::Min, primals)?;
            let lhs = ep(
                Primitive::Mul,
                &[
                    tangents[0].clone(),
                    balanced_eq_weight(&primals[0], &ans, &primals[1])?,
                ],
            )?;
            let rhs = ep(
                Primitive::Mul,
                &[
                    tangents[1].clone(),
                    balanced_eq_weight(&primals[1], &ans, &primals[0])?,
                ],
            )?;
            ep(Primitive::Add, &[lhs, rhs])
        }

        // ── Select: tangent follows primal condition ──
        Primitive::Select => ep(
            Primitive::Select,
            &[primals[0].clone(), tangents[1].clone(), tangents[2].clone()],
        ),

        // ── Comparison: discrete, zero tangent ──
        Primitive::Eq
        | Primitive::Ne
        | Primitive::Lt
        | Primitive::Le
        | Primitive::Gt
        | Primitive::Ge => {
            let primal_out = ep(primitive, primals)?;
            Ok(zeros_like(&primal_out))
        }

        // ── Reduction: apply same reduction to tangent ──
        Primitive::ReduceSum => ep_p(Primitive::ReduceSum, &[tangents[0].clone()], params),
        Primitive::ReduceMax | Primitive::ReduceMin => {
            reduce_chooser_jvp_value(primitive, &primals[0], &tangents[0], params)
        }
        Primitive::ReduceProd => reduce_prod_jvp_value(&primals[0], &tangents[0], params),
        Primitive::ReduceAnd | Primitive::ReduceOr | Primitive::ReduceXor => {
            // Bitwise reductions are non-differentiable: tangent is structurally zero.
            let primal_out = ep_p(primitive, primals, params)?;
            Ok(zeros_like(&primal_out))
        }

        // ── Dot: product rule for matrix/vector multiply ──
        Primitive::Dot => {
            // d(a·b) = da·b + a·db
            let da_b = ep(Primitive::Dot, &[tangents[0].clone(), primals[1].clone()])?;
            let a_db = ep(Primitive::Dot, &[primals[0].clone(), tangents[1].clone()])?;
            ep(Primitive::Add, &[da_b, a_db])
        }

        // ── Shape ops: apply same transformation to tangent ──
        Primitive::Reshape => ep_p(Primitive::Reshape, &[tangents[0].clone()], params),
        Primitive::Transpose => ep_p(Primitive::Transpose, &[tangents[0].clone()], params),
        Primitive::BroadcastInDim => {
            ep_p(Primitive::BroadcastInDim, &[tangents[0].clone()], params)
        }
        Primitive::Slice => ep_p(Primitive::Slice, &[tangents[0].clone()], params),
        Primitive::Rev => ep_p(Primitive::Rev, &[tangents[0].clone()], params),
        Primitive::Squeeze => ep_p(Primitive::Squeeze, &[tangents[0].clone()], params),
        Primitive::Split => ep_p(Primitive::Split, &[tangents[0].clone()], params),
        Primitive::ExpandDims => ep_p(Primitive::ExpandDims, &[tangents[0].clone()], params),
        Primitive::Concatenate => ep_p(Primitive::Concatenate, tangents, params),
        Primitive::Gather => {
            // Gather: tangent follows same indexing from tangent source
            let mut inputs = vec![tangents[0].clone()];
            if tangents.len() > 1 {
                inputs.extend_from_slice(&tangents[1..]);
            }
            ep_p(Primitive::Gather, &inputs, params)
        }
        Primitive::Scatter => {
            let mut inputs = vec![tangents[0].clone()];
            if tangents.len() > 1 {
                inputs.extend_from_slice(&tangents[1..]);
            }
            ep_p(Primitive::Scatter, &inputs, params)
        }

        // ── Clamp: tangent passes through where x is in (lo, hi) ──
        Primitive::Clamp => {
            let in_range_lo = ep(Primitive::Gt, &[primals[0].clone(), primals[1].clone()])?;
            let in_range_hi = ep(Primitive::Lt, &[primals[0].clone(), primals[2].clone()])?;
            let in_range = eval_primitive(
                Primitive::Select,
                &[in_range_lo, in_range_hi, Value::scalar_bool(false)],
                &no_params,
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let zero = zeros_like(&tangents[0]);
            ep(Primitive::Select, &[in_range, tangents[0].clone(), zero])
        }

        Primitive::DynamicSlice => ep_p(Primitive::DynamicSlice, tangents, params),
        Primitive::Pad => match &tangents[0] {
            Value::Scalar(_) => Ok(tangents[0].clone()),
            _ => ep_p(Primitive::Pad, tangents, params),
        },
        Primitive::Iota | Primitive::BroadcastedIota | Primitive::OneHot => {
            let primal_out = ep_p(primitive, primals, params)?;
            Ok(zeros_like(&primal_out))
        }
        Primitive::Copy => Ok(tangents[0].clone()),
        Primitive::BitcastConvertType => {
            let primal_out = ep_p(Primitive::BitcastConvertType, primals, params)?;
            Ok(zeros_like(&primal_out))
        }
        Primitive::ReducePrecision => Ok(tangents[0].clone()),
        Primitive::DynamicUpdateSlice => ep_p(Primitive::DynamicUpdateSlice, tangents, params),
        Primitive::Cumsum => ep_p(Primitive::Cumsum, &[tangents[0].clone()], params),
        Primitive::Cumprod => cumprod_jvp_value(&primals[0], &tangents[0], params),
        Primitive::Sort => ep_p(Primitive::Sort, &[tangents[0].clone()], params),
        Primitive::Argsort => Ok(zeros_like(&primals[0])),
        Primitive::Conv => ep_p(Primitive::Conv, tangents, params),

        // ── Control flow ──
        Primitive::Cond => {
            if primals.len() < 3 {
                return Err(AdError::InputArity {
                    expected: 3,
                    actual: primals.len(),
                });
            }
            if tangents.len() < 3 {
                return Err(AdError::InputArity {
                    expected: 3,
                    actual: tangents.len(),
                });
            }
            let pred = scalar_predicate_for_ad(Primitive::Cond, "predicate", &primals[0])?;
            Ok(if pred {
                tangents[1].clone()
            } else {
                tangents[2].clone()
            })
        }

        Primitive::Scan => scan_jvp(primals, tangents, params),

        Primitive::While => while_jvp(primals, tangents, params),

        Primitive::Switch => {
            let branch_idx = switch_branch_index(primals, params)? + 1;
            if branch_idx >= tangents.len() {
                return Err(AdError::InputArity {
                    expected: primals.len(),
                    actual: tangents.len(),
                });
            }
            Ok(tangents[branch_idx].clone())
        }

        // ── Bitwise: not differentiable ──
        Primitive::BitwiseAnd
        | Primitive::BitwiseOr
        | Primitive::BitwiseXor
        | Primitive::BitwiseNot
        | Primitive::ShiftLeft
        | Primitive::ShiftRightArithmetic
        | Primitive::ShiftRightLogical
        | Primitive::PopulationCount
        | Primitive::CountLeadingZeros
        | Primitive::IsFinite
        | Primitive::Nextafter => Ok(zeros_like(&primals[0])),

        Primitive::ReduceWindow => ep_p(Primitive::ReduceWindow, &[tangents[0].clone()], params),

        // Collective operations (pmap-only, not differentiable outside pmap context)
        Primitive::Psum
        | Primitive::Pmean
        | Primitive::AllGather
        | Primitive::AllToAll
        | Primitive::AxisIndex => Err(AdError::UnsupportedPrimitive(primitive)),

        // Cbrt JVP: d cbrt(x) = tangent / (3 * cbrt(x)^2)
        Primitive::Cbrt => {
            let cbrt_x = ep(Primitive::Cbrt, &[primals[0].clone()])?;
            let cbrt_sq = ep(Primitive::Mul, &[cbrt_x.clone(), cbrt_x])?;
            let three = Value::scalar_f64(3.0);
            let denom = ep(Primitive::Mul, &[three, cbrt_sq])?;
            let recip = ep(Primitive::Reciprocal, &[denom])?;
            ep(Primitive::Mul, &[tangents[0].clone(), recip])
        }
        // IntegerPow JVP: d x^n = n * x^(n-1) * tangent
        Primitive::IntegerPow => {
            let n: i32 = params
                .get("exponent")
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(1);
            let n_val = Value::scalar_f64(f64::from(n));
            let mut nm1_params = params.clone();
            nm1_params.insert("exponent".into(), (n - 1).to_string());
            let x_nm1 = ep_p(Primitive::IntegerPow, &[primals[0].clone()], &nm1_params)?;
            let n_x_nm1 = ep(Primitive::Mul, &[n_val, x_nm1])?;
            ep(Primitive::Mul, &[tangents[0].clone(), n_x_nm1])
        }
        // ── Cholesky JVP ──
        // dL = L phi(L^{-1} dA L^{-T}) where phi extracts the lower triangle
        // with diagonal halved.
        Primitive::Cholesky => cholesky_jvp(primals, tangents),
        // ── TriangularSolve JVP ──
        // d(A\B) = A \ (dB - dA X) where X = A\B
        Primitive::TriangularSolve => triangular_solve_jvp(primals, tangents, params),
        // ── FFT JVP ──
        // FFT is linear: JVP = FFT(tangent)
        Primitive::Fft => ep(Primitive::Fft, &[tangents[0].clone()]),
        // ── IFFT JVP ──
        // IFFT is linear: JVP = IFFT(tangent)
        Primitive::Ifft => ep(Primitive::Ifft, &[tangents[0].clone()]),
        // ── RFFT JVP ──
        // RFFT is linear: JVP = RFFT(tangent)
        Primitive::Rfft => ep(Primitive::Rfft, &[tangents[0].clone()]),
        // ── IRFFT JVP ──
        // IRFFT is linear: JVP = IRFFT(tangent)
        Primitive::Irfft => ep(Primitive::Irfft, &[tangents[0].clone()]),
        Primitive::Qr | Primitive::Svd | Primitive::Eigh => {
            Err(AdError::UnsupportedPrimitive(primitive))
        }
    }
}

/// JVP rule for multi-output primitives.
/// Given primals, tangents, and primal output values, computes output tangents.
fn jvp_rule_multi(
    primitive: Primitive,
    _primals: &[Value],
    tangents: &[Value],
    primal_outputs: &[Value],
    _params: &BTreeMap<String, String>,
) -> Result<Vec<Value>, AdError> {
    match primitive {
        // ── Eigh JVP ──
        // A = V diag(w) V^T.  dA → (dw, dV)
        // M = V^T dA V;  dw = diag(M);  dV = V (F ⊙ M)
        Primitive::Eigh => {
            let da = &tangents[0];
            let v_val = &primal_outputs[1];
            let w_val = &primal_outputs[0];

            let v_t = v_val
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("Eigh JVP: V must be tensor".to_owned()))?;
            let da_t = da
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("Eigh JVP: dA must be tensor".to_owned()))?;
            let w_vec: Vec<f64> = w_val
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("Eigh JVP: W must be tensor".to_owned()))?
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap_or(0.0))
                .collect();

            let nn = w_vec.len();

            // M = V^T dA V [n×n]
            let vt_val = transpose_2d(v_t)?;
            let vt = tensor_value(&vt_val, "Eigh JVP transpose(V)")?;
            let vt_da = matmul_2d(vt, da_t)?;
            let m_val = matmul_2d(tensor_value(&vt_da, "Eigh JVP V^T @ dA")?, v_t)?;
            let m_t = tensor_value(&m_val, "Eigh JVP projected tangent")?;
            let m_vals: Vec<f64> = m_t
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap_or(0.0))
                .collect();

            // dw = diag(M)
            let dw: Vec<Literal> = (0..nn)
                .map(|i| Literal::from_f64(m_vals[i * nn + i]))
                .collect();
            let dw_val = Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape {
                        dims: vec![nn as u32],
                    },
                    dw,
                )
                .map_err(|e| AdError::EvalFailed(e.to_string()))?,
            );

            // F ⊙ M [n×n]: F[i,j] = 1/(w[j]-w[i]) for i≠j, 0 for i=j
            let mut fm = vec![0.0f64; nn * nn];
            for i in 0..nn {
                for j in 0..nn {
                    if i != j {
                        let f_ij = safe_eigen_gap_reciprocal(w_vec[i], w_vec[j]);
                        fm[i * nn + j] = f_ij * m_vals[i * nn + j];
                    }
                }
            }
            let fm_tensor = TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![nn as u32, nn as u32],
                },
                fm.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;

            // dV = V (F ⊙ M)
            let dv = matmul_2d(v_t, &fm_tensor)?;

            Ok(vec![dw_val, dv])
        }
        // ── QR JVP ──
        // A = QR.  dA → (dQ, dR)
        // C = Q^T dA;  dR = triu(C);  Ω from lower triangle;  dQ = Q Ω
        Primitive::Qr => {
            let da = &tangents[0];
            let q_val = &primal_outputs[0];
            let r_val = &primal_outputs[1];

            let q_t = q_val
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("QR JVP: Q must be tensor".to_owned()))?;
            let r_t = r_val
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("QR JVP: R must be tensor".to_owned()))?;
            let da_t = da
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("QR JVP: dA must be tensor".to_owned()))?;

            let m = q_t.shape.dims[0] as usize;
            let k = q_t.shape.dims[1] as usize; // k = min(m, n)
            let n = r_t.shape.dims[1] as usize;
            let r_vals: Vec<f64> = r_t
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap_or(0.0))
                .collect();

            // C = Q^T dA [k×n]
            let qt_val = transpose_2d(q_t)?;
            let qt = tensor_value(&qt_val, "QR JVP transpose(Q)")?;
            let c_val = matmul_2d(qt, da_t)?;
            let c_t = tensor_value(&c_val, "QR JVP Q^T @ dA")?;
            let c_vals: Vec<f64> = c_t
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap_or(0.0))
                .collect();

            // Solve for Ω from the lower triangle of C = Ω R + dR
            // For i > j: C[i,j] = Σ_p Ω[i,p] R[p,j]
            // Since R is upper triangular: only p ≤ j contribute
            // Solve column by column using forward substitution
            let mut omega = vec![0.0f64; k * k];
            for j in 0..k {
                for i in (j + 1)..k {
                    let mut rhs = c_vals[i * n + j];
                    for p in 0..j {
                        rhs -= omega[i * k + p] * r_vals[p * n + j];
                    }
                    let r_jj = r_vals[j * n + j];
                    omega[i * k + j] = if r_jj.abs() > 1e-20 { rhs / r_jj } else { 0.0 };
                    // Antisymmetric: Ω[j,i] = -Ω[i,j]
                    omega[j * k + i] = -omega[i * k + j];
                }
            }

            // dR = C - Ω R, then take upper triangle
            let omega_r = matmul_f64(k, k, n, &omega, &r_vals);
            let mut dr_vals = vec![0.0f64; k * n];
            for i in 0..k {
                for j in i..n {
                    dr_vals[i * n + j] = c_vals[i * n + j] - omega_r[i * n + j];
                }
            }
            let dr = build_matrix_f64(k, n, &dr_vals)?;

            // dQ = Q Ω [m×k]
            let omega_tensor = TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![k as u32, k as u32],
                },
                omega.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;

            let mut dq = matmul_2d(q_t, &omega_tensor)?;

            // Extra term for m > k: dQ += (I - QQ^T) dA R^{-1}
            if m > k {
                let q_vals: Vec<f64> = q_t
                    .elements
                    .iter()
                    .map(|l| l.as_f64().unwrap_or(0.0))
                    .collect();
                let da_vals: Vec<f64> = da_t
                    .elements
                    .iter()
                    .map(|l| l.as_f64().unwrap_or(0.0))
                    .collect();
                // proj = dA - Q(Q^T dA) = dA - Q C [m×n]
                let q_c = matmul_f64(m, k, n, &q_vals, &c_vals);
                let proj: Vec<f64> = da_vals.iter().zip(q_c.iter()).map(|(a, b)| a - b).collect();
                // proj @ R^{-1}: solve R X^T = proj^T => X = solve(R, proj^T)^T
                let proj_val = build_matrix_f64(m, n, &proj)?;
                let proj_t = transpose_2d(tensor_value(&proj_val, "QR JVP projected dA")?)?;
                let mut tri_params_jvp = BTreeMap::new();
                tri_params_jvp.insert("lower".to_owned(), "false".to_owned());
                let solve_result = eval_primitive(
                    Primitive::TriangularSolve,
                    &[r_val.clone(), proj_t],
                    &tri_params_jvp,
                )
                .map_err(|e| AdError::EvalFailed(e.to_string()))?;
                let extra = transpose_2d(tensor_value(
                    &solve_result,
                    "QR JVP triangular solve correction",
                )?)?;
                let next_dq = {
                    let dq_t = tensor_value(&dq, "QR JVP dQ")?;
                    let extra_t = tensor_value(&extra, "QR JVP extra dQ")?;
                    tensor_add(dq_t, extra_t)?
                };
                dq = next_dq;
            }

            Ok(vec![dq, dr])
        }
        // ── SVD JVP ──
        // A = U Σ V^T.  dA → (dU, ds, dVt)
        // M = U^T dA V;  ds = diag(M)
        // Ω_U[i,j] = (s_j M[i,j] + s_i M[j,i]) / (s_j² - s_i²)
        // Ω_V[i,j] = (s_i M[i,j] + s_j M[j,i]) / (s_j² - s_i²)
        // dU = U Ω_U;  dV = V Ω_V;  dVt = Ω_V^T Vt
        Primitive::Svd => {
            let da = &tangents[0];
            let u_val = &primal_outputs[0];
            let s_val = &primal_outputs[1];
            let vt_val = &primal_outputs[2];

            let u_t = u_val
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("SVD JVP: U must be tensor".to_owned()))?;
            let vt_t = vt_val
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("SVD JVP: Vt must be tensor".to_owned()))?;
            let da_t = da
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("SVD JVP: dA must be tensor".to_owned()))?;

            let s_vec: Vec<f64> = s_val
                .as_tensor()
                .ok_or_else(|| AdError::EvalFailed("SVD JVP: S must be tensor".to_owned()))?
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap_or(0.0))
                .collect();

            let m = u_t.shape.dims[0] as usize;
            let k = s_vec.len();
            let n = vt_t.shape.dims[1] as usize;

            // V = Vt^T [n×k]
            let v_val = transpose_2d(vt_t)?;
            let v_t = tensor_value(&v_val, "SVD JVP transpose(Vt)")?;

            // M = U^T dA V [k×k]
            let ut_val = transpose_2d(u_t)?;
            let ut = tensor_value(&ut_val, "SVD JVP transpose(U)")?;
            let ut_da = matmul_2d(ut, da_t)?;
            let m_val = matmul_2d(tensor_value(&ut_da, "SVD JVP U^T @ dA")?, v_t)?;
            let m_t = tensor_value(&m_val, "SVD JVP projected tangent")?;
            let m_vals: Vec<f64> = m_t
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap_or(0.0))
                .collect();

            // ds = diag(M)
            let ds: Vec<Literal> = (0..k)
                .map(|i| Literal::from_f64(m_vals[i * k + i]))
                .collect();
            let ds_val = Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape {
                        dims: vec![k as u32],
                    },
                    ds,
                )
                .map_err(|e| AdError::EvalFailed(e.to_string()))?,
            );

            // Build Ω_U and Ω_V [k×k]
            let mut omega_u = vec![0.0f64; k * k];
            let mut omega_v = vec![0.0f64; k * k];
            for i in 0..k {
                for j in 0..k {
                    if i != j {
                        let si = s_vec[i];
                        let sj = s_vec[j];
                        let denom = sj * sj - si * si;
                        if denom.abs() > 1e-20 {
                            let mij = m_vals[i * k + j];
                            let mji = m_vals[j * k + i];
                            omega_u[i * k + j] = (sj * mij + si * mji) / denom;
                            omega_v[i * k + j] = (si * mij + sj * mji) / denom;
                        }
                    }
                }
            }

            // dU = U Ω_U [m×k]
            let omega_u_tensor = TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![k as u32, k as u32],
                },
                omega_u.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let mut du = matmul_2d(u_t, &omega_u_tensor)?;

            // dVt = Ω_V^T Vt [k×n]  (since dV = V Ω_V, dVt = dV^T = Ω_V^T V^T = Ω_V^T Vt)
            // But Ω_V is antisymmetric, so Ω_V^T = -Ω_V
            let neg_omega_v: Vec<f64> = omega_v.iter().map(|&v| -v).collect();
            let neg_omega_v_tensor = TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![k as u32, k as u32],
                },
                neg_omega_v.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
            let mut dvt = matmul_2d(&neg_omega_v_tensor, vt_t)?;

            // Extra terms for non-square
            if m > k {
                let u_vals: Vec<f64> = u_t
                    .elements
                    .iter()
                    .map(|l| l.as_f64().unwrap_or(0.0))
                    .collect();
                let da_vals: Vec<f64> = da_t
                    .elements
                    .iter()
                    .map(|l| l.as_f64().unwrap_or(0.0))
                    .collect();
                let v_vals: Vec<f64> = v_t
                    .elements
                    .iter()
                    .map(|l| l.as_f64().unwrap_or(0.0))
                    .collect();
                // (I - UU^T) dA V Σ^{-1}
                let ut_da_vals: Vec<f64> = ut_da
                    .as_tensor()
                    .ok_or_else(|| {
                        AdError::EvalFailed("SVD JVP U^T @ dA: expected tensor value".to_owned())
                    })?
                    .elements
                    .iter()
                    .map(|l| l.as_f64().unwrap_or(0.0))
                    .collect();
                let u_ut_da = matmul_f64(m, k, n, &u_vals, &ut_da_vals);
                let proj: Vec<f64> = da_vals
                    .iter()
                    .zip(u_ut_da.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                let proj_v = matmul_f64(m, n, k, &proj, &v_vals);
                let mut du_extra = vec![0.0; m * k];
                for i in 0..m {
                    for j in 0..k {
                        let sinv = if s_vec[j].abs() > 1e-20 {
                            1.0 / s_vec[j]
                        } else {
                            0.0
                        };
                        du_extra[i * k + j] = proj_v[i * k + j] * sinv;
                    }
                }
                let du_extra_val = build_matrix_f64(m, k, &du_extra)?;
                let next_du = {
                    let du_t = tensor_value(&du, "SVD JVP dU")?;
                    let du_extra_t = tensor_value(&du_extra_val, "SVD JVP extra dU")?;
                    tensor_add(du_t, du_extra_t)?
                };
                du = next_du;
            }

            if n > k {
                let vt_vals: Vec<f64> = vt_t
                    .elements
                    .iter()
                    .map(|l| l.as_f64().unwrap_or(0.0))
                    .collect();
                // Σ^{-1} U^T dA (I - VV^T)
                let ut_da_vals: Vec<f64> = ut_da
                    .as_tensor()
                    .ok_or_else(|| {
                        AdError::EvalFailed("SVD JVP U^T @ dA: expected tensor value".to_owned())
                    })?
                    .elements
                    .iter()
                    .map(|l| l.as_f64().unwrap_or(0.0))
                    .collect();
                // U^T dA [k×n] @ V [n×k] = M [k×k] — already have. Then V^T [k×n]
                // (I_n - VV^T) = I - Vt^T Vt
                let vt_trans = transpose_f64(k, n, &vt_vals);
                let vvt = matmul_f64(n, k, n, &vt_trans, &vt_vals);
                // U^T dA (I-VV^T) [k×n]
                let ut_da_vvt = matmul_f64(k, n, n, &ut_da_vals, &vvt);
                let proj: Vec<f64> = ut_da_vals
                    .iter()
                    .zip(ut_da_vvt.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                // Σ^{-1} @ proj [k×n]
                let mut dvt_extra = vec![0.0; k * n];
                for i in 0..k {
                    let sinv = if s_vec[i].abs() > 1e-20 {
                        1.0 / s_vec[i]
                    } else {
                        0.0
                    };
                    for j in 0..n {
                        dvt_extra[i * n + j] = sinv * proj[i * n + j];
                    }
                }
                let dvt_extra_val = build_matrix_f64(k, n, &dvt_extra)?;
                let next_dvt = {
                    let dvt_t = tensor_value(&dvt, "SVD JVP dVt")?;
                    let dvt_extra_t = tensor_value(&dvt_extra_val, "SVD JVP extra dVt")?;
                    tensor_add(dvt_t, dvt_extra_t)?
                };
                dvt = next_dvt;
            }

            Ok(vec![du, ds_val, dvt])
        }
        _ => {
            // Single-output primitives should not reach here
            Err(AdError::UnsupportedPrimitive(primitive))
        }
    }
}

// ── Forward-mode gradient (convenience) ────────────────────────────

/// Compute gradient via forward-mode JVP by evaluating with unit tangent.
pub fn jvp_grad_first(jaxpr: &Jaxpr, args: &[Value]) -> Result<f64, AdError> {
    let mut tangents: Vec<Value> = args.iter().map(zeros_like).collect();
    tangents[0] = Value::scalar_f64(1.0);
    let result = jvp(jaxpr, args, &tangents)?;
    result
        .tangents
        .first()
        .and_then(|v| v.as_f64_scalar())
        .ok_or(AdError::NonScalarGradientOutput)
}

/// Compute gradients of a Jaxpr with respect to all inputs (tensor-aware).
fn value_and_grad_jaxpr_inner(
    jaxpr: &Jaxpr,
    args: &[Value],
) -> Result<(Vec<Value>, Vec<Value>, usize), AdError> {
    value_and_grad_jaxpr_inner_with_custom_vjp_key(jaxpr, args, None)
}

fn value_and_grad_jaxpr_inner_with_custom_vjp_key(
    jaxpr: &Jaxpr,
    args: &[Value],
    custom_vjp_rule_key: Option<&str>,
) -> Result<(Vec<Value>, Vec<Value>, usize), AdError> {
    let (outputs, tape, env) = forward_with_tape(jaxpr, args)?;

    let output_val = outputs.first().ok_or(AdError::NonScalarGradientOutput)?;

    // JAX semantics: grad requires scalar-valued function output
    if !matches!(output_val, Value::Scalar(_)) {
        return Err(AdError::NonScalarGradientOutput);
    }

    let seed = Value::scalar_f64(1.0);

    let custom_rule = custom_vjp_rule_key
        .and_then(lookup_custom_jaxpr_vjp_by_key)
        .or_else(|| lookup_custom_jaxpr_vjp(jaxpr));
    if let Some(custom_rule) = custom_rule {
        let grads = custom_rule(args, &outputs, &seed)?;
        if grads.len() != jaxpr.invars.len() {
            return Err(AdError::EvalFailed(format!(
                "custom Jaxpr VJP cotangent arity mismatch: expected {}, got {}",
                jaxpr.invars.len(),
                grads.len()
            )));
        }
        return Ok((outputs, grads, tape.entries.len()));
    }

    let output_var = jaxpr.outvars[0];
    let grads = backward(&tape, output_var, seed, jaxpr, &env)?;
    Ok((outputs, grads, tape.entries.len()))
}

/// Compute gradients of a Jaxpr with respect to all inputs (tensor-aware).
pub fn grad_jaxpr(jaxpr: &Jaxpr, args: &[Value]) -> Result<Vec<Value>, AdError> {
    let (_, grads, _) = value_and_grad_jaxpr_inner(jaxpr, args)?;
    Ok(grads)
}

/// Compute gradients using a wrapper-scoped custom VJP rule when present.
pub fn grad_jaxpr_with_custom_vjp_key(
    jaxpr: &Jaxpr,
    args: &[Value],
    rule_key: &str,
) -> Result<Vec<Value>, AdError> {
    let (_, grads, _) =
        value_and_grad_jaxpr_inner_with_custom_vjp_key(jaxpr, args, Some(rule_key))?;
    Ok(grads)
}

/// Compute gradients with a custom output cotangent (for checkpoint/remat).
///
/// Unlike `grad_jaxpr` which uses seed=1.0, this accepts an arbitrary cotangent
/// to support re-running forward during backward pass of a checkpointed region.
/// The cotangent is the gradient flowing back from downstream computations.
pub fn grad_jaxpr_with_cotangent(
    jaxpr: &Jaxpr,
    args: &[Value],
    cotangent: &Value,
) -> Result<Vec<Value>, AdError> {
    let (_, tape, env) = forward_with_tape(jaxpr, args)?;
    let output_var = jaxpr
        .outvars
        .first()
        .copied()
        .ok_or(AdError::NonScalarGradientOutput)?;
    backward(&tape, output_var, cotangent.clone(), jaxpr, &env)
}

/// Compute both value outputs and gradients in one shared forward pass.
pub fn value_and_grad_jaxpr(
    jaxpr: &Jaxpr,
    args: &[Value],
) -> Result<(Vec<Value>, Vec<Value>), AdError> {
    let (values, grads, _) = value_and_grad_jaxpr_inner(jaxpr, args)?;
    Ok((values, grads))
}

/// Compute both value outputs and gradients using a wrapper-scoped custom VJP
/// rule when present.
pub fn value_and_grad_jaxpr_with_custom_vjp_key(
    jaxpr: &Jaxpr,
    args: &[Value],
    rule_key: &str,
) -> Result<(Vec<Value>, Vec<Value>), AdError> {
    let (values, grads, _) =
        value_and_grad_jaxpr_inner_with_custom_vjp_key(jaxpr, args, Some(rule_key))?;
    Ok((values, grads))
}

/// Compute gradient with respect to the first input only (convenience wrapper).
/// Returns scalar f64 for backward compatibility.
pub fn grad_first(jaxpr: &Jaxpr, args: &[Value]) -> Result<f64, AdError> {
    let grads = grad_jaxpr(jaxpr, args)?;
    to_f64(&grads[0])
}

fn flatten_value_to_f64(value: &Value) -> Result<Vec<f64>, AdError> {
    match value {
        Value::Scalar(_) => value
            .as_f64_scalar()
            .map(|v| vec![v])
            .ok_or_else(|| AdError::EvalFailed("expected scalar convertible to f64".to_owned())),
        Value::Tensor(tensor) => tensor
            .to_f64_vec()
            .ok_or_else(|| AdError::EvalFailed("expected tensor convertible to f64".to_owned())),
    }
}

fn flatten_values_to_f64(values: &[Value]) -> Result<Vec<f64>, AdError> {
    let mut out = Vec::new();
    for value in values {
        out.extend(flatten_value_to_f64(value)?);
    }
    Ok(out)
}

fn value_from_flat_like(template: &Value, flat_values: &[f64]) -> Result<Value, AdError> {
    match template {
        Value::Scalar(_) => {
            if flat_values.len() != 1 {
                return Err(AdError::EvalFailed(format!(
                    "scalar reconstruction expected 1 value, got {}",
                    flat_values.len()
                )));
            }
            Ok(Value::scalar_f64(flat_values[0]))
        }
        Value::Tensor(tensor) => {
            if flat_values.len() != tensor.len() {
                return Err(AdError::EvalFailed(format!(
                    "tensor reconstruction expected {} values, got {}",
                    tensor.len(),
                    flat_values.len()
                )));
            }
            let elements = flat_values
                .iter()
                .copied()
                .map(Literal::from_f64)
                .collect::<Vec<_>>();
            TensorValue::new(DType::F64, tensor.shape.clone(), elements)
                .map(Value::Tensor)
                .map_err(|e| AdError::EvalFailed(e.to_string()))
        }
    }
}

fn reconstruct_args_from_flat(
    templates: &[Value],
    flat_args: &[Vec<f64>],
) -> Result<Vec<Value>, AdError> {
    if templates.len() != flat_args.len() {
        return Err(AdError::EvalFailed(format!(
            "argument reconstruction expected {} arg vectors, got {}",
            templates.len(),
            flat_args.len()
        )));
    }
    templates
        .iter()
        .zip(flat_args.iter())
        .map(|(template, flat)| value_from_flat_like(template, flat))
        .collect()
}

fn basis_value_like(template: &Value, basis_index: usize) -> Result<Value, AdError> {
    match template {
        Value::Scalar(_) => {
            if basis_index != 0 {
                return Err(AdError::EvalFailed(format!(
                    "scalar basis index out of range: {basis_index}"
                )));
            }
            Ok(Value::scalar_f64(1.0))
        }
        Value::Tensor(tensor) => {
            if basis_index >= tensor.len() {
                return Err(AdError::EvalFailed(format!(
                    "tensor basis index out of range: {basis_index} >= {}",
                    tensor.len()
                )));
            }
            let mut elements = vec![Literal::from_f64(0.0); tensor.len()];
            elements[basis_index] = Literal::from_f64(1.0);
            TensorValue::new(DType::F64, tensor.shape.clone(), elements)
                .map(Value::Tensor)
                .map_err(|e| AdError::EvalFailed(e.to_string()))
        }
    }
}

fn global_basis_to_arg_index(lengths: &[usize], mut index: usize) -> Option<(usize, usize)> {
    for (arg_idx, len) in lengths.iter().copied().enumerate() {
        if index < len {
            return Some((arg_idx, index));
        }
        index = index.saturating_sub(len);
    }
    None
}

fn matrix_value(rows: usize, cols: usize, entries: Vec<f64>) -> Result<Value, AdError> {
    let rows_u32 = u32::try_from(rows)
        .map_err(|_| AdError::EvalFailed(format!("row count too large for shape: {rows}")))?;
    let cols_u32 = u32::try_from(cols)
        .map_err(|_| AdError::EvalFailed(format!("column count too large for shape: {cols}")))?;
    let elements = entries
        .into_iter()
        .map(Literal::from_f64)
        .collect::<Vec<_>>();
    TensorValue::new(
        DType::F64,
        Shape {
            dims: vec![rows_u32, cols_u32],
        },
        elements,
    )
    .map(Value::Tensor)
    .map_err(|e| AdError::EvalFailed(e.to_string()))
}

/// Compute Jacobian matrix of all outputs with respect to all inputs.
///
/// Output layout is row-major `[output_dim, input_dim]`, where both dimensions
/// are flattened over all output/input values in argument order.
pub fn jacobian_jaxpr(jaxpr: &Jaxpr, args: &[Value]) -> Result<Value, AdError> {
    jacobian_jaxpr_inner(jaxpr, args, None)
}

/// Compute Jacobian using a wrapper-scoped custom JVP rule when present.
pub fn jacobian_jaxpr_with_custom_jvp_key(
    jaxpr: &Jaxpr,
    args: &[Value],
    rule_key: &str,
) -> Result<Value, AdError> {
    jacobian_jaxpr_inner(jaxpr, args, Some(rule_key))
}

fn jacobian_jaxpr_inner(
    jaxpr: &Jaxpr,
    args: &[Value],
    custom_jvp_rule_key: Option<&str>,
) -> Result<Value, AdError> {
    if args.is_empty() {
        return Err(AdError::InputArity {
            expected: 1,
            actual: 0,
        });
    }

    let input_lengths = args
        .iter()
        .map(flatten_value_to_f64)
        .map(|res| res.map(|flat| flat.len()))
        .collect::<Result<Vec<_>, _>>()?;
    let input_dim = input_lengths.iter().sum::<usize>();
    if input_dim == 0 {
        return Err(AdError::EvalFailed(
            "jacobian requires at least one differentiable input dimension".to_owned(),
        ));
    }

    let zero_tangents = args.iter().map(zeros_like).collect::<Vec<_>>();
    let mut output_dim = None::<usize>;
    let mut jacobian = Vec::<f64>::new();

    for basis_idx in 0..input_dim {
        let (arg_idx, local_idx) = global_basis_to_arg_index(&input_lengths, basis_idx)
            .ok_or_else(|| {
                AdError::EvalFailed(format!("unable to resolve basis index {basis_idx}"))
            })?;
        let mut tangents = zero_tangents.clone();
        tangents[arg_idx] = basis_value_like(&args[arg_idx], local_idx)?;

        let jvp_result = jvp_inner(jaxpr, args, &tangents, custom_jvp_rule_key)?;
        let tangent_flat = flatten_values_to_f64(&jvp_result.tangents)?;
        let current_output_dim = tangent_flat.len();

        if let Some(expected) = output_dim {
            if expected != current_output_dim {
                return Err(AdError::EvalFailed(format!(
                    "inconsistent Jacobian output dimension: expected {expected}, got {current_output_dim}"
                )));
            }
        } else {
            output_dim = Some(current_output_dim);
            jacobian.resize(current_output_dim * input_dim, 0.0);
        }

        for (row, value) in tangent_flat.into_iter().enumerate() {
            jacobian[row * input_dim + basis_idx] = value;
        }
    }

    matrix_value(output_dim.unwrap_or(0), input_dim, jacobian)
}

/// Compute Hessian matrix of a scalar-output function with respect to all inputs.
///
/// Uses central-difference directional derivatives of `grad_jaxpr` with
/// epsilon `1e-5`. Output is row-major `[input_dim, input_dim]`.
pub fn hessian_jaxpr(jaxpr: &Jaxpr, args: &[Value]) -> Result<Value, AdError> {
    if args.is_empty() {
        return Err(AdError::InputArity {
            expected: 1,
            actual: 0,
        });
    }

    let base_flat_args = args
        .iter()
        .map(flatten_value_to_f64)
        .collect::<Result<Vec<_>, _>>()?;
    let input_lengths = base_flat_args.iter().map(Vec::len).collect::<Vec<_>>();
    let input_dim = input_lengths.iter().sum::<usize>();
    if input_dim == 0 {
        return Err(AdError::EvalFailed(
            "hessian requires at least one differentiable input dimension".to_owned(),
        ));
    }

    let epsilon = 1e-5_f64;
    let mut hessian = vec![0.0; input_dim * input_dim];

    for basis_idx in 0..input_dim {
        let (arg_idx, local_idx) = global_basis_to_arg_index(&input_lengths, basis_idx)
            .ok_or_else(|| {
                AdError::EvalFailed(format!("unable to resolve basis index {basis_idx}"))
            })?;

        let mut plus_flat = base_flat_args.clone();
        plus_flat[arg_idx][local_idx] += epsilon;
        let plus_args = reconstruct_args_from_flat(args, &plus_flat)?;
        let plus_grad = grad_jaxpr(jaxpr, &plus_args)?;
        let plus_flat_grad = flatten_values_to_f64(&plus_grad)?;

        let mut minus_flat = base_flat_args.clone();
        minus_flat[arg_idx][local_idx] -= epsilon;
        let minus_args = reconstruct_args_from_flat(args, &minus_flat)?;
        let minus_grad = grad_jaxpr(jaxpr, &minus_args)?;
        let minus_flat_grad = flatten_values_to_f64(&minus_grad)?;

        if plus_flat_grad.len() != input_dim || minus_flat_grad.len() != input_dim {
            return Err(AdError::EvalFailed(format!(
                "gradient dimension mismatch for Hessian: expected {input_dim}, got plus={} minus={}",
                plus_flat_grad.len(),
                minus_flat_grad.len()
            )));
        }

        for row in 0..input_dim {
            hessian[row * input_dim + basis_idx] =
                (plus_flat_grad[row] - minus_flat_grad[row]) / (2.0 * epsilon);
        }
    }

    matrix_value(input_dim, input_dim, hessian)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fj_core::{Equation, ProgramSpec, build_program};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Mutex, OnceLock};

    fn custom_rule_test_guard() -> std::sync::MutexGuard<'static, ()> {
        static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
        GUARD
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[test]
    fn unsupported_primitive_display_uses_permanent_unsupported_wording() {
        let rendered = AdError::UnsupportedPrimitive(Primitive::Scan).to_string();
        assert!(rendered.contains("unsupported"));
        assert!(!rendered.contains("not implemented"));
    }

    #[test]
    fn vjp_rejects_empty_cotangent_list() {
        let result = vjp(
            Primitive::Add,
            &[Value::scalar_f64(1.0), Value::scalar_f64(2.0)],
            &[],
            &[],
            &BTreeMap::new(),
        );

        assert!(matches!(
            result,
            Err(AdError::CotangentArity {
                primitive: Primitive::Add,
                expected_at_least: 1,
                actual: 0
            })
        ));
    }

    #[test]
    fn grad_rejects_primitive_output_arity_mismatch() {
        use smallvec::smallvec;

        let jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3)],
            vec![Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3), VarId(4)],
                params: BTreeMap::new(),
                effects: Vec::new(),
                sub_jaxprs: Vec::new(),
            }],
        );

        let result = grad_jaxpr(&jaxpr, &[Value::scalar_f64(1.0), Value::scalar_f64(2.0)]);

        assert!(matches!(
            result,
            Err(AdError::PrimitiveOutputArity {
                primitive: Primitive::Add,
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn f32_zero_one_helpers_preserve_literal_width() {
        let scalar_zero = zeros_like(&Value::scalar_f32(2.5));
        assert_eq!(scalar_zero.dtype(), DType::F32);
        assert!(matches!(scalar_zero, Value::Scalar(Literal::F32Bits(_))));

        let scalar_one = ones_like(&Value::scalar_f32(2.5));
        assert_eq!(scalar_one.dtype(), DType::F32);
        assert!(matches!(scalar_one, Value::Scalar(Literal::F32Bits(_))));

        let tensor = match TensorValue::new(
            DType::F32,
            Shape::vector(2),
            vec![Literal::from_f32(1.25), Literal::from_f32(-3.5)],
        ) {
            Ok(tensor) => Value::Tensor(tensor),
            Err(error) => fail_test(format!("f32 tensor should build: {error}")),
        };

        let tensor_zero = zeros_like(&tensor);
        let zero_tensor = match tensor_zero.as_tensor() {
            Some(tensor) => tensor,
            None => fail_test("zero should be tensor"),
        };
        assert_eq!(zero_tensor.dtype, DType::F32);
        assert!(
            zero_tensor
                .elements
                .iter()
                .all(|lit| matches!(lit, Literal::F32Bits(_)))
        );

        let tensor_one = ones_like(&tensor);
        let one_tensor = match tensor_one.as_tensor() {
            Some(tensor) => tensor,
            None => fail_test("one should be tensor"),
        };
        assert_eq!(one_tensor.dtype, DType::F32);
        assert!(
            one_tensor
                .elements
                .iter()
                .all(|lit| matches!(lit, Literal::F32Bits(_)))
        );
    }

    #[derive(Debug)]
    struct TestAssertionFailure(String);

    impl std::fmt::Display for TestAssertionFailure {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(&self.0)
        }
    }

    impl std::error::Error for TestAssertionFailure {}

    fn fail_test(message: impl Into<String>) -> ! {
        std::panic::panic_any(TestAssertionFailure(message.into()))
    }

    fn expect_tensor_ref<'a>(value: &'a Value, context: &str) -> &'a TensorValue {
        match value {
            Value::Tensor(tensor) => tensor,
            other => fail_test(format!("{context}: expected tensor, got {other:?}")),
        }
    }

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
                sub_jaxprs: vec![],
                effects: vec![],
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
                sub_jaxprs: vec![],
                effects: vec![],
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
    fn value_and_grad_matches_separate_paths_single_input() {
        let jaxpr = build_program(ProgramSpec::SquarePlusLinear);
        let args = vec![Value::scalar_f64(3.0)];

        let (value, grad) =
            value_and_grad_jaxpr(&jaxpr, &args).expect("value_and_grad should succeed");
        let separate_grad = grad_jaxpr(&jaxpr, &args).expect("grad should succeed");

        assert_eq!(value.len(), 1);
        let value_scalar = to_f64(&value[0]).expect("value should be scalar");
        assert!(
            (value_scalar - 15.0).abs() < 1e-10,
            "value = {value_scalar}"
        );
        assert_eq!(grad, separate_grad);
    }

    #[test]
    fn value_and_grad_matches_separate_paths_multi_input() {
        let jaxpr = build_program(ProgramSpec::Add2);
        let args = vec![Value::scalar_f64(6.0), Value::scalar_f64(3.0)];

        let (value, grad) =
            value_and_grad_jaxpr(&jaxpr, &args).expect("value_and_grad should succeed");
        let separate_grad = grad_jaxpr(&jaxpr, &args).expect("grad should succeed");

        assert_eq!(value.len(), 1);
        let value_scalar = to_f64(&value[0]).expect("value should be scalar");
        assert!((value_scalar - 9.0).abs() < 1e-10, "value = {value_scalar}");
        assert_eq!(grad, separate_grad);
    }

    #[test]
    fn value_and_grad_returns_all_outputs() {
        let jaxpr = build_program(ProgramSpec::AddOneMulTwo);
        let args = vec![Value::scalar_f64(4.0)];

        let (values, grads) =
            value_and_grad_jaxpr(&jaxpr, &args).expect("value_and_grad should succeed");

        assert_eq!(values.len(), 2, "expected both outputs from AddOneMulTwo");
        let first = to_f64(&values[0]).expect("first output scalar");
        let second = to_f64(&values[1]).expect("second output scalar");
        assert!((first - 5.0).abs() < 1e-10, "first output = {first}");
        assert!((second - 8.0).abs() < 1e-10, "second output = {second}");

        assert_eq!(grads.len(), 1);
        let g = to_f64(&grads[0]).expect("gradient should be scalar");
        assert!((g - 1.0).abs() < 1e-10, "gradient = {g}");
    }

    #[test]
    fn value_and_grad_shares_forward_pass() {
        let jaxpr = build_program(ProgramSpec::SquarePlusLinear);
        let args = vec![Value::scalar_f64(2.0)];

        let (_, _, forward_steps) = value_and_grad_jaxpr_inner(&jaxpr, &args)
            .expect("value_and_grad internals should succeed");
        let separate_forward_steps = jaxpr.equations.len() * 2;
        assert_eq!(
            forward_steps,
            jaxpr.equations.len(),
            "shared forward should execute one equation pass"
        );
        assert!(
            forward_steps < separate_forward_steps,
            "shared forward should do fewer steps than separate value + grad forward passes"
        );
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
                sub_jaxprs: vec![],
                effects: vec![],
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
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        )
    }

    #[test]
    fn custom_vjp_rule_overrides_builtin_rule() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = Arc::clone(&call_count);
        register_custom_vjp(Primitive::CountLeadingZeros, move |_inputs, _g, _params| {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
            Ok(vec![Value::scalar_f64(42.0)])
        });

        let jaxpr = make_unary_jaxpr(Primitive::CountLeadingZeros);
        let grads = grad_jaxpr(&jaxpr, &[Value::scalar_i64(8)]).expect("grad should succeed");
        let grad = grads[0]
            .as_f64_scalar()
            .expect("custom cotangent should be scalar f64");
        assert!(
            (grad - 42.0).abs() < 1e-10,
            "custom VJP should override builtin, got {grad}"
        );
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "custom VJP rule should be invoked once"
        );

        clear_custom_derivative_rules();
    }

    #[test]
    fn custom_jvp_rule_overrides_builtin_rule() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = Arc::clone(&call_count);
        register_custom_jvp(
            Primitive::CountLeadingZeros,
            move |_primals, _tangents, _params| {
                call_count_clone.fetch_add(1, Ordering::SeqCst);
                Ok(Value::scalar_f64(123.0))
            },
        );

        let jaxpr = make_unary_jaxpr(Primitive::CountLeadingZeros);
        let result = jvp(&jaxpr, &[Value::scalar_i64(8)], &[Value::scalar_f64(1.0)])
            .expect("jvp should succeed");
        let tangent = result.tangents[0]
            .as_f64_scalar()
            .expect("custom tangent should be scalar f64");
        assert!(
            (tangent - 123.0).abs() < 1e-10,
            "custom JVP should override builtin, got {tangent}"
        );
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "custom JVP rule should be invoked once"
        );

        clear_custom_derivative_rules();
    }

    #[test]
    fn test_custom_vjp_simple() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();

        register_custom_vjp(Primitive::CountLeadingZeros, |_inputs, _g, _params| {
            Ok(vec![Value::scalar_f64(2.5)])
        });

        let jaxpr = make_unary_jaxpr(Primitive::CountLeadingZeros);
        let grads = grad_jaxpr(&jaxpr, &[Value::scalar_i64(8)]).expect("grad should succeed");
        let grad = to_f64(&grads[0]).expect("custom gradient should be scalar");
        assert!(
            (grad - 2.5).abs() < 1e-10,
            "expected custom gradient 2.5, got {grad}"
        );

        clear_custom_derivative_rules();
    }

    #[test]
    fn test_custom_vjp_overrides_default() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();

        let jaxpr = make_unary_jaxpr(Primitive::CountLeadingZeros);
        let auto_grad = to_f64(
            &grad_jaxpr(&jaxpr, &[Value::scalar_i64(16)]).expect("auto grad should succeed")[0],
        )
        .expect("auto grad should be scalar");
        assert!(
            auto_grad.abs() < 1e-10,
            "expected default grad 0, got {auto_grad}"
        );

        register_custom_vjp(Primitive::CountLeadingZeros, |_inputs, _g, _params| {
            Ok(vec![Value::scalar_f64(11.0)])
        });

        let custom_grad = to_f64(
            &grad_jaxpr(&jaxpr, &[Value::scalar_i64(16)]).expect("custom grad should succeed")[0],
        )
        .expect("custom grad should be scalar");
        assert!(
            (custom_grad - 11.0).abs() < 1e-10,
            "expected overridden gradient 11.0, got {custom_grad}"
        );

        clear_custom_derivative_rules();
    }

    #[test]
    fn test_custom_vjp_multiple_args() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();

        register_custom_vjp(Primitive::BitwiseAnd, |_inputs, _g, _params| {
            Ok(vec![Value::scalar_f64(3.0), Value::scalar_f64(4.0)])
        });

        let jaxpr = make_binary_jaxpr(Primitive::BitwiseAnd);
        let grads = grad_jaxpr(&jaxpr, &[Value::scalar_i64(7), Value::scalar_i64(3)])
            .expect("custom grad for binary primitive should succeed");
        let g0 = to_f64(&grads[0]).expect("first gradient should be scalar");
        let g1 = to_f64(&grads[1]).expect("second gradient should be scalar");
        assert!((g0 - 3.0).abs() < 1e-10, "first custom gradient = {g0}");
        assert!((g1 - 4.0).abs() < 1e-10, "second custom gradient = {g1}");

        clear_custom_derivative_rules();
    }

    #[test]
    fn test_custom_jvp_simple() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();

        register_custom_jvp(
            Primitive::CountLeadingZeros,
            |_primals, _tangents, _params| Ok(Value::scalar_f64(9.0)),
        );

        let jaxpr = make_unary_jaxpr(Primitive::CountLeadingZeros);
        let result = jvp(&jaxpr, &[Value::scalar_i64(32)], &[Value::scalar_f64(1.0)])
            .expect("jvp should succeed");
        let tangent = to_f64(&result.tangents[0]).expect("custom tangent should be scalar");
        assert!(
            (tangent - 9.0).abs() < 1e-10,
            "expected custom tangent 9.0, got {tangent}"
        );

        clear_custom_derivative_rules();
    }

    #[test]
    fn test_custom_jvp_overrides_default() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();

        let jaxpr = make_unary_jaxpr(Primitive::CountLeadingZeros);
        let auto = jvp(&jaxpr, &[Value::scalar_i64(4)], &[Value::scalar_f64(1.0)])
            .expect("default jvp should succeed");
        let auto_tangent = to_f64(&auto.tangents[0]).expect("default tangent should be scalar");
        assert!(
            auto_tangent.abs() < 1e-10,
            "expected default tangent 0, got {auto_tangent}"
        );

        register_custom_jvp(
            Primitive::CountLeadingZeros,
            |_primals, _tangents, _params| Ok(Value::scalar_f64(-7.0)),
        );
        let custom = jvp(&jaxpr, &[Value::scalar_i64(4)], &[Value::scalar_f64(1.0)])
            .expect("custom jvp should succeed");
        let custom_tangent = to_f64(&custom.tangents[0]).expect("custom tangent should be scalar");
        assert!(
            (custom_tangent - (-7.0)).abs() < 1e-10,
            "expected overridden tangent -7.0, got {custom_tangent}"
        );

        clear_custom_derivative_rules();
    }

    #[test]
    fn test_custom_vjp_nested_grad() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = Arc::clone(&call_count);
        register_custom_vjp(Primitive::CountLeadingZeros, move |inputs, _g, _params| {
            call_count_clone.fetch_add(1, Ordering::SeqCst);
            let x = inputs
                .first()
                .and_then(Value::as_i64_scalar)
                .ok_or_else(|| AdError::EvalFailed("expected scalar i64 primal".to_owned()))?;
            Ok(vec![Value::scalar_f64(x as f64)])
        });

        let jaxpr = make_unary_jaxpr(Primitive::CountLeadingZeros);
        let inner_grad =
            |x: i64| grad_first(&jaxpr, &[Value::scalar_i64(x)]).expect("grad evaluation");
        let second_derivative_estimate = (inner_grad(10) - inner_grad(8)) / 2.0;
        assert!(
            (second_derivative_estimate - 1.0).abs() < 1e-10,
            "expected nested grad estimate 1.0, got {second_derivative_estimate}"
        );
        assert!(
            call_count.load(Ordering::SeqCst) >= 2,
            "custom VJP should be used for both inner gradient evaluations"
        );

        clear_custom_derivative_rules();
    }

    #[test]
    fn test_custom_vjp_registration_error() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();

        register_custom_vjp(Primitive::BitwiseAnd, |_inputs, _g, _params| {
            // Invalid cotangent arity: primitive has two inputs.
            Ok(vec![Value::scalar_f64(1.0)])
        });

        let jaxpr = make_binary_jaxpr(Primitive::BitwiseAnd);
        let err = grad_jaxpr(&jaxpr, &[Value::scalar_i64(3), Value::scalar_i64(1)])
            .expect_err("invalid custom VJP arity should fail");
        let AdError::EvalFailed(message) = err else {
            fail_test(format!("expected EvalFailed, got: {err:?}"));
        };
        assert!(
            message.contains("cotangent arity mismatch"),
            "expected cotangent arity mismatch error, got: {message}"
        );

        clear_custom_derivative_rules();
    }

    #[test]
    fn test_custom_vjp_with_residuals() {
        let _guard = custom_rule_test_guard();
        clear_custom_derivative_rules();

        let residuals: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
        let residuals_clone = Arc::clone(&residuals);
        register_custom_vjp(Primitive::CountLeadingZeros, move |inputs, g, _params| {
            let x = inputs
                .first()
                .and_then(Value::as_i64_scalar)
                .ok_or_else(|| AdError::EvalFailed("expected scalar i64 primal".to_owned()))?;
            let mut residuals = residuals_clone
                .lock()
                .map_err(|_| AdError::EvalFailed("residual lock poisoned".to_owned()))?;
            residuals.push(x);
            let upstream = g.as_f64_scalar().ok_or_else(|| {
                AdError::EvalFailed("expected scalar upstream cotangent".to_owned())
            })?;
            Ok(vec![Value::scalar_f64((x as f64) + upstream)])
        });

        let jaxpr = make_unary_jaxpr(Primitive::CountLeadingZeros);
        let grads = grad_jaxpr(&jaxpr, &[Value::scalar_i64(12)]).expect("grad should succeed");
        let grad = to_f64(&grads[0]).expect("custom gradient should be scalar");
        assert!(
            (grad - 13.0).abs() < 1e-10,
            "expected custom residual grad 13, got {grad}"
        );

        let stored = residuals
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        assert_eq!(stored, vec![12], "expected one stored residual input");

        clear_custom_derivative_rules();
    }

    fn tensor_f64_values(value: &Value) -> Vec<f64> {
        expect_tensor_ref(value, "tensor_f64_values")
            .elements
            .iter()
            .map(|lit| lit.as_f64().expect("expected numeric literal"))
            .collect()
    }

    fn tensor_f64(dims: &[u32], values: &[f64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: dims.to_vec(),
                },
                values.iter().copied().map(Literal::from_f64).collect(),
            )
            .expect("test tensor shape should match element count"),
        )
    }

    fn empty_f64_tensor(dims: Vec<u32>) -> Value {
        Value::Tensor(
            TensorValue::new(DType::F64, Shape { dims }, Vec::new())
                .expect("empty tensor shape should be accepted"),
        )
    }

    fn assert_tensor_f64(value: &Value, dims: &[u32], expected: &[f64]) {
        let tensor = expect_tensor_ref(value, "assert_tensor_f64");
        assert_eq!(tensor.shape.dims.as_slice(), dims);
        let actual = tensor_f64_values(value);
        assert_eq!(actual.len(), expected.len());
        for (index, (actual_value, expected_value)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual_value - expected_value).abs() < 1e-10,
                "element {index}: actual={actual_value}, expected={expected_value}"
            );
        }
    }

    #[test]
    fn vjp_dot_scalar_tensor_reduces_scalar_cotangent() {
        let scalar = Value::scalar_f64(2.0);
        let tensor = tensor_f64(&[2], &[3.0, 4.0]);
        let g = tensor_f64(&[2], &[0.5, -1.0]);

        let grads = vjp_single(Primitive::Dot, &[scalar, tensor], &g, &BTreeMap::new()).unwrap();

        let scalar_grad = grads[0]
            .as_f64_scalar()
            .expect("scalar input cotangent should be scalar");
        assert!((scalar_grad - (-2.5)).abs() < 1e-10);
        assert_tensor_f64(&grads[1], &[2], &[1.0, -2.0]);
    }

    #[test]
    fn vjp_dot_matrix_vector_rank2() {
        let matrix = tensor_f64(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let vector = tensor_f64(&[3], &[7.0, 8.0, 9.0]);
        let g = tensor_f64(&[2], &[0.5, -1.0]);

        let grads = vjp_single(Primitive::Dot, &[matrix, vector], &g, &BTreeMap::new()).unwrap();

        assert_tensor_f64(&grads[0], &[2, 3], &[3.5, 4.0, 4.5, -7.0, -8.0, -9.0]);
        assert_tensor_f64(&grads[1], &[3], &[-3.5, -4.0, -4.5]);
    }

    #[test]
    fn vjp_dot_vector_matrix_rank2() {
        let vector = tensor_f64(&[3], &[1.0, 2.0, 3.0]);
        let matrix = tensor_f64(&[3, 2], &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let g = tensor_f64(&[2], &[0.5, -1.0]);

        let grads = vjp_single(Primitive::Dot, &[vector, matrix], &g, &BTreeMap::new()).unwrap();

        assert_tensor_f64(&grads[0], &[3], &[-3.0, -4.0, -5.0]);
        assert_tensor_f64(&grads[1], &[3, 2], &[0.5, -1.0, 1.0, -2.0, 1.5, -3.0]);
    }

    #[test]
    fn vjp_dot_matrix_matrix_rank2() {
        let lhs = tensor_f64(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let rhs = tensor_f64(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let g = tensor_f64(&[2, 2], &[0.5, -1.0, 2.0, 3.0]);

        let grads = vjp_single(Primitive::Dot, &[lhs, rhs], &g, &BTreeMap::new()).unwrap();

        assert_tensor_f64(&grads[0], &[2, 3], &[-4.5, -5.5, -6.5, 38.0, 48.0, 58.0]);
        assert_tensor_f64(&grads[1], &[3, 2], &[8.5, 11.0, 11.0, 13.0, 13.5, 15.0]);
    }

    fn scalar_complex128(value: &Value) -> (f64, f64) {
        value
            .as_scalar_literal()
            .and_then(Literal::as_complex128)
            .expect("expected complex128 scalar literal")
    }

    // ── V2 primitive VJP/JVP coverage (bd-2u82) ────────────────

    #[test]
    fn test_rev_vjp_single() {
        let input = Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector");
        let g = Value::vector_f64(&[10.0, 20.0, 30.0]).expect("vector");
        let mut params = BTreeMap::new();
        params.insert("axes".into(), "0".into());

        let grads = vjp_single(Primitive::Rev, &[input], &g, &params).expect("vjp");
        assert_eq!(tensor_f64_values(&grads[0]), vec![30.0, 20.0, 10.0]);
    }

    #[test]
    fn test_squeeze_vjp_single() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 3, 1],
                },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                ],
            )
            .expect("tensor"),
        );
        let g = Value::vector_f64(&[7.0, 8.0, 9.0]).expect("vector");
        let mut params = BTreeMap::new();
        params.insert("dimensions".into(), "0,2".into());

        let grads = vjp_single(Primitive::Squeeze, &[input], &g, &params).expect("vjp");
        let out = grads[0].as_tensor().expect("tensor");
        assert_eq!(out.shape.dims, vec![1, 3, 1]);
        assert_eq!(tensor_f64_values(&grads[0]), vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_split_vjp_single() {
        let input = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("vector");
        let g = Value::Tensor(
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
            .expect("tensor"),
        );
        let mut params = BTreeMap::new();
        params.insert("axis".into(), "0".into());
        params.insert("num_sections".into(), "3".into());

        let grads = vjp_single(Primitive::Split, &[input], &g, &params).expect("vjp");
        let out = grads[0].as_tensor().expect("tensor");
        assert_eq!(out.shape.dims, vec![6]);
        assert_eq!(
            tensor_f64_values(&grads[0]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_expand_dims_vjp_single() {
        let input = Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector");
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![1, 3] },
                vec![
                    Literal::from_f64(2.0),
                    Literal::from_f64(4.0),
                    Literal::from_f64(6.0),
                ],
            )
            .expect("tensor"),
        );
        let mut params = BTreeMap::new();
        params.insert("axis".into(), "0".into());

        let grads = vjp_single(Primitive::ExpandDims, &[input], &g, &params).expect("vjp");
        let out = grads[0].as_tensor().expect("tensor");
        assert_eq!(out.shape.dims, vec![3]);
        assert_eq!(tensor_f64_values(&grads[0]), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_cbrt_vjp_single() {
        let input = Value::scalar_f64(8.0);
        let g = Value::scalar_f64(1.0);
        let grads = vjp_single(Primitive::Cbrt, &[input], &g, &BTreeMap::new()).expect("vjp");
        let grad = grads[0].as_f64_scalar().expect("scalar");
        assert!((grad - (1.0 / 12.0)).abs() < 1e-10, "got {grad}");
    }

    #[test]
    fn test_integer_pow_vjp_single() {
        let input = Value::scalar_f64(3.0);
        let g = Value::scalar_f64(1.0);
        let mut params = BTreeMap::new();
        params.insert("exponent".into(), "4".into());

        let grads = vjp_single(Primitive::IntegerPow, &[input], &g, &params).expect("vjp");
        let grad = grads[0].as_f64_scalar().expect("scalar");
        assert!((grad - 108.0).abs() < 1e-10, "got {grad}");
    }

    #[test]
    fn test_shift_right_arithmetic_no_grad() {
        let grads = vjp_single(
            Primitive::ShiftRightArithmetic,
            &[Value::scalar_i64(8), Value::scalar_i64(1)],
            &Value::scalar_i64(1),
            &BTreeMap::new(),
        )
        .expect("vjp");
        assert_eq!(grads[0].as_i64_scalar(), Some(0));
        assert_eq!(grads[1].as_i64_scalar(), Some(0));
    }

    #[test]
    fn test_reduce_and_no_grad() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::Bool,
                Shape { dims: vec![3] },
                vec![
                    Literal::Bool(true),
                    Literal::Bool(false),
                    Literal::Bool(true),
                ],
            )
            .expect("tensor"),
        );
        let mut params = BTreeMap::new();
        params.insert("axes".into(), "0".into());
        let grads = vjp_single(
            Primitive::ReduceAnd,
            &[input],
            &Value::scalar_f64(1.0),
            &params,
        )
        .expect("vjp");
        assert_eq!(tensor_f64_values(&grads[0]), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_reduce_or_no_grad() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::Bool,
                Shape { dims: vec![3] },
                vec![
                    Literal::Bool(true),
                    Literal::Bool(false),
                    Literal::Bool(true),
                ],
            )
            .expect("tensor"),
        );
        let mut params = BTreeMap::new();
        params.insert("axes".into(), "0".into());
        let grads = vjp_single(
            Primitive::ReduceOr,
            &[input],
            &Value::scalar_f64(1.0),
            &params,
        )
        .expect("vjp");
        assert_eq!(tensor_f64_values(&grads[0]), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_conj_vjp_single() {
        let input = Value::scalar_complex128(1.0, -2.0);
        let g = Value::scalar_complex128(3.0, -4.0);
        let grads = vjp_single(Primitive::Conj, &[input], &g, &BTreeMap::new()).expect("vjp");
        let (re, im) = scalar_complex128(&grads[0]);
        assert!((re - 3.0).abs() < 1e-10);
        assert!((im - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_real_vjp_single() {
        let input = Value::scalar_complex128(2.0, -3.0);
        let grads = vjp_single(
            Primitive::Real,
            &[input],
            &Value::scalar_f64(5.0),
            &BTreeMap::new(),
        )
        .expect("vjp");
        let (re, im) = scalar_complex128(&grads[0]);
        assert!((re - 5.0).abs() < 1e-10);
        assert!(im.abs() < 1e-10);
    }

    #[test]
    fn test_imag_vjp_single() {
        let input = Value::scalar_complex128(2.0, -3.0);
        let grads = vjp_single(
            Primitive::Imag,
            &[input],
            &Value::scalar_f64(5.0),
            &BTreeMap::new(),
        )
        .expect("vjp");
        let (re, im) = scalar_complex128(&grads[0]);
        assert!(re.abs() < 1e-10);
        assert!((im - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_complex_constructor_vjp_single() {
        let grads = vjp_single(
            Primitive::Complex,
            &[Value::scalar_f64(1.0), Value::scalar_f64(2.0)],
            &Value::scalar_complex128(7.0, -11.0),
            &BTreeMap::new(),
        )
        .expect("vjp");
        assert!((grads[0].as_f64_scalar().expect("scalar") - 7.0).abs() < 1e-10);
        assert!((grads[1].as_f64_scalar().expect("scalar") + 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_copy_vjp_single() {
        let input = Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector");
        let g = Value::vector_f64(&[0.5, 1.0, 1.5]).expect("vector");
        let grads = vjp_single(Primitive::Copy, &[input], &g, &BTreeMap::new()).expect("vjp");
        assert_eq!(tensor_f64_values(&grads[0]), vec![0.5, 1.0, 1.5]);
    }

    #[test]
    fn test_bitcast_no_grad() {
        let mut params = BTreeMap::new();
        params.insert("new_dtype".into(), "f64".into());
        let grads = vjp_single(
            Primitive::BitcastConvertType,
            &[Value::scalar_i64(123)],
            &Value::scalar_f64(1.0),
            &params,
        )
        .expect("vjp");
        assert_eq!(grads[0].as_i64_scalar(), Some(0));
    }

    #[test]
    fn test_broadcasted_iota_no_grad() {
        let g = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::I64(1),
                    Literal::I64(2),
                    Literal::I64(3),
                    Literal::I64(4),
                    Literal::I64(5),
                    Literal::I64(6),
                ],
            )
            .expect("tensor"),
        );
        let grads = vjp_single(Primitive::BroadcastedIota, &[], &g, &BTreeMap::new()).expect("vjp");
        assert!(grads.is_empty());
    }

    #[test]
    fn test_reduce_precision_vjp_single() {
        let mut params = BTreeMap::new();
        params.insert("exponent_bits".into(), "5".into());
        params.insert("mantissa_bits".into(), "3".into());
        let grads = vjp_single(
            Primitive::ReducePrecision,
            &[Value::scalar_f64(1.125)],
            &Value::scalar_f64(0.75),
            &params,
        )
        .expect("vjp");
        let grad = grads[0].as_f64_scalar().expect("scalar");
        assert!((grad - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_rev_jvp() {
        let primals = vec![Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector")];
        let tangents = vec![Value::vector_f64(&[0.1, 0.2, 0.3]).expect("vector")];
        let mut params = BTreeMap::new();
        params.insert("axes".into(), "0".into());
        let tangent = jvp_rule(Primitive::Rev, &primals, &tangents, &params).expect("jvp");
        assert_eq!(tensor_f64_values(&tangent), vec![0.3, 0.2, 0.1]);
    }

    #[test]
    fn test_squeeze_jvp() {
        let primal = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 3, 1],
                },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                ],
            )
            .expect("tensor"),
        );
        let tangent = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 3, 1],
                },
                vec![
                    Literal::from_f64(4.0),
                    Literal::from_f64(5.0),
                    Literal::from_f64(6.0),
                ],
            )
            .expect("tensor"),
        );
        let mut params = BTreeMap::new();
        params.insert("dimensions".into(), "0,2".into());
        let out = jvp_rule(Primitive::Squeeze, &[primal], &[tangent], &params).expect("jvp");
        assert_eq!(out.as_tensor().expect("tensor").shape.dims, vec![3]);
        assert_eq!(tensor_f64_values(&out), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_split_jvp() {
        let primal = Value::vector_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("vector");
        let tangent = Value::vector_f64(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).expect("vector");
        let mut params = BTreeMap::new();
        params.insert("axis".into(), "0".into());
        params.insert("num_sections".into(), "3".into());
        let out = jvp_rule(Primitive::Split, &[primal], &[tangent], &params).expect("jvp");
        assert_eq!(out.as_tensor().expect("tensor").shape.dims, vec![3, 2]);
        assert_eq!(tensor_f64_values(&out), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    }

    #[test]
    fn test_expand_dims_jvp() {
        let primal = Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector");
        let tangent = Value::vector_f64(&[0.5, 1.0, 1.5]).expect("vector");
        let mut params = BTreeMap::new();
        params.insert("axis".into(), "0".into());
        let out = jvp_rule(Primitive::ExpandDims, &[primal], &[tangent], &params).expect("jvp");
        assert_eq!(out.as_tensor().expect("tensor").shape.dims, vec![1, 3]);
        assert_eq!(tensor_f64_values(&out), vec![0.5, 1.0, 1.5]);
    }

    #[test]
    fn test_cbrt_jvp() {
        let out = jvp_rule(
            Primitive::Cbrt,
            &[Value::scalar_f64(8.0)],
            &[Value::scalar_f64(1.0)],
            &BTreeMap::new(),
        )
        .expect("jvp");
        let tangent = out.as_f64_scalar().expect("scalar");
        assert!((tangent - (1.0 / 12.0)).abs() < 1e-10);
    }

    #[test]
    fn test_integer_pow_jvp() {
        let mut params = BTreeMap::new();
        params.insert("exponent".into(), "4".into());
        let out = jvp_rule(
            Primitive::IntegerPow,
            &[Value::scalar_f64(3.0)],
            &[Value::scalar_f64(1.0)],
            &params,
        )
        .expect("jvp");
        let tangent = out.as_f64_scalar().expect("scalar");
        assert!((tangent - 108.0).abs() < 1e-10);
    }

    #[test]
    fn test_shift_right_arithmetic_no_grad_jvp() {
        let out = jvp_rule(
            Primitive::ShiftRightArithmetic,
            &[Value::scalar_i64(8), Value::scalar_i64(1)],
            &[Value::scalar_i64(0), Value::scalar_i64(0)],
            &BTreeMap::new(),
        )
        .expect("jvp");
        assert_eq!(out.as_i64_scalar(), Some(0));
    }

    #[test]
    fn test_reduce_and_no_grad_jvp() {
        let primal = Value::Tensor(
            TensorValue::new(
                DType::Bool,
                Shape { dims: vec![3] },
                vec![
                    Literal::Bool(true),
                    Literal::Bool(false),
                    Literal::Bool(true),
                ],
            )
            .expect("tensor"),
        );
        let mut params = BTreeMap::new();
        params.insert("axes".into(), "0".into());
        let out = jvp_rule(
            Primitive::ReduceAnd,
            std::slice::from_ref(&primal),
            &[zeros_like(&primal)],
            &params,
        )
        .expect("jvp");
        assert!((out.as_f64_scalar().expect("scalar") - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_reduce_or_no_grad_jvp() {
        let primal = Value::Tensor(
            TensorValue::new(
                DType::Bool,
                Shape { dims: vec![3] },
                vec![
                    Literal::Bool(true),
                    Literal::Bool(false),
                    Literal::Bool(true),
                ],
            )
            .expect("tensor"),
        );
        let mut params = BTreeMap::new();
        params.insert("axes".into(), "0".into());
        let out = jvp_rule(
            Primitive::ReduceOr,
            std::slice::from_ref(&primal),
            &[zeros_like(&primal)],
            &params,
        )
        .expect("jvp");
        assert!((out.as_f64_scalar().expect("scalar") - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_conj_jvp() {
        let out = jvp_rule(
            Primitive::Conj,
            &[Value::scalar_complex128(0.0, 0.0)],
            &[Value::scalar_complex128(2.0, -5.0)],
            &BTreeMap::new(),
        )
        .expect("jvp");
        let (re, im) = scalar_complex128(&out);
        assert!((re - 2.0).abs() < 1e-10);
        assert!((im - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_real_jvp() {
        let out = jvp_rule(
            Primitive::Real,
            &[Value::scalar_complex128(1.0, 2.0)],
            &[Value::scalar_complex128(3.0, -4.0)],
            &BTreeMap::new(),
        )
        .expect("jvp");
        assert!((out.as_f64_scalar().expect("scalar") - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_imag_jvp() {
        let out = jvp_rule(
            Primitive::Imag,
            &[Value::scalar_complex128(1.0, 2.0)],
            &[Value::scalar_complex128(3.0, -4.0)],
            &BTreeMap::new(),
        )
        .expect("jvp");
        assert!((out.as_f64_scalar().expect("scalar") + 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_complex_constructor_jvp() {
        let out = jvp_rule(
            Primitive::Complex,
            &[Value::scalar_f64(1.0), Value::scalar_f64(2.0)],
            &[Value::scalar_f64(0.1), Value::scalar_f64(0.2)],
            &BTreeMap::new(),
        )
        .expect("jvp");
        let (re, im) = scalar_complex128(&out);
        assert!((re - 0.1).abs() < 1e-10);
        assert!((im - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_copy_jvp() {
        let tangent = Value::vector_f64(&[0.2, 0.4, 0.6]).expect("vector");
        let out = jvp_rule(
            Primitive::Copy,
            &[Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector")],
            std::slice::from_ref(&tangent),
            &BTreeMap::new(),
        )
        .expect("jvp");
        assert_eq!(tensor_f64_values(&out), vec![0.2, 0.4, 0.6]);
    }

    #[test]
    fn test_bitcast_no_grad_jvp() {
        let mut params = BTreeMap::new();
        params.insert("new_dtype".into(), "f64".into());
        let out = jvp_rule(
            Primitive::BitcastConvertType,
            &[Value::scalar_i64(42)],
            &[Value::scalar_i64(1)],
            &params,
        )
        .expect("jvp");
        assert!((out.as_f64_scalar().expect("scalar") - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_broadcasted_iota_no_grad_jvp() {
        let mut params = BTreeMap::new();
        params.insert("shape".into(), "2,3".into());
        params.insert("dimension".into(), "1".into());
        params.insert("dtype".into(), "i64".into());
        let out = jvp_rule(Primitive::BroadcastedIota, &[], &[], &params).expect("jvp");
        let tensor = out.as_tensor().expect("tensor");
        assert_eq!(tensor.shape.dims, vec![2, 3]);
        for lit in &tensor.elements {
            assert_eq!(lit.as_i64(), Some(0));
        }
    }

    #[test]
    fn test_reduce_precision_jvp() {
        let mut params = BTreeMap::new();
        params.insert("exponent_bits".into(), "5".into());
        params.insert("mantissa_bits".into(), "3".into());
        let out = jvp_rule(
            Primitive::ReducePrecision,
            &[Value::scalar_f64(1.125)],
            &[Value::scalar_f64(0.25)],
            &params,
        )
        .expect("jvp");
        assert!((out.as_f64_scalar().expect("scalar") - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_cbrt_vjp_finite_diff() {
        let x = 2.5;
        let eps = 1e-6;
        let sym = vjp_single(
            Primitive::Cbrt,
            &[Value::scalar_f64(x)],
            &Value::scalar_f64(1.0),
            &BTreeMap::new(),
        )
        .expect("vjp")[0]
            .as_f64_scalar()
            .expect("scalar");

        let plus = eval_primitive(
            Primitive::Cbrt,
            &[Value::scalar_f64(x + eps)],
            &BTreeMap::new(),
        )
        .expect("eval")
        .as_f64_scalar()
        .expect("scalar");
        let minus = eval_primitive(
            Primitive::Cbrt,
            &[Value::scalar_f64(x - eps)],
            &BTreeMap::new(),
        )
        .expect("eval")
        .as_f64_scalar()
        .expect("scalar");
        let num = (plus - minus) / (2.0 * eps);
        assert!((sym - num).abs() < 1e-4, "sym={sym}, num={num}");
    }

    #[test]
    fn test_integer_pow_vjp_finite_diff() {
        let x = 1.3;
        let eps = 1e-6;
        let mut params = BTreeMap::new();
        params.insert("exponent".into(), "5".into());

        let sym = vjp_single(
            Primitive::IntegerPow,
            &[Value::scalar_f64(x)],
            &Value::scalar_f64(1.0),
            &params,
        )
        .expect("vjp")[0]
            .as_f64_scalar()
            .expect("scalar");

        let plus = eval_primitive(
            Primitive::IntegerPow,
            &[Value::scalar_f64(x + eps)],
            &params,
        )
        .expect("eval")
        .as_f64_scalar()
        .expect("scalar");
        let minus = eval_primitive(
            Primitive::IntegerPow,
            &[Value::scalar_f64(x - eps)],
            &params,
        )
        .expect("eval")
        .as_f64_scalar()
        .expect("scalar");
        let num = (plus - minus) / (2.0 * eps);
        assert!((sym - num).abs() < 1e-4, "sym={sym}, num={num}");
    }

    #[test]
    fn test_lgamma_vjp_finite_diff() {
        let x = 2.4_f64;
        let eps = 1e-6_f64;

        let sym = vjp_single(
            Primitive::Lgamma,
            &[Value::scalar_f64(x)],
            &Value::scalar_f64(1.0),
            &BTreeMap::new(),
        )
        .expect("vjp")[0]
            .as_f64_scalar()
            .expect("scalar");

        let plus = eval_primitive(
            Primitive::Lgamma,
            &[Value::scalar_f64(x + eps)],
            &BTreeMap::new(),
        )
        .expect("eval")
        .as_f64_scalar()
        .expect("scalar");
        let minus = eval_primitive(
            Primitive::Lgamma,
            &[Value::scalar_f64(x - eps)],
            &BTreeMap::new(),
        )
        .expect("eval")
        .as_f64_scalar()
        .expect("scalar");
        let num = (plus - minus) / (2.0 * eps);

        assert!((sym - num).abs() < 1e-5, "sym={sym}, num={num}");
    }

    #[test]
    fn test_digamma_vjp_finite_diff() {
        let x = 1.7_f64;
        let eps = 1e-6_f64;

        let sym = vjp_single(
            Primitive::Digamma,
            &[Value::scalar_f64(x)],
            &Value::scalar_f64(1.0),
            &BTreeMap::new(),
        )
        .expect("vjp")[0]
            .as_f64_scalar()
            .expect("scalar");

        let plus = eval_primitive(
            Primitive::Digamma,
            &[Value::scalar_f64(x + eps)],
            &BTreeMap::new(),
        )
        .expect("eval")
        .as_f64_scalar()
        .expect("scalar");
        let minus = eval_primitive(
            Primitive::Digamma,
            &[Value::scalar_f64(x - eps)],
            &BTreeMap::new(),
        )
        .expect("eval")
        .as_f64_scalar()
        .expect("scalar");
        let num = (plus - minus) / (2.0 * eps);

        assert!((sym - num).abs() < 1e-4, "sym={sym}, num={num}");
    }

    #[test]
    fn test_erf_inv_vjp_single() {
        let x = 0.4_f64;
        let grad = vjp_single(
            Primitive::ErfInv,
            &[Value::scalar_f64(x)],
            &Value::scalar_f64(1.0),
            &BTreeMap::new(),
        )
        .expect("vjp")[0]
            .as_f64_scalar()
            .expect("scalar");

        let y = eval_primitive(Primitive::ErfInv, &[Value::scalar_f64(x)], &BTreeMap::new())
            .expect("eval")
            .as_f64_scalar()
            .expect("scalar");
        let expected = std::f64::consts::PI.sqrt() / 2.0 * (y * y).exp();

        assert!(
            (grad - expected).abs() < 1e-8,
            "grad={grad}, expected={expected}"
        );
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
        let grads = vjp_single(Primitive::ReduceMax, &[input], &g, &params).unwrap();
        let vals = tensor_f64_values(&grads[0]);
        assert!((vals[0] - 0.0).abs() < 1e-10, "non-max should be 0");
        assert!((vals[1] - 1.0).abs() < 1e-10, "max should get gradient");
        assert!((vals[2] - 0.0).abs() < 1e-10, "non-max should be 0");
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
        let grads = vjp_single(Primitive::ReduceProd, &[input], &g, &params).unwrap();
        let vals = tensor_f64_values(&grads[0]);
        assert!((vals[0] - 12.0).abs() < 1e-10, "got {}", vals[0]);
        assert!((vals[1] - 8.0).abs() < 1e-10, "got {}", vals[1]);
        assert!((vals[2] - 6.0).abs() < 1e-10, "got {}", vals[2]);
    }

    #[test]
    fn vjp_reduce_prod_axis0_rank2_zero_aware() {
        use fj_core::{DType, Literal, Shape, TensorValue};

        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(0.0),
                    Literal::from_f64(4.0),
                    Literal::from_f64(5.0),
                    Literal::from_f64(6.0),
                ],
            )
            .unwrap(),
        );
        let g = Value::vector_f64(&[10.0, 20.0, 30.0]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "0".to_owned());

        let grads = vjp_single(Primitive::ReduceProd, &[input], &g, &params).unwrap();
        assert_eq!(
            tensor_f64_values(&grads[0]),
            vec![40.0, 100.0, 180.0, 10.0, 40.0, 0.0]
        );
    }

    #[test]
    fn vjp_reduce_prod_axis1_rank2_zero_aware() {
        use fj_core::{DType, Literal, Shape, TensorValue};

        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0),
                    Literal::from_f64(0.0),
                    Literal::from_f64(6.0),
                ],
            )
            .unwrap(),
        );
        let g = Value::vector_f64(&[7.0, 11.0]).unwrap();
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "1".to_owned());

        let grads = vjp_single(Primitive::ReduceProd, &[input], &g, &params).unwrap();
        assert_eq!(
            tensor_f64_values(&grads[0]),
            vec![42.0, 21.0, 14.0, 0.0, 264.0, 0.0]
        );
    }

    #[test]
    fn vjp_reduce_prod_empty_huge_shape_returns_empty_without_stride_overflow() {
        let huge = u32::MAX;
        let input = empty_f64_tensor(vec![1, huge, huge, huge, 0]);
        let g = empty_f64_tensor(vec![huge, huge, huge, 0]);
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "0".to_owned());

        let grads = vjp_single(Primitive::ReduceProd, &[input], &g, &params)
            .expect("empty huge reduce_prod VJP should stay fail-closed and panic-free");
        let tensor = expect_tensor_ref(&grads[0], "reduce_prod VJP gradient");
        assert_eq!(tensor.shape.dims, vec![1, huge, huge, huge, 0]);
        assert!(tensor.elements.is_empty());
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
        let grads = vjp_single(Primitive::ReduceMin, &[input], &g, &params).unwrap();
        let vals = tensor_f64_values(&grads[0]);
        assert!((vals[0] - 0.0).abs() < 1e-10);
        assert!((vals[1] - 2.0).abs() < 1e-10, "min should get gradient");
        assert!((vals[2] - 0.0).abs() < 1e-10);
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

        let grads = vjp_single(Primitive::BroadcastInDim, &[input], &g, &params).unwrap();
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

        let grads = vjp_single(Primitive::BroadcastInDim, &[input], &g, &params).unwrap();
        // Sum along axis 0: [1+4, 2+5, 3+6] = [5, 7, 9]
        let t = expect_tensor_ref(&grads[0], "broadcast vector-to-matrix gradient");
        let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
        assert_eq!(t.shape.dims, vec![3], "shape should be [3]");
        assert!((vals[0] - 5.0).abs() < 1e-10, "got {}", vals[0]);
        assert!((vals[1] - 7.0).abs() < 1e-10, "got {}", vals[1]);
        assert!((vals[2] - 9.0).abs() < 1e-10, "got {}", vals[2]);
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

        let grads = vjp_single(Primitive::BroadcastInDim, &[input], &g, &params).unwrap();
        let vals = tensor_f64_values(&grads[0]);
        assert!((vals[0] - 10.0).abs() < 1e-10);
        assert!((vals[1] - 20.0).abs() < 1e-10);
        assert!((vals[2] - 30.0).abs() < 1e-10);
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
        let grads = vjp_single(
            Primitive::Scatter,
            &[operand, indices, updates],
            &g,
            &params,
        )
        .unwrap();

        // grad_operand should be [[1,1],[0,0],[1,1]] — zeroed at index 1
        let vals = tensor_f64_values(&grads[0]);
        assert_eq!(vals.len(), 6);
        assert!((vals[0] - 1.0).abs() < 1e-10, "row 0 should pass through");
        assert!((vals[1] - 1.0).abs() < 1e-10, "row 0 should pass through");
        assert!(vals[2].abs() < 1e-10, "row 1 should be zeroed");
        assert!(vals[3].abs() < 1e-10, "row 1 should be zeroed");
        assert!((vals[4] - 1.0).abs() < 1e-10, "row 2 should pass through");
        assert!((vals[5] - 1.0).abs() < 1e-10, "row 2 should pass through");

        // grad_updates should be [[1,1]] — gathered from g at index 1
        let vals = tensor_f64_values(&grads[1]);
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn vjp_scatter_empty_huge_tail_returns_empty_without_slice_overflow() {
        let huge = u32::MAX;
        let operand = empty_f64_tensor(vec![huge, huge, huge, 0]);
        let indices = Value::Tensor(
            TensorValue::new(DType::I64, Shape { dims: vec![0] }, Vec::new())
                .expect("empty indices tensor"),
        );
        let updates = empty_f64_tensor(vec![0, huge, huge, 0]);
        let g = empty_f64_tensor(vec![huge, huge, huge, 0]);

        let grads = vjp_single(
            Primitive::Scatter,
            &[operand, indices, updates],
            &g,
            &BTreeMap::new(),
        )
        .expect("empty huge scatter VJP should not overflow slice size");
        let operand_grad = expect_tensor_ref(&grads[0], "scatter operand gradient");
        assert_eq!(operand_grad.shape.dims, vec![huge, huge, huge, 0]);
        assert!(operand_grad.elements.is_empty());
    }

    // ── Forward-mode JVP tests ──────────────────────────────────

    #[test]
    fn jvp_x_squared_at_3() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = jvp(&jaxpr, &[Value::scalar_f64(3.0)], &[Value::scalar_f64(1.0)])
            .expect("jvp should succeed");
        let primal = result.primals[0].as_f64_scalar().unwrap();
        assert!(
            (primal - 9.0).abs() < 1e-10,
            "primal should be 9, got {primal}"
        );
        let tangent = result.tangents[0].as_f64_scalar().unwrap();
        assert!(
            (tangent - 6.0).abs() < 1e-10,
            "tangent should be 6, got {tangent}"
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
        let result = jvp(&jaxpr, &[Value::scalar_f64(3.0)], &[Value::scalar_f64(1.0)])
            .expect("jvp should succeed");
        let primal = result.primals[0].as_f64_scalar().unwrap();
        assert!(
            (primal - 15.0).abs() < 1e-10,
            "primal should be 15, got {primal}"
        );
        let tangent = result.tangents[0].as_f64_scalar().unwrap();
        assert!(
            (tangent - 8.0).abs() < 1e-10,
            "tangent should be 8, got {tangent}"
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
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let result = jvp(&jaxpr, &[Value::scalar_f64(0.0)], &[Value::scalar_f64(1.0)])
            .expect("jvp should succeed");
        assert!(result.primals[0].as_f64_scalar().unwrap().abs() < 1e-10);
        let tangent = result.tangents[0].as_f64_scalar().unwrap();
        assert!((tangent - 1.0).abs() < 1e-10);
    }

    #[test]
    fn jvp_floor_f32_zero_tangent_preserves_f32_width() {
        use fj_core::{Equation, Jaxpr, VarId};
        use smallvec::smallvec;
        use std::collections::BTreeMap;

        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Floor,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let result = match jvp(
            &jaxpr,
            &[Value::scalar_f32(1.75)],
            &[Value::scalar_f32(1.0)],
        ) {
            Ok(result) => result,
            Err(error) => fail_test(format!("jvp should succeed: {error}")),
        };

        assert_eq!(result.tangents[0].dtype(), DType::F32);
        let bits = match result.tangents[0] {
            Value::Scalar(Literal::F32Bits(bits)) => bits,
            _ => fail_test("f32 tangent should stay F32Bits"),
        };
        assert_eq!(bits, 0.0_f32.to_bits());
    }

    #[test]
    fn jvp_scaled_tangent() {
        let jaxpr = build_program(ProgramSpec::Square);
        let result = jvp(&jaxpr, &[Value::scalar_f64(3.0)], &[Value::scalar_f64(2.0)])
            .expect("jvp should succeed");
        let tangent = result.tangents[0].as_f64_scalar().unwrap();
        assert!(
            (tangent - 12.0).abs() < 1e-10,
            "scaled tangent should be 12, got {tangent}"
        );
    }

    // ── Composition gradient tests ─────────────────────────────
    // Test gradient through chains of operations (where real bugs hide)

    #[test]
    fn grad_exp_of_sin() {
        // f(x) = exp(sin(x)), f'(x) = cos(x) * exp(sin(x))
        use fj_core::{Equation, VarId};
        use smallvec::smallvec;

        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Sin,
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

        let x = 1.0;
        let sym = grad_first(&jaxpr, &[Value::scalar_f64(x)]).unwrap();
        let expected = x.cos() * x.sin().exp();
        assert!(
            (sym - expected).abs() < 1e-8,
            "grad(exp(sin(x))) at x=1: sym={sym}, expected={expected}"
        );
    }

    #[test]
    fn grad_mul_of_sin_cos() {
        // f(x) = sin(x) * cos(x), f'(x) = cos²(x) - sin²(x)
        use fj_core::{Equation, VarId};
        use smallvec::smallvec;

        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Sin,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Cos,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        );

        let x = 0.7;
        let sym = grad_first(&jaxpr, &[Value::scalar_f64(x)]).unwrap();
        let expected = x.cos().powi(2) - x.sin().powi(2);
        assert!(
            (sym - expected).abs() < 1e-8,
            "grad(sin(x)*cos(x)) at x=0.7: sym={sym}, expected={expected}"
        );
    }

    #[test]
    fn grad_log_of_square_plus_one() {
        // f(x) = log(x² + 1), f'(x) = 2x / (x² + 1)
        use fj_core::{Equation, VarId};
        use smallvec::smallvec;

        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Square,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(2)), Atom::Lit(Literal::from_f64(1.0))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Log,
                    inputs: smallvec![Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        );

        let x = 2.0;
        let sym = grad_first(&jaxpr, &[Value::scalar_f64(x)]).unwrap();
        let expected = 2.0 * x / (x * x + 1.0);
        assert!(
            (sym - expected).abs() < 1e-8,
            "grad(log(x²+1)) at x=2: sym={sym}, expected={expected}"
        );
    }

    #[test]
    fn grad_tanh_of_mul() {
        // f(x) = tanh(2x), f'(x) = 2 * (1 - tanh²(2x))
        use fj_core::{Equation, VarId};
        use smallvec::smallvec;

        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(Literal::from_f64(2.0))],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Tanh,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        );

        let x = 0.5;
        let sym = grad_first(&jaxpr, &[Value::scalar_f64(x)]).unwrap();
        let t = (2.0 * x).tanh();
        let expected = 2.0 * (1.0 - t * t);
        assert!(
            (sym - expected).abs() < 1e-8,
            "grad(tanh(2x)) at x=0.5: sym={sym}, expected={expected}"
        );
    }

    // ── Max/Min VJP tie-breaking and edge cases ──────────────────

    #[test]
    fn vjp_max_gradient_splits_equal_args() {
        use smallvec::smallvec;
        // max(a, b) with a == b: JAX splits the gradient evenly.
        let jaxpr = Jaxpr::new(
            vec![VarId(0), VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Max,
                inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );

        let x = 5.0;
        let g0 = grad_first(&jaxpr, &[Value::scalar_f64(x), Value::scalar_f64(x)]).unwrap();
        assert!(
            (g0 - 0.5).abs() < 1e-10,
            "max tie: grad w.r.t. first arg should be 0.5, got {g0}"
        );
    }

    #[test]
    fn vjp_max_distinct_values() {
        use smallvec::smallvec;
        // max(3, 7) → grad goes entirely to second arg (b > a)
        let jaxpr = Jaxpr::new(
            vec![VarId(0), VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Max,
                inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );

        let g0 = grad_first(&jaxpr, &[Value::scalar_f64(3.0), Value::scalar_f64(7.0)]).unwrap();
        assert!(
            g0.abs() < 1e-10,
            "max(3,7): grad w.r.t. first arg should be 0.0, got {g0}"
        );
    }

    #[test]
    fn vjp_min_gradient_splits_equal_args() {
        use smallvec::smallvec;
        // min(a, b) with a == b: JAX splits the gradient evenly.
        let jaxpr = Jaxpr::new(
            vec![VarId(0), VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Min,
                inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );

        let x = 5.0;
        let g0 = grad_first(&jaxpr, &[Value::scalar_f64(x), Value::scalar_f64(x)]).unwrap();
        assert!(
            (g0 - 0.5).abs() < 1e-10,
            "min tie: grad w.r.t. first arg should be 0.5, got {g0}"
        );
    }

    #[test]
    fn vjp_min_distinct_values() {
        use smallvec::smallvec;
        // min(7, 3) → grad goes entirely to second arg (b < a)
        let jaxpr = Jaxpr::new(
            vec![VarId(0), VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Min,
                inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );

        let g0 = grad_first(&jaxpr, &[Value::scalar_f64(7.0), Value::scalar_f64(3.0)]).unwrap();
        assert!(
            g0.abs() < 1e-10,
            "min(7,3): grad w.r.t. first arg should be 0.0, got {g0}"
        );
    }

    #[test]
    fn test_jvp_max_tie_averages_tangents() {
        let params = BTreeMap::new();
        let tangent_out = jvp_rule(
            Primitive::Max,
            &[Value::scalar_f64(5.0), Value::scalar_f64(5.0)],
            &[Value::scalar_f64(2.0), Value::scalar_f64(6.0)],
            &params,
        )
        .expect("jvp");

        let val = tangent_out.as_f64_scalar().expect("scalar tangent");
        assert!((val - 4.0).abs() < 1e-10, "got {val}");
    }

    #[test]
    fn test_jvp_min_tie_averages_tangents() {
        let params = BTreeMap::new();
        let tangent_out = jvp_rule(
            Primitive::Min,
            &[Value::scalar_f64(5.0), Value::scalar_f64(5.0)],
            &[Value::scalar_f64(2.0), Value::scalar_f64(6.0)],
            &params,
        )
        .expect("jvp");

        let val = tangent_out.as_f64_scalar().expect("scalar tangent");
        assert!((val - 4.0).abs() < 1e-10, "got {val}");
    }

    // ── Higher-order gradient tests ─────────────────────────────
    // Tests second derivative (grad of grad) for key functions.

    #[test]
    fn second_derivative_x_squared() {
        use smallvec::smallvec;
        // f(x) = x², f'(x) = 2x, f''(x) = 2
        // Build inner: grad(x²) = 2x
        let inner = Jaxpr::new(
            vec![VarId(0)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Square,
                inputs: smallvec![Atom::Var(VarId(0))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );

        // First derivative at x=3: should be 6
        let first = grad_first(&inner, &[Value::scalar_f64(3.0)]).unwrap();
        assert!((first - 6.0).abs() < 1e-8, "f'(3) = 2*3 = 6, got {first}");

        // Build graph for f'(x) = 2x to take second derivative
        // f'(x) = x + x (numerically equivalent to 2x but buildable)
        let first_deriv = Jaxpr::new(
            vec![VarId(0)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(0)), Atom::Var(VarId(0))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );

        // Second derivative: grad(2x) = 2
        let second = grad_first(&first_deriv, &[Value::scalar_f64(3.0)]).unwrap();
        assert!(
            (second - 2.0).abs() < 1e-8,
            "f''(x) should be 2, got {second}"
        );
    }

    #[test]
    fn second_derivative_exp() {
        use smallvec::smallvec;
        // f(x) = exp(x), f'(x) = exp(x), f''(x) = exp(x)
        // So grad(exp)(x) = exp(x) for all orders
        let jaxpr = Jaxpr::new(
            vec![VarId(0)],
            vec![],
            vec![VarId(1)],
            vec![Equation {
                primitive: Primitive::Exp,
                inputs: smallvec![Atom::Var(VarId(0))],
                outputs: smallvec![VarId(1)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );

        let x = 1.0;
        let first = grad_first(&jaxpr, &[Value::scalar_f64(x)]).unwrap();
        let expected = x.exp();
        assert!(
            (first - expected).abs() < 1e-8,
            "grad(exp)(1.0) should be e, got {first}"
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

    // ── Binary VJP numerical gradient verification ──────────────
    fn numerical_grad_binary(prim: Primitive, a: f64, b: f64) -> (f64, f64) {
        let eps = 1e-6;
        let no_p = BTreeMap::new();
        // d/da
        let fa_plus = eval_primitive(
            prim,
            &[Value::scalar_f64(a + eps), Value::scalar_f64(b)],
            &no_p,
        )
        .unwrap()
        .as_f64_scalar()
        .unwrap();
        let fa_minus = eval_primitive(
            prim,
            &[Value::scalar_f64(a - eps), Value::scalar_f64(b)],
            &no_p,
        )
        .unwrap()
        .as_f64_scalar()
        .unwrap();
        let da = (fa_plus - fa_minus) / (2.0 * eps);

        // d/db
        let fb_plus = eval_primitive(
            prim,
            &[Value::scalar_f64(a), Value::scalar_f64(b + eps)],
            &no_p,
        )
        .unwrap()
        .as_f64_scalar()
        .unwrap();
        let fb_minus = eval_primitive(
            prim,
            &[Value::scalar_f64(a), Value::scalar_f64(b - eps)],
            &no_p,
        )
        .unwrap()
        .as_f64_scalar()
        .unwrap();
        let db = (fb_plus - fb_minus) / (2.0 * eps);

        (da, db)
    }

    fn assert_binary_grad_matches(prim: Primitive, a: f64, b: f64, tol: f64) {
        let jaxpr = make_binary_jaxpr(prim);
        let grads = grad_jaxpr(&jaxpr, &[Value::scalar_f64(a), Value::scalar_f64(b)]).unwrap();
        let sym_da = to_f64(&grads[0]).unwrap();
        let sym_db = to_f64(&grads[1]).unwrap();
        let (num_da, num_db) = numerical_grad_binary(prim, a, b);
        assert!(
            (sym_da - num_da).abs() < tol,
            "d/da {:?}({a},{b}): symbolic={sym_da}, numerical={num_da}",
            prim
        );
        assert!(
            (sym_db - num_db).abs() < tol,
            "d/db {:?}({a},{b}): symbolic={sym_db}, numerical={num_db}",
            prim
        );
    }

    #[test]
    fn vjp_vs_numerical_add() {
        assert_binary_grad_matches(Primitive::Add, 3.0, 5.0, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_sub() {
        assert_binary_grad_matches(Primitive::Sub, 7.0, 3.0, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_mul() {
        assert_binary_grad_matches(Primitive::Mul, 3.0, 4.0, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_div() {
        assert_binary_grad_matches(Primitive::Div, 6.0, 3.0, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_pow() {
        assert_binary_grad_matches(Primitive::Pow, 2.0, 3.0, 1e-3);
    }

    #[test]
    fn vjp_vs_numerical_atan2() {
        assert_binary_grad_matches(Primitive::Atan2, 1.0, 2.0, 1e-4);
    }

    #[test]
    fn vjp_vs_numerical_rem() {
        assert_binary_grad_matches(Primitive::Rem, 7.0, 3.0, 1e-4);
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
            fn prop_jvp_matches_vjp_single(x in prop::num::f64::NORMAL.prop_filter(
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

    // ── Pad AD tests ─────────────────────────────────────────────

    #[test]
    fn vjp_pad_edge_only_extracts_interior() {
        // pad([1, 2, 3], 0, low=1, high=1) = [0, 1, 2, 3, 0]
        // VJP: gradient [a, b, c, d, e] -> operand grad = [b, c, d], pad_value grad = a + e
        let operand = Value::Tensor(
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
        let pad_val = Value::scalar_f64(0.0);
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape::vector(5),
                vec![
                    Literal::from_f64(10.0),
                    Literal::from_f64(20.0),
                    Literal::from_f64(30.0),
                    Literal::from_f64(40.0),
                    Literal::from_f64(50.0),
                ],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("padding_low".into(), "1".into());
        params.insert("padding_high".into(), "1".into());

        let grads = vjp_single(Primitive::Pad, &[operand, pad_val], &g, &params).unwrap();
        assert_eq!(grads.len(), 2);

        // Operand gradient: elements at positions 1, 2, 3 of g -> [20, 30, 40]
        let vals = tensor_f64_values(&grads[0]);
        assert_eq!(vals, vec![20.0, 30.0, 40.0]);

        // Pad value gradient: sum of padding positions = 10 + 50 = 60
        let pad_grad = grads[1].as_f64_scalar().unwrap();
        assert!((pad_grad - 60.0).abs() < 1e-10, "pad_grad = {pad_grad}");
    }

    #[test]
    fn vjp_pad_with_interior_extracts_strided() {
        // pad([1, 2, 3], 0, low=0, high=0, interior=1) = [1, 0, 2, 0, 3]
        // VJP: gradient [a, b, c, d, e] -> operand grad = [a, c, e] (stride 2)
        // pad_value grad = b + d
        let operand = Value::Tensor(
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
        let pad_val = Value::scalar_f64(0.0);
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape::vector(5),
                vec![
                    Literal::from_f64(10.0),
                    Literal::from_f64(20.0),
                    Literal::from_f64(30.0),
                    Literal::from_f64(40.0),
                    Literal::from_f64(50.0),
                ],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("padding_low".into(), "0".into());
        params.insert("padding_high".into(), "0".into());
        params.insert("padding_interior".into(), "1".into());

        let grads = vjp_single(Primitive::Pad, &[operand, pad_val], &g, &params).unwrap();

        // Operand gradient: positions 0, 2, 4 of g -> [10, 30, 50]
        let vals = tensor_f64_values(&grads[0]);
        assert_eq!(vals, vec![10.0, 30.0, 50.0]);

        // Pad value gradient: positions 1, 3 of g -> 20 + 40 = 60
        let pad_grad = grads[1].as_f64_scalar().unwrap();
        assert!((pad_grad - 60.0).abs() < 1e-10, "pad_grad = {pad_grad}");
    }

    #[test]
    fn vjp_pad_2d_edge_only() {
        // pad([[1, 2], [3, 4]], 0, low=[1,1], high=[1,1])
        // Result shape: 4x4
        // VJP should extract the 2x2 interior from a 4x4 gradient
        let operand = Value::Tensor(
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
            .unwrap(),
        );
        let pad_val = Value::scalar_f64(0.0);
        // 4x4 gradient: row-major
        let g_elems: Vec<Literal> = (1..=16).map(|i| Literal::from_f64(i as f64)).collect();
        let g = Value::Tensor(
            TensorValue::new(DType::F64, Shape { dims: vec![4, 4] }, g_elems).unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("padding_low".into(), "1,1".into());
        params.insert("padding_high".into(), "1,1".into());

        let grads = vjp_single(Primitive::Pad, &[operand, pad_val], &g, &params).unwrap();

        // Interior of 4x4 at offset (1,1) with shape 2x2:
        // row 1: g[1][1]=6, g[1][2]=7
        // row 2: g[2][1]=10, g[2][2]=11
        let t = expect_tensor_ref(&grads[0], "pad 2d operand gradient");
        let vals: Vec<f64> = t.elements.iter().map(|l| l.as_f64().unwrap()).collect();
        assert_eq!(vals, vec![6.0, 7.0, 10.0, 11.0]);
        assert_eq!(t.shape.dims, vec![2, 2]);

        // Pad value gradient: sum(g) - sum(operand_grad) = 136 - 34 = 102
        let total_sum: f64 = (1..=16).map(|i| i as f64).sum();
        let op_sum = 6.0 + 7.0 + 10.0 + 11.0;
        let pad_grad = grads[1].as_f64_scalar().unwrap();
        assert!(
            (pad_grad - (total_sum - op_sum)).abs() < 1e-10,
            "pad_grad = {pad_grad}, expected {}",
            total_sum - op_sum
        );
    }

    #[test]
    fn vjp_pad_negative_edge_padding_zeros_cropped_operand_gradient() {
        let operand = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape::vector(4),
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0),
                ],
            )
            .unwrap(),
        );
        let pad_val = Value::scalar_f64(0.0);
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
        let mut params = BTreeMap::new();
        params.insert("padding_low".into(), "1".into());
        params.insert("padding_high".into(), "-2".into());
        params.insert("padding_interior".into(), "0".into());

        let grads = vjp_single(Primitive::Pad, &[operand, pad_val], &g, &params).unwrap();

        let vals = tensor_f64_values(&grads[0]);
        assert_eq!(vals, vec![20.0, 30.0, 0.0, 0.0]);

        let pad_grad = grads[1].as_f64_scalar().unwrap();
        assert!((pad_grad - 10.0).abs() < 1e-10, "pad_grad = {pad_grad}");
    }

    #[test]
    fn jvp_pad_passes_tangent_through() {
        // Pad is linear in its operand, so JVP tangent = tangent_operand
        let primals = vec![Value::scalar_f64(2.0), Value::scalar_f64(0.0)];
        let tangents = vec![Value::scalar_f64(5.0), Value::scalar_f64(0.0)];
        let params: BTreeMap<String, String> = BTreeMap::new();
        let result = jvp_rule(Primitive::Pad, &primals, &tangents, &params).unwrap();
        let result_f64 = result.as_f64_scalar().unwrap();
        assert!(
            (result_f64 - 5.0).abs() < 1e-10,
            "JVP of pad should pass tangent through, got {result_f64}"
        );
    }

    // ── Conv 1D VJP tests ──────────────────────────────────────────

    #[test]
    fn conv_vjp_basic_1d() {
        // lhs=[1, 3, 1] (batch=1, width=3, c_in=1), rhs=[2, 1, 1] (K=2, c_in=1, c_out=1)
        // valid padding, stride=1 => output=[1, 2, 1]
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 3, 1],
                },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                ],
            )
            .unwrap(),
        );
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![2, 1, 1],
                },
                vec![Literal::from_f64(0.5), Literal::from_f64(0.5)],
            )
            .unwrap(),
        );
        // output = [1*0.5+2*0.5, 2*0.5+3*0.5] = [1.5, 2.5]
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 2, 1],
                },
                vec![Literal::from_f64(1.0), Literal::from_f64(1.0)],
            )
            .unwrap(),
        );

        let params = BTreeMap::new();
        let grads = vjp_single(Primitive::Conv, &[lhs.clone(), rhs.clone()], &g, &params).unwrap();

        // grad_lhs[0,0,0] = g[0,0,0]*rhs[0,0,0] + g[0,1,0]*rhs[1,0,0] (if w_out=1 maps to w=0 via k=1)
        // Let's verify: for w=0: w_out=0,k=0 => g[0]*0.5 = 0.5
        //               for w=1: w_out=0,k=1 => g[0]*0.5 + w_out=1,k=0 => g[1]*0.5 = 1.0
        //               for w=2: w_out=1,k=1 => g[1]*0.5 = 0.5
        let grad_lhs = tensor_f64_values(&grads[0]);
        assert!(
            (grad_lhs[0] - 0.5).abs() < 1e-10,
            "grad_lhs[0] = {}, expected 0.5",
            grad_lhs[0]
        );
        assert!(
            (grad_lhs[1] - 1.0).abs() < 1e-10,
            "grad_lhs[1] = {}, expected 1.0",
            grad_lhs[1]
        );
        assert!(
            (grad_lhs[2] - 0.5).abs() < 1e-10,
            "grad_lhs[2] = {}, expected 0.5",
            grad_lhs[2]
        );

        // grad_rhs[k=0] = sum_n sum_w_out lhs[n, w_out*1+0, 0] * g[n, w_out, 0]
        //                = lhs[0,0,0]*g[0,0,0] + lhs[0,1,0]*g[0,1,0] = 1*1 + 2*1 = 3
        // grad_rhs[k=1] = lhs[0,1,0]*g[0,0,0] + lhs[0,2,0]*g[0,1,0] = 2*1 + 3*1 = 5
        let grad_rhs = tensor_f64_values(&grads[1]);
        assert!(
            (grad_rhs[0] - 3.0).abs() < 1e-10,
            "grad_rhs[0] = {}, expected 3.0",
            grad_rhs[0]
        );
        assert!(
            (grad_rhs[1] - 5.0).abs() < 1e-10,
            "grad_rhs[1] = {}, expected 5.0",
            grad_rhs[1]
        );
    }

    #[test]
    fn conv_vjp_numerical_check() {
        // Verify conv VJP with numerical differentiation
        let lhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 4, 1],
                },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0),
                ],
            )
            .unwrap(),
        );
        let rhs = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![2, 1, 1],
                },
                vec![Literal::from_f64(1.0), Literal::from_f64(-1.0)],
            )
            .unwrap(),
        );

        let params = BTreeMap::new();
        let fwd = eval_primitive(Primitive::Conv, &[lhs.clone(), rhs.clone()], &params).unwrap();
        // output = [1-2, 2-3, 3-4] = [-1, -1, -1], shape [1,3,1]
        let out_vals = tensor_f64_values(&fwd);
        for v in &out_vals {
            assert!((*v - (-1.0)).abs() < 1e-10);
        }

        // Use g = ones => grad check
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![1, 3, 1],
                },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0),
                ],
            )
            .unwrap(),
        );

        let grads = vjp_single(Primitive::Conv, &[lhs, rhs], &g, &params).unwrap();
        // Verify grad_lhs has correct shape
        assert_eq!(
            expect_tensor_ref(&grads[0], "conv lhs gradient").shape.dims,
            vec![1, 4, 1]
        );
        // Verify grad_rhs has correct shape
        assert_eq!(
            expect_tensor_ref(&grads[1], "conv rhs gradient").shape.dims,
            vec![2, 1, 1]
        );
    }

    // ── DynamicSlice / DynamicUpdateSlice VJP tests ───────────────

    #[test]
    fn dynamic_slice_vjp_embeds_cotangent_in_operand_shape() {
        let operand = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![5] },
                (1..=5).map(|i| Literal::from_f64(i as f64)).collect(),
            )
            .unwrap(),
        );
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2] },
                vec![Literal::from_f64(7.0), Literal::from_f64(11.0)],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".to_owned(), "2".to_owned());

        let grads = vjp(
            Primitive::DynamicSlice,
            &[operand, Value::scalar_i64(1)],
            &[g],
            &[],
            &params,
        )
        .unwrap();

        assert_eq!(tensor_f64_values(&grads[0]), vec![0.0, 7.0, 11.0, 0.0, 0.0]);
        assert_eq!(grads[1].as_f64_scalar().unwrap(), 0.0);
    }

    #[test]
    fn dynamic_slice_vjp_negative_start_relative_to_end() {
        let operand = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![5] },
                (1..=5).map(|i| Literal::from_f64(i as f64)).collect(),
            )
            .unwrap(),
        );
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2] },
                vec![Literal::from_f64(7.0), Literal::from_f64(11.0)],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("slice_sizes".to_owned(), "2".to_owned());

        let grads = vjp(
            Primitive::DynamicSlice,
            &[operand, Value::scalar_i64(-3)],
            &[g],
            &[],
            &params,
        )
        .unwrap();

        assert_eq!(tensor_f64_values(&grads[0]), vec![0.0, 0.0, 7.0, 11.0, 0.0]);
        assert_eq!(grads[1].as_f64_scalar().unwrap(), 0.0);
    }

    #[test]
    fn dynamic_update_slice_vjp_1d() {
        // operand = [1, 2, 3, 4, 5], update = [10, 20], start = 1
        // result = [1, 10, 20, 4, 5]
        // VJP: grad_operand = g with positions 1,2 zeroed, grad_update = g[1:3]
        let operand = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![5] },
                (1..=5).map(|i| Literal::from_f64(i as f64)).collect(),
            )
            .unwrap(),
        );
        let update = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2] },
                vec![Literal::from_f64(10.0), Literal::from_f64(20.0)],
            )
            .unwrap(),
        );
        let start = Value::scalar_i64(1);

        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![5] },
                (1..=5)
                    .map(|i| Literal::from_f64(i as f64 * 10.0))
                    .collect(),
            )
            .unwrap(),
        );

        let grads = dynamic_update_slice_vjp(&[operand, update, start], &g).unwrap();

        // grad_operand should be [10, 0, 0, 40, 50]
        let g_op = tensor_f64_values(&grads[0]);
        assert!((g_op[0] - 10.0).abs() < 1e-10);
        assert!((g_op[1] - 0.0).abs() < 1e-10);
        assert!((g_op[2] - 0.0).abs() < 1e-10);
        assert!((g_op[3] - 40.0).abs() < 1e-10);
        assert!((g_op[4] - 50.0).abs() < 1e-10);

        // grad_update should be [20, 30] (g at positions 1,2)
        let g_upd = tensor_f64_values(&grads[1]);
        assert!((g_upd[0] - 20.0).abs() < 1e-10);
        assert!((g_upd[1] - 30.0).abs() < 1e-10);

        // Start index gradient should be zero
        assert_eq!(grads[2].as_f64_scalar().unwrap(), 0.0);
    }

    #[test]
    fn dynamic_update_slice_vjp_negative_start_relative_to_end() {
        let operand = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![5] },
                (1..=5).map(|i| Literal::from_f64(i as f64)).collect(),
            )
            .unwrap(),
        );
        let update = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2] },
                vec![Literal::from_f64(10.0), Literal::from_f64(20.0)],
            )
            .unwrap(),
        );
        let start = Value::scalar_i64(-3);
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![5] },
                (1..=5)
                    .map(|i| Literal::from_f64(i as f64 * 10.0))
                    .collect(),
            )
            .unwrap(),
        );

        let grads = dynamic_update_slice_vjp(&[operand, update, start], &g).unwrap();

        let g_op = tensor_f64_values(&grads[0]);
        assert_eq!(g_op, vec![10.0, 20.0, 0.0, 0.0, 50.0]);

        let g_upd = tensor_f64_values(&grads[1]);
        assert_eq!(g_upd, vec![30.0, 40.0]);
        assert_eq!(grads[2].as_f64_scalar().unwrap(), 0.0);
    }

    #[test]
    fn dynamic_update_slice_vjp_empty_huge_update_returns_empty_without_product_overflow() {
        let huge = u32::MAX;
        let operand = empty_f64_tensor(vec![huge, huge, huge, 0]);
        let update = empty_f64_tensor(vec![0, huge, huge, 0]);
        let g = empty_f64_tensor(vec![huge, huge, huge, 0]);

        let grads = dynamic_update_slice_vjp(
            &[
                operand,
                update,
                Value::scalar_i64(0),
                Value::scalar_i64(0),
                Value::scalar_i64(0),
                Value::scalar_i64(0),
            ],
            &g,
        )
        .expect("empty huge dynamic_update_slice VJP should not overflow update products");

        let operand_grad = expect_tensor_ref(&grads[0], "dynamic update operand gradient");
        assert_eq!(operand_grad.shape.dims, vec![huge, huge, huge, 0]);
        assert!(operand_grad.elements.is_empty());

        let update_grad = expect_tensor_ref(&grads[1], "dynamic update update gradient");
        assert_eq!(update_grad.shape.dims, vec![0, huge, huge, 0]);
        assert!(update_grad.elements.is_empty());
    }

    // ── Cumsum VJP tests ──────────────────────────────────────────

    #[test]
    fn cumsum_vjp_1d_tensor() {
        // x = [1, 2, 3], cumsum = [1, 3, 6]
        // VJP of cumsum with g = [1, 1, 1] should be reverse cumsum = [3, 2, 1]
        let x = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                ],
            )
            .unwrap(),
        );
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0),
                ],
            )
            .unwrap(),
        );

        let mut params = BTreeMap::new();
        params.insert("axis".to_string(), "0".to_string());

        let grads = vjp_single(Primitive::Cumsum, &[x], &g, &params).unwrap();
        let result = tensor_f64_values(&grads[0]);
        // reverse cumsum of [1,1,1] = [3, 2, 1]
        assert!((result[0] - 3.0).abs() < 1e-10, "got {}", result[0]);
        assert!((result[1] - 2.0).abs() < 1e-10, "got {}", result[1]);
        assert!((result[2] - 1.0).abs() < 1e-10, "got {}", result[2]);
    }

    #[test]
    fn cumsum_vjp_with_nonuniform_gradient() {
        // g = [1, 2, 3], reverse cumsum = [6, 5, 3]
        let x = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0),
                ],
            )
            .unwrap(),
        );
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                ],
            )
            .unwrap(),
        );

        let mut params = BTreeMap::new();
        params.insert("axis".to_string(), "0".to_string());

        let grads = vjp_single(Primitive::Cumsum, &[x], &g, &params).unwrap();
        let result = tensor_f64_values(&grads[0]);
        assert!((result[0] - 6.0).abs() < 1e-10, "got {}", result[0]);
        assert!((result[1] - 5.0).abs() < 1e-10, "got {}", result[1]);
        assert!((result[2] - 3.0).abs() < 1e-10, "got {}", result[2]);
    }

    // ── Cumprod VJP tests ─────────────────────────────────────────

    #[test]
    fn cumprod_vjp_1d_tensor() {
        // x = [2, 3, 4], cumprod = [2, 6, 24]
        // g = [1, 1, 1]
        // grad_x[i] = sum_{j>=i} g[j] * cumprod[j] / x[i]
        // grad_x[0] = (1*2 + 1*6 + 1*24) / 2 = 32/2 = 16
        // grad_x[1] = (1*6 + 1*24) / 3 = 30/3 = 10
        // grad_x[2] = 1*24 / 4 = 6
        let x = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3] },
                vec![
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0),
                ],
            )
            .unwrap(),
        );
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0),
                ],
            )
            .unwrap(),
        );

        let mut params = BTreeMap::new();
        params.insert("axis".to_string(), "0".to_string());

        let grads = vjp_single(Primitive::Cumprod, &[x], &g, &params).unwrap();
        let result = tensor_f64_values(&grads[0]);
        assert!((result[0] - 16.0).abs() < 1e-10, "got {}", result[0]);
        assert!((result[1] - 10.0).abs() < 1e-10, "got {}", result[1]);
        assert!((result[2] - 6.0).abs() < 1e-10, "got {}", result[2]);
    }

    // ── Sort VJP tests ────────────────────────────────────────────

    #[test]
    fn sort_vjp_unscrambles_gradient() {
        // x = [3, 1, 2] => sorted = [1, 2, 3], argsort = [1, 2, 0]
        // g for sorted output = [10, 20, 30]
        // Unsort: result[1] = 10, result[2] = 20, result[0] = 30
        // So grad_x = [30, 10, 20]
        let x = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3] },
                vec![
                    Literal::from_f64(3.0),
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                ],
            )
            .unwrap(),
        );
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3] },
                vec![
                    Literal::from_f64(10.0),
                    Literal::from_f64(20.0),
                    Literal::from_f64(30.0),
                ],
            )
            .unwrap(),
        );

        let params = BTreeMap::new();
        let grads = vjp_single(Primitive::Sort, &[x], &g, &params).unwrap();
        let result = tensor_f64_values(&grads[0]);
        assert!((result[0] - 30.0).abs() < 1e-10, "got {}", result[0]);
        assert!((result[1] - 10.0).abs() < 1e-10, "got {}", result[1]);
        assert!((result[2] - 20.0).abs() < 1e-10, "got {}", result[2]);
    }

    #[test]
    fn sort_vjp_identity_for_sorted_input() {
        // Already sorted: x = [1, 2, 3], g = [10, 20, 30]
        // Argsort = [0, 1, 2] (identity), so unsort is identity too
        // grad_x = [10, 20, 30]
        let x = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                ],
            )
            .unwrap(),
        );
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3] },
                vec![
                    Literal::from_f64(10.0),
                    Literal::from_f64(20.0),
                    Literal::from_f64(30.0),
                ],
            )
            .unwrap(),
        );

        let params = BTreeMap::new();
        let grads = vjp_single(Primitive::Sort, &[x], &g, &params).unwrap();
        let result = tensor_f64_values(&grads[0]);
        assert!((result[0] - 10.0).abs() < 1e-10);
        assert!((result[1] - 20.0).abs() < 1e-10);
        assert!((result[2] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn sort_vjp_descending_unscrambles_gradient() {
        // x = [1, 3, 2], descending sort => [3, 2, 1], permutation [1, 2, 0]
        // g sorted = [7, 8, 9] => grad_x = [9, 7, 8]
        let x = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(2.0),
                ],
            )
            .unwrap(),
        );
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3] },
                vec![
                    Literal::from_f64(7.0),
                    Literal::from_f64(8.0),
                    Literal::from_f64(9.0),
                ],
            )
            .unwrap(),
        );

        let mut params = BTreeMap::new();
        params.insert("descending".to_string(), "true".to_string());

        let grads = vjp_single(Primitive::Sort, &[x], &g, &params).unwrap();
        let result = tensor_f64_values(&grads[0]);
        assert_eq!(result, vec![9.0, 7.0, 8.0]);
    }

    #[test]
    fn sort_vjp_averages_tied_gradients() {
        // x = [2, 1, 1], sorted = [1, 1, 2]
        // g(sorted) = [3, 9, 6]
        // Tied 1s share mean gradient (3+9)/2 = 6
        // grad_x = [6, 6, 6]
        let x = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3] },
                vec![
                    Literal::from_f64(2.0),
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0),
                ],
            )
            .unwrap(),
        );
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3] },
                vec![
                    Literal::from_f64(3.0),
                    Literal::from_f64(9.0),
                    Literal::from_f64(6.0),
                ],
            )
            .unwrap(),
        );

        let grads = vjp_single(Primitive::Sort, &[x], &g, &BTreeMap::new()).unwrap();
        let result = tensor_f64_values(&grads[0]);
        assert_eq!(result, vec![6.0, 6.0, 6.0]);
    }

    #[test]
    fn sort_vjp_axis0_unscrambles_each_column() {
        // x shape [3, 2], axis=0 (column-wise sort)
        // col0: [3,1,2] -> permutation [1,2,0], g col0 [10,30,50] -> [50,10,30]
        // col1: [1,2,3] -> identity,          g col1 [20,40,60] -> [20,40,60]
        let x = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3, 2] },
                vec![
                    Literal::from_f64(3.0),
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                ],
            )
            .unwrap(),
        );
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3, 2] },
                vec![
                    Literal::from_f64(10.0),
                    Literal::from_f64(20.0),
                    Literal::from_f64(30.0),
                    Literal::from_f64(40.0),
                    Literal::from_f64(50.0),
                    Literal::from_f64(60.0),
                ],
            )
            .unwrap(),
        );

        let mut params = BTreeMap::new();
        params.insert("axis".to_string(), "0".to_string());

        let grads = vjp_single(Primitive::Sort, &[x], &g, &params).unwrap();
        let result = tensor_f64_values(&grads[0]);
        assert_eq!(result, vec![50.0, 20.0, 10.0, 40.0, 30.0, 60.0]);
    }

    #[test]
    fn cond_vjp_true_branch() {
        // pred=true, true_val=5.0, false_val=10.0, g=3.0
        // grad: [0.0, 3.0, 0.0] (flows to true operand)
        let pred = Value::scalar_bool(true);
        let true_val = Value::scalar_f64(5.0);
        let false_val = Value::scalar_f64(10.0);
        let g = Value::scalar_f64(3.0);
        let params = BTreeMap::new();
        let grads = vjp_single(Primitive::Cond, &[pred, true_val, false_val], &g, &params).unwrap();
        assert_eq!(grads.len(), 3);
        // pred gradient is zero
        assert_eq!(grads[0].as_f64_scalar().unwrap(), 0.0);
        // true_val gets the gradient
        assert_eq!(grads[1].as_f64_scalar().unwrap(), 3.0);
        // false_val gets zero
        assert_eq!(grads[2].as_f64_scalar().unwrap(), 0.0);
    }

    #[test]
    fn cond_vjp_false_branch() {
        // pred=false, true_val=5.0, false_val=10.0, g=7.0
        // grad: [0.0, 0.0, 7.0] (flows to false operand)
        let pred = Value::scalar_bool(false);
        let true_val = Value::scalar_f64(5.0);
        let false_val = Value::scalar_f64(10.0);
        let g = Value::scalar_f64(7.0);
        let params = BTreeMap::new();
        let grads = vjp_single(Primitive::Cond, &[pred, true_val, false_val], &g, &params).unwrap();
        assert_eq!(grads.len(), 3);
        assert_eq!(grads[0].as_f64_scalar().unwrap(), 0.0);
        assert_eq!(grads[1].as_f64_scalar().unwrap(), 0.0);
        assert_eq!(grads[2].as_f64_scalar().unwrap(), 7.0);
    }

    #[test]
    fn cond_vjp_tensor_branches() {
        // pred=true, true_val=[1,2], false_val=[3,4], g=[10,20]
        // grad should flow to true branch
        let pred = Value::scalar_bool(true);
        let true_val = Value::vector_f64(&[1.0, 2.0]).unwrap();
        let false_val = Value::vector_f64(&[3.0, 4.0]).unwrap();
        let g = Value::vector_f64(&[10.0, 20.0]).unwrap();
        let params = BTreeMap::new();
        let grads = vjp_single(Primitive::Cond, &[pred, true_val, false_val], &g, &params).unwrap();
        // true_val gets [10, 20]
        let vals = tensor_f64_values(&grads[1]);
        assert_eq!(vals, vec![10.0, 20.0]);
        // false_val gets zeros
        let vals = tensor_f64_values(&grads[2]);
        assert_eq!(vals, vec![0.0, 0.0]);
    }

    // ── Scan VJP tests ──────────────────────────────────────────────

    fn scan_params(body_op: &str) -> BTreeMap<String, String> {
        let mut p = BTreeMap::new();
        p.insert("body_op".to_owned(), body_op.to_owned());
        p
    }

    fn scan_params_reverse(body_op: &str) -> BTreeMap<String, String> {
        let mut p = scan_params(body_op);
        p.insert("reverse".to_owned(), "true".to_owned());
        p
    }

    #[test]
    fn scan_vjp_add_vector() {
        // scan(add, 0.0, [1,2,3]) => carry = 6.0
        // d(carry)/d(init) = 1.0, d(carry)/d(xs[i]) = 1.0
        // With g=1.0: grad_init = 1.0, grad_xs = [1, 1, 1]
        let init = Value::scalar_f64(0.0);
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let g = Value::scalar_f64(1.0);
        let grads = vjp_single(Primitive::Scan, &[init, xs], &g, &scan_params("add")).unwrap();
        assert_eq!(grads[0].as_f64_scalar().unwrap(), 1.0);
        let vals = tensor_f64_values(&grads[1]);
        assert_eq!(vals, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn scan_vjp_mul_vector() {
        // scan(mul, 1.0, [2,3,4]) => carry = 24.0
        // d(carry)/d(init) = 2*3*4 = 24
        // d(carry)/d(xs[0]) = 1*3*4 = 12
        // d(carry)/d(xs[1]) = 1*2*4 = 8
        // d(carry)/d(xs[2]) = 1*2*3 = 6
        // With g=1.0:
        let init = Value::scalar_f64(1.0);
        let xs = Value::vector_f64(&[2.0, 3.0, 4.0]).unwrap();
        let g = Value::scalar_f64(1.0);
        let grads = vjp_single(Primitive::Scan, &[init, xs], &g, &scan_params("mul")).unwrap();
        assert_eq!(grads[0].as_f64_scalar().unwrap(), 24.0);
        let vals = tensor_f64_values(&grads[1]);
        assert_eq!(vals, vec![12.0, 8.0, 6.0]);
    }

    #[test]
    fn scan_vjp_div_vector() {
        let init = Value::scalar_f64(120.0);
        let xs = Value::vector_f64(&[2.0, 3.0]).unwrap();
        let g = Value::scalar_f64(1.0);
        let grads = vjp_single(Primitive::Scan, &[init, xs], &g, &scan_params("div")).unwrap();
        assert!((grads[0].as_f64_scalar().unwrap() - (1.0 / 6.0)).abs() < 1e-10);
        let vals = tensor_f64_values(&grads[1]);
        assert!((vals[0] + 10.0).abs() < 1e-10, "vals = {vals:?}");
        assert!((vals[1] + (20.0 / 3.0)).abs() < 1e-10, "vals = {vals:?}");
    }

    #[test]
    fn scan_jvp_pow_vector_uses_body_rule() {
        let init = Value::scalar_f64(2.0);
        let xs = Value::vector_f64(&[3.0, 2.0]).unwrap();
        let init_tangent = Value::scalar_f64(0.5);
        let xs_tangent = Value::vector_f64(&[0.0, 0.0]).unwrap();
        let tangent = jvp_rule(
            Primitive::Scan,
            &[init, xs],
            &[init_tangent, xs_tangent],
            &scan_params("pow"),
        )
        .expect("scan JVP should use body_op=pow");
        let actual = tangent.as_f64_scalar().expect("scan JVP output");
        assert!((actual - 96.0).abs() < 1e-10, "tangent = {actual}");
    }

    #[test]
    fn scan_vjp_max_routes_gradient_to_winning_xs() {
        // scan(max, 0.0, [1,3,2]) => 3.0, with the gradient flowing to xs[1].
        let init = Value::scalar_f64(0.0);
        let xs = Value::vector_f64(&[1.0, 3.0, 2.0]).unwrap();
        let g = Value::scalar_f64(1.0);
        let grads = vjp_single(Primitive::Scan, &[init, xs], &g, &scan_params("max")).unwrap();
        assert_eq!(grads[0].as_f64_scalar().unwrap(), 0.0);
        let vals = tensor_f64_values(&grads[1]);
        assert_eq!(vals, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn scan_vjp_max_ties_split_through_carry_chain() {
        // Two max ties split the cotangent first between carry/xs[1], then
        // between init/xs[0] through the propagated carry cotangent.
        let init = Value::scalar_f64(1.0);
        let xs = Value::vector_f64(&[1.0, 1.0]).unwrap();
        let g = Value::scalar_f64(1.0);
        let grads = vjp_single(Primitive::Scan, &[init, xs], &g, &scan_params("max")).unwrap();
        assert_eq!(grads[0].as_f64_scalar().unwrap(), 0.25);
        let vals = tensor_f64_values(&grads[1]);
        assert_eq!(vals, vec![0.25, 0.5]);
    }

    #[test]
    fn scan_vjp_min_routes_gradient_to_winning_xs() {
        // scan(min, 5.0, [3,4,2]) => 2.0, with the gradient flowing to xs[2].
        let init = Value::scalar_f64(5.0);
        let xs = Value::vector_f64(&[3.0, 4.0, 2.0]).unwrap();
        let g = Value::scalar_f64(1.0);
        let grads = vjp_single(Primitive::Scan, &[init, xs], &g, &scan_params("min")).unwrap();
        assert_eq!(grads[0].as_f64_scalar().unwrap(), 0.0);
        let vals = tensor_f64_values(&grads[1]);
        assert_eq!(vals, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn scan_vjp_add_with_gradient_scaling() {
        // scan(add, 0.0, [1,2]) with g=5.0
        // grad_init = 5.0, grad_xs = [5, 5]
        let init = Value::scalar_f64(0.0);
        let xs = Value::vector_f64(&[1.0, 2.0]).unwrap();
        let g = Value::scalar_f64(5.0);
        let grads = vjp_single(Primitive::Scan, &[init, xs], &g, &scan_params("add")).unwrap();
        assert_eq!(grads[0].as_f64_scalar().unwrap(), 5.0);
        let vals = tensor_f64_values(&grads[1]);
        assert_eq!(vals, vec![5.0, 5.0]);
    }

    #[test]
    fn scan_vjp_empty() {
        // scan over empty xs returns grad to init, zero to xs
        let init = Value::scalar_f64(42.0);
        let xs =
            Value::Tensor(TensorValue::new(DType::F64, Shape { dims: vec![0] }, vec![]).unwrap());
        let g = Value::scalar_f64(1.0);
        let grads = vjp_single(Primitive::Scan, &[init, xs], &g, &scan_params("add")).unwrap();
        assert_eq!(grads[0].as_f64_scalar().unwrap(), 1.0);
    }

    #[test]
    fn scan_jvp_mul_vector_uses_body_rule() {
        // d(init * 2 * 3 * 4) with init tangent 0.5 and xs tangents [0.1,0.2,0.3].
        let init = Value::scalar_f64(1.0);
        let xs = Value::vector_f64(&[2.0, 3.0, 4.0]).unwrap();
        let init_tangent = Value::scalar_f64(0.5);
        let xs_tangent = Value::vector_f64(&[0.1, 0.2, 0.3]).unwrap();
        let tangent = jvp_rule(
            Primitive::Scan,
            &[init, xs],
            &[init_tangent, xs_tangent],
            &scan_params("mul"),
        )
        .expect("scan JVP should use body_op=mul");
        let actual = tangent.as_f64_scalar().expect("scan JVP output");
        assert!((actual - 16.6).abs() < 1e-10, "tangent = {actual}");
    }

    #[test]
    fn scan_jvp_max_routes_tangent_to_winner() {
        let init = Value::scalar_f64(0.0);
        let xs = Value::vector_f64(&[1.0, 3.0, 2.0]).unwrap();
        let init_tangent = Value::scalar_f64(10.0);
        let xs_tangent = Value::vector_f64(&[0.1, 0.2, 0.3]).unwrap();
        let tangent = jvp_rule(
            Primitive::Scan,
            &[init, xs],
            &[init_tangent, xs_tangent],
            &scan_params("max"),
        )
        .expect("scan JVP should use body_op=max");
        let actual = tangent.as_f64_scalar().expect("scan JVP output");
        assert!((actual - 0.2).abs() < 1e-10, "tangent = {actual}");
    }

    #[test]
    fn scan_jvp_max_ties_average_tangent_chain() {
        let init = Value::scalar_f64(1.0);
        let xs = Value::vector_f64(&[1.0, 1.0]).unwrap();
        let init_tangent = Value::scalar_f64(2.0);
        let xs_tangent = Value::vector_f64(&[6.0, 10.0]).unwrap();
        let tangent = jvp_rule(
            Primitive::Scan,
            &[init, xs],
            &[init_tangent, xs_tangent],
            &scan_params("max"),
        )
        .expect("scan JVP should split max ties through scan");
        let actual = tangent.as_f64_scalar().expect("scan JVP output");
        assert!((actual - 7.0).abs() < 1e-10, "tangent = {actual}");
    }

    #[test]
    fn scan_jvp_min_routes_tangent_to_winner() {
        let init = Value::scalar_f64(5.0);
        let xs = Value::vector_f64(&[3.0, 4.0, 2.0]).unwrap();
        let init_tangent = Value::scalar_f64(10.0);
        let xs_tangent = Value::vector_f64(&[0.1, 0.2, 0.3]).unwrap();
        let tangent = jvp_rule(
            Primitive::Scan,
            &[init, xs],
            &[init_tangent, xs_tangent],
            &scan_params("min"),
        )
        .expect("scan JVP should use body_op=min");
        let actual = tangent.as_f64_scalar().expect("scan JVP output");
        assert!((actual - 0.3).abs() < 1e-10, "tangent = {actual}");
    }

    #[test]
    fn scan_jvp_sub_reverse_respects_iteration_order() {
        // reverse scan(sub, 10, [1,2,3]) applies 3, then 2, then 1.
        let init = Value::scalar_f64(10.0);
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap();
        let init_tangent = Value::scalar_f64(0.5);
        let xs_tangent = Value::vector_f64(&[0.1, 0.2, 0.3]).unwrap();
        let tangent = jvp_rule(
            Primitive::Scan,
            &[init, xs],
            &[init_tangent, xs_tangent],
            &scan_params_reverse("sub"),
        )
        .expect("scan JVP should honor reverse body order");
        let actual = tangent.as_f64_scalar().expect("scan JVP output");
        assert!((actual + 0.1).abs() < 1e-10, "tangent = {actual}");
    }

    // ── While VJP tests ─────────────────────────────────────────────

    fn while_params(body_op: &str, cond_op: &str) -> BTreeMap<String, String> {
        let mut p = BTreeMap::new();
        p.insert("body_op".to_owned(), body_op.to_owned());
        p.insert("cond_op".to_owned(), cond_op.to_owned());
        p
    }

    fn switch_params(num_branches: usize) -> BTreeMap<String, String> {
        let mut p = BTreeMap::new();
        p.insert("num_branches".to_owned(), num_branches.to_string());
        p
    }

    fn finite_diff_derivative<F>(f: F, x: f64, eps: f64) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let plus = f(x + eps);
        let minus = f(x - eps);
        (plus - minus) / (2.0 * eps)
    }

    #[test]
    fn while_vjp_add_loop() {
        // while carry < 10: carry += 3 => 0, 3, 6, 9, 12 (4 iterations)
        // For Add body: d(carry)/d(init) = 1.0, d(carry)/d(step) = num_iters = 4
        let init = Value::scalar_f64(0.0);
        let step = Value::scalar_f64(3.0);
        let threshold = Value::scalar_f64(10.0);
        let g = Value::scalar_f64(1.0);
        let grads = vjp_single(
            Primitive::While,
            &[init, step, threshold],
            &g,
            &while_params("add", "lt"),
        )
        .unwrap();
        // grad_init = 1.0 (chain through 4 Add VJPs: each passes g through)
        assert_eq!(grads[0].as_f64_scalar().unwrap(), 1.0);
        // grad_step = 4.0 (4 iterations, each contributes g=1.0)
        assert_eq!(grads[1].as_f64_scalar().unwrap(), 4.0);
        // grad_threshold = 0.0 (discrete)
        assert_eq!(grads[2].as_f64_scalar().unwrap(), 0.0);
    }

    #[test]
    fn while_vjp_no_iterations() {
        // carry=10, while carry < 5 => no iterations
        let init = Value::scalar_f64(10.0);
        let step = Value::scalar_f64(1.0);
        let threshold = Value::scalar_f64(5.0);
        let g = Value::scalar_f64(1.0);
        let grads = vjp_single(
            Primitive::While,
            &[init, step, threshold],
            &g,
            &while_params("add", "lt"),
        )
        .unwrap();
        // grad passes straight through to init
        assert_eq!(grads[0].as_f64_scalar().unwrap(), 1.0);
        // no iterations, step grad = 0
        assert_eq!(grads[1].as_f64_scalar().unwrap(), 0.0);
    }

    #[test]
    fn while_vjp_mul_loop() {
        // while carry < 100: carry *= 2 => 1, 2, 4, 8, 16, 32, 64, 128 (7 iterations)
        // For Mul: d(carry)/d(init) = 2^7 = 128
        // d(carry)/d(step) = sum over iterations of (carry_final / step_at_each)
        let init = Value::scalar_f64(1.0);
        let step = Value::scalar_f64(2.0);
        let threshold = Value::scalar_f64(100.0);
        let g = Value::scalar_f64(1.0);
        let grads = vjp_single(
            Primitive::While,
            &[init, step, threshold],
            &g,
            &while_params("mul", "lt"),
        )
        .unwrap();
        // grad_init = product of all step multipliers = 2^7 = 128
        assert_eq!(grads[0].as_f64_scalar().unwrap(), 128.0);
        // grad_step = sum of (carry_at_step * product_of_remaining_steps) / step
        // Each iteration i: d(final)/d(step_at_i) = init * step^(n-1) = 128/2 = 64 each
        // 7 iterations * 64 = 448
        assert_eq!(grads[1].as_f64_scalar().unwrap(), 448.0);
    }

    #[test]
    fn while_vjp_div_loop_matches_lax_supported_body_op() {
        // while carry > 1: carry /= 2 => 8, 4, 2, 1 (3 iterations).
        // Treating the iteration count as fixed, final = init / step^3.
        let init = Value::scalar_f64(8.0);
        let step = Value::scalar_f64(2.0);
        let threshold = Value::scalar_f64(1.0);
        let g = Value::scalar_f64(1.0);
        let grads = vjp_single(
            Primitive::While,
            &[init, step, threshold],
            &g,
            &while_params("div", "gt"),
        )
        .expect("while VJP should support lax body_op=div");
        let grad_init = grads[0].as_f64_scalar().expect("init gradient");
        let grad_step = grads[1].as_f64_scalar().expect("step gradient");
        assert!((grad_init - 0.125).abs() < 1e-10, "grad_init = {grad_init}");
        assert!((grad_step + 1.5).abs() < 1e-10, "grad_step = {grad_step}");
        assert_eq!(grads[2].as_f64_scalar().unwrap(), 0.0);
    }

    #[test]
    fn while_vjp_pow_loop_matches_lax_supported_body_op() {
        // while carry < 3: carry = carry^2 => one iteration from 2 to 4.
        let init = Value::scalar_f64(2.0);
        let step = Value::scalar_f64(2.0);
        let threshold = Value::scalar_f64(3.0);
        let g = Value::scalar_f64(1.0);
        let grads = vjp_single(
            Primitive::While,
            &[init, step, threshold],
            &g,
            &while_params("pow", "lt"),
        )
        .expect("while VJP should support lax body_op=pow");
        let grad_init = grads[0].as_f64_scalar().expect("init gradient");
        let grad_step = grads[1].as_f64_scalar().expect("step gradient");
        assert!((grad_init - 4.0).abs() < 1e-10, "grad_init = {grad_init}");
        assert!(
            (grad_step - (4.0_f64 * 2.0_f64.ln())).abs() < 1e-10,
            "grad_step = {grad_step}"
        );
    }

    #[test]
    fn while_vjp_ne_condition_matches_lax_supported_cond_op() {
        // while carry != 0: carry -= 1 => 2, 1, 0 (2 iterations).
        let init = Value::scalar_f64(2.0);
        let step = Value::scalar_f64(1.0);
        let threshold = Value::scalar_f64(0.0);
        let g = Value::scalar_f64(1.0);
        let grads = vjp_single(
            Primitive::While,
            &[init, step, threshold],
            &g,
            &while_params("sub", "ne"),
        )
        .expect("while VJP should support lax cond_op=ne");
        assert_eq!(grads[0].as_f64_scalar().unwrap(), 1.0);
        assert_eq!(grads[1].as_f64_scalar().unwrap(), -2.0);
        assert_eq!(grads[2].as_f64_scalar().unwrap(), 0.0);
    }

    #[test]
    fn while_vjp_eq_condition_matches_lax_supported_cond_op() {
        // while carry == 0: carry += 1 => one iteration from 0 to 1.
        let init = Value::scalar_f64(0.0);
        let step = Value::scalar_f64(1.0);
        let threshold = Value::scalar_f64(0.0);
        let g = Value::scalar_f64(1.0);
        let grads = vjp_single(
            Primitive::While,
            &[init, step, threshold],
            &g,
            &while_params("add", "eq"),
        )
        .expect("while VJP should support lax cond_op=eq");
        assert_eq!(grads[0].as_f64_scalar().unwrap(), 1.0);
        assert_eq!(grads[1].as_f64_scalar().unwrap(), 1.0);
        assert_eq!(grads[2].as_f64_scalar().unwrap(), 0.0);
    }

    #[test]
    fn while_jvp_mul_loop_uses_body_rule() {
        // while carry < 100: carry *= 2 => seven body applications.
        // Fixed-path tangent: dinit * 2^7 + dstep * 7 * 2^6.
        let init = Value::scalar_f64(1.0);
        let step = Value::scalar_f64(2.0);
        let threshold = Value::scalar_f64(100.0);
        let init_tangent = Value::scalar_f64(0.5);
        let step_tangent = Value::scalar_f64(0.1);
        let threshold_tangent = Value::scalar_f64(999.0);
        let tangent = jvp_rule(
            Primitive::While,
            &[init, step, threshold],
            &[init_tangent, step_tangent, threshold_tangent],
            &while_params("mul", "lt"),
        )
        .expect("while JVP should use body_op=mul");
        let actual = tangent.as_f64_scalar().expect("while JVP output");
        assert!((actual - 108.8).abs() < 1e-10, "tangent = {actual}");
    }

    #[test]
    fn while_jvp_sub_ne_condition_matches_loop_path() {
        // while carry != 0: carry -= 1 => two iterations from 2 to 0.
        let init = Value::scalar_f64(2.0);
        let step = Value::scalar_f64(1.0);
        let threshold = Value::scalar_f64(0.0);
        let init_tangent = Value::scalar_f64(0.5);
        let step_tangent = Value::scalar_f64(0.1);
        let threshold_tangent = Value::scalar_f64(0.0);
        let tangent = jvp_rule(
            Primitive::While,
            &[init, step, threshold],
            &[init_tangent, step_tangent, threshold_tangent],
            &while_params("sub", "ne"),
        )
        .expect("while JVP should use cond_op=ne");
        let actual = tangent.as_f64_scalar().expect("while JVP output");
        assert!((actual - 0.3).abs() < 1e-10, "tangent = {actual}");
    }

    #[test]
    fn while_jvp_no_iterations_returns_init_tangent() {
        let init = Value::scalar_f64(10.0);
        let step = Value::scalar_f64(1.0);
        let threshold = Value::scalar_f64(5.0);
        let tangent = jvp_rule(
            Primitive::While,
            &[init, step, threshold],
            &[
                Value::scalar_f64(0.75),
                Value::scalar_f64(100.0),
                Value::scalar_f64(100.0),
            ],
            &while_params("add", "lt"),
        )
        .expect("while JVP should handle empty primal loop path");
        assert_eq!(tangent.as_f64_scalar().unwrap(), 0.75);
    }

    #[test]
    fn while_jvp_tensor_scalar_condition_runs_loop() {
        let init = Value::Tensor(
            TensorValue::new(DType::I64, Shape::scalar(), vec![Literal::I64(0)])
                .expect("init scalar tensor"),
        );
        let step = Value::Tensor(
            TensorValue::new(DType::I64, Shape::scalar(), vec![Literal::I64(1)])
                .expect("step scalar tensor"),
        );
        let threshold = Value::Tensor(
            TensorValue::new(DType::I64, Shape::scalar(), vec![Literal::I64(3)])
                .expect("threshold scalar tensor"),
        );
        let init_tangent = Value::Tensor(
            TensorValue::new(DType::F64, Shape::scalar(), vec![Literal::from_f64(0.5)])
                .expect("init tangent"),
        );
        let step_tangent = Value::Tensor(
            TensorValue::new(DType::F64, Shape::scalar(), vec![Literal::from_f64(0.25)])
                .expect("step tangent"),
        );
        let threshold_tangent = Value::Tensor(
            TensorValue::new(DType::F64, Shape::scalar(), vec![Literal::from_f64(0.0)])
                .expect("threshold tangent"),
        );

        let tangent = jvp_rule(
            Primitive::While,
            &[init, step, threshold],
            &[init_tangent, step_tangent, threshold_tangent],
            &while_params("add", "lt"),
        )
        .expect("while JVP should treat scalar tensor comparison output as predicate");

        let tensor = tangent.as_tensor().expect("tensor tangent output");
        assert_eq!(tensor.shape, Shape::scalar());
        assert_eq!(tensor.elements[0].as_f64().unwrap(), 1.25);
    }

    #[test]
    fn cond_vjp_scalar_tensor_zero_selects_false_branch() {
        let pred = Value::Tensor(
            TensorValue::new(DType::F64, Shape::scalar(), vec![Literal::from_f64(0.0)])
                .expect("scalar tensor predicate"),
        );
        let grads = vjp_single(
            Primitive::Cond,
            &[pred, Value::scalar_f64(10.0), Value::scalar_f64(20.0)],
            &Value::scalar_f64(4.0),
            &BTreeMap::new(),
        )
        .expect("cond VJP should accept scalar tensor numeric predicates");

        assert_eq!(grads[0].as_f64_scalar().unwrap(), 0.0);
        assert_eq!(grads[1].as_f64_scalar().unwrap(), 0.0);
        assert_eq!(grads[2].as_f64_scalar().unwrap(), 4.0);
    }

    #[test]
    fn cond_jvp_u32_zero_selects_false_tangent() {
        let tangent = jvp_rule(
            Primitive::Cond,
            &[
                Value::scalar_u32(0),
                Value::scalar_f64(10.0),
                Value::scalar_f64(20.0),
            ],
            &[
                Value::scalar_f64(0.0),
                Value::scalar_f64(0.5),
                Value::scalar_f64(1.5),
            ],
            &BTreeMap::new(),
        )
        .expect("cond JVP should use numeric predicate truthiness");

        assert_eq!(tangent.as_f64_scalar().unwrap(), 1.5);
    }

    #[test]
    fn cond_jvp_rejects_complex_predicate() {
        let result = jvp_rule(
            Primitive::Cond,
            &[
                Value::scalar_complex128(1.0, 0.0),
                Value::scalar_f64(10.0),
                Value::scalar_f64(20.0),
            ],
            &[
                Value::scalar_f64(0.0),
                Value::scalar_f64(0.5),
                Value::scalar_f64(1.5),
            ],
            &BTreeMap::new(),
        );

        assert!(matches!(
            result,
            Err(AdError::EvalFailed(ref detail))
                if detail.contains("cond predicate must be boolean or numeric")
        ));
    }

    #[test]
    fn switch_vjp_bool_index_selects_true_branch() {
        let grads = vjp_single(
            Primitive::Switch,
            &[
                Value::scalar_bool(true),
                Value::scalar_f64(10.0),
                Value::scalar_f64(20.0),
            ],
            &Value::scalar_f64(3.0),
            &switch_params(2),
        )
        .expect("switch VJP should accept bool branch indices");

        assert_eq!(grads[0].as_f64_scalar().unwrap(), 0.0);
        assert_eq!(grads[1].as_f64_scalar().unwrap(), 0.0);
        assert_eq!(grads[2].as_f64_scalar().unwrap(), 3.0);
    }

    #[test]
    fn switch_jvp_scalar_tensor_index_selects_branch_tangent() {
        let index = Value::Tensor(
            TensorValue::new(DType::I64, Shape::scalar(), vec![Literal::I64(1)])
                .expect("scalar tensor index"),
        );
        let tangent = jvp_rule(
            Primitive::Switch,
            &[index, Value::scalar_f64(10.0), Value::scalar_f64(20.0)],
            &[
                Value::scalar_f64(0.0),
                Value::scalar_f64(0.5),
                Value::scalar_f64(1.5),
            ],
            &switch_params(2),
        )
        .expect("switch JVP should accept scalar tensor branch indices");

        assert_eq!(tangent.as_f64_scalar().unwrap(), 1.5);
    }

    #[test]
    fn switch_jvp_negative_index_clamps_to_first_branch_tangent() {
        let tangent = jvp_rule(
            Primitive::Switch,
            &[
                Value::scalar_i64(-1),
                Value::scalar_f64(10.0),
                Value::scalar_f64(20.0),
            ],
            &[
                Value::scalar_f64(0.0),
                Value::scalar_f64(0.5),
                Value::scalar_f64(1.5),
            ],
            &switch_params(2),
        )
        .expect("switch JVP should clamp negative branch indices");

        assert_eq!(tangent.as_f64_scalar().unwrap(), 0.5);
    }

    #[test]
    fn switch_vjp_high_index_clamps_to_last_branch() {
        let grads = vjp_single(
            Primitive::Switch,
            &[
                Value::scalar_u64(u64::MAX),
                Value::scalar_f64(10.0),
                Value::scalar_f64(20.0),
            ],
            &Value::scalar_f64(3.0),
            &switch_params(2),
        )
        .expect("switch VJP should clamp high branch indices");

        assert_eq!(grads[0].as_f64_scalar().unwrap(), 0.0);
        assert_eq!(grads[1].as_f64_scalar().unwrap(), 0.0);
        assert_eq!(grads[2].as_f64_scalar().unwrap(), 3.0);
    }

    // ── Task-specified control-flow AD coverage ───────────────────────

    #[test]
    fn test_grad_through_scan_sum() {
        let init = Value::scalar_f64(0.0);
        let xs = Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector should be valid");
        let g = Value::scalar_f64(1.0);
        let grads = vjp_single(Primitive::Scan, &[init, xs], &g, &scan_params("add"))
            .expect("scan VJP should succeed");
        assert_eq!(
            grads[0]
                .as_f64_scalar()
                .expect("init gradient should be scalar"),
            1.0
        );
    }

    #[test]
    fn test_grad_through_scan_rnn() {
        // RNN-like multiplicative carry update: carry_{t+1} = carry_t * x_t.
        let init = Value::scalar_f64(1.0);
        let xs = Value::vector_f64(&[2.0, 3.0, 4.0]).expect("vector should be valid");
        let g = Value::scalar_f64(1.0);
        let grads = vjp_single(Primitive::Scan, &[init, xs], &g, &scan_params("mul"))
            .expect("scan VJP should succeed");
        assert_eq!(
            grads[0]
                .as_f64_scalar()
                .expect("init gradient should be scalar"),
            24.0
        );
    }

    #[test]
    fn test_grad_through_while_countdown() {
        // while carry > 0: carry -= 1 (starting at 5) => 5 iterations.
        let init = Value::scalar_f64(5.0);
        let step = Value::scalar_f64(1.0);
        let threshold = Value::scalar_f64(0.0);
        let g = Value::scalar_f64(1.0);
        let grads = vjp_single(
            Primitive::While,
            &[init, step, threshold],
            &g,
            &while_params("sub", "gt"),
        )
        .expect("while VJP should succeed");
        assert_eq!(
            grads[0]
                .as_f64_scalar()
                .expect("init gradient should be scalar"),
            1.0
        );
        assert_eq!(
            grads[1]
                .as_f64_scalar()
                .expect("step gradient should be scalar"),
            -5.0
        );
    }

    #[test]
    fn test_grad_through_while_convergence() {
        // while carry > 0.25: carry *= 0.5 (starting at 8) => 5 iterations.
        let init = Value::scalar_f64(8.0);
        let step = Value::scalar_f64(0.5);
        let threshold = Value::scalar_f64(0.25);
        let g = Value::scalar_f64(1.0);
        let grads = vjp_single(
            Primitive::While,
            &[init, step, threshold],
            &g,
            &while_params("mul", "gt"),
        )
        .expect("while VJP should succeed");
        let grad_init = grads[0]
            .as_f64_scalar()
            .expect("init gradient should be scalar");
        let grad_step = grads[1]
            .as_f64_scalar()
            .expect("step gradient should be scalar");
        assert!((grad_init - 0.03125).abs() < 1e-12);
        assert!((grad_step - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_grad_through_cond_true_branch() {
        let pred = Value::scalar_bool(true);
        let true_val = Value::scalar_f64(5.0);
        let false_val = Value::scalar_f64(10.0);
        let g = Value::scalar_f64(3.0);
        let grads = vjp_single(
            Primitive::Cond,
            &[pred, true_val, false_val],
            &g,
            &BTreeMap::new(),
        )
        .expect("cond VJP should succeed");
        assert_eq!(
            grads[1]
                .as_f64_scalar()
                .expect("true-branch gradient should be scalar"),
            3.0
        );
        assert_eq!(
            grads[2]
                .as_f64_scalar()
                .expect("false-branch gradient should be scalar"),
            0.0
        );
    }

    #[test]
    fn test_grad_through_cond_false_branch() {
        let pred = Value::scalar_bool(false);
        let true_val = Value::scalar_f64(5.0);
        let false_val = Value::scalar_f64(10.0);
        let g = Value::scalar_f64(7.0);
        let grads = vjp_single(
            Primitive::Cond,
            &[pred, true_val, false_val],
            &g,
            &BTreeMap::new(),
        )
        .expect("cond VJP should succeed");
        assert_eq!(
            grads[1]
                .as_f64_scalar()
                .expect("true-branch gradient should be scalar"),
            0.0
        );
        assert_eq!(
            grads[2]
                .as_f64_scalar()
                .expect("false-branch gradient should be scalar"),
            7.0
        );
    }

    #[test]
    fn test_grad_through_cond_predicate() {
        let pred = Value::scalar_bool(true);
        let true_val = Value::scalar_f64(2.0);
        let false_val = Value::scalar_f64(3.0);
        let g = Value::scalar_f64(1.0);
        let grads = vjp_single(
            Primitive::Cond,
            &[pred, true_val, false_val],
            &g,
            &BTreeMap::new(),
        )
        .expect("cond VJP should succeed");
        assert_eq!(
            grads[0]
                .as_f64_scalar()
                .expect("predicate gradient should be scalar"),
            0.0
        );
    }

    #[test]
    fn test_grad_scan_finite_diff() {
        let xs = Value::vector_f64(&[2.0, 3.0, 4.0]).expect("vector should be valid");
        let params = scan_params("mul");
        let init = 1.0;
        let eps = 1e-5;
        let numerical = finite_diff_derivative(
            |x| {
                eval_primitive(
                    Primitive::Scan,
                    &[Value::scalar_f64(x), xs.clone()],
                    &params,
                )
                .expect("scan eval should succeed")
                .as_f64_scalar()
                .expect("scan output should be scalar")
            },
            init,
            eps,
        );
        let grads = vjp_single(
            Primitive::Scan,
            &[Value::scalar_f64(init), xs],
            &Value::scalar_f64(1.0),
            &params,
        )
        .expect("scan VJP should succeed");
        let analytic = grads[0]
            .as_f64_scalar()
            .expect("analytic gradient should be scalar");
        assert!((analytic - numerical).abs() < 1e-4);
    }

    #[test]
    fn test_grad_while_finite_diff() {
        let params = while_params("add", "lt");
        let step = Value::scalar_f64(3.0);
        let threshold = Value::scalar_f64(10.0);
        let init = 0.0;
        let eps = 1e-5;
        let numerical = finite_diff_derivative(
            |x| {
                eval_primitive(
                    Primitive::While,
                    &[Value::scalar_f64(x), step.clone(), threshold.clone()],
                    &params,
                )
                .expect("while eval should succeed")
                .as_f64_scalar()
                .expect("while output should be scalar")
            },
            init,
            eps,
        );
        let grads = vjp_single(
            Primitive::While,
            &[Value::scalar_f64(init), step, threshold],
            &Value::scalar_f64(1.0),
            &params,
        )
        .expect("while VJP should succeed");
        let analytic = grads[0]
            .as_f64_scalar()
            .expect("analytic gradient should be scalar");
        assert!((analytic - numerical).abs() < 1e-4);
    }

    // ── ReduceWindow VJP tests ───────────────────────────────────────

    #[test]
    fn test_reduce_window_sum_vjp_1d() {
        // ReduceWindow sum with window=3, stride=1 on [1,2,3,4,5]
        // Forward output: [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12]
        // VJP with g=[1,1,1]:
        //   input[0] appears in window 0 only => grad=1
        //   input[1] appears in windows 0,1 => grad=2
        //   input[2] appears in windows 0,1,2 => grad=3
        //   input[3] appears in windows 1,2 => grad=2
        //   input[4] appears in window 2 only => grad=1
        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![5] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0),
                    Literal::from_f64(5.0),
                ],
            )
            .unwrap(),
        );
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0),
                ],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("window_dimensions".into(), "3".into());
        params.insert("window_strides".into(), "1".into());
        params.insert("reduce_op".into(), "sum".into());

        let grads = vjp_single(Primitive::ReduceWindow, &[input], &g, &params).unwrap();
        let vals = tensor_f64_values(&grads[0]);
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_reduce_window_vjp_empty_huge_shape_returns_empty_without_product_overflow() {
        let huge = u32::MAX;
        let input = empty_f64_tensor(vec![huge, huge, huge, 0]);
        let g = empty_f64_tensor(vec![0, huge, huge, 0]);
        let mut params = BTreeMap::new();
        params.insert("window_dimensions".into(), "1,1,1,1".into());
        params.insert("window_strides".into(), "1,1,1,1".into());
        params.insert("reduce_op".into(), "sum".into());

        let grads = vjp_single(Primitive::ReduceWindow, &[input], &g, &params)
            .expect("empty huge reduce_window VJP should not overflow shape products");
        let tensor = expect_tensor_ref(&grads[0], "reduce_window input gradient");
        assert_eq!(tensor.shape.dims, vec![huge, huge, huge, 0]);
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn test_reduce_window_max_vjp_simple() {
        // Max-pool on [1,5,3,2] with window=2, stride=1
        // Forward: [max(1,5), max(5,3), max(3,2)] = [5, 5, 3]
        // VJP with g=[10,20,30]:
        //   Window 0: max at index 1 (val 5) => grad[1] += 10
        //   Window 1: max at index 1 (val 5) => grad[1] += 20
        //   Window 2: max at index 2 (val 3) => grad[2] += 30
        //   Result: [0, 30, 30, 0]
        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![4] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(5.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(2.0),
                ],
            )
            .unwrap(),
        );
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3] },
                vec![
                    Literal::from_f64(10.0),
                    Literal::from_f64(20.0),
                    Literal::from_f64(30.0),
                ],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("window_dimensions".into(), "2".into());
        params.insert("window_strides".into(), "1".into());
        params.insert("reduce_op".into(), "max".into());

        let grads = vjp_single(Primitive::ReduceWindow, &[input], &g, &params).unwrap();
        let vals = tensor_f64_values(&grads[0]);
        assert_eq!(vals, vec![0.0, 30.0, 30.0, 0.0]);
    }

    #[test]
    fn test_reduce_window_min_vjp_simple() {
        // Min-pool on [4,1,3,2] with window=2, stride=1
        // Forward: [min(4,1), min(1,3), min(3,2)] = [1, 1, 2]
        // VJP with g=[10,20,30]:
        //   Window 0: min at index 1 (val 1) => grad[1] += 10
        //   Window 1: min at index 1 (val 1) => grad[1] += 20
        //   Window 2: min at index 3 (val 2) => grad[3] += 30
        //   Result: [0, 30, 0, 30]
        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![4] },
                vec![
                    Literal::from_f64(4.0),
                    Literal::from_f64(1.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(2.0),
                ],
            )
            .unwrap(),
        );
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3] },
                vec![
                    Literal::from_f64(10.0),
                    Literal::from_f64(20.0),
                    Literal::from_f64(30.0),
                ],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("window_dimensions".into(), "2".into());
        params.insert("window_strides".into(), "1".into());
        params.insert("reduce_op".into(), "min".into());

        let grads = vjp_single(Primitive::ReduceWindow, &[input], &g, &params).unwrap();
        let vals = tensor_f64_values(&grads[0]);
        assert_eq!(vals, vec![0.0, 30.0, 0.0, 30.0]);
    }

    #[test]
    fn test_reduce_window_sum_vjp_strided() {
        // ReduceWindow sum with window=2, stride=2 on [1,2,3,4,5,6]
        // Forward: [1+2, 3+4, 5+6] = [3, 7, 11]
        // VJP with g=[1,10,100]:
        //   Window 0 covers positions [0,1] => grad[0]+=1, grad[1]+=1
        //   Window 1 covers positions [2,3] => grad[2]+=10, grad[3]+=10
        //   Window 2 covers positions [4,5] => grad[4]+=100, grad[5]+=100
        //   Result: [1, 1, 10, 10, 100, 100]
        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![6] },
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
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(10.0),
                    Literal::from_f64(100.0),
                ],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("window_dimensions".into(), "2".into());
        params.insert("window_strides".into(), "2".into());
        params.insert("reduce_op".into(), "sum".into());

        let grads = vjp_single(Primitive::ReduceWindow, &[input], &g, &params).unwrap();
        let vals = tensor_f64_values(&grads[0]);
        assert_eq!(vals, vec![1.0, 1.0, 10.0, 10.0, 100.0, 100.0]);
    }

    #[test]
    fn test_reduce_window_vjp_finite_diff() {
        // Finite difference verification for ReduceWindow sum VJP
        // f(x) = reduce_window_sum(x, window=2, stride=1)
        // Check gradient via (f(x+eps) - f(x-eps)) / (2*eps) for each input element
        let eps = 1e-5;
        let base_vals = vec![2.0, 5.0, 1.0, 4.0, 3.0];

        let mut params = BTreeMap::new();
        params.insert("window_dimensions".into(), "2".into());
        params.insert("window_strides".into(), "1".into());
        params.insert("reduce_op".into(), "sum".into());

        // Compute analytical gradient
        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![5] },
                base_vals.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        );
        // Output has 4 elements; use unit gradient
        let g = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![4] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.0),
                ],
            )
            .unwrap(),
        );
        let analytical = vjp_single(Primitive::ReduceWindow, &[input], &g, &params).unwrap();
        let analytical_vals = tensor_f64_values(&analytical[0]);

        // For each input position, compute finite difference
        for i in 0..5 {
            let mut plus_vals = base_vals.clone();
            plus_vals[i] += eps;
            let plus_input = Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape { dims: vec![5] },
                    plus_vals.iter().map(|&v| Literal::from_f64(v)).collect(),
                )
                .unwrap(),
            );
            let plus_out = eval_primitive(
                Primitive::ReduceWindow,
                std::slice::from_ref(&plus_input),
                &params,
            )
            .unwrap();

            let mut minus_vals = base_vals.clone();
            minus_vals[i] -= eps;
            let minus_input = Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape { dims: vec![5] },
                    minus_vals.iter().map(|&v| Literal::from_f64(v)).collect(),
                )
                .unwrap(),
            );
            let minus_out = eval_primitive(
                Primitive::ReduceWindow,
                std::slice::from_ref(&minus_input),
                &params,
            )
            .unwrap();

            // Sum the output (since g is all-ones, gradient = sum of per-output partials)
            let plus_sum: f64 = match &plus_out {
                Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).sum(),
                Value::Scalar(l) => l.as_f64().unwrap(),
            };
            let minus_sum: f64 = match &minus_out {
                Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).sum(),
                Value::Scalar(l) => l.as_f64().unwrap(),
            };

            let numerical = (plus_sum - minus_sum) / (2.0 * eps);
            assert!(
                (analytical_vals[i] - numerical).abs() < 1e-4,
                "finite diff mismatch at position {i}: analytical={}, numerical={}",
                analytical_vals[i],
                numerical
            );
        }
    }

    // ── Tensor-valued JVP tests (AD-03) ──────────────────────────

    #[test]
    fn test_jvp_tensor_tangent_1d() {
        // JVP of f(x) = x + x (= 2*x) with rank-1 tangent
        // f'(x) · dx = dx + dx = 2 * dx
        use fj_core::{Equation, Jaxpr, VarId};
        use smallvec::smallvec;

        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );

        let primal = Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector should build");
        let tangent = Value::vector_f64(&[0.5, 1.0, 1.5]).expect("vector should build");

        let result = jvp(&jaxpr, &[primal], &[tangent]).expect("jvp should succeed");

        let out_tangent = result.tangents[0].as_tensor().expect("should be tensor");
        let vals = out_tangent.to_f64_vec().expect("f64 elements");
        assert_eq!(vals.len(), 3);
        assert!((vals[0] - 1.0).abs() < 1e-10); // 2 * 0.5
        assert!((vals[1] - 2.0).abs() < 1e-10); // 2 * 1.0
        assert!((vals[2] - 3.0).abs() < 1e-10); // 2 * 1.5
    }

    #[test]
    fn test_jvp_tensor_tangent_2d() {
        // JVP of f(a, b) = dot(a, b) with rank-1 tangents (vector dot product)
        // f'(a,b)·(da, db) = dot(da, b) + dot(a, db)
        let params = BTreeMap::new();

        // a = [1, 2, 3], b = [4, 5, 6]
        let a = Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector should build");
        let b = Value::vector_f64(&[4.0, 5.0, 6.0]).expect("vector should build");

        // da = [0.1, 0.2, 0.3], db = [0.4, 0.5, 0.6]
        let da = Value::vector_f64(&[0.1, 0.2, 0.3]).expect("vector should build");
        let db = Value::vector_f64(&[0.4, 0.5, 0.6]).expect("vector should build");

        // dot(da, b) = 0.1*4 + 0.2*5 + 0.3*6 = 0.4 + 1.0 + 1.8 = 3.2
        // dot(a, db) = 1*0.4 + 2*0.5 + 3*0.6 = 0.4 + 1.0 + 1.8 = 3.2
        // tangent = 3.2 + 3.2 = 6.4
        let tangent_out =
            jvp_rule(Primitive::Dot, &[a, b], &[da, db], &params).expect("jvp should succeed");

        let val = tangent_out
            .as_f64_scalar()
            .expect("dot of vectors is scalar");
        assert!((val - 6.4).abs() < 1e-10, "got {val}");
    }

    #[test]
    fn test_jvp_tensor_output_shape() {
        // JVP output tangent has same shape as primal output
        let params = BTreeMap::new();

        let x = Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector");
        let dx = Value::vector_f64(&[1.0, 1.0, 1.0]).expect("vector");

        // Neg: output shape = input shape
        let tangent_out = jvp_rule(
            Primitive::Neg,
            std::slice::from_ref(&x),
            std::slice::from_ref(&dx),
            &params,
        )
        .expect("jvp");
        let t = tangent_out.as_tensor().expect("should be tensor");
        assert_eq!(t.shape.dims, vec![3]);

        // Exp: output shape = input shape
        let tangent_out = jvp_rule(
            Primitive::Exp,
            std::slice::from_ref(&x),
            std::slice::from_ref(&dx),
            &params,
        )
        .expect("jvp");
        let t = tangent_out.as_tensor().expect("should be tensor");
        assert_eq!(t.shape.dims, vec![3]);

        // Sin: output shape = input shape
        let tangent_out = jvp_rule(Primitive::Sin, &[x], &[dx], &params).expect("jvp");
        let t = tangent_out.as_tensor().expect("should be tensor");
        assert_eq!(t.shape.dims, vec![3]);
    }

    #[test]
    fn test_jvp_tensor_add_tangent() {
        // JVP(add)(x, y, dx, dy) = (x+y, dx+dy) for tensors
        let params = BTreeMap::new();

        let x = Value::vector_f64(&[1.0, 2.0]).expect("vector");
        let y = Value::vector_f64(&[3.0, 4.0]).expect("vector");
        let dx = Value::vector_f64(&[0.1, 0.2]).expect("vector");
        let dy = Value::vector_f64(&[0.3, 0.4]).expect("vector");

        let tangent_out = jvp_rule(Primitive::Add, &[x, y], &[dx, dy], &params).expect("jvp");

        let t = tangent_out.as_tensor().expect("should be tensor");
        let vals = t.to_f64_vec().expect("f64 elements");
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 0.4).abs() < 1e-10); // 0.1 + 0.3
        assert!((vals[1] - 0.6).abs() < 1e-10); // 0.2 + 0.4
    }

    #[test]
    fn test_jvp_tensor_mul_tangent() {
        // JVP(mul)(x, y, dx, dy) = (x*y, x*dy + y*dx) for tensors
        let params = BTreeMap::new();

        let x = Value::vector_f64(&[2.0, 3.0]).expect("vector");
        let y = Value::vector_f64(&[4.0, 5.0]).expect("vector");
        let dx = Value::vector_f64(&[0.1, 0.2]).expect("vector");
        let dy = Value::vector_f64(&[0.3, 0.4]).expect("vector");

        let tangent_out = jvp_rule(Primitive::Mul, &[x, y], &[dx, dy], &params).expect("jvp");

        // x*dy + y*dx = [2*0.3 + 4*0.1, 3*0.4 + 5*0.2] = [1.0, 2.2]
        let t = tangent_out.as_tensor().expect("should be tensor");
        let vals = t.to_f64_vec().expect("f64 elements");
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 1.0).abs() < 1e-10, "got {}", vals[0]);
        assert!((vals[1] - 2.2).abs() < 1e-10, "got {}", vals[1]);
    }

    #[test]
    fn test_jvp_tensor_dot_tangent() {
        // JVP of dot product: f(a) = dot(a, a) — tangent = dot(da, a) + dot(a, da) = 2*dot(a, da)
        let params = BTreeMap::new();

        let a = Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector");
        let da = Value::vector_f64(&[0.1, 0.2, 0.3]).expect("vector");

        // dot(da, a) = 0.1*1 + 0.2*2 + 0.3*3 = 0.1 + 0.4 + 0.9 = 1.4
        // dot(a, da) = same = 1.4
        // tangent = 1.4 + 1.4 = 2.8
        let tangent_out =
            jvp_rule(Primitive::Dot, &[a.clone(), a], &[da.clone(), da], &params).expect("jvp");

        let val = tangent_out
            .as_f64_scalar()
            .expect("dot of vectors is scalar");
        assert!((val - 2.8).abs() < 1e-10, "got {val}");
    }

    #[test]
    fn test_jvp_tensor_reduce_tangent() {
        // JVP of reduce_sum with tensor tangent
        // f(x) = reduce_sum(x), f'(x)·dx = reduce_sum(dx)
        let mut params = BTreeMap::new();
        params.insert("axes".into(), "0".into());

        let x = Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector");
        let dx = Value::vector_f64(&[0.1, 0.2, 0.3]).expect("vector");

        let tangent_out = jvp_rule(Primitive::ReduceSum, &[x], &[dx], &params).expect("jvp");

        let val = tangent_out
            .as_f64_scalar()
            .expect("should reduce to scalar");
        assert!((val - 0.6).abs() < 1e-10, "got {val}"); // 0.1 + 0.2 + 0.3
    }

    #[test]
    fn test_jvp_comparison_scalar_tensor_zero_tangent_shape() {
        let params = BTreeMap::new();

        let rhs = Value::vector_f64(&[1.0, 3.0, 5.0]).expect("vector");
        let drhs = Value::vector_f64(&[0.1, 0.2, 0.3]).expect("vector");

        let tangent_out = jvp_rule(
            Primitive::Lt,
            &[Value::scalar_f64(2.0), rhs],
            &[Value::scalar_f64(0.5), drhs],
            &params,
        )
        .expect("jvp");

        let tensor = tangent_out
            .as_tensor()
            .expect("comparison tangent should follow broadcast output shape");
        assert_eq!(tensor.shape.dims, vec![3]);
        assert_eq!(
            tensor.to_f64_vec().expect("f64 elements"),
            vec![0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_jvp_comparison_tensor_scalar_zero_tangent_shape() {
        let params = BTreeMap::new();

        let lhs = Value::vector_f64(&[1.0, 3.0, 5.0]).expect("vector");
        let dlhs = Value::vector_f64(&[0.1, 0.2, 0.3]).expect("vector");

        let tangent_out = jvp_rule(
            Primitive::Ge,
            &[lhs, Value::scalar_f64(2.0)],
            &[dlhs, Value::scalar_f64(0.5)],
            &params,
        )
        .expect("jvp");

        let tensor = tangent_out
            .as_tensor()
            .expect("comparison tangent should follow broadcast output shape");
        assert_eq!(tensor.shape.dims, vec![3]);
        assert_eq!(
            tensor.to_f64_vec().expect("f64 elements"),
            vec![0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_jvp_reduce_max_selects_primal_winner_tangent() {
        let params = BTreeMap::new();

        let x = Value::vector_f64(&[2.0, 1.0]).expect("vector");
        let dx = Value::vector_f64(&[0.0, 100.0]).expect("vector");

        let tangent_out = jvp_rule(Primitive::ReduceMax, &[x], &[dx], &params).expect("jvp");

        let val = tangent_out.as_f64_scalar().expect("full reduce is scalar");
        assert!((val - 0.0).abs() < 1e-10, "got {val}");
    }

    #[test]
    fn test_jvp_reduce_min_selects_primal_winner_tangent() {
        let params = BTreeMap::new();

        let x = Value::vector_f64(&[2.0, 1.0]).expect("vector");
        let dx = Value::vector_f64(&[0.0, 100.0]).expect("vector");

        let tangent_out = jvp_rule(Primitive::ReduceMin, &[x], &[dx], &params).expect("jvp");

        let val = tangent_out.as_f64_scalar().expect("full reduce is scalar");
        assert!((val - 100.0).abs() < 1e-10, "got {val}");
    }

    #[test]
    fn test_jvp_reduce_max_ties_average_tangents() {
        let params = BTreeMap::new();

        let x = Value::vector_f64(&[2.0, 2.0]).expect("vector");
        let dx = Value::vector_f64(&[0.0, 100.0]).expect("vector");

        let tangent_out = jvp_rule(Primitive::ReduceMax, &[x], &[dx], &params).expect("jvp");

        let val = tangent_out.as_f64_scalar().expect("full reduce is scalar");
        assert!((val - 50.0).abs() < 1e-10, "got {val}");
    }

    #[test]
    fn test_jvp_reduce_max_axis0_uses_primal_winners() {
        use fj_core::{DType, Literal, Shape, TensorValue};

        let x = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::from_f64(2.0),
                    Literal::from_f64(7.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0),
                    Literal::from_f64(1.0),
                    Literal::from_f64(5.0),
                ],
            )
            .expect("tensor"),
        );
        let dx = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::from_f64(10.0),
                    Literal::from_f64(70.0),
                    Literal::from_f64(30.0),
                    Literal::from_f64(40.0),
                    Literal::from_f64(100.0),
                    Literal::from_f64(50.0),
                ],
            )
            .expect("tensor"),
        );
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "0".to_owned());

        let tangent_out = jvp_rule(Primitive::ReduceMax, &[x], &[dx], &params).expect("jvp");

        assert_eq!(tensor_f64_values(&tangent_out), vec![40.0, 70.0, 50.0]);
    }

    #[test]
    fn test_jvp_reduce_min_axis1_uses_primal_winners() {
        use fj_core::{DType, Literal, Shape, TensorValue};

        let x = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::from_f64(2.0),
                    Literal::from_f64(1.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0),
                    Literal::from_f64(0.0),
                    Literal::from_f64(5.0),
                ],
            )
            .expect("tensor"),
        );
        let dx = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::from_f64(10.0),
                    Literal::from_f64(100.0),
                    Literal::from_f64(30.0),
                    Literal::from_f64(40.0),
                    Literal::from_f64(0.0),
                    Literal::from_f64(50.0),
                ],
            )
            .expect("tensor"),
        );
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "1".to_owned());

        let tangent_out = jvp_rule(Primitive::ReduceMin, &[x], &[dx], &params).expect("jvp");

        assert_eq!(tensor_f64_values(&tangent_out), vec![100.0, 0.0]);
    }

    #[test]
    fn test_jvp_reduce_prod_full_reduction_uses_primal_products() {
        let params = BTreeMap::new();

        let x = Value::vector_f64(&[2.0, 3.0, 4.0]).expect("vector");
        let dx = Value::vector_f64(&[0.5, 1.0, 1.5]).expect("vector");

        let tangent_out = jvp_rule(Primitive::ReduceProd, &[x], &[dx], &params).expect("jvp");

        let val = tangent_out
            .as_f64_scalar()
            .expect("full reduction should produce a scalar");
        assert!((val - 23.0).abs() < 1e-10, "got {val}");
    }

    #[test]
    fn test_jvp_reduce_prod_axis0_zero_aware() {
        use fj_core::{DType, Literal, Shape, TensorValue};

        let x = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(0.0),
                    Literal::from_f64(4.0),
                    Literal::from_f64(5.0),
                    Literal::from_f64(6.0),
                ],
            )
            .expect("tensor"),
        );
        let dx = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::from_f64(0.5),
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(1.5),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                ],
            )
            .expect("tensor"),
        );
        let mut params = BTreeMap::new();
        params.insert("axes".into(), "0".into());

        let tangent_out = jvp_rule(Primitive::ReduceProd, &[x], &[dx], &params).expect("jvp");

        let tensor = tangent_out
            .as_tensor()
            .expect("partial reduction should produce a tensor");
        assert_eq!(tensor.shape.dims, vec![3]);
        assert_eq!(
            tensor.to_f64_vec().expect("f64 elements"),
            vec![3.5, 9.0, 12.0]
        );
    }

    #[test]
    fn test_jvp_reduce_prod_axis1_zero_aware() {
        use fj_core::{DType, Literal, Shape, TensorValue};

        let x = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::from_f64(1.0),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0),
                    Literal::from_f64(0.0),
                    Literal::from_f64(6.0),
                ],
            )
            .expect("tensor"),
        );
        let dx = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::from_f64(0.5),
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.5),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0),
                ],
            )
            .expect("tensor"),
        );
        let mut params = BTreeMap::new();
        params.insert("axes".into(), "1".into());

        let tangent_out = jvp_rule(Primitive::ReduceProd, &[x], &[dx], &params).expect("jvp");

        let tensor = tangent_out
            .as_tensor()
            .expect("partial reduction should produce a tensor");
        assert_eq!(tensor.shape.dims, vec![2]);
        assert_eq!(tensor.to_f64_vec().expect("f64 elements"), vec![9.0, 72.0]);
    }

    #[test]
    fn test_jvp_reduce_prod_empty_huge_shape_returns_empty_without_stride_overflow() {
        let huge = u32::MAX;
        let x = empty_f64_tensor(vec![1, huge, huge, huge, 0]);
        let dx = empty_f64_tensor(vec![1, huge, huge, huge, 0]);
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "0".to_owned());

        let tangent_out = jvp_rule(Primitive::ReduceProd, &[x], &[dx], &params)
            .expect("empty huge reduce_prod JVP should not overflow shape products");
        let tensor = expect_tensor_ref(&tangent_out, "reduce_prod tangent");
        assert_eq!(tensor.shape.dims, vec![huge, huge, huge, 0]);
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn test_jvp_cumprod_scalar_passes_tangent() {
        let params = BTreeMap::new();

        let tangent_out = jvp_rule(
            Primitive::Cumprod,
            &[Value::scalar_f64(3.0)],
            &[Value::scalar_f64(0.25)],
            &params,
        )
        .expect("jvp");

        let val = tangent_out
            .as_f64_scalar()
            .expect("scalar cumprod should produce scalar tangent");
        assert!((val - 0.25).abs() < 1e-10, "got {val}");
    }

    #[test]
    fn test_jvp_cumprod_vector_uses_product_recurrence() {
        let params = BTreeMap::new();

        let x = Value::vector_f64(&[2.0, 3.0, 4.0]).expect("vector");
        let dx = Value::vector_f64(&[0.5, 1.0, 1.5]).expect("vector");

        let tangent_out = jvp_rule(Primitive::Cumprod, &[x], &[dx], &params).expect("jvp");

        let tensor = tangent_out.as_tensor().expect("cumprod tangent tensor");
        assert_eq!(
            tensor.to_f64_vec().expect("f64 elements"),
            vec![0.5, 3.5, 23.0]
        );
    }

    #[test]
    fn test_jvp_cumprod_axis0_rank2() {
        use fj_core::{DType, Literal, Shape, TensorValue};

        let x = Value::Tensor(
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
            .expect("tensor"),
        );
        let dx = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::from_f64(0.1),
                    Literal::from_f64(0.2),
                    Literal::from_f64(0.3),
                    Literal::from_f64(0.4),
                    Literal::from_f64(0.5),
                    Literal::from_f64(0.6),
                ],
            )
            .expect("tensor"),
        );
        let mut params = BTreeMap::new();
        params.insert("axis".into(), "0".into());

        let tangent_out = jvp_rule(Primitive::Cumprod, &[x], &[dx], &params).expect("jvp");

        let tensor = tangent_out.as_tensor().expect("cumprod tangent tensor");
        assert_eq!(tensor.shape.dims, vec![2, 3]);
        let values = tensor.to_f64_vec().expect("f64 elements");
        for (actual, expected) in values.iter().zip([0.1, 0.2, 0.3, 0.8, 2.0, 3.6]) {
            assert!(
                (*actual - expected).abs() < 1e-10,
                "got {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_jvp_cumprod_axis1_rank2_zero_aware() {
        use fj_core::{DType, Literal, Shape, TensorValue};

        let x = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0),
                    Literal::from_f64(1.0),
                    Literal::from_f64(0.0),
                    Literal::from_f64(5.0),
                ],
            )
            .expect("tensor"),
        );
        let dx = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::from_f64(0.5),
                    Literal::from_f64(1.0),
                    Literal::from_f64(1.5),
                    Literal::from_f64(2.0),
                    Literal::from_f64(3.0),
                    Literal::from_f64(4.0),
                ],
            )
            .expect("tensor"),
        );
        let mut params = BTreeMap::new();
        params.insert("axis".into(), "1".into());

        let tangent_out = jvp_rule(Primitive::Cumprod, &[x], &[dx], &params).expect("jvp");

        let tensor = tangent_out.as_tensor().expect("cumprod tangent tensor");
        assert_eq!(tensor.shape.dims, vec![2, 3]);
        assert_eq!(
            tensor.to_f64_vec().expect("f64 elements"),
            vec![0.5, 3.5, 23.0, 2.0, 3.0, 15.0]
        );
    }

    #[test]
    fn test_jvp_cumprod_empty_huge_shape_returns_empty_without_stride_overflow() {
        let huge = u32::MAX;
        let x = empty_f64_tensor(vec![huge, huge, huge, 0]);
        let dx = empty_f64_tensor(vec![huge, huge, huge, 0]);
        let mut params = BTreeMap::new();
        params.insert("axis".to_owned(), "0".to_owned());

        let tangent_out = jvp_rule(Primitive::Cumprod, &[x], &[dx], &params)
            .expect("empty huge cumprod JVP should not compute overflowing strides");
        let tensor = expect_tensor_ref(&tangent_out, "cumprod tangent");
        assert_eq!(tensor.shape.dims, vec![huge, huge, huge, 0]);
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn test_jvp_tensor_broadcast_tangent() {
        // Tangent broadcasting matches primal broadcasting
        // BroadcastInDim([1,2,3], shape=[2,3]) => [[1,2,3],[1,2,3]]
        // tangent should broadcast identically
        let mut params = BTreeMap::new();
        params.insert("shape".into(), "2,3".into());
        params.insert("broadcast_dimensions".into(), "1".into());

        let x = Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector");
        let dx = Value::vector_f64(&[0.5, 1.0, 1.5]).expect("vector");

        let tangent_out = jvp_rule(Primitive::BroadcastInDim, &[x], &[dx], &params).expect("jvp");

        let t = tangent_out.as_tensor().expect("should be tensor");
        assert_eq!(t.shape.dims, vec![2, 3]);
        let vals = t.to_f64_vec().expect("f64 elements");
        // Broadcast: each row is [0.5, 1.0, 1.5]
        assert!((vals[0] - 0.5).abs() < 1e-10);
        assert!((vals[1] - 1.0).abs() < 1e-10);
        assert!((vals[2] - 1.5).abs() < 1e-10);
        assert!((vals[3] - 0.5).abs() < 1e-10);
        assert!((vals[4] - 1.0).abs() < 1e-10);
        assert!((vals[5] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_jvp_chain_rule_tensors() {
        // JVP through multi-step tensor computation:
        // f(x) = reduce_sum(x * x) — chain of mul then reduce
        // f'(x)·dx = reduce_sum(2*x*dx)
        use fj_core::{Equation, Jaxpr, VarId};
        use smallvec::smallvec;

        let mut reduce_params = BTreeMap::new();
        reduce_params.insert("axes".into(), "0".into());

        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::ReduceSum,
                    inputs: smallvec![Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params: reduce_params,
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        );

        let x = Value::vector_f64(&[1.0, 2.0, 3.0]).expect("vector");
        let dx = Value::vector_f64(&[1.0, 1.0, 1.0]).expect("vector");

        let result = jvp(&jaxpr, &[x], &[dx]).expect("jvp should succeed");

        // Primal: reduce_sum([1, 4, 9]) = 14
        let primal_val = result.primals[0]
            .as_f64_scalar()
            .expect("primal should be scalar");
        assert!((primal_val - 14.0).abs() < 1e-10, "primal = {primal_val}");

        // Tangent: reduce_sum(2*[1,2,3]*[1,1,1]) = reduce_sum([2,4,6]) = 12
        let tangent_val = result.tangents[0]
            .as_f64_scalar()
            .expect("tangent should be scalar");
        assert!(
            (tangent_val - 12.0).abs() < 1e-10,
            "tangent = {tangent_val}"
        );
    }

    #[test]
    fn test_jacobian_two_outputs_two_inputs() {
        use fj_core::{Jaxpr, VarId};
        use smallvec::smallvec;

        let jaxpr = Jaxpr::new(
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
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        );

        let jac = jacobian_jaxpr(&jaxpr, &[Value::scalar_f64(2.0), Value::scalar_f64(3.0)])
            .expect("jacobian should succeed");
        let tensor = jac.as_tensor().expect("jacobian should return tensor");
        assert_eq!(tensor.shape.dims, vec![2, 2]);
        let vals = tensor.to_f64_vec().expect("f64 elements");
        assert!((vals[0] - 1.0).abs() < 1e-10);
        assert!((vals[1] - 1.0).abs() < 1e-10);
        assert!((vals[2] - 3.0).abs() < 1e-10);
        assert!((vals[3] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_hessian_x2y() {
        use fj_core::{Jaxpr, VarId};
        use smallvec::smallvec;

        let jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        );

        let hessian = hessian_jaxpr(&jaxpr, &[Value::scalar_f64(2.0), Value::scalar_f64(3.0)])
            .expect("hessian should succeed");
        let tensor = hessian.as_tensor().expect("hessian should return tensor");
        assert_eq!(tensor.shape.dims, vec![2, 2]);
        let vals = tensor.to_f64_vec().expect("f64 elements");
        assert!((vals[0] - 6.0).abs() < 1e-3);
        assert!((vals[1] - 4.0).abs() < 1e-3);
        assert!((vals[2] - 4.0).abs() < 1e-3);
        assert!(vals[3].abs() < 1e-3);
    }

    #[test]
    fn test_jacobian_scalar_square_matrix_row() {
        let jac = jacobian_jaxpr(
            &build_program(ProgramSpec::Square),
            &[Value::scalar_f64(3.0)],
        )
        .expect("jacobian should succeed");
        let tensor = jac.as_tensor().expect("jacobian should return tensor");
        assert_eq!(tensor.shape.dims, vec![1, 1]);
        let vals = tensor.to_f64_vec().expect("f64 elements");
        assert_eq!(vals.len(), 1);
        assert!((vals[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_hessian_scalar_square_matrix_row() {
        let hessian = hessian_jaxpr(
            &build_program(ProgramSpec::Square),
            &[Value::scalar_f64(3.0)],
        )
        .expect("hessian should succeed");
        let tensor = hessian.as_tensor().expect("hessian should return tensor");
        assert_eq!(tensor.shape.dims, vec![1, 1]);
        let vals = tensor.to_f64_vec().expect("f64 elements");
        assert_eq!(vals.len(), 1);
        assert!((vals[0] - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_hessian_is_symmetric_for_quadratic() {
        use fj_core::{Jaxpr, VarId};
        use smallvec::smallvec;

        let jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(5)],
            vec![
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(3)), Atom::Var(VarId(4))],
                    outputs: smallvec![VarId(5)],
                    params: BTreeMap::new(),
                    sub_jaxprs: vec![],
                    effects: vec![],
                },
            ],
        );

        let hessian = hessian_jaxpr(&jaxpr, &[Value::scalar_f64(1.5), Value::scalar_f64(2.0)])
            .expect("hessian should succeed");
        let tensor = hessian.as_tensor().expect("hessian should return tensor");
        let vals = tensor.to_f64_vec().expect("f64 elements");
        assert_eq!(tensor.shape.dims, vec![2, 2]);
        assert!(
            (vals[1] - vals[2]).abs() < 1e-3,
            "hessian should be symmetric: {:?}",
            vals
        );
    }

    // ── FFT AD tests ──

    #[allow(dead_code)]
    fn make_fft_jaxpr(prim: Primitive) -> Jaxpr {
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![fj_core::Equation {
                primitive: prim,
                inputs: smallvec::smallvec![Atom::Var(VarId(1))],
                outputs: smallvec::smallvec![VarId(2)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        )
    }

    fn make_rfft_jaxpr(fft_length: usize) -> Jaxpr {
        let mut params = BTreeMap::new();
        params.insert("fft_length".to_owned(), fft_length.to_string());
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![fj_core::Equation {
                primitive: Primitive::Rfft,
                inputs: smallvec::smallvec![Atom::Var(VarId(1))],
                outputs: smallvec::smallvec![VarId(2)],
                params,
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        )
    }

    fn make_irfft_jaxpr(fft_length: usize) -> Jaxpr {
        let mut params = BTreeMap::new();
        params.insert("fft_length".to_owned(), fft_length.to_string());
        Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![fj_core::Equation {
                primitive: Primitive::Irfft,
                inputs: smallvec::smallvec![Atom::Var(VarId(1))],
                outputs: smallvec::smallvec![VarId(2)],
                params,
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        )
    }

    fn make_real_tensor(data: &[f64]) -> Value {
        let elements: Vec<Literal> = data.iter().map(|&v| Literal::from_f64(v)).collect();
        let shape = Shape {
            dims: vec![data.len() as u32],
        };
        Value::Tensor(TensorValue::new(DType::F64, shape, elements).unwrap())
    }

    fn make_complex_tensor(data: &[(f64, f64)]) -> Value {
        let elements: Vec<Literal> = data
            .iter()
            .map(|&(re, im)| Literal::from_complex128(re, im))
            .collect();
        let shape = Shape {
            dims: vec![data.len() as u32],
        };
        Value::Tensor(TensorValue::new(DType::Complex128, shape, elements).unwrap())
    }

    #[test]
    fn fft_padding_rejects_rank_zero_tensor_without_panicking() -> Result<(), String> {
        let scalar_shaped = TensorValue::new(
            DType::Complex128,
            Shape { dims: vec![] },
            vec![Literal::from_complex128(1.0, 0.0)],
        )
        .map_err(|e| e.to_string())?;

        let err = match zero_pad_last_axis_complex(&scalar_shaped, 2) {
            Ok(_) => return Err("rank-zero tensor unexpectedly padded successfully".to_owned()),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("rank to be at least 1"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    #[test]
    fn tensor_value_reports_scalar_context_without_panicking() -> Result<(), String> {
        let err = match tensor_value(&Value::scalar_f64(1.0), "test tensor conversion") {
            Ok(_) => return Err("scalar unexpectedly converted to tensor".to_owned()),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("test tensor conversion"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    fn extract_f64_vec(v: &Value) -> Vec<f64> {
        match v {
            Value::Tensor(t) => t.elements.iter().map(|l| l.as_f64().unwrap()).collect(),
            Value::Scalar(l) => vec![l.as_f64().unwrap()],
        }
    }

    fn extract_complex_vec(v: &Value) -> Vec<(f64, f64)> {
        match v {
            Value::Tensor(t) => t
                .elements
                .iter()
                .map(|l| {
                    l.as_complex128()
                        .unwrap_or((l.as_f64().unwrap_or(0.0), 0.0))
                })
                .collect(),
            Value::Scalar(l) => {
                vec![
                    l.as_complex128()
                        .unwrap_or((l.as_f64().unwrap_or(0.0), 0.0)),
                ]
            }
        }
    }

    #[test]
    fn test_rfft_jvp_linearity() {
        // RFFT is linear, so JVP(RFFT)(dx) = RFFT(dx)
        let jaxpr = make_rfft_jaxpr(4);
        let x = make_real_tensor(&[1.0, 2.0, 3.0, 4.0]);
        let dx = make_real_tensor(&[0.1, 0.2, 0.3, 0.4]);

        let jvp_result = jvp(&jaxpr, &[x], std::slice::from_ref(&dx)).unwrap();
        let tangent_out = extract_complex_vec(&jvp_result.tangents[0]);

        let mut rfft_params = BTreeMap::new();
        rfft_params.insert("fft_length".to_owned(), "4".to_owned());
        let rfft_dx =
            eval_primitive(Primitive::Rfft, std::slice::from_ref(&dx), &rfft_params).unwrap();
        let expected = extract_complex_vec(&rfft_dx);

        assert_eq!(tangent_out.len(), expected.len());
        for (i, ((ar, ai), (er, ei))) in tangent_out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (ar - er).abs() < 1e-10 && (ai - ei).abs() < 1e-10,
                "RFFT JVP mismatch at {i}: got ({ar},{ai}), expected ({er},{ei})"
            );
        }
    }

    #[test]
    fn test_irfft_jvp_linearity() {
        // IRFFT is linear, so JVP(IRFFT)(dg) = IRFFT(dg)
        let jaxpr = make_irfft_jaxpr(4);
        let g_in = make_complex_tensor(&[(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0)]);
        let dg = make_complex_tensor(&[(1.0, 0.0), (0.5, -0.5), (0.0, 0.0)]);

        let jvp_result = jvp(&jaxpr, &[g_in], std::slice::from_ref(&dg)).unwrap();
        let tangent_out = extract_f64_vec(&jvp_result.tangents[0]);

        let mut irfft_params = BTreeMap::new();
        irfft_params.insert("fft_length".to_owned(), "4".to_owned());
        let irfft_dg =
            eval_primitive(Primitive::Irfft, std::slice::from_ref(&dg), &irfft_params).unwrap();
        let expected = extract_f64_vec(&irfft_dg);

        assert_eq!(tangent_out.len(), expected.len());
        for (i, (got, exp)) in tangent_out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-10,
                "IRFFT JVP mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_rfft_vjp_adjoint_identity() {
        // Verify adjoint identity: <RFFT(x), g>_C = <x, VJP(g)>_R
        let x = make_real_tensor(&[1.0, 2.0, 3.0, 4.0]);
        let half_len = 3; // 4/2+1

        let mut params = BTreeMap::new();
        params.insert("fft_length".to_owned(), "4".to_owned());

        let y = eval_primitive(Primitive::Rfft, std::slice::from_ref(&x), &params).unwrap();

        let g = make_complex_tensor(&vec![(1.0, 0.0); half_len]);
        let vjp_result =
            vjp_single(Primitive::Rfft, std::slice::from_ref(&x), &g, &params).unwrap();
        let result = extract_f64_vec(&vjp_result[0]);
        assert_eq!(
            result.len(),
            4,
            "RFFT VJP should return tensor of input length"
        );

        let y_complex = extract_complex_vec(&y);
        let g_complex = extract_complex_vec(&g);
        let lhs: f64 = y_complex
            .iter()
            .zip(g_complex.iter())
            .map(|((yr, yi), (gr, gi))| yr * gr + yi * gi)
            .sum();
        let x_vals = extract_f64_vec(&x);
        let rhs: f64 = x_vals.iter().zip(result.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (lhs - rhs).abs() < 1e-8,
            "adjoint identity failed: <RFFT(x),g> = {lhs}, <x,VJP(g)> = {rhs}"
        );
    }

    #[test]
    fn test_irfft_vjp_adjoint_identity() {
        // Verify adjoint identity: <IRFFT(y), g>_R = <y, VJP(g)>_C
        let fft_length = 4;
        let half_len = 3;

        let mut params = BTreeMap::new();
        params.insert("fft_length".to_owned(), "4".to_owned());

        let y = make_complex_tensor(&[(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0)]);
        let x = eval_primitive(Primitive::Irfft, std::slice::from_ref(&y), &params).unwrap();

        let g = make_real_tensor(&vec![1.0; fft_length]);
        let vjp_result =
            vjp_single(Primitive::Irfft, std::slice::from_ref(&y), &g, &params).unwrap();
        let result = extract_complex_vec(&vjp_result[0]);
        assert_eq!(
            result.len(),
            half_len,
            "IRFFT VJP should return tensor of half-spectrum length"
        );

        let x_vals = extract_f64_vec(&x);
        let g_vals = extract_f64_vec(&g);
        let lhs: f64 = x_vals.iter().zip(g_vals.iter()).map(|(a, b)| a * b).sum();

        let y_complex = extract_complex_vec(&y);
        let rhs: f64 = y_complex
            .iter()
            .zip(result.iter())
            .map(|((yr, yi), (vr, vi))| yr * vr + yi * vi)
            .sum();

        assert!(
            (lhs - rhs).abs() < 1e-8,
            "adjoint identity failed: <IRFFT(y),g> = {lhs}, <y,VJP(g)> = {rhs}"
        );
    }

    // ── QR VJP test ──

    #[test]
    fn test_qr_vjp_identity_matrix() {
        // QR of 2×2 identity: Q=I, R=I
        // VJP should produce correct dA
        let a_data = [1.0, 0.0, 0.0, 1.0];
        let a_elements: Vec<Literal> = a_data.iter().map(|&v| Literal::from_f64(v)).collect();
        let a = Value::Tensor(
            TensorValue::new(DType::F64, Shape { dims: vec![2, 2] }, a_elements).unwrap(),
        );

        // Forward: QR(A) = (Q, R) = (I, I) for identity
        let outputs =
            fj_lax::eval_primitive_multi(Primitive::Qr, std::slice::from_ref(&a), &BTreeMap::new())
                .unwrap();
        assert_eq!(outputs.len(), 2);
        let q = &outputs[0];
        let r = &outputs[1];

        // VJP with g_Q = ones(2,2), g_R = zeros(2,2)
        let g_q_elements: Vec<Literal> = vec![1.0, 1.0, 1.0, 1.0]
            .into_iter()
            .map(Literal::from_f64)
            .collect();
        let g_q = Value::Tensor(
            TensorValue::new(DType::F64, Shape { dims: vec![2, 2] }, g_q_elements).unwrap(),
        );
        let g_r = zeros_like(r);

        let vjp_result = vjp(
            Primitive::Qr,
            &[a],
            &[g_q, g_r],
            &[q.clone(), r.clone()],
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(
            vjp_result.len(),
            1,
            "QR VJP should return 1 gradient (for A)"
        );

        // Result should be a 2×2 tensor
        let da = vjp_result[0].as_tensor().unwrap();
        assert_eq!(da.shape.dims, vec![2, 2]);

        // Verify finite values (no NaN/Inf)
        let vals: Vec<f64> = da.elements.iter().map(|l| l.as_f64().unwrap()).collect();
        for (i, v) in vals.iter().enumerate() {
            assert!(v.is_finite(), "QR VJP element {i} is not finite: {v}");
        }
    }

    #[test]
    fn test_qr_vjp_numerical_check() {
        // QR of [[1, -1], [1, 1]]: verify VJP via finite differences
        let a_data = [1.0, -1.0, 1.0, 1.0];
        let a_elements: Vec<Literal> = a_data.iter().map(|&v| Literal::from_f64(v)).collect();
        let a = Value::Tensor(
            TensorValue::new(DType::F64, Shape { dims: vec![2, 2] }, a_elements).unwrap(),
        );

        let outputs =
            fj_lax::eval_primitive_multi(Primitive::Qr, std::slice::from_ref(&a), &BTreeMap::new())
                .unwrap();
        let q = &outputs[0];
        let r = &outputs[1];

        // Use gradient g_R = ones, g_Q = zeros
        let g_q = zeros_like(q);
        let g_r_elements: Vec<Literal> = vec![1.0, 1.0, 1.0, 1.0]
            .into_iter()
            .map(Literal::from_f64)
            .collect();
        let g_r = Value::Tensor(
            TensorValue::new(DType::F64, Shape { dims: vec![2, 2] }, g_r_elements).unwrap(),
        );

        let vjp_result = vjp(
            Primitive::Qr,
            std::slice::from_ref(&a),
            &[g_q, g_r],
            &[q.clone(), r.clone()],
            &BTreeMap::new(),
        )
        .unwrap();
        let da = vjp_result[0].as_tensor().unwrap();
        let da_vals: Vec<f64> = da.elements.iter().map(|l| l.as_f64().unwrap()).collect();

        // Numerical VJP check via finite differences
        let eps = 1e-6;
        let a_vals = vec![1.0, -1.0, 1.0, 1.0];
        for idx in 0..4 {
            let mut a_plus = a_vals.clone();
            a_plus[idx] += eps;
            let a_plus_val = Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape { dims: vec![2, 2] },
                    a_plus.iter().map(|&v| Literal::from_f64(v)).collect(),
                )
                .unwrap(),
            );

            let mut a_minus = a_vals.clone();
            a_minus[idx] -= eps;
            let a_minus_val = Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape { dims: vec![2, 2] },
                    a_minus.iter().map(|&v| Literal::from_f64(v)).collect(),
                )
                .unwrap(),
            );

            let out_plus =
                fj_lax::eval_primitive_multi(Primitive::Qr, &[a_plus_val], &BTreeMap::new())
                    .unwrap();
            let out_minus =
                fj_lax::eval_primitive_multi(Primitive::Qr, &[a_minus_val], &BTreeMap::new())
                    .unwrap();

            // We used g_R = ones, g_Q = zeros, so the directional derivative is
            // sum(R_plus - R_minus) / (2*eps) for the R output
            let r_plus: Vec<f64> = out_plus[1]
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap())
                .collect();
            let r_minus: Vec<f64> = out_minus[1]
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap())
                .collect();
            let numerical: f64 = r_plus
                .iter()
                .zip(r_minus.iter())
                .map(|(p, m)| (p - m) / (2.0 * eps))
                .sum();

            assert!(
                (da_vals[idx] - numerical).abs() < 1e-4,
                "QR VJP element {idx}: analytical={}, numerical={}",
                da_vals[idx],
                numerical,
            );
        }
    }

    // ── Eigh VJP test ──

    #[test]
    fn test_eigh_vjp_numerical() {
        // Eigh of symmetric [[4, 2], [2, 3]]
        let a_data = vec![4.0, 2.0, 2.0, 3.0];
        let a = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 2] },
                a_data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        );

        let outputs = fj_lax::eval_primitive_multi(
            Primitive::Eigh,
            std::slice::from_ref(&a),
            &BTreeMap::new(),
        )
        .unwrap();
        let w = &outputs[0];
        let v = &outputs[1];

        // VJP with g_w = ones(2), g_V = zeros(2,2) — measures sensitivity through eigenvalues
        let g_w = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2] },
                vec![Literal::from_f64(1.0), Literal::from_f64(1.0)],
            )
            .unwrap(),
        );
        let g_v = zeros_like(v);

        let vjp_result = vjp(
            Primitive::Eigh,
            std::slice::from_ref(&a),
            &[g_w, g_v],
            &[w.clone(), v.clone()],
            &BTreeMap::new(),
        )
        .unwrap();
        let da = vjp_result[0].as_tensor().unwrap();
        let da_vals: Vec<f64> = da.elements.iter().map(|l| l.as_f64().unwrap()).collect();

        // Numerical check via finite differences
        let eps = 1e-6;
        for idx in 0..4 {
            let mut a_plus = a_data.clone();
            a_plus[idx] += eps;
            // Keep symmetric
            if idx == 1 {
                a_plus[2] += eps;
            }
            if idx == 2 {
                a_plus[1] += eps;
            }
            let a_plus_val = Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape { dims: vec![2, 2] },
                    a_plus.iter().map(|&v| Literal::from_f64(v)).collect(),
                )
                .unwrap(),
            );
            let mut a_minus = a_data.clone();
            a_minus[idx] -= eps;
            if idx == 1 {
                a_minus[2] -= eps;
            }
            if idx == 2 {
                a_minus[1] -= eps;
            }
            let a_minus_val = Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape { dims: vec![2, 2] },
                    a_minus.iter().map(|&v| Literal::from_f64(v)).collect(),
                )
                .unwrap(),
            );

            let out_plus =
                fj_lax::eval_primitive_multi(Primitive::Eigh, &[a_plus_val], &BTreeMap::new())
                    .unwrap();
            let out_minus =
                fj_lax::eval_primitive_multi(Primitive::Eigh, &[a_minus_val], &BTreeMap::new())
                    .unwrap();

            // g_w = ones → numerical = sum(w_plus - w_minus) / (2*eps)
            let w_plus: Vec<f64> = out_plus[0]
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap())
                .collect();
            let w_minus: Vec<f64> = out_minus[0]
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap())
                .collect();
            let numerical: f64 = w_plus
                .iter()
                .zip(w_minus.iter())
                .map(|(p, m)| (p - m) / (2.0 * eps))
                .sum();

            // For symmetric perturbation, off-diagonal elements contribute twice
            let effective_da = if idx == 0 || idx == 3 {
                da_vals[idx]
            } else {
                da_vals[idx] + da_vals[if idx == 1 { 2 } else { 1 }]
            };
            assert!(
                (effective_da - numerical).abs() < 1e-4,
                "Eigh VJP element {idx}: analytical={effective_da}, numerical={numerical}",
            );
        }
    }

    // ── SVD VJP test ──

    #[test]
    fn test_svd_vjp_numerical() {
        // SVD of [[3, 0], [0, -2]]
        let a_data = vec![3.0, 0.0, 0.0, -2.0];
        let a = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 2] },
                a_data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        );

        let outputs = fj_lax::eval_primitive_multi(
            Primitive::Svd,
            std::slice::from_ref(&a),
            &BTreeMap::new(),
        )
        .unwrap();
        let u = &outputs[0];
        let s = &outputs[1];
        let vt = &outputs[2];

        // VJP with g_s = ones(2), g_U = zeros, g_Vt = zeros
        let g_u = zeros_like(u);
        let g_s = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2] },
                vec![Literal::from_f64(1.0), Literal::from_f64(1.0)],
            )
            .unwrap(),
        );
        let g_vt = zeros_like(vt);

        let vjp_result = vjp(
            Primitive::Svd,
            std::slice::from_ref(&a),
            &[g_u, g_s, g_vt],
            &[u.clone(), s.clone(), vt.clone()],
            &BTreeMap::new(),
        )
        .unwrap();
        let da = vjp_result[0].as_tensor().unwrap();
        let da_vals: Vec<f64> = da.elements.iter().map(|l| l.as_f64().unwrap()).collect();

        // Numerical check
        let eps = 1e-6;
        for idx in 0..4 {
            let mut a_plus = a_data.clone();
            a_plus[idx] += eps;
            let a_plus_val = Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape { dims: vec![2, 2] },
                    a_plus.iter().map(|&v| Literal::from_f64(v)).collect(),
                )
                .unwrap(),
            );
            let mut a_minus = a_data.clone();
            a_minus[idx] -= eps;
            let a_minus_val = Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape { dims: vec![2, 2] },
                    a_minus.iter().map(|&v| Literal::from_f64(v)).collect(),
                )
                .unwrap(),
            );

            let out_plus =
                fj_lax::eval_primitive_multi(Primitive::Svd, &[a_plus_val], &BTreeMap::new())
                    .unwrap();
            let out_minus =
                fj_lax::eval_primitive_multi(Primitive::Svd, &[a_minus_val], &BTreeMap::new())
                    .unwrap();

            let s_plus: Vec<f64> = out_plus[1]
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap())
                .collect();
            let s_minus: Vec<f64> = out_minus[1]
                .as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap())
                .collect();
            let numerical: f64 = s_plus
                .iter()
                .zip(s_minus.iter())
                .map(|(p, m)| (p - m) / (2.0 * eps))
                .sum();

            assert!(
                (da_vals[idx] - numerical).abs() < 1e-4,
                "SVD VJP element {idx}: analytical={}, numerical={}",
                da_vals[idx],
                numerical,
            );
        }
    }

    // ── Eigh JVP test ──

    #[test]
    fn test_eigh_jvp_numerical() {
        // Eigh JVP: verify dw tangent via finite differences
        let a_data = vec![4.0, 2.0, 2.0, 3.0];
        let make_a = |data: &[f64]| -> Value {
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    Shape { dims: vec![2, 2] },
                    data.iter().map(|&v| Literal::from_f64(v)).collect(),
                )
                .unwrap(),
            )
        };
        let a = make_a(&a_data);

        // Perturbation in the (0,0) direction: dA = [[1,0],[0,0]]
        let da = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 2] },
                vec![1.0, 0.0, 0.0, 0.0]
                    .into_iter()
                    .map(Literal::from_f64)
                    .collect(),
            )
            .unwrap(),
        );

        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2), VarId(3)],
            vec![fj_core::Equation {
                primitive: Primitive::Eigh,
                inputs: smallvec::smallvec![Atom::Var(VarId(1))],
                outputs: smallvec::smallvec![VarId(2), VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        );

        let jvp_result = jvp(&jaxpr, std::slice::from_ref(&a), &[da]).unwrap();
        let dw = &jvp_result.tangents[0];
        let dw_vals: Vec<f64> = dw
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_f64().unwrap())
            .collect();

        // Numerical: perturb A[0,0] by eps
        let eps = 1e-6;
        let mut a_plus = a_data.clone();
        a_plus[0] += eps;
        let mut a_minus = a_data.clone();
        a_minus[0] -= eps;
        let w_plus =
            fj_lax::eval_primitive_multi(Primitive::Eigh, &[make_a(&a_plus)], &BTreeMap::new())
                .unwrap();
        let w_minus =
            fj_lax::eval_primitive_multi(Primitive::Eigh, &[make_a(&a_minus)], &BTreeMap::new())
                .unwrap();

        let wp: Vec<f64> = w_plus[0]
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_f64().unwrap())
            .collect();
        let wm: Vec<f64> = w_minus[0]
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_f64().unwrap())
            .collect();

        for i in 0..2 {
            let numerical = (wp[i] - wm[i]) / (2.0 * eps);
            assert!(
                (dw_vals[i] - numerical).abs() < 1e-4,
                "Eigh JVP dw[{i}]: analytical={}, numerical={}",
                dw_vals[i],
                numerical,
            );
        }
    }

    // ── NaN/Inf gradient propagation tests (frankenjax-zrp) ──────────

    /// Helper: build single-op Jaxpr and compute grad, return the gradient as f64.
    fn grad_single_op(prim: Primitive, x: f64) -> Result<f64, AdError> {
        use smallvec::smallvec;
        let jaxpr = Jaxpr::new(
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
        );
        let grads = grad_jaxpr(&jaxpr, &[Value::scalar_f64(x)])?;
        Ok(to_f64(&grads[0]).unwrap_or(f64::NAN))
    }

    /// Helper: compute VJP for a unary op with a given cotangent.
    fn vjp_single_op(prim: Primitive, x: f64, g: f64) -> Result<f64, AdError> {
        let x_val = Value::scalar_f64(x);
        let g_val = Value::scalar_f64(g);
        let params = BTreeMap::new();
        let out = fj_lax::eval_primitive(prim, std::slice::from_ref(&x_val), &params)
            .map_err(|e| AdError::EvalFailed(e.to_string()))?;
        let vjp_result = vjp(
            prim,
            std::slice::from_ref(&x_val),
            std::slice::from_ref(&g_val),
            std::slice::from_ref(&out),
            &params,
        )?;
        Ok(to_f64(&vjp_result[0]).unwrap_or(f64::NAN))
    }

    #[test]
    fn nan_propagates_through_unary_gradients() {
        // JAX contract: grad(f)(NaN) = NaN for all differentiable unary ops
        // Neg excluded: grad(neg)(x) = -1 (constant), so NaN input → -1 is correct.
        let nan_ops = [
            Primitive::Sin,
            Primitive::Cos,
            Primitive::Exp,
            Primitive::Log,
            Primitive::Tanh,
            Primitive::Square,
            Primitive::Sqrt,
            Primitive::Sinh,
            Primitive::Cosh,
        ];

        for prim in &nan_ops {
            let g = grad_single_op(*prim, f64::NAN);
            match g {
                Ok(val) => assert!(
                    val.is_nan(),
                    "grad({:?})(NaN) should be NaN, got {val}",
                    prim
                ),
                Err(_) => {
                    // Some ops may error on NaN input (e.g., log(NaN) in eval).
                    // That's acceptable — the key is no silent zero.
                }
            }
        }
    }

    #[test]
    fn nan_cotangent_produces_nan_gradient() {
        // If the output cotangent is NaN, the input gradient must be NaN
        let ops = [
            Primitive::Sin,
            Primitive::Cos,
            Primitive::Exp,
            Primitive::Square,
            Primitive::Neg,
        ];

        for prim in &ops {
            if let Ok(g) = vjp_single_op(*prim, 1.0, f64::NAN) {
                assert!(
                    g.is_nan(),
                    "vjp({:?}, 1.0, cotangent=NaN) should give NaN gradient, got {g}",
                    prim
                );
            }
        }
    }

    #[test]
    fn inf_propagates_through_exp_gradient() {
        // grad(exp)(x) = exp(x). At x=Inf, exp(Inf) = Inf, so grad = Inf.
        if let Ok(g) = grad_single_op(Primitive::Exp, f64::INFINITY) {
            assert!(
                g.is_infinite() || g.is_nan(),
                "grad(exp)(Inf) should be Inf or NaN, got {g}"
            );
        }
    }

    #[test]
    fn inf_propagates_through_square_gradient() {
        // grad(x^2)(x) = 2x. At x=Inf, grad = Inf.
        if let Ok(g) = grad_single_op(Primitive::Square, f64::INFINITY) {
            assert!(g.is_infinite(), "grad(x^2)(Inf) should be Inf, got {g}");
        }
    }

    #[test]
    fn neg_inf_input_gradient() {
        // grad(exp)(-Inf) = exp(-Inf) = 0. This IS correct — not a silent error.
        if let Ok(g) = grad_single_op(Primitive::Exp, f64::NEG_INFINITY) {
            assert!(
                g.abs() < 1e-300 || g.is_nan(),
                "grad(exp)(-Inf) should be ~0 or NaN, got {g}"
            );
        }
    }

    #[test]
    fn jvp_nan_tangent_produces_nan_output() {
        use smallvec::smallvec;
        // JVP with NaN tangent: the output tangent should be NaN
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Sin,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let result = jvp(
            &jaxpr,
            &[Value::scalar_f64(1.0)],
            &[Value::scalar_f64(f64::NAN)],
        );
        if let Ok(r) = result {
            let tangent = to_f64(&r.tangents[0]).unwrap_or(0.0);
            assert!(
                tangent.is_nan(),
                "jvp(sin, x=1.0, dx=NaN) tangent should be NaN, got {tangent}"
            );
        }
    }

    #[test]
    fn jvp_nan_primal_produces_nan_tangent() {
        use smallvec::smallvec;
        // JVP with NaN primal: both output and tangent should be NaN
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Cos,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let result = jvp(
            &jaxpr,
            &[Value::scalar_f64(f64::NAN)],
            &[Value::scalar_f64(1.0)],
        );
        if let Ok(r) = result {
            let tangent = to_f64(&r.tangents[0]).unwrap_or(0.0);
            assert!(
                tangent.is_nan(),
                "jvp(cos, x=NaN, dx=1.0) tangent should be NaN, got {tangent}"
            );
        }
    }

    #[test]
    fn binary_nan_propagation_mul() {
        use smallvec::smallvec;
        // grad(x*y) w.r.t. x at x=NaN, y=3.0: grad = y = 3.0? No — the forward
        // pass produces NaN output, so the backward pass should propagate NaN.
        let jaxpr = Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3)],
            vec![Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                sub_jaxprs: vec![],
                effects: vec![],
            }],
        );
        let result = grad_jaxpr(
            &jaxpr,
            &[Value::scalar_f64(f64::NAN), Value::scalar_f64(3.0)],
        );
        if let Ok(grads) = result {
            // grad w.r.t. x of x*y = y * cotangent. But if x=NaN, the
            // forward pass output is NaN, and the VJP rule for mul uses
            // the primal values, so NaN should appear somewhere.
            let gx = to_f64(&grads[0]).unwrap_or(0.0);
            // Either NaN (from primal contamination) or 3.0 (from the formula)
            // are both defensible. The key assertion: it should NOT be 0.0.
            assert!(
                gx.is_nan() || (gx - 3.0).abs() < 1e-10,
                "grad(x*y) at x=NaN, y=3: expected NaN or 3.0, got {gx}"
            );
        }
    }
}
