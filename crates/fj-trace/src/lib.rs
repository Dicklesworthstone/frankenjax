#![forbid(unsafe_code)]

//! Tracing and abstract-value machinery for FJ-P2C-001.

use fj_core::{Atom, DType, Equation, Jaxpr, Primitive, Shape, Value, VarId};
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TracerId(pub u32);

pub trait AbstractValue {
    fn dtype(&self) -> DType;
    fn shape(&self) -> &Shape;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapedArray {
    pub dtype: DType,
    pub shape: Shape,
}

impl AbstractValue for ShapedArray {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcreteArray {
    pub dtype: DType,
    pub shape: Shape,
    pub is_known_constant: bool,
}

impl AbstractValue for ConcreteArray {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

impl ShapedArray {
    #[must_use]
    pub fn from_value(value: &Value) -> Self {
        match value {
            Value::Scalar(lit) => {
                let dtype = match lit {
                    fj_core::Literal::I64(_) => DType::I64,
                    fj_core::Literal::Bool(_) => DType::Bool,
                    fj_core::Literal::F64Bits(_) => DType::F64,
                };
                Self {
                    dtype,
                    shape: Shape::scalar(),
                }
            }
            Value::Tensor(tensor) => Self {
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
            },
        }
    }
}

pub trait Tracer {
    fn tracer_id(&self) -> TracerId;
    fn abstract_value(&self) -> &dyn AbstractValue;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasicTracer {
    tracer_id: TracerId,
    aval: ShapedArray,
}

impl BasicTracer {
    #[must_use]
    pub fn new(tracer_id: TracerId, aval: ShapedArray) -> Self {
        Self { tracer_id, aval }
    }
}

impl Tracer for BasicTracer {
    fn tracer_id(&self) -> TracerId {
        self.tracer_id
    }

    fn abstract_value(&self) -> &dyn AbstractValue {
        &self.aval
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JaxprTrace {
    pub trace_id: u64,
    pub in_avals: Vec<ShapedArray>,
    pub out_avals: Vec<ShapedArray>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClosedJaxpr {
    pub jaxpr: Jaxpr,
    pub const_values: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceError {
    UnsupportedPrimitive {
        primitive: Primitive,
    },
    InvalidAbstractValue,
    UnboundTracerInput {
        tracer_id: TracerId,
    },
    OutputShadowing {
        tracer_id: TracerId,
    },
    UnresolvedOutvar {
        tracer_id: TracerId,
    },
    CompositionViolation,
    MissingPrimitiveParam {
        primitive: Primitive,
        key: &'static str,
    },
    InvalidPrimitiveParam {
        primitive: Primitive,
        key: &'static str,
        value: String,
    },
    ShapeInferenceFailed {
        primitive: Primitive,
        detail: String,
    },
    NestedTraceNotClosed,
    UnknownTrace {
        trace_id: u64,
    },
}

impl std::fmt::Display for TraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedPrimitive { primitive } => {
                write!(f, "unsupported primitive {}", primitive.as_str())
            }
            Self::InvalidAbstractValue => write!(f, "invalid abstract value"),
            Self::UnboundTracerInput { tracer_id } => {
                write!(f, "unbound tracer input id {}", tracer_id.0)
            }
            Self::OutputShadowing { tracer_id } => {
                write!(f, "output shadows an existing tracer id {}", tracer_id.0)
            }
            Self::UnresolvedOutvar { tracer_id } => {
                write!(
                    f,
                    "unable to resolve output var for tracer id {}",
                    tracer_id.0
                )
            }
            Self::CompositionViolation => write!(f, "trace composition invariant violated"),
            Self::MissingPrimitiveParam { primitive, key } => {
                write!(
                    f,
                    "missing primitive parameter {} for {}",
                    key,
                    primitive.as_str()
                )
            }
            Self::InvalidPrimitiveParam {
                primitive,
                key,
                value,
            } => {
                write!(
                    f,
                    "invalid primitive parameter {}={} for {}",
                    key,
                    value,
                    primitive.as_str()
                )
            }
            Self::ShapeInferenceFailed { primitive, detail } => {
                write!(
                    f,
                    "shape inference failed for {}: {}",
                    primitive.as_str(),
                    detail
                )
            }
            Self::NestedTraceNotClosed => {
                write!(f, "nested trace frame not closed before finalize")
            }
            Self::UnknownTrace { trace_id } => write!(f, "unknown trace id {}", trace_id),
        }
    }
}

impl std::error::Error for TraceError {}

pub trait TraceContext {
    fn process_primitive(
        &mut self,
        primitive: Primitive,
        input_ids: &[TracerId],
        params: BTreeMap<String, String>,
    ) -> Result<Vec<TracerId>, TraceError>;

    fn finalize(self) -> Result<ClosedJaxpr, TraceError>
    where
        Self: Sized;
}

pub trait TraceToJaxpr {
    fn trace_to_jaxpr(&mut self, trace: JaxprTrace) -> Result<ClosedJaxpr, TraceError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TraceEquation {
    primitive: Primitive,
    inputs: Vec<TracerId>,
    outputs: Vec<TracerId>,
    params: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TraceFrame {
    trace_id: u64,
    in_ids: Vec<TracerId>,
    const_ids: Vec<TracerId>,
    equations: Vec<TraceEquation>,
    last_output_ids: Vec<TracerId>,
}

#[derive(Debug, Clone)]
pub struct SimpleTraceContext {
    next_tracer_id: u32,
    next_trace_id: u64,
    tracer_avals: BTreeMap<TracerId, ShapedArray>,
    const_values: BTreeMap<TracerId, Value>,
    frame_stack: Vec<TraceFrame>,
}

impl Default for SimpleTraceContext {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleTraceContext {
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_tracer_id: 1,
            next_trace_id: 2,
            tracer_avals: BTreeMap::new(),
            const_values: BTreeMap::new(),
            frame_stack: vec![TraceFrame {
                trace_id: 1,
                in_ids: Vec::new(),
                const_ids: Vec::new(),
                equations: Vec::new(),
                last_output_ids: Vec::new(),
            }],
        }
    }

    #[must_use]
    pub fn with_inputs(in_avals: Vec<ShapedArray>) -> Self {
        let mut ctx = Self::new();
        for aval in in_avals {
            let _ = ctx.bind_input(aval);
        }
        ctx
    }

    #[must_use]
    pub fn active_trace_id(&self) -> u64 {
        self.frame_stack
            .last()
            .map(|frame| frame.trace_id)
            .unwrap_or(1)
    }

    pub fn bind_input(&mut self, aval: ShapedArray) -> TracerId {
        let tracer_id = self.allocate_tracer(aval);
        self.active_frame_mut().in_ids.push(tracer_id);
        tracer_id
    }

    pub fn bind_const_value(&mut self, value: Value) -> TracerId {
        let aval = ShapedArray::from_value(&value);
        let tracer_id = self.allocate_tracer(aval);
        self.const_values.insert(tracer_id, value);
        self.active_frame_mut().const_ids.push(tracer_id);
        tracer_id
    }

    pub fn push_subtrace(&mut self, in_avals: Vec<ShapedArray>) -> u64 {
        let trace_id = self.next_trace_id;
        self.next_trace_id += 1;

        let mut frame = TraceFrame {
            trace_id,
            in_ids: Vec::with_capacity(in_avals.len()),
            const_ids: Vec::new(),
            equations: Vec::new(),
            last_output_ids: Vec::new(),
        };

        for aval in in_avals {
            let tracer_id = self.allocate_tracer(aval);
            frame.in_ids.push(tracer_id);
        }

        self.frame_stack.push(frame);
        trace_id
    }

    pub fn pop_subtrace(&mut self) -> Result<u64, TraceError> {
        if self.frame_stack.len() <= 1 {
            return Err(TraceError::CompositionViolation);
        }
        let frame = self
            .frame_stack
            .pop()
            .ok_or(TraceError::CompositionViolation)?;
        Ok(frame.trace_id)
    }

    fn active_frame(&self) -> &TraceFrame {
        self.frame_stack
            .last()
            .expect("trace context should always have at least one frame")
    }

    fn active_frame_mut(&mut self) -> &mut TraceFrame {
        self.frame_stack
            .last_mut()
            .expect("trace context should always have at least one frame")
    }

    fn allocate_tracer(&mut self, aval: ShapedArray) -> TracerId {
        let tracer_id = TracerId(self.next_tracer_id);
        self.next_tracer_id += 1;
        self.tracer_avals.insert(tracer_id, aval);
        tracer_id
    }

    fn tracer_aval(&self, tracer_id: TracerId) -> Result<&ShapedArray, TraceError> {
        self.tracer_avals
            .get(&tracer_id)
            .ok_or(TraceError::UnboundTracerInput { tracer_id })
    }

    fn infer_primitive_output_avals(
        primitive: Primitive,
        inputs: &[ShapedArray],
        params: &BTreeMap<String, String>,
    ) -> Result<Vec<ShapedArray>, TraceError> {
        match primitive {
            // Binary elementwise: output shape = broadcast(lhs, rhs)
            Primitive::Add
            | Primitive::Sub
            | Primitive::Mul
            | Primitive::Max
            | Primitive::Min
            | Primitive::Pow
            | Primitive::Div
            | Primitive::Rem
            | Primitive::Atan2 => {
                if inputs.len() != 2 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected 2 inputs, got {}", inputs.len()),
                    });
                }
                let shape = broadcast_shape(&inputs[0].shape, &inputs[1].shape).ok_or(
                    TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!(
                            "cannot broadcast {:?} with {:?}",
                            inputs[0].shape.dims, inputs[1].shape.dims
                        ),
                    },
                )?;
                let dtype = promote_dtype(inputs[0].dtype, inputs[1].dtype);
                Ok(vec![ShapedArray { dtype, shape }])
            }
            // Comparison: output shape = broadcast(lhs, rhs), dtype = Bool
            Primitive::Eq
            | Primitive::Ne
            | Primitive::Lt
            | Primitive::Le
            | Primitive::Gt
            | Primitive::Ge => {
                if inputs.len() != 2 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected 2 inputs, got {}", inputs.len()),
                    });
                }
                let shape = broadcast_shape(&inputs[0].shape, &inputs[1].shape).ok_or(
                    TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!(
                            "cannot broadcast {:?} with {:?}",
                            inputs[0].shape.dims, inputs[1].shape.dims
                        ),
                    },
                )?;
                Ok(vec![ShapedArray {
                    dtype: DType::Bool,
                    shape,
                }])
            }
            Primitive::Dot => infer_dot(inputs),
            // Unary elementwise: output shape = input shape
            Primitive::Neg
            | Primitive::Abs
            | Primitive::Exp
            | Primitive::Log
            | Primitive::Sqrt
            | Primitive::Rsqrt
            | Primitive::Floor
            | Primitive::Ceil
            | Primitive::Round
            | Primitive::Sin
            | Primitive::Cos
            | Primitive::Tan
            | Primitive::Asin
            | Primitive::Acos
            | Primitive::Atan
            | Primitive::Sinh
            | Primitive::Cosh
            | Primitive::Tanh
            | Primitive::Expm1
            | Primitive::Log1p
            | Primitive::Sign
            | Primitive::Square
            | Primitive::Reciprocal
            | Primitive::Logistic
            | Primitive::Erf
            | Primitive::Erfc => {
                if inputs.len() != 1 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected 1 input, got {}", inputs.len()),
                    });
                }
                Ok(vec![inputs[0].clone()])
            }
            // Reductions: all use the same reduce shape inference
            Primitive::ReduceSum
            | Primitive::ReduceMax
            | Primitive::ReduceMin
            | Primitive::ReduceProd => infer_reduce_sum(primitive, inputs, params),
            Primitive::Reshape => infer_reshape(inputs, params),
            Primitive::Slice => infer_slice(inputs, params),
            Primitive::Gather => infer_gather(inputs, params),
            Primitive::Scatter => infer_scatter(inputs),
            Primitive::Transpose => infer_transpose(inputs, params),
            Primitive::BroadcastInDim => infer_broadcast_in_dim(inputs, params),
            Primitive::Select => {
                if inputs.len() != 3 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected 3 inputs, got {}", inputs.len()),
                    });
                }
                if inputs[0].shape != inputs[1].shape || inputs[0].shape != inputs[2].shape {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!(
                            "select requires matching shapes, got {:?}, {:?}, {:?}",
                            inputs[0].shape.dims, inputs[1].shape.dims, inputs[2].shape.dims
                        ),
                    });
                }
                Ok(vec![ShapedArray {
                    dtype: promote_dtype(inputs[1].dtype, inputs[2].dtype),
                    shape: inputs[1].shape.clone(),
                }])
            }
            Primitive::Concatenate => infer_concatenate(inputs, params),
        }
    }

    fn build_closed_jaxpr(mut self, frame: TraceFrame) -> Result<ClosedJaxpr, TraceError> {
        let mut tracer_to_var: BTreeMap<TracerId, VarId> = BTreeMap::new();
        let mut next_var = 1_u32;

        let ensure_var = |tracer_to_var: &mut BTreeMap<TracerId, VarId>,
                          next_var: &mut u32,
                          tracer_id: TracerId| {
            *tracer_to_var.entry(tracer_id).or_insert_with(|| {
                let var = VarId(*next_var);
                *next_var += 1;
                var
            })
        };

        for tracer_id in &frame.in_ids {
            ensure_var(&mut tracer_to_var, &mut next_var, *tracer_id);
        }
        for tracer_id in &frame.const_ids {
            ensure_var(&mut tracer_to_var, &mut next_var, *tracer_id);
        }

        let mut equations = Vec::with_capacity(frame.equations.len());
        for eqn in &frame.equations {
            let mut in_atoms: SmallVec<[Atom; 4]> = SmallVec::with_capacity(eqn.inputs.len());
            for input_id in &eqn.inputs {
                let var =
                    tracer_to_var
                        .get(input_id)
                        .copied()
                        .ok_or(TraceError::UnboundTracerInput {
                            tracer_id: *input_id,
                        })?;
                in_atoms.push(Atom::Var(var));
            }

            let mut out_vars: SmallVec<[VarId; 2]> = SmallVec::with_capacity(eqn.outputs.len());
            for output_id in &eqn.outputs {
                if tracer_to_var.contains_key(output_id) {
                    return Err(TraceError::OutputShadowing {
                        tracer_id: *output_id,
                    });
                }
                let out_var = ensure_var(&mut tracer_to_var, &mut next_var, *output_id);
                out_vars.push(out_var);
            }

            equations.push(Equation {
                primitive: eqn.primitive,
                inputs: in_atoms,
                outputs: out_vars,
                params: eqn.params.clone(),
            });
        }

        let invars = frame
            .in_ids
            .iter()
            .map(|tracer_id| {
                tracer_to_var
                    .get(tracer_id)
                    .copied()
                    .ok_or(TraceError::UnboundTracerInput {
                        tracer_id: *tracer_id,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let constvars = frame
            .const_ids
            .iter()
            .map(|tracer_id| {
                tracer_to_var
                    .get(tracer_id)
                    .copied()
                    .ok_or(TraceError::UnboundTracerInput {
                        tracer_id: *tracer_id,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let out_ids = if frame.last_output_ids.is_empty() {
            frame.in_ids.clone()
        } else {
            frame.last_output_ids.clone()
        };

        let outvars = out_ids
            .iter()
            .map(|tracer_id| {
                tracer_to_var
                    .get(tracer_id)
                    .copied()
                    .ok_or(TraceError::UnresolvedOutvar {
                        tracer_id: *tracer_id,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let const_values = frame
            .const_ids
            .iter()
            .map(|tracer_id| {
                self.const_values
                    .remove(tracer_id)
                    .ok_or(TraceError::UnboundTracerInput {
                        tracer_id: *tracer_id,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(ClosedJaxpr {
            jaxpr: Jaxpr::new(invars, constvars, outvars, equations),
            const_values,
        })
    }
}

impl TraceContext for SimpleTraceContext {
    fn process_primitive(
        &mut self,
        primitive: Primitive,
        input_ids: &[TracerId],
        params: BTreeMap<String, String>,
    ) -> Result<Vec<TracerId>, TraceError> {
        let input_avals = input_ids
            .iter()
            .map(|tracer_id| self.tracer_aval(*tracer_id).cloned())
            .collect::<Result<Vec<_>, _>>()?;

        let output_avals = Self::infer_primitive_output_avals(primitive, &input_avals, &params)?;
        let output_ids = output_avals
            .into_iter()
            .map(|aval| self.allocate_tracer(aval))
            .collect::<Vec<_>>();

        let frame = self.active_frame_mut();
        frame.equations.push(TraceEquation {
            primitive,
            inputs: input_ids.to_vec(),
            outputs: output_ids.clone(),
            params,
        });
        frame.last_output_ids = output_ids.clone();

        Ok(output_ids)
    }

    fn finalize(self) -> Result<ClosedJaxpr, TraceError>
    where
        Self: Sized,
    {
        let mut this = self;
        if this.frame_stack.len() != 1 {
            return Err(TraceError::NestedTraceNotClosed);
        }

        let frame = this
            .frame_stack
            .pop()
            .ok_or(TraceError::CompositionViolation)?;

        this.build_closed_jaxpr(frame)
    }
}

impl TraceToJaxpr for SimpleTraceContext {
    fn trace_to_jaxpr(&mut self, trace: JaxprTrace) -> Result<ClosedJaxpr, TraceError> {
        if trace.trace_id != self.active_trace_id() {
            return Err(TraceError::UnknownTrace {
                trace_id: trace.trace_id,
            });
        }

        let frame = self.active_frame();
        if trace.in_avals.len() != frame.in_ids.len() {
            return Err(TraceError::InvalidAbstractValue);
        }

        for (idx, tracer_id) in frame.in_ids.iter().enumerate() {
            let Some(actual_aval) = self.tracer_avals.get(tracer_id) else {
                return Err(TraceError::UnboundTracerInput {
                    tracer_id: *tracer_id,
                });
            };
            if actual_aval != &trace.in_avals[idx] {
                return Err(TraceError::InvalidAbstractValue);
            }
        }

        if !trace.out_avals.is_empty() {
            if trace.out_avals.len() != frame.last_output_ids.len() {
                return Err(TraceError::InvalidAbstractValue);
            }
            for (idx, tracer_id) in frame.last_output_ids.iter().enumerate() {
                let Some(actual_aval) = self.tracer_avals.get(tracer_id) else {
                    return Err(TraceError::UnboundTracerInput {
                        tracer_id: *tracer_id,
                    });
                };
                if actual_aval != &trace.out_avals[idx] {
                    return Err(TraceError::InvalidAbstractValue);
                }
            }
        }

        let owned = std::mem::take(self);
        owned.finalize()
    }
}

fn infer_dot(inputs: &[ShapedArray]) -> Result<Vec<ShapedArray>, TraceError> {
    let primitive = Primitive::Dot;
    if inputs.len() != 2 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!("expected 2 inputs, got {}", inputs.len()),
        });
    }

    let lhs = &inputs[0];
    let rhs = &inputs[1];
    let out_shape = match (lhs.shape.rank(), rhs.shape.rank()) {
        (0, 0) => Shape::scalar(),
        (1, 1) => {
            if lhs.shape.dims[0] != rhs.shape.dims[0] {
                return Err(TraceError::ShapeInferenceFailed {
                    primitive,
                    detail: format!(
                        "dot rank-1 mismatch lhs={:?} rhs={:?}",
                        lhs.shape.dims, rhs.shape.dims
                    ),
                });
            }
            Shape::scalar()
        }
        (2, 1) => {
            if lhs.shape.dims[1] != rhs.shape.dims[0] {
                return Err(TraceError::ShapeInferenceFailed {
                    primitive,
                    detail: format!(
                        "dot rank-2x1 mismatch lhs={:?} rhs={:?}",
                        lhs.shape.dims, rhs.shape.dims
                    ),
                });
            }
            Shape {
                dims: vec![lhs.shape.dims[0]],
            }
        }
        (1, 2) => {
            if lhs.shape.dims[0] != rhs.shape.dims[0] {
                return Err(TraceError::ShapeInferenceFailed {
                    primitive,
                    detail: format!(
                        "dot rank-1x2 mismatch lhs={:?} rhs={:?}",
                        lhs.shape.dims, rhs.shape.dims
                    ),
                });
            }
            Shape {
                dims: vec![rhs.shape.dims[1]],
            }
        }
        (2, 2) => {
            if lhs.shape.dims[1] != rhs.shape.dims[0] {
                return Err(TraceError::ShapeInferenceFailed {
                    primitive,
                    detail: format!(
                        "dot rank-2x2 mismatch lhs={:?} rhs={:?}",
                        lhs.shape.dims, rhs.shape.dims
                    ),
                });
            }
            Shape {
                dims: vec![lhs.shape.dims[0], rhs.shape.dims[1]],
            }
        }
        _ => {
            return Err(TraceError::ShapeInferenceFailed {
                primitive,
                detail: format!(
                    "dot supports rank combinations (0,0), (1,1), (2,1), (1,2), (2,2); got ({},{})",
                    lhs.shape.rank(),
                    rhs.shape.rank()
                ),
            });
        }
    };

    Ok(vec![ShapedArray {
        dtype: promote_dtype(lhs.dtype, rhs.dtype),
        shape: out_shape,
    }])
}

fn infer_reduce_sum(
    primitive: Primitive,
    inputs: &[ShapedArray],
    params: &BTreeMap<String, String>,
) -> Result<Vec<ShapedArray>, TraceError> {
    if inputs.len() != 1 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!("expected 1 input, got {}", inputs.len()),
        });
    }

    let input = &inputs[0];
    let mut out_dims = input.shape.dims.clone();
    if let Some(raw_axes) = params.get("axes") {
        let axes = parse_usize_list(primitive, "axes", raw_axes)?;
        let axes_set = axes.into_iter().collect::<BTreeSet<_>>();
        for axis in axes_set.iter().rev() {
            if *axis >= out_dims.len() {
                return Err(TraceError::ShapeInferenceFailed {
                    primitive,
                    detail: format!(
                        "axis {} out of bounds for shape {:?}",
                        axis, input.shape.dims
                    ),
                });
            }
            out_dims.remove(*axis);
        }
    } else {
        out_dims.clear();
    }

    Ok(vec![ShapedArray {
        dtype: input.dtype,
        shape: Shape { dims: out_dims },
    }])
}

fn infer_reshape(
    inputs: &[ShapedArray],
    params: &BTreeMap<String, String>,
) -> Result<Vec<ShapedArray>, TraceError> {
    let primitive = Primitive::Reshape;
    if inputs.len() != 1 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!("expected 1 input, got {}", inputs.len()),
        });
    }

    let input = &inputs[0];
    let raw_shape = params
        .get("new_shape")
        .ok_or(TraceError::MissingPrimitiveParam {
            primitive,
            key: "new_shape",
        })?;
    let shape_spec = parse_i64_list(primitive, "new_shape", raw_shape)?;

    let mut inferred_axis = None;
    let mut known_product = 1_u64;
    let mut dims = Vec::with_capacity(shape_spec.len());
    for (idx, dim) in shape_spec.iter().enumerate() {
        if *dim == -1 {
            if inferred_axis.is_some() {
                return Err(TraceError::ShapeInferenceFailed {
                    primitive,
                    detail: "only one -1 inferred axis allowed".to_owned(),
                });
            }
            inferred_axis = Some(idx);
            dims.push(0_u32);
            continue;
        }
        if *dim <= 0 {
            return Err(TraceError::ShapeInferenceFailed {
                primitive,
                detail: format!("reshape dimension must be positive, got {}", dim),
            });
        }
        let dim_u32 = u32::try_from(*dim).map_err(|_| TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!("reshape dimension out of range: {}", dim),
        })?;
        known_product = known_product
            .checked_mul(u64::from(dim_u32))
            .ok_or_else(|| TraceError::ShapeInferenceFailed {
                primitive,
                detail: "reshape target element count overflow".to_owned(),
            })?;
        dims.push(dim_u32);
    }

    let input_elements =
        input
            .shape
            .element_count()
            .ok_or_else(|| TraceError::ShapeInferenceFailed {
                primitive,
                detail: "input shape overflow".to_owned(),
            })?;

    if let Some(infer_idx) = inferred_axis {
        if known_product == 0 || input_elements % known_product != 0 {
            return Err(TraceError::ShapeInferenceFailed {
                primitive,
                detail: format!(
                    "cannot infer reshape dimension: input elements {} not divisible by {}",
                    input_elements, known_product
                ),
            });
        }
        dims[infer_idx] = (input_elements / known_product) as u32;
    }

    let target_elements = dims
        .iter()
        .try_fold(1_u64, |acc, dim| acc.checked_mul(u64::from(*dim)))
        .ok_or_else(|| TraceError::ShapeInferenceFailed {
            primitive,
            detail: "target shape overflow".to_owned(),
        })?;

    if target_elements != input_elements {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!(
                "reshape element mismatch: input={} target={}",
                input_elements, target_elements
            ),
        });
    }

    Ok(vec![ShapedArray {
        dtype: input.dtype,
        shape: Shape { dims },
    }])
}

fn infer_slice(
    inputs: &[ShapedArray],
    params: &BTreeMap<String, String>,
) -> Result<Vec<ShapedArray>, TraceError> {
    let primitive = Primitive::Slice;
    if inputs.len() != 1 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!("expected 1 input, got {}", inputs.len()),
        });
    }

    let input = &inputs[0];
    let starts = parse_u32_param_list(primitive, params, "start_indices")?;
    let limits = parse_u32_param_list(primitive, params, "limit_indices")?;

    if starts.len() != input.shape.rank() || limits.len() != input.shape.rank() {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!(
                "slice start/limit rank mismatch starts={} limits={} rank={}",
                starts.len(),
                limits.len(),
                input.shape.rank()
            ),
        });
    }

    let mut dims = Vec::with_capacity(starts.len());
    for axis in 0..starts.len() {
        let start = starts[axis];
        let limit = limits[axis];
        let bound = input.shape.dims[axis];
        if start > limit || limit > bound {
            return Err(TraceError::ShapeInferenceFailed {
                primitive,
                detail: format!(
                    "invalid slice bounds axis {}: start={} limit={} bound={}",
                    axis, start, limit, bound
                ),
            });
        }
        dims.push(limit - start);
    }

    Ok(vec![ShapedArray {
        dtype: input.dtype,
        shape: Shape { dims },
    }])
}

fn infer_gather(
    inputs: &[ShapedArray],
    params: &BTreeMap<String, String>,
) -> Result<Vec<ShapedArray>, TraceError> {
    let primitive = Primitive::Gather;
    if inputs.len() < 2 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!("expected at least 2 inputs, got {}", inputs.len()),
        });
    }

    let operand = &inputs[0];
    let indices = &inputs[1];
    let slice_sizes = if let Some(raw) = params.get("slice_sizes") {
        parse_u32_list(primitive, "slice_sizes", raw)?
    } else {
        vec![1_u32; operand.shape.rank()]
    };

    let mut dims = indices.shape.dims.clone();
    dims.extend(slice_sizes);

    Ok(vec![ShapedArray {
        dtype: operand.dtype,
        shape: Shape { dims },
    }])
}

fn infer_scatter(inputs: &[ShapedArray]) -> Result<Vec<ShapedArray>, TraceError> {
    let primitive = Primitive::Scatter;
    if inputs.len() < 2 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!("expected at least 2 inputs, got {}", inputs.len()),
        });
    }
    Ok(vec![inputs[0].clone()])
}

fn infer_transpose(
    inputs: &[ShapedArray],
    params: &BTreeMap<String, String>,
) -> Result<Vec<ShapedArray>, TraceError> {
    let primitive = Primitive::Transpose;
    if inputs.len() != 1 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!("expected 1 input, got {}", inputs.len()),
        });
    }

    let input = &inputs[0];
    let rank = input.shape.rank();
    let permutation = if let Some(raw) = params.get("permutation") {
        parse_usize_list(primitive, "permutation", raw)?
    } else {
        (0..rank).rev().collect()
    };

    if permutation.len() != rank {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!(
                "permutation rank mismatch: perm={} rank={}",
                permutation.len(),
                rank
            ),
        });
    }

    let mut seen = BTreeSet::new();
    let mut dims = Vec::with_capacity(rank);
    for axis in permutation {
        if axis >= rank || !seen.insert(axis) {
            return Err(TraceError::ShapeInferenceFailed {
                primitive,
                detail: "permutation must be a unique in-range axis list".to_owned(),
            });
        }
        dims.push(input.shape.dims[axis]);
    }

    Ok(vec![ShapedArray {
        dtype: input.dtype,
        shape: Shape { dims },
    }])
}

fn infer_broadcast_in_dim(
    inputs: &[ShapedArray],
    params: &BTreeMap<String, String>,
) -> Result<Vec<ShapedArray>, TraceError> {
    let primitive = Primitive::BroadcastInDim;
    if inputs.len() != 1 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!("expected 1 input, got {}", inputs.len()),
        });
    }

    let input = &inputs[0];
    let target_dims = parse_u32_param_list(primitive, params, "shape")?;

    if let Some(raw) = params.get("broadcast_dimensions") {
        let dims = parse_usize_list(primitive, "broadcast_dimensions", raw)?;
        if dims.len() != input.shape.rank() {
            return Err(TraceError::ShapeInferenceFailed {
                primitive,
                detail: format!(
                    "broadcast_dimensions length {} must equal input rank {}",
                    dims.len(),
                    input.shape.rank()
                ),
            });
        }

        for (input_axis, target_axis) in dims.iter().enumerate() {
            if *target_axis >= target_dims.len() {
                return Err(TraceError::ShapeInferenceFailed {
                    primitive,
                    detail: format!("target axis {} out of range", target_axis),
                });
            }
            let in_dim = input.shape.dims[input_axis];
            let target_dim = target_dims[*target_axis];
            if in_dim != 1 && in_dim != target_dim {
                return Err(TraceError::ShapeInferenceFailed {
                    primitive,
                    detail: format!(
                        "cannot broadcast input dim {} into target dim {} at axis {}",
                        in_dim, target_dim, target_axis
                    ),
                });
            }
        }
    } else {
        if input.shape.rank() > target_dims.len() {
            return Err(TraceError::ShapeInferenceFailed {
                primitive,
                detail: "input rank cannot exceed broadcast target rank".to_owned(),
            });
        }

        let offset = target_dims.len() - input.shape.rank();
        for (idx, in_dim) in input.shape.dims.iter().enumerate() {
            let target_dim = target_dims[offset + idx];
            if *in_dim != 1 && *in_dim != target_dim {
                return Err(TraceError::ShapeInferenceFailed {
                    primitive,
                    detail: format!(
                        "cannot broadcast input dim {} into trailing target dim {}",
                        in_dim, target_dim
                    ),
                });
            }
        }
    }

    Ok(vec![ShapedArray {
        dtype: input.dtype,
        shape: Shape { dims: target_dims },
    }])
}

fn infer_concatenate(
    inputs: &[ShapedArray],
    params: &BTreeMap<String, String>,
) -> Result<Vec<ShapedArray>, TraceError> {
    let primitive = Primitive::Concatenate;
    if inputs.is_empty() {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: "expected at least 1 input".to_owned(),
        });
    }

    let axis = if let Some(raw_axis) = params.get("dimension") {
        raw_axis
            .parse::<usize>()
            .map_err(|_| TraceError::InvalidPrimitiveParam {
                primitive,
                key: "dimension",
                value: raw_axis.clone(),
            })?
    } else {
        0
    };

    let rank = inputs[0].shape.rank();
    if axis >= rank {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!("concat axis {} out of bounds for rank {}", axis, rank),
        });
    }

    let mut out_dims = inputs[0].shape.dims.clone();
    let mut out_dtype = inputs[0].dtype;
    for input in &inputs[1..] {
        if input.shape.rank() != rank {
            return Err(TraceError::ShapeInferenceFailed {
                primitive,
                detail: "concat rank mismatch across inputs".to_owned(),
            });
        }
        for (dim_idx, (expected_dim, actual_dim)) in
            out_dims.iter().zip(input.shape.dims.iter()).enumerate()
        {
            if dim_idx == axis {
                continue;
            }
            if actual_dim != expected_dim {
                return Err(TraceError::ShapeInferenceFailed {
                    primitive,
                    detail: format!(
                        "concat non-axis shape mismatch at dim {}: {} vs {}",
                        dim_idx, expected_dim, actual_dim
                    ),
                });
            }
        }
        out_dims[axis] = out_dims[axis]
            .checked_add(input.shape.dims[axis])
            .ok_or_else(|| TraceError::ShapeInferenceFailed {
                primitive,
                detail: "concat axis size overflow".to_owned(),
            })?;
        out_dtype = promote_dtype(out_dtype, input.dtype);
    }

    Ok(vec![ShapedArray {
        dtype: out_dtype,
        shape: Shape { dims: out_dims },
    }])
}

fn broadcast_shape(lhs: &Shape, rhs: &Shape) -> Option<Shape> {
    let max_rank = lhs.rank().max(rhs.rank());
    let mut dims = Vec::with_capacity(max_rank);

    for offset in 0..max_rank {
        let lhs_dim = lhs
            .dims
            .get(lhs.rank().wrapping_sub(1 + offset))
            .copied()
            .unwrap_or(1);
        let rhs_dim = rhs
            .dims
            .get(rhs.rank().wrapping_sub(1 + offset))
            .copied()
            .unwrap_or(1);

        let out_dim = if lhs_dim == rhs_dim {
            lhs_dim
        } else if lhs_dim == 1 {
            rhs_dim
        } else if rhs_dim == 1 {
            lhs_dim
        } else {
            return None;
        };
        dims.push(out_dim);
    }

    dims.reverse();
    Some(Shape { dims })
}

fn promote_dtype(lhs: DType, rhs: DType) -> DType {
    use DType::{Bool, F32, F64, I32, I64};

    match (lhs, rhs) {
        (F64, _) | (_, F64) => F64,
        (F32, _) | (_, F32) => F32,
        (I64, _) | (_, I64) => I64,
        (I32, _) | (_, I32) => I32,
        (Bool, Bool) => Bool,
    }
}

fn parse_u32_param_list(
    primitive: Primitive,
    params: &BTreeMap<String, String>,
    key: &'static str,
) -> Result<Vec<u32>, TraceError> {
    let raw = params
        .get(key)
        .ok_or(TraceError::MissingPrimitiveParam { primitive, key })?;
    parse_u32_list(primitive, key, raw)
}

fn parse_u32_list(
    primitive: Primitive,
    key: &'static str,
    raw: &str,
) -> Result<Vec<u32>, TraceError> {
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }

    raw.split(',')
        .map(str::trim)
        .map(|piece| {
            piece
                .parse::<u32>()
                .map_err(|_| TraceError::InvalidPrimitiveParam {
                    primitive,
                    key,
                    value: raw.to_owned(),
                })
        })
        .collect()
}

fn parse_usize_list(
    primitive: Primitive,
    key: &'static str,
    raw: &str,
) -> Result<Vec<usize>, TraceError> {
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }

    raw.split(',')
        .map(str::trim)
        .map(|piece| {
            piece
                .parse::<usize>()
                .map_err(|_| TraceError::InvalidPrimitiveParam {
                    primitive,
                    key,
                    value: raw.to_owned(),
                })
        })
        .collect()
}

fn parse_i64_list(
    primitive: Primitive,
    key: &'static str,
    raw: &str,
) -> Result<Vec<i64>, TraceError> {
    if raw.trim().is_empty() {
        return Ok(Vec::new());
    }

    raw.split(',')
        .map(str::trim)
        .map(|piece| {
            piece
                .parse::<i64>()
                .map_err(|_| TraceError::InvalidPrimitiveParam {
                    primitive,
                    key,
                    value: raw.to_owned(),
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        JaxprTrace, ShapedArray, SimpleTraceContext, TraceContext, TraceToJaxpr, TracerId,
    };
    use fj_core::{DType, Primitive, Shape, Value};
    use proptest::prelude::*;
    use proptest::test_runner::{Config as ProptestConfig, TestCaseError, TestRunner};
    use std::any::Any;
    use std::collections::BTreeMap;
    use std::fs;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    const PACKET_ID: &str = "FJ-P2C-001";
    const SUITE_ID: &str = "fj-trace";

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn test_log_path(test_id: &str) -> PathBuf {
        let file_name = test_id.replace("::", "__");
        repo_root()
            .join("artifacts")
            .join("testing")
            .join("logs")
            .join(SUITE_ID)
            .join(format!("{file_name}.json"))
    }

    fn replay_command(test_id: &str) -> String {
        format!("cargo test -p fj-trace --lib {test_id} -- --exact --nocapture")
    }

    fn duration_ms(start: Instant) -> u64 {
        u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    fn write_log(path: &Path, log: &fj_test_utils::TestLogV1) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| format!("log dir create failed: {err}"))?;
        }
        let payload = serde_json::to_string_pretty(log)
            .map_err(|err| format!("log serialize failed: {err}"))?;
        fs::write(path, payload).map_err(|err| format!("log write failed: {err}"))
    }

    fn panic_payload_to_string(payload: &(dyn Any + Send)) -> String {
        if let Some(msg) = payload.downcast_ref::<String>() {
            return msg.clone();
        }
        if let Some(msg) = payload.downcast_ref::<&str>() {
            return (*msg).to_owned();
        }
        "non-string panic payload".to_owned()
    }

    fn run_logged_test<F>(
        test_name: &str,
        fixture_id: String,
        mode: fj_test_utils::TestMode,
        body: F,
    ) where
        F: FnOnce() -> Result<Vec<String>, String> + std::panic::UnwindSafe,
    {
        let overall_start = Instant::now();
        let setup_start = Instant::now();
        let test_id = fj_test_utils::test_id(module_path!(), test_name);
        let mut log = fj_test_utils::TestLogV1::unit(
            test_id.clone(),
            fixture_id,
            mode,
            fj_test_utils::TestResult::Fail,
        );
        log.phase_timings.setup_ms = duration_ms(setup_start);

        let execute_start = Instant::now();
        let outcome = catch_unwind(AssertUnwindSafe(body));
        log.phase_timings.execute_ms = duration_ms(execute_start);

        let verify_start = Instant::now();
        let mut panic_payload: Option<Box<dyn Any + Send>> = None;
        let mut failure_detail: Option<String> = None;

        match outcome {
            Ok(Ok(mut artifact_refs)) => {
                log.result = fj_test_utils::TestResult::Pass;
                artifact_refs.push(format!("packet:{PACKET_ID}"));
                artifact_refs.push(format!("replay: {}", replay_command(&test_id)));
                log.artifact_refs = artifact_refs;
                log.details = Some(format!(
                    "packet_id={PACKET_ID};suite_id={SUITE_ID};result=pass"
                ));
            }
            Ok(Err(detail)) => {
                failure_detail = Some(detail.clone());
                log.result = fj_test_utils::TestResult::Fail;
                log.artifact_refs = vec![
                    format!("packet:{PACKET_ID}"),
                    format!("replay: {}", replay_command(&test_id)),
                ];
                log.details = Some(detail);
            }
            Err(payload) => {
                let detail = panic_payload_to_string(payload.as_ref());
                failure_detail = Some(detail.clone());
                log.result = fj_test_utils::TestResult::Fail;
                log.artifact_refs = vec![
                    format!("packet:{PACKET_ID}"),
                    format!("replay: {}", replay_command(&test_id)),
                ];
                log.details = Some(detail);
                panic_payload = Some(payload);
            }
        }
        log.phase_timings.verify_ms = duration_ms(verify_start);

        let log_path = test_log_path(&test_id);
        log.artifact_refs.push(log_path.display().to_string());
        log.duration_ms = duration_ms(overall_start);

        let teardown_start = Instant::now();
        write_log(&log_path, &log).expect("test log write should succeed");
        log.phase_timings.teardown_ms = duration_ms(teardown_start);
        log.duration_ms = duration_ms(overall_start);
        write_log(&log_path, &log).expect("test log rewrite should succeed");

        if let Some(payload) = panic_payload {
            std::panic::resume_unwind(payload);
        }
        if let Some(detail) = failure_detail {
            panic!("{detail}");
        }
    }

    #[test]
    fn infer_reshape_and_finalize_closed_jaxpr() {
        run_logged_test(
            "infer_reshape_and_finalize_closed_jaxpr",
            fj_test_utils::fixture_id_from_json(&("reshape-finalize", [2_u32, 3_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::F64,
                    shape: Shape { dims: vec![2, 3] },
                }]);

                let input_id = TracerId(1);
                let const_id = ctx.bind_const_value(Value::scalar_f64(10.0));

                let mut reshape_params = BTreeMap::new();
                reshape_params.insert("new_shape".to_owned(), "3,2".to_owned());
                let reshaped = ctx
                    .process_primitive(Primitive::Reshape, &[input_id], reshape_params)
                    .expect("reshape inference should succeed");

                let _ = ctx
                    .process_primitive(Primitive::Add, &[reshaped[0], const_id], BTreeMap::new())
                    .expect("add inference should succeed");

                let closed = ctx.finalize().expect("trace finalize should succeed");
                assert_eq!(closed.jaxpr.constvars.len(), 1);
                assert_eq!(closed.const_values, vec![Value::scalar_f64(10.0)]);
                assert_eq!(closed.jaxpr.equations.len(), 2);
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn nested_trace_contexts_can_open_and_close() {
        run_logged_test(
            "nested_trace_contexts_can_open_and_close",
            fj_test_utils::fixture_id_from_json(&("nested-trace", [4_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::I64,
                    shape: Shape::vector(4),
                }]);
                let nested_id = ctx.push_subtrace(vec![ShapedArray {
                    dtype: DType::I64,
                    shape: Shape::vector(4),
                }]);

                assert_eq!(ctx.active_trace_id(), nested_id);

                let nested_in = TracerId(2);
                let _ = ctx
                    .process_primitive(Primitive::ReduceSum, &[nested_in], BTreeMap::new())
                    .expect("nested reduce_sum should infer");
                ctx.pop_subtrace().expect("nested frame should close");

                assert_eq!(ctx.active_trace_id(), 1);
                let root_in = TracerId(1);
                let _ = ctx
                    .process_primitive(Primitive::ReduceSum, &[root_in], BTreeMap::new())
                    .expect("root reduce_sum should infer");
                let closed = ctx.finalize().expect("root finalize should succeed");
                assert_eq!(closed.jaxpr.equations.len(), 1);
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn infer_new_primitives_shape_rules() {
        run_logged_test(
            "infer_new_primitives_shape_rules",
            fj_test_utils::fixture_id_from_json(&("new-primitives", [5_u32, 7_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![
                    ShapedArray {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![5, 7] },
                    },
                    ShapedArray {
                        dtype: DType::I64,
                        shape: Shape { dims: vec![2] },
                    },
                    ShapedArray {
                        dtype: DType::Bool,
                        shape: Shape { dims: vec![5, 7] },
                    },
                ]);

                let x = TracerId(1);
                let idx = TracerId(2);
                let cond = TracerId(3);

                let mut slice_params = BTreeMap::new();
                slice_params.insert("start_indices".to_owned(), "1,2".to_owned());
                slice_params.insert("limit_indices".to_owned(), "4,6".to_owned());
                let sliced = ctx
                    .process_primitive(Primitive::Slice, &[x], slice_params)
                    .expect("slice inference should succeed");
                assert_eq!(
                    ctx.tracer_aval(sliced[0]).expect("aval present").shape,
                    Shape { dims: vec![3, 4] }
                );

                let mut gather_params = BTreeMap::new();
                gather_params.insert("slice_sizes".to_owned(), "2,2".to_owned());
                let gathered = ctx
                    .process_primitive(Primitive::Gather, &[x, idx], gather_params)
                    .expect("gather inference should succeed");
                assert_eq!(
                    ctx.tracer_aval(gathered[0]).expect("aval present").shape,
                    Shape {
                        dims: vec![2, 2, 2]
                    }
                );

                let mut transpose_params = BTreeMap::new();
                transpose_params.insert("permutation".to_owned(), "1,0".to_owned());
                let transposed = ctx
                    .process_primitive(Primitive::Transpose, &[x], transpose_params)
                    .expect("transpose inference should succeed");
                assert_eq!(
                    ctx.tracer_aval(transposed[0]).expect("aval present").shape,
                    Shape { dims: vec![7, 5] }
                );

                let mut broadcast_params = BTreeMap::new();
                broadcast_params.insert("shape".to_owned(), "3,5,7".to_owned());
                broadcast_params.insert("broadcast_dimensions".to_owned(), "1,2".to_owned());
                let broadcasted = ctx
                    .process_primitive(Primitive::BroadcastInDim, &[x], broadcast_params)
                    .expect("broadcast inference should succeed");
                assert_eq!(
                    ctx.tracer_aval(broadcasted[0]).expect("aval present").shape,
                    Shape {
                        dims: vec![3, 5, 7]
                    }
                );

                let mut concat_params = BTreeMap::new();
                concat_params.insert("dimension".to_owned(), "0".to_owned());
                let concatenated = ctx
                    .process_primitive(Primitive::Concatenate, &[x, x], concat_params)
                    .expect("concat inference should succeed");
                assert_eq!(
                    ctx.tracer_aval(concatenated[0])
                        .expect("aval present")
                        .shape,
                    Shape { dims: vec![10, 7] }
                );

                let selected = ctx
                    .process_primitive(Primitive::Select, &[cond, x, x], BTreeMap::new())
                    .expect("select inference should succeed");
                assert_eq!(
                    ctx.tracer_aval(selected[0]).expect("aval present").shape,
                    Shape { dims: vec![5, 7] }
                );
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn infer_select_rejects_shape_mismatch() {
        run_logged_test(
            "infer_select_rejects_shape_mismatch",
            fj_test_utils::fixture_id_from_json(&("select-shape-mismatch", [5_u32, 7_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![
                    ShapedArray {
                        dtype: DType::Bool,
                        shape: Shape { dims: vec![5, 7] },
                    },
                    ShapedArray {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![5, 7] },
                    },
                    ShapedArray {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![1, 7] },
                    },
                ]);

                let err = ctx
                    .process_primitive(
                        Primitive::Select,
                        &[TracerId(1), TracerId(2), TracerId(3)],
                        BTreeMap::new(),
                    )
                    .expect_err("select mismatch should fail");
                let detail = err.to_string();
                assert!(detail.contains("select requires matching shapes"));
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn trace_to_jaxpr_validates_in_and_out_avals() {
        run_logged_test(
            "trace_to_jaxpr_validates_in_and_out_avals",
            fj_test_utils::fixture_id_from_json(&("trace-to-jaxpr", [4_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::I64,
                    shape: Shape::vector(4),
                }]);
                let input = TracerId(1);
                let outputs = ctx
                    .process_primitive(Primitive::ReduceSum, &[input], BTreeMap::new())
                    .expect("reduce_sum inference");

                let trace = JaxprTrace {
                    trace_id: ctx.active_trace_id(),
                    in_avals: vec![ShapedArray {
                        dtype: DType::I64,
                        shape: Shape::vector(4),
                    }],
                    out_avals: vec![ctx.tracer_aval(outputs[0]).expect("aval present").clone()],
                };
                let closed = ctx
                    .trace_to_jaxpr(trace)
                    .expect("trace_to_jaxpr should pass");
                assert_eq!(closed.jaxpr.equations.len(), 1);
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn trace_to_jaxpr_rejects_mismatched_out_avals() {
        run_logged_test(
            "trace_to_jaxpr_rejects_mismatched_out_avals",
            fj_test_utils::fixture_id_from_json(&("trace-bad-out-aval", [4_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::I64,
                    shape: Shape::vector(4),
                }]);
                let input = TracerId(1);
                let _ = ctx
                    .process_primitive(Primitive::ReduceSum, &[input], BTreeMap::new())
                    .expect("reduce_sum inference");

                let trace = JaxprTrace {
                    trace_id: ctx.active_trace_id(),
                    in_avals: vec![ShapedArray {
                        dtype: DType::I64,
                        shape: Shape::vector(4),
                    }],
                    out_avals: vec![ShapedArray {
                        dtype: DType::I64,
                        shape: Shape::vector(99),
                    }],
                };
                let err = ctx
                    .trace_to_jaxpr(trace)
                    .expect_err("mismatched out aval should fail");
                assert!(matches!(err, super::TraceError::InvalidAbstractValue));
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn finalize_rejects_unclosed_subtrace() {
        run_logged_test(
            "finalize_rejects_unclosed_subtrace",
            fj_test_utils::fixture_id_from_json(&("nested-not-closed", 1_u32))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::I64,
                    shape: Shape::vector(4),
                }]);
                let _ = ctx.push_subtrace(vec![ShapedArray {
                    dtype: DType::I64,
                    shape: Shape::vector(4),
                }]);
                let err = ctx
                    .finalize()
                    .expect_err("nested subtrace should block finalize");
                assert!(matches!(err, super::TraceError::NestedTraceNotClosed));
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn prop_add_broadcast_shape_inference_consistent() {
        run_logged_test(
            "prop_add_broadcast_shape_inference_consistent",
            fj_test_utils::fixture_id_from_json(&(
                "prop-add-broadcast",
                fj_test_utils::property_test_case_count(),
            ))
            .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut runner = TestRunner::new(ProptestConfig::with_cases(
                    fj_test_utils::property_test_case_count(),
                ));
                let strategy = (1_u32..=8_u32, 1_u32..=8_u32);
                runner
                    .run(&strategy, |(lhs, rhs)| {
                        prop_assume!(lhs == rhs || lhs == 1 || rhs == 1);
                        let mut ctx = SimpleTraceContext::with_inputs(vec![
                            ShapedArray {
                                dtype: DType::I64,
                                shape: Shape::vector(lhs),
                            },
                            ShapedArray {
                                dtype: DType::I64,
                                shape: Shape::vector(rhs),
                            },
                        ]);
                        let out = ctx
                            .process_primitive(
                                Primitive::Add,
                                &[TracerId(1), TracerId(2)],
                                BTreeMap::new(),
                            )
                            .map_err(|err| TestCaseError::fail(err.to_string()))?;
                        let aval = ctx
                            .tracer_aval(out[0])
                            .map_err(|err| TestCaseError::fail(err.to_string()))?;
                        let expected = if lhs == 1 { rhs } else { lhs };
                        prop_assert_eq!(aval.shape.clone(), Shape::vector(expected));
                        Ok(())
                    })
                    .map_err(|err| err.to_string())?;
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn prop_reduce_sum_axes_shape_consistency() {
        run_logged_test(
            "prop_reduce_sum_axes_shape_consistency",
            fj_test_utils::fixture_id_from_json(&(
                "prop-reduce-sum",
                fj_test_utils::property_test_case_count(),
            ))
            .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut runner = TestRunner::new(ProptestConfig::with_cases(
                    fj_test_utils::property_test_case_count(),
                ));
                let strategy = (
                    proptest::collection::vec(1_u32..=5_u32, 1..=4),
                    0_u8..=15_u8,
                );
                runner
                    .run(&strategy, |(dims, mask)| {
                        let rank = dims.len();
                        prop_assume!(rank > 0);
                        let axes = (0..rank)
                            .filter(|axis| (mask & (1_u8 << axis)) != 0)
                            .collect::<Vec<_>>();
                        let mut params = BTreeMap::new();
                        params.insert(
                            "axes".to_owned(),
                            axes.iter()
                                .map(ToString::to_string)
                                .collect::<Vec<_>>()
                                .join(","),
                        );

                        let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                            dtype: DType::I64,
                            shape: Shape { dims: dims.clone() },
                        }]);
                        let out = ctx
                            .process_primitive(Primitive::ReduceSum, &[TracerId(1)], params)
                            .map_err(|err| TestCaseError::fail(err.to_string()))?;
                        let aval = ctx
                            .tracer_aval(out[0])
                            .map_err(|err| TestCaseError::fail(err.to_string()))?;

                        let mut expected_dims = dims.clone();
                        let mut axes_sorted = axes.clone();
                        axes_sorted.sort_unstable();
                        axes_sorted.dedup();
                        for axis in axes_sorted.into_iter().rev() {
                            expected_dims.remove(axis);
                        }
                        prop_assert_eq!(
                            aval.shape.clone(),
                            Shape {
                                dims: expected_dims
                            }
                        );
                        Ok(())
                    })
                    .map_err(|err| err.to_string())?;
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn test_trace_test_log_schema_contract() {
        run_logged_test(
            "test_trace_test_log_schema_contract",
            fj_test_utils::fixture_id_from_json(&("trace-schema", 1_u32)).expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let fixture_id = fj_test_utils::fixture_id_from_json(&("trace", "schema-contract"))
                    .expect("fixture digest");
                let log = fj_test_utils::TestLogV1::unit(
                    fj_test_utils::test_id(module_path!(), "test_trace_test_log_schema_contract"),
                    fixture_id,
                    fj_test_utils::TestMode::Strict,
                    fj_test_utils::TestResult::Pass,
                );
                assert_eq!(log.schema_version, fj_test_utils::TEST_LOG_SCHEMA_VERSION);
                Ok(Vec::new())
            },
        );
    }
}
