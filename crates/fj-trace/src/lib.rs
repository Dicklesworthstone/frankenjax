#![forbid(unsafe_code)]

//! Tracing and abstract-value machinery for FJ-P2C-001.

use fj_core::{Atom, DType, Equation, Jaxpr, Primitive, Shape, Transform, Value, VarId};
use smallvec::SmallVec;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

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
                    fj_core::Literal::U32(_) => DType::U32,
                    fj_core::Literal::U64(_) => DType::U64,
                    fj_core::Literal::Bool(_) => DType::Bool,
                    fj_core::Literal::BF16Bits(_) => DType::BF16,
                    fj_core::Literal::F16Bits(_) => DType::F16,
                    fj_core::Literal::F64Bits(_) => DType::F64,
                    fj_core::Literal::Complex64Bits(..) => DType::Complex64,
                    fj_core::Literal::Complex128Bits(..) => DType::Complex128,
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
    ForeignTracerContext {
        tracer_id: TracerId,
    },
    TracerInvariantViolation {
        tracer_id: TracerId,
    },
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
            Self::ForeignTracerContext { tracer_id } => {
                write!(
                    f,
                    "tracer id {} escaped its originating trace context",
                    tracer_id.0
                )
            }
            Self::TracerInvariantViolation { tracer_id } => {
                write!(
                    f,
                    "tracer id {} violated cached abstract-value invariants",
                    tracer_id.0
                )
            }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NestedTraceFrameSummary {
    pub transform: Transform,
    pub trace_id: u64,
    pub depth: usize,
    pub equation_count: usize,
    pub invar_count: usize,
    pub outvar_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NestedTraceSummary {
    pub max_depth: usize,
    pub frames: Vec<NestedTraceFrameSummary>,
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

    #[must_use]
    pub fn nesting_depth(&self) -> usize {
        self.frame_stack.len()
    }

    #[must_use]
    pub fn next_tracer_id_hint(&self) -> u32 {
        self.next_tracer_id
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

    pub fn pop_subtrace_closed(&mut self) -> Result<ClosedJaxpr, TraceError> {
        if self.frame_stack.len() <= 1 {
            return Err(TraceError::CompositionViolation);
        }
        let frame = self
            .frame_stack
            .pop()
            .ok_or(TraceError::CompositionViolation)?;
        self.build_closed_jaxpr(frame)
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
            Primitive::Complex => {
                if inputs.len() != 2 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected 2 inputs, got {}", inputs.len()),
                    });
                }
                if matches!(
                    inputs[0].dtype,
                    DType::Bool | DType::Complex64 | DType::Complex128
                ) || matches!(
                    inputs[1].dtype,
                    DType::Bool | DType::Complex64 | DType::Complex128
                ) {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: "complex expects real-valued numeric inputs".to_owned(),
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
                let dtype = match (inputs[0].dtype, inputs[1].dtype) {
                    (DType::F32, DType::F32) => DType::Complex64,
                    _ => DType::Complex128,
                };
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
            | Primitive::Erfc
            | Primitive::Lgamma
            | Primitive::Digamma
            | Primitive::ErfInv
            | Primitive::Cbrt
            | Primitive::IntegerPow
            | Primitive::Copy
            | Primitive::ReducePrecision => {
                if inputs.len() != 1 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected 1 input, got {}", inputs.len()),
                    });
                }
                Ok(vec![inputs[0].clone()])
            }
            Primitive::BitcastConvertType => {
                if inputs.len() != 1 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected 1 input, got {}", inputs.len()),
                    });
                }
                let raw_dtype =
                    params
                        .get("new_dtype")
                        .ok_or(TraceError::MissingPrimitiveParam {
                            primitive,
                            key: "new_dtype",
                        })?;
                let dtype = parse_dtype_name(primitive, "new_dtype", raw_dtype)?;
                Ok(vec![ShapedArray {
                    dtype,
                    shape: inputs[0].shape.clone(),
                }])
            }
            Primitive::Conj => {
                if inputs.len() != 1 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected 1 input, got {}", inputs.len()),
                    });
                }
                if !matches!(inputs[0].dtype, DType::Complex64 | DType::Complex128) {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: "conj expects complex-valued input".to_owned(),
                    });
                }
                Ok(vec![inputs[0].clone()])
            }
            Primitive::Real | Primitive::Imag => {
                if inputs.len() != 1 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected 1 input, got {}", inputs.len()),
                    });
                }
                let dtype = match inputs[0].dtype {
                    DType::Complex64 => DType::F32,
                    DType::Complex128 => DType::F64,
                    _ => {
                        return Err(TraceError::ShapeInferenceFailed {
                            primitive,
                            detail: "real/imag expect complex-valued input".to_owned(),
                        });
                    }
                };
                Ok(vec![ShapedArray {
                    dtype,
                    shape: inputs[0].shape.clone(),
                }])
            }
            // IsFinite: same shape, output dtype is Bool
            Primitive::IsFinite => {
                if inputs.len() != 1 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected 1 input, got {}", inputs.len()),
                    });
                }
                Ok(vec![ShapedArray {
                    dtype: DType::Bool,
                    shape: inputs[0].shape.clone(),
                }])
            }
            // Nextafter: binary, same shape as inputs
            Primitive::Nextafter => {
                if inputs.len() != 2 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected 2 inputs, got {}", inputs.len()),
                    });
                }
                Ok(vec![inputs[0].clone()])
            }
            Primitive::Cholesky => infer_cholesky(inputs),
            Primitive::Qr => infer_qr(inputs, params),
            Primitive::Svd => infer_svd(inputs, params),
            Primitive::TriangularSolve => infer_triangular_solve(inputs),
            Primitive::Eigh => infer_eigh(inputs),
            Primitive::Fft => infer_fft(inputs),
            Primitive::Ifft => infer_ifft(inputs),
            Primitive::Rfft => infer_rfft(inputs, params),
            Primitive::Irfft => infer_irfft(inputs, params),
            // Reductions: all use the same reduce shape inference
            Primitive::ReduceSum
            | Primitive::ReduceMax
            | Primitive::ReduceMin
            | Primitive::ReduceProd
            | Primitive::ReduceAnd
            | Primitive::ReduceOr
            | Primitive::ReduceXor => infer_reduce_sum(primitive, inputs, params),
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
            Primitive::Pad => infer_pad(inputs, params),
            Primitive::Rev => {
                // Rev preserves shape
                if inputs.len() != 1 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected 1 input, got {}", inputs.len()),
                    });
                }
                Ok(vec![inputs[0].clone()])
            }
            Primitive::Squeeze => {
                // Squeeze removes specified singleton dimensions
                if inputs.len() != 1 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected 1 input, got {}", inputs.len()),
                    });
                }
                let dims = &inputs[0].shape.dims;
                let squeeze_dims: Vec<usize> = if let Some(raw) = params.get("dimensions") {
                    raw.split(',')
                        .map(|s| s.trim().parse::<usize>().unwrap_or(0))
                        .collect()
                } else {
                    dims.iter()
                        .enumerate()
                        .filter(|&(_, &d)| d == 1)
                        .map(|(i, _)| i)
                        .collect()
                };
                let new_dims: Vec<u32> = dims
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !squeeze_dims.contains(i))
                    .map(|(_, &d)| d)
                    .collect();
                Ok(vec![ShapedArray {
                    dtype: inputs[0].dtype,
                    shape: Shape { dims: new_dims },
                }])
            }
            Primitive::Split => {
                // Split along axis: output has extra leading dim
                if inputs.len() != 1 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected 1 input, got {}", inputs.len()),
                    });
                }
                let axis: usize = params
                    .get("axis")
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);
                let dims = &inputs[0].shape.dims;
                let num_sections: usize = if let Some(raw) = params.get("num_sections") {
                    raw.trim().parse().unwrap_or(1)
                } else if let Some(raw) = params.get("sizes") {
                    raw.split(',').count()
                } else {
                    1
                };
                let section_size = dims[axis] as usize / num_sections;
                let mut new_dims = Vec::with_capacity(dims.len() + 1);
                for (i, &d) in dims.iter().enumerate() {
                    if i == axis {
                        new_dims.push(num_sections as u32);
                        new_dims.push(section_size as u32);
                    } else {
                        new_dims.push(d);
                    }
                }
                Ok(vec![ShapedArray {
                    dtype: inputs[0].dtype,
                    shape: Shape { dims: new_dims },
                }])
            }
            Primitive::ExpandDims => {
                // ExpandDims inserts a size-1 dim at the given axis
                if inputs.len() != 1 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected 1 input, got {}", inputs.len()),
                    });
                }
                let axis: usize = params
                    .get("axis")
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);
                let mut new_dims = inputs[0].shape.dims.clone();
                new_dims.insert(axis, 1);
                Ok(vec![ShapedArray {
                    dtype: inputs[0].dtype,
                    shape: Shape { dims: new_dims },
                }])
            }
            Primitive::DynamicSlice => {
                // Output shape = slice_sizes param
                if inputs.is_empty() {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: "expected at least 1 input".to_owned(),
                    });
                }
                let slice_sizes = if let Some(raw) = params.get("slice_sizes") {
                    parse_u32_list(primitive, "slice_sizes", raw)?
                } else {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: "missing required param 'slice_sizes'".to_owned(),
                    });
                };
                Ok(vec![ShapedArray {
                    dtype: inputs[0].dtype,
                    shape: Shape { dims: slice_sizes },
                }])
            }
            Primitive::DynamicUpdateSlice => {
                // dynamic_update_slice(operand, update, ...starts): output = operand shape
                if inputs.len() < 2 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected at least 2 inputs, got {}", inputs.len()),
                    });
                }
                Ok(vec![ShapedArray {
                    dtype: inputs[0].dtype,
                    shape: inputs[0].shape.clone(),
                }])
            }
            Primitive::Clamp => {
                // clamp(x, lo, hi): 3 inputs, output shape matches first
                if inputs.len() != 3 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected 3 inputs, got {}", inputs.len()),
                    });
                }
                Ok(vec![inputs[0].clone()])
            }
            Primitive::Iota => {
                // Iota: no inputs, output shape from params
                let length = params
                    .get("length")
                    .and_then(|s| s.trim().parse::<u32>().ok())
                    .unwrap_or(0);
                let dtype_str = params.get("dtype").map(String::as_str).unwrap_or("I64");
                let dtype = match dtype_str {
                    "F64" | "f64" => DType::F64,
                    "U32" | "u32" => DType::U32,
                    "U64" | "u64" => DType::U64,
                    _ => DType::I64,
                };
                Ok(vec![ShapedArray {
                    dtype,
                    shape: Shape::vector(length),
                }])
            }
            Primitive::BroadcastedIota => {
                if !inputs.is_empty() {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected 0 inputs, got {}", inputs.len()),
                    });
                }
                let raw_shape = params
                    .get("shape")
                    .ok_or(TraceError::MissingPrimitiveParam {
                        primitive,
                        key: "shape",
                    })?;
                let dims = parse_u32_list(primitive, "shape", raw_shape)?;
                let rank = dims.len();
                let dimension = params
                    .get("dimension")
                    .and_then(|raw| raw.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                if rank > 0 && dimension >= rank {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("dimension {dimension} out of bounds for rank {rank}"),
                    });
                }
                let dtype = if let Some(raw_dtype) = params.get("dtype") {
                    parse_dtype_name(primitive, "dtype", raw_dtype)?
                } else {
                    DType::I64
                };
                Ok(vec![ShapedArray {
                    dtype,
                    shape: Shape { dims },
                }])
            }
            Primitive::OneHot => {
                // one_hot(indices): output appends num_classes dimension
                if inputs.len() != 1 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected 1 input, got {}", inputs.len()),
                    });
                }
                let num_classes = params
                    .get("num_classes")
                    .and_then(|s| s.trim().parse::<u32>().ok())
                    .ok_or_else(|| TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: "missing or invalid 'num_classes' param".to_owned(),
                    })?;
                let dtype_str = params.get("dtype").map(String::as_str).unwrap_or("F64");
                let dtype = match dtype_str {
                    "I64" | "i64" => DType::I64,
                    "I32" | "i32" => DType::I32,
                    "U32" | "u32" => DType::U32,
                    "U64" | "u64" => DType::U64,
                    _ => DType::F64,
                };
                let mut out_dims = inputs[0].shape.dims.clone();
                out_dims.push(num_classes);
                Ok(vec![ShapedArray {
                    dtype,
                    shape: Shape { dims: out_dims },
                }])
            }
            Primitive::Cumsum | Primitive::Cumprod | Primitive::Sort => {
                // Cumulative/sort ops: output shape = input shape
                if inputs.len() != 1 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected 1 input, got {}", inputs.len()),
                    });
                }
                Ok(vec![inputs[0].clone()])
            }
            Primitive::Argsort => {
                // Argsort: output shape = input shape, dtype = I64
                if inputs.len() != 1 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected 1 input, got {}", inputs.len()),
                    });
                }
                Ok(vec![ShapedArray {
                    dtype: DType::I64,
                    shape: inputs[0].shape.clone(),
                }])
            }
            Primitive::Conv => {
                // Conv: output shape depends on input, kernel, strides, padding.
                // For now: 1D conv with [batch, spatial, channels] layout.
                if inputs.len() != 2 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("expected 2 inputs (lhs, rhs), got {}", inputs.len()),
                    });
                }
                let lhs = &inputs[0];
                let rhs = &inputs[1];
                // Simple 1D: lhs=[N, W, C_in], rhs=[K, C_in, C_out]
                // output=[N, W_out, C_out] where W_out depends on stride/padding
                let lhs_rank = lhs.shape.rank();
                if lhs_rank < 2 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: "lhs must have rank >= 2".to_owned(),
                    });
                }
                let strides: Vec<usize> = params
                    .get("strides")
                    .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
                    .unwrap_or_else(|| vec![1; lhs_rank - 2]);
                let padding_mode = params.get("padding").map(String::as_str).unwrap_or("valid");

                let mut out_dims = Vec::new();
                out_dims.push(lhs.shape.dims[0]); // batch
                for i in 0..(lhs_rank - 2) {
                    let spatial = lhs.shape.dims[i + 1] as usize;
                    let kernel = rhs.shape.dims[i] as usize;
                    let stride = *strides.get(i).unwrap_or(&1);
                    let out_spatial = match padding_mode {
                        "same" | "SAME" => spatial.div_ceil(stride),
                        _ => {
                            // "valid" / default
                            if spatial >= kernel {
                                (spatial - kernel) / stride + 1
                            } else {
                                0
                            }
                        }
                    };
                    out_dims.push(out_spatial as u32);
                }
                // Last dim of rhs is output channels
                out_dims.push(*rhs.shape.dims.last().unwrap_or(&1));
                let dtype = promote_dtype(lhs.dtype, rhs.dtype);
                Ok(vec![ShapedArray {
                    dtype,
                    shape: Shape { dims: out_dims },
                }])
            }

            Primitive::Cond => {
                // Cond: inputs are [pred, true_val, false_val]
                // Output shape = shape of true_val (both branches must match)
                if inputs.len() != 3 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!(
                            "cond expects 3 inputs (pred, true, false), got {}",
                            inputs.len()
                        ),
                    });
                }
                let true_branch = &inputs[1];
                let false_branch = &inputs[2];
                if true_branch.shape != false_branch.shape {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!(
                            "cond branches must have same shape: {:?} vs {:?}",
                            true_branch.shape.dims, false_branch.shape.dims
                        ),
                    });
                }
                let dtype = promote_dtype(true_branch.dtype, false_branch.dtype);
                Ok(vec![ShapedArray {
                    dtype,
                    shape: true_branch.shape.clone(),
                }])
            }

            Primitive::Scan => {
                // Scan: inputs are [init_carry, xs_tensor]
                // Output shape = shape of init_carry (carry is threaded through)
                if inputs.len() < 2 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!(
                            "scan expects at least 2 inputs (init_carry, xs), got {}",
                            inputs.len()
                        ),
                    });
                }
                let init_carry = &inputs[0];
                Ok(vec![ShapedArray {
                    dtype: init_carry.dtype,
                    shape: init_carry.shape.clone(),
                }])
            }

            Primitive::While => {
                // While: inputs are [init_carry, step_value, threshold]
                // Output shape = shape of init_carry (carry is threaded through)
                if inputs.is_empty() {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: "while expects at least 1 input (init_carry)".to_owned(),
                    });
                }
                let init_carry = &inputs[0];
                Ok(vec![ShapedArray {
                    dtype: init_carry.dtype,
                    shape: init_carry.shape.clone(),
                }])
            }

            Primitive::Switch => {
                // Switch: inputs are [index, branch0_val, branch1_val, ...]
                // Output shape = shape of first branch value (all branches must match)
                if inputs.len() < 2 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!(
                            "switch expects at least 2 inputs (index + branch), got {}",
                            inputs.len()
                        ),
                    });
                }
                let branch = &inputs[1];
                Ok(vec![ShapedArray {
                    dtype: branch.dtype,
                    shape: branch.shape.clone(),
                }])
            }

            // Bitwise binary: same shape as inputs, integer type preserved
            Primitive::BitwiseAnd
            | Primitive::BitwiseOr
            | Primitive::BitwiseXor
            | Primitive::ShiftLeft
            | Primitive::ShiftRightArithmetic
            | Primitive::ShiftRightLogical => {
                if inputs.len() != 2 {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("bitwise binary op expects 2 inputs, got {}", inputs.len()),
                    });
                }
                Ok(vec![ShapedArray {
                    dtype: inputs[0].dtype,
                    shape: inputs[0].shape.clone(),
                }])
            }

            // Bitwise/integer unary: same shape and type as input
            Primitive::BitwiseNot | Primitive::PopulationCount | Primitive::CountLeadingZeros => {
                if inputs.is_empty() {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: format!("{} expects 1 input", primitive.as_str()),
                    });
                }
                Ok(vec![ShapedArray {
                    dtype: inputs[0].dtype,
                    shape: inputs[0].shape.clone(),
                }])
            }

            Primitive::ReduceWindow => {
                if inputs.is_empty() {
                    return Err(TraceError::ShapeInferenceFailed {
                        primitive,
                        detail: "reduce_window expects at least 1 input".to_owned(),
                    });
                }
                let input = &inputs[0];
                // Calculate output shape from window_dimensions and strides
                let window_dims: Vec<usize> = params
                    .get("window_dimensions")
                    .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
                    .unwrap_or_else(|| vec![2; input.shape.rank()]);
                let strides: Vec<usize> = params
                    .get("window_strides")
                    .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
                    .unwrap_or_else(|| vec![1; input.shape.rank()]);
                let padding = params.get("padding").map(|s| s.as_str()).unwrap_or("valid");

                let mut out_dims = Vec::with_capacity(input.shape.rank());
                for d in 0..input.shape.rank() {
                    let input_dim = input.shape.dims[d] as usize;
                    let win = *window_dims.get(d).unwrap_or(&2);
                    let stride = *strides.get(d).unwrap_or(&1);
                    let out_dim = match padding {
                        "same" => input_dim.div_ceil(stride),
                        _ => {
                            if input_dim >= win {
                                (input_dim - win) / stride + 1
                            } else {
                                0
                            }
                        }
                    };
                    out_dims.push(out_dim as u32);
                }

                Ok(vec![ShapedArray {
                    dtype: promote_dtype(input.dtype, input.dtype),
                    shape: Shape { dims: out_dims },
                }])
            }
        }
    }

    fn build_closed_jaxpr(&mut self, frame: TraceFrame) -> Result<ClosedJaxpr, TraceError> {
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
                effects: vec![],
                sub_jaxprs: vec![],
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

fn expect_single_matrix_input(
    primitive: Primitive,
    inputs: &[ShapedArray],
) -> Result<&ShapedArray, TraceError> {
    if inputs.len() != 1 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!("expected 1 input, got {}", inputs.len()),
        });
    }
    let input = &inputs[0];
    if input.shape.rank() < 2 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!(
                "expected rank >= 2 matrix input, got rank {}",
                input.shape.rank()
            ),
        });
    }
    Ok(input)
}

fn expect_square_trailing_dims(
    primitive: Primitive,
    input: &ShapedArray,
) -> Result<u32, TraceError> {
    let rank = input.shape.rank();
    let rows = input.shape.dims[rank - 2];
    let cols = input.shape.dims[rank - 1];
    if rows != cols {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!("expected square trailing dims, got ({rows}, {cols})"),
        });
    }
    Ok(rows)
}

fn parse_bool_param(
    primitive: Primitive,
    params: &BTreeMap<String, String>,
    key: &'static str,
    default: bool,
) -> Result<bool, TraceError> {
    let Some(raw) = params.get(key) else {
        return Ok(default);
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" => Ok(true),
        "0" | "false" | "no" => Ok(false),
        _ => Err(TraceError::InvalidPrimitiveParam {
            primitive,
            key,
            value: raw.clone(),
        }),
    }
}

fn parse_optional_fft_length(
    primitive: Primitive,
    params: &BTreeMap<String, String>,
    default: u32,
) -> Result<u32, TraceError> {
    if let Some(raw) = params.get("fft_length") {
        let len = raw
            .trim()
            .parse::<u32>()
            .map_err(|_| TraceError::InvalidPrimitiveParam {
                primitive,
                key: "fft_length",
                value: raw.clone(),
            })?;
        return Ok(len);
    }
    if let Some(raw) = params.get("fft_lengths") {
        let lengths = parse_u32_list(primitive, "fft_lengths", raw)?;
        let len = *lengths.last().ok_or(TraceError::InvalidPrimitiveParam {
            primitive,
            key: "fft_lengths",
            value: raw.clone(),
        })?;
        return Ok(len);
    }
    Ok(default)
}

fn to_complex_dtype(dtype: DType) -> DType {
    match dtype {
        DType::Complex64 | DType::Complex128 => dtype,
        DType::BF16 | DType::F16 | DType::F32 => DType::Complex64,
        _ => DType::Complex128,
    }
}

fn to_real_dtype(dtype: DType) -> DType {
    match dtype {
        DType::Complex64 => DType::F32,
        DType::Complex128 => DType::F64,
        other => other,
    }
}

fn infer_cholesky(inputs: &[ShapedArray]) -> Result<Vec<ShapedArray>, TraceError> {
    let primitive = Primitive::Cholesky;
    let input = expect_single_matrix_input(primitive, inputs)?;
    let _ = expect_square_trailing_dims(primitive, input)?;
    Ok(vec![input.clone()])
}

fn infer_qr(
    inputs: &[ShapedArray],
    params: &BTreeMap<String, String>,
) -> Result<Vec<ShapedArray>, TraceError> {
    let primitive = Primitive::Qr;
    let input = expect_single_matrix_input(primitive, inputs)?;
    let full_matrices = parse_bool_param(primitive, params, "full_matrices", false)?;

    let rank = input.shape.rank();
    let m = input.shape.dims[rank - 2];
    let n = input.shape.dims[rank - 1];
    let k = m.min(n);
    let batch_dims = &input.shape.dims[..rank - 2];

    let mut q_dims = batch_dims.to_vec();
    let mut r_dims = batch_dims.to_vec();
    if full_matrices {
        q_dims.extend([m, m]);
        r_dims.extend([m, n]);
    } else {
        q_dims.extend([m, k]);
        r_dims.extend([k, n]);
    }

    Ok(vec![
        ShapedArray {
            dtype: input.dtype,
            shape: Shape { dims: q_dims },
        },
        ShapedArray {
            dtype: input.dtype,
            shape: Shape { dims: r_dims },
        },
    ])
}

fn infer_svd(
    inputs: &[ShapedArray],
    params: &BTreeMap<String, String>,
) -> Result<Vec<ShapedArray>, TraceError> {
    let primitive = Primitive::Svd;
    let input = expect_single_matrix_input(primitive, inputs)?;
    let full_matrices = parse_bool_param(primitive, params, "full_matrices", false)?;

    let rank = input.shape.rank();
    let m = input.shape.dims[rank - 2];
    let n = input.shape.dims[rank - 1];
    let k = m.min(n);
    let batch_dims = &input.shape.dims[..rank - 2];

    let mut u_dims = batch_dims.to_vec();
    let mut s_dims = batch_dims.to_vec();
    let mut vt_dims = batch_dims.to_vec();

    if full_matrices {
        u_dims.extend([m, m]);
        vt_dims.extend([n, n]);
    } else {
        u_dims.extend([m, k]);
        vt_dims.extend([k, n]);
    }
    s_dims.push(k);

    Ok(vec![
        ShapedArray {
            dtype: input.dtype,
            shape: Shape { dims: u_dims },
        },
        ShapedArray {
            dtype: to_real_dtype(input.dtype),
            shape: Shape { dims: s_dims },
        },
        ShapedArray {
            dtype: input.dtype,
            shape: Shape { dims: vt_dims },
        },
    ])
}

fn infer_triangular_solve(inputs: &[ShapedArray]) -> Result<Vec<ShapedArray>, TraceError> {
    let primitive = Primitive::TriangularSolve;
    if inputs.len() != 2 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!("expected 2 inputs, got {}", inputs.len()),
        });
    }

    let lhs = &inputs[0];
    let rhs = &inputs[1];
    if lhs.shape.rank() < 2 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!("lhs must be rank >= 2, got {}", lhs.shape.rank()),
        });
    }
    let lhs_batch_rank = lhs.shape.rank() - 2;
    if rhs.shape.rank() < lhs_batch_rank + 1 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!(
                "rhs rank {} is incompatible with lhs batch rank {}",
                rhs.shape.rank(),
                lhs_batch_rank
            ),
        });
    }

    let lhs_batch_dims = &lhs.shape.dims[..lhs_batch_rank];
    let rhs_batch_dims = &rhs.shape.dims[..lhs_batch_rank];
    if lhs_batch_dims != rhs_batch_dims {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!(
                "lhs/rhs batch dims mismatch: lhs={:?}, rhs={:?}",
                lhs_batch_dims, rhs_batch_dims
            ),
        });
    }

    let n = expect_square_trailing_dims(primitive, lhs)?;
    let rhs_contract_dim = rhs.shape.dims[lhs_batch_rank];
    if rhs_contract_dim != n {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!(
                "rhs leading matrix dim {} must match lhs size {}",
                rhs_contract_dim, n
            ),
        });
    }

    let trailing = rhs.shape.rank() - lhs_batch_rank;
    if trailing != 1 && trailing != 2 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!(
                "rhs must be batched vector or matrix (trailing dims 1 or 2), got {}",
                trailing
            ),
        });
    }

    Ok(vec![ShapedArray {
        dtype: promote_dtype(lhs.dtype, rhs.dtype),
        shape: rhs.shape.clone(),
    }])
}

fn infer_eigh(inputs: &[ShapedArray]) -> Result<Vec<ShapedArray>, TraceError> {
    let primitive = Primitive::Eigh;
    let input = expect_single_matrix_input(primitive, inputs)?;
    let n = expect_square_trailing_dims(primitive, input)?;

    let rank = input.shape.rank();
    let batch_dims = &input.shape.dims[..rank - 2];

    let mut eigenvalues_dims = batch_dims.to_vec();
    eigenvalues_dims.push(n);

    Ok(vec![
        ShapedArray {
            dtype: to_real_dtype(input.dtype),
            shape: Shape {
                dims: eigenvalues_dims,
            },
        },
        input.clone(),
    ])
}

fn infer_fft(inputs: &[ShapedArray]) -> Result<Vec<ShapedArray>, TraceError> {
    let primitive = Primitive::Fft;
    if inputs.len() != 1 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!("expected 1 input, got {}", inputs.len()),
        });
    }
    let input = &inputs[0];
    Ok(vec![ShapedArray {
        dtype: to_complex_dtype(input.dtype),
        shape: input.shape.clone(),
    }])
}

fn infer_ifft(inputs: &[ShapedArray]) -> Result<Vec<ShapedArray>, TraceError> {
    let primitive = Primitive::Ifft;
    if inputs.len() != 1 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!("expected 1 input, got {}", inputs.len()),
        });
    }
    let input = &inputs[0];
    if !matches!(input.dtype, DType::Complex64 | DType::Complex128) {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: "ifft expects complex-valued input".to_owned(),
        });
    }
    Ok(vec![ShapedArray {
        dtype: to_complex_dtype(input.dtype),
        shape: input.shape.clone(),
    }])
}

fn infer_rfft(
    inputs: &[ShapedArray],
    params: &BTreeMap<String, String>,
) -> Result<Vec<ShapedArray>, TraceError> {
    let primitive = Primitive::Rfft;
    if inputs.len() != 1 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!("expected 1 input, got {}", inputs.len()),
        });
    }

    let input = &inputs[0];
    if input.shape.rank() == 0 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: "rfft expects rank >= 1 input".to_owned(),
        });
    }
    if matches!(input.dtype, DType::Complex64 | DType::Complex128) {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: "rfft expects real-valued input".to_owned(),
        });
    }

    let mut out_dims = input.shape.dims.clone();
    let last_axis = out_dims.len() - 1;
    let input_last = out_dims[last_axis];
    let fft_len = parse_optional_fft_length(primitive, params, input_last)?;
    if fft_len == 0 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: "rfft fft_length must be > 0".to_owned(),
        });
    }
    out_dims[last_axis] = fft_len / 2 + 1;

    Ok(vec![ShapedArray {
        dtype: to_complex_dtype(input.dtype),
        shape: Shape { dims: out_dims },
    }])
}

fn infer_irfft(
    inputs: &[ShapedArray],
    params: &BTreeMap<String, String>,
) -> Result<Vec<ShapedArray>, TraceError> {
    let primitive = Primitive::Irfft;
    if inputs.len() != 1 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!("expected 1 input, got {}", inputs.len()),
        });
    }

    let input = &inputs[0];
    if input.shape.rank() == 0 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: "irfft expects rank >= 1 input".to_owned(),
        });
    }
    if !matches!(input.dtype, DType::Complex64 | DType::Complex128) {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: "irfft expects complex-valued input".to_owned(),
        });
    }

    let mut out_dims = input.shape.dims.clone();
    let last_axis = out_dims.len() - 1;
    let inferred_len = out_dims[last_axis].saturating_sub(1).saturating_mul(2);
    let fft_len = parse_optional_fft_length(primitive, params, inferred_len)?;
    if fft_len == 0 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: "irfft fft_length must be > 0".to_owned(),
        });
    }
    out_dims[last_axis] = fft_len;

    Ok(vec![ShapedArray {
        dtype: to_real_dtype(input.dtype),
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
    if inputs.len() != 3 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!(
                "expected 3 inputs (operand, indices, updates), got {}",
                inputs.len()
            ),
        });
    }

    let operand = &inputs[0];
    if operand.shape.rank() == 0 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: "cannot scatter into a rank-0 tensor".to_owned(),
        });
    }

    // Output shape always matches operand shape
    Ok(vec![operand.clone()])
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

fn infer_pad(
    inputs: &[ShapedArray],
    params: &BTreeMap<String, String>,
) -> Result<Vec<ShapedArray>, TraceError> {
    let primitive = Primitive::Pad;
    if inputs.len() != 2 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!("expected 2 inputs, got {}", inputs.len()),
        });
    }

    let operand = &inputs[0];
    if inputs[1].shape.rank() != 0 {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: "pad value must be scalar-shaped".to_owned(),
        });
    }

    let rank = operand.shape.rank();
    let lows = parse_u32_param_list(primitive, params, "padding_low")?;
    let highs = parse_u32_param_list(primitive, params, "padding_high")?;
    let interiors = if let Some(raw) = params.get("padding_interior") {
        parse_u32_list(primitive, "padding_interior", raw)?
    } else {
        vec![0_u32; rank]
    };

    if lows.len() != rank || highs.len() != rank || interiors.len() != rank {
        return Err(TraceError::ShapeInferenceFailed {
            primitive,
            detail: format!(
                "padding rank mismatch: rank={} low={} high={} interior={}",
                rank,
                lows.len(),
                highs.len(),
                interiors.len()
            ),
        });
    }

    let mut out_dims = Vec::with_capacity(rank);
    for axis in 0..rank {
        let dim = operand.shape.dims[axis];
        let interior_span = dim.saturating_sub(1).checked_mul(interiors[axis]).ok_or(
            TraceError::ShapeInferenceFailed {
                primitive,
                detail: format!("padding interior overflow on axis {axis}"),
            },
        )?;
        let out_dim = lows[axis]
            .checked_add(dim)
            .and_then(|v| v.checked_add(interior_span))
            .and_then(|v| v.checked_add(highs[axis]))
            .ok_or(TraceError::ShapeInferenceFailed {
                primitive,
                detail: format!("padded dimension overflow on axis {axis}"),
            })?;
        out_dims.push(out_dim);
    }

    Ok(vec![ShapedArray {
        dtype: promote_dtype(operand.dtype, inputs[1].dtype),
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
    use DType::{BF16, Bool, Complex64, Complex128, F16, F32, F64, I32, I64, U32, U64};

    // JAX type promotion lattice — must match fj-lax/src/type_promotion.rs
    match (lhs, rhs) {
        (Complex128, _) | (_, Complex128) => Complex128,
        (Complex64, _) | (_, Complex64) => Complex64,
        (F64, _) | (_, F64) => F64,
        (F32, _) | (_, F32) => F32,
        (BF16, F16) | (F16, BF16) => F32,
        (BF16, BF16) => BF16,
        (F16, F16) => F16,
        (BF16, Bool | I32 | I64 | U32 | U64) | (Bool | I32 | I64 | U32 | U64, BF16) => BF16,
        (F16, Bool | I32 | I64 | U32 | U64) | (Bool | I32 | I64 | U32 | U64, F16) => F16,
        (U64, I64) | (I64, U64) => F64,
        (I32, U32) | (U32, I32) => I64,
        (I64, U32) | (U32, I64) => I64,
        (I64, _) | (_, I64) => I64,
        (I32, _) | (_, I32) => I32,
        (U64, _) | (_, U64) => U64,
        (U32, _) | (_, U32) => U32,
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

fn parse_dtype_name(
    primitive: Primitive,
    key: &'static str,
    raw: &str,
) -> Result<DType, TraceError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "bf16" | "bfloat16" => Ok(DType::BF16),
        "f16" | "float16" => Ok(DType::F16),
        "f32" | "float32" => Ok(DType::F32),
        "f64" | "float64" => Ok(DType::F64),
        "i32" => Ok(DType::I32),
        "i64" => Ok(DType::I64),
        "u32" => Ok(DType::U32),
        "u64" => Ok(DType::U64),
        "bool" => Ok(DType::Bool),
        "complex64" => Ok(DType::Complex64),
        "complex128" => Ok(DType::Complex128),
        _ => Err(TraceError::InvalidPrimitiveParam {
            primitive,
            key,
            value: raw.to_owned(),
        }),
    }
}

// ── make_jaxpr: Trace Rust closures into Jaxpr ────────────────────

/// A tracer reference that records primitive operations into a shared trace context.
///
/// Supports operator overloading (Add, Sub, Mul, Neg) so users can write
/// natural mathematical expressions that get traced into a Jaxpr.
#[derive(Clone)]
pub struct TracerRef {
    id: TracerId,
    aval: ShapedArray,
    ctx: Rc<RefCell<SimpleTraceContext>>,
}

impl std::fmt::Debug for TracerRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TracerRef")
            .field("id", &self.id)
            .field("aval", &self.aval)
            .finish()
    }
}

impl TracerRef {
    fn validate_against_ctx(
        &self,
        expected_ctx: &Rc<RefCell<SimpleTraceContext>>,
    ) -> Result<(), TraceError> {
        if !Rc::ptr_eq(&self.ctx, expected_ctx) {
            return Err(TraceError::ForeignTracerContext { tracer_id: self.id });
        }
        let actual = self.ctx.borrow().tracer_aval(self.id)?.clone();
        if actual != self.aval {
            return Err(TraceError::TracerInvariantViolation { tracer_id: self.id });
        }
        Ok(())
    }

    /// Get the tracer ID.
    #[must_use]
    pub fn id(&self) -> TracerId {
        self.id
    }

    /// Get the abstract value (dtype + shape).
    #[must_use]
    pub fn aval(&self) -> &ShapedArray {
        &self.aval
    }

    /// Apply a unary primitive operation (e.g., sin, cos, neg, exp).
    pub fn unary_op(&self, primitive: Primitive) -> Result<TracerRef, TraceError> {
        self.validate_against_ctx(&self.ctx)?;
        let output_ids =
            self.ctx
                .borrow_mut()
                .process_primitive(primitive, &[self.id], BTreeMap::new())?;
        let ctx = self.ctx.borrow();
        let aval = ctx.tracer_aval(output_ids[0])?.clone();
        drop(ctx);
        Ok(TracerRef {
            id: output_ids[0],
            aval,
            ctx: Rc::clone(&self.ctx),
        })
    }

    /// Apply a binary primitive operation (e.g., add, sub, mul).
    pub fn binary_op(
        &self,
        primitive: Primitive,
        other: &TracerRef,
    ) -> Result<TracerRef, TraceError> {
        self.validate_against_ctx(&self.ctx)?;
        other.validate_against_ctx(&self.ctx)?;
        let output_ids = self.ctx.borrow_mut().process_primitive(
            primitive,
            &[self.id, other.id],
            BTreeMap::new(),
        )?;
        let ctx = self.ctx.borrow();
        let aval = ctx.tracer_aval(output_ids[0])?.clone();
        drop(ctx);
        Ok(TracerRef {
            id: output_ids[0],
            aval,
            ctx: Rc::clone(&self.ctx),
        })
    }

    /// Apply a primitive with custom parameters.
    pub fn primitive_with_params(
        &self,
        primitive: Primitive,
        other_inputs: &[&TracerRef],
        params: BTreeMap<String, String>,
    ) -> Result<Vec<TracerRef>, TraceError> {
        self.validate_against_ctx(&self.ctx)?;
        let mut input_ids = vec![self.id];
        for other in other_inputs {
            other.validate_against_ctx(&self.ctx)?;
            input_ids.push(other.id);
        }
        let output_ids = self
            .ctx
            .borrow_mut()
            .process_primitive(primitive, &input_ids, params)?;
        let ctx = self.ctx.borrow();
        let refs = output_ids
            .into_iter()
            .map(|oid| {
                let aval = ctx.tracer_aval(oid).unwrap().clone();
                TracerRef {
                    id: oid,
                    aval,
                    ctx: Rc::clone(&self.ctx),
                }
            })
            .collect();
        Ok(refs)
    }
}

// ── Operator overloading for TracerRef ────────────────────────────

impl std::ops::Add for &TracerRef {
    type Output = TracerRef;
    fn add(self, rhs: Self) -> TracerRef {
        self.binary_op(Primitive::Add, rhs)
            .expect("add tracing failed")
    }
}

impl std::ops::Add for TracerRef {
    type Output = TracerRef;
    fn add(self, rhs: Self) -> TracerRef {
        self.binary_op(Primitive::Add, &rhs)
            .expect("add tracing failed")
    }
}

impl std::ops::Sub for &TracerRef {
    type Output = TracerRef;
    fn sub(self, rhs: Self) -> TracerRef {
        self.binary_op(Primitive::Sub, rhs)
            .expect("sub tracing failed")
    }
}

impl std::ops::Sub for TracerRef {
    type Output = TracerRef;
    fn sub(self, rhs: Self) -> TracerRef {
        self.binary_op(Primitive::Sub, &rhs)
            .expect("sub tracing failed")
    }
}

impl std::ops::Mul for &TracerRef {
    type Output = TracerRef;
    fn mul(self, rhs: Self) -> TracerRef {
        self.binary_op(Primitive::Mul, rhs)
            .expect("mul tracing failed")
    }
}

impl std::ops::Mul for TracerRef {
    type Output = TracerRef;
    fn mul(self, rhs: Self) -> TracerRef {
        self.binary_op(Primitive::Mul, &rhs)
            .expect("mul tracing failed")
    }
}

impl std::ops::Neg for &TracerRef {
    type Output = TracerRef;
    fn neg(self) -> TracerRef {
        self.unary_op(Primitive::Neg).expect("neg tracing failed")
    }
}

impl std::ops::Neg for TracerRef {
    type Output = TracerRef;
    fn neg(self) -> TracerRef {
        self.unary_op(Primitive::Neg).expect("neg tracing failed")
    }
}

/// Trace a Rust closure into a Jaxpr, analogous to `jax.make_jaxpr(f)(*args)`.
///
/// The closure receives `TracerRef` values representing abstract inputs.
/// All primitive operations on these tracers are recorded into the Jaxpr.
///
/// # Arguments
/// - `f`: Closure that takes `&[TracerRef]` (abstract inputs) and returns `Vec<TracerRef>` (outputs).
/// - `in_avals`: Abstract values describing each input (dtype + shape).
///
/// # Returns
/// A `ClosedJaxpr` representing the traced computation.
pub fn make_jaxpr<F>(f: F, in_avals: Vec<ShapedArray>) -> Result<ClosedJaxpr, TraceError>
where
    F: FnOnce(&[TracerRef]) -> Vec<TracerRef>,
{
    let ctx = Rc::new(RefCell::new(SimpleTraceContext::new()));

    // Allocate input tracers
    let input_refs: Vec<TracerRef> = in_avals
        .into_iter()
        .map(|aval| {
            let id = ctx.borrow_mut().bind_input(aval.clone());
            TracerRef {
                id,
                aval,
                ctx: Rc::clone(&ctx),
            }
        })
        .collect();

    // Run the user's closure
    let output_refs = f(&input_refs);
    for output in &output_refs {
        output.validate_against_ctx(&ctx)?;
    }

    // Extract output IDs before dropping refs
    let output_ids: Vec<TracerId> = output_refs.iter().map(|r| r.id).collect();

    // Drop all TracerRef handles so Rc refcount goes to 1
    drop(input_refs);
    drop(output_refs);

    // Set the output tracer IDs on the active frame
    {
        let mut ctx_mut = ctx.borrow_mut();
        let frame = ctx_mut.active_frame_mut();
        frame.last_output_ids = output_ids;
    }

    // Finalize: extract the context and build Jaxpr
    let ctx_inner = Rc::try_unwrap(ctx)
        .map_err(|_| TraceError::CompositionViolation)?
        .into_inner();

    ctx_inner.finalize()
}

/// Trace a fallible closure into a Jaxpr.
///
/// Like `make_jaxpr`, but the closure can return `Err(TraceError)`.
pub fn make_jaxpr_fallible<F>(f: F, in_avals: Vec<ShapedArray>) -> Result<ClosedJaxpr, TraceError>
where
    F: FnOnce(&[TracerRef]) -> Result<Vec<TracerRef>, TraceError>,
{
    let ctx = Rc::new(RefCell::new(SimpleTraceContext::new()));

    let input_refs: Vec<TracerRef> = in_avals
        .into_iter()
        .map(|aval| {
            let id = ctx.borrow_mut().bind_input(aval.clone());
            TracerRef {
                id,
                aval,
                ctx: Rc::clone(&ctx),
            }
        })
        .collect();

    let output_refs = f(&input_refs)?;
    for output in &output_refs {
        output.validate_against_ctx(&ctx)?;
    }
    let output_ids: Vec<TracerId> = output_refs.iter().map(|r| r.id).collect();
    drop(input_refs);
    drop(output_refs);

    {
        let mut ctx_mut = ctx.borrow_mut();
        let frame = ctx_mut.active_frame_mut();
        frame.last_output_ids = output_ids;
    }

    let ctx_inner = Rc::try_unwrap(ctx)
        .map_err(|_| TraceError::CompositionViolation)?
        .into_inner();

    ctx_inner.finalize()
}

/// Build a nested trace-frame summary for a composed transform stack.
///
/// This opens one subtrace per transform (outer-to-inner), then closes them
/// (inner-to-outer) to verify stack discipline.
pub fn simulate_nested_trace_contexts(
    transforms: &[Transform],
    args: &[Value],
) -> Result<NestedTraceSummary, TraceError> {
    let in_avals: Vec<ShapedArray> = args.iter().map(ShapedArray::from_value).collect();
    let mut ctx = SimpleTraceContext::with_inputs(in_avals.clone());
    let mut opened = Vec::with_capacity(transforms.len());

    for transform in transforms {
        let trace_id = ctx.push_subtrace(in_avals.clone());
        opened.push((*transform, trace_id, ctx.nesting_depth()));
    }

    let mut frames = Vec::with_capacity(opened.len());
    for (transform, trace_id, depth) in opened.into_iter().rev() {
        let closed = ctx.pop_subtrace_closed()?;
        frames.push(NestedTraceFrameSummary {
            transform,
            trace_id,
            depth,
            equation_count: closed.jaxpr.equations.len(),
            invar_count: closed.jaxpr.invars.len(),
            outvar_count: closed.jaxpr.outvars.len(),
        });
    }
    frames.reverse();

    Ok(NestedTraceSummary {
        max_depth: transforms.len() + 1,
        frames,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        JaxprTrace, ShapedArray, SimpleTraceContext, TraceContext, TraceToJaxpr, TracerId,
    };
    use fj_core::{Atom, DType, Literal, Primitive, Shape, TensorValue, Value};
    use proptest::prelude::*;
    use proptest::test_runner::{Config as ProptestConfig, TestCaseError, TestRunner};
    use std::any::Any;
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::fs;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::path::{Path, PathBuf};
    use std::rc::Rc;
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
    fn test_infer_shift_right_arithmetic_shape() {
        run_logged_test(
            "test_infer_shift_right_arithmetic_shape",
            fj_test_utils::fixture_id_from_json(&("shift-right-arithmetic-shape", [8_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![
                    ShapedArray {
                        dtype: DType::I64,
                        shape: Shape { dims: vec![8] },
                    },
                    ShapedArray {
                        dtype: DType::I64,
                        shape: Shape { dims: vec![8] },
                    },
                ]);
                let out = ctx
                    .process_primitive(
                        Primitive::ShiftRightArithmetic,
                        &[TracerId(1), TracerId(2)],
                        BTreeMap::new(),
                    )
                    .expect("shift-right arithmetic inference");
                let aval = ctx.tracer_aval(out[0]).expect("aval present");
                assert_eq!(aval.dtype, DType::I64);
                assert_eq!(aval.shape, Shape { dims: vec![8] });
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn test_infer_shift_right_logical_shape() {
        run_logged_test(
            "test_infer_shift_right_logical_shape",
            fj_test_utils::fixture_id_from_json(&("shift-right-logical-shape", [8_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![
                    ShapedArray {
                        dtype: DType::I64,
                        shape: Shape { dims: vec![8] },
                    },
                    ShapedArray {
                        dtype: DType::I64,
                        shape: Shape { dims: vec![8] },
                    },
                ]);
                let out = ctx
                    .process_primitive(
                        Primitive::ShiftRightLogical,
                        &[TracerId(1), TracerId(2)],
                        BTreeMap::new(),
                    )
                    .expect("shift-right logical inference");
                let aval = ctx.tracer_aval(out[0]).expect("aval present");
                assert_eq!(aval.dtype, DType::I64);
                assert_eq!(aval.shape, Shape { dims: vec![8] });
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn test_infer_rev_shape() {
        run_logged_test(
            "test_infer_rev_shape",
            fj_test_utils::fixture_id_from_json(&("rev-shape", [2_u32, 3_u32, 4_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::F64,
                    shape: Shape {
                        dims: vec![2, 3, 4],
                    },
                }]);
                let mut params = BTreeMap::new();
                params.insert("dimensions".to_owned(), "1".to_owned());
                let out = ctx
                    .process_primitive(Primitive::Rev, &[TracerId(1)], params)
                    .expect("rev inference should succeed");
                let aval = ctx.tracer_aval(out[0]).expect("aval present");
                assert_eq!(
                    aval.shape,
                    Shape {
                        dims: vec![2, 3, 4]
                    }
                );
                assert_eq!(aval.dtype, DType::F64);
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn test_infer_squeeze_shape() {
        run_logged_test(
            "test_infer_squeeze_shape",
            fj_test_utils::fixture_id_from_json(&("squeeze-shape", [1_u32, 3_u32, 1_u32, 4_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::F64,
                    shape: Shape {
                        dims: vec![1, 3, 1, 4],
                    },
                }]);
                let mut params = BTreeMap::new();
                params.insert("dimensions".to_owned(), "0,2".to_owned());
                let out = ctx
                    .process_primitive(Primitive::Squeeze, &[TracerId(1)], params)
                    .expect("squeeze inference should succeed");
                let aval = ctx.tracer_aval(out[0]).expect("aval present");
                assert_eq!(aval.shape, Shape { dims: vec![3, 4] });
                assert_eq!(aval.dtype, DType::F64);
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn test_infer_split_shapes() {
        run_logged_test(
            "test_infer_split_shapes",
            fj_test_utils::fixture_id_from_json(&("split-shape", [8_u32])).expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::F64,
                    shape: Shape::vector(8),
                }]);
                let mut params = BTreeMap::new();
                params.insert("axis".to_owned(), "0".to_owned());
                params.insert("num_sections".to_owned(), "4".to_owned());
                let out = ctx
                    .process_primitive(Primitive::Split, &[TracerId(1)], params)
                    .expect("split inference should succeed");
                let aval = ctx.tracer_aval(out[0]).expect("aval present");
                assert_eq!(aval.shape, Shape { dims: vec![4, 2] });
                assert_eq!(aval.dtype, DType::F64);
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn test_infer_expand_dims_shape() {
        run_logged_test(
            "test_infer_expand_dims_shape",
            fj_test_utils::fixture_id_from_json(&("expand-dims-shape", [3_u32, 4_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::F64,
                    shape: Shape { dims: vec![3, 4] },
                }]);
                let mut params = BTreeMap::new();
                params.insert("axis".to_owned(), "1".to_owned());
                let out = ctx
                    .process_primitive(Primitive::ExpandDims, &[TracerId(1)], params)
                    .expect("expand_dims inference should succeed");
                let aval = ctx.tracer_aval(out[0]).expect("aval present");
                assert_eq!(
                    aval.shape,
                    Shape {
                        dims: vec![3, 1, 4]
                    }
                );
                assert_eq!(aval.dtype, DType::F64);
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn test_infer_cbrt_shape() {
        run_logged_test(
            "test_infer_cbrt_shape",
            fj_test_utils::fixture_id_from_json(&("cbrt-shape", [6_u32])).expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::F64,
                    shape: Shape::vector(6),
                }]);
                let out = ctx
                    .process_primitive(Primitive::Cbrt, &[TracerId(1)], BTreeMap::new())
                    .expect("cbrt inference should succeed");
                let aval = ctx.tracer_aval(out[0]).expect("aval present");
                assert_eq!(aval.shape, Shape::vector(6));
                assert_eq!(aval.dtype, DType::F64);
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn test_infer_is_finite_shape() {
        run_logged_test(
            "test_infer_is_finite_shape",
            fj_test_utils::fixture_id_from_json(&("is-finite-shape", [2_u32, 3_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::F64,
                    shape: Shape { dims: vec![2, 3] },
                }]);
                let out = ctx
                    .process_primitive(Primitive::IsFinite, &[TracerId(1)], BTreeMap::new())
                    .expect("is_finite inference should succeed");
                let aval = ctx.tracer_aval(out[0]).expect("aval present");
                assert_eq!(aval.shape, Shape { dims: vec![2, 3] });
                assert_eq!(aval.dtype, DType::Bool);
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn test_infer_integer_pow_shape() {
        run_logged_test(
            "test_infer_integer_pow_shape",
            fj_test_utils::fixture_id_from_json(&("integer-pow-shape", [4_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::F64,
                    shape: Shape::vector(4),
                }]);
                let mut params = BTreeMap::new();
                params.insert("exponent".to_owned(), "3".to_owned());
                let out = ctx
                    .process_primitive(Primitive::IntegerPow, &[TracerId(1)], params)
                    .expect("integer_pow inference should succeed");
                let aval = ctx.tracer_aval(out[0]).expect("aval present");
                assert_eq!(aval.shape, Shape::vector(4));
                assert_eq!(aval.dtype, DType::F64);
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn test_infer_nextafter_shape() {
        run_logged_test(
            "test_infer_nextafter_shape",
            fj_test_utils::fixture_id_from_json(&("nextafter-shape", [5_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![
                    ShapedArray {
                        dtype: DType::F64,
                        shape: Shape::vector(5),
                    },
                    ShapedArray {
                        dtype: DType::F64,
                        shape: Shape::vector(5),
                    },
                ]);
                let out = ctx
                    .process_primitive(
                        Primitive::Nextafter,
                        &[TracerId(1), TracerId(2)],
                        BTreeMap::new(),
                    )
                    .expect("nextafter inference should succeed");
                let aval = ctx.tracer_aval(out[0]).expect("aval present");
                assert_eq!(aval.shape, Shape::vector(5));
                assert_eq!(aval.dtype, DType::F64);
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
    fn prop_all_primitives_have_shape_rule() {
        run_logged_test(
            "prop_all_primitives_have_shape_rule",
            fj_test_utils::fixture_id_from_json(&("prop-all-v2-shape-rules", 1_u32))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut cases = vec![
                    (
                        Primitive::Rev,
                        vec![ShapedArray {
                            dtype: DType::F64,
                            shape: Shape {
                                dims: vec![2, 3, 4],
                            },
                        }],
                        vec![TracerId(1)],
                        {
                            let mut p = BTreeMap::new();
                            p.insert("dimensions".to_owned(), "1".to_owned());
                            p
                        },
                    ),
                    (
                        Primitive::Squeeze,
                        vec![ShapedArray {
                            dtype: DType::F64,
                            shape: Shape {
                                dims: vec![1, 3, 1, 4],
                            },
                        }],
                        vec![TracerId(1)],
                        {
                            let mut p = BTreeMap::new();
                            p.insert("dimensions".to_owned(), "0,2".to_owned());
                            p
                        },
                    ),
                    (
                        Primitive::Split,
                        vec![ShapedArray {
                            dtype: DType::F64,
                            shape: Shape::vector(8),
                        }],
                        vec![TracerId(1)],
                        {
                            let mut p = BTreeMap::new();
                            p.insert("axis".to_owned(), "0".to_owned());
                            p.insert("num_sections".to_owned(), "4".to_owned());
                            p
                        },
                    ),
                    (
                        Primitive::ExpandDims,
                        vec![ShapedArray {
                            dtype: DType::F64,
                            shape: Shape { dims: vec![3, 4] },
                        }],
                        vec![TracerId(1)],
                        {
                            let mut p = BTreeMap::new();
                            p.insert("axis".to_owned(), "1".to_owned());
                            p
                        },
                    ),
                    (
                        Primitive::Cbrt,
                        vec![ShapedArray {
                            dtype: DType::F64,
                            shape: Shape::vector(5),
                        }],
                        vec![TracerId(1)],
                        BTreeMap::new(),
                    ),
                    (
                        Primitive::IsFinite,
                        vec![ShapedArray {
                            dtype: DType::F64,
                            shape: Shape::vector(5),
                        }],
                        vec![TracerId(1)],
                        BTreeMap::new(),
                    ),
                    (
                        Primitive::IntegerPow,
                        vec![ShapedArray {
                            dtype: DType::F64,
                            shape: Shape::vector(5),
                        }],
                        vec![TracerId(1)],
                        {
                            let mut p = BTreeMap::new();
                            p.insert("exponent".to_owned(), "3".to_owned());
                            p
                        },
                    ),
                    (
                        Primitive::Nextafter,
                        vec![
                            ShapedArray {
                                dtype: DType::F64,
                                shape: Shape::vector(5),
                            },
                            ShapedArray {
                                dtype: DType::F64,
                                shape: Shape::vector(5),
                            },
                        ],
                        vec![TracerId(1), TracerId(2)],
                        BTreeMap::new(),
                    ),
                    (
                        Primitive::Cholesky,
                        vec![ShapedArray {
                            dtype: DType::F64,
                            shape: Shape { dims: vec![4, 4] },
                        }],
                        vec![TracerId(1)],
                        BTreeMap::new(),
                    ),
                    (
                        Primitive::Qr,
                        vec![ShapedArray {
                            dtype: DType::F64,
                            shape: Shape { dims: vec![5, 3] },
                        }],
                        vec![TracerId(1)],
                        BTreeMap::new(),
                    ),
                    (
                        Primitive::Svd,
                        vec![ShapedArray {
                            dtype: DType::F64,
                            shape: Shape { dims: vec![5, 3] },
                        }],
                        vec![TracerId(1)],
                        BTreeMap::new(),
                    ),
                    (
                        Primitive::TriangularSolve,
                        vec![
                            ShapedArray {
                                dtype: DType::F64,
                                shape: Shape {
                                    dims: vec![2, 4, 4],
                                },
                            },
                            ShapedArray {
                                dtype: DType::F64,
                                shape: Shape {
                                    dims: vec![2, 4, 3],
                                },
                            },
                        ],
                        vec![TracerId(1), TracerId(2)],
                        BTreeMap::new(),
                    ),
                    (
                        Primitive::Eigh,
                        vec![ShapedArray {
                            dtype: DType::F64,
                            shape: Shape { dims: vec![4, 4] },
                        }],
                        vec![TracerId(1)],
                        BTreeMap::new(),
                    ),
                    (
                        Primitive::Fft,
                        vec![ShapedArray {
                            dtype: DType::F64,
                            shape: Shape::vector(8),
                        }],
                        vec![TracerId(1)],
                        BTreeMap::new(),
                    ),
                    (
                        Primitive::Ifft,
                        vec![ShapedArray {
                            dtype: DType::Complex128,
                            shape: Shape::vector(8),
                        }],
                        vec![TracerId(1)],
                        BTreeMap::new(),
                    ),
                    (
                        Primitive::Rfft,
                        vec![ShapedArray {
                            dtype: DType::F64,
                            shape: Shape::vector(8),
                        }],
                        vec![TracerId(1)],
                        {
                            let mut p = BTreeMap::new();
                            p.insert("fft_length".to_owned(), "8".to_owned());
                            p
                        },
                    ),
                    (
                        Primitive::Irfft,
                        vec![ShapedArray {
                            dtype: DType::Complex128,
                            shape: Shape::vector(5),
                        }],
                        vec![TracerId(1)],
                        {
                            let mut p = BTreeMap::new();
                            p.insert("fft_length".to_owned(), "8".to_owned());
                            p
                        },
                    ),
                    (
                        Primitive::ShiftRightArithmetic,
                        vec![
                            ShapedArray {
                                dtype: DType::I64,
                                shape: Shape::vector(5),
                            },
                            ShapedArray {
                                dtype: DType::I64,
                                shape: Shape::vector(5),
                            },
                        ],
                        vec![TracerId(1), TracerId(2)],
                        BTreeMap::new(),
                    ),
                    (
                        Primitive::ShiftRightLogical,
                        vec![
                            ShapedArray {
                                dtype: DType::I64,
                                shape: Shape::vector(5),
                            },
                            ShapedArray {
                                dtype: DType::I64,
                                shape: Shape::vector(5),
                            },
                        ],
                        vec![TracerId(1), TracerId(2)],
                        BTreeMap::new(),
                    ),
                    (
                        Primitive::ReduceAnd,
                        vec![ShapedArray {
                            dtype: DType::Bool,
                            shape: Shape {
                                dims: vec![2, 3, 4],
                            },
                        }],
                        vec![TracerId(1)],
                        {
                            let mut p = BTreeMap::new();
                            p.insert("axes".to_owned(), "1".to_owned());
                            p
                        },
                    ),
                    (
                        Primitive::ReduceOr,
                        vec![ShapedArray {
                            dtype: DType::Bool,
                            shape: Shape {
                                dims: vec![2, 3, 4],
                            },
                        }],
                        vec![TracerId(1)],
                        {
                            let mut p = BTreeMap::new();
                            p.insert("axes".to_owned(), "0,2".to_owned());
                            p
                        },
                    ),
                    (
                        Primitive::ReduceXor,
                        vec![ShapedArray {
                            dtype: DType::Bool,
                            shape: Shape {
                                dims: vec![2, 3, 4],
                            },
                        }],
                        vec![TracerId(1)],
                        {
                            let mut p = BTreeMap::new();
                            p.insert("axes".to_owned(), "2".to_owned());
                            p
                        },
                    ),
                    (
                        Primitive::Complex,
                        vec![
                            ShapedArray {
                                dtype: DType::F64,
                                shape: Shape { dims: vec![2, 4] },
                            },
                            ShapedArray {
                                dtype: DType::F64,
                                shape: Shape { dims: vec![2, 4] },
                            },
                        ],
                        vec![TracerId(1), TracerId(2)],
                        BTreeMap::new(),
                    ),
                    (
                        Primitive::Conj,
                        vec![ShapedArray {
                            dtype: DType::Complex128,
                            shape: Shape { dims: vec![2, 4] },
                        }],
                        vec![TracerId(1)],
                        BTreeMap::new(),
                    ),
                    (
                        Primitive::Real,
                        vec![ShapedArray {
                            dtype: DType::Complex128,
                            shape: Shape { dims: vec![2, 4] },
                        }],
                        vec![TracerId(1)],
                        BTreeMap::new(),
                    ),
                    (
                        Primitive::Imag,
                        vec![ShapedArray {
                            dtype: DType::Complex128,
                            shape: Shape { dims: vec![2, 4] },
                        }],
                        vec![TracerId(1)],
                        BTreeMap::new(),
                    ),
                    (
                        Primitive::Copy,
                        vec![ShapedArray {
                            dtype: DType::F64,
                            shape: Shape { dims: vec![2, 4] },
                        }],
                        vec![TracerId(1)],
                        BTreeMap::new(),
                    ),
                    (
                        Primitive::BitcastConvertType,
                        vec![ShapedArray {
                            dtype: DType::F64,
                            shape: Shape::vector(4),
                        }],
                        vec![TracerId(1)],
                        {
                            let mut p = BTreeMap::new();
                            p.insert("new_dtype".to_owned(), "i64".to_owned());
                            p
                        },
                    ),
                    (Primitive::BroadcastedIota, vec![], vec![], {
                        let mut p = BTreeMap::new();
                        p.insert("shape".to_owned(), "2,3".to_owned());
                        p.insert("dimension".to_owned(), "1".to_owned());
                        p.insert("dtype".to_owned(), "i64".to_owned());
                        p
                    }),
                    (
                        Primitive::ReducePrecision,
                        vec![ShapedArray {
                            dtype: DType::F64,
                            shape: Shape::vector(4),
                        }],
                        vec![TracerId(1)],
                        {
                            let mut p = BTreeMap::new();
                            p.insert("exponent_bits".to_owned(), "8".to_owned());
                            p.insert("mantissa_bits".to_owned(), "7".to_owned());
                            p
                        },
                    ),
                ];

                for (primitive, in_avals, tracer_ids, params) in cases.drain(..) {
                    let mut ctx = if in_avals.is_empty() {
                        SimpleTraceContext::new()
                    } else {
                        SimpleTraceContext::with_inputs(in_avals)
                    };
                    let out = ctx
                        .process_primitive(primitive, &tracer_ids, params)
                        .map_err(|err| {
                            format!("shape inference failed for {:?}: {err}", primitive)
                        })?;
                    let _ = ctx
                        .tracer_aval(out[0])
                        .map_err(|err| format!("missing output aval for {:?}: {err}", primitive))?;
                }
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn v2_linalg_fft_shape_contracts() {
        run_logged_test(
            "v2_linalg_fft_shape_contracts",
            fj_test_utils::fixture_id_from_json(&("v2-shape-contracts", 1_u32))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut artifacts = Vec::new();

                {
                    let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![4, 4] },
                    }]);
                    let out = ctx
                        .process_primitive(Primitive::Cholesky, &[TracerId(1)], BTreeMap::new())
                        .map_err(|err| err.to_string())?;
                    if out.len() != 1 {
                        return Err(format!("cholesky output arity mismatch: {}", out.len()));
                    }
                    let aval = ctx.tracer_aval(out[0]).map_err(|err| err.to_string())?;
                    if aval.shape != (Shape { dims: vec![4, 4] }) {
                        return Err(format!("cholesky shape mismatch: {:?}", aval.shape.dims));
                    }
                    artifacts.push(format!("cholesky:{:?}", aval.shape.dims));
                }

                {
                    let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![5, 3] },
                    }]);
                    let out = ctx
                        .process_primitive(Primitive::Qr, &[TracerId(1)], BTreeMap::new())
                        .map_err(|err| err.to_string())?;
                    if out.len() != 2 {
                        return Err(format!("qr output arity mismatch: {}", out.len()));
                    }
                    let q = ctx.tracer_aval(out[0]).map_err(|err| err.to_string())?;
                    let r = ctx.tracer_aval(out[1]).map_err(|err| err.to_string())?;
                    if q.shape != (Shape { dims: vec![5, 3] }) {
                        return Err(format!("qr q-shape mismatch: {:?}", q.shape.dims));
                    }
                    if r.shape != (Shape { dims: vec![3, 3] }) {
                        return Err(format!("qr r-shape mismatch: {:?}", r.shape.dims));
                    }
                    artifacts.push(format!("qr_q:{:?}", q.shape.dims));
                    artifacts.push(format!("qr_r:{:?}", r.shape.dims));
                }

                {
                    let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![5, 3] },
                    }]);
                    let out = ctx
                        .process_primitive(Primitive::Svd, &[TracerId(1)], BTreeMap::new())
                        .map_err(|err| err.to_string())?;
                    if out.len() != 3 {
                        return Err(format!("svd output arity mismatch: {}", out.len()));
                    }
                    let u = ctx.tracer_aval(out[0]).map_err(|err| err.to_string())?;
                    let s = ctx.tracer_aval(out[1]).map_err(|err| err.to_string())?;
                    let vt = ctx.tracer_aval(out[2]).map_err(|err| err.to_string())?;
                    if u.shape != (Shape { dims: vec![5, 3] }) {
                        return Err(format!("svd u-shape mismatch: {:?}", u.shape.dims));
                    }
                    if s.shape != (Shape { dims: vec![3] }) {
                        return Err(format!("svd s-shape mismatch: {:?}", s.shape.dims));
                    }
                    if vt.shape != (Shape { dims: vec![3, 3] }) {
                        return Err(format!("svd vt-shape mismatch: {:?}", vt.shape.dims));
                    }
                    artifacts.push(format!("svd_u:{:?}", u.shape.dims));
                    artifacts.push(format!("svd_s:{:?}", s.shape.dims));
                    artifacts.push(format!("svd_vt:{:?}", vt.shape.dims));
                }

                {
                    let mut ctx = SimpleTraceContext::with_inputs(vec![
                        ShapedArray {
                            dtype: DType::F64,
                            shape: Shape {
                                dims: vec![2, 4, 4],
                            },
                        },
                        ShapedArray {
                            dtype: DType::F64,
                            shape: Shape {
                                dims: vec![2, 4, 3],
                            },
                        },
                    ]);
                    let out = ctx
                        .process_primitive(
                            Primitive::TriangularSolve,
                            &[TracerId(1), TracerId(2)],
                            BTreeMap::new(),
                        )
                        .map_err(|err| err.to_string())?;
                    if out.len() != 1 {
                        return Err(format!(
                            "triangular_solve output arity mismatch: {}",
                            out.len()
                        ));
                    }
                    let solved = ctx.tracer_aval(out[0]).map_err(|err| err.to_string())?;
                    if solved.shape
                        != (Shape {
                            dims: vec![2, 4, 3],
                        })
                    {
                        return Err(format!(
                            "triangular_solve shape mismatch: {:?}",
                            solved.shape.dims
                        ));
                    }
                    artifacts.push(format!("triangular_solve:{:?}", solved.shape.dims));
                }

                {
                    let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![4, 4] },
                    }]);
                    let out = ctx
                        .process_primitive(Primitive::Eigh, &[TracerId(1)], BTreeMap::new())
                        .map_err(|err| err.to_string())?;
                    if out.len() != 2 {
                        return Err(format!("eigh output arity mismatch: {}", out.len()));
                    }
                    let evals = ctx.tracer_aval(out[0]).map_err(|err| err.to_string())?;
                    let evecs = ctx.tracer_aval(out[1]).map_err(|err| err.to_string())?;
                    if evals.shape != (Shape { dims: vec![4] }) {
                        return Err(format!(
                            "eigh eigenvalue shape mismatch: {:?}",
                            evals.shape.dims
                        ));
                    }
                    if evecs.shape != (Shape { dims: vec![4, 4] }) {
                        return Err(format!(
                            "eigh eigenvector shape mismatch: {:?}",
                            evecs.shape.dims
                        ));
                    }
                    artifacts.push(format!("eigh_vals:{:?}", evals.shape.dims));
                    artifacts.push(format!("eigh_vecs:{:?}", evecs.shape.dims));
                }

                {
                    let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                        dtype: DType::F64,
                        shape: Shape::vector(8),
                    }]);
                    let fft = ctx
                        .process_primitive(Primitive::Fft, &[TracerId(1)], BTreeMap::new())
                        .map_err(|err| err.to_string())?;
                    let (fft_shape, fft_dtype) = {
                        let fft_aval = ctx.tracer_aval(fft[0]).map_err(|err| err.to_string())?;
                        (fft_aval.shape.clone(), fft_aval.dtype)
                    };
                    if fft_shape != Shape::vector(8) || fft_dtype != DType::Complex128 {
                        return Err(format!(
                            "fft mismatch shape={:?} dtype={:?}",
                            fft_shape.dims, fft_dtype
                        ));
                    }

                    let ifft = ctx
                        .process_primitive(Primitive::Ifft, &[fft[0]], BTreeMap::new())
                        .map_err(|err| err.to_string())?;
                    let (ifft_shape, ifft_dtype) = {
                        let ifft_aval = ctx.tracer_aval(ifft[0]).map_err(|err| err.to_string())?;
                        (ifft_aval.shape.clone(), ifft_aval.dtype)
                    };
                    if ifft_shape != Shape::vector(8) || ifft_dtype != DType::Complex128 {
                        return Err(format!(
                            "ifft mismatch shape={:?} dtype={:?}",
                            ifft_shape.dims, ifft_dtype
                        ));
                    }

                    let mut rfft_params = BTreeMap::new();
                    rfft_params.insert("fft_length".to_owned(), "8".to_owned());
                    let rfft = ctx
                        .process_primitive(Primitive::Rfft, &[TracerId(1)], rfft_params)
                        .map_err(|err| err.to_string())?;
                    let (rfft_shape, rfft_dtype) = {
                        let rfft_aval = ctx.tracer_aval(rfft[0]).map_err(|err| err.to_string())?;
                        (rfft_aval.shape.clone(), rfft_aval.dtype)
                    };
                    if rfft_shape != Shape::vector(5) || rfft_dtype != DType::Complex128 {
                        return Err(format!(
                            "rfft mismatch shape={:?} dtype={:?}",
                            rfft_shape.dims, rfft_dtype
                        ));
                    }

                    let mut irfft_params = BTreeMap::new();
                    irfft_params.insert("fft_length".to_owned(), "8".to_owned());
                    let irfft = ctx
                        .process_primitive(Primitive::Irfft, &[rfft[0]], irfft_params)
                        .map_err(|err| err.to_string())?;
                    let (irfft_shape, irfft_dtype) = {
                        let irfft_aval =
                            ctx.tracer_aval(irfft[0]).map_err(|err| err.to_string())?;
                        (irfft_aval.shape.clone(), irfft_aval.dtype)
                    };
                    if irfft_shape != Shape::vector(8) || irfft_dtype != DType::F64 {
                        return Err(format!(
                            "irfft mismatch shape={:?} dtype={:?}",
                            irfft_shape.dims, irfft_dtype
                        ));
                    }
                    artifacts.push(format!("fft:{:?}", fft_shape.dims));
                    artifacts.push(format!("rfft:{:?}", rfft_shape.dims));
                    artifacts.push(format!("irfft:{:?}", irfft_shape.dims));
                }

                Ok(artifacts)
            },
        );
    }

    #[test]
    fn prop_shape_inference_deterministic() {
        run_logged_test(
            "prop_shape_inference_deterministic",
            fj_test_utils::fixture_id_from_json(&(
                "prop-shape-deterministic",
                fj_test_utils::property_test_case_count(),
            ))
            .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut runner = TestRunner::new(ProptestConfig::with_cases(
                    fj_test_utils::property_test_case_count(),
                ));
                let strategy = (proptest::collection::vec(1_u32..=5_u32, 1..=4), 0_u8..=1_u8);
                runner
                    .run(&strategy, |(dims, axis_sel)| {
                        let mut ctx_a = SimpleTraceContext::with_inputs(vec![ShapedArray {
                            dtype: DType::F64,
                            shape: Shape { dims: dims.clone() },
                        }]);
                        let mut ctx_b = SimpleTraceContext::with_inputs(vec![ShapedArray {
                            dtype: DType::F64,
                            shape: Shape { dims: dims.clone() },
                        }]);

                        let mut params = BTreeMap::new();
                        let axis = usize::from(axis_sel) % dims.len();
                        params.insert("axis".to_owned(), axis.to_string());

                        let primitive = if axis_sel % 2 == 0 {
                            Primitive::ExpandDims
                        } else {
                            Primitive::Squeeze
                        };
                        if primitive == Primitive::Squeeze {
                            // Ensure squeeze has at least one singleton axis.
                            let mut singleton_dims = dims.clone();
                            singleton_dims[axis] = 1;
                            ctx_a = SimpleTraceContext::with_inputs(vec![ShapedArray {
                                dtype: DType::F64,
                                shape: Shape {
                                    dims: singleton_dims.clone(),
                                },
                            }]);
                            ctx_b = SimpleTraceContext::with_inputs(vec![ShapedArray {
                                dtype: DType::F64,
                                shape: Shape {
                                    dims: singleton_dims,
                                },
                            }]);
                            params.clear();
                            params.insert("dimensions".to_owned(), axis.to_string());
                        }

                        let out_a = ctx_a
                            .process_primitive(primitive, &[TracerId(1)], params.clone())
                            .map_err(|err| TestCaseError::fail(err.to_string()))?;
                        let out_b = ctx_b
                            .process_primitive(primitive, &[TracerId(1)], params)
                            .map_err(|err| TestCaseError::fail(err.to_string()))?;
                        let aval_a = ctx_a
                            .tracer_aval(out_a[0])
                            .map_err(|err| TestCaseError::fail(err.to_string()))?;
                        let aval_b = ctx_b
                            .tracer_aval(out_b[0])
                            .map_err(|err| TestCaseError::fail(err.to_string()))?;

                        prop_assert_eq!(aval_a.shape.clone(), aval_b.shape.clone());
                        prop_assert_eq!(aval_a.dtype, aval_b.dtype);
                        Ok(())
                    })
                    .map_err(|err| err.to_string())?;
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn e2e_shape_inference_all_v2_prims() {
        run_logged_test(
            "e2e_shape_inference_all_v2_prims",
            fj_test_utils::fixture_id_from_json(&("e2e-v2-shape-inference", 1_u32))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut records = Vec::new();

                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::F64,
                    shape: Shape { dims: vec![2, 3] },
                }]);
                let rev_out = ctx
                    .process_primitive(Primitive::Rev, &[TracerId(1)], BTreeMap::new())
                    .map_err(|err| err.to_string())?;
                let rev_aval = ctx
                    .tracer_aval(rev_out[0])
                    .map_err(|err| format!("missing rev aval: {err}"))?;
                records.push(serde_json::json!({
                    "primitive": "rev",
                    "input_shapes": [[2, 3]],
                    "inferred_output_shape": rev_aval.shape.dims,
                    "expected_shape": [2, 3],
                    "match": rev_aval.shape == Shape { dims: vec![2, 3] },
                }));

                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::F64,
                    shape: Shape::vector(4),
                }]);
                let mut bitcast_params = BTreeMap::new();
                bitcast_params.insert("new_dtype".to_owned(), "i64".to_owned());
                let bitcast_out = ctx
                    .process_primitive(
                        Primitive::BitcastConvertType,
                        &[TracerId(1)],
                        bitcast_params,
                    )
                    .map_err(|err| err.to_string())?;
                let bitcast_aval = ctx
                    .tracer_aval(bitcast_out[0])
                    .map_err(|err| format!("missing bitcast aval: {err}"))?;
                records.push(serde_json::json!({
                    "primitive": "bitcast_convert_type",
                    "input_shapes": [[4]],
                    "inferred_output_shape": bitcast_aval.shape.dims,
                    "expected_shape": [4],
                    "match": bitcast_aval.shape == Shape::vector(4),
                }));

                let mut ctx = SimpleTraceContext::new();
                let mut bcast_iota_params = BTreeMap::new();
                bcast_iota_params.insert("shape".to_owned(), "2,3".to_owned());
                bcast_iota_params.insert("dimension".to_owned(), "1".to_owned());
                bcast_iota_params.insert("dtype".to_owned(), "i64".to_owned());
                let bcast_iota_out = ctx
                    .process_primitive(Primitive::BroadcastedIota, &[], bcast_iota_params)
                    .map_err(|err| err.to_string())?;
                let bcast_iota_aval = ctx
                    .tracer_aval(bcast_iota_out[0])
                    .map_err(|err| format!("missing broadcasted_iota aval: {err}"))?;
                records.push(serde_json::json!({
                    "primitive": "broadcasted_iota",
                    "input_shapes": [],
                    "inferred_output_shape": bcast_iota_aval.shape.dims,
                    "expected_shape": [2, 3],
                    "match": bcast_iota_aval.shape == Shape { dims: vec![2, 3] },
                }));

                let all_match = records.iter().all(|row| {
                    row.get("match")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false)
                });
                if !all_match {
                    return Err(
                        "one or more V2 primitive shape-inference checks mismatched".to_owned()
                    );
                }

                let log_path = repo_root()
                    .join("artifacts")
                    .join("e2e")
                    .join("e2e_v2_shape_inference.e2e.json");
                if let Some(parent) = log_path.parent() {
                    fs::create_dir_all(parent)
                        .map_err(|err| format!("failed to create e2e log dir: {err}"))?;
                }
                let payload = serde_json::to_string_pretty(&records)
                    .map_err(|err| format!("failed to serialize e2e log: {err}"))?;
                fs::write(&log_path, payload)
                    .map_err(|err| format!("failed to write e2e log: {err}"))?;

                Ok(vec![log_path.display().to_string()])
            },
        );
    }

    #[test]
    fn test_infer_reduce_and_shape() {
        run_logged_test(
            "test_infer_reduce_and_shape",
            fj_test_utils::fixture_id_from_json(&("reduce-and-shape", [2_u32, 3_u32, 4_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::Bool,
                    shape: Shape {
                        dims: vec![2, 3, 4],
                    },
                }]);
                let mut params = BTreeMap::new();
                params.insert("axes".to_owned(), "1".to_owned());
                let out = ctx
                    .process_primitive(Primitive::ReduceAnd, &[TracerId(1)], params)
                    .expect("reduce_and inference should succeed");
                let aval = ctx.tracer_aval(out[0]).expect("aval present");
                assert_eq!(aval.dtype, DType::Bool);
                assert_eq!(aval.shape, Shape { dims: vec![2, 4] });
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn test_infer_reduce_or_full_reduce() {
        run_logged_test(
            "test_infer_reduce_or_full_reduce",
            fj_test_utils::fixture_id_from_json(&("reduce-or-full", [2_u32, 2_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::Bool,
                    shape: Shape { dims: vec![2, 2] },
                }]);
                let out = ctx
                    .process_primitive(Primitive::ReduceOr, &[TracerId(1)], BTreeMap::new())
                    .expect("reduce_or inference should succeed");
                let aval = ctx.tracer_aval(out[0]).expect("aval present");
                assert_eq!(aval.dtype, DType::Bool);
                assert_eq!(aval.shape, Shape::scalar());
                Ok(Vec::new())
            },
        );
    }

    // ── Scatter / Gather shape inference edge cases ──────────────────

    #[test]
    fn scatter_requires_exactly_three_inputs() {
        run_logged_test(
            "scatter_requires_exactly_three_inputs",
            fj_test_utils::fixture_id_from_json(&("scatter", "arity")).expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![
                    ShapedArray {
                        dtype: DType::F64,
                        shape: Shape::vector(5),
                    },
                    ShapedArray {
                        dtype: DType::I64,
                        shape: Shape::scalar(),
                    },
                ]);
                // Only 2 inputs — should fail
                let err = ctx
                    .process_primitive(
                        Primitive::Scatter,
                        &[TracerId(1), TracerId(2)],
                        BTreeMap::new(),
                    )
                    .unwrap_err();
                assert!(
                    format!("{err:?}").contains("expected 3 inputs"),
                    "error should mention arity: {err:?}"
                );
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn scatter_rejects_rank0_operand() {
        run_logged_test(
            "scatter_rejects_rank0_operand",
            fj_test_utils::fixture_id_from_json(&("scatter", "rank0")).expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![
                    ShapedArray {
                        dtype: DType::F64,
                        shape: Shape::scalar(),
                    },
                    ShapedArray {
                        dtype: DType::I64,
                        shape: Shape::scalar(),
                    },
                    ShapedArray {
                        dtype: DType::F64,
                        shape: Shape::scalar(),
                    },
                ]);
                let err = ctx
                    .process_primitive(
                        Primitive::Scatter,
                        &[TracerId(1), TracerId(2), TracerId(3)],
                        BTreeMap::new(),
                    )
                    .unwrap_err();
                assert!(
                    format!("{err:?}").contains("rank-0"),
                    "error should mention rank-0: {err:?}"
                );
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn scatter_output_shape_matches_operand() {
        run_logged_test(
            "scatter_output_shape_matches_operand",
            fj_test_utils::fixture_id_from_json(&("scatter", "shape")).expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![
                    ShapedArray {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![3, 4] },
                    },
                    ShapedArray {
                        dtype: DType::I64,
                        shape: Shape::vector(2),
                    },
                    ShapedArray {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![2, 4] },
                    },
                ]);
                let out = ctx
                    .process_primitive(
                        Primitive::Scatter,
                        &[TracerId(1), TracerId(2), TracerId(3)],
                        BTreeMap::new(),
                    )
                    .expect("scatter should succeed");
                assert_eq!(out.len(), 1);
                let out_aval = ctx.tracer_aval(out[0]).unwrap();
                assert_eq!(
                    out_aval.shape.dims,
                    vec![3, 4],
                    "scatter output should match operand shape"
                );
                assert_eq!(out_aval.dtype, DType::F64);
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn gather_output_shape_with_slice_sizes() {
        run_logged_test(
            "gather_output_shape_with_slice_sizes",
            fj_test_utils::fixture_id_from_json(&("gather", "slice_sizes"))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![
                    ShapedArray {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![10, 8] },
                    },
                    ShapedArray {
                        dtype: DType::I64,
                        shape: Shape::vector(3),
                    },
                ]);
                let mut params = BTreeMap::new();
                params.insert("slice_sizes".to_owned(), "1,4".to_owned());
                let out = ctx
                    .process_primitive(Primitive::Gather, &[TracerId(1), TracerId(2)], params)
                    .expect("gather should succeed");
                assert_eq!(out.len(), 1);
                let out_aval = ctx.tracer_aval(out[0]).unwrap();
                // Output shape = indices.shape ++ slice_sizes = [3] ++ [1,4] = [3,1,4]
                assert_eq!(
                    out_aval.shape.dims,
                    vec![3, 1, 4],
                    "gather output should be indices.shape ++ slice_sizes"
                );
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn gather_default_slice_sizes_are_ones() {
        run_logged_test(
            "gather_default_slice_sizes_are_ones",
            fj_test_utils::fixture_id_from_json(&("gather", "default_slices"))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![
                    ShapedArray {
                        dtype: DType::I64,
                        shape: Shape { dims: vec![5, 3] },
                    },
                    ShapedArray {
                        dtype: DType::I64,
                        shape: Shape::vector(2),
                    },
                ]);
                let out = ctx
                    .process_primitive(
                        Primitive::Gather,
                        &[TracerId(1), TracerId(2)],
                        BTreeMap::new(),
                    )
                    .expect("gather should succeed with default slice_sizes");
                let out_aval = ctx.tracer_aval(out[0]).unwrap();
                // Default slice_sizes = [1,1] for rank-2 operand
                // Output = indices.shape ++ [1,1] = [2,1,1]
                assert_eq!(
                    out_aval.shape.dims,
                    vec![2, 1, 1],
                    "default slice_sizes should be [1,1] for rank-2"
                );
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn gather_requires_two_inputs() {
        run_logged_test(
            "gather_requires_two_inputs",
            fj_test_utils::fixture_id_from_json(&("gather", "arity")).expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::F64,
                    shape: Shape::vector(5),
                }]);
                let err = ctx
                    .process_primitive(Primitive::Gather, &[TracerId(1)], BTreeMap::new())
                    .unwrap_err();
                assert!(
                    format!("{err:?}").contains("at least 2 inputs"),
                    "error should mention arity: {err:?}"
                );
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

    #[test]
    fn infer_one_hot_appends_num_classes_dim() {
        run_logged_test(
            "infer_one_hot_appends_num_classes_dim",
            fj_test_utils::fixture_id_from_json(&("one-hot-shape", [4_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::I64,
                    shape: Shape::vector(4),
                }]);
                let idx = TracerId(1);

                let mut params = BTreeMap::new();
                params.insert("num_classes".to_owned(), "5".to_owned());

                let out = ctx
                    .process_primitive(Primitive::OneHot, &[idx], params)
                    .expect("one_hot inference should succeed");
                let aval = ctx.tracer_aval(out[0]).expect("aval present");
                assert_eq!(aval.shape, Shape { dims: vec![4, 5] });
                assert_eq!(aval.dtype, DType::F64);
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn test_infer_complex_shape_and_dtype() {
        run_logged_test(
            "test_infer_complex_shape_and_dtype",
            fj_test_utils::fixture_id_from_json(&("complex-shape", [3_u32, 2_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![
                    ShapedArray {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![3, 2] },
                    },
                    ShapedArray {
                        dtype: DType::F64,
                        shape: Shape { dims: vec![3, 2] },
                    },
                ]);
                let out = ctx
                    .process_primitive(
                        Primitive::Complex,
                        &[TracerId(1), TracerId(2)],
                        BTreeMap::new(),
                    )
                    .expect("complex inference should succeed");
                let aval = ctx.tracer_aval(out[0]).expect("aval present");
                assert_eq!(aval.dtype, DType::Complex128);
                assert_eq!(aval.shape, Shape { dims: vec![3, 2] });
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn test_infer_conj_shape_and_dtype() {
        run_logged_test(
            "test_infer_conj_shape_and_dtype",
            fj_test_utils::fixture_id_from_json(&("conj-shape", [5_u32])).expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::Complex128,
                    shape: Shape::vector(5),
                }]);
                let out = ctx
                    .process_primitive(Primitive::Conj, &[TracerId(1)], BTreeMap::new())
                    .expect("conj inference should succeed");
                let aval = ctx.tracer_aval(out[0]).expect("aval present");
                assert_eq!(aval.dtype, DType::Complex128);
                assert_eq!(aval.shape, Shape::vector(5));
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn test_infer_real_shape_and_dtype() {
        run_logged_test(
            "test_infer_real_shape_and_dtype",
            fj_test_utils::fixture_id_from_json(&("real-shape", [2_u32, 4_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::Complex128,
                    shape: Shape { dims: vec![2, 4] },
                }]);
                let out = ctx
                    .process_primitive(Primitive::Real, &[TracerId(1)], BTreeMap::new())
                    .expect("real inference should succeed");
                let aval = ctx.tracer_aval(out[0]).expect("aval present");
                assert_eq!(aval.dtype, DType::F64);
                assert_eq!(aval.shape, Shape { dims: vec![2, 4] });
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn test_infer_imag_shape_and_dtype() {
        run_logged_test(
            "test_infer_imag_shape_and_dtype",
            fj_test_utils::fixture_id_from_json(&("imag-shape", [2_u32, 4_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::Complex128,
                    shape: Shape { dims: vec![2, 4] },
                }]);
                let out = ctx
                    .process_primitive(Primitive::Imag, &[TracerId(1)], BTreeMap::new())
                    .expect("imag inference should succeed");
                let aval = ctx.tracer_aval(out[0]).expect("aval present");
                assert_eq!(aval.dtype, DType::F64);
                assert_eq!(aval.shape, Shape { dims: vec![2, 4] });
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn test_infer_copy_shape() {
        run_logged_test(
            "test_infer_copy_shape",
            fj_test_utils::fixture_id_from_json(&("copy-shape", [3_u32, 2_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::F64,
                    shape: Shape { dims: vec![3, 2] },
                }]);
                let out = ctx
                    .process_primitive(Primitive::Copy, &[TracerId(1)], BTreeMap::new())
                    .expect("copy inference should succeed");
                let aval = ctx.tracer_aval(out[0]).expect("aval present");
                assert_eq!(aval.dtype, DType::F64);
                assert_eq!(aval.shape, Shape { dims: vec![3, 2] });
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn test_infer_bitcast_convert_type_shape() {
        run_logged_test(
            "test_infer_bitcast_convert_type_shape",
            fj_test_utils::fixture_id_from_json(&("bitcast-shape", [4_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::F64,
                    shape: Shape::vector(4),
                }]);
                let mut params = BTreeMap::new();
                params.insert("new_dtype".to_owned(), "i64".to_owned());
                let out = ctx
                    .process_primitive(Primitive::BitcastConvertType, &[TracerId(1)], params)
                    .expect("bitcast inference should succeed");
                let aval = ctx.tracer_aval(out[0]).expect("aval present");
                assert_eq!(aval.dtype, DType::I64);
                assert_eq!(aval.shape, Shape::vector(4));
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn test_infer_broadcasted_iota_shape() {
        run_logged_test(
            "test_infer_broadcasted_iota_shape",
            fj_test_utils::fixture_id_from_json(&("broadcasted-iota-shape", [2_u32, 3_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::new();
                let mut params = BTreeMap::new();
                params.insert("shape".to_owned(), "2,3".to_owned());
                params.insert("dimension".to_owned(), "1".to_owned());
                params.insert("dtype".to_owned(), "i64".to_owned());
                let out = ctx
                    .process_primitive(Primitive::BroadcastedIota, &[], params)
                    .expect("broadcasted_iota inference should succeed");
                let aval = ctx.tracer_aval(out[0]).expect("aval present");
                assert_eq!(aval.dtype, DType::I64);
                assert_eq!(aval.shape, Shape { dims: vec![2, 3] });
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn test_infer_reduce_precision_shape() {
        run_logged_test(
            "test_infer_reduce_precision_shape",
            fj_test_utils::fixture_id_from_json(&("reduce-precision-shape", [5_u32]))
                .expect("fixture digest"),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
                    dtype: DType::F64,
                    shape: Shape::vector(5),
                }]);
                let mut params = BTreeMap::new();
                params.insert("exponent_bits".to_owned(), "8".to_owned());
                params.insert("mantissa_bits".to_owned(), "7".to_owned());
                let out = ctx
                    .process_primitive(Primitive::ReducePrecision, &[TracerId(1)], params)
                    .expect("reduce_precision inference should succeed");
                let aval = ctx.tracer_aval(out[0]).expect("aval present");
                assert_eq!(aval.dtype, DType::F64);
                assert_eq!(aval.shape, Shape::vector(5));
                Ok(Vec::new())
            },
        );
    }

    // ── make_jaxpr tests ─────────────────────────────────────────────

    #[test]
    fn test_make_jaxpr_identity() {
        use super::make_jaxpr;
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        };
        let closed = make_jaxpr(|inputs| vec![inputs[0].clone()], vec![aval]).unwrap();
        assert_eq!(closed.jaxpr.invars.len(), 1);
        assert_eq!(closed.jaxpr.outvars.len(), 1);
        assert!(
            closed.jaxpr.equations.is_empty(),
            "identity should have no equations"
        );
        assert_eq!(closed.jaxpr.invars[0], closed.jaxpr.outvars[0]);
    }

    #[test]
    fn test_make_jaxpr_add() {
        use super::make_jaxpr;
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        };
        let closed = make_jaxpr(
            |inputs| vec![&inputs[0] + &inputs[1]],
            vec![aval.clone(), aval],
        )
        .unwrap();
        assert_eq!(closed.jaxpr.invars.len(), 2);
        assert_eq!(closed.jaxpr.outvars.len(), 1);
        assert_eq!(closed.jaxpr.equations.len(), 1);
        assert_eq!(closed.jaxpr.equations[0].primitive, Primitive::Add);
    }

    #[test]
    fn test_make_jaxpr_chain() {
        use super::make_jaxpr;
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        };
        // sin(cos(x)) — two-equation chain
        let closed = make_jaxpr(
            |inputs| {
                let cos_x = inputs[0].unary_op(Primitive::Cos).unwrap();
                let sin_cos_x = cos_x.unary_op(Primitive::Sin).unwrap();
                vec![sin_cos_x]
            },
            vec![aval],
        )
        .unwrap();
        assert_eq!(closed.jaxpr.equations.len(), 2);
        assert_eq!(closed.jaxpr.equations[0].primitive, Primitive::Cos);
        assert_eq!(closed.jaxpr.equations[1].primitive, Primitive::Sin);
    }

    #[test]
    fn test_make_jaxpr_multi_output() {
        use super::make_jaxpr;
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        };
        // f(x) -> (x+x, x*x)
        let closed = make_jaxpr(
            |inputs| {
                let doubled = &inputs[0] + &inputs[0];
                let squared = &inputs[0] * &inputs[0];
                vec![doubled, squared]
            },
            vec![aval],
        )
        .unwrap();
        assert_eq!(closed.jaxpr.invars.len(), 1);
        assert_eq!(closed.jaxpr.outvars.len(), 2);
        assert_eq!(closed.jaxpr.equations.len(), 2);
    }

    #[test]
    fn test_make_jaxpr_shape_validated() {
        use super::make_jaxpr;
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape { dims: vec![3] },
        };
        // f(x) -> x + x, should infer shape [3]
        let closed = make_jaxpr(|inputs| vec![&inputs[0] + &inputs[0]], vec![aval]).unwrap();
        assert_eq!(closed.jaxpr.equations.len(), 1);
        // Output exists and is well-formed
        assert_eq!(closed.jaxpr.outvars.len(), 1);
    }

    #[test]
    fn test_make_jaxpr_broadcast() {
        use super::make_jaxpr;
        let scalar = ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        };
        let vector = ShapedArray {
            dtype: DType::F64,
            shape: Shape { dims: vec![3] },
        };
        // f(scalar, vector) -> scalar + vector = vector
        let closed =
            make_jaxpr(|inputs| vec![&inputs[0] + &inputs[1]], vec![scalar, vector]).unwrap();
        assert_eq!(closed.jaxpr.equations.len(), 1);
        assert_eq!(closed.jaxpr.equations[0].primitive, Primitive::Add);
    }

    #[test]
    fn test_make_jaxpr_reduction() {
        use super::make_jaxpr;
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape { dims: vec![3] },
        };
        // f(x) -> reduce_sum(x)
        let closed = make_jaxpr(
            |inputs| {
                let mut params = BTreeMap::new();
                params.insert("axes".to_owned(), "0".to_owned());
                inputs[0]
                    .primitive_with_params(Primitive::ReduceSum, &[], params)
                    .unwrap()
            },
            vec![aval],
        )
        .unwrap();
        assert_eq!(closed.jaxpr.equations.len(), 1);
        assert_eq!(closed.jaxpr.equations[0].primitive, Primitive::ReduceSum);
    }

    #[test]
    fn test_make_jaxpr_dot() {
        use super::make_jaxpr;
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape { dims: vec![3] },
        };
        // f(x, y) -> dot(x, y)
        let closed = make_jaxpr(
            |inputs| {
                inputs[0]
                    .primitive_with_params(Primitive::Dot, &[&inputs[1]], BTreeMap::new())
                    .unwrap()
            },
            vec![aval.clone(), aval],
        )
        .unwrap();
        assert_eq!(closed.jaxpr.equations.len(), 1);
        assert_eq!(closed.jaxpr.equations[0].primitive, Primitive::Dot);
    }

    #[test]
    fn test_make_jaxpr_deterministic() {
        use super::make_jaxpr;
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        };
        let f = |inputs: &[super::TracerRef]| {
            let sq = &inputs[0] * &inputs[0];
            vec![sq + inputs[0].clone()]
        };
        let closed1 = make_jaxpr(f, vec![aval.clone()]).unwrap();
        let closed2 = make_jaxpr(
            |inputs: &[super::TracerRef]| {
                let sq = &inputs[0] * &inputs[0];
                vec![sq + inputs[0].clone()]
            },
            vec![aval],
        )
        .unwrap();
        // Same structure
        assert_eq!(closed1.jaxpr.equations.len(), closed2.jaxpr.equations.len());
        for (e1, e2) in closed1
            .jaxpr
            .equations
            .iter()
            .zip(closed2.jaxpr.equations.iter())
        {
            assert_eq!(e1.primitive, e2.primitive);
        }
    }

    #[test]
    fn test_make_jaxpr_rejects_foreign_output_tracer() {
        use super::{TraceError, TracerRef, make_jaxpr};
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        };

        let foreign = make_jaxpr(|inputs| vec![inputs[0].clone()], vec![aval.clone()]).unwrap();
        assert!(foreign.jaxpr.equations.is_empty());

        let foreign_ctx = Rc::new(RefCell::new(SimpleTraceContext::new()));
        let foreign_id = foreign_ctx.borrow_mut().bind_input(aval.clone());
        let foreign_ref = TracerRef {
            id: foreign_id,
            aval: aval.clone(),
            ctx: Rc::clone(&foreign_ctx),
        };

        let err = make_jaxpr(|_inputs| vec![foreign_ref.clone()], vec![aval]).unwrap_err();
        assert!(matches!(err, TraceError::ForeignTracerContext { .. }));
    }

    #[test]
    fn test_tracer_binary_op_rejects_foreign_context() {
        use super::{TraceError, TracerRef};
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        };

        let ctx_a = Rc::new(RefCell::new(SimpleTraceContext::new()));
        let id_a = ctx_a.borrow_mut().bind_input(aval.clone());
        let a = TracerRef {
            id: id_a,
            aval: aval.clone(),
            ctx: Rc::clone(&ctx_a),
        };

        let ctx_b = Rc::new(RefCell::new(SimpleTraceContext::new()));
        let id_b = ctx_b.borrow_mut().bind_input(aval.clone());
        let b = TracerRef {
            id: id_b,
            aval: aval.clone(),
            ctx: Rc::clone(&ctx_b),
        };

        let err = a.binary_op(Primitive::Add, &b).unwrap_err();
        assert!(matches!(err, TraceError::ForeignTracerContext { .. }));
    }

    #[test]
    fn test_tracer_unary_op_rejects_cached_aval_mismatch() {
        use super::{TraceError, TracerRef};
        let actual_aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        };
        let stale_aval = ShapedArray {
            dtype: DType::Bool,
            shape: Shape::scalar(),
        };

        let ctx = Rc::new(RefCell::new(SimpleTraceContext::new()));
        let id = ctx.borrow_mut().bind_input(actual_aval);
        let stale = TracerRef {
            id,
            aval: stale_aval,
            ctx,
        };

        let err = stale.unary_op(Primitive::Neg).unwrap_err();
        assert!(matches!(err, TraceError::TracerInvariantViolation { .. }));
    }

    // ── Trace-time validation evidence tests ─────────────────

    #[test]
    fn test_binary_op_dtype_mismatch_detection() {
        // When two tracers have different dtypes, binary op should still succeed
        // (type promotion happens at eval time), but shape inference should catch
        // incompatible shapes.
        use super::make_jaxpr;
        let aval_f64 = ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        };

        let result = make_jaxpr(
            |inputs| {
                let sum = inputs[0].binary_op(Primitive::Add, &inputs[1]).unwrap();
                vec![sum]
            },
            vec![aval_f64.clone(), aval_f64],
        );
        assert!(result.is_ok(), "same-shape binary op should succeed");
    }

    #[test]
    fn test_foreign_tracer_in_binary_op_rhs() {
        // Ensure foreign tracer detection works for RHS as well as LHS
        use super::{TraceError, TracerRef};
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        };

        let ctx_a = Rc::new(RefCell::new(SimpleTraceContext::new()));
        let id_a = ctx_a.borrow_mut().bind_input(aval.clone());
        let a = TracerRef {
            id: id_a,
            aval: aval.clone(),
            ctx: Rc::clone(&ctx_a),
        };

        let ctx_b = Rc::new(RefCell::new(SimpleTraceContext::new()));
        let id_b = ctx_b.borrow_mut().bind_input(aval.clone());
        let b = TracerRef {
            id: id_b,
            aval: aval.clone(),
            ctx: Rc::clone(&ctx_b),
        };

        // Both directions should fail
        assert!(matches!(
            a.binary_op(Primitive::Add, &b),
            Err(TraceError::ForeignTracerContext { .. })
        ));
        assert!(matches!(
            b.binary_op(Primitive::Mul, &a),
            Err(TraceError::ForeignTracerContext { .. })
        ));
    }

    #[test]
    fn test_binary_op_cached_aval_mismatch() {
        // Stale aval on either side of a binary op should be caught
        use super::{TraceError, TracerRef};
        let actual = ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        };
        let stale = ShapedArray {
            dtype: DType::I64,
            shape: Shape::scalar(),
        };

        let ctx = Rc::new(RefCell::new(SimpleTraceContext::new()));
        let id_good = ctx.borrow_mut().bind_input(actual.clone());
        let id_bad = ctx.borrow_mut().bind_input(actual);
        let good = TracerRef {
            id: id_good,
            aval: ShapedArray {
                dtype: DType::F64,
                shape: Shape::scalar(),
            },
            ctx: Rc::clone(&ctx),
        };
        let bad = TracerRef {
            id: id_bad,
            aval: stale,
            ctx: Rc::clone(&ctx),
        };

        let err = good.binary_op(Primitive::Add, &bad).unwrap_err();
        assert!(matches!(err, TraceError::TracerInvariantViolation { .. }));
    }

    #[test]
    fn test_make_jaxpr_identity_function() {
        // Identity function: output = input. Should produce empty equations.
        use super::make_jaxpr;
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        };

        let closed = make_jaxpr(|inputs| vec![inputs[0].clone()], vec![aval]).unwrap();
        assert!(
            closed.jaxpr.equations.is_empty(),
            "identity should have no equations"
        );
        assert_eq!(closed.jaxpr.invars.len(), 1);
        assert_eq!(closed.jaxpr.outvars.len(), 1);
    }

    #[test]
    fn test_make_jaxpr_multi_output_neg_abs() {
        // Function returning multiple outputs (neg and abs)
        use super::make_jaxpr;
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        };

        let closed = make_jaxpr(
            |inputs| {
                let neg = inputs[0].unary_op(Primitive::Neg).unwrap();
                let abs = inputs[0].unary_op(Primitive::Abs).unwrap();
                vec![neg, abs]
            },
            vec![aval],
        )
        .unwrap();
        assert_eq!(closed.jaxpr.equations.len(), 2);
        assert_eq!(closed.jaxpr.outvars.len(), 2);
    }

    #[test]
    fn test_make_jaxpr_chain_preserves_data_flow() {
        // Chain: x -> neg -> exp -> output. Verify equation connectivity.
        use super::make_jaxpr;
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        };

        let closed = make_jaxpr(
            |inputs| {
                let neg = inputs[0].unary_op(Primitive::Neg).unwrap();
                let exp = neg.unary_op(Primitive::Exp).unwrap();
                vec![exp]
            },
            vec![aval],
        )
        .unwrap();

        assert_eq!(closed.jaxpr.equations.len(), 2);
        assert_eq!(closed.jaxpr.equations[0].primitive, Primitive::Neg);
        assert_eq!(closed.jaxpr.equations[1].primitive, Primitive::Exp);
        // Second equation's input should be first equation's output
        let neg_output = closed.jaxpr.equations[0].outputs[0];
        let exp_input = match &closed.jaxpr.equations[1].inputs[0] {
            fj_core::Atom::Var(v) => *v,
            _ => panic!("expected var input"),
        };
        assert_eq!(neg_output, exp_input, "data flow should be connected");
    }

    #[test]
    fn test_make_jaxpr_vector_shape_inference() {
        // Trace with vector inputs, verify shape propagation
        use super::make_jaxpr;
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape { dims: vec![4] },
        };

        let closed = make_jaxpr(
            |inputs| {
                let neg = inputs[0].unary_op(Primitive::Neg).unwrap();
                vec![neg]
            },
            vec![aval],
        )
        .unwrap();
        assert_eq!(closed.jaxpr.equations.len(), 1);
    }

    #[test]
    fn test_make_jaxpr_fallible_wrong_output_count() {
        // make_jaxpr_fallible should reject wrong output count
        use super::make_jaxpr_fallible;
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        };

        // Return 2 outputs when traced with 1 input — should succeed (multi-output is valid)
        let result = make_jaxpr_fallible(
            |inputs| {
                let neg = inputs[0].unary_op(Primitive::Neg)?;
                let abs = inputs[0].unary_op(Primitive::Abs)?;
                Ok(vec![neg, abs])
            },
            vec![aval],
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_make_jaxpr_fallible_error_propagation() {
        // Errors inside the traced function should propagate
        use super::{TraceError, make_jaxpr_fallible};
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        };

        let result =
            make_jaxpr_fallible(|_inputs| Err(TraceError::InvalidAbstractValue), vec![aval]);
        assert!(matches!(result, Err(TraceError::InvalidAbstractValue)));
    }

    // ================================================================
    // Trace-time shape inference validation (frankenjax-cim)
    // ================================================================

    #[test]
    fn test_trace_multi_input_broadcasting() {
        // f(scalar, [3], [3]) = scalar * [3] + [3]
        use super::make_jaxpr;
        let s = ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        };
        let v = ShapedArray {
            dtype: DType::F64,
            shape: Shape { dims: vec![3] },
        };

        let closed = make_jaxpr(
            |inputs| {
                let scaled = &inputs[0] * &inputs[1]; // scalar * [3] → [3]
                let sum = &scaled + &inputs[2]; // [3] + [3] → [3]
                vec![sum]
            },
            vec![s, v.clone(), v],
        )
        .unwrap();

        assert_eq!(closed.jaxpr.invars.len(), 3);
        assert_eq!(closed.jaxpr.equations.len(), 2);
        assert_eq!(closed.jaxpr.equations[0].primitive, Primitive::Mul);
        assert_eq!(closed.jaxpr.equations[1].primitive, Primitive::Add);

        // Evaluate to verify correctness
        let result = fj_interpreters::eval_jaxpr(
            &closed.jaxpr,
            &[
                Value::scalar_f64(2.0),
                Value::vector_f64(&[1.0, 2.0, 3.0]).unwrap(),
                Value::vector_f64(&[10.0, 20.0, 30.0]).unwrap(),
            ],
        )
        .unwrap();
        let vals = result[0].as_tensor().unwrap().to_f64_vec().unwrap();
        assert_eq!(vals, vec![12.0, 24.0, 36.0]);
    }

    #[test]
    fn test_trace_unary_chain_shape_preservation() {
        // f(x) = exp(neg(sin(x))) preserves shape through unary chain
        use super::make_jaxpr;
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape { dims: vec![4] },
        };

        let closed = make_jaxpr(
            |inputs| {
                let s = inputs[0].unary_op(Primitive::Sin).unwrap();
                let n = s.unary_op(Primitive::Neg).unwrap();
                let e = n.unary_op(Primitive::Exp).unwrap();
                vec![e]
            },
            vec![aval],
        )
        .unwrap();

        assert_eq!(closed.jaxpr.equations.len(), 3);
        assert_eq!(closed.jaxpr.equations[0].primitive, Primitive::Sin);
        assert_eq!(closed.jaxpr.equations[1].primitive, Primitive::Neg);
        assert_eq!(closed.jaxpr.equations[2].primitive, Primitive::Exp);

        // Evaluate: exp(neg(sin([0, π/2, π, 3π/2])))
        let input = Value::vector_f64(&[
            0.0,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::PI,
            3.0 * std::f64::consts::FRAC_PI_2,
        ])
        .unwrap();
        let result = fj_interpreters::eval_jaxpr(&closed.jaxpr, &[input]).unwrap();
        let vals = result[0].as_tensor().unwrap().to_f64_vec().unwrap();
        assert_eq!(vals.len(), 4);
        // exp(-sin(0)) = exp(0) = 1
        assert!((vals[0] - 1.0).abs() < 1e-10);
        // exp(-sin(π/2)) = exp(-1) ≈ 0.3679
        assert!((vals[1] - (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_trace_reduction_changes_shape() {
        // f([3, 4]) -> reduce_sum(axis=1) -> [3]
        use super::make_jaxpr;
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape { dims: vec![3, 4] },
        };

        let closed = make_jaxpr(
            |inputs| {
                let mut params = BTreeMap::new();
                params.insert("axes".to_owned(), "1".to_owned());
                inputs[0]
                    .primitive_with_params(Primitive::ReduceSum, &[], params)
                    .unwrap()
            },
            vec![aval],
        )
        .unwrap();

        assert_eq!(closed.jaxpr.equations.len(), 1);
        assert_eq!(closed.jaxpr.equations[0].primitive, Primitive::ReduceSum);

        // Evaluate: reduce_sum([[1,2,3,4],[5,6,7,8],[9,10,11,12]], axis=1) = [10, 26, 42]
        let data: Vec<f64> = (1..=12).map(|i| i as f64).collect();
        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3, 4] },
                data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        );
        let result = fj_interpreters::eval_jaxpr(&closed.jaxpr, &[input]).unwrap();
        let vals = result[0].as_tensor().unwrap().to_f64_vec().unwrap();
        assert_eq!(vals, vec![10.0, 26.0, 42.0]);
    }

    #[test]
    fn test_ifft_rejects_real_input_shape_inference() {
        let mut ctx = SimpleTraceContext::with_inputs(vec![ShapedArray {
            dtype: DType::F64,
            shape: Shape::vector(8),
        }]);

        let err = ctx
            .process_primitive(Primitive::Ifft, &[TracerId(1)], BTreeMap::new())
            .expect_err("ifft should reject real-valued input during tracing");
        assert!(matches!(
            err,
            super::TraceError::ShapeInferenceFailed {
                primitive: Primitive::Ifft,
                ..
            }
        ));
        assert!(
            err.to_string().contains("complex-valued input"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_trace_mixed_dtype_promotion() {
        // f(i64, f64) = i64 + f64 should produce f64 output
        use super::make_jaxpr;
        let i64_aval = ShapedArray {
            dtype: DType::I64,
            shape: Shape::scalar(),
        };
        let f64_aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        };

        let closed = make_jaxpr(
            |inputs| vec![&inputs[0] + &inputs[1]],
            vec![i64_aval, f64_aval],
        )
        .unwrap();

        assert_eq!(closed.jaxpr.equations.len(), 1);
        // The traced equation should handle type promotion
        let result = fj_interpreters::eval_jaxpr(
            &closed.jaxpr,
            &[Value::scalar_i64(3), Value::scalar_f64(0.14)],
        )
        .unwrap();
        let val = result[0].as_f64_scalar().unwrap();
        #[allow(clippy::approx_constant)]
        let expected = 3.14;
        assert!((val - expected).abs() < 1e-10);
    }

    #[test]
    fn test_trace_diamond_dag() {
        // Diamond DAG: y = sin(x) + cos(x) — x feeds into two paths that merge
        use super::make_jaxpr;
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        };

        let closed = make_jaxpr(
            |inputs| {
                let s = inputs[0].unary_op(Primitive::Sin).unwrap();
                let c = inputs[0].unary_op(Primitive::Cos).unwrap();
                vec![&s + &c]
            },
            vec![aval],
        )
        .unwrap();

        // Should have 3 equations: sin, cos, add
        assert_eq!(closed.jaxpr.equations.len(), 3);
        // Input var used in both sin and cos equations
        assert_eq!(closed.jaxpr.invars.len(), 1);
        let x_var = closed.jaxpr.invars[0];
        // Both sin and cos should reference the same input
        assert!(closed.jaxpr.equations[0].inputs.contains(&Atom::Var(x_var)));
        assert!(closed.jaxpr.equations[1].inputs.contains(&Atom::Var(x_var)));

        // Evaluate: sin(π/4) + cos(π/4) = √2 ≈ 1.4142
        let x = std::f64::consts::FRAC_PI_4;
        let result = fj_interpreters::eval_jaxpr(&closed.jaxpr, &[Value::scalar_f64(x)]).unwrap();
        let val = result[0].as_f64_scalar().unwrap();
        assert!(
            (val - std::f64::consts::SQRT_2).abs() < 1e-10,
            "sin(π/4)+cos(π/4) should be √2, got {val}"
        );
    }

    #[test]
    fn test_trace_multi_output_program() {
        // f(x) returns (sin(x), cos(x)) — multi-output tracing
        use super::make_jaxpr;
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape::scalar(),
        };

        let closed = make_jaxpr(
            |inputs| {
                let s = inputs[0].unary_op(Primitive::Sin).unwrap();
                let c = inputs[0].unary_op(Primitive::Cos).unwrap();
                vec![s, c]
            },
            vec![aval],
        )
        .unwrap();

        assert_eq!(closed.jaxpr.outvars.len(), 2);
        assert_eq!(closed.jaxpr.equations.len(), 2);

        let result = fj_interpreters::eval_jaxpr(&closed.jaxpr, &[Value::scalar_f64(1.0)]).unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0].as_f64_scalar().unwrap() - 1.0_f64.sin()).abs() < 1e-10);
        assert!((result[1].as_f64_scalar().unwrap() - 1.0_f64.cos()).abs() < 1e-10);
    }

    #[test]
    fn test_trace_multiple_reductions() {
        // f([2,3]) returns (reduce_sum(axis=0), reduce_sum(axis=1))
        // axis=0: [3] shape, axis=1: [2] shape
        use super::make_jaxpr;
        let aval = ShapedArray {
            dtype: DType::F64,
            shape: Shape { dims: vec![2, 3] },
        };

        let closed = make_jaxpr(
            |inputs| {
                let mut params0 = BTreeMap::new();
                params0.insert("axes".to_owned(), "0".to_owned());
                let sum0 = inputs[0]
                    .primitive_with_params(Primitive::ReduceSum, &[], params0)
                    .unwrap();
                let mut params1 = BTreeMap::new();
                params1.insert("axes".to_owned(), "1".to_owned());
                let sum1 = inputs[0]
                    .primitive_with_params(Primitive::ReduceSum, &[], params1)
                    .unwrap();
                let mut out = sum0;
                out.extend(sum1);
                out
            },
            vec![aval],
        )
        .unwrap();

        assert_eq!(closed.jaxpr.outvars.len(), 2);
        assert_eq!(closed.jaxpr.equations.len(), 2);

        // [[1,2,3],[4,5,6]]: sum(axis=0) = [5,7,9], sum(axis=1) = [6,15]
        let data: Vec<f64> = (1..=6).map(|i| i as f64).collect();
        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![2, 3] },
                data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        );
        let result = fj_interpreters::eval_jaxpr(&closed.jaxpr, &[input]).unwrap();
        assert_eq!(result.len(), 2);
        let sum0 = result[0].as_tensor().unwrap().to_f64_vec().unwrap();
        let sum1 = result[1].as_tensor().unwrap().to_f64_vec().unwrap();
        assert_eq!(sum0, vec![5.0, 7.0, 9.0]);
        assert_eq!(sum1, vec![6.0, 15.0]);
    }
}
