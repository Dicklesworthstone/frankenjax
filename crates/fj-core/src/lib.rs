#![forbid(unsafe_code)]

#[cfg(test)]
pub mod proptest_strategies;

use serde::{Deserialize, Serialize};
use smallvec::{SmallVec, smallvec};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompatibilityMode {
    Strict,
    Hardened,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    Bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    pub dims: Vec<u32>,
}

impl Shape {
    #[must_use]
    pub fn scalar() -> Self {
        Self { dims: Vec::new() }
    }

    #[must_use]
    pub fn vector(len: u32) -> Self {
        Self { dims: vec![len] }
    }

    #[must_use]
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    #[must_use]
    pub fn element_count(&self) -> Option<u64> {
        self.dims
            .iter()
            .try_fold(1_u64, |acc, dim| acc.checked_mul(u64::from(*dim)))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbstractValue {
    pub dtype: DType,
    pub shape: Shape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Primitive {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Neg,
    Abs,
    Max,
    Min,
    Pow,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Floor,
    Ceil,
    Round,
    // Trigonometric
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    // Hyperbolic
    Sinh,
    Cosh,
    Tanh,
    // Additional math
    Expm1,
    Log1p,
    Sign,
    Square,
    Reciprocal,
    Logistic,
    Erf,
    Erfc,
    // Binary math
    Div,
    Rem,
    Atan2,
    // Selection
    Select,
    // Dot product
    Dot,
    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    // Reduction
    ReduceSum,
    ReduceMax,
    ReduceMin,
    ReduceProd,
    // Shape manipulation
    Reshape,
    Slice,
    DynamicSlice,
    Gather,
    Scatter,
    Transpose,
    BroadcastInDim,
    Concatenate,
    // Clamping
    Clamp,
    // Index generation
    Iota,
}

impl Primitive {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Neg => "neg",
            Self::Abs => "abs",
            Self::Max => "max",
            Self::Min => "min",
            Self::Pow => "pow",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sqrt => "sqrt",
            Self::Rsqrt => "rsqrt",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Round => "round",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tanh => "tanh",
            Self::Expm1 => "expm1",
            Self::Log1p => "log1p",
            Self::Sign => "sign",
            Self::Square => "square",
            Self::Reciprocal => "reciprocal",
            Self::Logistic => "logistic",
            Self::Erf => "erf",
            Self::Erfc => "erfc",
            Self::Div => "div",
            Self::Rem => "rem",
            Self::Atan2 => "atan2",
            Self::Select => "select",
            Self::Dot => "dot",
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
            Self::ReduceSum => "reduce_sum",
            Self::ReduceMax => "reduce_max",
            Self::ReduceMin => "reduce_min",
            Self::ReduceProd => "reduce_prod",
            Self::Reshape => "reshape",
            Self::Slice => "slice",
            Self::DynamicSlice => "dynamic_slice",
            Self::Gather => "gather",
            Self::Scatter => "scatter",
            Self::Transpose => "transpose",
            Self::BroadcastInDim => "broadcast_in_dim",
            Self::Concatenate => "concatenate",
            Self::Clamp => "clamp",
            Self::Iota => "iota",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Transform {
    Jit,
    Grad,
    Vmap,
}

impl Transform {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Jit => "jit",
            Self::Grad => "grad",
            Self::Vmap => "vmap",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VarId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Literal {
    I64(i64),
    Bool(bool),
    F64Bits(u64),
}

impl Literal {
    #[must_use]
    pub fn from_f64(value: f64) -> Self {
        Self::F64Bits(value.to_bits())
    }

    #[must_use]
    pub fn as_f64(self) -> Option<f64> {
        match self {
            Self::F64Bits(bits) => Some(f64::from_bits(bits)),
            Self::I64(value) => Some(value as f64),
            Self::Bool(_) => None,
        }
    }

    #[must_use]
    pub fn as_i64(self) -> Option<i64> {
        match self {
            Self::I64(value) => Some(value),
            Self::Bool(_) | Self::F64Bits(_) => None,
        }
    }

    #[must_use]
    pub fn is_integral(self) -> bool {
        matches!(self, Self::I64(_))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Value {
    Scalar(Literal),
    Tensor(TensorValue),
}

impl Value {
    #[must_use]
    pub fn scalar_i64(value: i64) -> Self {
        Self::Scalar(Literal::I64(value))
    }

    #[must_use]
    pub fn scalar_f64(value: f64) -> Self {
        Self::Scalar(Literal::from_f64(value))
    }

    #[must_use]
    pub fn scalar_bool(value: bool) -> Self {
        Self::Scalar(Literal::Bool(value))
    }

    pub fn vector_i64(values: &[i64]) -> Result<Self, ValueError> {
        let elements = values.iter().copied().map(Literal::I64).collect::<Vec<_>>();
        Ok(Self::Tensor(TensorValue::new(
            DType::I64,
            Shape::vector(values.len() as u32),
            elements,
        )?))
    }

    pub fn vector_f64(values: &[f64]) -> Result<Self, ValueError> {
        let elements = values
            .iter()
            .copied()
            .map(Literal::from_f64)
            .collect::<Vec<_>>();
        Ok(Self::Tensor(TensorValue::new(
            DType::F64,
            Shape::vector(values.len() as u32),
            elements,
        )?))
    }

    #[must_use]
    pub fn as_scalar_literal(&self) -> Option<Literal> {
        match self {
            Self::Scalar(lit) => Some(*lit),
            Self::Tensor(_) => None,
        }
    }

    #[must_use]
    pub fn as_f64_scalar(&self) -> Option<f64> {
        self.as_scalar_literal().and_then(Literal::as_f64)
    }

    #[must_use]
    pub fn as_i64_scalar(&self) -> Option<i64> {
        self.as_scalar_literal().and_then(Literal::as_i64)
    }

    #[must_use]
    pub fn as_bool_scalar(&self) -> Option<bool> {
        match self.as_scalar_literal() {
            Some(Literal::Bool(b)) => Some(b),
            _ => None,
        }
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        match self {
            Self::Scalar(lit) => match lit {
                Literal::I64(_) => DType::I64,
                Literal::Bool(_) => DType::Bool,
                Literal::F64Bits(_) => DType::F64,
            },
            Self::Tensor(t) => t.dtype,
        }
    }

    #[must_use]
    pub fn as_tensor(&self) -> Option<&TensorValue> {
        match self {
            Self::Scalar(_) => None,
            Self::Tensor(tensor) => Some(tensor),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorValue {
    pub dtype: DType,
    pub shape: Shape,
    pub elements: Vec<Literal>,
}

impl TensorValue {
    pub fn new(dtype: DType, shape: Shape, elements: Vec<Literal>) -> Result<Self, ValueError> {
        let expected_count = shape.element_count().ok_or(ValueError::ShapeOverflow {
            shape: shape.clone(),
        })?;

        if expected_count != elements.len() as u64 {
            return Err(ValueError::ElementCountMismatch {
                shape,
                expected_count,
                actual_count: elements.len(),
            });
        }

        Ok(Self {
            dtype,
            shape,
            elements,
        })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    #[must_use]
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    #[must_use]
    pub fn leading_dim(&self) -> Option<u32> {
        self.shape.dims.first().copied()
    }

    pub fn slice_axis0(&self, index: usize) -> Result<Value, ValueError> {
        let axis_size = self
            .leading_dim()
            .ok_or(ValueError::RankZeroAxisSliceUnsupported)?;
        if index >= axis_size as usize {
            return Err(ValueError::SliceIndexOutOfBounds {
                index,
                axis_size: axis_size as usize,
            });
        }

        if self.rank() == 1 {
            return Ok(Value::Scalar(self.elements[index]));
        }

        let slice_len = self
            .shape
            .dims
            .iter()
            .skip(1)
            .try_fold(1_usize, |acc, dim| acc.checked_mul(*dim as usize))
            .ok_or(ValueError::ShapeOverflow {
                shape: self.shape.clone(),
            })?;

        let start = index
            .checked_mul(slice_len)
            .ok_or(ValueError::ShapeOverflow {
                shape: self.shape.clone(),
            })?;
        let end = start
            .checked_add(slice_len)
            .ok_or(ValueError::ShapeOverflow {
                shape: self.shape.clone(),
            })?;
        let elements = self.elements[start..end].to_vec();
        let subshape = Shape {
            dims: self.shape.dims[1..].to_vec(),
        };
        Ok(Value::Tensor(TensorValue::new(
            self.dtype, subshape, elements,
        )?))
    }

    pub fn stack_axis0(slices: &[Value]) -> Result<Self, ValueError> {
        if slices.is_empty() {
            return Err(ValueError::EmptyAxisStack);
        }

        match &slices[0] {
            Value::Scalar(first) => {
                let mut elements = Vec::with_capacity(slices.len());
                elements.push(*first);
                for value in &slices[1..] {
                    let Value::Scalar(lit) = value else {
                        return Err(ValueError::MixedAxisStackKinds);
                    };
                    elements.push(*lit);
                }
                let dtype = infer_dtype_from_literals(&elements);
                TensorValue::new(dtype, Shape::vector(slices.len() as u32), elements)
            }
            Value::Tensor(first) => {
                let mut elements = Vec::with_capacity(first.elements.len() * slices.len());
                elements.extend_from_slice(&first.elements);
                for value in &slices[1..] {
                    let Value::Tensor(tensor) = value else {
                        return Err(ValueError::MixedAxisStackKinds);
                    };
                    if tensor.dtype != first.dtype {
                        return Err(ValueError::AxisStackDTypeMismatch {
                            expected: first.dtype,
                            actual: tensor.dtype,
                        });
                    }
                    if tensor.shape != first.shape {
                        return Err(ValueError::AxisStackShapeMismatch {
                            expected: first.shape.clone(),
                            actual: tensor.shape.clone(),
                        });
                    }
                    elements.extend_from_slice(&tensor.elements);
                }

                let mut dims = Vec::with_capacity(first.shape.rank() + 1);
                dims.push(slices.len() as u32);
                dims.extend_from_slice(&first.shape.dims);
                TensorValue::new(first.dtype, Shape { dims }, elements)
            }
        }
    }

    pub fn to_f64_vec(&self) -> Option<Vec<f64>> {
        self.elements.iter().copied().map(Literal::as_f64).collect()
    }

    pub fn to_i64_vec(&self) -> Option<Vec<i64>> {
        self.elements.iter().copied().map(Literal::as_i64).collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueError {
    ShapeOverflow {
        shape: Shape,
    },
    ElementCountMismatch {
        shape: Shape,
        expected_count: u64,
        actual_count: usize,
    },
    RankZeroAxisSliceUnsupported,
    SliceIndexOutOfBounds {
        index: usize,
        axis_size: usize,
    },
    EmptyAxisStack,
    MixedAxisStackKinds,
    AxisStackShapeMismatch {
        expected: Shape,
        actual: Shape,
    },
    AxisStackDTypeMismatch {
        expected: DType,
        actual: DType,
    },
}

impl std::fmt::Display for ValueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeOverflow { shape } => {
                write!(f, "shape element count overflowed: {:?}", shape.dims)
            }
            Self::ElementCountMismatch {
                shape,
                expected_count,
                actual_count,
            } => {
                write!(
                    f,
                    "tensor element count mismatch for shape {:?}: expected {}, got {}",
                    shape.dims, expected_count, actual_count
                )
            }
            Self::RankZeroAxisSliceUnsupported => {
                write!(f, "cannot axis-slice rank-0 scalar tensor")
            }
            Self::SliceIndexOutOfBounds { index, axis_size } => {
                write!(
                    f,
                    "axis-slice index {} out of bounds for axis size {}",
                    index, axis_size
                )
            }
            Self::EmptyAxisStack => {
                write!(f, "cannot stack empty slice list")
            }
            Self::MixedAxisStackKinds => {
                write!(f, "cannot stack mixed scalar/tensor slice kinds")
            }
            Self::AxisStackShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "stack shape mismatch: expected {:?}, got {:?}",
                    expected.dims, actual.dims
                )
            }
            Self::AxisStackDTypeMismatch { expected, actual } => {
                write!(
                    f,
                    "stack dtype mismatch: expected {:?}, got {:?}",
                    expected, actual
                )
            }
        }
    }
}

impl std::error::Error for ValueError {}

fn infer_dtype_from_literals(elements: &[Literal]) -> DType {
    if elements.is_empty() {
        return DType::F64;
    }
    if elements
        .iter()
        .all(|literal| matches!(literal, Literal::I64(_)))
    {
        DType::I64
    } else if elements
        .iter()
        .all(|literal| matches!(literal, Literal::Bool(_)))
    {
        DType::Bool
    } else {
        DType::F64
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Atom {
    Var(VarId),
    Lit(Literal),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Equation {
    pub primitive: Primitive,
    pub inputs: SmallVec<[Atom; 4]>,
    pub outputs: SmallVec<[VarId; 2]>,
    pub params: BTreeMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Jaxpr {
    pub invars: Vec<VarId>,
    pub constvars: Vec<VarId>,
    pub outvars: Vec<VarId>,
    pub equations: Vec<Equation>,
    #[serde(skip)]
    fingerprint_cache: std::sync::OnceLock<String>,
}

impl Clone for Jaxpr {
    fn clone(&self) -> Self {
        Self {
            invars: self.invars.clone(),
            constvars: self.constvars.clone(),
            outvars: self.outvars.clone(),
            equations: self.equations.clone(),
            fingerprint_cache: std::sync::OnceLock::new(),
        }
    }
}

impl PartialEq for Jaxpr {
    fn eq(&self, other: &Self) -> bool {
        self.invars == other.invars
            && self.constvars == other.constvars
            && self.outvars == other.outvars
            && self.equations == other.equations
    }
}

impl Eq for Jaxpr {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JaxprValidationError {
    DuplicateBinding { section: &'static str, var: VarId },
    UnboundInputVar { equation_index: usize, var: VarId },
    OutputShadowsBinding { equation_index: usize, var: VarId },
    UnknownOutvar { var: VarId },
}

impl std::fmt::Display for JaxprValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateBinding { section, var } => {
                write!(f, "duplicate binding in {} for var v{}", section, var.0)
            }
            Self::UnboundInputVar {
                equation_index,
                var,
            } => {
                write!(
                    f,
                    "equation {} references unbound input var v{}",
                    equation_index, var.0
                )
            }
            Self::OutputShadowsBinding {
                equation_index,
                var,
            } => {
                write!(
                    f,
                    "equation {} output var v{} shadows an existing binding",
                    equation_index, var.0
                )
            }
            Self::UnknownOutvar { var } => {
                write!(f, "outvar v{} does not have a defining binding", var.0)
            }
        }
    }
}

impl std::error::Error for JaxprValidationError {}

impl Jaxpr {
    #[must_use]
    pub fn new(
        invars: Vec<VarId>,
        constvars: Vec<VarId>,
        outvars: Vec<VarId>,
        equations: Vec<Equation>,
    ) -> Self {
        Self {
            invars,
            constvars,
            outvars,
            equations,
            fingerprint_cache: std::sync::OnceLock::new(),
        }
    }

    #[must_use]
    pub fn canonical_fingerprint(&self) -> &str {
        self.fingerprint_cache.get_or_init(|| {
            let mut out = String::new();
            write_var_list(&mut out, "in", &self.invars);
            write_var_list(&mut out, "const", &self.constvars);
            write_var_list(&mut out, "out", &self.outvars);

            for eqn in &self.equations {
                let _ = write!(&mut out, "eqn:{}(", eqn.primitive.as_str());
                for atom in &eqn.inputs {
                    write_atom(&mut out, atom);
                    out.push(',');
                }
                out.push(')');
                out.push_str("->");
                for outvar in &eqn.outputs {
                    let _ = write!(&mut out, "v{},", outvar.0);
                }
                out.push('{');
                for (key, value) in &eqn.params {
                    let _ = write!(&mut out, "{key}={value};");
                }
                out.push('}');
                out.push('|');
            }

            out
        })
    }

    pub fn validate_well_formed(&self) -> Result<(), JaxprValidationError> {
        let mut bindings = BTreeSet::new();

        for var in &self.invars {
            if !bindings.insert(*var) {
                return Err(JaxprValidationError::DuplicateBinding {
                    section: "invars",
                    var: *var,
                });
            }
        }
        for var in &self.constvars {
            if !bindings.insert(*var) {
                return Err(JaxprValidationError::DuplicateBinding {
                    section: "constvars",
                    var: *var,
                });
            }
        }

        for (equation_index, eqn) in self.equations.iter().enumerate() {
            for atom in &eqn.inputs {
                if let Atom::Var(var) = atom
                    && !bindings.contains(var)
                {
                    return Err(JaxprValidationError::UnboundInputVar {
                        equation_index,
                        var: *var,
                    });
                }
            }
            for out_var in &eqn.outputs {
                if !bindings.insert(*out_var) {
                    return Err(JaxprValidationError::OutputShadowsBinding {
                        equation_index,
                        var: *out_var,
                    });
                }
            }
        }

        let mut seen_outvars = BTreeSet::new();
        for outvar in &self.outvars {
            if !seen_outvars.insert(*outvar) {
                return Err(JaxprValidationError::DuplicateBinding {
                    section: "outvars",
                    var: *outvar,
                });
            }
            if !bindings.contains(outvar) {
                return Err(JaxprValidationError::UnknownOutvar { var: *outvar });
            }
        }

        Ok(())
    }
}

fn write_var_list(out: &mut String, label: &str, vars: &[VarId]) {
    let _ = write!(out, "{label}=[");
    for var in vars {
        let _ = write!(out, "v{},", var.0);
    }
    out.push(']');
}

fn write_atom(out: &mut String, atom: &Atom) {
    match atom {
        Atom::Var(var) => {
            let _ = write!(out, "v{}", var.0);
        }
        Atom::Lit(lit) => write_literal(out, *lit),
    }
}

fn write_literal(out: &mut String, lit: Literal) {
    match lit {
        Literal::I64(value) => {
            let _ = write!(out, "i64:{value}");
        }
        Literal::Bool(value) => {
            let _ = write!(out, "bool:{value}");
        }
        Literal::F64Bits(value) => {
            let _ = write!(out, "f64bits:{value}");
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgramSpec {
    Add2,
    Square,
    SquarePlusLinear,
    AddOne,
    SinX,
    CosX,
    Dot3,
    ReduceSumVec,
}

#[must_use]
pub fn build_program(spec: ProgramSpec) -> Jaxpr {
    match spec {
        ProgramSpec::Add2 => Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3)],
            vec![Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
            }],
        ),
        ProgramSpec::Square => Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Mul,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        ),
        ProgramSpec::SquarePlusLinear => Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(4)],
            vec![
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(2))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                },
            ],
        ),
        ProgramSpec::AddOne => Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Add,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        ),
        ProgramSpec::SinX => Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Sin,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        ),
        ProgramSpec::CosX => Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Cos,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        ),
        ProgramSpec::Dot3 => Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3)],
            vec![Equation {
                primitive: Primitive::Dot,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
            }],
        ),
        ProgramSpec::ReduceSumVec => Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::ReduceSum,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: BTreeMap::new(),
            }],
        ),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceTransformLedger {
    pub root_jaxpr: Jaxpr,
    pub transform_stack: Vec<Transform>,
    pub transform_evidence: Vec<String>,
}

impl TraceTransformLedger {
    #[must_use]
    pub fn new(root_jaxpr: Jaxpr) -> Self {
        Self {
            root_jaxpr,
            transform_stack: Vec::new(),
            transform_evidence: Vec::new(),
        }
    }

    pub fn push_transform(&mut self, transform: Transform, evidence_id: impl Into<String>) {
        self.transform_stack.push(transform);
        self.transform_evidence.push(evidence_id.into());
    }

    #[must_use]
    pub fn composition_signature(&self) -> String {
        let mut out = String::new();
        out.push_str("stack=");
        for transform in &self.transform_stack {
            let _ = write!(&mut out, "{}>", transform.as_str());
        }
        out.push_str("|jaxpr=");
        out.push_str(self.root_jaxpr.canonical_fingerprint());
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformCompositionProof {
    pub stack_signature: String,
    pub stack_hash_hex: String,
    pub transform_count: usize,
    pub evidence_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformCompositionError {
    EvidenceCountMismatch {
        transform_count: usize,
        evidence_count: usize,
    },
    EmptyEvidence {
        index: usize,
        transform: Transform,
    },
}

impl std::fmt::Display for TransformCompositionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EvidenceCountMismatch {
                transform_count,
                evidence_count,
            } => {
                write!(
                    f,
                    "transform/evidence cardinality mismatch: transforms={}, evidence={}",
                    transform_count, evidence_count
                )
            }
            Self::EmptyEvidence { index, transform } => {
                write!(
                    f,
                    "transform evidence at index {} for {} is empty",
                    index,
                    transform.as_str()
                )
            }
        }
    }
}

impl std::error::Error for TransformCompositionError {}

pub fn verify_transform_composition(
    ledger: &TraceTransformLedger,
) -> Result<TransformCompositionProof, TransformCompositionError> {
    if ledger.transform_stack.len() != ledger.transform_evidence.len() {
        return Err(TransformCompositionError::EvidenceCountMismatch {
            transform_count: ledger.transform_stack.len(),
            evidence_count: ledger.transform_evidence.len(),
        });
    }

    for (index, transform) in ledger.transform_stack.iter().enumerate() {
        if ledger.transform_evidence[index].trim().is_empty() {
            return Err(TransformCompositionError::EmptyEvidence {
                index,
                transform: *transform,
            });
        }
    }

    let stack_signature = ledger.composition_signature();
    let stack_hash_hex = format!("{:016x}", fnv1a_64(stack_signature.as_bytes()));

    Ok(TransformCompositionProof {
        stack_signature,
        stack_hash_hex,
        transform_count: ledger.transform_stack.len(),
        evidence_count: ledger.transform_evidence.len(),
    })
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::{
        Atom, DType, Equation, Jaxpr, JaxprValidationError, Literal, Primitive, ProgramSpec, Shape,
        TensorValue, TraceTransformLedger, Transform, Value, ValueError, VarId, build_program,
        verify_transform_composition,
    };
    use proptest::prelude::*;
    use proptest::test_runner::{Config as ProptestConfig, TestCaseError, TestRunner};
    use serde::Serialize;
    use smallvec::smallvec;
    use std::any::Any;
    use std::collections::BTreeMap;
    use std::fs;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    const PACKET_ID: &str = "FJ-P2C-001";
    const SUITE_ID: &str = "fj-core";

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
        format!("cargo test -p fj-core --lib {test_id} -- --exact --nocapture")
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

    fn run_logged_test<Fixture, F>(
        test_name: &str,
        fixture: &Fixture,
        mode: fj_test_utils::TestMode,
        body: F,
    ) where
        Fixture: Serialize,
        F: FnOnce() -> Result<Vec<String>, String> + std::panic::UnwindSafe,
    {
        let overall_start = Instant::now();
        let setup_start = Instant::now();
        let fixture_id = fj_test_utils::fixture_id_from_json(fixture).expect("fixture digest");
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

    fn all_primitives_jaxpr() -> Jaxpr {
        let mut equations = Vec::new();

        let mut reshape_params = BTreeMap::new();
        reshape_params.insert("new_shape".to_owned(), "2,3".to_owned());
        equations.push(Equation {
            primitive: Primitive::Reshape,
            inputs: smallvec![Atom::Var(VarId(1))],
            outputs: smallvec![VarId(5)],
            params: reshape_params,
        });

        let mut slice_params = BTreeMap::new();
        slice_params.insert("start_indices".to_owned(), "0,0".to_owned());
        slice_params.insert("limit_indices".to_owned(), "2,2".to_owned());
        equations.push(Equation {
            primitive: Primitive::Slice,
            inputs: smallvec![Atom::Var(VarId(5))],
            outputs: smallvec![VarId(6)],
            params: slice_params,
        });

        let mut gather_params = BTreeMap::new();
        gather_params.insert("slice_sizes".to_owned(), "1,2".to_owned());
        equations.push(Equation {
            primitive: Primitive::Gather,
            inputs: smallvec![Atom::Var(VarId(5)), Atom::Var(VarId(2))],
            outputs: smallvec![VarId(7)],
            params: gather_params,
        });

        let mut transpose_params = BTreeMap::new();
        transpose_params.insert("permutation".to_owned(), "1,0".to_owned());
        equations.push(Equation {
            primitive: Primitive::Transpose,
            inputs: smallvec![Atom::Var(VarId(5))],
            outputs: smallvec![VarId(8)],
            params: transpose_params,
        });

        let mut broadcast_params = BTreeMap::new();
        broadcast_params.insert("shape".to_owned(), "3,2,2".to_owned());
        broadcast_params.insert("broadcast_dimensions".to_owned(), "1,2".to_owned());
        equations.push(Equation {
            primitive: Primitive::BroadcastInDim,
            inputs: smallvec![Atom::Var(VarId(6))],
            outputs: smallvec![VarId(9)],
            params: broadcast_params,
        });

        let mut concat_params = BTreeMap::new();
        concat_params.insert("dimension".to_owned(), "0".to_owned());
        equations.push(Equation {
            primitive: Primitive::Concatenate,
            inputs: smallvec![Atom::Var(VarId(8)), Atom::Var(VarId(8))],
            outputs: smallvec![VarId(10)],
            params: concat_params,
        });

        equations.push(Equation {
            primitive: Primitive::Scatter,
            inputs: smallvec![Atom::Var(VarId(10)), Atom::Var(VarId(2))],
            outputs: smallvec![VarId(11)],
            params: BTreeMap::new(),
        });

        let mut reduce_params = BTreeMap::new();
        reduce_params.insert("axes".to_owned(), "0".to_owned());
        equations.push(Equation {
            primitive: Primitive::ReduceSum,
            inputs: smallvec![Atom::Var(VarId(11))],
            outputs: smallvec![VarId(12)],
            params: reduce_params,
        });

        equations.push(Equation {
            primitive: Primitive::Sin,
            inputs: smallvec![Atom::Var(VarId(12))],
            outputs: smallvec![VarId(13)],
            params: BTreeMap::new(),
        });
        equations.push(Equation {
            primitive: Primitive::Cos,
            inputs: smallvec![Atom::Var(VarId(13))],
            outputs: smallvec![VarId(14)],
            params: BTreeMap::new(),
        });
        equations.push(Equation {
            primitive: Primitive::Add,
            inputs: smallvec![Atom::Var(VarId(14)), Atom::Var(VarId(12))],
            outputs: smallvec![VarId(15)],
            params: BTreeMap::new(),
        });
        equations.push(Equation {
            primitive: Primitive::Mul,
            inputs: smallvec![Atom::Var(VarId(15)), Atom::Var(VarId(15))],
            outputs: smallvec![VarId(16)],
            params: BTreeMap::new(),
        });
        equations.push(Equation {
            primitive: Primitive::Dot,
            inputs: smallvec![Atom::Var(VarId(8)), Atom::Var(VarId(3))],
            outputs: smallvec![VarId(17)],
            params: BTreeMap::new(),
        });
        equations.push(Equation {
            primitive: Primitive::Add,
            inputs: smallvec![Atom::Var(VarId(16)), Atom::Var(VarId(17))],
            outputs: smallvec![VarId(24)],
            params: BTreeMap::new(),
        });

        Jaxpr::new(
            vec![VarId(1), VarId(2), VarId(3)],
            vec![VarId(4)],
            vec![VarId(24)],
            equations,
        )
    }

    #[test]
    fn jaxpr_construction_supports_all_primitives() {
        run_logged_test(
            "jaxpr_construction_supports_all_primitives",
            &("all_primitives", 14_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = all_primitives_jaxpr();
                jaxpr
                    .validate_well_formed()
                    .map_err(|err| format!("well-formed validation failed: {err}"))?;
                let covered = jaxpr
                    .equations
                    .iter()
                    .map(|eqn| eqn.primitive)
                    .collect::<std::collections::BTreeSet<_>>();
                let expected = [
                    Primitive::Add,
                    Primitive::Mul,
                    Primitive::Dot,
                    Primitive::Sin,
                    Primitive::Cos,
                    Primitive::ReduceSum,
                    Primitive::Reshape,
                    Primitive::Slice,
                    Primitive::Gather,
                    Primitive::Scatter,
                    Primitive::Transpose,
                    Primitive::BroadcastInDim,
                    Primitive::Concatenate,
                ]
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>();
                assert_eq!(covered, expected);
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn jaxpr_constvars_binding_is_valid() {
        run_logged_test(
            "jaxpr_constvars_binding_is_valid",
            &("constvars", 1_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = Jaxpr::new(
                    vec![VarId(1)],
                    vec![VarId(2)],
                    vec![VarId(3)],
                    vec![Equation {
                        primitive: Primitive::Add,
                        inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                        outputs: smallvec![VarId(3)],
                        params: BTreeMap::new(),
                    }],
                );
                jaxpr
                    .validate_well_formed()
                    .map_err(|err| format!("constvar jaxpr should be valid: {err}"))?;
                assert!(jaxpr.canonical_fingerprint().contains("const=[v2,]"));
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn jaxpr_multi_equation_chain_is_valid() {
        run_logged_test(
            "jaxpr_multi_equation_chain_is_valid",
            &("square_plus_linear", 3_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = build_program(ProgramSpec::SquarePlusLinear);
                assert_eq!(jaxpr.equations.len(), 3);
                jaxpr
                    .validate_well_formed()
                    .map_err(|err| format!("program should validate: {err}"))?;
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn jaxpr_empty_program_is_valid() {
        run_logged_test(
            "jaxpr_empty_program_is_valid",
            &("empty", 0_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = Jaxpr::new(vec![], vec![], vec![], vec![]);
                jaxpr
                    .validate_well_formed()
                    .map_err(|err| format!("empty jaxpr should validate: {err}"))?;
                assert_eq!(jaxpr.equations.len(), 0);
                assert!(
                    jaxpr
                        .canonical_fingerprint()
                        .starts_with("in=[]const=[]out=[]")
                );
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn jaxpr_duplicate_varid_detection() {
        run_logged_test(
            "jaxpr_duplicate_varid_detection",
            &("duplicate-varid", 1_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let bad = Jaxpr::new(vec![VarId(1), VarId(1)], vec![], vec![VarId(1)], vec![]);
                let err = bad
                    .validate_well_formed()
                    .expect_err("duplicate invar should fail");
                assert!(matches!(
                    err,
                    JaxprValidationError::DuplicateBinding {
                        section: "invars",
                        var: VarId(1)
                    }
                ));
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn canonical_fingerprint_is_deterministic() {
        run_logged_test(
            "canonical_fingerprint_is_deterministic",
            &("fingerprint-deterministic", 1_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = build_program(ProgramSpec::SquarePlusLinear);
                let fp_a = jaxpr.canonical_fingerprint().to_owned();
                let fp_b = jaxpr.canonical_fingerprint().to_owned();
                assert_eq!(fp_a, fp_b);
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn canonical_fingerprint_changes_on_program_change() {
        run_logged_test(
            "canonical_fingerprint_changes_on_program_change",
            &("fingerprint-sensitivity", 2_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let base = build_program(ProgramSpec::SquarePlusLinear);
                let mut modified = base.clone();
                modified.equations[0]
                    .params
                    .insert("tweak".to_owned(), "1".to_owned());
                assert_ne!(
                    base.canonical_fingerprint(),
                    modified.canonical_fingerprint()
                );
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn canonical_fingerprint_round_trip_stable() {
        run_logged_test(
            "canonical_fingerprint_round_trip_stable",
            &("fingerprint-roundtrip", 1_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = build_program(ProgramSpec::SquarePlusLinear);
                let encoded = serde_json::to_string(&jaxpr)
                    .map_err(|err| format!("serialize failed: {err}"))?;
                let decoded: Jaxpr = serde_json::from_str(&encoded)
                    .map_err(|err| format!("decode failed: {err}"))?;
                decoded
                    .validate_well_formed()
                    .map_err(|err| format!("decoded jaxpr should validate: {err}"))?;
                assert_eq!(
                    jaxpr.canonical_fingerprint(),
                    decoded.canonical_fingerprint()
                );
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn transform_composition_valid_single_transforms() {
        run_logged_test(
            "transform_composition_valid_single_transforms",
            &["jit", "grad", "vmap"],
            fj_test_utils::TestMode::Strict,
            || {
                for transform in [Transform::Jit, Transform::Grad, Transform::Vmap] {
                    let mut ttl = TraceTransformLedger::new(build_program(ProgramSpec::Square));
                    ttl.push_transform(transform, format!("evidence-{}", transform.as_str()));
                    let proof = verify_transform_composition(&ttl)
                        .map_err(|err| format!("single transform should validate: {err}"))?;
                    assert_eq!(proof.transform_count, 1);
                    assert_eq!(proof.evidence_count, 1);
                }
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn transform_composition_valid_double_compositions() {
        run_logged_test(
            "transform_composition_valid_double_compositions",
            &[
                "jit+grad",
                "jit+vmap",
                "vmap+grad",
                "grad+grad",
                "vmap+vmap",
            ],
            fj_test_utils::TestMode::Strict,
            || {
                let stacks = [
                    [Transform::Jit, Transform::Grad],
                    [Transform::Jit, Transform::Vmap],
                    [Transform::Vmap, Transform::Grad],
                    [Transform::Grad, Transform::Grad],
                    [Transform::Vmap, Transform::Vmap],
                ];
                for stack in stacks {
                    let mut ttl = TraceTransformLedger::new(build_program(ProgramSpec::Square));
                    ttl.push_transform(stack[0], "a");
                    ttl.push_transform(stack[1], "b");
                    verify_transform_composition(&ttl)
                        .map_err(|err| format!("double transform should validate: {err}"))?;
                }
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn transform_composition_valid_triple_stack() {
        run_logged_test(
            "transform_composition_valid_triple_stack",
            &["jit", "vmap", "grad"],
            fj_test_utils::TestMode::Strict,
            || {
                let mut ttl = TraceTransformLedger::new(build_program(ProgramSpec::Square));
                ttl.push_transform(Transform::Jit, "jit");
                ttl.push_transform(Transform::Vmap, "vmap");
                ttl.push_transform(Transform::Grad, "grad");
                let proof = verify_transform_composition(&ttl)
                    .map_err(|err| format!("triple transform should validate: {err}"))?;
                assert_eq!(proof.transform_count, 3);
                assert_eq!(proof.evidence_count, 3);
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn transform_composition_allows_double_grad() {
        run_logged_test(
            "transform_composition_allows_double_grad",
            &["grad", "grad"],
            fj_test_utils::TestMode::Strict,
            || {
                let mut ttl = TraceTransformLedger::new(build_program(ProgramSpec::Square));
                ttl.push_transform(Transform::Grad, "g1");
                ttl.push_transform(Transform::Grad, "g2");
                let proof =
                    verify_transform_composition(&ttl).expect("double grad should validate");
                assert_eq!(proof.transform_count, 2);
                assert_eq!(proof.evidence_count, 2);
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn transform_composition_allows_double_vmap() {
        run_logged_test(
            "transform_composition_allows_double_vmap",
            &["vmap", "vmap"],
            fj_test_utils::TestMode::Strict,
            || {
                let mut ttl = TraceTransformLedger::new(build_program(ProgramSpec::Square));
                ttl.push_transform(Transform::Vmap, "v1");
                ttl.push_transform(Transform::Vmap, "v2");
                let proof =
                    verify_transform_composition(&ttl).expect("double vmap should validate");
                assert_eq!(proof.transform_count, 2);
                assert_eq!(proof.evidence_count, 2);
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn transform_composition_rejects_evidence_count_mismatch() {
        run_logged_test(
            "transform_composition_rejects_evidence_count_mismatch",
            &("mismatch", 1_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let ttl = TraceTransformLedger {
                    root_jaxpr: build_program(ProgramSpec::Square),
                    transform_stack: vec![Transform::Jit],
                    transform_evidence: vec![],
                };
                let err = verify_transform_composition(&ttl)
                    .expect_err("evidence mismatch should reject");
                assert!(format!("{err}").contains("cardinality mismatch"));
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn transform_composition_rejects_empty_evidence() {
        run_logged_test(
            "transform_composition_rejects_empty_evidence",
            &("empty-evidence", 1_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ttl = TraceTransformLedger::new(build_program(ProgramSpec::Square));
                ttl.push_transform(Transform::Jit, "   ");
                let err =
                    verify_transform_composition(&ttl).expect_err("empty evidence should fail");
                assert!(format!("{err}").contains("is empty"));
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn ttl_construction_with_empty_stack_is_deterministic() {
        run_logged_test(
            "ttl_construction_with_empty_stack_is_deterministic",
            &("ttl-empty-stack", 0_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let ttl = TraceTransformLedger::new(build_program(ProgramSpec::Add2));
                assert!(ttl.transform_stack.is_empty());
                assert!(ttl.transform_evidence.is_empty());
                let sig_a = ttl.composition_signature();
                let sig_b = ttl.composition_signature();
                assert_eq!(sig_a, sig_b);
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn ttl_hash_stability_across_runs() {
        run_logged_test(
            "ttl_hash_stability_across_runs",
            &("ttl-hash", 1_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ttl = TraceTransformLedger::new(build_program(ProgramSpec::Square));
                ttl.push_transform(Transform::Jit, "jit-evidence");
                ttl.push_transform(Transform::Grad, "grad-evidence");
                let proof_a = verify_transform_composition(&ttl)
                    .map_err(|err| format!("proof failed: {err}"))?;
                let proof_b = verify_transform_composition(&ttl)
                    .map_err(|err| format!("proof failed: {err}"))?;
                assert_eq!(proof_a.stack_hash_hex, proof_b.stack_hash_hex);
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn shape_scalar_rank_is_zero() {
        run_logged_test(
            "shape_scalar_rank_is_zero",
            &("shape-scalar", 0_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let scalar = Shape::scalar();
                assert_eq!(scalar.rank(), 0);
                assert_eq!(scalar.element_count(), Some(1));
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn vector_constructor_builds_tensor_value() {
        run_logged_test(
            "vector_constructor_builds_tensor_value",
            &("vector", [1_i64, 2, 3]),
            fj_test_utils::TestMode::Strict,
            || {
                let value =
                    Value::vector_i64(&[1, 2, 3]).expect("vector constructor should succeed");
                let tensor = value.as_tensor().expect("expected tensor value");
                assert_eq!(tensor.shape, Shape::vector(3));
                assert_eq!(tensor.len(), 3);
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn tensor_slice_axis0_on_rank2_returns_rank1_slice() {
        run_logged_test(
            "tensor_slice_axis0_on_rank2_returns_rank1_slice",
            &("slice-axis0", [2_u32, 3_u32]),
            fj_test_utils::TestMode::Strict,
            || {
                let tensor = TensorValue::new(
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
                .expect("rank2 tensor should build");

                let slice = tensor.slice_axis0(1).expect("slice should succeed");
                let Value::Tensor(slice_tensor) = slice else {
                    return Err("slice should be tensor".to_owned());
                };
                assert_eq!(slice_tensor.shape, Shape::vector(3));
                assert_eq!(
                    slice_tensor.elements,
                    vec![Literal::I64(4), Literal::I64(5), Literal::I64(6)]
                );
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn stack_axis0_restores_rank2_from_rank1_slices() {
        run_logged_test(
            "stack_axis0_restores_rank2_from_rank1_slices",
            &("stack-axis0", [2_u32, 3_u32]),
            fj_test_utils::TestMode::Strict,
            || {
                let row_a = Value::vector_i64(&[1, 2, 3]).expect("vector should build");
                let row_b = Value::vector_i64(&[4, 5, 6]).expect("vector should build");
                let stacked =
                    TensorValue::stack_axis0(&[row_a, row_b]).expect("stack should succeed");
                assert_eq!(stacked.shape, Shape { dims: vec![2, 3] });
                assert_eq!(
                    stacked.elements,
                    vec![
                        Literal::I64(1),
                        Literal::I64(2),
                        Literal::I64(3),
                        Literal::I64(4),
                        Literal::I64(5),
                        Literal::I64(6)
                    ]
                );
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn tensor_new_rejects_wrong_element_count() {
        run_logged_test(
            "tensor_new_rejects_wrong_element_count",
            &("tensor-mismatch", 4_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let err = TensorValue::new(
                    DType::I64,
                    Shape::vector(4),
                    vec![Literal::I64(1), Literal::I64(2)],
                )
                .expect_err("shape mismatch should fail");
                assert!(matches!(err, ValueError::ElementCountMismatch { .. }));
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn literal_and_value_scalar_helpers_cover_paths() {
        run_logged_test(
            "literal_and_value_scalar_helpers_cover_paths",
            &("literal-value-helpers", 1_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let primitive_names = [
                    Primitive::Add,
                    Primitive::Mul,
                    Primitive::Dot,
                    Primitive::Sin,
                    Primitive::Cos,
                    Primitive::ReduceSum,
                    Primitive::Reshape,
                    Primitive::Slice,
                    Primitive::Gather,
                    Primitive::Scatter,
                    Primitive::Transpose,
                    Primitive::BroadcastInDim,
                    Primitive::Concatenate,
                ]
                .into_iter()
                .map(Primitive::as_str)
                .collect::<Vec<_>>();
                assert_eq!(primitive_names.len(), 13);

                let scalar_i64 = Value::scalar_i64(7);
                let scalar_bool = Value::scalar_bool(true);
                let scalar_f64 = Value::scalar_f64(3.5);
                assert_eq!(
                    scalar_i64.as_scalar_literal().and_then(Literal::as_i64),
                    Some(7)
                );
                assert_eq!(
                    scalar_bool.as_scalar_literal().and_then(Literal::as_i64),
                    None
                );
                assert_eq!(scalar_f64.as_f64_scalar(), Some(3.5));
                assert_eq!(Literal::I64(9).as_i64(), Some(9));
                assert_eq!(Literal::from_f64(1.25).as_f64(), Some(1.25));
                assert!(Literal::I64(1).is_integral());
                assert!(!Literal::Bool(false).is_integral());

                let vector = Value::vector_f64(&[0.5, 1.5, 2.5]).expect("vector_f64 should work");
                let vector_tensor = vector.as_tensor().expect("expected tensor");
                assert_eq!(vector_tensor.to_f64_vec(), Some(vec![0.5, 1.5, 2.5]));
                assert!(!vector_tensor.is_empty());
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn tensor_axis0_error_variants_are_reported() {
        run_logged_test(
            "tensor_axis0_error_variants_are_reported",
            &("tensor-errors", 1_u32),
            fj_test_utils::TestMode::Hardened,
            || {
                let overflow = TensorValue::new(
                    DType::I64,
                    Shape {
                        dims: vec![u32::MAX, u32::MAX, 2],
                    },
                    vec![],
                )
                .expect_err("overflow shape should fail");
                assert!(matches!(overflow, ValueError::ShapeOverflow { .. }));
                let _overflow_text = format!("{overflow}");

                let scalar_tensor =
                    TensorValue::new(DType::I64, Shape::scalar(), vec![Literal::I64(1)])
                        .expect("scalar tensor should build");
                let rank_zero_err = scalar_tensor
                    .slice_axis0(0)
                    .expect_err("rank-0 axis slice should fail");
                assert!(matches!(
                    rank_zero_err,
                    ValueError::RankZeroAxisSliceUnsupported
                ));
                let _rank_zero_text = format!("{rank_zero_err}");

                let one_vec = TensorValue::new(DType::I64, Shape::vector(1), vec![Literal::I64(1)])
                    .expect("vector should build");
                let oob_err = one_vec
                    .slice_axis0(1)
                    .expect_err("out-of-bounds slice should fail");
                assert!(matches!(oob_err, ValueError::SliceIndexOutOfBounds { .. }));
                let _oob_text = format!("{oob_err}");

                let empty_stack =
                    TensorValue::stack_axis0(&[]).expect_err("empty stack should fail");
                assert!(matches!(empty_stack, ValueError::EmptyAxisStack));
                let _empty_stack_text = format!("{empty_stack}");

                let mixed = TensorValue::stack_axis0(&[
                    Value::scalar_i64(1),
                    Value::vector_i64(&[1]).expect("vector"),
                ])
                .expect_err("mixed stack should fail");
                assert!(matches!(mixed, ValueError::MixedAxisStackKinds));
                let _mixed_text = format!("{mixed}");

                let shape_mismatch = TensorValue::stack_axis0(&[
                    Value::vector_i64(&[1, 2]).expect("vector"),
                    Value::vector_i64(&[3, 4, 5]).expect("vector"),
                ])
                .expect_err("shape mismatch should fail");
                assert!(matches!(
                    shape_mismatch,
                    ValueError::AxisStackShapeMismatch { .. }
                ));
                let _shape_mismatch_text = format!("{shape_mismatch}");

                let dtype_mismatch = TensorValue::stack_axis0(&[
                    Value::vector_i64(&[1, 2]).expect("vector"),
                    Value::vector_f64(&[1.0, 2.0]).expect("vector"),
                ])
                .expect_err("dtype mismatch should fail");
                assert!(matches!(
                    dtype_mismatch,
                    ValueError::AxisStackDTypeMismatch { .. }
                ));
                let _dtype_mismatch_text = format!("{dtype_mismatch}");
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn jaxpr_validation_error_variants_have_actionable_messages() {
        run_logged_test(
            "jaxpr_validation_error_variants_have_actionable_messages",
            &("jaxpr-validation-errors", 4_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let dup_const = Jaxpr::new(vec![VarId(1)], vec![VarId(1)], vec![VarId(1)], vec![]);
                let err = dup_const
                    .validate_well_formed()
                    .expect_err("duplicate const should fail");
                assert!(matches!(
                    err,
                    JaxprValidationError::DuplicateBinding {
                        section: "constvars",
                        ..
                    }
                ));
                assert!(format!("{err}").contains("duplicate binding"));

                let unbound_input = Jaxpr::new(
                    vec![],
                    vec![],
                    vec![VarId(1)],
                    vec![Equation {
                        primitive: Primitive::Add,
                        inputs: smallvec![Atom::Var(VarId(99)), Atom::Lit(Literal::I64(1))],
                        outputs: smallvec![VarId(1)],
                        params: BTreeMap::new(),
                    }],
                );
                let err = unbound_input
                    .validate_well_formed()
                    .expect_err("unbound input should fail");
                assert!(matches!(err, JaxprValidationError::UnboundInputVar { .. }));
                assert!(format!("{err}").contains("unbound input"));

                let output_shadow = Jaxpr::new(
                    vec![VarId(1)],
                    vec![],
                    vec![VarId(1)],
                    vec![Equation {
                        primitive: Primitive::Add,
                        inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(1))],
                        outputs: smallvec![VarId(1)],
                        params: BTreeMap::new(),
                    }],
                );
                let err = output_shadow
                    .validate_well_formed()
                    .expect_err("shadowing output should fail");
                assert!(matches!(
                    err,
                    JaxprValidationError::OutputShadowsBinding { .. }
                ));
                assert!(format!("{err}").contains("shadows"));

                let unknown_out = Jaxpr::new(vec![VarId(1)], vec![], vec![VarId(9)], vec![]);
                let err = unknown_out
                    .validate_well_formed()
                    .expect_err("unknown outvar should fail");
                assert!(matches!(err, JaxprValidationError::UnknownOutvar { .. }));
                assert!(format!("{err}").contains("outvar"));
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn build_program_variants_cover_all_specs() {
        run_logged_test(
            "build_program_variants_cover_all_specs",
            &("build-program-specs", 8_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let specs = [
                    ProgramSpec::Add2,
                    ProgramSpec::Square,
                    ProgramSpec::SquarePlusLinear,
                    ProgramSpec::AddOne,
                    ProgramSpec::SinX,
                    ProgramSpec::CosX,
                    ProgramSpec::Dot3,
                    ProgramSpec::ReduceSumVec,
                ];
                for spec in specs {
                    let jaxpr = build_program(spec);
                    jaxpr
                        .validate_well_formed()
                        .map_err(|err| format!("program {:?} invalid: {err}", spec))?;
                    let _fingerprint = jaxpr.canonical_fingerprint();
                }
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn canonical_fingerprint_supports_bool_and_f64_literals() {
        run_logged_test(
            "canonical_fingerprint_supports_bool_and_f64_literals",
            &("fingerprint-literal-types", 2_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = Jaxpr::new(
                    vec![VarId(1)],
                    vec![],
                    vec![VarId(3)],
                    vec![
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(Literal::Bool(true))],
                            outputs: smallvec![VarId(2)],
                            params: BTreeMap::new(),
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![
                                Atom::Var(VarId(2)),
                                Atom::Lit(Literal::from_f64(1.0)),
                            ],
                            outputs: smallvec![VarId(3)],
                            params: BTreeMap::new(),
                        },
                    ],
                );
                let fingerprint = jaxpr.canonical_fingerprint().to_owned();
                assert!(fingerprint.contains("bool:true"));
                assert!(fingerprint.contains("f64bits:"));
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn proptest_strategy_entrypoints_are_exercised() {
        run_logged_test(
            "proptest_strategy_entrypoints_are_exercised",
            &("strategy-entrypoints", 1_u32),
            fj_test_utils::TestMode::Hardened,
            || {
                let mut runner = TestRunner::new(ProptestConfig::with_cases(1));
                runner
                    .run(&super::proptest_strategies::arb_literal(), |_| Ok(()))
                    .map_err(|err| err.to_string())?;
                runner
                    .run(&super::proptest_strategies::arb_var_id(), |_| Ok(()))
                    .map_err(|err| err.to_string())?;
                runner
                    .run(&super::proptest_strategies::arb_atom(), |_| Ok(()))
                    .map_err(|err| err.to_string())?;
                runner
                    .run(&super::proptest_strategies::arb_value(), |_| Ok(()))
                    .map_err(|err| err.to_string())?;
                runner
                    .run(&super::proptest_strategies::arb_shape(), |_| Ok(()))
                    .map_err(|err| err.to_string())?;
                runner
                    .run(&super::proptest_strategies::arb_primitive(), |_| Ok(()))
                    .map_err(|err| err.to_string())?;
                runner
                    .run(&super::proptest_strategies::arb_transform(), |_| Ok(()))
                    .map_err(|err| err.to_string())?;
                runner
                    .run(&super::proptest_strategies::arb_dtype(), |_| Ok(()))
                    .map_err(|err| err.to_string())?;
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn prop_jaxpr_generation_is_well_formed() {
        run_logged_test(
            "prop_jaxpr_generation_is_well_formed",
            &(
                "prop-jaxpr-well-formed",
                fj_test_utils::property_test_case_count(),
            ),
            fj_test_utils::TestMode::Strict,
            || {
                let mut runner = TestRunner::new(ProptestConfig::with_cases(
                    fj_test_utils::property_test_case_count(),
                ));
                runner
                    .run(&super::proptest_strategies::arb_binary_jaxpr(), |jaxpr| {
                        prop_assert!(jaxpr.validate_well_formed().is_ok());
                        Ok(())
                    })
                    .map_err(|err| err.to_string())?;
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn prop_fingerprint_uniqueness_for_distinct_jaxprs() {
        run_logged_test(
            "prop_fingerprint_uniqueness_for_distinct_jaxprs",
            &(
                "prop-fingerprint-unique",
                fj_test_utils::property_test_case_count(),
            ),
            fj_test_utils::TestMode::Strict,
            || {
                let mut runner = TestRunner::new(ProptestConfig::with_cases(
                    fj_test_utils::property_test_case_count(),
                ));
                let strategy = (
                    super::proptest_strategies::arb_binary_jaxpr(),
                    super::proptest_strategies::arb_binary_jaxpr(),
                );
                runner
                    .run(&strategy, |(lhs, rhs)| {
                        if lhs != rhs {
                            prop_assert_ne!(
                                lhs.canonical_fingerprint(),
                                rhs.canonical_fingerprint()
                            );
                        }
                        Ok(())
                    })
                    .map_err(|err| err.to_string())?;
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn prop_transform_composition_proof_idempotent() {
        run_logged_test(
            "prop_transform_composition_proof_idempotent",
            &(
                "prop-composition-idempotent",
                fj_test_utils::property_test_case_count(),
            ),
            fj_test_utils::TestMode::Strict,
            || {
                let mut runner = TestRunner::new(ProptestConfig::with_cases(
                    fj_test_utils::property_test_case_count(),
                ));
                let stack_strategy =
                    proptest::collection::vec(super::proptest_strategies::arb_transform(), 0..=3);

                runner
                    .run(&stack_strategy, |stack| {
                        let grad_count = stack
                            .iter()
                            .filter(|transform| **transform == Transform::Grad)
                            .count();
                        let vmap_count = stack
                            .iter()
                            .filter(|transform| **transform == Transform::Vmap)
                            .count();
                        prop_assume!(grad_count <= 1 && vmap_count <= 1);

                        let mut ttl = TraceTransformLedger::new(build_program(ProgramSpec::Square));
                        for (idx, transform) in stack.iter().enumerate() {
                            ttl.push_transform(*transform, format!("ev-{idx}"));
                        }
                        let proof_a = verify_transform_composition(&ttl)
                            .map_err(|err| TestCaseError::fail(err.to_string()))?;
                        let proof_b = verify_transform_composition(&ttl)
                            .map_err(|err| TestCaseError::fail(err.to_string()))?;
                        prop_assert_eq!(proof_a, proof_b);
                        Ok(())
                    })
                    .map_err(|err| err.to_string())?;
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn test_core_test_log_schema_contract() {
        run_logged_test(
            "test_core_test_log_schema_contract",
            &("schema-contract", 1_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let fixture_id =
                    fj_test_utils::fixture_id_from_json(&(1_u32, 2_u32)).expect("fixture digest");
                let log = fj_test_utils::TestLogV1::unit(
                    fj_test_utils::test_id(module_path!(), "test_core_test_log_schema_contract"),
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
    fn value_as_i64_scalar() {
        assert_eq!(Value::scalar_i64(42).as_i64_scalar(), Some(42));
        assert_eq!(Value::scalar_i64(-7).as_i64_scalar(), Some(-7));
        assert_eq!(Value::scalar_f64(2.78).as_i64_scalar(), None);
        assert_eq!(Value::scalar_bool(true).as_i64_scalar(), None);
    }

    #[test]
    fn value_as_bool_scalar() {
        assert_eq!(Value::scalar_bool(true).as_bool_scalar(), Some(true));
        assert_eq!(Value::scalar_bool(false).as_bool_scalar(), Some(false));
        assert_eq!(Value::scalar_i64(1).as_bool_scalar(), None);
        assert_eq!(Value::scalar_f64(1.0).as_bool_scalar(), None);
    }

    #[test]
    fn value_dtype_scalars() {
        assert_eq!(Value::scalar_i64(0).dtype(), DType::I64);
        assert_eq!(Value::scalar_f64(0.0).dtype(), DType::F64);
        assert_eq!(Value::scalar_bool(false).dtype(), DType::Bool);
    }

    #[test]
    fn value_dtype_tensors() {
        let t = Value::vector_i64(&[1, 2, 3]).unwrap();
        assert_eq!(t.dtype(), DType::I64);
        let t = Value::vector_f64(&[1.0, 2.0]).unwrap();
        assert_eq!(t.dtype(), DType::F64);
    }

    #[test]
    fn tensor_to_i64_vec() {
        let t = TensorValue::new(
            DType::I64,
            Shape::vector(3),
            vec![Literal::I64(10), Literal::I64(20), Literal::I64(30)],
        )
        .unwrap();
        assert_eq!(t.to_i64_vec(), Some(vec![10, 20, 30]));
    }

    #[test]
    fn tensor_to_i64_vec_wrong_dtype_returns_none() {
        let t = TensorValue::new(
            DType::F64,
            Shape::vector(2),
            vec![Literal::from_f64(1.0), Literal::from_f64(2.0)],
        )
        .unwrap();
        assert_eq!(t.to_i64_vec(), None);
    }
}
