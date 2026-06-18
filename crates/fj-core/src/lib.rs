#![forbid(unsafe_code)]

#[cfg(test)]
pub mod proptest_strategies;

use half::{bf16, f16};
use rustc_hash::FxHashSet;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use smallvec::{SmallVec, smallvec};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::ops::{Deref, Index, IndexMut};
use std::slice::SliceIndex;
use std::sync::{Arc, OnceLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompatibilityMode {
    Strict,
    Hardened,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    BF16,
    F16,
    F32,
    F64,
    I32,
    I64,
    U32,
    U64,
    Bool,
    Complex64,
    Complex128,
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
        if self.dims.contains(&0) {
            return Some(0);
        }

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
    Fma,
    Pow,
    Hypot,
    LogAddExp,
    LogAddExp2,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Floor,
    Ceil,
    Round,
    Trunc,
    // Trigonometric
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Deg2Rad,
    Rad2Deg,
    // Hyperbolic
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    // Additional math
    Expm1,
    Log1p,
    Log2,
    Exp2,
    Sinc,
    Sign,
    Square,
    Reciprocal,
    Logistic,
    Erf,
    Erfc,
    // Binary math
    Div,
    Rem,
    Gcd,
    Lcm,
    Atan2,
    // Complex number primitives
    Complex,
    Conj,
    Real,
    Imag,
    // Selection
    Select,
    SelectN,
    // Dot product
    Dot,
    DotGeneral,
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
    ReduceAnd,
    ReduceOr,
    ReduceXor,
    // Shape manipulation
    Reshape,
    Slice,
    DynamicSlice,
    DynamicUpdateSlice,
    Gather,
    Scatter,
    Transpose,
    BroadcastInDim,
    Concatenate,
    Pad,
    Rev,
    Squeeze,
    Split,
    ExpandDims,
    Tile,
    // Special math
    Cbrt,
    Lgamma,
    Digamma,
    Polygamma,
    ErfInv,
    Igamma,
    Igammac,
    Betainc,
    Zeta,
    BesselI0e,
    BesselI1e,
    IsFinite,
    IsNan,
    IsInf,
    Signbit,
    Heaviside,
    CopySign,
    Ldexp,
    XLogY,
    XLog1PY,
    IntegerPow,
    Nextafter,
    // Clamping
    Clamp,
    // Index generation
    Iota,
    BroadcastedIota,
    // Utility operations
    Copy,
    StopGradient,
    ConvertElementType,
    BitcastConvertType,
    ReducePrecision,
    // Linear algebra
    Cholesky,
    Qr,
    Lu,
    Svd,
    TriangularSolve,
    Eigh,
    Eig,
    Solve,
    Det,
    Slogdet,
    // FFT
    Fft,
    Ifft,
    Rfft,
    Irfft,
    // Encoding
    OneHot,
    // Cumulative
    Cumsum,
    Cumprod,
    Cummax,
    Cummin,
    // Sorting
    Sort,
    Argsort,
    TopK,
    // Index-of-extremum
    Argmin,
    Argmax,
    // Convolution
    Conv,
    // Control flow
    Cond,
    Scan,
    AssociativeScan,
    While,
    Switch,
    // Collective operations (pmap axis-aware reductions)
    Psum,
    Pmean,
    AllGather,
    AllToAll,
    AxisIndex,
    // Bitwise
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    BitwiseNot,
    ShiftLeft,
    ShiftRightArithmetic,
    ShiftRightLogical,
    // Windowed reduction (pooling)
    ReduceWindow,
    // Integer intrinsics
    PopulationCount,
    CountLeadingZeros,
    CountTrailingZeros,
}

impl Primitive {
    /// Canonical inventory of every primitive variant.
    pub const ALL: &[Self] = &[
        Self::Add,
        Self::Sub,
        Self::Mul,
        Self::Neg,
        Self::Abs,
        Self::Max,
        Self::Min,
        Self::Fma,
        Self::Pow,
        Self::Hypot,
        Self::LogAddExp,
        Self::LogAddExp2,
        Self::Exp,
        Self::Log,
        Self::Sqrt,
        Self::Rsqrt,
        Self::Floor,
        Self::Ceil,
        Self::Round,
        Self::Trunc,
        Self::Sin,
        Self::Cos,
        Self::Tan,
        Self::Asin,
        Self::Acos,
        Self::Atan,
        Self::Deg2Rad,
        Self::Rad2Deg,
        Self::Sinh,
        Self::Cosh,
        Self::Tanh,
        Self::Asinh,
        Self::Acosh,
        Self::Atanh,
        Self::Expm1,
        Self::Log1p,
        Self::Log2,
        Self::Exp2,
        Self::Sinc,
        Self::Sign,
        Self::Square,
        Self::Reciprocal,
        Self::Logistic,
        Self::Erf,
        Self::Erfc,
        Self::Div,
        Self::Rem,
        Self::Gcd,
        Self::Lcm,
        Self::Atan2,
        Self::Complex,
        Self::Conj,
        Self::Real,
        Self::Imag,
        Self::Select,
        Self::SelectN,
        Self::Dot,
        Self::DotGeneral,
        Self::Eq,
        Self::Ne,
        Self::Lt,
        Self::Le,
        Self::Gt,
        Self::Ge,
        Self::ReduceSum,
        Self::ReduceMax,
        Self::ReduceMin,
        Self::ReduceProd,
        Self::ReduceAnd,
        Self::ReduceOr,
        Self::ReduceXor,
        Self::Reshape,
        Self::Slice,
        Self::DynamicSlice,
        Self::DynamicUpdateSlice,
        Self::Gather,
        Self::Scatter,
        Self::Transpose,
        Self::BroadcastInDim,
        Self::Concatenate,
        Self::Pad,
        Self::Rev,
        Self::Squeeze,
        Self::Split,
        Self::ExpandDims,
        Self::Tile,
        Self::Cbrt,
        Self::Lgamma,
        Self::Digamma,
        Self::Polygamma,
        Self::ErfInv,
        Self::Igamma,
        Self::Igammac,
        Self::Betainc,
        Self::Zeta,
        Self::BesselI0e,
        Self::BesselI1e,
        Self::IsFinite,
        Self::IsNan,
        Self::IsInf,
        Self::Signbit,
        Self::Heaviside,
        Self::CopySign,
        Self::Ldexp,
        Self::XLogY,
        Self::XLog1PY,
        Self::IntegerPow,
        Self::Nextafter,
        Self::Clamp,
        Self::Iota,
        Self::BroadcastedIota,
        Self::Copy,
        Self::StopGradient,
        Self::ConvertElementType,
        Self::BitcastConvertType,
        Self::ReducePrecision,
        Self::Cholesky,
        Self::Qr,
        Self::Lu,
        Self::Svd,
        Self::TriangularSolve,
        Self::Eigh,
        Self::Eig,
        Self::Solve,
        Self::Det,
        Self::Slogdet,
        Self::Fft,
        Self::Ifft,
        Self::Rfft,
        Self::Irfft,
        Self::OneHot,
        Self::Cumsum,
        Self::Cumprod,
        Self::Cummax,
        Self::Cummin,
        Self::Sort,
        Self::Argsort,
        Self::TopK,
        Self::Argmin,
        Self::Argmax,
        Self::Conv,
        Self::Cond,
        Self::Scan,
        Self::AssociativeScan,
        Self::While,
        Self::Switch,
        Self::Psum,
        Self::Pmean,
        Self::AllGather,
        Self::AllToAll,
        Self::AxisIndex,
        Self::BitwiseAnd,
        Self::BitwiseOr,
        Self::BitwiseXor,
        Self::BitwiseNot,
        Self::ShiftLeft,
        Self::ShiftRightArithmetic,
        Self::ShiftRightLogical,
        Self::ReduceWindow,
        Self::PopulationCount,
        Self::CountLeadingZeros,
        Self::CountTrailingZeros,
    ];

    /// Pmap-only collective primitives that fail closed without an active pmap context.
    pub const PMAP_COLLECTIVES: &[Self] = &[
        Self::Psum,
        Self::Pmean,
        Self::AllGather,
        Self::AllToAll,
        Self::AxisIndex,
    ];

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
            Self::Fma => "fma",
            Self::Pow => "pow",
            Self::Hypot => "hypot",
            Self::LogAddExp => "logaddexp",
            Self::LogAddExp2 => "logaddexp2",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sqrt => "sqrt",
            Self::Rsqrt => "rsqrt",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Round => "round",
            Self::Trunc => "trunc",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Deg2Rad => "deg2rad",
            Self::Rad2Deg => "rad2deg",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tanh => "tanh",
            Self::Asinh => "asinh",
            Self::Acosh => "acosh",
            Self::Atanh => "atanh",
            Self::Expm1 => "expm1",
            Self::Log1p => "log1p",
            Self::Log2 => "log2",
            Self::Exp2 => "exp2",
            Self::Sinc => "sinc",
            Self::Sign => "sign",
            Self::Square => "square",
            Self::Reciprocal => "reciprocal",
            Self::Logistic => "logistic",
            Self::Erf => "erf",
            Self::Erfc => "erfc",
            Self::Div => "div",
            Self::Rem => "rem",
            Self::Gcd => "gcd",
            Self::Lcm => "lcm",
            Self::Atan2 => "atan2",
            Self::Complex => "complex",
            Self::Conj => "conj",
            Self::Real => "real",
            Self::Imag => "imag",
            Self::Select => "select",
            Self::SelectN => "select_n",
            Self::Dot => "dot",
            Self::DotGeneral => "dot_general",
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
            Self::ReduceAnd => "reduce_and",
            Self::ReduceOr => "reduce_or",
            Self::ReduceXor => "reduce_xor",
            Self::Reshape => "reshape",
            Self::Slice => "slice",
            Self::DynamicSlice => "dynamic_slice",
            Self::DynamicUpdateSlice => "dynamic_update_slice",
            Self::Gather => "gather",
            Self::Scatter => "scatter",
            Self::Transpose => "transpose",
            Self::BroadcastInDim => "broadcast_in_dim",
            Self::Concatenate => "concatenate",
            Self::Pad => "pad",
            Self::Rev => "rev",
            Self::Squeeze => "squeeze",
            Self::Split => "split",
            Self::ExpandDims => "expand_dims",
            Self::Tile => "tile",
            Self::Cbrt => "cbrt",
            Self::Lgamma => "lgamma",
            Self::Digamma => "digamma",
            Self::Polygamma => "polygamma",
            Self::ErfInv => "erf_inv",
            Self::Igamma => "igamma",
            Self::Igammac => "igammac",
            Self::Betainc => "betainc",
            Self::Zeta => "zeta",
            Self::BesselI0e => "bessel_i0e",
            Self::BesselI1e => "bessel_i1e",
            Self::IsFinite => "is_finite",
            Self::IsNan => "is_nan",
            Self::IsInf => "is_inf",
            Self::Signbit => "signbit",
            Self::Heaviside => "heaviside",
            Self::CopySign => "copysign",
            Self::Ldexp => "ldexp",
            Self::XLogY => "xlogy",
            Self::XLog1PY => "xlog1py",
            Self::IntegerPow => "integer_pow",
            Self::Nextafter => "nextafter",
            Self::Clamp => "clamp",
            Self::Iota => "iota",
            Self::BroadcastedIota => "broadcasted_iota",
            Self::Copy => "copy",
            Self::StopGradient => "stop_gradient",
            Self::ConvertElementType => "convert_element_type",
            Self::BitcastConvertType => "bitcast_convert_type",
            Self::ReducePrecision => "reduce_precision",
            Self::Cholesky => "cholesky",
            Self::Qr => "qr",
            Self::Lu => "lu",
            Self::Svd => "svd",
            Self::TriangularSolve => "triangular_solve",
            Self::Eigh => "eigh",
            Self::Eig => "eig",
            Self::Solve => "solve",
            Self::Det => "det",
            Self::Slogdet => "slogdet",
            Self::Fft => "fft",
            Self::Ifft => "ifft",
            Self::Rfft => "rfft",
            Self::Irfft => "irfft",
            Self::OneHot => "one_hot",
            Self::Cumsum => "cumsum",
            Self::Cumprod => "cumprod",
            Self::Cummax => "cummax",
            Self::Cummin => "cummin",
            Self::Sort => "sort",
            Self::Argsort => "argsort",
            Self::TopK => "top_k",
            Self::Argmin => "argmin",
            Self::Argmax => "argmax",
            Self::Conv => "conv",
            Self::Cond => "cond",
            Self::Scan => "scan",
            Self::AssociativeScan => "associative_scan",
            Self::While => "while_loop",
            Self::Switch => "switch",
            Self::Psum => "psum",
            Self::Pmean => "pmean",
            Self::AllGather => "all_gather",
            Self::AllToAll => "all_to_all",
            Self::AxisIndex => "axis_index",
            Self::BitwiseAnd => "bitwise_and",
            Self::BitwiseOr => "bitwise_or",
            Self::BitwiseXor => "bitwise_xor",
            Self::BitwiseNot => "bitwise_not",
            Self::ShiftLeft => "shift_left",
            Self::ShiftRightArithmetic => "shift_right_arithmetic",
            Self::ShiftRightLogical => "shift_right_logical",
            Self::ReduceWindow => "reduce_window",
            Self::PopulationCount => "population_count",
            Self::CountLeadingZeros => "count_leading_zeros",
            Self::CountTrailingZeros => "count_trailing_zeros",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Transform {
    Jit,
    Grad,
    Vmap,
    /// Parallel map for multi-device SPMD execution.
    /// V1: Fails closed until multi-device execution support lands.
    Pmap,
}

impl Transform {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Jit => "jit",
            Self::Grad => "grad",
            Self::Vmap => "vmap",
            Self::Pmap => "pmap",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VarId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Literal {
    I32(i32),
    I64(i64),
    U32(u32),
    U64(u64),
    Bool(bool),
    BF16Bits(u16),
    F16Bits(u16),
    F32Bits(u32),
    F64Bits(u64),
    Complex64Bits(u32, u32),
    Complex128Bits(u64, u64),
}

impl Literal {
    #[must_use]
    pub fn from_bf16_f32(value: f32) -> Self {
        Self::BF16Bits(bf16::from_f32(value).to_bits())
    }

    #[must_use]
    pub fn from_f16_f32(value: f32) -> Self {
        Self::F16Bits(f16::from_f32(value).to_bits())
    }

    /// Round f64 -> f32 with round-to-odd: exact values pass through, otherwise
    /// return whichever of the two bracketing f32 values has an odd mantissa LSB.
    /// Composing this with a round-to-nearest f32 -> f16/bf16 narrowing yields a
    /// correctly single-rounded f64 -> f16/bf16 (Boldo–Melquiond), avoiding the
    /// double rounding of `value as f32` followed by narrowing. f32's 24
    /// significand bits suffice for both f16 (11) and bf16 (8).
    fn f64_to_f32_round_to_odd(x: f64) -> f32 {
        let nearest = x as f32;
        if !nearest.is_finite() || f64::from(nearest) == x || (nearest.to_bits() & 1) == 1 {
            return nearest; // non-finite, exact, or already odd
        }
        let bits = nearest.to_bits();
        let toward_larger = x > f64::from(nearest);
        let negative = (bits >> 31) == 1;
        // `toward_larger == !negative` minimized (clippy::nonminimal_bool); same
        // truth table: step the f32 magnitude up when moving toward +inf for a
        // positive (or away from 0 for a negative), down otherwise.
        let neighbor = if toward_larger != negative {
            bits + 1
        } else {
            bits - 1
        };
        f32::from_bits(neighbor)
    }

    /// f64 -> bf16, correctly single-rounded (round-to-nearest-even) like XLA,
    /// via a round-to-odd f32 intermediate (see [`Self::f64_to_f32_round_to_odd`]).
    #[must_use]
    pub fn from_bf16_f64(value: f64) -> Self {
        Self::BF16Bits(bf16::from_f32(Self::f64_to_f32_round_to_odd(value)).to_bits())
    }

    /// f64 -> f16, correctly single-rounded (round-to-nearest-even) like XLA, via
    /// a round-to-odd f32 intermediate (see [`Self::f64_to_f32_round_to_odd`]).
    #[must_use]
    pub fn from_f16_f64(value: f64) -> Self {
        Self::F16Bits(f16::from_f32(Self::f64_to_f32_round_to_odd(value)).to_bits())
    }

    #[must_use]
    pub fn from_f32(value: f32) -> Self {
        Self::F32Bits(value.to_bits())
    }

    #[must_use]
    pub fn from_f64(value: f64) -> Self {
        Self::F64Bits(value.to_bits())
    }

    #[must_use]
    pub fn from_complex64(re: f32, im: f32) -> Self {
        Self::Complex64Bits(re.to_bits(), im.to_bits())
    }

    #[must_use]
    pub fn from_complex128(re: f64, im: f64) -> Self {
        Self::Complex128Bits(re.to_bits(), im.to_bits())
    }

    /// Check whether this literal's kind is consistent with the given dtype.
    ///
    /// Used to assert the dtype/element invariant: a `TensorValue` declaring
    /// `DType::X` must contain only literals whose kind corresponds to `X`.
    /// `I32` tensors may still use `Literal::I64` as their dense backing;
    /// scalar `I32` values use `Literal::I32`. Everything else is a 1:1 pairing.
    #[must_use]
    pub fn matches_dtype(self, dtype: DType) -> bool {
        match dtype {
            DType::Bool => matches!(self, Self::Bool(_)),
            DType::I32 => matches!(self, Self::I32(_) | Self::I64(_)),
            DType::I64 => matches!(self, Self::I64(_)),
            DType::U32 => matches!(self, Self::U32(_)),
            DType::U64 => matches!(self, Self::U64(_)),
            DType::BF16 => matches!(self, Self::BF16Bits(_)),
            DType::F16 => matches!(self, Self::F16Bits(_)),
            DType::F32 => matches!(self, Self::F32Bits(_)),
            DType::F64 => matches!(self, Self::F64Bits(_)),
            DType::Complex64 => matches!(self, Self::Complex64Bits(..)),
            DType::Complex128 => matches!(self, Self::Complex128Bits(..)),
        }
    }

    #[must_use]
    pub fn as_f64(self) -> Option<f64> {
        match self {
            Self::F64Bits(bits) => Some(f64::from_bits(bits)),
            Self::F32Bits(bits) => Some(f64::from(f32::from_bits(bits))),
            Self::I32(value) => Some(f64::from(value)),
            Self::I64(value) => Some(value as f64),
            Self::U32(value) => Some(value as f64),
            Self::U64(value) => Some(value as f64),
            Self::BF16Bits(bits) => Some(f64::from(f32::from(bf16::from_bits(bits)))),
            Self::F16Bits(bits) => Some(f64::from(f32::from(f16::from_bits(bits)))),
            Self::Bool(_) | Self::Complex64Bits(..) | Self::Complex128Bits(..) => None,
        }
    }

    #[must_use]
    pub fn as_i64(self) -> Option<i64> {
        match self {
            Self::I32(value) => Some(i64::from(value)),
            Self::I64(value) => Some(value),
            Self::U32(value) => Some(i64::from(value)),
            Self::U64(value) => i64::try_from(value).ok(),
            Self::Bool(_)
            | Self::BF16Bits(_)
            | Self::F16Bits(_)
            | Self::F32Bits(_)
            | Self::F64Bits(_)
            | Self::Complex64Bits(..)
            | Self::Complex128Bits(..) => None,
        }
    }

    #[must_use]
    pub fn as_u64(self) -> Option<u64> {
        match self {
            Self::U32(value) => Some(u64::from(value)),
            Self::U64(value) => Some(value),
            Self::I32(value) => u64::try_from(value).ok(),
            Self::I64(value) => u64::try_from(value).ok(),
            Self::Bool(_)
            | Self::BF16Bits(_)
            | Self::F16Bits(_)
            | Self::F32Bits(_)
            | Self::F64Bits(_)
            | Self::Complex64Bits(..)
            | Self::Complex128Bits(..) => None,
        }
    }

    #[must_use]
    pub fn as_bf16_f32(self) -> Option<f32> {
        match self {
            Self::BF16Bits(bits) => Some(f32::from(bf16::from_bits(bits))),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_f16_f32(self) -> Option<f32> {
        match self {
            Self::F16Bits(bits) => Some(f32::from(f16::from_bits(bits))),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_complex64(self) -> Option<(f32, f32)> {
        match self {
            Self::Complex64Bits(re, im) => Some((f32::from_bits(re), f32::from_bits(im))),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_complex128(self) -> Option<(f64, f64)> {
        match self {
            Self::Complex128Bits(re, im) => Some((f64::from_bits(re), f64::from_bits(im))),
            _ => None,
        }
    }

    #[must_use]
    pub fn is_integral(self) -> bool {
        matches!(
            self,
            Self::I32(_) | Self::I64(_) | Self::U32(_) | Self::U64(_)
        )
    }

    #[must_use]
    pub fn is_complex(self) -> bool {
        matches!(self, Self::Complex64Bits(..) | Self::Complex128Bits(..))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Value {
    Scalar(Literal),
    Tensor(TensorValue),
}

impl Value {
    #[must_use]
    pub fn scalar_i32(value: i32) -> Self {
        Self::Scalar(Literal::I32(value))
    }

    #[must_use]
    pub fn scalar_i64(value: i64) -> Self {
        Self::Scalar(Literal::I64(value))
    }

    #[must_use]
    pub fn scalar_u32(value: u32) -> Self {
        Self::Scalar(Literal::U32(value))
    }

    #[must_use]
    pub fn scalar_u64(value: u64) -> Self {
        Self::Scalar(Literal::U64(value))
    }

    #[must_use]
    pub fn scalar_bf16(value: f32) -> Self {
        Self::Scalar(Literal::from_bf16_f32(value))
    }

    #[must_use]
    pub fn scalar_f16(value: f32) -> Self {
        Self::Scalar(Literal::from_f16_f32(value))
    }

    #[must_use]
    pub fn scalar_f32(value: f32) -> Self {
        Self::Scalar(Literal::from_f32(value))
    }

    #[must_use]
    pub fn scalar_f64(value: f64) -> Self {
        Self::Scalar(Literal::from_f64(value))
    }

    #[must_use]
    pub fn scalar_bool(value: bool) -> Self {
        Self::Scalar(Literal::Bool(value))
    }

    #[must_use]
    pub fn scalar_complex64(re: f32, im: f32) -> Self {
        Self::Scalar(Literal::from_complex64(re, im))
    }

    #[must_use]
    pub fn scalar_complex128(re: f64, im: f64) -> Self {
        Self::Scalar(Literal::from_complex128(re, im))
    }

    pub fn vector_i64(values: &[i64]) -> Result<Self, ValueError> {
        Ok(Self::Tensor(TensorValue::new_i64_values(
            Shape::vector(values.len() as u32),
            values.to_vec(),
        )?))
    }

    pub fn vector_bool(values: &[bool]) -> Result<Self, ValueError> {
        Ok(Self::Tensor(TensorValue::new_bool_values(
            Shape::vector(values.len() as u32),
            values.to_vec(),
        )?))
    }

    pub fn vector_f64(values: &[f64]) -> Result<Self, ValueError> {
        Ok(Self::Tensor(TensorValue::new_f64_values(
            Shape::vector(values.len() as u32),
            values.to_vec(),
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

    /// Extract complex128 scalar as (re, im) f64 pair.
    #[must_use]
    pub fn as_complex128_scalar(&self) -> Option<(f64, f64)> {
        match self.as_scalar_literal() {
            Some(Literal::Complex128Bits(re_bits, im_bits)) => {
                Some((f64::from_bits(re_bits), f64::from_bits(im_bits)))
            }
            _ => None,
        }
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        match self {
            Self::Scalar(lit) => match lit {
                Literal::I32(_) => DType::I32,
                Literal::I64(_) => DType::I64,
                Literal::U32(_) => DType::U32,
                Literal::U64(_) => DType::U64,
                Literal::Bool(_) => DType::Bool,
                Literal::BF16Bits(_) => DType::BF16,
                Literal::F16Bits(_) => DType::F16,
                Literal::F32Bits(_) => DType::F32,
                Literal::F64Bits(_) => DType::F64,
                Literal::Complex64Bits(..) => DType::Complex64,
                Literal::Complex128Bits(..) => DType::Complex128,
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

pub struct LiteralBuffer {
    storage: LiteralBufferStorage,
}

/// A lazily-materialized dense lane of a `Concat` buffer: `None` once computed if
/// the parts aren't all that dtype, else the concatenated dense `Vec<T>`.
type ConcatLane<T> = Arc<OnceLock<Option<Arc<Vec<T>>>>>;

enum LiteralBufferStorage {
    Literals(Arc<Vec<Literal>>),
    F64 {
        values: Arc<Vec<f64>>,
        literals: Arc<OnceLock<Arc<Vec<Literal>>>>,
    },
    F64OnePlusXPlusX {
        base: Arc<Vec<f64>>,
        values: Arc<OnceLock<Arc<Vec<f64>>>>,
        literals: Arc<OnceLock<Arc<Vec<Literal>>>>,
    },
    /// Dense `f32` storage. f32 is JAX's DEFAULT float dtype, so a packed
    /// `&[f32]` slice (`as_f32_slice`) lets the hottest ML paths (f32 elementwise
    /// transcendentals, threaded activations) avoid per-element `Literal`
    /// materialization. Materializing back via `Literal::from_f32(v)` is
    /// bit-identical because the stored `f32` IS the logical value.
    F32 {
        values: Arc<Vec<f32>>,
        literals: Arc<OnceLock<Arc<Vec<Literal>>>>,
    },
    I64 {
        values: Arc<Vec<i64>>,
        literals: Arc<OnceLock<Arc<Vec<Literal>>>>,
    },
    Bool {
        values: Arc<Vec<bool>>,
        literals: Arc<OnceLock<Arc<Vec<Literal>>>>,
    },
    BoolWords {
        words: Arc<Vec<u64>>,
        len: usize,
        literals: Arc<OnceLock<Arc<Vec<Literal>>>>,
    },
    /// Dense complex storage: `(re, im)` pairs as `f64`, tagged with the logical
    /// complex dtype (`Complex64` or `Complex128`). Lets complex-heavy ops (FFT,
    /// complex elementwise) borrow a packed `&[(f64, f64)]` slice and emit dense
    /// outputs without per-element `Literal` materialization. For `Complex64` the
    /// stored `f64` pairs are exactly the widened `f32` values, so materializing
    /// back via `from_complex64(re as f32, im as f32)` is bit-identical.
    Complex {
        values: Arc<Vec<(f64, f64)>>,
        dtype: DType,
        literals: Arc<OnceLock<Arc<Vec<Literal>>>>,
    },
    /// Dense half-float storage: raw 16-bit values, tagged with the logical dtype
    /// (`BF16` or `F16`). Lets half-precision ML paths (bf16 is the dominant
    /// training dtype) borrow a packed `&[u16]` slice and emit dense outputs
    /// without per-element `Literal` materialization. The stored `u16` IS the
    /// logical value's bit pattern, so materializing back via `BF16Bits(bits)` /
    /// `F16Bits(bits)` is bit-identical.
    HalfFloat {
        values: Arc<Vec<u16>>,
        dtype: DType,
        literals: Arc<OnceLock<Arc<Vec<Literal>>>>,
    },
    /// Dense unsigned 32-bit storage. `u32` is JAX's RNG word type (threefry
    /// output) and a common bit-manipulation / index dtype, so a packed `&[u32]`
    /// slice (`as_u32_slice`) lets those paths avoid per-element `Literal`
    /// materialization. The stored `u32` IS the logical value, so materializing
    /// back via `Literal::U32(v)` is bit-identical.
    U32 {
        values: Arc<Vec<u32>>,
        literals: Arc<OnceLock<Arc<Vec<Literal>>>>,
    },
    /// Dense unsigned 64-bit storage. Mirror of [`Self::U32`] for `u64`
    /// (`as_u64_slice`); materializing back via `Literal::U64(v)` is bit-identical.
    U64 {
        values: Arc<Vec<u64>>,
        literals: Arc<OnceLock<Arc<Vec<Literal>>>>,
    },
    RepeatedPatches {
        base: Arc<Vec<Literal>>,
        repeats: usize,
        patches: Arc<Vec<(usize, Literal)>>,
        literals: Arc<OnceLock<Arc<Vec<Literal>>>>,
    },
    Concat {
        parts: Arc<Vec<LiteralBufferSlice>>,
        len: usize,
        literals: Arc<OnceLock<Arc<Vec<Literal>>>>,
        f64_values: ConcatLane<f64>,
        f32_values: ConcatLane<f32>,
        i64_values: ConcatLane<i64>,
        half_values: ConcatLane<u16>,
        complex_values: ConcatLane<(f64, f64)>,
        bool_values: ConcatLane<bool>,
        u32_values: ConcatLane<u32>,
        u64_values: ConcatLane<u64>,
    },
}

struct LiteralBufferSlice {
    buffer: LiteralBuffer,
    start: usize,
    len: usize,
}

impl LiteralBuffer {
    #[must_use]
    pub fn new(elements: Vec<Literal>) -> Self {
        Self {
            storage: LiteralBufferStorage::Literals(Arc::new(elements)),
        }
    }

    #[must_use]
    pub fn from_f64_values(values: Vec<f64>) -> Self {
        Self {
            storage: LiteralBufferStorage::F64 {
                values: Arc::new(values),
                literals: Arc::new(OnceLock::new()),
            },
        }
    }

    #[must_use]
    pub fn from_f64_one_plus_x_plus_x(base: Arc<Vec<f64>>) -> Self {
        Self {
            storage: LiteralBufferStorage::F64OnePlusXPlusX {
                base,
                values: Arc::new(OnceLock::new()),
                literals: Arc::new(OnceLock::new()),
            },
        }
    }

    #[must_use]
    pub fn from_f32_values(values: Vec<f32>) -> Self {
        Self {
            storage: LiteralBufferStorage::F32 {
                values: Arc::new(values),
                literals: Arc::new(OnceLock::new()),
            },
        }
    }

    #[must_use]
    pub fn from_i64_values(values: Vec<i64>) -> Self {
        Self {
            storage: LiteralBufferStorage::I64 {
                values: Arc::new(values),
                literals: Arc::new(OnceLock::new()),
            },
        }
    }

    #[must_use]
    pub fn from_u32_values(values: Vec<u32>) -> Self {
        Self {
            storage: LiteralBufferStorage::U32 {
                values: Arc::new(values),
                literals: Arc::new(OnceLock::new()),
            },
        }
    }

    #[must_use]
    pub fn from_u64_values(values: Vec<u64>) -> Self {
        Self {
            storage: LiteralBufferStorage::U64 {
                values: Arc::new(values),
                literals: Arc::new(OnceLock::new()),
            },
        }
    }

    #[must_use]
    pub fn from_bool_values(values: Vec<bool>) -> Self {
        Self {
            storage: LiteralBufferStorage::Bool {
                values: Arc::new(values),
                literals: Arc::new(OnceLock::new()),
            },
        }
    }

    #[must_use]
    pub fn from_bool_words(words: Vec<u64>, len: usize) -> Option<Self> {
        let expected_words = len.div_ceil(u64::BITS as usize);
        if words.len() != expected_words {
            return None;
        }
        let mut words = words;
        clear_unused_bool_word_bits(&mut words, len);
        Some(Self {
            storage: LiteralBufferStorage::BoolWords {
                words: Arc::new(words),
                len,
                literals: Arc::new(OnceLock::new()),
            },
        })
    }

    /// Build a dense complex buffer from `(re, im)` `f64` pairs tagged with a
    /// complex dtype. `dtype` must be `Complex64` or `Complex128`.
    #[must_use]
    pub fn from_complex_values(values: Vec<(f64, f64)>, dtype: DType) -> Self {
        debug_assert!(
            matches!(dtype, DType::Complex64 | DType::Complex128),
            "from_complex_values requires a complex dtype"
        );
        // Invariant: a `Complex64` dense buffer always holds f32-exact values, so
        // the packed slice (`as_complex_slice`) and the lazily materialized
        // `Literal` agree bit-for-bit. `Complex128` keeps full f64 precision.
        let mut values = values;
        if dtype == DType::Complex64 {
            for pair in &mut values {
                pair.0 = pair.0 as f32 as f64;
                pair.1 = pair.1 as f32 as f64;
            }
        }
        Self {
            storage: LiteralBufferStorage::Complex {
                values: Arc::new(values),
                dtype,
                literals: Arc::new(OnceLock::new()),
            },
        }
    }

    /// Build a dense half-float buffer from raw 16-bit values tagged with a
    /// half-float dtype. `dtype` must be `BF16` or `F16`. The `u16` bits ARE the
    /// logical values, so materialization is bit-exact.
    #[must_use]
    pub fn from_half_float_values(values: Vec<u16>, dtype: DType) -> Self {
        debug_assert!(
            matches!(dtype, DType::BF16 | DType::F16),
            "from_half_float_values requires a half-float dtype"
        );
        Self {
            storage: LiteralBufferStorage::HalfFloat {
                values: Arc::new(values),
                dtype,
                literals: Arc::new(OnceLock::new()),
            },
        }
    }

    #[must_use]
    pub fn from_repeated_with_patches(
        base: Vec<Literal>,
        repeats: usize,
        patches: Vec<(usize, Literal)>,
    ) -> Option<Self> {
        let len = base.len().checked_mul(repeats)?;
        if patches.iter().any(|(index, _)| *index >= len) {
            return None;
        }

        Some(Self {
            storage: LiteralBufferStorage::RepeatedPatches {
                base: Arc::new(base),
                repeats,
                patches: Arc::new(patches),
                literals: Arc::new(OnceLock::new()),
            },
        })
    }

    #[must_use]
    pub fn from_concat_slices(slices: Vec<(Self, usize, usize)>) -> Option<Self> {
        let mut len = 0_usize;
        let mut parts = Vec::with_capacity(slices.len());
        for (buffer, start, part_len) in slices {
            let end = start.checked_add(part_len)?;
            if end > buffer.len() {
                return None;
            }
            len = len.checked_add(part_len)?;
            if part_len != 0 {
                parts.push(LiteralBufferSlice {
                    buffer,
                    start,
                    len: part_len,
                });
            }
        }

        Some(Self {
            storage: LiteralBufferStorage::Concat {
                parts: Arc::new(parts),
                len,
                literals: Arc::new(OnceLock::new()),
                f64_values: Arc::new(OnceLock::new()),
                f32_values: Arc::new(OnceLock::new()),
                i64_values: Arc::new(OnceLock::new()),
                half_values: Arc::new(OnceLock::new()),
                complex_values: Arc::new(OnceLock::new()),
                bool_values: Arc::new(OnceLock::new()),
                u32_values: Arc::new(OnceLock::new()),
                u64_values: Arc::new(OnceLock::new()),
            },
        })
    }

    #[must_use]
    pub fn as_slice(&self) -> &[Literal] {
        match &self.storage {
            LiteralBufferStorage::Literals(elements) => elements.as_slice(),
            LiteralBufferStorage::F64 { values, literals } => literals
                .get_or_init(|| Arc::new(values.iter().copied().map(Literal::from_f64).collect()))
                .as_slice(),
            LiteralBufferStorage::F64OnePlusXPlusX {
                base,
                values,
                literals,
            } => {
                let values =
                    values.get_or_init(|| Arc::new(materialize_f64_one_plus_x_plus_x(base)));
                literals
                    .get_or_init(|| {
                        Arc::new(values.iter().copied().map(Literal::from_f64).collect())
                    })
                    .as_slice()
            }
            LiteralBufferStorage::F32 { values, literals } => literals
                .get_or_init(|| Arc::new(values.iter().copied().map(Literal::from_f32).collect()))
                .as_slice(),
            LiteralBufferStorage::I64 { values, literals } => literals
                .get_or_init(|| Arc::new(values.iter().copied().map(Literal::I64).collect()))
                .as_slice(),
            LiteralBufferStorage::Bool { values, literals } => literals
                .get_or_init(|| Arc::new(values.iter().copied().map(Literal::Bool).collect()))
                .as_slice(),
            LiteralBufferStorage::U32 { values, literals } => literals
                .get_or_init(|| Arc::new(values.iter().copied().map(Literal::U32).collect()))
                .as_slice(),
            LiteralBufferStorage::U64 { values, literals } => literals
                .get_or_init(|| Arc::new(values.iter().copied().map(Literal::U64).collect()))
                .as_slice(),
            LiteralBufferStorage::BoolWords {
                words,
                len,
                literals,
            } => literals
                .get_or_init(|| Arc::new(materialize_bool_words(words, *len)))
                .as_slice(),
            LiteralBufferStorage::Complex {
                values,
                dtype,
                literals,
            } => literals
                .get_or_init(|| {
                    Arc::new(
                        values
                            .iter()
                            .map(|&(re, im)| complex_pair_to_literal(re, im, *dtype))
                            .collect(),
                    )
                })
                .as_slice(),
            LiteralBufferStorage::HalfFloat {
                values,
                dtype,
                literals,
            } => literals
                .get_or_init(|| {
                    Arc::new(
                        values
                            .iter()
                            .map(|&bits| half_bits_to_literal(bits, *dtype))
                            .collect(),
                    )
                })
                .as_slice(),
            LiteralBufferStorage::RepeatedPatches {
                base,
                repeats,
                patches,
                literals,
            } => literals
                .get_or_init(|| {
                    Arc::new(materialize_repeated_patches(
                        base,
                        *repeats,
                        patches.as_slice(),
                    ))
                })
                .as_slice(),
            LiteralBufferStorage::Concat {
                parts,
                len,
                literals,
                ..
            } => literals
                .get_or_init(|| Arc::new(materialize_concat_slices(parts, *len)))
                .as_slice(),
        }
    }

    #[must_use]
    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        match &self.storage {
            LiteralBufferStorage::Literals(_) => None,
            LiteralBufferStorage::F64 { values, .. } => Some(values.as_slice()),
            LiteralBufferStorage::F64OnePlusXPlusX { base, values, .. } => Some(
                values
                    .get_or_init(|| Arc::new(materialize_f64_one_plus_x_plus_x(base)))
                    .as_slice(),
            ),
            LiteralBufferStorage::F32 { .. } => None,
            LiteralBufferStorage::HalfFloat { .. } => None,
            LiteralBufferStorage::I64 { .. } => None,
            LiteralBufferStorage::Bool { .. } => None,
            LiteralBufferStorage::BoolWords { .. } => None,
            LiteralBufferStorage::Complex { .. } => None,
            LiteralBufferStorage::U32 { .. } => None,
            LiteralBufferStorage::U64 { .. } => None,
            LiteralBufferStorage::RepeatedPatches { .. } => None,
            LiteralBufferStorage::Concat {
                parts,
                len,
                f64_values,
                ..
            } => f64_values
                .get_or_init(|| materialize_concat_f64_slices(parts, *len).map(Arc::new))
                .as_deref()
                .map(Vec::as_slice),
        }
    }

    /// Borrow the dense unsigned-32 storage as a packed `&[u32]` slice, if this
    /// buffer is `u32`-backed. Returns `None` otherwise.
    #[must_use]
    pub fn as_u32_slice(&self) -> Option<&[u32]> {
        match &self.storage {
            LiteralBufferStorage::U32 { values, .. } => Some(values.as_slice()),
            LiteralBufferStorage::Concat {
                parts,
                len,
                u32_values,
                ..
            } => u32_values
                .get_or_init(|| materialize_concat_u32_slices(parts, *len).map(Arc::new))
                .as_deref()
                .map(Vec::as_slice),
            _ => None,
        }
    }

    /// Borrow the dense unsigned-64 storage as a packed `&[u64]` slice, if this
    /// buffer is `u64`-backed. Returns `None` otherwise.
    #[must_use]
    pub fn as_u64_slice(&self) -> Option<&[u64]> {
        match &self.storage {
            LiteralBufferStorage::U64 { values, .. } => Some(values.as_slice()),
            LiteralBufferStorage::Concat {
                parts,
                len,
                u64_values,
                ..
            } => u64_values
                .get_or_init(|| materialize_concat_u64_slices(parts, *len).map(Arc::new))
                .as_deref()
                .map(Vec::as_slice),
            _ => None,
        }
    }

    #[must_use]
    pub fn f64_values_arc(&self) -> Option<Arc<Vec<f64>>> {
        match &self.storage {
            LiteralBufferStorage::F64 { values, .. } => Some(Arc::clone(values)),
            LiteralBufferStorage::F64OnePlusXPlusX { base, values, .. } => {
                Some(Arc::clone(values.get_or_init(|| {
                    Arc::new(materialize_f64_one_plus_x_plus_x(base))
                })))
            }
            _ => None,
        }
    }

    /// Borrow the dense half-float storage as a packed `&[u16]` bit slice, if this
    /// buffer is half-float-backed. Returns `None` otherwise. The logical dtype
    /// (`BF16`/`F16`) is available via [`Self::half_float_dtype`].
    #[must_use]
    pub fn as_half_float_slice(&self) -> Option<&[u16]> {
        match &self.storage {
            LiteralBufferStorage::HalfFloat { values, .. } => Some(values.as_slice()),
            LiteralBufferStorage::Concat {
                parts,
                len,
                half_values,
                ..
            } => half_values
                .get_or_init(|| materialize_concat_half_slices(parts, *len).map(Arc::new))
                .as_deref()
                .map(Vec::as_slice),
            _ => None,
        }
    }

    /// The logical half-float dtype of a half-float-backed buffer (`BF16`/`F16`),
    /// or `None` if this buffer is not dense-half-float-backed.
    #[must_use]
    pub fn half_float_dtype(&self) -> Option<DType> {
        match &self.storage {
            LiteralBufferStorage::HalfFloat { dtype, .. } => Some(*dtype),
            _ => None,
        }
    }

    /// Borrow the dense `f32` storage as a packed `&[f32]` slice, if this buffer
    /// is `f32`-backed. Returns `None` for `Literal`-backed or non-`f32` buffers
    /// (callers fall back to per-element extraction).
    #[must_use]
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match &self.storage {
            LiteralBufferStorage::F32 { values, .. } => Some(values.as_slice()),
            LiteralBufferStorage::Concat {
                parts,
                len,
                f32_values,
                ..
            } => f32_values
                .get_or_init(|| materialize_concat_f32_slices(parts, *len).map(Arc::new))
                .as_deref()
                .map(Vec::as_slice),
            _ => None,
        }
    }

    /// Borrow the dense complex storage as packed `(re, im)` `f64` pairs, if this
    /// buffer is complex-backed. Returns `None` for `Literal`-backed or non-complex
    /// buffers (callers fall back to per-element extraction). The logical complex
    /// dtype is available via [`Self::complex_dtype`].
    #[must_use]
    pub fn as_complex_slice(&self) -> Option<&[(f64, f64)]> {
        match &self.storage {
            LiteralBufferStorage::Complex { values, .. } => Some(values.as_slice()),
            // Lazy concat of complex parts (e.g. concatenating spectra in an FFT/
            // signal pipeline): materialize the contiguous (re,im) backing once,
            // bypassing the per-Literal Vec<Literal> the generic path would build.
            LiteralBufferStorage::Concat {
                parts,
                len,
                complex_values,
                ..
            } => complex_values
                .get_or_init(|| materialize_concat_complex_slices(parts, *len).map(Arc::new))
                .as_deref()
                .map(Vec::as_slice),
            _ => None,
        }
    }

    /// The logical complex dtype of a complex-backed buffer (`Complex64`/`Complex128`),
    /// or `None` if this buffer is not dense-complex-backed.
    #[must_use]
    pub fn complex_dtype(&self) -> Option<DType> {
        match &self.storage {
            LiteralBufferStorage::Complex { dtype, .. } => Some(*dtype),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_i64_slice(&self) -> Option<&[i64]> {
        match &self.storage {
            LiteralBufferStorage::I64 { values, .. } => Some(values.as_slice()),
            LiteralBufferStorage::Concat {
                parts,
                len,
                i64_values,
                ..
            } => i64_values
                .get_or_init(|| materialize_concat_i64_slices(parts, *len).map(Arc::new))
                .as_deref()
                .map(Vec::as_slice),
            LiteralBufferStorage::Literals(_)
            | LiteralBufferStorage::F64 { .. }
            | LiteralBufferStorage::F64OnePlusXPlusX { .. }
            | LiteralBufferStorage::F32 { .. }
            | LiteralBufferStorage::HalfFloat { .. }
            | LiteralBufferStorage::Bool { .. }
            | LiteralBufferStorage::BoolWords { .. }
            | LiteralBufferStorage::Complex { .. }
            | LiteralBufferStorage::U32 { .. }
            | LiteralBufferStorage::U64 { .. }
            | LiteralBufferStorage::RepeatedPatches { .. } => None,
        }
    }

    #[must_use]
    pub fn as_bool_slice(&self) -> Option<&[bool]> {
        match &self.storage {
            LiteralBufferStorage::Bool { values, .. } => Some(values.as_slice()),
            LiteralBufferStorage::Concat {
                parts,
                len,
                bool_values,
                ..
            } => bool_values
                .get_or_init(|| materialize_concat_bool_slices(parts, *len).map(Arc::new))
                .as_deref()
                .map(Vec::as_slice),
            LiteralBufferStorage::Literals(_)
            | LiteralBufferStorage::F64 { .. }
            | LiteralBufferStorage::F64OnePlusXPlusX { .. }
            | LiteralBufferStorage::F32 { .. }
            | LiteralBufferStorage::HalfFloat { .. }
            | LiteralBufferStorage::I64 { .. }
            | LiteralBufferStorage::BoolWords { .. }
            | LiteralBufferStorage::Complex { .. }
            | LiteralBufferStorage::U32 { .. }
            | LiteralBufferStorage::U64 { .. }
            | LiteralBufferStorage::RepeatedPatches { .. } => None,
        }
    }

    #[must_use]
    pub fn as_bool_words(&self) -> Option<(&[u64], usize)> {
        match &self.storage {
            LiteralBufferStorage::BoolWords { words, len, .. } => Some((words.as_slice(), *len)),
            _ => None,
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        match &self.storage {
            LiteralBufferStorage::Literals(elements) => elements.len(),
            LiteralBufferStorage::F64 { values, .. } => values.len(),
            LiteralBufferStorage::F64OnePlusXPlusX { base, .. } => base.len(),
            LiteralBufferStorage::F32 { values, .. } => values.len(),
            LiteralBufferStorage::HalfFloat { values, .. } => values.len(),
            LiteralBufferStorage::I64 { values, .. } => values.len(),
            LiteralBufferStorage::Bool { values, .. } => values.len(),
            LiteralBufferStorage::BoolWords { len, .. } => *len,
            LiteralBufferStorage::Complex { values, .. } => values.len(),
            LiteralBufferStorage::U32 { values, .. } => values.len(),
            LiteralBufferStorage::U64 { values, .. } => values.len(),
            LiteralBufferStorage::RepeatedPatches { base, repeats, .. } => base.len() * repeats,
            LiteralBufferStorage::Concat { len, .. } => *len,
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn to_vec(&self) -> Vec<Literal> {
        self.as_slice().to_vec()
    }

    fn make_mut(&mut self) -> &mut Vec<Literal> {
        if matches!(
            self.storage,
            LiteralBufferStorage::F64 { .. }
                | LiteralBufferStorage::F64OnePlusXPlusX { .. }
                | LiteralBufferStorage::F32 { .. }
                | LiteralBufferStorage::HalfFloat { .. }
                | LiteralBufferStorage::I64 { .. }
                | LiteralBufferStorage::Bool { .. }
                | LiteralBufferStorage::BoolWords { .. }
                | LiteralBufferStorage::Complex { .. }
                | LiteralBufferStorage::U32 { .. }
                | LiteralBufferStorage::U64 { .. }
                | LiteralBufferStorage::RepeatedPatches { .. }
                | LiteralBufferStorage::Concat { .. }
        ) {
            let elements = self.as_slice().to_vec();
            self.storage = LiteralBufferStorage::Literals(Arc::new(elements));
        }

        match &mut self.storage {
            LiteralBufferStorage::Literals(elements) => Arc::make_mut(elements),
            LiteralBufferStorage::F64 { .. }
            | LiteralBufferStorage::F64OnePlusXPlusX { .. }
            | LiteralBufferStorage::F32 { .. }
            | LiteralBufferStorage::HalfFloat { .. }
            | LiteralBufferStorage::I64 { .. }
            | LiteralBufferStorage::Bool { .. }
            | LiteralBufferStorage::BoolWords { .. }
            | LiteralBufferStorage::Complex { .. }
            | LiteralBufferStorage::U32 { .. }
            | LiteralBufferStorage::U64 { .. }
            | LiteralBufferStorage::RepeatedPatches { .. }
            | LiteralBufferStorage::Concat { .. } => unreachable!("lazy buffer was materialized"),
        }
    }

    pub fn push(&mut self, value: Literal) {
        self.make_mut().push(value);
    }

    pub fn extend_from_slice(&mut self, values: &[Literal]) {
        self.make_mut().extend_from_slice(values);
    }

    pub fn copy_from_slice(&mut self, values: &[Literal]) {
        self.make_mut().as_mut_slice().copy_from_slice(values);
    }

    pub fn sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&Literal, &Literal) -> std::cmp::Ordering,
    {
        self.make_mut().sort_by(compare);
    }
}

impl Default for LiteralBuffer {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl Clone for LiteralBuffer {
    fn clone(&self) -> Self {
        match &self.storage {
            LiteralBufferStorage::Literals(elements) => Self {
                storage: LiteralBufferStorage::Literals(Arc::clone(elements)),
            },
            LiteralBufferStorage::F64 { values, literals } => Self {
                storage: LiteralBufferStorage::F64 {
                    values: Arc::clone(values),
                    literals: Arc::clone(literals),
                },
            },
            LiteralBufferStorage::F64OnePlusXPlusX {
                base,
                values,
                literals,
            } => Self {
                storage: LiteralBufferStorage::F64OnePlusXPlusX {
                    base: Arc::clone(base),
                    values: Arc::clone(values),
                    literals: Arc::clone(literals),
                },
            },
            LiteralBufferStorage::F32 { values, literals } => Self {
                storage: LiteralBufferStorage::F32 {
                    values: Arc::clone(values),
                    literals: Arc::clone(literals),
                },
            },
            LiteralBufferStorage::I64 { values, literals } => Self {
                storage: LiteralBufferStorage::I64 {
                    values: Arc::clone(values),
                    literals: Arc::clone(literals),
                },
            },
            LiteralBufferStorage::Bool { values, literals } => Self {
                storage: LiteralBufferStorage::Bool {
                    values: Arc::clone(values),
                    literals: Arc::clone(literals),
                },
            },
            LiteralBufferStorage::BoolWords {
                words,
                len,
                literals,
            } => Self {
                storage: LiteralBufferStorage::BoolWords {
                    words: Arc::clone(words),
                    len: *len,
                    literals: Arc::clone(literals),
                },
            },
            LiteralBufferStorage::Complex {
                values,
                dtype,
                literals,
            } => Self {
                storage: LiteralBufferStorage::Complex {
                    values: Arc::clone(values),
                    dtype: *dtype,
                    literals: Arc::clone(literals),
                },
            },
            LiteralBufferStorage::HalfFloat {
                values,
                dtype,
                literals,
            } => Self {
                storage: LiteralBufferStorage::HalfFloat {
                    values: Arc::clone(values),
                    dtype: *dtype,
                    literals: Arc::clone(literals),
                },
            },
            LiteralBufferStorage::U32 { values, literals } => Self {
                storage: LiteralBufferStorage::U32 {
                    values: Arc::clone(values),
                    literals: Arc::clone(literals),
                },
            },
            LiteralBufferStorage::U64 { values, literals } => Self {
                storage: LiteralBufferStorage::U64 {
                    values: Arc::clone(values),
                    literals: Arc::clone(literals),
                },
            },
            LiteralBufferStorage::RepeatedPatches {
                base,
                repeats,
                patches,
                literals,
            } => Self {
                storage: LiteralBufferStorage::RepeatedPatches {
                    base: Arc::clone(base),
                    repeats: *repeats,
                    patches: Arc::clone(patches),
                    literals: Arc::clone(literals),
                },
            },
            LiteralBufferStorage::Concat {
                parts,
                len,
                literals,
                f64_values,
                f32_values,
                i64_values,
                half_values,
                complex_values,
                bool_values,
                u32_values,
                u64_values,
            } => Self {
                storage: LiteralBufferStorage::Concat {
                    parts: Arc::clone(parts),
                    len: *len,
                    literals: Arc::clone(literals),
                    f64_values: Arc::clone(f64_values),
                    f32_values: Arc::clone(f32_values),
                    i64_values: Arc::clone(i64_values),
                    half_values: Arc::clone(half_values),
                    complex_values: Arc::clone(complex_values),
                    bool_values: Arc::clone(bool_values),
                    u32_values: Arc::clone(u32_values),
                    u64_values: Arc::clone(u64_values),
                },
            },
        }
    }
}

impl PartialEq for LiteralBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for LiteralBuffer {}

impl std::fmt::Debug for LiteralBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl From<Vec<Literal>> for LiteralBuffer {
    fn from(elements: Vec<Literal>) -> Self {
        Self::new(elements)
    }
}

impl FromIterator<Literal> for LiteralBuffer {
    fn from_iter<T: IntoIterator<Item = Literal>>(iter: T) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

impl Serialize for LiteralBuffer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.as_slice().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for LiteralBuffer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Vec::<Literal>::deserialize(deserializer).map(Self::new)
    }
}

impl Deref for LiteralBuffer {
    type Target = [Literal];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl AsRef<[Literal]> for LiteralBuffer {
    fn as_ref(&self) -> &[Literal] {
        self.as_slice()
    }
}

impl<'a> IntoIterator for &'a LiteralBuffer {
    type Item = &'a Literal;
    type IntoIter = std::slice::Iter<'a, Literal>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl IntoIterator for LiteralBuffer {
    type Item = Literal;
    type IntoIter = std::vec::IntoIter<Literal>;

    fn into_iter(self) -> Self::IntoIter {
        match self.storage {
            LiteralBufferStorage::Literals(elements) => Arc::try_unwrap(elements)
                .unwrap_or_else(|elements| (*elements).clone())
                .into_iter(),
            LiteralBufferStorage::F64 { values, literals } => {
                if let Some(materialized) = literals.get() {
                    return Arc::try_unwrap(Arc::clone(materialized))
                        .unwrap_or_else(|elements| (*elements).clone())
                        .into_iter();
                }

                values
                    .iter()
                    .copied()
                    .map(Literal::from_f64)
                    .collect::<Vec<_>>()
                    .into_iter()
            }
            LiteralBufferStorage::F64OnePlusXPlusX {
                base,
                values,
                literals,
            } => {
                if let Some(materialized) = literals.get() {
                    return Arc::try_unwrap(Arc::clone(materialized))
                        .unwrap_or_else(|elements| (*elements).clone())
                        .into_iter();
                }

                values
                    .get_or_init(|| Arc::new(materialize_f64_one_plus_x_plus_x(&base)))
                    .iter()
                    .copied()
                    .map(Literal::from_f64)
                    .collect::<Vec<_>>()
                    .into_iter()
            }
            LiteralBufferStorage::F32 { values, literals } => {
                if let Some(materialized) = literals.get() {
                    return Arc::try_unwrap(Arc::clone(materialized))
                        .unwrap_or_else(|elements| (*elements).clone())
                        .into_iter();
                }

                values
                    .iter()
                    .copied()
                    .map(Literal::from_f32)
                    .collect::<Vec<_>>()
                    .into_iter()
            }
            LiteralBufferStorage::I64 { values, literals } => {
                if let Some(materialized) = literals.get() {
                    return Arc::try_unwrap(Arc::clone(materialized))
                        .unwrap_or_else(|elements| (*elements).clone())
                        .into_iter();
                }

                values
                    .iter()
                    .copied()
                    .map(Literal::I64)
                    .collect::<Vec<_>>()
                    .into_iter()
            }
            LiteralBufferStorage::Bool { values, literals } => {
                if let Some(materialized) = literals.get() {
                    return Arc::try_unwrap(Arc::clone(materialized))
                        .unwrap_or_else(|elements| (*elements).clone())
                        .into_iter();
                }

                values
                    .iter()
                    .copied()
                    .map(Literal::Bool)
                    .collect::<Vec<_>>()
                    .into_iter()
            }
            LiteralBufferStorage::BoolWords {
                words,
                len,
                literals,
            } => {
                if let Some(materialized) = literals.get() {
                    return Arc::try_unwrap(Arc::clone(materialized))
                        .unwrap_or_else(|elements| (*elements).clone())
                        .into_iter();
                }

                materialize_bool_words(&words, len).into_iter()
            }
            LiteralBufferStorage::Complex {
                values,
                dtype,
                literals,
            } => {
                if let Some(materialized) = literals.get() {
                    return Arc::try_unwrap(Arc::clone(materialized))
                        .unwrap_or_else(|elements| (*elements).clone())
                        .into_iter();
                }

                values
                    .iter()
                    .map(|&(re, im)| complex_pair_to_literal(re, im, dtype))
                    .collect::<Vec<_>>()
                    .into_iter()
            }
            LiteralBufferStorage::HalfFloat {
                values,
                dtype,
                literals,
            } => {
                if let Some(materialized) = literals.get() {
                    return Arc::try_unwrap(Arc::clone(materialized))
                        .unwrap_or_else(|elements| (*elements).clone())
                        .into_iter();
                }

                values
                    .iter()
                    .map(|&bits| half_bits_to_literal(bits, dtype))
                    .collect::<Vec<_>>()
                    .into_iter()
            }
            LiteralBufferStorage::U32 { values, literals } => {
                if let Some(materialized) = literals.get() {
                    return Arc::try_unwrap(Arc::clone(materialized))
                        .unwrap_or_else(|elements| (*elements).clone())
                        .into_iter();
                }

                values
                    .iter()
                    .copied()
                    .map(Literal::U32)
                    .collect::<Vec<_>>()
                    .into_iter()
            }
            LiteralBufferStorage::U64 { values, literals } => {
                if let Some(materialized) = literals.get() {
                    return Arc::try_unwrap(Arc::clone(materialized))
                        .unwrap_or_else(|elements| (*elements).clone())
                        .into_iter();
                }

                values
                    .iter()
                    .copied()
                    .map(Literal::U64)
                    .collect::<Vec<_>>()
                    .into_iter()
            }
            LiteralBufferStorage::RepeatedPatches {
                base,
                repeats,
                patches,
                literals,
            } => {
                if let Some(materialized) = literals.get() {
                    return Arc::try_unwrap(Arc::clone(materialized))
                        .unwrap_or_else(|elements| (*elements).clone())
                        .into_iter();
                }

                materialize_repeated_patches(&base, repeats, &patches).into_iter()
            }
            LiteralBufferStorage::Concat {
                parts,
                len,
                literals,
                ..
            } => {
                if let Some(materialized) = literals.get() {
                    return Arc::try_unwrap(Arc::clone(materialized))
                        .unwrap_or_else(|elements| (*elements).clone())
                        .into_iter();
                }

                materialize_concat_slices(&parts, len).into_iter()
            }
        }
    }
}

impl<I> Index<I> for LiteralBuffer
where
    I: SliceIndex<[Literal]>,
{
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<I> IndexMut<I> for LiteralBuffer
where
    I: SliceIndex<[Literal]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.make_mut()[index]
    }
}

impl PartialEq<Vec<Literal>> for LiteralBuffer {
    fn eq(&self, other: &Vec<Literal>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl PartialEq<LiteralBuffer> for Vec<Literal> {
    fn eq(&self, other: &LiteralBuffer) -> bool {
        self.as_slice() == other.as_slice()
    }
}

/// Materialize one dense complex `(re, im)` pair into a `Literal` of the given
/// complex dtype. For `Complex64` the `f64` inputs are narrowed back to `f32`
/// (bit-identical to the round-trip that produced the dense buffer).
#[inline]
fn complex_pair_to_literal(re: f64, im: f64, dtype: DType) -> Literal {
    match dtype {
        DType::Complex64 => Literal::from_complex64(re as f32, im as f32),
        _ => Literal::from_complex128(re, im),
    }
}

/// Materialize a dense half-float bit pattern into the matching `Literal`. The
/// stored `u16` IS the value's bits, so this is bit-exact.
fn half_bits_to_literal(bits: u16, dtype: DType) -> Literal {
    match dtype {
        DType::F16 => Literal::F16Bits(bits),
        _ => Literal::BF16Bits(bits),
    }
}

fn clear_unused_bool_word_bits(words: &mut [u64], len: usize) {
    let used_bits = len % u64::BITS as usize;
    if used_bits == 0 || words.is_empty() {
        return;
    }
    let mask = (1_u64 << used_bits) - 1;
    if let Some(last) = words.last_mut() {
        *last &= mask;
    }
}

fn materialize_bool_words(words: &[u64], len: usize) -> Vec<Literal> {
    let mut out = Vec::with_capacity(len);
    for index in 0..len {
        let word = words[index / u64::BITS as usize];
        let bit = (word >> (index % u64::BITS as usize)) & 1;
        out.push(Literal::Bool(bit != 0));
    }
    out
}

fn materialize_f64_one_plus_x_plus_x(base: &[f64]) -> Vec<f64> {
    let mut values = Vec::with_capacity(base.len());
    for &x in base {
        let mut value = 1.0_f64;
        value += x;
        value += x;
        values.push(value);
    }
    values
}

fn materialize_repeated_patches(
    base: &[Literal],
    repeats: usize,
    patches: &[(usize, Literal)],
) -> Vec<Literal> {
    let mut elements = Vec::with_capacity(base.len() * repeats);
    for _ in 0..repeats {
        elements.extend_from_slice(base);
    }
    for &(index, literal) in patches {
        if let Some(slot) = elements.get_mut(index) {
            *slot = literal;
        }
    }
    elements
}

fn materialize_concat_slices(parts: &[LiteralBufferSlice], len: usize) -> Vec<Literal> {
    let mut elements = Vec::with_capacity(len);
    for part in parts {
        let end = part.start + part.len;
        elements.extend_from_slice(&part.buffer.as_slice()[part.start..end]);
    }
    elements
}

fn materialize_concat_f64_slices(parts: &[LiteralBufferSlice], len: usize) -> Option<Vec<f64>> {
    let mut values = Vec::with_capacity(len);
    for part in parts {
        let source = part.buffer.as_f64_slice()?;
        let end = part.start.checked_add(part.len)?;
        values.extend_from_slice(source.get(part.start..end)?);
    }
    Some(values)
}

fn materialize_concat_f32_slices(parts: &[LiteralBufferSlice], len: usize) -> Option<Vec<f32>> {
    let mut values = Vec::with_capacity(len);
    for part in parts {
        let source = part.buffer.as_f32_slice()?;
        let end = part.start.checked_add(part.len)?;
        values.extend_from_slice(source.get(part.start..end)?);
    }
    Some(values)
}

fn materialize_concat_i64_slices(parts: &[LiteralBufferSlice], len: usize) -> Option<Vec<i64>> {
    let mut values = Vec::with_capacity(len);
    for part in parts {
        let source = part.buffer.as_i64_slice()?;
        let end = part.start.checked_add(part.len)?;
        values.extend_from_slice(source.get(part.start..end)?);
    }
    Some(values)
}

fn materialize_concat_half_slices(parts: &[LiteralBufferSlice], len: usize) -> Option<Vec<u16>> {
    let mut values = Vec::with_capacity(len);
    for part in parts {
        let source = part.buffer.as_half_float_slice()?;
        let end = part.start.checked_add(part.len)?;
        values.extend_from_slice(source.get(part.start..end)?);
    }
    Some(values)
}

fn materialize_concat_complex_slices(
    parts: &[LiteralBufferSlice],
    len: usize,
) -> Option<Vec<(f64, f64)>> {
    let mut values = Vec::with_capacity(len);
    for part in parts {
        let source = part.buffer.as_complex_slice()?;
        let end = part.start.checked_add(part.len)?;
        values.extend_from_slice(source.get(part.start..end)?);
    }
    Some(values)
}

fn materialize_concat_bool_slices(parts: &[LiteralBufferSlice], len: usize) -> Option<Vec<bool>> {
    let mut values = Vec::with_capacity(len);
    for part in parts {
        let source = part.buffer.as_bool_slice()?;
        let end = part.start.checked_add(part.len)?;
        values.extend_from_slice(source.get(part.start..end)?);
    }
    Some(values)
}

fn materialize_concat_u32_slices(parts: &[LiteralBufferSlice], len: usize) -> Option<Vec<u32>> {
    let mut values = Vec::with_capacity(len);
    for part in parts {
        let source = part.buffer.as_u32_slice()?;
        let end = part.start.checked_add(part.len)?;
        values.extend_from_slice(source.get(part.start..end)?);
    }
    Some(values)
}

fn materialize_concat_u64_slices(parts: &[LiteralBufferSlice], len: usize) -> Option<Vec<u64>> {
    let mut values = Vec::with_capacity(len);
    for part in parts {
        let source = part.buffer.as_u64_slice()?;
        let end = part.start.checked_add(part.len)?;
        values.extend_from_slice(source.get(part.start..end)?);
    }
    Some(values)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorValue {
    pub dtype: DType,
    pub shape: Shape,
    pub elements: LiteralBuffer,
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

        let elements = if matches!(dtype, DType::I32 | DType::I64)
            && elements
                .iter()
                .all(|literal| matches!(literal, Literal::I32(_) | Literal::I64(_)))
        {
            let values = elements
                .into_iter()
                .map(|literal| match literal {
                    Literal::I32(value) => i64::from(value),
                    Literal::I64(value) => value,
                    _ => unreachable!("all elements were checked as signed integers"),
                })
                .collect();
            LiteralBuffer::from_i64_values(values)
        } else if dtype == DType::U32
            && elements
                .iter()
                .all(|literal| matches!(literal, Literal::U32(_)))
        {
            let values = elements
                .into_iter()
                .map(|literal| match literal {
                    Literal::U32(value) => value,
                    _ => unreachable!("all elements were checked as u32"),
                })
                .collect();
            LiteralBuffer::from_u32_values(values)
        } else if dtype == DType::U64
            && elements
                .iter()
                .all(|literal| matches!(literal, Literal::U64(_)))
        {
            let values = elements
                .into_iter()
                .map(|literal| match literal {
                    Literal::U64(value) => value,
                    _ => unreachable!("all elements were checked as u64"),
                })
                .collect();
            LiteralBuffer::from_u64_values(values)
        } else {
            elements.into()
        };

        Ok(Self {
            dtype,
            shape,
            elements,
        })
    }

    pub fn new_with_literal_buffer(
        dtype: DType,
        shape: Shape,
        elements: LiteralBuffer,
    ) -> Result<Self, ValueError> {
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

    pub fn new_f64_values(shape: Shape, values: Vec<f64>) -> Result<Self, ValueError> {
        let expected_count = shape.element_count().ok_or(ValueError::ShapeOverflow {
            shape: shape.clone(),
        })?;

        if expected_count != values.len() as u64 {
            return Err(ValueError::ElementCountMismatch {
                shape,
                expected_count,
                actual_count: values.len(),
            });
        }

        Ok(Self {
            dtype: DType::F64,
            shape,
            elements: LiteralBuffer::from_f64_values(values),
        })
    }

    /// Build an `f32` tensor from dense `f32` values, backed by the packed
    /// [`LiteralBuffer::from_f32_values`] storage (no per-element `Literal`
    /// materialization). Lets f32-heavy ops borrow a `&[f32]` slice and emit
    /// dense outputs.
    pub fn new_f32_values(shape: Shape, values: Vec<f32>) -> Result<Self, ValueError> {
        let expected_count = shape.element_count().ok_or(ValueError::ShapeOverflow {
            shape: shape.clone(),
        })?;

        if expected_count != values.len() as u64 {
            return Err(ValueError::ElementCountMismatch {
                shape,
                expected_count,
                actual_count: values.len(),
            });
        }

        Ok(Self {
            dtype: DType::F32,
            shape,
            elements: LiteralBuffer::from_f32_values(values),
        })
    }

    /// Build a half-float (`BF16`/`F16`) tensor from dense 16-bit values, backed
    /// by the packed [`LiteralBuffer::from_half_float_values`] storage. `dtype`
    /// must be `BF16` or `F16`; the `u16`s are the values' bit patterns.
    pub fn new_half_float_values(
        dtype: DType,
        shape: Shape,
        values: Vec<u16>,
    ) -> Result<Self, ValueError> {
        let expected_count = shape.element_count().ok_or(ValueError::ShapeOverflow {
            shape: shape.clone(),
        })?;

        if expected_count != values.len() as u64 {
            return Err(ValueError::ElementCountMismatch {
                shape,
                expected_count,
                actual_count: values.len(),
            });
        }

        Ok(Self {
            dtype,
            shape,
            elements: LiteralBuffer::from_half_float_values(values, dtype),
        })
    }

    pub fn new_i64_values(shape: Shape, values: Vec<i64>) -> Result<Self, ValueError> {
        let expected_count = shape.element_count().ok_or(ValueError::ShapeOverflow {
            shape: shape.clone(),
        })?;

        if expected_count != values.len() as u64 {
            return Err(ValueError::ElementCountMismatch {
                shape,
                expected_count,
                actual_count: values.len(),
            });
        }

        Ok(Self {
            dtype: DType::I64,
            shape,
            elements: LiteralBuffer::from_i64_values(values),
        })
    }

    pub fn new_i32_values(shape: Shape, values: Vec<i64>) -> Result<Self, ValueError> {
        let expected_count = shape.element_count().ok_or(ValueError::ShapeOverflow {
            shape: shape.clone(),
        })?;

        if expected_count != values.len() as u64 {
            return Err(ValueError::ElementCountMismatch {
                shape,
                expected_count,
                actual_count: values.len(),
            });
        }

        Ok(Self {
            dtype: DType::I32,
            shape,
            elements: LiteralBuffer::from_i64_values(values),
        })
    }

    /// Build a `u32` tensor from dense `u32` values, backed by the packed
    /// [`LiteralBuffer::from_u32_values`] storage (no per-element `Literal`
    /// materialization). Lets `u32`-typed ops (RNG output, bit manipulation)
    /// borrow a `&[u32]` slice (`as_u32_slice`) and emit dense outputs.
    pub fn new_u32_values(shape: Shape, values: Vec<u32>) -> Result<Self, ValueError> {
        let expected_count = shape.element_count().ok_or(ValueError::ShapeOverflow {
            shape: shape.clone(),
        })?;

        if expected_count != values.len() as u64 {
            return Err(ValueError::ElementCountMismatch {
                shape,
                expected_count,
                actual_count: values.len(),
            });
        }

        Ok(Self {
            dtype: DType::U32,
            shape,
            elements: LiteralBuffer::from_u32_values(values),
        })
    }

    /// Build a `u64` tensor from dense `u64` values, backed by the packed
    /// [`LiteralBuffer::from_u64_values`] storage. Mirror of [`Self::new_u32_values`].
    pub fn new_u64_values(shape: Shape, values: Vec<u64>) -> Result<Self, ValueError> {
        let expected_count = shape.element_count().ok_or(ValueError::ShapeOverflow {
            shape: shape.clone(),
        })?;

        if expected_count != values.len() as u64 {
            return Err(ValueError::ElementCountMismatch {
                shape,
                expected_count,
                actual_count: values.len(),
            });
        }

        Ok(Self {
            dtype: DType::U64,
            shape,
            elements: LiteralBuffer::from_u64_values(values),
        })
    }

    pub fn new_bool_values(shape: Shape, values: Vec<bool>) -> Result<Self, ValueError> {
        let expected_count = shape.element_count().ok_or(ValueError::ShapeOverflow {
            shape: shape.clone(),
        })?;

        if expected_count != values.len() as u64 {
            return Err(ValueError::ElementCountMismatch {
                shape,
                expected_count,
                actual_count: values.len(),
            });
        }

        Ok(Self {
            dtype: DType::Bool,
            shape,
            elements: LiteralBuffer::from_bool_values(values),
        })
    }

    /// Build a complex tensor from dense `(re, im)` `f64` pairs, backed by the
    /// packed [`LiteralBuffer::from_complex_values`] storage (no per-element
    /// `Literal` materialization). `dtype` must be `Complex64` or `Complex128`.
    pub fn new_complex_values(
        dtype: DType,
        shape: Shape,
        values: Vec<(f64, f64)>,
    ) -> Result<Self, ValueError> {
        let expected_count = shape.element_count().ok_or(ValueError::ShapeOverflow {
            shape: shape.clone(),
        })?;

        if expected_count != values.len() as u64 {
            return Err(ValueError::ElementCountMismatch {
                shape,
                expected_count,
                actual_count: values.len(),
            });
        }

        Ok(Self {
            dtype,
            shape,
            elements: LiteralBuffer::from_complex_values(values, dtype),
        })
    }

    /// Verify that every element's literal kind agrees with the declared
    /// tensor dtype. Returns the first mismatch as
    /// `ValueError::ElementDTypeMismatch`.
    ///
    /// This is an opt-in invariant check — `TensorValue::new` only validates
    /// element-count, so callers (or tests) can call this to enforce the
    /// stricter dtype/element invariant on demand.
    pub fn validate_dtype_consistency(&self) -> Result<(), ValueError> {
        for (index, literal) in self.elements.iter().enumerate() {
            if !literal.matches_dtype(self.dtype) {
                return Err(ValueError::ElementDTypeMismatch {
                    index,
                    declared: self.dtype,
                    literal: *literal,
                });
            }
        }
        Ok(())
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
            let literal = self.elements[index];
            if self.dtype == DType::I32 {
                let value = literal.as_i64().ok_or(ValueError::ElementDTypeMismatch {
                    index,
                    declared: self.dtype,
                    literal,
                })?;
                return Ok(Value::scalar_i32(value as i32));
            }
            return Ok(Value::Scalar(literal));
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

    pub fn repeat_axis0(value: &Value, repeat_count: usize) -> Result<Self, ValueError> {
        if repeat_count == 0 {
            return Err(ValueError::EmptyAxisStack);
        }

        match value {
            Value::Scalar(lit) => {
                let elements = vec![*lit; repeat_count];
                let dtype = infer_dtype_from_repeated_literal(*lit);
                TensorValue::new(dtype, Shape::vector(repeat_count as u32), elements)
            }
            Value::Tensor(tensor) => {
                let mut dims = Vec::with_capacity(tensor.shape.rank() + 1);
                dims.push(repeat_count as u32);
                dims.extend_from_slice(&tensor.shape.dims);
                let shape = Shape { dims };

                let total_len = tensor.elements.len().checked_mul(repeat_count).ok_or(
                    ValueError::ShapeOverflow {
                        shape: shape.clone(),
                    },
                )?;
                let mut elements = Vec::with_capacity(total_len);
                for _ in 0..repeat_count {
                    elements.extend_from_slice(&tensor.elements);
                }

                TensorValue::new(tensor.dtype, shape, elements)
            }
        }
    }

    pub fn to_f64_vec(&self) -> Option<Vec<f64>> {
        if let Some(values) = self.elements.as_f64_slice() {
            return Some(values.to_vec());
        }

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
    /// `TensorValue::validate_dtype_consistency` found an element whose
    /// literal kind disagrees with the declared tensor dtype.
    ElementDTypeMismatch {
        index: usize,
        declared: DType,
        literal: Literal,
    },
    InvalidArgument {
        op: &'static str,
        reason: &'static str,
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
            Self::ElementDTypeMismatch {
                index,
                declared,
                literal,
            } => {
                write!(
                    f,
                    "tensor element {index} literal {literal:?} disagrees with declared dtype {declared:?}"
                )
            }
            Self::InvalidArgument { op, reason } => {
                write!(f, "{op} invalid argument: {reason}")
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
        .all(|literal| matches!(literal, Literal::I32(_)))
    {
        DType::I32
    } else if elements
        .iter()
        .all(|literal| matches!(literal, Literal::I64(_)))
    {
        DType::I64
    } else if elements
        .iter()
        .all(|literal| matches!(literal, Literal::U32(_)))
    {
        DType::U32
    } else if elements
        .iter()
        .all(|literal| matches!(literal, Literal::U64(_)))
    {
        DType::U64
    } else if elements
        .iter()
        .all(|literal| matches!(literal, Literal::Bool(_)))
    {
        DType::Bool
    } else if elements
        .iter()
        .all(|literal| matches!(literal, Literal::BF16Bits(_)))
    {
        DType::BF16
    } else if elements
        .iter()
        .all(|literal| matches!(literal, Literal::F16Bits(_)))
    {
        DType::F16
    } else if elements
        .iter()
        .all(|literal| matches!(literal, Literal::F32Bits(_)))
    {
        DType::F32
    } else if elements
        .iter()
        .all(|literal| matches!(literal, Literal::Complex64Bits(..)))
    {
        DType::Complex64
    } else if elements
        .iter()
        .all(|literal| matches!(literal, Literal::Complex128Bits(..)))
    {
        DType::Complex128
    } else {
        DType::F64
    }
}

fn infer_dtype_from_repeated_literal(literal: Literal) -> DType {
    match literal {
        Literal::I32(_) => DType::I32,
        Literal::I64(_) => DType::I64,
        Literal::U32(_) => DType::U32,
        Literal::U64(_) => DType::U64,
        Literal::Bool(_) => DType::Bool,
        Literal::BF16Bits(_) => DType::BF16,
        Literal::F16Bits(_) => DType::F16,
        Literal::F32Bits(_) => DType::F32,
        Literal::F64Bits(_) => DType::F64,
        Literal::Complex64Bits(..) => DType::Complex64,
        Literal::Complex128Bits(..) => DType::Complex128,
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<String>,
    /// Nested sub-jaxprs for control-flow primitives.
    /// For Cond: `[true_branch, false_branch]`.
    /// For Scan: `[body]`.
    /// For While: `[cond, body]`.
    /// For Switch: `[branch0, branch1, ...]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sub_jaxprs: Vec<Jaxpr>,
}

pub struct EquationList {
    elements: Arc<Vec<Equation>>,
}

impl EquationList {
    #[must_use]
    pub fn new(equations: Vec<Equation>) -> Self {
        Self {
            elements: Arc::new(equations),
        }
    }

    #[must_use]
    pub fn as_slice(&self) -> &[Equation] {
        self.elements.as_slice()
    }

    #[must_use]
    pub fn into_vec(self) -> Vec<Equation> {
        Arc::try_unwrap(self.elements).unwrap_or_else(|elements| elements.as_ref().clone())
    }

    fn make_mut(&mut self) -> &mut Vec<Equation> {
        Arc::make_mut(&mut self.elements)
    }

    pub fn push(&mut self, equation: Equation) {
        self.make_mut().push(equation);
    }

    pub fn extend<I>(&mut self, equations: I)
    where
        I: IntoIterator<Item = Equation>,
    {
        self.make_mut().extend(equations);
    }
}

impl Default for EquationList {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl Clone for EquationList {
    fn clone(&self) -> Self {
        Self {
            elements: Arc::clone(&self.elements),
        }
    }
}

impl PartialEq for EquationList {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for EquationList {}

impl std::fmt::Debug for EquationList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}

impl From<Vec<Equation>> for EquationList {
    fn from(equations: Vec<Equation>) -> Self {
        Self::new(equations)
    }
}

impl FromIterator<Equation> for EquationList {
    fn from_iter<T: IntoIterator<Item = Equation>>(iter: T) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

impl Serialize for EquationList {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.as_slice().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EquationList {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Vec::<Equation>::deserialize(deserializer).map(Self::new)
    }
}

impl Deref for EquationList {
    type Target = [Equation];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<I> Index<I> for EquationList
where
    I: SliceIndex<[Equation]>,
{
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<I> IndexMut<I> for EquationList
where
    I: SliceIndex<[Equation]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.make_mut()[index]
    }
}

impl IntoIterator for EquationList {
    type Item = Equation;
    type IntoIter = std::vec::IntoIter<Equation>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

impl<'a> IntoIterator for &'a EquationList {
    type Item = &'a Equation;
    type IntoIter = std::slice::Iter<'a, Equation>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a> IntoIterator for &'a mut EquationList {
    type Item = &'a mut Equation;
    type IntoIter = std::slice::IterMut<'a, Equation>;

    fn into_iter(self) -> Self::IntoIter {
        self.make_mut().iter_mut()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Jaxpr {
    pub invars: Vec<VarId>,
    pub constvars: Vec<VarId>,
    pub outvars: Vec<VarId>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub effects: Vec<String>,
    pub equations: EquationList,
    #[serde(skip)]
    fingerprint_cache: std::sync::OnceLock<String>,
}

impl Clone for Jaxpr {
    fn clone(&self) -> Self {
        Self {
            invars: self.invars.clone(),
            constvars: self.constvars.clone(),
            outvars: self.outvars.clone(),
            effects: self.effects.clone(),
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
            && self.effects == other.effects
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
        equations: impl Into<EquationList>,
    ) -> Self {
        Self {
            invars,
            constvars,
            outvars,
            effects: Vec::new(),
            equations: equations.into(),
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
            if !self.effects.is_empty() {
                write_effect_list(&mut out, "effects", &self.effects);
            }

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
                if !eqn.effects.is_empty() {
                    write_effect_list(&mut out, "eqn_effects", &eqn.effects);
                }
                // Nested sub-jaxprs (cond branches, scan/while/switch bodies)
                // carry the actual control-flow computation. Without folding
                // them into the fingerprint, two programs that differ ONLY in a
                // branch/body body collide to the same cache key — returning a
                // stale, wrong cached result. Recurse, length-framed by count so
                // sub-fingerprints cannot run together ambiguously.
                if !eqn.sub_jaxprs.is_empty() {
                    let _ = write!(&mut out, "sub[{}", eqn.sub_jaxprs.len());
                    for sub in &eqn.sub_jaxprs {
                        out.push(':');
                        out.push_str(sub.canonical_fingerprint());
                    }
                    out.push(']');
                }
                out.push('|');
            }

            out
        })
    }

    pub fn validate_well_formed(&self) -> Result<(), JaxprValidationError> {
        let binding_capacity = self.invars.len()
            + self.constvars.len()
            + self
                .equations
                .iter()
                .map(|eqn| eqn.outputs.len())
                .sum::<usize>();
        let mut bindings =
            FxHashSet::with_capacity_and_hasher(binding_capacity, Default::default());

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
            // Recursively validate any sub-jaxprs attached to this equation
            // (cond branches, scan/while bodies, etc.). Sub-jaxprs are
            // independent VarId scopes so this is a structural check on the
            // nested IR — bindings from the parent are NOT propagated.
            for sub_jaxpr in &eqn.sub_jaxprs {
                sub_jaxpr.validate_well_formed()?;
            }
        }

        let mut seen_outvars =
            FxHashSet::with_capacity_and_hasher(self.outvars.len(), Default::default());
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

fn write_effect_list(out: &mut String, label: &str, effects: &[String]) {
    let _ = write!(out, "{label}=[");
    for effect in effects {
        let _ = write!(out, "{effect},");
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
        Literal::I32(value) => {
            let _ = write!(out, "i32:{value}");
        }
        Literal::I64(value) => {
            let _ = write!(out, "i64:{value}");
        }
        Literal::U32(value) => {
            let _ = write!(out, "u32:{value}");
        }
        Literal::U64(value) => {
            let _ = write!(out, "u64:{value}");
        }
        Literal::Bool(value) => {
            let _ = write!(out, "bool:{value}");
        }
        Literal::BF16Bits(value) => {
            let _ = write!(out, "bf16bits:{value}");
        }
        Literal::F16Bits(value) => {
            let _ = write!(out, "f16bits:{value}");
        }
        Literal::F32Bits(value) => {
            let _ = write!(out, "f32bits:{value}");
        }
        Literal::F64Bits(value) => {
            let _ = write!(out, "f64bits:{value}");
        }
        Literal::Complex64Bits(re, im) => {
            let _ = write!(out, "c64:{re},{im}");
        }
        Literal::Complex128Bits(re, im) => {
            let _ = write!(out, "c128:{re},{im}");
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
    // Lax unary primitives (scalar → scalar)
    LaxNeg,
    LaxAbs,
    LaxExp,
    LaxLog,
    LaxSqrt,
    LaxRsqrt,
    LaxFloor,
    LaxCeil,
    LaxRound,
    LaxTan,
    LaxAsin,
    LaxAcos,
    LaxAtan,
    LaxSinh,
    LaxCosh,
    LaxTanh,
    LaxAsinh,
    LaxAcosh,
    LaxAtanh,
    LaxExpm1,
    LaxLog1p,
    LaxSign,
    LaxSquare,
    LaxReciprocal,
    LaxLogistic,
    LaxErf,
    LaxErfc,
    // Lax binary primitives (scalar, scalar → scalar)
    LaxSub,
    LaxMul,
    LaxDiv,
    LaxRem,
    LaxPow,
    LaxAtan2,
    LaxMax,
    LaxMin,
    LaxEq,
    LaxNe,
    LaxLt,
    LaxLe,
    LaxGt,
    LaxGe,
    // Lax ternary primitives
    LaxSelect,
    LaxClamp,
    // Lax reduction primitives (vector → scalar)
    LaxReduceMax,
    LaxReduceMin,
    LaxReduceProd,
    // Lax special math unary primitives
    LaxCbrt,
    LaxLgamma,
    LaxDigamma,
    LaxErfInv,
    LaxIsFinite,
    LaxNextafter,
    // Lax cumulative primitives (vector → vector)
    LaxCumsum,
    LaxCumprod,
    // Lax boolean reduction primitives (vector → scalar)
    LaxReduceAnd,
    LaxReduceOr,
    // Lax bitwise binary primitives (i64, i64 → i64)
    LaxBitwiseAnd,
    LaxBitwiseOr,
    LaxBitwiseXor,
    // Lax bitwise unary primitive (i64 → i64)
    LaxBitwiseNot,
    // Lax integer intrinsics (i64 → i64)
    LaxPopulationCount,
    LaxCountLeadingZeros,
    // Lax boolean reduction: XOR (vector → scalar)
    LaxReduceXor,
    // Lax sorting (vector → vector)
    LaxSort,
    // Integer power (scalar → scalar, exponent in params)
    LaxIntegerPow2,
    LaxIntegerPow3,
    LaxIntegerPowNeg1,
    // Lax shape manipulation primitives (tensor → tensor)
    LaxReshape6To2x3,
    LaxReshape6To3x2,
    LaxSlice1To4,
    LaxTranspose2x3,
    LaxRev,
    LaxSqueeze,
    LaxConcatenate,
    // Lax structural primitives
    LaxIota5,
    LaxCopy,
    LaxExpandDimsAxis0,
    LaxPadLow1High2,
    LaxBroadcastInDimScalar3,
    // Lax bitwise shift primitives (i64, i64 → i64)
    LaxShiftLeft,
    LaxShiftRightArithmetic,
    LaxShiftRightLogical,
    // Lax advanced shape/index primitives
    LaxDynamicSlice,
    LaxDynamicUpdateSlice,
    LaxSplit2,
    LaxBroadcastedIota2x3,
    // Lax advanced primitives
    LaxWhileAddLt,
    LaxSwitch3,
    LaxArgsort,
    LaxOneHot4,
    LaxReduceWindowSum,
    // Lax utility/data primitives
    LaxBitcastF64ToI64,
    LaxReducePrecisionF64,
    LaxGather1d,
    // Lax convolution (1D valid padding)
    LaxConv1dValid,
    // Lax scatter (overwrite mode)
    LaxScatterOverwrite,
    // Lax complex number primitives
    LaxComplex,
    LaxConj,
    LaxReal,
    LaxImag,
    // Linalg primitives
    LaxCholesky,
    LaxTriangularSolve,
    LaxQr,
    LaxSvd,
    LaxEigh,
    // FFT primitives
    LaxFft,
    LaxIfft,
    LaxRfft,
    LaxIrfft,
    // Utility programs for testing
    Identity,
    AddOneMulTwo,
    // Control flow programs
    CondSelect,
    ScanAdd,
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
                effects: vec![],
                sub_jaxprs: vec![],
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
                effects: vec![],
                sub_jaxprs: vec![],
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
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(2))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(2)), Atom::Var(VarId(3))],
                    outputs: smallvec![VarId(4)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
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
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        ),
        ProgramSpec::SinX => unary_program(Primitive::Sin),
        ProgramSpec::CosX => unary_program(Primitive::Cos),
        ProgramSpec::Dot3 => binary_program(Primitive::Dot),
        ProgramSpec::ReduceSumVec => unary_program(Primitive::ReduceSum),
        // Lax unary
        ProgramSpec::LaxNeg => unary_program(Primitive::Neg),
        ProgramSpec::LaxAbs => unary_program(Primitive::Abs),
        ProgramSpec::LaxExp => unary_program(Primitive::Exp),
        ProgramSpec::LaxLog => unary_program(Primitive::Log),
        ProgramSpec::LaxSqrt => unary_program(Primitive::Sqrt),
        ProgramSpec::LaxRsqrt => unary_program(Primitive::Rsqrt),
        ProgramSpec::LaxFloor => unary_program(Primitive::Floor),
        ProgramSpec::LaxCeil => unary_program(Primitive::Ceil),
        ProgramSpec::LaxRound => unary_program(Primitive::Round),
        ProgramSpec::LaxTan => unary_program(Primitive::Tan),
        ProgramSpec::LaxAsin => unary_program(Primitive::Asin),
        ProgramSpec::LaxAcos => unary_program(Primitive::Acos),
        ProgramSpec::LaxAtan => unary_program(Primitive::Atan),
        ProgramSpec::LaxSinh => unary_program(Primitive::Sinh),
        ProgramSpec::LaxCosh => unary_program(Primitive::Cosh),
        ProgramSpec::LaxTanh => unary_program(Primitive::Tanh),
        ProgramSpec::LaxAsinh => unary_program(Primitive::Asinh),
        ProgramSpec::LaxAcosh => unary_program(Primitive::Acosh),
        ProgramSpec::LaxAtanh => unary_program(Primitive::Atanh),
        ProgramSpec::LaxExpm1 => unary_program(Primitive::Expm1),
        ProgramSpec::LaxLog1p => unary_program(Primitive::Log1p),
        ProgramSpec::LaxSign => unary_program(Primitive::Sign),
        ProgramSpec::LaxSquare => unary_program(Primitive::Square),
        ProgramSpec::LaxReciprocal => unary_program(Primitive::Reciprocal),
        ProgramSpec::LaxLogistic => unary_program(Primitive::Logistic),
        ProgramSpec::LaxErf => unary_program(Primitive::Erf),
        ProgramSpec::LaxErfc => unary_program(Primitive::Erfc),
        // Lax binary
        ProgramSpec::LaxSub => binary_program(Primitive::Sub),
        ProgramSpec::LaxMul => binary_program(Primitive::Mul),
        ProgramSpec::LaxDiv => binary_program(Primitive::Div),
        ProgramSpec::LaxRem => binary_program(Primitive::Rem),
        ProgramSpec::LaxPow => binary_program(Primitive::Pow),
        ProgramSpec::LaxAtan2 => binary_program(Primitive::Atan2),
        ProgramSpec::LaxMax => binary_program(Primitive::Max),
        ProgramSpec::LaxMin => binary_program(Primitive::Min),
        ProgramSpec::LaxEq => binary_program(Primitive::Eq),
        ProgramSpec::LaxNe => binary_program(Primitive::Ne),
        ProgramSpec::LaxLt => binary_program(Primitive::Lt),
        ProgramSpec::LaxLe => binary_program(Primitive::Le),
        ProgramSpec::LaxGt => binary_program(Primitive::Gt),
        ProgramSpec::LaxGe => binary_program(Primitive::Ge),
        // Lax ternary
        ProgramSpec::LaxSelect => ternary_program(Primitive::Select),
        ProgramSpec::LaxClamp => ternary_program(Primitive::Clamp),
        // Lax reduction
        ProgramSpec::LaxReduceMax => unary_program(Primitive::ReduceMax),
        ProgramSpec::LaxReduceMin => unary_program(Primitive::ReduceMin),
        ProgramSpec::LaxReduceProd => unary_program(Primitive::ReduceProd),
        // Lax special math unary
        ProgramSpec::LaxCbrt => unary_program(Primitive::Cbrt),
        ProgramSpec::LaxLgamma => unary_program(Primitive::Lgamma),
        ProgramSpec::LaxDigamma => unary_program(Primitive::Digamma),
        ProgramSpec::LaxErfInv => unary_program(Primitive::ErfInv),
        ProgramSpec::LaxIsFinite => unary_program(Primitive::IsFinite),
        ProgramSpec::LaxNextafter => binary_program(Primitive::Nextafter),
        // Lax cumulative (vector → vector)
        ProgramSpec::LaxCumsum => unary_program(Primitive::Cumsum),
        ProgramSpec::LaxCumprod => unary_program(Primitive::Cumprod),
        // Lax boolean reduction (vector → scalar)
        ProgramSpec::LaxReduceAnd => unary_program(Primitive::ReduceAnd),
        ProgramSpec::LaxReduceOr => unary_program(Primitive::ReduceOr),
        // Lax bitwise binary (i64, i64 → i64)
        ProgramSpec::LaxBitwiseAnd => binary_program(Primitive::BitwiseAnd),
        ProgramSpec::LaxBitwiseOr => binary_program(Primitive::BitwiseOr),
        ProgramSpec::LaxBitwiseXor => binary_program(Primitive::BitwiseXor),
        // Lax bitwise unary (i64 → i64)
        ProgramSpec::LaxBitwiseNot => unary_program(Primitive::BitwiseNot),
        ProgramSpec::LaxPopulationCount => unary_program(Primitive::PopulationCount),
        ProgramSpec::LaxCountLeadingZeros => unary_program(Primitive::CountLeadingZeros),
        ProgramSpec::LaxReduceXor => unary_program(Primitive::ReduceXor),
        ProgramSpec::LaxSort => unary_program(Primitive::Sort),
        ProgramSpec::LaxIntegerPow2 => {
            let mut params = BTreeMap::new();
            params.insert("exponent".to_owned(), "2".to_owned());
            Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(2)],
                vec![Equation {
                    primitive: Primitive::IntegerPow,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        ProgramSpec::LaxIntegerPow3 => {
            let mut params = BTreeMap::new();
            params.insert("exponent".to_owned(), "3".to_owned());
            Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(2)],
                vec![Equation {
                    primitive: Primitive::IntegerPow,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        ProgramSpec::LaxIntegerPowNeg1 => {
            let mut params = BTreeMap::new();
            params.insert("exponent".to_owned(), "-1".to_owned());
            Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(2)],
                vec![Equation {
                    primitive: Primitive::IntegerPow,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        ProgramSpec::LaxReshape6To2x3 => {
            let mut params = BTreeMap::new();
            params.insert("new_shape".to_owned(), "2,3".to_owned());
            Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(2)],
                vec![Equation {
                    primitive: Primitive::Reshape,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        ProgramSpec::LaxReshape6To3x2 => {
            let mut params = BTreeMap::new();
            params.insert("new_shape".to_owned(), "3,2".to_owned());
            Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(2)],
                vec![Equation {
                    primitive: Primitive::Reshape,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        ProgramSpec::LaxSlice1To4 => {
            let mut params = BTreeMap::new();
            params.insert("start_indices".to_owned(), "1".to_owned());
            params.insert("limit_indices".to_owned(), "4".to_owned());
            Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(2)],
                vec![Equation {
                    primitive: Primitive::Slice,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        ProgramSpec::LaxTranspose2x3 => {
            // Default transpose (no permutation param) reverses axes
            Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(2)],
                vec![Equation {
                    primitive: Primitive::Transpose,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        ProgramSpec::LaxRev => {
            let mut params = BTreeMap::new();
            params.insert("axes".to_owned(), "0".to_owned());
            Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(2)],
                vec![Equation {
                    primitive: Primitive::Rev,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        ProgramSpec::LaxSqueeze => {
            let mut params = BTreeMap::new();
            params.insert("dimensions".to_owned(), "0".to_owned());
            Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(2)],
                vec![Equation {
                    primitive: Primitive::Squeeze,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        ProgramSpec::LaxConcatenate => {
            // Two-input concatenation along axis 0 (default)
            Jaxpr::new(
                vec![VarId(1), VarId(2)],
                vec![],
                vec![VarId(3)],
                vec![Equation {
                    primitive: Primitive::Concatenate,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        // While loop: init=0, step=1, threshold=5, body_op=add, cond_op=lt → 5
        ProgramSpec::LaxWhileAddLt => {
            let mut params = BTreeMap::new();
            params.insert("body_op".to_owned(), "add".to_owned());
            params.insert("cond_op".to_owned(), "lt".to_owned());
            params.insert("max_iter".to_owned(), "100".to_owned());
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
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        // Switch: index selects among branch values
        ProgramSpec::LaxSwitch3 => {
            let mut params = BTreeMap::new();
            params.insert("num_branches".to_owned(), "3".to_owned());
            Jaxpr::new(
                vec![VarId(1), VarId(2), VarId(3), VarId(4)],
                vec![],
                vec![VarId(5)],
                vec![Equation {
                    primitive: Primitive::Switch,
                    inputs: smallvec![
                        Atom::Var(VarId(1)),
                        Atom::Var(VarId(2)),
                        Atom::Var(VarId(3)),
                        Atom::Var(VarId(4))
                    ],
                    outputs: smallvec![VarId(5)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        // Argsort: vector → indices
        ProgramSpec::LaxArgsort => unary_program(Primitive::Argsort),
        // OneHot: scalar index → vector of length num_classes
        ProgramSpec::LaxOneHot4 => {
            let mut params = BTreeMap::new();
            params.insert("num_classes".to_owned(), "4".to_owned());
            Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(2)],
                vec![Equation {
                    primitive: Primitive::OneHot,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        // ReduceWindow: vector with window_dimensions=3, reduce_op=sum, stride=1
        ProgramSpec::LaxReduceWindowSum => {
            let mut params = BTreeMap::new();
            params.insert("window_dimensions".to_owned(), "3".to_owned());
            params.insert("reduce_op".to_owned(), "sum".to_owned());
            params.insert("window_strides".to_owned(), "1".to_owned());
            Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(2)],
                vec![Equation {
                    primitive: Primitive::ReduceWindow,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        // BitcastConvertType: f64 → i64 (reinterpret bits)
        ProgramSpec::LaxBitcastF64ToI64 => {
            let mut params = BTreeMap::new();
            params.insert("new_dtype".to_owned(), "i64".to_owned());
            Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(2)],
                vec![Equation {
                    primitive: Primitive::BitcastConvertType,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        // ReducePrecision: f64 with reduced mantissa (8 bits)
        ProgramSpec::LaxReducePrecisionF64 => {
            let mut params = BTreeMap::new();
            params.insert("exponent_bits".to_owned(), "11".to_owned());
            params.insert("mantissa_bits".to_owned(), "8".to_owned());
            Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(2)],
                vec![Equation {
                    primitive: Primitive::ReducePrecision,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        // Conv 1D (valid padding): [lhs=[N,W,C_in], rhs=[K,C_in,C_out]] → [N,out_W,C_out]
        ProgramSpec::LaxConv1dValid => {
            let mut params = BTreeMap::new();
            params.insert("padding".to_owned(), "valid".to_owned());
            Jaxpr::new(
                vec![VarId(1), VarId(2)],
                vec![],
                vec![VarId(3)],
                vec![Equation {
                    primitive: Primitive::Conv,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        // Scatter (overwrite): [operand, indices, updates] → scattered tensor
        ProgramSpec::LaxScatterOverwrite => {
            let mut params = BTreeMap::new();
            params.insert("mode".to_owned(), "overwrite".to_owned());
            Jaxpr::new(
                vec![VarId(1), VarId(2), VarId(3)],
                vec![],
                vec![VarId(4)],
                vec![Equation {
                    primitive: Primitive::Scatter,
                    inputs: smallvec![
                        Atom::Var(VarId(1)),
                        Atom::Var(VarId(2)),
                        Atom::Var(VarId(3))
                    ],
                    outputs: smallvec![VarId(4)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        // Gather: [operand, indices] with slice_sizes=1
        ProgramSpec::LaxGather1d => {
            let mut params = BTreeMap::new();
            params.insert("slice_sizes".to_owned(), "1".to_owned());
            Jaxpr::new(
                vec![VarId(1), VarId(2)],
                vec![],
                vec![VarId(3)],
                vec![Equation {
                    primitive: Primitive::Gather,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        // DynamicSlice: [vector, start_idx] → slice of length 3
        ProgramSpec::LaxDynamicSlice => {
            let mut params = BTreeMap::new();
            params.insert("slice_sizes".to_owned(), "3".to_owned());
            Jaxpr::new(
                vec![VarId(1), VarId(2)],
                vec![],
                vec![VarId(3)],
                vec![Equation {
                    primitive: Primitive::DynamicSlice,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        // DynamicUpdateSlice: [operand, update, start_idx] → updated tensor
        ProgramSpec::LaxDynamicUpdateSlice => Jaxpr::new(
            vec![VarId(1), VarId(2), VarId(3)],
            vec![],
            vec![VarId(4)],
            vec![Equation {
                primitive: Primitive::DynamicUpdateSlice,
                inputs: smallvec![
                    Atom::Var(VarId(1)),
                    Atom::Var(VarId(2)),
                    Atom::Var(VarId(3))
                ],
                outputs: smallvec![VarId(4)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        ),
        // Split: vector of 6 into 2 equal sections → reshaped [2,3]
        ProgramSpec::LaxSplit2 => {
            let mut params = BTreeMap::new();
            params.insert("axis".to_owned(), "0".to_owned());
            params.insert("num_sections".to_owned(), "2".to_owned());
            Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(2)],
                vec![Equation {
                    primitive: Primitive::Split,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        // BroadcastedIota: no inputs → 2x3 tensor with iota along dimension 1
        ProgramSpec::LaxBroadcastedIota2x3 => {
            let mut params = BTreeMap::new();
            params.insert("shape".to_owned(), "2,3".to_owned());
            params.insert("dimension".to_owned(), "1".to_owned());
            Jaxpr::new(
                vec![],
                vec![],
                vec![VarId(1)],
                vec![Equation {
                    primitive: Primitive::BroadcastedIota,
                    inputs: smallvec![],
                    outputs: smallvec![VarId(1)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        // Bitwise shifts: binary i64 operations
        ProgramSpec::LaxShiftLeft => binary_program(Primitive::ShiftLeft),
        ProgramSpec::LaxShiftRightArithmetic => binary_program(Primitive::ShiftRightArithmetic),
        ProgramSpec::LaxShiftRightLogical => binary_program(Primitive::ShiftRightLogical),
        // Iota: no inputs, length=5 → [0,1,2,3,4]
        ProgramSpec::LaxIota5 => {
            let mut params = BTreeMap::new();
            params.insert("length".to_owned(), "5".to_owned());
            Jaxpr::new(
                vec![],
                vec![],
                vec![VarId(1)],
                vec![Equation {
                    primitive: Primitive::Iota,
                    inputs: smallvec![],
                    outputs: smallvec![VarId(1)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        // Copy: 1 input, no params, returns clone
        ProgramSpec::LaxCopy => unary_program(Primitive::Copy),
        // ExpandDims: 1 input, axis=0 → inserts leading size-1 dim
        ProgramSpec::LaxExpandDimsAxis0 => {
            let mut params = BTreeMap::new();
            params.insert("axis".to_owned(), "0".to_owned());
            Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(2)],
                vec![Equation {
                    primitive: Primitive::ExpandDims,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        // Pad: 2 inputs (tensor + pad_value), low=1 high=2 interior=0
        ProgramSpec::LaxPadLow1High2 => {
            let mut params = BTreeMap::new();
            params.insert("padding_low".to_owned(), "1".to_owned());
            params.insert("padding_high".to_owned(), "2".to_owned());
            params.insert("padding_interior".to_owned(), "0".to_owned());
            Jaxpr::new(
                vec![VarId(1), VarId(2)],
                vec![],
                vec![VarId(3)],
                vec![Equation {
                    primitive: Primitive::Pad,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                    outputs: smallvec![VarId(3)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        // BroadcastInDim: scalar → vector of length 3
        ProgramSpec::LaxBroadcastInDimScalar3 => {
            let mut params = BTreeMap::new();
            params.insert("shape".to_owned(), "3".to_owned());
            Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(2)],
                vec![Equation {
                    primitive: Primitive::BroadcastInDim,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        // Complex: (real, imag) → complex128
        ProgramSpec::LaxComplex => binary_program(Primitive::Complex),
        // Conj: complex → complex (negate imag)
        ProgramSpec::LaxConj => unary_program(Primitive::Conj),
        // Real: complex → f64 (extract real part)
        ProgramSpec::LaxReal => unary_program(Primitive::Real),
        // Imag: complex → f64 (extract imaginary part)
        ProgramSpec::LaxImag => unary_program(Primitive::Imag),
        // Linalg: Cholesky — single matrix input → lower-triangular factor
        ProgramSpec::LaxCholesky => unary_program(Primitive::Cholesky),
        // Linalg: TriangularSolve — [A, B] → X where A X = B, A lower-triangular
        ProgramSpec::LaxTriangularSolve => Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3)],
            vec![Equation {
                primitive: Primitive::TriangularSolve,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        ),
        // Linalg: QR — A → [Q, R] where A = Q R
        ProgramSpec::LaxQr => Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2), VarId(3)],
            vec![Equation {
                primitive: Primitive::Qr,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2), VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        ),
        // Linalg: SVD — A → [U, S, Vt] where A = U diag(S) Vt
        ProgramSpec::LaxSvd => Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2), VarId(3), VarId(4)],
            vec![Equation {
                primitive: Primitive::Svd,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2), VarId(3), VarId(4)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        ),
        // Linalg: Eigh — A → [W, V] where A = V diag(W) V^T
        ProgramSpec::LaxEigh => Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2), VarId(3)],
            vec![Equation {
                primitive: Primitive::Eigh,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2), VarId(3)],
                params: BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        ),
        // FFT: 1D DFT along last axis
        ProgramSpec::LaxFft => unary_program(Primitive::Fft),
        // IFFT: 1D inverse DFT along last axis
        ProgramSpec::LaxIfft => unary_program(Primitive::Ifft),
        // RFFT: real-to-complex FFT (needs fft_length param)
        ProgramSpec::LaxRfft => {
            let mut params = BTreeMap::new();
            params.insert("fft_length".to_owned(), "8".to_owned());
            Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(2)],
                vec![Equation {
                    primitive: Primitive::Rfft,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        // IRFFT: complex-to-real inverse FFT (needs fft_length param)
        ProgramSpec::LaxIrfft => {
            let mut params = BTreeMap::new();
            params.insert("fft_length".to_owned(), "8".to_owned());
            Jaxpr::new(
                vec![VarId(1)],
                vec![],
                vec![VarId(2)],
                vec![Equation {
                    primitive: Primitive::Irfft,
                    inputs: smallvec![Atom::Var(VarId(1))],
                    outputs: smallvec![VarId(2)],
                    params,
                    effects: vec![],
                    sub_jaxprs: vec![],
                }],
            )
        }
        // Utility programs
        ProgramSpec::Identity => Jaxpr::new(vec![VarId(1)], vec![], vec![VarId(1)], vec![]),
        ProgramSpec::AddOneMulTwo => Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2), VarId(3)],
            vec![
                Equation {
                    primitive: Primitive::Add,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(1))],
                    outputs: smallvec![VarId(2)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
                Equation {
                    primitive: Primitive::Mul,
                    inputs: smallvec![Atom::Var(VarId(1)), Atom::Lit(Literal::I64(2))],
                    outputs: smallvec![VarId(3)],
                    params: BTreeMap::new(),
                    effects: vec![],
                    sub_jaxprs: vec![],
                },
            ],
        ),
        // cond(pred, on_true, on_false) -> selected
        ProgramSpec::CondSelect => Jaxpr::new(
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
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        ),
        // scan(add, init, xs) -> carry (cumulative sum)
        ProgramSpec::ScanAdd => Jaxpr::new(
            vec![VarId(1), VarId(2)],
            vec![],
            vec![VarId(3)],
            vec![Equation {
                primitive: Primitive::Scan,
                inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                outputs: smallvec![VarId(3)],
                params: BTreeMap::from([("body_op".to_owned(), "add".to_owned())]),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        ),
    }
}

fn unary_program(primitive: Primitive) -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1)],
        vec![],
        vec![VarId(2)],
        vec![Equation {
            primitive,
            inputs: smallvec![Atom::Var(VarId(1))],
            outputs: smallvec![VarId(2)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    )
}

fn binary_program(primitive: Primitive) -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1), VarId(2)],
        vec![],
        vec![VarId(3)],
        vec![Equation {
            primitive,
            inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
            outputs: smallvec![VarId(3)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    )
}

fn ternary_program(primitive: Primitive) -> Jaxpr {
    Jaxpr::new(
        vec![VarId(1), VarId(2), VarId(3)],
        vec![],
        vec![VarId(4)],
        vec![Equation {
            primitive,
            inputs: smallvec![
                Atom::Var(VarId(1)),
                Atom::Var(VarId(2)),
                Atom::Var(VarId(3))
            ],
            outputs: smallvec![VarId(4)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        }],
    )
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
        composition_signature_for(
            &self.root_jaxpr,
            &self.transform_stack,
            &self.transform_evidence,
        )
    }
}

#[must_use]
pub fn composition_signature_for(
    root_jaxpr: &Jaxpr,
    transform_stack: &[Transform],
    transform_evidence: &[String],
) -> String {
    let mut out = String::new();
    out.push_str("stack=");
    for transform in transform_stack {
        let _ = write!(&mut out, "{}>", transform.as_str());
    }
    out.push_str("|evidence=");
    for (transform, evidence) in transform_stack.iter().zip(transform_evidence.iter()) {
        let _ = write!(
            &mut out,
            "{}:{}:{};",
            transform.as_str(),
            evidence.len(),
            evidence
        );
    }
    out.push_str("|jaxpr=");
    out.push_str(root_jaxpr.canonical_fingerprint());
    out
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
    EvidenceTransformMismatch {
        index: usize,
        transform: Transform,
        evidence: String,
    },
    DuplicateEvidence {
        index: usize,
        evidence: String,
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
            Self::EvidenceTransformMismatch {
                index,
                transform,
                evidence,
            } => {
                write!(
                    f,
                    "transform evidence at index {} does not bind to {}: {}",
                    index,
                    transform.as_str(),
                    evidence
                )
            }
            Self::DuplicateEvidence { index, evidence } => {
                write!(
                    f,
                    "transform evidence at index {} duplicates an earlier evidence id: {}",
                    index, evidence
                )
            }
        }
    }
}

impl std::error::Error for TransformCompositionError {}

pub fn verify_transform_composition(
    ledger: &TraceTransformLedger,
) -> Result<TransformCompositionProof, TransformCompositionError> {
    verify_transform_composition_parts(
        &ledger.root_jaxpr,
        &ledger.transform_stack,
        &ledger.transform_evidence,
    )
}

pub fn verify_transform_composition_parts(
    root_jaxpr: &Jaxpr,
    transform_stack: &[Transform],
    transform_evidence: &[String],
) -> Result<TransformCompositionProof, TransformCompositionError> {
    if transform_stack.len() != transform_evidence.len() {
        return Err(TransformCompositionError::EvidenceCountMismatch {
            transform_count: transform_stack.len(),
            evidence_count: transform_evidence.len(),
        });
    }

    let mut seen_evidence = BTreeSet::new();
    for (index, transform) in transform_stack.iter().enumerate() {
        let evidence = transform_evidence[index].trim();
        if evidence.is_empty() {
            return Err(TransformCompositionError::EmptyEvidence {
                index,
                transform: *transform,
            });
        }
        if !evidence_mentions_transform(evidence, *transform) {
            return Err(TransformCompositionError::EvidenceTransformMismatch {
                index,
                transform: *transform,
                evidence: evidence.to_owned(),
            });
        }
        if !seen_evidence.insert(evidence) {
            return Err(TransformCompositionError::DuplicateEvidence {
                index,
                evidence: evidence.to_owned(),
            });
        }
    }

    let stack_signature =
        composition_signature_for(root_jaxpr, transform_stack, transform_evidence);
    let stack_hash_hex = format!("{:016x}", fnv1a_64(stack_signature.as_bytes()));

    Ok(TransformCompositionProof {
        stack_signature,
        stack_hash_hex,
        transform_count: transform_stack.len(),
        evidence_count: transform_evidence.len(),
    })
}

fn evidence_mentions_transform(evidence: &str, transform: Transform) -> bool {
    let expected = transform.as_str();
    evidence
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .any(|part| part.eq_ignore_ascii_case(expected))
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
        Atom, DType, Equation, Jaxpr, JaxprValidationError, Literal, LiteralBuffer, Primitive,
        ProgramSpec, Shape, TensorValue, TraceTransformLedger, Transform, Value, ValueError, VarId,
        build_program, verify_transform_composition, verify_transform_composition_parts,
    };
    use proptest::prelude::*;
    use proptest::test_runner::{Config as ProptestConfig, TestCaseError, TestRunner};
    use serde::Serialize;
    use serde_json::json;
    use smallvec::smallvec;
    use std::any::Any;
    use std::collections::BTreeMap;
    use std::fmt::Write;
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

        let validation_start = Instant::now();
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
        log.phase_timings.verify_ms = duration_ms(validation_start);

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
            std::panic::panic_any(detail);
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
            effects: vec![],
            sub_jaxprs: vec![],
        });

        let mut slice_params = BTreeMap::new();
        slice_params.insert("start_indices".to_owned(), "0,0".to_owned());
        slice_params.insert("limit_indices".to_owned(), "2,2".to_owned());
        equations.push(Equation {
            primitive: Primitive::Slice,
            inputs: smallvec![Atom::Var(VarId(5))],
            outputs: smallvec![VarId(6)],
            params: slice_params,
            effects: vec![],
            sub_jaxprs: vec![],
        });

        let mut gather_params = BTreeMap::new();
        gather_params.insert("slice_sizes".to_owned(), "1,2".to_owned());
        equations.push(Equation {
            primitive: Primitive::Gather,
            inputs: smallvec![Atom::Var(VarId(5)), Atom::Var(VarId(2))],
            outputs: smallvec![VarId(7)],
            params: gather_params,
            effects: vec![],
            sub_jaxprs: vec![],
        });

        let mut transpose_params = BTreeMap::new();
        transpose_params.insert("permutation".to_owned(), "1,0".to_owned());
        equations.push(Equation {
            primitive: Primitive::Transpose,
            inputs: smallvec![Atom::Var(VarId(5))],
            outputs: smallvec![VarId(8)],
            params: transpose_params,
            effects: vec![],
            sub_jaxprs: vec![],
        });

        let mut broadcast_params = BTreeMap::new();
        broadcast_params.insert("shape".to_owned(), "3,2,2".to_owned());
        broadcast_params.insert("broadcast_dimensions".to_owned(), "1,2".to_owned());
        equations.push(Equation {
            primitive: Primitive::BroadcastInDim,
            inputs: smallvec![Atom::Var(VarId(6))],
            outputs: smallvec![VarId(9)],
            params: broadcast_params,
            effects: vec![],
            sub_jaxprs: vec![],
        });

        let mut concat_params = BTreeMap::new();
        concat_params.insert("dimension".to_owned(), "0".to_owned());
        equations.push(Equation {
            primitive: Primitive::Concatenate,
            inputs: smallvec![Atom::Var(VarId(8)), Atom::Var(VarId(8))],
            outputs: smallvec![VarId(10)],
            params: concat_params,
            effects: vec![],
            sub_jaxprs: vec![],
        });

        let mut pad_params = BTreeMap::new();
        pad_params.insert("padding_low".to_owned(), "1,0".to_owned());
        pad_params.insert("padding_high".to_owned(), "0,1".to_owned());
        pad_params.insert("padding_interior".to_owned(), "0,1".to_owned());
        equations.push(Equation {
            primitive: Primitive::Pad,
            inputs: smallvec![Atom::Var(VarId(10)), Atom::Var(VarId(4))],
            outputs: smallvec![VarId(18)],
            params: pad_params,
            effects: vec![],
            sub_jaxprs: vec![],
        });

        equations.push(Equation {
            primitive: Primitive::Scatter,
            inputs: smallvec![Atom::Var(VarId(18)), Atom::Var(VarId(2))],
            outputs: smallvec![VarId(11)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        });

        let mut reduce_params = BTreeMap::new();
        reduce_params.insert("axes".to_owned(), "0".to_owned());
        equations.push(Equation {
            primitive: Primitive::ReduceSum,
            inputs: smallvec![Atom::Var(VarId(11))],
            outputs: smallvec![VarId(12)],
            params: reduce_params,
            effects: vec![],
            sub_jaxprs: vec![],
        });

        equations.push(Equation {
            primitive: Primitive::Sin,
            inputs: smallvec![Atom::Var(VarId(12))],
            outputs: smallvec![VarId(13)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        });
        equations.push(Equation {
            primitive: Primitive::Cos,
            inputs: smallvec![Atom::Var(VarId(13))],
            outputs: smallvec![VarId(14)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        });
        equations.push(Equation {
            primitive: Primitive::Add,
            inputs: smallvec![Atom::Var(VarId(14)), Atom::Var(VarId(12))],
            outputs: smallvec![VarId(15)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        });
        equations.push(Equation {
            primitive: Primitive::Mul,
            inputs: smallvec![Atom::Var(VarId(15)), Atom::Var(VarId(15))],
            outputs: smallvec![VarId(16)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        });
        equations.push(Equation {
            primitive: Primitive::Dot,
            inputs: smallvec![Atom::Var(VarId(8)), Atom::Var(VarId(3))],
            outputs: smallvec![VarId(17)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
        });
        equations.push(Equation {
            primitive: Primitive::Add,
            inputs: smallvec![Atom::Var(VarId(16)), Atom::Var(VarId(17))],
            outputs: smallvec![VarId(24)],
            params: BTreeMap::new(),
            effects: vec![],
            sub_jaxprs: vec![],
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
                    Primitive::Pad,
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
                        effects: vec![],
                        sub_jaxprs: vec![],
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
    fn validate_well_formed_recurses_into_sub_jaxprs() {
        run_logged_test(
            "validate_well_formed_recurses_into_sub_jaxprs",
            &("sub-jaxpr-validation", 1_u32),
            fj_test_utils::TestMode::Strict,
            || {
                // Build a malformed sub_jaxpr (duplicate invar).
                let malformed_sub =
                    Jaxpr::new(vec![VarId(1), VarId(1)], vec![], vec![VarId(1)], vec![]);

                // Parent is otherwise well-formed: a single Cond equation
                // whose `sub_jaxprs` contains the malformed branch.
                let parent = Jaxpr::new(
                    vec![VarId(10)],
                    vec![],
                    vec![VarId(11)],
                    vec![Equation {
                        primitive: Primitive::Cond,
                        inputs: smallvec![Atom::Var(VarId(10))],
                        outputs: smallvec![VarId(11)],
                        params: BTreeMap::new(),
                        effects: vec![],
                        sub_jaxprs: vec![malformed_sub],
                    }],
                );

                let err = parent
                    .validate_well_formed()
                    .expect_err("nested malformed sub_jaxpr should propagate");
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
    fn validate_well_formed_accepts_well_formed_sub_jaxprs() {
        run_logged_test(
            "validate_well_formed_accepts_well_formed_sub_jaxprs",
            &("sub-jaxpr-validation-ok", 1_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let body = Jaxpr::new(vec![VarId(1)], vec![], vec![VarId(1)], vec![]);

                let parent = Jaxpr::new(
                    vec![VarId(10)],
                    vec![],
                    vec![VarId(11)],
                    vec![Equation {
                        primitive: Primitive::Cond,
                        inputs: smallvec![Atom::Var(VarId(10))],
                        outputs: smallvec![VarId(11)],
                        params: BTreeMap::new(),
                        effects: vec![],
                        sub_jaxprs: vec![body],
                    }],
                );

                parent
                    .validate_well_formed()
                    .expect("well-formed parent + sub_jaxpr should pass");
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
    fn jaxpr_equation_storage_clone_isolation_and_plain_serde() {
        run_logged_test(
            "jaxpr_equation_storage_clone_isolation_and_plain_serde",
            &("equation-storage-cow-serde", 3_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let original = build_program(ProgramSpec::SquarePlusLinear);
                let original_fingerprint = original.canonical_fingerprint().to_owned();

                let mut modified = original.clone();
                modified.equations[0]
                    .params
                    .insert("cow_mutation".to_owned(), "1".to_owned());

                assert!(!original.equations[0].params.contains_key("cow_mutation"));
                assert!(modified.equations[0].params.contains_key("cow_mutation"));
                assert_eq!(original.canonical_fingerprint(), original_fingerprint);
                assert_ne!(
                    original.canonical_fingerprint(),
                    modified.canonical_fingerprint()
                );

                let encoded = serde_json::to_value(&original)
                    .map_err(|err| format!("to_value failed: {err}"))?;
                assert!(encoded["equations"].is_array());
                assert!(!encoded.to_string().contains("EquationList"));

                let decoded: Jaxpr = serde_json::from_value(encoded)
                    .map_err(|err| format!("decode failed: {err}"))?;
                assert_eq!(decoded.equations, original.equations);
                assert_eq!(
                    decoded.canonical_fingerprint(),
                    original.canonical_fingerprint()
                );

                let debug = format!("{original:?}");
                assert!(debug.contains("equations: ["));
                assert!(!debug.contains("EquationList"));
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn jaxpr_effects_default_empty() {
        run_logged_test(
            "jaxpr_effects_default_empty",
            &("effects-default-empty", 1_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = build_program(ProgramSpec::Square);
                assert!(jaxpr.effects.is_empty());
                assert!(jaxpr.equations.iter().all(|eqn| eqn.effects.is_empty()));
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn jaxpr_effects_serde_skip_when_empty_and_backcompat() {
        run_logged_test(
            "jaxpr_effects_serde_skip_when_empty_and_backcompat",
            &("effects-serde-backcompat", 1_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let jaxpr = build_program(ProgramSpec::SquarePlusLinear);
                let encoded = serde_json::to_string(&jaxpr)
                    .map_err(|err| format!("serialize failed: {err}"))?;
                assert!(!encoded.contains("\"effects\":"));

                let mut value = serde_json::to_value(&jaxpr)
                    .map_err(|err| format!("to_value failed: {err}"))?;
                if let serde_json::Value::Object(map) = &mut value {
                    map.remove("effects");
                    if let Some(serde_json::Value::Array(eqns)) = map.get_mut("equations") {
                        for eqn in eqns {
                            if let serde_json::Value::Object(eq_map) = eqn {
                                eq_map.remove("effects");
                            }
                        }
                    }
                }
                let decoded: Jaxpr =
                    serde_json::from_value(value).map_err(|err| format!("decode failed: {err}"))?;
                assert!(decoded.effects.is_empty());
                assert!(decoded.equations.iter().all(|eqn| eqn.effects.is_empty()));
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn jaxpr_effects_serde_roundtrip_with_non_empty_effects() {
        run_logged_test(
            "jaxpr_effects_serde_roundtrip_with_non_empty_effects",
            &("effects-serde-roundtrip", 2_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let mut jaxpr = build_program(ProgramSpec::Square);
                jaxpr.effects.push("global_io".to_owned());
                jaxpr.equations[0].effects.push("eqn_debug".to_owned());

                let encoded = serde_json::to_string(&jaxpr)
                    .map_err(|err| format!("serialize failed: {err}"))?;
                assert!(encoded.contains("\"effects\":[\"global_io\"]"));
                assert!(encoded.contains("\"effects\":[\"eqn_debug\"]"));

                let decoded: Jaxpr = serde_json::from_str(&encoded)
                    .map_err(|err| format!("decode failed: {err}"))?;
                assert_eq!(decoded.effects, vec!["global_io".to_owned()]);
                assert_eq!(decoded.equations[0].effects, vec!["eqn_debug".to_owned()]);
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn canonical_fingerprint_changes_when_effects_change() {
        run_logged_test(
            "canonical_fingerprint_changes_when_effects_change",
            &("effects-fingerprint", 2_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let base = build_program(ProgramSpec::SquarePlusLinear);

                let mut with_jaxpr_effect = base.clone();
                with_jaxpr_effect.effects.push("global_io".to_owned());
                assert_ne!(
                    base.canonical_fingerprint(),
                    with_jaxpr_effect.canonical_fingerprint()
                );

                let mut with_eqn_effect = base.clone();
                with_eqn_effect.equations[0]
                    .effects
                    .push("eqn_debug".to_owned());
                assert_ne!(
                    base.canonical_fingerprint(),
                    with_eqn_effect.canonical_fingerprint()
                );
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn canonical_fingerprint_changes_when_sub_jaxpr_body_changes() {
        run_logged_test(
            "canonical_fingerprint_changes_when_sub_jaxpr_body_changes",
            &("sub-jaxpr-fingerprint", 1_u32),
            fj_test_utils::TestMode::Strict,
            || {
                // A single-equation sub-jaxpr computing `prim(v1) -> v2`.
                fn unary_sub(prim: Primitive) -> Jaxpr {
                    Jaxpr::new(
                        vec![VarId(1)],
                        vec![],
                        vec![VarId(2)],
                        vec![Equation {
                            primitive: prim,
                            inputs: smallvec![Atom::Var(VarId(1))],
                            outputs: smallvec![VarId(2)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
                        }],
                    )
                }
                // A parent `switch` whose only varying part is its branch body.
                fn switch_with_branch(branch: Jaxpr) -> Jaxpr {
                    Jaxpr::new(
                        vec![VarId(1), VarId(2)],
                        vec![],
                        vec![VarId(3)],
                        vec![Equation {
                            primitive: Primitive::Switch,
                            inputs: smallvec![Atom::Var(VarId(1)), Atom::Var(VarId(2))],
                            outputs: smallvec![VarId(3)],
                            params: BTreeMap::from([("num_branches".to_owned(), "1".to_owned())]),
                            effects: vec![],
                            sub_jaxprs: vec![branch],
                        }],
                    )
                }

                let with_neg = switch_with_branch(unary_sub(Primitive::Neg));
                let with_abs = switch_with_branch(unary_sub(Primitive::Abs));

                // Regression: parents are structurally identical except for the
                // branch body. Before folding sub_jaxprs into the fingerprint
                // these collided to the same cache key.
                assert_ne!(
                    with_neg.canonical_fingerprint(),
                    with_abs.canonical_fingerprint(),
                    "differing branch bodies must yield different fingerprints"
                );

                // Determinism: identical branch bodies still match.
                let with_neg_again = switch_with_branch(unary_sub(Primitive::Neg));
                assert_eq!(
                    with_neg.canonical_fingerprint(),
                    with_neg_again.canonical_fingerprint()
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
    fn transform_composition_parts_match_owned_ledger_proof() {
        run_logged_test(
            "transform_composition_parts_match_owned_ledger_proof",
            &("transform-parts-proof", 1_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ttl = TraceTransformLedger::new(build_program(ProgramSpec::Square));
                ttl.push_transform(Transform::Jit, "evidence-jit");
                ttl.push_transform(Transform::Grad, "evidence-grad");

                let owned = verify_transform_composition(&ttl)
                    .map_err(|err| format!("owned proof should validate: {err}"))?;
                let borrowed = verify_transform_composition_parts(
                    &ttl.root_jaxpr,
                    &ttl.transform_stack,
                    &ttl.transform_evidence,
                )
                .map_err(|err| format!("borrowed proof should validate: {err}"))?;
                assert_eq!(owned, borrowed);
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
                    ttl.push_transform(stack[0], format!("evidence-{}-0", stack[0].as_str()));
                    ttl.push_transform(stack[1], format!("evidence-{}-1", stack[1].as_str()));
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
                ttl.push_transform(Transform::Grad, "grad-1");
                ttl.push_transform(Transform::Grad, "grad-2");
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
                ttl.push_transform(Transform::Vmap, "vmap-1");
                ttl.push_transform(Transform::Vmap, "vmap-2");
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
    fn transform_composition_rejects_transform_mismatched_evidence() {
        run_logged_test(
            "transform_composition_rejects_transform_mismatched_evidence",
            &("mismatched-evidence", 1_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ttl = TraceTransformLedger::new(build_program(ProgramSpec::Square));
                ttl.push_transform(Transform::Jit, "evidence-grad-0");
                let err = verify_transform_composition(&ttl)
                    .expect_err("evidence must bind to its transform");
                assert!(format!("{err}").contains("does not bind to jit"));
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn transform_composition_rejects_duplicate_evidence_ids() {
        run_logged_test(
            "transform_composition_rejects_duplicate_evidence_ids",
            &("duplicate-evidence", 2_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let mut ttl = TraceTransformLedger::new(build_program(ProgramSpec::Square));
                ttl.push_transform(Transform::Grad, "grad");
                ttl.push_transform(Transform::Grad, "grad");
                let err = verify_transform_composition(&ttl)
                    .expect_err("duplicate evidence ids should fail");
                assert!(format!("{err}").contains("duplicates an earlier evidence id"));
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn transform_composition_signature_changes_when_evidence_changes() {
        run_logged_test(
            "transform_composition_signature_changes_when_evidence_changes",
            &("evidence-bound-signature", 2_u32),
            fj_test_utils::TestMode::Strict,
            || {
                let mut lhs = TraceTransformLedger::new(build_program(ProgramSpec::Square));
                lhs.push_transform(Transform::Jit, "jit-evidence-a");
                let mut rhs = TraceTransformLedger::new(build_program(ProgramSpec::Square));
                rhs.push_transform(Transform::Jit, "jit-evidence-b");
                assert_ne!(lhs.composition_signature(), rhs.composition_signature());
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
    fn repeat_axis0_matches_repeated_stack_for_scalars_and_tensors() {
        let scalar = Value::scalar_i64(7);
        let repeated_scalar =
            TensorValue::repeat_axis0(&scalar, 3).expect("scalar repeat should succeed");
        let stacked_scalar = TensorValue::stack_axis0(&[scalar.clone(), scalar.clone(), scalar])
            .expect("scalar stack should succeed");
        assert_eq!(repeated_scalar, stacked_scalar);

        let vector = Value::vector_i64(&[1, 2]).expect("vector should build");
        let repeated_vector =
            TensorValue::repeat_axis0(&vector, 3).expect("tensor repeat should succeed");
        assert_eq!(
            repeated_vector,
            TensorValue::stack_axis0(&[vector.clone(), vector.clone(), vector])
                .expect("tensor stack should succeed")
        );

        let empty_repeat = TensorValue::repeat_axis0(&Value::scalar_i64(0), 0)
            .expect_err("empty repeat should preserve stack's empty-axis error");
        assert!(matches!(empty_repeat, ValueError::EmptyAxisStack));
    }

    #[test]
    fn validate_dtype_consistency_accepts_matching_tensor() {
        let tensor = TensorValue::new(
            DType::Complex64,
            Shape::vector(2),
            vec![
                Literal::from_complex64(1.0, 0.0),
                Literal::from_complex64(2.0, 0.0),
            ],
        )
        .expect("complex64 tensor builds");
        assert!(tensor.validate_dtype_consistency().is_ok());
    }

    #[test]
    fn validate_dtype_consistency_rejects_mismatched_tensor() {
        // Construct a tensor declaring DType::F32 but containing F64Bits.
        // This is exactly the invariant-violation shape that bugs like
        // eldm/2chb/1x85/e8g4 surfaced.
        let tensor = TensorValue::new(
            DType::F32,
            Shape::vector(2),
            vec![Literal::from_f64(1.0), Literal::from_f64(2.0)],
        )
        .expect("element count matches, count check passes");
        let mismatch = tensor.validate_dtype_consistency();
        assert!(matches!(
            mismatch,
            Err(ValueError::ElementDTypeMismatch { .. })
        ));
        let Err(ValueError::ElementDTypeMismatch {
            index,
            declared,
            literal,
        }) = mismatch
        else {
            return;
        };
        assert_eq!(index, 0);
        assert_eq!(declared, DType::F32);
        assert!(matches!(literal, Literal::F64Bits(_)));
    }

    #[test]
    fn stack_axis0_preserves_complex_scalar_dtype() {
        // Regression test for frankenjax-znx7: infer_dtype_from_literals
        // previously fell through to DType::F64 for all-Complex literal lists,
        // so stack_axis0 of complex scalars produced a tensor declaring F64
        // while containing Complex elements.
        let c1 = Value::Scalar(Literal::from_complex64(1.0, 0.0));
        let c2 = Value::Scalar(Literal::from_complex64(2.0, 0.0));
        let stacked = TensorValue::stack_axis0(&[c1, c2]).expect("complex64 stack");
        assert_eq!(stacked.dtype, DType::Complex64);
        stacked
            .validate_dtype_consistency()
            .expect("complex64 stack dtype invariant");

        let d1 = Value::Scalar(Literal::from_complex128(1.0, 0.0));
        let d2 = Value::Scalar(Literal::from_complex128(2.0, 0.0));
        let stacked128 = TensorValue::stack_axis0(&[d1, d2]).expect("complex128 stack");
        assert_eq!(stacked128.dtype, DType::Complex128);
        stacked128
            .validate_dtype_consistency()
            .expect("complex128 stack dtype invariant");
    }

    #[test]
    fn repeat_axis0_preserves_complex_scalar_dtype() {
        // Regression test for frankenjax-e8g4: infer_dtype_from_repeated_literal
        // previously mapped Complex64Bits / Complex128Bits to DType::F64, so
        // repeat_axis0 on a complex scalar produced a tensor declaring F64
        // while containing Complex literals (invariant violation).
        let c64 = Value::Scalar(Literal::from_complex64(1.0, 2.0));
        let rep = TensorValue::repeat_axis0(&c64, 3).expect("complex64 repeat");
        assert_eq!(rep.dtype, DType::Complex64);
        rep.validate_dtype_consistency()
            .expect("complex64 repeat dtype invariant");

        let c128 = Value::Scalar(Literal::from_complex128(1.0, 2.0));
        let rep128 = TensorValue::repeat_axis0(&c128, 2).expect("complex128 repeat");
        assert_eq!(rep128.dtype, DType::Complex128);
        rep128
            .validate_dtype_consistency()
            .expect("complex128 repeat dtype invariant");
    }

    #[test]
    fn test_stack_axis0_rank2_to_rank3() {
        // Stacking [2,3] matrices should produce a [N,2,3] rank-3 tensor
        let mat_a = Value::Tensor(
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
            .expect("matrix a should build"),
        );
        let mat_b = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                vec![
                    Literal::I64(7),
                    Literal::I64(8),
                    Literal::I64(9),
                    Literal::I64(10),
                    Literal::I64(11),
                    Literal::I64(12),
                ],
            )
            .expect("matrix b should build"),
        );
        let stacked =
            TensorValue::stack_axis0(&[mat_a, mat_b]).expect("stacking matrices should succeed");
        assert_eq!(
            stacked.shape,
            Shape {
                dims: vec![2, 2, 3]
            },
            "stacking two [2,3] matrices should produce [2,2,3]"
        );
        let elements: Vec<i64> = stacked
            .elements
            .iter()
            .map(|l| l.as_i64().unwrap())
            .collect();
        assert_eq!(elements, (1..=12).collect::<Vec<i64>>());
    }

    #[test]
    fn test_stack_axis0_shape_mismatch_error() {
        // Stacking tensors with different inner shapes should fail
        let mat_a = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 3] },
                (1..=6).map(Literal::I64).collect(),
            )
            .expect("matrix a should build"),
        );
        let mat_b = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![3, 2] },
                (1..=6).map(Literal::I64).collect(),
            )
            .expect("matrix b should build"),
        );
        let err = TensorValue::stack_axis0(&[mat_a, mat_b])
            .expect_err("stacking mismatched shapes should fail");
        assert!(
            matches!(err, ValueError::AxisStackShapeMismatch { .. }),
            "should get shape mismatch error, got: {err}"
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
                    Primitive::Pad,
                ]
                .into_iter()
                .map(Primitive::as_str)
                .collect::<Vec<_>>();
                assert_eq!(primitive_names.len(), 14);

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
                        effects: vec![],
                        sub_jaxprs: vec![],
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
                        effects: vec![],
                        sub_jaxprs: vec![],
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
                            effects: vec![],
                            sub_jaxprs: vec![],
                        },
                        Equation {
                            primitive: Primitive::Add,
                            inputs: smallvec![
                                Atom::Var(VarId(2)),
                                Atom::Lit(Literal::from_f64(1.0)),
                            ],
                            outputs: smallvec![VarId(3)],
                            params: BTreeMap::new(),
                            effects: vec![],
                            sub_jaxprs: vec![],
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
                            ttl.push_transform(
                                *transform,
                                format!("ev-{}-{idx}", transform.as_str()),
                            );
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
        assert_eq!(Value::scalar_u32(0).dtype(), DType::U32);
        assert_eq!(Value::scalar_u64(0).dtype(), DType::U64);
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

    // ── Unsigned type tests (bd-2983) ─────────────────────────────

    #[test]
    fn test_dtype_u32_exists() {
        let dt = DType::U32;
        assert_ne!(dt, DType::I32);
        assert_ne!(dt, DType::U64);
    }

    #[test]
    fn test_dtype_u64_exists() {
        let dt = DType::U64;
        assert_ne!(dt, DType::I64);
        assert_ne!(dt, DType::U32);
    }

    #[test]
    fn test_literal_u32_max() {
        let lit = Literal::U32(u32::MAX);
        assert_eq!(lit.as_u64(), Some(u64::from(u32::MAX)));
        assert_eq!(lit.as_i64(), Some(i64::from(u32::MAX)));
    }

    #[test]
    fn test_literal_u64_max() {
        let lit = Literal::U64(u64::MAX);
        assert_eq!(lit.as_u64(), Some(u64::MAX));
        assert_eq!(lit.as_i64(), None);
    }

    #[test]
    fn test_u32_serde_roundtrip() {
        let lit = Literal::U32(3_000_000_000);
        let json = serde_json::to_string(&lit).unwrap();
        let deser: Literal = serde_json::from_str(&json).unwrap();
        assert_eq!(lit, deser);
    }

    #[test]
    fn test_tensor_value_u32() {
        let t = TensorValue::new(
            DType::U32,
            Shape::vector(3),
            vec![Literal::U32(1), Literal::U32(2), Literal::U32(u32::MAX)],
        )
        .unwrap();
        assert_eq!(t.dtype, DType::U32);
        assert_eq!(t.shape, Shape::vector(3));
        assert_eq!(t.elements.len(), 3);
    }

    // ── BF16/F16 tests (bd-gsad) ──────────────────────────────────

    #[test]
    fn test_dtype_bfloat16_exists() {
        let dt = DType::BF16;
        assert_ne!(dt, DType::F16);
        assert_ne!(dt, DType::F64);
    }

    #[test]
    fn test_dtype_float16_exists() {
        let dt = DType::F16;
        assert_ne!(dt, DType::BF16);
        assert_ne!(dt, DType::F64);
    }

    #[test]
    fn test_literal_bfloat16_from_f32() {
        let lit = Literal::from_bf16_f32(1.5);
        let roundtrip = lit.as_bf16_f32().unwrap();
        assert_eq!(roundtrip, 1.5);
    }

    #[test]
    fn test_literal_float16_from_f32() {
        let lit = Literal::from_f16_f32(1.5);
        let roundtrip = lit.as_f16_f32().unwrap();
        assert_eq!(roundtrip, 1.5);
    }

    #[test]
    fn test_bfloat16_precision_loss() {
        let value = 1.0000001_f32;
        let lit = Literal::from_bf16_f32(value);
        let roundtrip = lit.as_bf16_f32().unwrap();
        assert_ne!(value.to_bits(), roundtrip.to_bits());
    }

    #[test]
    fn test_float16_range_overflow() {
        let lit = Literal::from_f16_f32(100_000.0);
        let roundtrip = lit.as_f16_f32().unwrap();
        assert!(roundtrip.is_infinite());
    }

    #[test]
    fn test_bfloat16_serde_roundtrip() {
        let lit = Literal::from_bf16_f32(3.25);
        let json = serde_json::to_string(&lit).unwrap();
        let deser: Literal = serde_json::from_str(&json).unwrap();
        assert_eq!(lit, deser);
    }

    #[test]
    fn test_tensor_value_bfloat16() {
        let t = TensorValue::new(
            DType::BF16,
            Shape::vector(2),
            vec![Literal::from_bf16_f32(1.0), Literal::from_bf16_f32(2.0)],
        )
        .unwrap();
        assert_eq!(t.dtype, DType::BF16);
        assert_eq!(t.shape, Shape::vector(2));
    }

    #[test]
    fn test_tensor_value_float16() {
        let t = TensorValue::new(
            DType::F16,
            Shape::vector(2),
            vec![Literal::from_f16_f32(1.0), Literal::from_f16_f32(2.0)],
        )
        .unwrap();
        assert_eq!(t.dtype, DType::F16);
        assert_eq!(t.shape, Shape::vector(2));
    }

    #[test]
    fn test_dtype_size_bfloat16() {
        assert_eq!(std::mem::size_of::<u16>(), 2);
    }

    #[test]
    fn test_dtype_size_float16() {
        assert_eq!(std::mem::size_of::<u16>(), 2);
    }

    #[test]
    fn prop_bfloat16_roundtrip_preserves_bits() {
        run_logged_test(
            "prop_bfloat16_roundtrip_preserves_bits",
            &(
                "prop-bfloat16-roundtrip-preserves-bits",
                fj_test_utils::property_test_case_count(),
            ),
            fj_test_utils::TestMode::Strict,
            || {
                let mut runner = TestRunner::new(ProptestConfig::with_cases(
                    fj_test_utils::property_test_case_count(),
                ));
                runner
                    .run(&any::<u16>(), |bits| {
                        let literal = Literal::BF16Bits(bits);
                        let serialized = serde_json::to_string(&literal)
                            .map_err(|err| TestCaseError::fail(err.to_string()))?;
                        let decoded: Literal = serde_json::from_str(&serialized)
                            .map_err(|err| TestCaseError::fail(err.to_string()))?;
                        let recovered = match decoded {
                            Literal::BF16Bits(value) => value,
                            other => {
                                return Err(TestCaseError::fail(format!(
                                    "expected BF16Bits after roundtrip, got {other:?}"
                                )));
                            }
                        };
                        prop_assert_eq!(recovered, bits);
                        Ok(())
                    })
                    .map_err(|err| err.to_string())?;
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn prop_float16_finite_values_convert() {
        run_logged_test(
            "prop_float16_finite_values_convert",
            &(
                "prop-float16-finite-values-convert",
                fj_test_utils::property_test_case_count(),
            ),
            fj_test_utils::TestMode::Strict,
            || {
                let mut runner = TestRunner::new(ProptestConfig::with_cases(
                    fj_test_utils::property_test_case_count(),
                ));
                let strategy = any::<f32>().prop_filter("finite float16 domain", |value| {
                    value.is_finite() && value.abs() <= 65_504.0
                });
                runner
                    .run(&strategy, |value| {
                        let expected = f32::from(half::f16::from_f32(value));
                        let roundtrip = Literal::from_f16_f32(value)
                            .as_f16_f32()
                            .ok_or_else(|| TestCaseError::fail("expected f16 literal"))?;
                        prop_assert!(roundtrip.is_finite());
                        prop_assert_eq!(roundtrip.to_bits(), expected.to_bits());
                        Ok(())
                    })
                    .map_err(|err| err.to_string())?;
                Ok(Vec::new())
            },
        );
    }

    #[test]
    fn e2e_bfloat16_computation() {
        run_logged_test(
            "e2e_bfloat16_computation",
            &(
                "e2e-bfloat16-computation",
                [1.000_000_1_f32, 2.000_000_2_f32],
            ),
            fj_test_utils::TestMode::Hardened,
            || {
                let input_a = 1.000_000_1_f32;
                let input_b = 2.000_000_2_f32;

                let quantized_a = Literal::from_bf16_f32(input_a)
                    .as_bf16_f32()
                    .ok_or_else(|| "expected BF16 scalar A roundtrip".to_owned())?;
                let quantized_b = Literal::from_bf16_f32(input_b)
                    .as_bf16_f32()
                    .ok_or_else(|| "expected BF16 scalar B roundtrip".to_owned())?;

                let bits_of = |value: f32| -> Result<u16, String> {
                    match Literal::from_bf16_f32(value) {
                        Literal::BF16Bits(bits) => Ok(bits),
                        other => Err(format!("expected BF16Bits, got {other:?}")),
                    }
                };
                let input_a_bits = bits_of(input_a)?;
                let input_b_bits = bits_of(input_b)?;

                let sum_expected = input_a + input_b;
                let product_expected = input_a * input_b;
                let sum_roundtrip = Literal::from_bf16_f32(quantized_a + quantized_b)
                    .as_bf16_f32()
                    .ok_or_else(|| "expected BF16 sum roundtrip".to_owned())?;
                let product_roundtrip = Literal::from_bf16_f32(quantized_a * quantized_b)
                    .as_bf16_f32()
                    .ok_or_else(|| "expected BF16 product roundtrip".to_owned())?;

                let input_a_loss = f64::from((input_a - quantized_a).abs());
                let input_b_loss = f64::from((input_b - quantized_b).abs());
                let sum_loss = f64::from((sum_expected - sum_roundtrip).abs());
                let product_loss = f64::from((product_expected - product_roundtrip).abs());
                assert!(
                    input_a_loss > 0.0 || input_b_loss > 0.0,
                    "expected BF16 quantization to lose precision"
                );

                let records = vec![
                    json!({
                        "test_name": "e2e_bfloat16_computation.input_a",
                        "dtype": "BF16",
                        "bit_pattern": format!("0x{input_a_bits:04x}"),
                        "f32_value": input_a,
                        "roundtrip_value": quantized_a,
                        "precision_loss": input_a_loss,
                        "pass": quantized_a.is_finite()
                    }),
                    json!({
                        "test_name": "e2e_bfloat16_computation.input_b",
                        "dtype": "BF16",
                        "bit_pattern": format!("0x{input_b_bits:04x}"),
                        "f32_value": input_b,
                        "roundtrip_value": quantized_b,
                        "precision_loss": input_b_loss,
                        "pass": quantized_b.is_finite()
                    }),
                    json!({
                        "test_name": "e2e_bfloat16_computation.sum",
                        "dtype": "BF16",
                        "bit_pattern": null,
                        "f32_value": sum_expected,
                        "roundtrip_value": sum_roundtrip,
                        "precision_loss": sum_loss,
                        "pass": sum_roundtrip.is_finite() && sum_loss <= 5e-3
                    }),
                    json!({
                        "test_name": "e2e_bfloat16_computation.mul",
                        "dtype": "BF16",
                        "bit_pattern": null,
                        "f32_value": product_expected,
                        "roundtrip_value": product_roundtrip,
                        "precision_loss": product_loss,
                        "pass": product_roundtrip.is_finite() && product_loss <= 5e-3
                    }),
                ];
                let pass = records
                    .iter()
                    .all(|record| record["pass"].as_bool().unwrap_or(false));

                let e2e_payload = json!({
                    "schema_version": "frankenjax.e2e-forensic-log.v1",
                    "test_name": "e2e_bfloat16_computation",
                    "records": records,
                    "pass": pass
                });
                let test_log_payload = json!({
                    "test_name": "e2e_bfloat16_computation",
                    "dtype": "BF16",
                    "records": e2e_payload["records"].clone(),
                    "pass": pass
                });

                let e2e_path = repo_root()
                    .join("artifacts")
                    .join("e2e")
                    .join("e2e_bfloat16.e2e.json");
                let test_log_path = repo_root()
                    .join("artifacts")
                    .join("testing")
                    .join("logs")
                    .join("fj-core")
                    .join("e2e_bfloat16_computation.json");

                for artifact_path in [&e2e_path, &test_log_path] {
                    if let Some(parent) = artifact_path.parent() {
                        fs::create_dir_all(parent).map_err(|err| {
                            format!(
                                "failed creating artifact directory {}: {err}",
                                parent.display()
                            )
                        })?;
                    }
                }
                fs::write(
                    &e2e_path,
                    serde_json::to_string_pretty(&e2e_payload)
                        .map_err(|err| format!("failed serializing e2e payload: {err}"))?,
                )
                .map_err(|err| format!("failed writing {}: {err}", e2e_path.display()))?;
                fs::write(
                    &test_log_path,
                    serde_json::to_string_pretty(&test_log_payload)
                        .map_err(|err| format!("failed serializing test log payload: {err}"))?,
                )
                .map_err(|err| format!("failed writing {}: {err}", test_log_path.display()))?;

                if !pass {
                    return Err("bf16 e2e forensic checks did not pass".to_owned());
                }

                Ok(vec![
                    e2e_path.display().to_string(),
                    test_log_path.display().to_string(),
                ])
            },
        );
    }

    // ── Complex type tests (bd-1hx8) ────────────────────────────

    #[test]
    fn test_dtype_complex64_exists() {
        let dt = DType::Complex64;
        assert_ne!(dt, DType::F64);
        assert_ne!(dt, DType::Complex128);
    }

    #[test]
    fn test_dtype_complex128_exists() {
        let dt = DType::Complex128;
        assert_ne!(dt, DType::F64);
        assert_ne!(dt, DType::Complex64);
    }

    #[test]
    fn test_literal_complex64_roundtrip() {
        let lit = Literal::from_complex64(1.0_f32, 2.0_f32);
        let (re, im) = lit.as_complex64().unwrap();
        assert_eq!(re, 1.0_f32);
        assert_eq!(im, 2.0_f32);
    }

    #[test]
    fn test_literal_complex128_roundtrip() {
        let lit = Literal::from_complex128(std::f64::consts::PI, -2.71_f64);
        let (re, im) = lit.as_complex128().unwrap();
        assert_eq!(re, std::f64::consts::PI);
        assert_eq!(im, -2.71_f64);
    }

    #[test]
    fn test_literal_complex64_zero() {
        let lit = Literal::from_complex64(0.0, 0.0);
        let (re, im) = lit.as_complex64().unwrap();
        assert_eq!(re, 0.0_f32);
        assert_eq!(im, 0.0_f32);
    }

    #[test]
    fn test_literal_complex64_pure_imaginary() {
        let lit = Literal::from_complex64(0.0, 1.0);
        let (re, im) = lit.as_complex64().unwrap();
        assert_eq!(re, 0.0_f32);
        assert_eq!(im, 1.0_f32);
    }

    #[test]
    fn test_tensor_value_complex64_storage() {
        let elements = vec![
            Literal::from_complex64(1.0, 2.0),
            Literal::from_complex64(3.0, 4.0),
        ];
        let t = TensorValue::new(DType::Complex64, Shape::vector(2), elements).unwrap();
        assert_eq!(t.dtype, DType::Complex64);
        assert_eq!(t.elements.len(), 2);
        assert_eq!(t.elements[0].as_complex64(), Some((1.0, 2.0)));
        assert_eq!(t.elements[1].as_complex64(), Some((3.0, 4.0)));
    }

    #[test]
    fn test_tensor_value_complex128_storage() {
        let elements = vec![
            Literal::from_complex128(1.5, -0.5),
            Literal::from_complex128(2.5, 3.5),
            Literal::from_complex128(0.0, 0.0),
        ];
        let t = TensorValue::new(DType::Complex128, Shape::vector(3), elements).unwrap();
        assert_eq!(t.dtype, DType::Complex128);
        assert_eq!(t.elements.len(), 3);
        assert_eq!(t.elements[0].as_complex128(), Some((1.5, -0.5)));
        assert_eq!(t.elements[2].as_complex128(), Some((0.0, 0.0)));
    }

    #[test]
    fn test_complex_serde_roundtrip() {
        let lit = Literal::from_complex128(std::f64::consts::PI, std::f64::consts::E);
        let json = serde_json::to_string(&lit).unwrap();
        let deser: Literal = serde_json::from_str(&json).unwrap();
        assert_eq!(lit, deser);
        let (re, im) = deser.as_complex128().unwrap();
        assert_eq!(re, std::f64::consts::PI);
        assert_eq!(im, std::f64::consts::E);
    }

    #[test]
    fn test_complex_value_dtype() {
        let v = Value::scalar_complex64(1.0, 2.0);
        assert_eq!(v.dtype(), DType::Complex64);
        let v = Value::scalar_complex128(3.0, 4.0);
        assert_eq!(v.dtype(), DType::Complex128);
    }

    #[test]
    fn test_literal_is_complex() {
        assert!(Literal::from_complex64(1.0, 0.0).is_complex());
        assert!(Literal::from_complex128(1.0, 0.0).is_complex());
        assert!(!Literal::from_f64(1.0).is_complex());
        assert!(!Literal::I64(1).is_complex());
    }

    #[test]
    fn test_complex64_as_f64_returns_none() {
        let lit = Literal::from_complex64(1.0, 2.0);
        assert!(lit.as_f64().is_none());
        assert!(lit.as_i64().is_none());
    }

    #[test]
    fn test_complex128_as_f64_returns_none() {
        let lit = Literal::from_complex128(1.0, 2.0);
        assert!(lit.as_f64().is_none());
        assert!(lit.as_i64().is_none());
    }

    #[test]
    fn dense_f32_literal_buffer_preserves_slice_api_and_cow() {
        // Dense f32 storage must present the same `Literal` API as the boxed
        // form, materializing via `Literal::from_f32` bit-for-bit (including a
        // signaling-NaN payload and -0.0), and fall back to literal storage on
        // mutation (COW).
        let snan = f32::from_bits(0x7fc0_0001);
        let mut buffer = LiteralBuffer::from_f32_values(vec![1.25, -0.0, snan]);
        let expected = vec![
            Literal::from_f32(1.25),
            Literal::from_f32(-0.0),
            Literal::from_f32(snan),
        ];

        assert_eq!(buffer.len(), 3);
        assert!(!buffer.is_empty());
        assert_eq!(buffer.as_f32_slice().expect("dense f32 values").len(), 3);
        assert!(
            buffer.as_f64_slice().is_none(),
            "f32 storage is not f64-backed"
        );
        assert_eq!(buffer.as_slice(), expected.as_slice());
        assert_eq!(buffer.to_vec(), expected);
        assert_eq!(buffer[1], Literal::from_f32(-0.0));
        assert_eq!(format!("{buffer:?}"), format!("{:?}", expected));
        assert_eq!(
            serde_json::to_string(&buffer).expect("serialize dense f32 buffer"),
            serde_json::to_string(&expected).expect("serialize literal vec")
        );

        let owned = buffer.clone().into_iter().collect::<Vec<_>>();
        assert_eq!(owned, expected);
        assert_eq!(buffer, expected);

        let original = buffer.clone();
        buffer[0] = Literal::from_f32(9.0);
        assert_eq!(original.as_slice(), expected.as_slice());
        assert_eq!(buffer[0], Literal::from_f32(9.0));
        assert!(
            buffer.as_f32_slice().is_none(),
            "mutating dense f32 storage should materialize to literal storage"
        );
    }

    #[test]
    fn dense_f32_tensor_value_new_f32_values_roundtrips() {
        let data: Vec<f32> = vec![0.0, -1.5, 3.25, f32::from_bits(0x7f80_0000)]; // incl +inf
        let t = TensorValue::new_f32_values(Shape::vector(4), data.clone()).unwrap();
        assert_eq!(t.dtype, DType::F32);
        assert_eq!(t.elements.as_f32_slice().expect("dense"), data.as_slice());
        for (i, &v) in data.iter().enumerate() {
            assert_eq!(t.elements[i], Literal::from_f32(v));
        }
        // element-count mismatch is rejected like the other dense constructors.
        assert!(TensorValue::new_f32_values(Shape::vector(2), vec![1.0]).is_err());
    }

    #[test]
    fn dense_half_float_storage_roundtrips_and_preserves_api() {
        // Dense BF16/F16 storage must present the same `Literal` API as the boxed
        // form, materializing bit-exact (the stored u16 IS the value's bits), with
        // the right dtype tag, and fall back to literal storage on mutation (COW).
        for (dtype, mk_lit) in [
            (DType::BF16, Literal::BF16Bits as fn(u16) -> Literal),
            (DType::F16, Literal::F16Bits as fn(u16) -> Literal),
        ] {
            let raw: Vec<u16> = vec![0x0000, 0x8000, 0x3f80, 0x7fc1, 0xffff]; // +0,-0,~1,NaN,etc
            let expected: Vec<Literal> = raw.iter().copied().map(mk_lit).collect();
            let mut buffer = LiteralBuffer::from_half_float_values(raw.clone(), dtype);
            assert_eq!(buffer.len(), raw.len());
            assert_eq!(
                buffer.as_half_float_slice().expect("dense half"),
                raw.as_slice()
            );
            assert_eq!(buffer.half_float_dtype(), Some(dtype));
            assert!(buffer.as_f64_slice().is_none());
            assert_eq!(buffer.as_slice(), expected.as_slice());
            assert_eq!(buffer.to_vec(), expected);
            assert_eq!(format!("{buffer:?}"), format!("{:?}", expected));
            assert_eq!(
                serde_json::to_string(&buffer).expect("serialize"),
                serde_json::to_string(&expected).expect("serialize literals")
            );
            assert_eq!(buffer.clone().into_iter().collect::<Vec<_>>(), expected);
            assert_eq!(buffer, expected);
            // COW on mutation -> materializes to literal storage.
            let original = buffer.clone();
            buffer[0] = mk_lit(0x1234);
            assert_eq!(original.as_slice(), expected.as_slice());
            assert_eq!(buffer[0], mk_lit(0x1234));
            assert!(buffer.as_half_float_slice().is_none());

            // TensorValue constructor round-trips with the right dtype.
            let t = TensorValue::new_half_float_values(
                dtype,
                Shape::vector(raw.len() as u32),
                raw.clone(),
            )
            .unwrap();
            assert_eq!(t.dtype, dtype);
            assert_eq!(
                t.elements.as_half_float_slice().expect("dense"),
                raw.as_slice()
            );
            for (i, &b) in raw.iter().enumerate() {
                assert_eq!(t.elements[i], mk_lit(b));
            }
            assert!(TensorValue::new_half_float_values(dtype, Shape::vector(2), vec![1]).is_err());
        }
    }

    #[test]
    fn dense_f64_pass44_literal_buffer_preserves_slice_api_and_cow() {
        let mut buffer =
            LiteralBuffer::from_f64_values(vec![1.25, -0.0, f64::from_bits(0x7ff8_0000_0000_0001)]);
        let expected = vec![
            Literal::from_f64(1.25),
            Literal::from_f64(-0.0),
            Literal::from_f64(f64::from_bits(0x7ff8_0000_0000_0001)),
        ];

        assert_eq!(buffer.len(), 3);
        assert!(!buffer.is_empty());
        assert_eq!(buffer.as_f64_slice().expect("dense values").len(), 3);
        assert_eq!(buffer.as_slice(), expected.as_slice());
        assert_eq!(buffer.to_vec(), expected);
        assert_eq!(buffer[1], Literal::from_f64(-0.0));
        assert_eq!(format!("{buffer:?}"), format!("{:?}", expected));
        assert_eq!(
            serde_json::to_string(&buffer).expect("serialize dense buffer"),
            serde_json::to_string(&expected).expect("serialize literal vec")
        );

        let owned = buffer.clone().into_iter().collect::<Vec<_>>();
        assert_eq!(owned, expected);
        assert_eq!(buffer, expected);

        let original = buffer.clone();
        buffer[0] = Literal::from_f64(9.0);
        assert_eq!(original.as_slice(), expected.as_slice());
        assert_eq!(buffer[0], Literal::from_f64(9.0));
        assert!(
            buffer.as_f64_slice().is_none(),
            "mutating dense storage should materialize to literal storage"
        );
    }

    #[test]
    fn dense_i64_literal_buffer_preserves_slice_api_and_cow() {
        let mut buffer = LiteralBuffer::from_i64_values(vec![7, -3, i64::MIN, i64::MAX, 0]);
        let expected = vec![
            Literal::I64(7),
            Literal::I64(-3),
            Literal::I64(i64::MIN),
            Literal::I64(i64::MAX),
            Literal::I64(0),
        ];

        assert_eq!(buffer.len(), 5);
        assert!(!buffer.is_empty());
        assert_eq!(buffer.as_i64_slice().expect("dense values").len(), 5);
        assert!(
            buffer.as_f64_slice().is_none(),
            "i64 dense storage is not an f64 slice"
        );
        assert_eq!(buffer.as_slice(), expected.as_slice());
        assert_eq!(buffer.to_vec(), expected);
        assert_eq!(buffer[2], Literal::I64(i64::MIN));
        assert_eq!(format!("{buffer:?}"), format!("{:?}", expected));
        assert_eq!(
            serde_json::to_string(&buffer).expect("serialize dense buffer"),
            serde_json::to_string(&expected).expect("serialize literal vec")
        );

        let owned = buffer.clone().into_iter().collect::<Vec<_>>();
        assert_eq!(owned, expected);
        assert_eq!(buffer, expected);

        let original = buffer.clone();
        buffer[0] = Literal::I64(99);
        assert_eq!(original.as_slice(), expected.as_slice());
        assert_eq!(buffer[0], Literal::I64(99));
        assert!(
            buffer.as_i64_slice().is_none(),
            "mutating dense storage should materialize to literal storage"
        );
    }

    #[test]
    fn dense_u32_buffer_round_trips_bit_identically_to_literals() {
        let values: Vec<u32> = vec![0, 1, 7, 3_000_000_000, u32::MAX, u32::MAX - 1];
        let expected: Vec<Literal> = values.iter().copied().map(Literal::U32).collect();

        let mut buffer = LiteralBuffer::from_u32_values(values.clone());
        assert_eq!(buffer.len(), expected.len());
        assert_eq!(buffer.as_u32_slice(), Some(values.as_slice()));
        // Dense u32 must NOT masquerade as i64/bool (would re-introduce the widening bug class).
        assert!(buffer.as_i64_slice().is_none());
        assert!(buffer.as_bool_slice().is_none());
        // Materialization, indexing, serialization, and consuming-iter all match the
        // boxed `Literal::U32` representation bit-for-bit.
        assert_eq!(buffer.as_slice(), expected.as_slice());
        assert_eq!(buffer.to_vec(), expected);
        assert_eq!(buffer[4], Literal::U32(u32::MAX));
        assert_eq!(
            serde_json::to_string(&buffer).expect("serialize dense u32"),
            serde_json::to_string(&expected).expect("serialize literal u32")
        );
        assert_eq!(buffer.clone().into_iter().collect::<Vec<_>>(), expected);
        assert_eq!(buffer, expected);

        // Mutating materializes back to literal storage (clone stays dense + unchanged).
        let original = buffer.clone();
        buffer[0] = Literal::U32(42);
        assert_eq!(original.as_u32_slice(), Some(values.as_slice()));
        assert_eq!(buffer[0], Literal::U32(42));
        assert!(buffer.as_u32_slice().is_none());

        // TensorValue::new densifies a U32 dtype + all-U32 literals.
        let tv = TensorValue::new(DType::U32, Shape { dims: vec![6] }, expected.clone())
            .expect("build u32 tensor");
        assert_eq!(tv.elements.as_u32_slice(), Some(values.as_slice()));
        let tv2 = TensorValue::new_u32_values(Shape { dims: vec![6] }, values.clone())
            .expect("build dense u32 tensor");
        assert_eq!(tv2.elements.as_slice(), expected.as_slice());
    }

    #[test]
    fn dense_u64_buffer_round_trips_bit_identically_to_literals() {
        let values: Vec<u64> = vec![0, 1, 7, u32::MAX as u64 + 1, u64::MAX, u64::MAX - 1];
        let expected: Vec<Literal> = values.iter().copied().map(Literal::U64).collect();

        let mut buffer = LiteralBuffer::from_u64_values(values.clone());
        assert_eq!(buffer.len(), expected.len());
        assert_eq!(buffer.as_u64_slice(), Some(values.as_slice()));
        assert!(buffer.as_i64_slice().is_none());
        assert!(buffer.as_u32_slice().is_none());
        assert_eq!(buffer.as_slice(), expected.as_slice());
        assert_eq!(buffer.to_vec(), expected);
        assert_eq!(buffer[4], Literal::U64(u64::MAX));
        assert_eq!(
            serde_json::to_string(&buffer).expect("serialize dense u64"),
            serde_json::to_string(&expected).expect("serialize literal u64")
        );
        assert_eq!(buffer.clone().into_iter().collect::<Vec<_>>(), expected);
        assert_eq!(buffer, expected);

        let original = buffer.clone();
        buffer[0] = Literal::U64(42);
        assert_eq!(original.as_u64_slice(), Some(values.as_slice()));
        assert!(buffer.as_u64_slice().is_none());

        let tv = TensorValue::new(DType::U64, Shape { dims: vec![6] }, expected.clone())
            .expect("build u64 tensor");
        assert_eq!(tv.elements.as_u64_slice(), Some(values.as_slice()));
        let tv2 = TensorValue::new_u64_values(Shape { dims: vec![6] }, values.clone())
            .expect("build dense u64 tensor");
        assert_eq!(tv2.elements.as_slice(), expected.as_slice());
    }

    #[test]
    fn bool_word_literal_buffer_preserves_literal_api_and_tail_canonicalization() {
        for len in [0usize, 1, 63, 64, 65, 127, 128, 129] {
            let expected_flags: Vec<bool> = (0..len).map(|i| i % 3 == 0 || i % 7 == 1).collect();
            let expected_literals: Vec<Literal> =
                expected_flags.iter().copied().map(Literal::Bool).collect();
            let mut words = vec![0_u64; len.div_ceil(u64::BITS as usize)];
            for (index, &flag) in expected_flags.iter().enumerate() {
                if flag {
                    words[index / u64::BITS as usize] |= 1_u64 << (index % u64::BITS as usize);
                }
            }
            if let Some(last) = words.last_mut() {
                let tail_bits = len % u64::BITS as usize;
                if tail_bits != 0 {
                    *last |= !((1_u64 << tail_bits) - 1);
                }
            }

            let mut buffer =
                LiteralBuffer::from_bool_words(words, len).expect("valid bool word buffer");
            assert_eq!(buffer.len(), len);
            assert_eq!(buffer.as_bool_words().expect("word-backed").1, len);
            assert_eq!(buffer.as_slice(), expected_literals.as_slice(), "len={len}");
            assert_eq!(buffer.to_vec(), expected_literals, "len={len}");
            assert_eq!(
                serde_json::to_string(&buffer).expect("serialize bool words"),
                serde_json::to_string(&expected_literals).expect("serialize literal bools"),
                "len={len}"
            );
            assert_eq!(
                buffer.clone().into_iter().collect::<Vec<_>>(),
                expected_literals,
                "len={len}"
            );

            if len != 0 {
                buffer[0] = Literal::Bool(!expected_flags[0]);
                assert!(
                    buffer.as_bool_words().is_none(),
                    "mutating bool words must materialize to literal storage"
                );
                assert_eq!(buffer[0], Literal::Bool(!expected_flags[0]));
            }
        }

        assert!(LiteralBuffer::from_bool_words(vec![0], 65).is_none());
    }

    #[test]
    fn i32_tensor_constructor_uses_dense_i64_storage() {
        let values = vec![7, -3, i64::from(i32::MIN), i64::from(i32::MAX), i64::MAX];
        let literals = values.iter().copied().map(Literal::I64).collect::<Vec<_>>();
        let shape = Shape::vector(values.len() as u32);
        let tensor = TensorValue::new(DType::I32, shape.clone(), literals.clone())
            .expect("valid i32 tensor");

        assert_eq!(tensor.dtype, DType::I32);
        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.elements.as_i64_slice(), Some(values.as_slice()));
        assert_eq!(tensor.elements.as_slice(), literals.as_slice());

        let explicit =
            TensorValue::new_i32_values(shape.clone(), values.clone()).expect("valid dense i32");
        assert_eq!(explicit.dtype, DType::I32);
        assert_eq!(explicit.elements.as_i64_slice(), Some(values.as_slice()));
        assert_eq!(explicit.elements.as_slice(), literals.as_slice());

        let digest = fj_test_utils::fixture_id_from_json(&(shape.dims, values))
            .expect("i32 dense golden digest should build");
        assert_eq!(
            digest,
            "a5749ee53dedc45fde6e86c9ec1b6fa9bc13cde391b6eac6a6c1f5d0a8d54daa"
        );
    }

    #[test]
    fn i64_tensor_constructor_uses_dense_i64_storage() {
        let values = vec![7, -3, i64::MIN, i64::MAX, 0];
        let literals = values.iter().copied().map(Literal::I64).collect::<Vec<_>>();
        let shape = Shape::vector(values.len() as u32);
        let tensor = TensorValue::new(DType::I64, shape.clone(), literals.clone())
            .expect("valid i64 tensor");

        assert_eq!(tensor.dtype, DType::I64);
        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.elements.as_i64_slice(), Some(values.as_slice()));
        assert_eq!(tensor.elements.as_slice(), literals.as_slice());

        let explicit =
            TensorValue::new_i64_values(shape.clone(), values.clone()).expect("valid dense i64");
        assert_eq!(explicit.dtype, DType::I64);
        assert_eq!(explicit.elements.as_i64_slice(), Some(values.as_slice()));
        assert_eq!(explicit.elements.as_slice(), literals.as_slice());

        let digest = fj_test_utils::fixture_id_from_json(&(shape.dims, values))
            .expect("i64 dense golden digest should build");
        assert_eq!(
            digest,
            "bf3d1069d10c35e03e75849746c04f0066b8f7879af1a6b92a05efc336f6cbce"
        );
    }

    #[test]
    fn repeated_patches_pass60_literal_buffer_materializes_in_update_order() {
        let mut buffer = LiteralBuffer::from_repeated_with_patches(
            vec![Literal::I64(0), Literal::I64(1)],
            3,
            vec![
                (1, Literal::I64(9)),
                (1, Literal::I64(10)),
                (4, Literal::I64(8)),
            ],
        )
        .expect("valid repeated-patch buffer");
        let expected = vec![
            Literal::I64(0),
            Literal::I64(10),
            Literal::I64(0),
            Literal::I64(1),
            Literal::I64(8),
            Literal::I64(1),
        ];

        assert_eq!(buffer.len(), 6);
        assert!(!buffer.is_empty());
        assert!(buffer.as_f64_slice().is_none());
        assert_eq!(buffer.as_slice(), expected.as_slice());
        assert_eq!(buffer.to_vec(), expected);
        assert_eq!(buffer[1], Literal::I64(10));
        assert_eq!(buffer.clone().into_iter().collect::<Vec<_>>(), expected);
        assert_eq!(
            serde_json::to_string(&buffer).expect("serialize repeated-patch buffer"),
            serde_json::to_string(&expected).expect("serialize literal vec")
        );

        let original = buffer.clone();
        buffer[0] = Literal::I64(7);
        assert_eq!(original.as_slice(), expected.as_slice());
        assert_eq!(buffer[0], Literal::I64(7));
        assert_eq!(buffer[1], Literal::I64(10));
    }

    #[test]
    fn concat_pass65_literal_buffer_materializes_in_slice_order() -> Result<(), String> {
        let left = LiteralBuffer::new(vec![Literal::I64(0), Literal::I64(1), Literal::I64(2)]);
        let right = LiteralBuffer::from_f64_values(vec![3.5, -0.0, 7.25]);
        let mut buffer = LiteralBuffer::from_concat_slices(vec![
            (left.clone(), 1, 2),
            (right.clone(), 0, 2),
            (left.clone(), 0, 1),
        ])
        .ok_or_else(|| "valid concat slices should construct".to_owned())?;
        let expected = vec![
            Literal::I64(1),
            Literal::I64(2),
            Literal::from_f64(3.5),
            Literal::from_f64(-0.0),
            Literal::I64(0),
        ];

        assert_eq!(buffer.len(), 5);
        assert!(!buffer.is_empty());
        assert!(buffer.as_f64_slice().is_none());
        assert_eq!(buffer.as_slice(), expected.as_slice());
        assert_eq!(buffer.to_vec(), expected);
        assert_eq!(buffer[3], Literal::from_f64(-0.0));
        assert_eq!(
            serde_json::to_string(&buffer).map_err(|err| err.to_string())?,
            serde_json::to_string(&expected).map_err(|err| err.to_string())?
        );
        assert_eq!(buffer.clone().into_iter().collect::<Vec<_>>(), expected);
        assert_eq!(buffer, expected);

        assert!(
            LiteralBuffer::from_concat_slices(vec![(left.clone(), 2, 2)]).is_none(),
            "out-of-range concat slices should be rejected"
        );

        let original = buffer.clone();
        buffer[0] = Literal::I64(9);
        assert_eq!(original.as_slice(), expected.as_slice());
        assert_eq!(buffer[0], Literal::I64(9));
        assert_eq!(buffer[1], Literal::I64(2));
        Ok(())
    }

    #[test]
    fn concat_dense_typed_slices_materialize_in_slice_order() -> Result<(), String> {
        let left_f64 =
            LiteralBuffer::from_f64_values(vec![1.0, -0.0, f64::from_bits(0x7ff8_0000_0000_0042)]);
        let right_f64 = LiteralBuffer::from_f64_values(vec![2.5, 3.5]);
        let mut f64_concat = LiteralBuffer::from_concat_slices(vec![
            (left_f64.clone(), 1, 2),
            (right_f64.clone(), 0, 2),
        ])
        .ok_or_else(|| "valid f64 concat slices should construct".to_owned())?;
        let f64_expected = [-0.0_f64, f64::from_bits(0x7ff8_0000_0000_0042), 2.5, 3.5];
        assert_eq!(
            f64_concat
                .as_f64_slice()
                .expect("dense f64 concat")
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            f64_expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            f64_concat
                .as_slice()
                .iter()
                .filter_map(|literal| literal.as_f64().map(f64::to_bits))
                .collect::<Vec<_>>(),
            f64_expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        let original = f64_concat.clone();
        f64_concat[0] = Literal::from_f64(9.0);
        assert_eq!(
            original.as_f64_slice().expect("clone keeps dense concat")[0].to_bits(),
            (-0.0_f64).to_bits()
        );
        assert!(
            f64_concat.as_f64_slice().is_none(),
            "mutating a lazy concat materializes literal storage"
        );

        let f32_concat = LiteralBuffer::from_concat_slices(vec![
            (LiteralBuffer::from_f32_values(vec![1.0, -0.0]), 0, 2),
            (LiteralBuffer::from_f32_values(vec![3.25]), 0, 1),
        ])
        .ok_or_else(|| "valid f32 concat slices should construct".to_owned())?;
        assert_eq!(
            f32_concat
                .as_f32_slice()
                .expect("dense f32 concat")
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![1.0_f32.to_bits(), (-0.0_f32).to_bits(), 3.25_f32.to_bits()]
        );

        let i64_concat = LiteralBuffer::from_concat_slices(vec![
            (LiteralBuffer::from_i64_values(vec![1, 2, 3]), 1, 2),
            (LiteralBuffer::from_i64_values(vec![-4]), 0, 1),
        ])
        .ok_or_else(|| "valid i64 concat slices should construct".to_owned())?;
        assert_eq!(
            i64_concat.as_i64_slice().expect("dense i64 concat"),
            &[2, 3, -4]
        );

        let half_concat = LiteralBuffer::from_concat_slices(vec![
            (
                LiteralBuffer::from_half_float_values(vec![0x3c00, 0xbc00], DType::F16),
                0,
                2,
            ),
            (
                LiteralBuffer::from_half_float_values(vec![0x7e00], DType::F16),
                0,
                1,
            ),
        ])
        .ok_or_else(|| "valid half concat slices should construct".to_owned())?;
        assert_eq!(
            half_concat
                .as_half_float_slice()
                .expect("dense half concat"),
            &[0x3c00, 0xbc00, 0x7e00]
        );

        // Complex concat (FFT/signal pipelines): as_complex_slice materializes the
        // contiguous (re,im) backing in slice order, and as_slice (the per-Literal
        // path) yields the identical complex literals — dense path is bit-identical.
        let complex_concat = LiteralBuffer::from_concat_slices(vec![
            (
                LiteralBuffer::from_complex_values(
                    vec![(1.0, 2.0), (3.0, -4.0)],
                    DType::Complex128,
                ),
                0,
                2,
            ),
            (
                LiteralBuffer::from_complex_values(vec![(5.0, 6.0)], DType::Complex128),
                0,
                1,
            ),
        ])
        .ok_or_else(|| "valid complex concat slices should construct".to_owned())?;
        assert_eq!(
            complex_concat
                .as_complex_slice()
                .expect("dense complex concat"),
            &[(1.0, 2.0), (3.0, -4.0), (5.0, 6.0)]
        );
        assert_eq!(
            complex_concat.as_slice(),
            &[
                Literal::from_complex128(1.0, 2.0),
                Literal::from_complex128(3.0, -4.0),
                Literal::from_complex128(5.0, 6.0),
            ]
        );

        // bool / u32 / u64 concat: dense accessors materialize in slice order, and
        // as_slice (per-Literal) yields the identical literals — dense bit-identical.
        let bool_concat = LiteralBuffer::from_concat_slices(vec![
            (
                LiteralBuffer::from_bool_values(vec![true, false, true]),
                1,
                2,
            ),
            (LiteralBuffer::from_bool_values(vec![false]), 0, 1),
        ])
        .ok_or_else(|| "valid bool concat slices should construct".to_owned())?;
        assert_eq!(
            bool_concat.as_bool_slice().expect("dense bool concat"),
            &[false, true, false]
        );
        assert_eq!(
            bool_concat.as_slice(),
            &[
                Literal::Bool(false),
                Literal::Bool(true),
                Literal::Bool(false)
            ]
        );

        let u32_concat = LiteralBuffer::from_concat_slices(vec![
            (LiteralBuffer::from_u32_values(vec![10, 20, 30]), 0, 2),
            (LiteralBuffer::from_u32_values(vec![40]), 0, 1),
        ])
        .ok_or_else(|| "valid u32 concat slices should construct".to_owned())?;
        assert_eq!(
            u32_concat.as_u32_slice().expect("dense u32 concat"),
            &[10, 20, 40]
        );
        assert_eq!(
            u32_concat.as_slice(),
            &[Literal::U32(10), Literal::U32(20), Literal::U32(40)]
        );

        let u64_concat = LiteralBuffer::from_concat_slices(vec![
            (LiteralBuffer::from_u64_values(vec![100, 200]), 1, 1),
            (LiteralBuffer::from_u64_values(vec![300, 400]), 0, 2),
        ])
        .ok_or_else(|| "valid u64 concat slices should construct".to_owned())?;
        assert_eq!(
            u64_concat.as_u64_slice().expect("dense u64 concat"),
            &[200, 300, 400]
        );
        assert_eq!(
            u64_concat.as_slice(),
            &[Literal::U64(200), Literal::U64(300), Literal::U64(400)]
        );

        let digest = fj_test_utils::fixture_id_from_json(&(
            f64_expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![1.0_f32.to_bits(), (-0.0_f32).to_bits(), 3.25_f32.to_bits()],
            vec![2_i64, 3, -4],
            vec![0x3c00_u16, 0xbc00, 0x7e00],
        ))
        .expect("concat dense typed-slice golden digest should build");
        assert_eq!(
            digest,
            "2865e0a49dc4739b8bc41f080918ed9114f39d6b453bef575e86becc9772a3ce"
        );

        let mixed = LiteralBuffer::from_concat_slices(vec![
            (left_f64, 0, 1),
            (LiteralBuffer::new(vec![Literal::from_f64(4.0)]), 0, 1),
        ])
        .ok_or_else(|| "valid mixed concat slices should construct".to_owned())?;
        assert!(
            mixed.as_f64_slice().is_none(),
            "literal-backed parts keep the dense accessor unavailable"
        );
        Ok(())
    }

    #[test]
    fn dense_f64_pass44_vector_f64_uses_dense_storage_with_identical_literals() {
        let input = [1.25, -0.0, f64::from_bits(0x7ff8_0000_0000_0001)];
        let value = Value::vector_f64(&input).expect("vector_f64 should build");
        let tensor = value.as_tensor().expect("expected tensor");
        assert_eq!(tensor.shape, Shape::vector(3));
        assert_eq!(tensor.dtype, DType::F64);
        assert_eq!(
            tensor
                .elements
                .as_f64_slice()
                .expect("dense values")
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            input
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        let literal_bits = tensor
            .elements
            .iter()
            .filter_map(|literal| literal.as_f64().map(f64::to_bits))
            .collect::<Vec<_>>();
        assert_eq!(
            literal_bits,
            input
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn dense_f64_pass44_tensor_new_keeps_malformed_f64_constructible() {
        let tensor = TensorValue::new(DType::F64, Shape::vector(1), vec![Literal::I64(7)])
            .expect("TensorValue::new only checks element count");
        assert!(tensor.elements.as_f64_slice().is_none());
        let mismatch = tensor.validate_dtype_consistency();
        assert!(matches!(
            mismatch,
            Err(ValueError::ElementDTypeMismatch { .. })
        ));
        let Err(ValueError::ElementDTypeMismatch {
            index,
            declared,
            literal,
        }) = mismatch
        else {
            return;
        };
        assert_eq!(index, 0);
        assert_eq!(declared, DType::F64);
        assert_eq!(literal, Literal::I64(7));
    }

    // ── Shape tests ──────────────────────────────────────────

    #[test]
    fn shape_scalar_is_empty() {
        let s = Shape::scalar();
        assert_eq!(s.dims, Vec::<u32>::new());
        assert_eq!(s.rank(), 0);
        assert_eq!(s.element_count(), Some(1));
    }

    #[test]
    fn shape_vector_rank_1() {
        let s = Shape { dims: vec![5] };
        assert_eq!(s.rank(), 1);
        assert_eq!(s.element_count(), Some(5));
    }

    #[test]
    fn shape_matrix_rank_2() {
        let s = Shape { dims: vec![3, 4] };
        assert_eq!(s.rank(), 2);
        assert_eq!(s.element_count(), Some(12));
    }

    #[test]
    fn shape_zero_dim_element_count() {
        let s = Shape {
            dims: vec![3, 0, 4],
        };
        assert_eq!(s.element_count(), Some(0));
    }

    #[test]
    fn shape_zero_dim_short_circuits_huge_prefix() {
        let s = Shape {
            dims: vec![u32::MAX, u32::MAX, u32::MAX, 0],
        };
        assert_eq!(s.element_count(), Some(0));
    }

    // ── f64 -> f16/bf16 single-rounding (round-to-odd) ───────

    #[test]
    fn from_f16_f64_single_rounds_not_double() {
        // Input just above the f16 tie between 0x3C00 (1.0) and 0x3C01:
        // 1 + 2^-11 is exactly halfway; +2^-30 nudges it above the tie, so a
        // correct single rounding gives 0x3C01. Rounding f64->f32 first loses
        // the +2^-30 (below f32 half-ULP), leaving an exact tie that round-to-
        // even sends back down to 0x3C00 — the double-rounding bug.
        let x = 1.00048828125_f64 + 2f64.powi(-30);
        let literal = Literal::from_f16_f64(x);
        assert!(matches!(literal, Literal::F16Bits(_)));
        let Literal::F16Bits(bits) = literal else {
            return;
        };
        assert_eq!(bits, 0x3C01, "f64->f16 must single-round (XLA parity)");
        // Naive double rounding would land on 0x3C00.
        assert_eq!(half::f16::from_f32(x as f32).to_bits(), 0x3C00);
    }

    #[test]
    fn from_f16_bf16_f64_preserve_exact_and_special() {
        // Exact values and non-finites pass through unchanged.
        for x in [0.0_f64, -0.0, 1.0, -2.0, f64::INFINITY, f64::NEG_INFINITY] {
            let Literal::F16Bits(b16) = Literal::from_f16_f64(x) else {
                unreachable!()
            };
            assert_eq!(b16, half::f16::from_f64(x).to_bits(), "f16 {x}");
            let Literal::BF16Bits(bb) = Literal::from_bf16_f64(x) else {
                unreachable!()
            };
            assert_eq!(bb, half::bf16::from_f64(x).to_bits(), "bf16 {x}");
        }
        assert!(
            matches!(Literal::from_f16_f64(f64::NAN), Literal::F16Bits(b) if b & 0x7C00 == 0x7C00 && b & 0x03FF != 0)
        );
    }

    // ── TensorValue edge cases ───────────────────────────────

    #[test]
    fn tensor_value_empty() {
        let tv = TensorValue::new(DType::F64, Shape { dims: vec![0] }, vec![]).unwrap();
        assert!(tv.is_empty());
        assert_eq!(tv.len(), 0);
        assert_eq!(tv.rank(), 1);
    }

    #[test]
    fn tensor_value_element_count_mismatch() {
        let err = TensorValue::new(
            DType::F64,
            Shape { dims: vec![3] },
            vec![Literal::from_f64(1.0), Literal::from_f64(2.0)],
        );
        assert!(err.is_err());
    }

    #[test]
    fn tensor_value_scalar_shape() {
        // Scalar shape = 1 element
        let tv =
            TensorValue::new(DType::F64, Shape::scalar(), vec![Literal::from_f64(42.0)]).unwrap();
        assert_eq!(tv.len(), 1);
        assert_eq!(tv.rank(), 0);
    }

    #[test]
    fn tensor_value_leading_dim() {
        let tv = TensorValue::new(
            DType::F64,
            Shape { dims: vec![3, 2] },
            (0..6).map(|i| Literal::from_f64(i as f64)).collect(),
        )
        .unwrap();
        assert_eq!(tv.leading_dim(), Some(3));
    }

    #[test]
    fn tensor_value_leading_dim_scalar() {
        let tv =
            TensorValue::new(DType::F64, Shape::scalar(), vec![Literal::from_f64(1.0)]).unwrap();
        assert_eq!(tv.leading_dim(), None);
    }

    // ── Value construction helpers ───────────────────────────

    #[test]
    fn value_scalar_i64_construction() {
        let v = Value::scalar_i64(42);
        assert_eq!(v.as_i64_scalar(), Some(42));
    }

    #[test]
    fn value_scalar_f64_construction() {
        let v = Value::scalar_f64(std::f64::consts::PI);
        let f = v.as_f64_scalar().unwrap();
        assert!((f - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn value_scalar_bool_construction() {
        let v = Value::scalar_bool(true);
        assert!(matches!(v, Value::Scalar(Literal::Bool(true))));
    }

    #[test]
    fn value_vector_i64_construction() {
        let v = Value::vector_i64(&[1, 2, 3]).unwrap();
        let t = v.as_tensor().unwrap();
        assert_eq!(t.shape.dims, vec![3]);
        assert_eq!(t.dtype, DType::I64);
    }

    #[test]
    fn value_vector_f64_construction() {
        let v = Value::vector_f64(&[1.0, 2.0]).unwrap();
        let t = v.as_tensor().unwrap();
        assert_eq!(t.shape.dims, vec![2]);
        assert_eq!(t.dtype, DType::F64);
    }

    #[test]
    fn literal_buffer_semantic_golden() -> Result<(), String> {
        let value = Value::vector_f64(&[1.25, -0.0, f64::from_bits(0x7ff8_0000_0000_0001)])
            .map_err(|err| err.to_string())?;
        let tensor = value
            .as_tensor()
            .ok_or_else(|| "vector_f64 did not produce a tensor".to_string())?;
        let cloned = value.clone();
        let clone_tensor = cloned
            .as_tensor()
            .ok_or_else(|| "cloned vector_f64 value did not produce a tensor".to_string())?;
        let slice = tensor.slice_axis0(1).map_err(|err| err.to_string())?;
        let Value::Scalar(Literal::F64Bits(slice_bits)) = slice else {
            return Err("rank-1 axis slice should produce an f64 scalar".to_string());
        };

        let bits = tensor
            .elements
            .iter()
            .map(|literal| match literal {
                Literal::F64Bits(bits) => Ok(*bits),
                other => Err(format!("expected f64 literal, got {other:?}")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let clone_bits = clone_tensor
            .elements
            .iter()
            .map(|literal| match literal {
                Literal::F64Bits(bits) => Ok(*bits),
                other => Err(format!("expected f64 literal, got {other:?}")),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut actual = String::new();
        writeln!(&mut actual, "dtype={:?}", tensor.dtype).map_err(|err| err.to_string())?;
        writeln!(&mut actual, "shape={:?}", tensor.shape.dims).map_err(|err| err.to_string())?;
        writeln!(&mut actual, "bits={bits:?}").map_err(|err| err.to_string())?;
        writeln!(&mut actual, "clone_bits={clone_bits:?}").map_err(|err| err.to_string())?;
        writeln!(&mut actual, "slice_axis0_1_bits={slice_bits}").map_err(|err| err.to_string())?;
        writeln!(
            &mut actual,
            "json={}",
            serde_json::to_string(&value).map_err(|err| err.to_string())?
        )
        .map_err(|err| err.to_string())?;

        assert_eq!(
            actual,
            include_str!(
                "../../../artifacts/performance/evidence/fj_core_literal_buffer_pass42_2026-06-03.golden"
            )
        );
        Ok(())
    }

    // ── Jaxpr construction ───────────────────────────────────

    #[test]
    fn jaxpr_new_basic() {
        let jaxpr = Jaxpr::new(
            vec![VarId(1)],
            vec![],
            vec![VarId(2)],
            vec![Equation {
                primitive: Primitive::Neg,
                inputs: smallvec![Atom::Var(VarId(1))],
                outputs: smallvec![VarId(2)],
                params: std::collections::BTreeMap::new(),
                effects: vec![],
                sub_jaxprs: vec![],
            }],
        );
        assert_eq!(jaxpr.invars.len(), 1);
        assert_eq!(jaxpr.outvars.len(), 1);
        assert_eq!(jaxpr.equations.len(), 1);
        assert!(jaxpr.constvars.is_empty());
    }

    #[test]
    fn jaxpr_empty_equations() {
        // Identity function: output = input
        let jaxpr = Jaxpr::new(vec![VarId(1)], vec![], vec![VarId(1)], vec![]);
        assert!(jaxpr.equations.is_empty());
        assert_eq!(jaxpr.invars, jaxpr.outvars);
    }

    // ── Primitive as_str round-trip ──────────────────────────

    #[test]
    fn all_primitives_have_nonempty_name() {
        // Ensure every variant of Primitive returns a non-empty as_str()
        let prims = [
            Primitive::Add,
            Primitive::Sub,
            Primitive::Mul,
            Primitive::Neg,
            Primitive::Abs,
            Primitive::Sin,
            Primitive::Cos,
            Primitive::Exp,
            Primitive::Log,
            Primitive::Sqrt,
            Primitive::ReduceSum,
            Primitive::Cholesky,
            Primitive::Qr,
            Primitive::Svd,
            Primitive::Fft,
            Primitive::Ifft,
            Primitive::Rfft,
            Primitive::Irfft,
            Primitive::Cond,
            Primitive::Scan,
            Primitive::While,
            Primitive::Switch,
        ];
        for p in prims {
            assert!(
                !p.as_str().is_empty(),
                "primitive {:?} should have non-empty name",
                p
            );
        }
    }

    // ── DType properties ─────────────────────────────────────

    #[test]
    fn dtype_all_variants_distinct() {
        let dtypes = [
            DType::BF16,
            DType::F16,
            DType::F32,
            DType::F64,
            DType::I32,
            DType::I64,
            DType::U32,
            DType::U64,
            DType::Bool,
            DType::Complex64,
            DType::Complex128,
        ];
        // Verify all 11 are distinct via pairwise comparison
        for (i, a) in dtypes.iter().enumerate() {
            for (j, b) in dtypes.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "dtypes at index {i} and {j} should differ");
                }
            }
        }
    }

    mod proptest_tests {
        use super::*;

        proptest::proptest! {
            #![proptest_config(proptest::test_runner::Config::with_cases(
                fj_test_utils::property_test_case_count()
            ))]

            #[test]
            fn metamorphic_shape_element_count_matches_dims_product(
                dims in proptest::collection::vec(1u32..10, 0..4)
            ) {
                let shape = Shape { dims: dims.clone() };
                let expected: u64 = dims.iter().map(|&d| u64::from(d)).product();
                let actual = shape.element_count();
                prop_assert_eq!(actual, Some(expected), "element_count mismatch for dims {:?}", dims);
            }

            #[test]
            fn metamorphic_shape_scalar_has_zero_rank(_seed in 0u64..1000) {
                let scalar = Shape::scalar();
                prop_assert_eq!(scalar.rank(), 0, "scalar should have rank 0");
                prop_assert_eq!(scalar.element_count(), Some(1), "scalar should have 1 element");
            }

            #[test]
            fn metamorphic_shape_vector_has_rank_one(len in 1u32..100) {
                let vec_shape = Shape::vector(len);
                prop_assert_eq!(vec_shape.rank(), 1, "vector should have rank 1");
                prop_assert_eq!(vec_shape.element_count(), Some(u64::from(len)), "vector element count mismatch");
            }

            #[test]
            fn metamorphic_value_scalar_f64_roundtrip(bits in proptest::num::u64::ANY) {
                let original = Value::Scalar(Literal::F64Bits(bits));
                if let Value::Scalar(Literal::F64Bits(recovered)) = original.clone() {
                    prop_assert_eq!(recovered, bits, "f64 scalar bits should roundtrip");
                } else {
                    prop_assert!(false, "scalar should remain scalar");
                }
            }

        }
    }
}
