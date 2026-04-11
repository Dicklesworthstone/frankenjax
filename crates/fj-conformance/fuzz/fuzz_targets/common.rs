#![allow(dead_code)]

use fj_core::{
    CompatibilityMode, DType, Literal, Primitive, ProgramSpec, Shape, TensorValue, Transform, Value,
};
use std::collections::BTreeMap;

const ALL_PRIMITIVES: &[Primitive] = &[
    Primitive::Add,
    Primitive::Sub,
    Primitive::Mul,
    Primitive::Neg,
    Primitive::Abs,
    Primitive::Max,
    Primitive::Min,
    Primitive::Pow,
    Primitive::Exp,
    Primitive::Log,
    Primitive::Sqrt,
    Primitive::Rsqrt,
    Primitive::Floor,
    Primitive::Ceil,
    Primitive::Round,
    Primitive::Sin,
    Primitive::Cos,
    Primitive::Tan,
    Primitive::Asin,
    Primitive::Acos,
    Primitive::Atan,
    Primitive::Sinh,
    Primitive::Cosh,
    Primitive::Tanh,
    Primitive::Expm1,
    Primitive::Log1p,
    Primitive::Sign,
    Primitive::Square,
    Primitive::Reciprocal,
    Primitive::Logistic,
    Primitive::Erf,
    Primitive::Erfc,
    Primitive::Div,
    Primitive::Rem,
    Primitive::Atan2,
    Primitive::Complex,
    Primitive::Conj,
    Primitive::Real,
    Primitive::Imag,
    Primitive::Select,
    Primitive::Dot,
    Primitive::Eq,
    Primitive::Ne,
    Primitive::Lt,
    Primitive::Le,
    Primitive::Gt,
    Primitive::Ge,
    Primitive::ReduceSum,
    Primitive::ReduceMax,
    Primitive::ReduceMin,
    Primitive::ReduceProd,
    Primitive::ReduceAnd,
    Primitive::ReduceOr,
    Primitive::ReduceXor,
    Primitive::Reshape,
    Primitive::Slice,
    Primitive::DynamicSlice,
    Primitive::DynamicUpdateSlice,
    Primitive::Gather,
    Primitive::Scatter,
    Primitive::Transpose,
    Primitive::BroadcastInDim,
    Primitive::Concatenate,
    Primitive::Pad,
    Primitive::Rev,
    Primitive::Squeeze,
    Primitive::Split,
    Primitive::ExpandDims,
    Primitive::Cbrt,
    Primitive::Lgamma,
    Primitive::Digamma,
    Primitive::ErfInv,
    Primitive::IsFinite,
    Primitive::IntegerPow,
    Primitive::Nextafter,
    Primitive::Clamp,
    Primitive::Iota,
    Primitive::BroadcastedIota,
    Primitive::Copy,
    Primitive::BitcastConvertType,
    Primitive::ReducePrecision,
    Primitive::Cholesky,
    Primitive::Qr,
    Primitive::Svd,
    Primitive::TriangularSolve,
    Primitive::Eigh,
    Primitive::Fft,
    Primitive::Ifft,
    Primitive::Rfft,
    Primitive::Irfft,
    Primitive::OneHot,
    Primitive::Cumsum,
    Primitive::Cumprod,
    Primitive::Sort,
    Primitive::Argsort,
    Primitive::Conv,
    Primitive::Cond,
    Primitive::Scan,
    Primitive::While,
    Primitive::Switch,
    Primitive::BitwiseAnd,
    Primitive::BitwiseOr,
    Primitive::BitwiseXor,
    Primitive::BitwiseNot,
    Primitive::ShiftLeft,
    Primitive::ShiftRightArithmetic,
    Primitive::ShiftRightLogical,
    Primitive::ReduceWindow,
    Primitive::PopulationCount,
    Primitive::CountLeadingZeros,
];

const ALL_DTYPES: &[DType] = &[
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

const NON_COMPLEX_DTYPES: &[DType] = &[
    DType::BF16,
    DType::F16,
    DType::F32,
    DType::F64,
    DType::I32,
    DType::I64,
    DType::U32,
    DType::U64,
    DType::Bool,
];

pub struct ByteCursor<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> ByteCursor<'a> {
    #[must_use]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    fn take_raw_u8(&mut self) -> u8 {
        if self.data.is_empty() {
            return 0;
        }

        let byte = self.data[self.offset % self.data.len()];
        self.offset = self.offset.saturating_add(1);
        byte
    }

    #[must_use]
    pub fn take_u8(&mut self) -> u8 {
        self.take_raw_u8()
    }

    #[must_use]
    pub fn take_bool(&mut self) -> bool {
        self.take_raw_u8() % 2 == 0
    }

    #[must_use]
    pub fn take_u16(&mut self) -> u16 {
        u16::from(self.take_raw_u8()) | (u16::from(self.take_raw_u8()) << 8)
    }

    #[must_use]
    pub fn take_u32(&mut self) -> u32 {
        u32::from(self.take_raw_u8())
            | (u32::from(self.take_raw_u8()) << 8)
            | (u32::from(self.take_raw_u8()) << 16)
            | (u32::from(self.take_raw_u8()) << 24)
    }

    #[must_use]
    pub fn take_u64(&mut self) -> u64 {
        u64::from(self.take_u32()) | (u64::from(self.take_u32()) << 32)
    }

    #[must_use]
    pub fn take_usize(&mut self, inclusive_max: usize) -> usize {
        if inclusive_max == 0 {
            return 0;
        }
        usize::from(self.take_raw_u8()) % (inclusive_max + 1)
    }

    #[must_use]
    pub fn take_string(&mut self, max_len: usize) -> String {
        let len = self.take_usize(max_len);
        let alphabet = b"abcdefghijklmnopqrstuvwxyz0123456789_-";
        let mut out = String::with_capacity(len);
        for _ in 0..len {
            let idx = usize::from(self.take_raw_u8()) % alphabet.len();
            out.push(char::from(alphabet[idx]));
        }
        out
    }
}

#[must_use]
pub fn sample_mode(cursor: &mut ByteCursor<'_>) -> CompatibilityMode {
    if cursor.take_bool() {
        CompatibilityMode::Strict
    } else {
        CompatibilityMode::Hardened
    }
}

#[must_use]
pub fn sample_program(cursor: &mut ByteCursor<'_>) -> ProgramSpec {
    match cursor.take_u8() % 8 {
        0 => ProgramSpec::Add2,
        1 => ProgramSpec::Square,
        2 => ProgramSpec::SquarePlusLinear,
        3 => ProgramSpec::AddOne,
        4 => ProgramSpec::SinX,
        5 => ProgramSpec::CosX,
        6 => ProgramSpec::Dot3,
        _ => ProgramSpec::ReduceSumVec,
    }
}

#[must_use]
pub fn sample_transform(cursor: &mut ByteCursor<'_>) -> Transform {
    match cursor.take_u8() % 3 {
        0 => Transform::Jit,
        1 => Transform::Grad,
        _ => Transform::Vmap,
    }
}

#[must_use]
pub fn sample_primitive(cursor: &mut ByteCursor<'_>) -> Primitive {
    let idx = cursor.take_usize(ALL_PRIMITIVES.len().saturating_sub(1));
    ALL_PRIMITIVES[idx]
}

#[must_use]
pub fn primitive_arity(primitive: Primitive) -> usize {
    match primitive {
        // Binary ops
        Primitive::Add
        | Primitive::Sub
        | Primitive::Mul
        | Primitive::Max
        | Primitive::Min
        | Primitive::Pow
        | Primitive::Div
        | Primitive::Rem
        | Primitive::Atan2
        | Primitive::Dot
        | Primitive::Gather
        | Primitive::Eq
        | Primitive::Ne
        | Primitive::Lt
        | Primitive::Le
        | Primitive::Gt
        | Primitive::Ge
        | Primitive::Concatenate
        | Primitive::Pad
        | Primitive::Complex
        | Primitive::Nextafter
        | Primitive::TriangularSolve
        | Primitive::BitwiseAnd
        | Primitive::BitwiseOr
        | Primitive::BitwiseXor
        | Primitive::ShiftLeft
        | Primitive::ShiftRightArithmetic
        | Primitive::ShiftRightLogical => 2,
        // Ternary ops
        Primitive::Select | Primitive::Scatter | Primitive::Clamp | Primitive::Cond => 3,
        // Unary ops
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
        | Primitive::ReduceSum
        | Primitive::ReduceMax
        | Primitive::ReduceMin
        | Primitive::ReduceProd
        | Primitive::ReduceAnd
        | Primitive::ReduceOr
        | Primitive::ReduceXor
        | Primitive::Reshape
        | Primitive::Slice
        | Primitive::Transpose
        | Primitive::BroadcastInDim
        | Primitive::DynamicSlice
        | Primitive::Conj
        | Primitive::Real
        | Primitive::Imag
        | Primitive::Rev
        | Primitive::Squeeze
        | Primitive::Split
        | Primitive::ExpandDims
        | Primitive::Cbrt
        | Primitive::Lgamma
        | Primitive::Digamma
        | Primitive::ErfInv
        | Primitive::IsFinite
        | Primitive::IntegerPow
        | Primitive::Copy
        | Primitive::BitcastConvertType
        | Primitive::ReducePrecision
        | Primitive::Cholesky
        | Primitive::Qr
        | Primitive::Svd
        | Primitive::Eigh
        | Primitive::Fft
        | Primitive::Ifft
        | Primitive::Rfft
        | Primitive::Irfft
        | Primitive::BitwiseNot
        | Primitive::ReduceWindow
        | Primitive::PopulationCount
        | Primitive::CountLeadingZeros => 1,
        // Nullary ops
        Primitive::Iota | Primitive::BroadcastedIota => 0,
        // Parameterized primitives with fixed or minimum input counts.
        Primitive::DynamicUpdateSlice => 3,
        Primitive::OneHot
        | Primitive::Cumsum
        | Primitive::Cumprod
        | Primitive::Sort
        | Primitive::Argsort => 1,
        Primitive::Conv => 2,
        Primitive::Scan => 2,
        Primitive::While => 3,
        Primitive::Switch => 2,
    }
}

#[must_use]
pub fn sample_dtype(cursor: &mut ByteCursor<'_>) -> DType {
    let idx = cursor.take_usize(ALL_DTYPES.len().saturating_sub(1));
    ALL_DTYPES[idx]
}

#[must_use]
pub fn sample_shape(cursor: &mut ByteCursor<'_>, max_rank: usize, max_dim: u32) -> Shape {
    let rank = cursor.take_usize(max_rank);
    let mut dims = Vec::with_capacity(rank);
    for _ in 0..rank {
        let limit = max_dim.max(1);
        dims.push(cursor.take_u32() % (limit + 1));
    }
    Shape { dims }
}

#[must_use]
pub fn sample_literal(cursor: &mut ByteCursor<'_>, dtype: DType) -> Literal {
    match dtype {
        DType::Bool => Literal::Bool(cursor.take_bool()),
        DType::I32 => Literal::I64(i64::from((cursor.take_u32() % 10_000) as i32)),
        DType::I64 => Literal::I64((cursor.take_u64() % 1_000_000) as i64),
        DType::U32 => Literal::U32(cursor.take_u32()),
        DType::U64 => Literal::U64(cursor.take_u64()),
        DType::BF16 => Literal::BF16Bits(cursor.take_u16()),
        DType::F16 => Literal::F16Bits(cursor.take_u16()),
        DType::F32 => {
            let value = f32::from_bits(cursor.take_u32());
            Literal::from_f64(f64::from(value))
        }
        DType::F64 => Literal::F64Bits(cursor.take_u64()),
        DType::Complex64 => Literal::Complex64Bits(cursor.take_u32(), cursor.take_u32()),
        DType::Complex128 => Literal::Complex128Bits(cursor.take_u64(), cursor.take_u64()),
    }
}

#[must_use]
pub fn sample_value(cursor: &mut ByteCursor<'_>) -> Value {
    if cursor.take_bool() {
        let dtype = sample_dtype(cursor);
        return Value::Scalar(sample_literal(cursor, dtype));
    }

    let dtype = sample_dtype(cursor);
    let shape = sample_shape(cursor, 3, 4);
    let Some(element_count) = shape.element_count() else {
        return Value::Scalar(sample_literal(cursor, dtype));
    };

    let Ok(element_count) = usize::try_from(element_count) else {
        return Value::Scalar(sample_literal(cursor, dtype));
    };

    if element_count > 64 {
        return Value::Scalar(sample_literal(cursor, dtype));
    }

    let mut elements = Vec::with_capacity(element_count);
    for _ in 0..element_count {
        elements.push(sample_literal(cursor, dtype));
    }

    match TensorValue::new(dtype, shape, elements) {
        Ok(tensor) => Value::Tensor(tensor),
        Err(_) => Value::Scalar(sample_literal(cursor, dtype)),
    }
}

#[must_use]
pub fn sample_values(cursor: &mut ByteCursor<'_>, max_len: usize) -> Vec<Value> {
    let len = cursor.take_usize(max_len);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(sample_value(cursor));
    }
    out
}

#[must_use]
pub fn sample_backend(cursor: &mut ByteCursor<'_>) -> String {
    let backend = cursor.take_string(10);
    if backend.trim().is_empty() {
        "cpu".to_owned()
    } else {
        backend
    }
}

#[must_use]
pub fn sample_compile_options(
    cursor: &mut ByteCursor<'_>,
    max_entries: usize,
) -> BTreeMap<String, String> {
    let mut options = BTreeMap::new();
    let entries = cursor.take_usize(max_entries);
    for idx in 0..entries {
        let key = format!("opt{}_{}", idx, cursor.take_string(6));
        let value = cursor.take_string(14);
        options.insert(key, value);
    }
    options
}

#[must_use]
pub fn sample_unknown_features(cursor: &mut ByteCursor<'_>, max_entries: usize) -> Vec<String> {
    let entries = cursor.take_usize(max_entries);
    let mut out = Vec::with_capacity(entries);
    for idx in 0..entries {
        let base = cursor.take_string(12);
        let feature = if base.is_empty() {
            format!("unknown_feature_{}", idx)
        } else {
            base
        };
        out.push(feature);
    }
    out
}

#[must_use]
pub fn sample_evidence_id(cursor: &mut ByteCursor<'_>, index: usize) -> String {
    if cursor.take_u8() % 5 == 0 {
        return String::new();
    }

    let value = cursor.take_string(16);
    if value.is_empty() {
        format!("ev-{}", index)
    } else {
        value
    }
}

fn dtype_to_param(dtype: DType) -> &'static str {
    match dtype {
        DType::BF16 => "BF16",
        DType::F16 => "F16",
        DType::F32 => "F32",
        DType::F64 => "F64",
        DType::I32 => "I32",
        DType::I64 => "I64",
        DType::U32 => "U32",
        DType::U64 => "U64",
        DType::Bool => "Bool",
        DType::Complex64 => "Complex64",
        DType::Complex128 => "Complex128",
    }
}

fn sample_dtype_param(cursor: &mut ByteCursor<'_>, allow_complex: bool) -> &'static str {
    let pool = if allow_complex {
        ALL_DTYPES
    } else {
        NON_COMPLEX_DTYPES
    };
    let idx = cursor.take_usize(pool.len().saturating_sub(1));
    dtype_to_param(pool[idx])
}

#[must_use]
pub fn sample_primitive_params(
    cursor: &mut ByteCursor<'_>,
    primitive: Primitive,
) -> BTreeMap<String, String> {
    let mut params = BTreeMap::new();

    match primitive {
        Primitive::ReduceSum
        | Primitive::ReduceMax
        | Primitive::ReduceMin
        | Primitive::ReduceProd
        | Primitive::ReduceAnd
        | Primitive::ReduceOr
        | Primitive::ReduceXor => {
            if cursor.take_bool() {
                let axis_count = 1 + cursor.take_usize(2);
                let axes = (0..axis_count)
                    .map(|_| cursor.take_usize(3).to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                params.insert("axes".to_owned(), axes);
            }
        }
        Primitive::Reshape => {
            let rank = 1 + cursor.take_usize(3);
            let shape = (0..rank)
                .map(|_| (1 + cursor.take_usize(4)).to_string())
                .collect::<Vec<_>>()
                .join(",");
            params.insert("new_shape".to_owned(), shape);
        }
        Primitive::Slice => {
            let rank = 1 + cursor.take_usize(3);
            let mut starts = Vec::with_capacity(rank);
            let mut limits = Vec::with_capacity(rank);
            for _ in 0..rank {
                let start = cursor.take_usize(3);
                let width = 1 + cursor.take_usize(3);
                starts.push(start.to_string());
                limits.push((start + width).to_string());
            }
            params.insert("start_indices".to_owned(), starts.join(","));
            params.insert("limit_indices".to_owned(), limits.join(","));
        }
        Primitive::DynamicSlice => {
            let rank = 1 + cursor.take_usize(3);
            let sizes = (0..rank)
                .map(|_| (1 + cursor.take_usize(3)).to_string())
                .collect::<Vec<_>>()
                .join(",");
            params.insert("slice_sizes".to_owned(), sizes);
        }
        Primitive::Gather => {
            if cursor.take_bool() {
                let rank = 1 + cursor.take_usize(3);
                let sizes = (0..rank)
                    .map(|_| (1 + cursor.take_usize(3)).to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                params.insert("slice_sizes".to_owned(), sizes);
            }
        }
        Primitive::Transpose => {
            let rank = 1 + cursor.take_usize(3);
            let mut axes = (0..rank).collect::<Vec<_>>();
            if cursor.take_bool() {
                axes.reverse();
            }
            let encoded = axes
                .into_iter()
                .map(|axis| axis.to_string())
                .collect::<Vec<_>>()
                .join(",");
            params.insert("permutation".to_owned(), encoded);
        }
        Primitive::BroadcastInDim => {
            let rank = 1 + cursor.take_usize(4);
            let target_shape = (0..rank)
                .map(|_| (1 + cursor.take_usize(4)).to_string())
                .collect::<Vec<_>>()
                .join(",");
            params.insert("shape".to_owned(), target_shape);

            if cursor.take_bool() {
                let dims = (0..rank)
                    .map(|axis| axis.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                params.insert("broadcast_dimensions".to_owned(), dims);
            }
        }
        Primitive::Concatenate => {
            params.insert("dimension".to_owned(), cursor.take_usize(3).to_string());
        }
        Primitive::Pad => {
            let rank = 1 + cursor.take_usize(3);
            let lows = (0..rank)
                .map(|_| cursor.take_usize(2).to_string())
                .collect::<Vec<_>>()
                .join(",");
            let highs = (0..rank)
                .map(|_| cursor.take_usize(2).to_string())
                .collect::<Vec<_>>()
                .join(",");
            let interiors = (0..rank)
                .map(|_| cursor.take_usize(2).to_string())
                .collect::<Vec<_>>()
                .join(",");
            params.insert("padding_low".to_owned(), lows);
            params.insert("padding_high".to_owned(), highs);
            params.insert("padding_interior".to_owned(), interiors);
        }
        Primitive::Iota => {
            params.insert("length".to_owned(), (1 + cursor.take_usize(8)).to_string());
            let dtype = match cursor.take_u8() % 4 {
                0 => "I64",
                1 => "F64",
                2 => "I32",
                _ => "F32",
            };
            params.insert("dtype".to_owned(), dtype.to_owned());
        }
        Primitive::BroadcastedIota => {
            let rank = 1 + cursor.take_usize(3);
            let shape = (0..rank)
                .map(|_| (1 + cursor.take_usize(4)).to_string())
                .collect::<Vec<_>>()
                .join(",");
            params.insert("shape".to_owned(), shape);
            if cursor.take_bool() {
                let axis = cursor.take_usize(rank.saturating_sub(1));
                params.insert("dimension".to_owned(), axis.to_string());
            }
            if cursor.take_bool() {
                params.insert(
                    "dtype".to_owned(),
                    sample_dtype_param(cursor, false).to_owned(),
                );
            }
        }
        Primitive::OneHot => {
            params.insert(
                "num_classes".to_owned(),
                (1 + cursor.take_usize(8)).to_string(),
            );
            if cursor.take_bool() {
                params.insert(
                    "dtype".to_owned(),
                    sample_dtype_param(cursor, false).to_owned(),
                );
            }
            if cursor.take_bool() {
                let on_value = (cursor.take_u8() % 5) as f64;
                let off_value = (cursor.take_u8() % 3) as f64;
                params.insert("on_value".to_owned(), on_value.to_string());
                params.insert("off_value".to_owned(), off_value.to_string());
            }
        }
        Primitive::BitcastConvertType => {
            params.insert(
                "new_dtype".to_owned(),
                sample_dtype_param(cursor, true).to_owned(),
            );
        }
        Primitive::ReducePrecision => {
            if cursor.take_bool() {
                params.insert(
                    "exponent_bits".to_owned(),
                    (1 + cursor.take_usize(10)).to_string(),
                );
            }
            if cursor.take_bool() {
                params.insert(
                    "mantissa_bits".to_owned(),
                    (1 + cursor.take_usize(23)).to_string(),
                );
            }
        }
        Primitive::IntegerPow => {
            let raw = cursor.take_u8() as i32;
            let exponent = (raw % 11) - 5;
            params.insert("exponent".to_owned(), exponent.to_string());
        }
        _ => {}
    }

    params
}

#[cfg(test)]
mod tests {
    use super::primitive_arity;
    use fj_core::Primitive;

    #[test]
    fn primitive_arity_uses_real_counts_for_parameterized_primitives() {
        assert_eq!(primitive_arity(Primitive::DynamicUpdateSlice), 3);
        assert_eq!(primitive_arity(Primitive::OneHot), 1);
        assert_eq!(primitive_arity(Primitive::Cumsum), 1);
        assert_eq!(primitive_arity(Primitive::Cumprod), 1);
        assert_eq!(primitive_arity(Primitive::Sort), 1);
        assert_eq!(primitive_arity(Primitive::Argsort), 1);
        assert_eq!(primitive_arity(Primitive::Conv), 2);
        assert_eq!(primitive_arity(Primitive::Cond), 3);
        assert_eq!(primitive_arity(Primitive::Scan), 2);
        assert_eq!(primitive_arity(Primitive::While), 3);
        assert_eq!(primitive_arity(Primitive::Switch), 2);
        assert_eq!(primitive_arity(Primitive::BroadcastedIota), 0);
    }
}
