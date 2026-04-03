#![allow(dead_code)]

use fj_core::{
    CompatibilityMode, DType, Literal, Primitive, ProgramSpec, Shape, TensorValue, Transform, Value,
};
use std::collections::BTreeMap;

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
    // All 44 LAX primitives covered (indices 0..=43, modulo 44).
    match cursor.take_u8() % 44 {
        0 => Primitive::Add,
        1 => Primitive::Sub,
        2 => Primitive::Mul,
        3 => Primitive::Neg,
        4 => Primitive::Abs,
        5 => Primitive::Max,
        6 => Primitive::Min,
        7 => Primitive::Pow,
        8 => Primitive::Exp,
        9 => Primitive::Log,
        10 => Primitive::Sqrt,
        11 => Primitive::Rsqrt,
        12 => Primitive::Floor,
        13 => Primitive::Ceil,
        14 => Primitive::Round,
        15 => Primitive::Sin,
        16 => Primitive::Cos,
        17 => Primitive::Tan,
        18 => Primitive::Asin,
        19 => Primitive::Acos,
        20 => Primitive::Atan,
        21 => Primitive::Sinh,
        22 => Primitive::Cosh,
        23 => Primitive::Tanh,
        24 => Primitive::Expm1,
        25 => Primitive::Log1p,
        26 => Primitive::Sign,
        27 => Primitive::Square,
        28 => Primitive::Reciprocal,
        29 => Primitive::Logistic,
        30 => Primitive::Erf,
        31 => Primitive::Erfc,
        32 => Primitive::Div,
        33 => Primitive::Rem,
        34 => Primitive::Atan2,
        35 => Primitive::Select,
        36 => Primitive::Dot,
        37 => Primitive::Eq,
        38 => Primitive::Ne,
        39 => Primitive::Lt,
        40 => Primitive::Le,
        41 => Primitive::Gt,
        42 => Primitive::Ge,
        // Remaining: reductions + shape ops (index 43 catches them all via _)
        _ => {
            // Distribute among remaining primitives using a second byte
            match cursor.take_u8() % 15 {
                0 => Primitive::ReduceSum,
                1 => Primitive::ReduceMax,
                2 => Primitive::ReduceMin,
                3 => Primitive::ReduceProd,
                4 => Primitive::ReduceAnd,
                5 => Primitive::ReduceOr,
                6 => Primitive::ReduceXor,
                7 => Primitive::Reshape,
                8 => Primitive::Slice,
                9 => Primitive::Gather,
                10 => Primitive::Scatter,
                11 => Primitive::Transpose,
                12 => Primitive::BroadcastInDim,
                13 => Primitive::Concatenate,
                _ => Primitive::Pad,
            }
        }
    }
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
        | Primitive::BitwiseAnd
        | Primitive::BitwiseOr
        | Primitive::BitwiseXor
        | Primitive::ShiftLeft
        | Primitive::ShiftRightArithmetic
        | Primitive::ShiftRightLogical => 2,
        // Ternary ops
        Primitive::Select | Primitive::Scatter | Primitive::Clamp => 3,
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
        | Primitive::BitwiseNot
        | Primitive::ReduceWindow
        | Primitive::PopulationCount
        | Primitive::CountLeadingZeros => 1,
        // Nullary ops
        Primitive::Iota => 0,
        // Parameterized primitives with fixed or minimum input counts.
        Primitive::DynamicUpdateSlice => 3,
        Primitive::OneHot
        | Primitive::Cumsum
        | Primitive::Cumprod
        | Primitive::Sort
        | Primitive::Argsort => 1,
        Primitive::Conv => 2,
        Primitive::Cond => 3,
        Primitive::Scan => 2,
        Primitive::While => 1,
    }
}

#[must_use]
pub fn sample_dtype(cursor: &mut ByteCursor<'_>) -> DType {
    match cursor.take_u8() % 5 {
        0 => DType::F32,
        1 => DType::F64,
        2 => DType::I32,
        3 => DType::I64,
        _ => DType::Bool,
    }
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
        DType::F32 | DType::F64 => Literal::F64Bits(cursor.take_u64()),
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
            let dtype = match cursor.take_u8() % 3 {
                0 => "I64",
                1 => "F64",
                _ => "I32",
            };
            params.insert("dtype".to_owned(), dtype.to_owned());
        }
        Primitive::Add
        | Primitive::Sub
        | Primitive::Mul
        | Primitive::Neg
        | Primitive::Abs
        | Primitive::Max
        | Primitive::Min
        | Primitive::Pow
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
        | Primitive::Div
        | Primitive::Rem
        | Primitive::Atan2
        | Primitive::Select
        | Primitive::Dot
        | Primitive::Eq
        | Primitive::Ne
        | Primitive::Lt
        | Primitive::Le
        | Primitive::Gt
        | Primitive::Ge
        | Primitive::Scatter
        | Primitive::Clamp => {}
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
        assert_eq!(primitive_arity(Primitive::While), 1);
    }
}
