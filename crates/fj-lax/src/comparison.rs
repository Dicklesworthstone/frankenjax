#![forbid(unsafe_code)]

use fj_core::{DType, Literal, LiteralBuffer, Primitive, Shape, TensorValue, Value};
use std::simd::{
    Simd,
    cmp::{SimdPartialEq, SimdPartialOrd},
};

use crate::EvalError;
use crate::type_promotion::compare_literals;

fn broadcast_shape(lhs: &Shape, rhs: &Shape) -> Option<Shape> {
    let max_rank = lhs.rank().max(rhs.rank());
    let mut dims = Vec::with_capacity(max_rank);

    for offset in 0..max_rank {
        let lhs_dim = if offset < lhs.rank() {
            lhs.dims[lhs.rank() - 1 - offset]
        } else {
            1
        };
        let rhs_dim = if offset < rhs.rank() {
            rhs.dims[rhs.rank() - 1 - offset]
        } else {
            1
        };

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

fn compute_strides(dims: &[u32]) -> Vec<usize> {
    let mut strides = vec![1_usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1] as usize;
    }
    strides
}

fn flat_to_multi(flat: usize, strides: &[usize]) -> Vec<usize> {
    let mut multi = Vec::with_capacity(strides.len());
    let mut remainder = flat;
    for &stride in strides {
        multi.push(remainder / stride);
        remainder %= stride;
    }
    multi
}

fn broadcast_strides(shape: &Shape, out_shape: &Shape) -> Vec<usize> {
    let rank = shape.rank();
    let out_rank = out_shape.rank();
    let real_strides = compute_strides(&shape.dims);

    let mut result = vec![0_usize; out_rank];
    for (i, &stride) in real_strides.iter().enumerate().take(rank) {
        let out_axis = out_rank - rank + i;
        if shape.dims[i] == 1 {
            result[out_axis] = 0;
        } else {
            result[out_axis] = stride;
        }
    }
    result
}

fn broadcast_flat_index(multi: &[usize], strides: &[usize]) -> usize {
    multi.iter().zip(strides.iter()).map(|(&m, &s)| m * s).sum()
}

fn eval_f64_rank2_row_broadcast_compare(
    lhs: &TensorValue,
    rhs: &TensorValue,
    out_shape: &Shape,
    float_cmp: &impl Fn(f64, f64) -> bool,
) -> Result<Option<Value>, EvalError> {
    let (Some(left), Some(right)) = (lhs.elements.as_f64_slice(), rhs.elements.as_f64_slice())
    else {
        return Ok(None);
    };

    if lhs.shape.rank() == 2
        && rhs.shape.rank() == 1
        && lhs.shape.dims[1] == rhs.shape.dims[0]
        && out_shape.dims == lhs.shape.dims
    {
        let rows = lhs.shape.dims[0] as usize;
        let cols = lhs.shape.dims[1] as usize;
        if cols == 0 {
            return Ok(Some(Value::Tensor(TensorValue::new_bool_values(
                out_shape.clone(),
                Vec::new(),
            )?)));
        }
        let mut out = Vec::with_capacity(rows * cols);
        for row in left.chunks_exact(cols).take(rows) {
            for (&a, &b) in row.iter().zip(right) {
                out.push(float_cmp(a, b));
            }
        }
        return Ok(Some(Value::Tensor(TensorValue::new_bool_values(
            out_shape.clone(),
            out,
        )?)));
    }

    if lhs.shape.rank() == 1
        && rhs.shape.rank() == 2
        && lhs.shape.dims[0] == rhs.shape.dims[1]
        && out_shape.dims == rhs.shape.dims
    {
        let rows = rhs.shape.dims[0] as usize;
        let cols = rhs.shape.dims[1] as usize;
        if cols == 0 {
            return Ok(Some(Value::Tensor(TensorValue::new_bool_values(
                out_shape.clone(),
                Vec::new(),
            )?)));
        }
        let mut out = Vec::with_capacity(rows * cols);
        for row in right.chunks_exact(cols).take(rows) {
            for (&a, &b) in left.iter().zip(row) {
                out.push(float_cmp(a, b));
            }
        }
        return Ok(Some(Value::Tensor(TensorValue::new_bool_values(
            out_shape.clone(),
            out,
        )?)));
    }

    Ok(None)
}

#[inline]
fn push_bool_word(words: &mut [u64], index: usize, value: bool) {
    if value {
        words[index / u64::BITS as usize] |= 1_u64 << (index % u64::BITS as usize);
    }
}

fn bool_words_tensor(
    primitive: Primitive,
    shape: Shape,
    len: usize,
    words: Vec<u64>,
) -> Result<Value, EvalError> {
    let elements = LiteralBuffer::from_bool_words(words, len).ok_or(EvalError::Unsupported {
        primitive,
        detail: "invalid bool word mask length".to_owned(),
    })?;
    Ok(Value::Tensor(TensorValue::new_with_literal_buffer(
        DType::Bool,
        shape,
        elements,
    )?))
}

fn f64_compare_words(
    primitive: Primitive,
    left: &[f64],
    right: &[f64],
    float_cmp: &impl Fn(f64, f64) -> bool,
) -> Vec<u64> {
    const WORD_BITS: usize = u64::BITS as usize;
    const LANES: usize = 8;
    debug_assert!(matches!(
        primitive,
        Primitive::Eq
            | Primitive::Ne
            | Primitive::Lt
            | Primitive::Le
            | Primitive::Gt
            | Primitive::Ge
    ));
    let mut words = vec![0_u64; left.len().div_ceil(WORD_BITS)];
    let full_words = left.len() / WORD_BITS;
    for (word_index, word_slot) in words.iter_mut().take(full_words).enumerate() {
        let base = word_index * WORD_BITS;
        let mut word = 0_u64;
        for chunk in 0..(WORD_BITS / LANES) {
            let offset = base + chunk * LANES;
            let left_values = Simd::<f64, LANES>::from_slice(&left[offset..offset + LANES]);
            let right_values = Simd::<f64, LANES>::from_slice(&right[offset..offset + LANES]);
            let mask = match primitive {
                Primitive::Eq => left_values.simd_eq(right_values).to_bitmask(),
                Primitive::Ne => left_values.simd_ne(right_values).to_bitmask(),
                Primitive::Lt => left_values.simd_lt(right_values).to_bitmask(),
                Primitive::Le => left_values.simd_le(right_values).to_bitmask(),
                Primitive::Gt => left_values.simd_gt(right_values).to_bitmask(),
                Primitive::Ge => left_values.simd_ge(right_values).to_bitmask(),
                _ => 0,
            };
            word |= mask << (chunk * LANES);
        }
        *word_slot = word;
    }
    for index in (full_words * WORD_BITS)..left.len() {
        push_bool_word(&mut words, index, float_cmp(left[index], right[index]));
    }
    words
}

/// Comparison operators: return Bool scalars/tensors.
#[inline]
pub(crate) fn eval_comparison(
    primitive: Primitive,
    inputs: &[Value],
    int_cmp: impl Fn(i128, i128) -> bool,
    float_cmp: impl Fn(f64, f64) -> bool,
) -> Result<Value, EvalError> {
    if inputs.len() != 2 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 2,
            actual: inputs.len(),
        });
    }

    match (&inputs[0], &inputs[1]) {
        (Value::Scalar(lhs), Value::Scalar(rhs)) => {
            let result = compare_literals(*lhs, *rhs, primitive, &int_cmp, &float_cmp)?;
            Ok(Value::scalar_bool(result))
        }
        (Value::Tensor(lhs), Value::Tensor(rhs)) => {
            if lhs.shape == rhs.shape {
                // I64 AND I32 share the dense i64 backing; an i32 value is sign-extended
                // in its i64 slot, so `int_cmp(i128::from(..))` is identical for both (the
                // bool output needs no width-narrowing). I32 is JAX's default int, so this
                // is the common integer-compare case that otherwise hit the generic loop.
                if matches!(lhs.dtype, DType::I64 | DType::I32)
                    && matches!(rhs.dtype, DType::I64 | DType::I32)
                    && let Some(value) = eval_same_shape_i64_compare(lhs, rhs, &int_cmp)?
                {
                    return Ok(value);
                }
                // Unsigned u32/u64 dense same-shape compare. Generic compare_literals
                // maps U32/U64 through literal_to_i128 (always non-negative), so
                // `int_cmp(i128::from(left), i128::from(right))` on the packed slices is
                // bit-for-bit identical with no signed/unsigned ambiguity.
                if let Some(value) = eval_same_shape_unsigned_compare(lhs, rhs, &int_cmp)? {
                    return Ok(value);
                }
                if lhs.dtype == DType::F64
                    && rhs.dtype == DType::F64
                    && let Some(value) =
                        eval_same_shape_f64_compare(primitive, lhs, rhs, &float_cmp)?
                {
                    return Ok(value);
                }
                if lhs.dtype == DType::F32
                    && rhs.dtype == DType::F32
                    && let Some(value) = eval_same_shape_f32_compare(lhs, rhs, &float_cmp)?
                {
                    return Ok(value);
                }
                if matches!(lhs.dtype, DType::BF16 | DType::F16)
                    && lhs.dtype == rhs.dtype
                    && let Some(value) = eval_same_shape_half_float_compare(lhs, rhs, &float_cmp)?
                {
                    return Ok(value);
                }
                if matches!(lhs.dtype, DType::Complex64 | DType::Complex128)
                    && let Some(value) = eval_same_shape_complex_compare(primitive, lhs, rhs)?
                {
                    return Ok(value);
                }

                let mut elements = Vec::with_capacity(lhs.elements.len());
                for (lhs, rhs) in lhs
                    .elements
                    .iter()
                    .copied()
                    .zip(rhs.elements.iter().copied())
                {
                    elements.push(Literal::Bool(compare_literals(
                        lhs, rhs, primitive, &int_cmp, &float_cmp,
                    )?));
                }
                return Ok(Value::Tensor(TensorValue::new(
                    DType::Bool,
                    lhs.shape.clone(),
                    elements,
                )?));
            }

            let out_shape =
                broadcast_shape(&lhs.shape, &rhs.shape).ok_or(EvalError::ShapeMismatch {
                    primitive,
                    left: lhs.shape.clone(),
                    right: rhs.shape.clone(),
                })?;

            let out_count = out_shape.element_count().unwrap_or(0) as usize;
            let out_strides = compute_strides(&out_shape.dims);
            let lhs_strides = broadcast_strides(&lhs.shape, &out_shape);
            let rhs_strides = broadcast_strides(&rhs.shape, &out_shape);

            // Dense broadcast-compare fast paths via the shared BroadcastOdometer:
            // gather the two source offsets incrementally (no per-element
            // flat_to_multi decode) from the contiguous typed slices, and apply
            // the same cmp closure. Bit-for-bit identical to the generic
            // compare_literals path for F64⊗F64 (float_cmp on as_f64) and I64⊗I64
            // (int_cmp on i128::from). Bool output (no dense Bool storage).
            if let (Some(left), Some(right)) =
                (lhs.elements.as_i64_slice(), rhs.elements.as_i64_slice())
            {
                let mut out = Vec::with_capacity(out_count);
                crate::arithmetic::broadcast_visit_row_major(
                    &out_shape.dims,
                    &lhs_strides,
                    &rhs_strides,
                    |li, ri| {
                        out.push(int_cmp(i128::from(left[li]), i128::from(right[ri])));
                    },
                );
                return Ok(Value::Tensor(TensorValue::new_bool_values(out_shape, out)?));
            }
            if let (Some(left), Some(right)) =
                (lhs.elements.as_u32_slice(), rhs.elements.as_u32_slice())
            {
                let mut out = Vec::with_capacity(out_count);
                crate::arithmetic::broadcast_visit_row_major(
                    &out_shape.dims,
                    &lhs_strides,
                    &rhs_strides,
                    |li, ri| {
                        out.push(int_cmp(i128::from(left[li]), i128::from(right[ri])));
                    },
                );
                return Ok(Value::Tensor(TensorValue::new_bool_values(out_shape, out)?));
            }
            if let (Some(left), Some(right)) =
                (lhs.elements.as_u64_slice(), rhs.elements.as_u64_slice())
            {
                let mut out = Vec::with_capacity(out_count);
                crate::arithmetic::broadcast_visit_row_major(
                    &out_shape.dims,
                    &lhs_strides,
                    &rhs_strides,
                    |li, ri| {
                        out.push(int_cmp(i128::from(left[li]), i128::from(right[ri])));
                    },
                );
                return Ok(Value::Tensor(TensorValue::new_bool_values(out_shape, out)?));
            }
            if let (Some(left), Some(right)) =
                (lhs.elements.as_f64_slice(), rhs.elements.as_f64_slice())
            {
                if let Some(value) =
                    eval_f64_rank2_row_broadcast_compare(lhs, rhs, &out_shape, &float_cmp)?
                {
                    return Ok(value);
                }
                let mut out = Vec::with_capacity(out_count);
                crate::arithmetic::broadcast_visit_row_major(
                    &out_shape.dims,
                    &lhs_strides,
                    &rhs_strides,
                    |li, ri| {
                        out.push(float_cmp(left[li], right[ri]));
                    },
                );
                return Ok(Value::Tensor(TensorValue::new_bool_values(out_shape, out)?));
            }
            if let (Some(left), Some(right)) =
                (lhs.elements.as_f32_slice(), rhs.elements.as_f32_slice())
            {
                let mut out = Vec::with_capacity(out_count);
                crate::arithmetic::broadcast_visit_row_major(
                    &out_shape.dims,
                    &lhs_strides,
                    &rhs_strides,
                    |li, ri| {
                        out.push(float_cmp(f64::from(left[li]), f64::from(right[ri])));
                    },
                );
                return Ok(Value::Tensor(TensorValue::new_bool_values(out_shape, out)?));
            }
            // Half-float (BF16/F16) dense broadcast compare — the broadcast sibling of
            // eval_same_shape_half_float_compare. Decode each u16 to f64 exactly via
            // as_f64() (same widen the generic float branch uses) and apply float_cmp.
            if matches!(lhs.dtype, DType::BF16 | DType::F16)
                && lhs.dtype == rhs.dtype
                && let (Some(left), Some(right)) = (
                    lhs.elements.as_half_float_slice(),
                    rhs.elements.as_half_float_slice(),
                )
            {
                let decode: fn(u16) -> f64 = if lhs.dtype == DType::BF16 {
                    |b| Literal::BF16Bits(b).as_f64().unwrap_or(f64::NAN)
                } else {
                    |b| Literal::F16Bits(b).as_f64().unwrap_or(f64::NAN)
                };
                let mut out = Vec::with_capacity(out_count);
                crate::arithmetic::broadcast_visit_row_major(
                    &out_shape.dims,
                    &lhs_strides,
                    &rhs_strides,
                    |li, ri| {
                        out.push(float_cmp(decode(left[li]), decode(right[ri])));
                    },
                );
                return Ok(Value::Tensor(TensorValue::new_bool_values(out_shape, out)?));
            }

            // Complex broadcast Eq/Ne — the broadcast sibling of
            // eval_same_shape_complex_compare. Only Eq/Ne (ordered comparisons skip
            // this and fall through to compare_literals below, which raises the
            // correct "ordered comparison is not supported for complex operands"
            // error). Component equality (re == re && im == im, negated for Ne) over
            // the packed (re, im) backings — bit-for-bit identical to compare_literals'
            // complex arm, into dense Bool.
            if matches!(primitive, Primitive::Eq | Primitive::Ne)
                && let (Some(left), Some(right)) = (
                    lhs.elements.as_complex_slice(),
                    rhs.elements.as_complex_slice(),
                )
            {
                let is_eq = primitive == Primitive::Eq;
                let mut out = Vec::with_capacity(out_count);
                crate::arithmetic::broadcast_visit_row_major(
                    &out_shape.dims,
                    &lhs_strides,
                    &rhs_strides,
                    |li, ri| {
                        let (lre, lim) = left[li];
                        let (rre, rim) = right[ri];
                        let equal = lre == rre && lim == rim;
                        out.push(if is_eq { equal } else { !equal });
                    },
                );
                return Ok(Value::Tensor(TensorValue::new_bool_values(out_shape, out)?));
            }

            let mut elements = Vec::with_capacity(out_count);
            for flat_idx in 0..out_count {
                let multi = flat_to_multi(flat_idx, &out_strides);
                let lhs_idx = broadcast_flat_index(&multi, &lhs_strides);
                let rhs_idx = broadcast_flat_index(&multi, &rhs_strides);

                let l = lhs.elements[lhs_idx];
                let r = rhs.elements[rhs_idx];
                elements.push(Literal::Bool(compare_literals(
                    l, r, primitive, &int_cmp, &float_cmp,
                )?));
            }

            Ok(Value::Tensor(TensorValue::new(
                DType::Bool,
                out_shape,
                elements,
            )?))
        }
        (Value::Scalar(lhs), Value::Tensor(rhs)) => {
            if let Some(value) = eval_f64_scalar_compare(*lhs, rhs, true, &float_cmp)? {
                return Ok(value);
            }
            if let Some(value) = eval_f32_scalar_compare(*lhs, rhs, true, &float_cmp)? {
                return Ok(value);
            }
            if let Some(value) = eval_i64_scalar_compare(*lhs, rhs, true, &int_cmp)? {
                return Ok(value);
            }
            if let Some(value) = eval_unsigned_scalar_compare(*lhs, rhs, true, &int_cmp)? {
                return Ok(value);
            }

            let mut elements = Vec::with_capacity(rhs.elements.len());
            for rhs in rhs.elements.iter().copied() {
                elements.push(Literal::Bool(compare_literals(
                    *lhs, rhs, primitive, &int_cmp, &float_cmp,
                )?));
            }
            Ok(Value::Tensor(TensorValue::new(
                DType::Bool,
                rhs.shape.clone(),
                elements,
            )?))
        }
        (Value::Tensor(lhs), Value::Scalar(rhs)) => {
            if let Some(value) = eval_f64_scalar_compare(*rhs, lhs, false, &float_cmp)? {
                return Ok(value);
            }
            if let Some(value) = eval_f32_scalar_compare(*rhs, lhs, false, &float_cmp)? {
                return Ok(value);
            }
            if let Some(value) = eval_i64_scalar_compare(*rhs, lhs, false, &int_cmp)? {
                return Ok(value);
            }
            if let Some(value) = eval_unsigned_scalar_compare(*rhs, lhs, false, &int_cmp)? {
                return Ok(value);
            }

            let mut elements = Vec::with_capacity(lhs.elements.len());
            for lhs in lhs.elements.iter().copied() {
                elements.push(Literal::Bool(compare_literals(
                    lhs, *rhs, primitive, &int_cmp, &float_cmp,
                )?));
            }
            Ok(Value::Tensor(TensorValue::new(
                DType::Bool,
                lhs.shape.clone(),
                elements,
            )?))
        }
    }
}

/// Same-shape F64⊗F64 comparison fast path producing a `DType::Bool` tensor.
///
/// Bit-for-bit identical to the generic `compare_literals` path: for F64
/// operands that path falls through to `float_cmp(lhs.as_f64(), rhs.as_f64())`,
/// which is exactly what this applies here (same closure), skipping the
/// complex-operand check and the integral `literal_to_i128` probe per element.
/// Returns `Ok(None)` if any element is not `F64Bits` so the caller falls
/// through to the generic path.
#[inline]
fn eval_same_shape_f64_compare(
    primitive: Primitive,
    lhs: &TensorValue,
    rhs: &TensorValue,
    float_cmp: &impl Fn(f64, f64) -> bool,
) -> Result<Option<Value>, EvalError> {
    // Dense F64 path: a contiguous `float_cmp` map straight into a dense `Vec<bool>`
    // backing (DType::Bool). This vectorizes the compare-and-write and shrinks the
    // mask from 24 bytes/elem (Literal) to 1, the dominant cost at scale. Bit-for-
    // bit identical: same `float_cmp` in the same order, and a Bool's value carries
    // no representation ambiguity, so dense-bool output equals the Literal output.
    if let (Some(left), Some(right)) = (lhs.elements.as_f64_slice(), rhs.elements.as_f64_slice()) {
        let out = f64_compare_words(primitive, left, right, float_cmp);
        return Ok(Some(bool_words_tensor(
            primitive,
            lhs.shape.clone(),
            left.len(),
            out,
        )?));
    }

    let mut out = Vec::with_capacity(lhs.elements.len());
    for (left, right) in lhs.elements.iter().zip(&rhs.elements) {
        let (Literal::F64Bits(left_bits), Literal::F64Bits(right_bits)) = (*left, *right) else {
            return Ok(None);
        };
        out.push(float_cmp(
            f64::from_bits(left_bits),
            f64::from_bits(right_bits),
        ));
    }

    Ok(Some(Value::Tensor(TensorValue::new_bool_values(
        lhs.shape.clone(),
        out,
    )?)))
}

/// Same-shape complex Eq/Ne fast path producing a `DType::Bool` tensor. JAX allows
/// eq/ne on complex (operands `_any`) but NOT ordered lt/le/gt/ge (`_ordered` excludes
/// complex); this handles ONLY Eq/Ne and returns `None` otherwise, so ordered
/// comparisons fall through to `compare_literals`, which raises the
/// "ordered comparison is not supported for complex operands" error (matching JAX).
/// Bit-for-bit identical to `compare_literals`' complex arm: component equality
/// (`re == re && im == im`, negated for Ne — so NaN compares unequal, ±0 compares equal)
/// over the packed `(re, im)` backing, into a dense `Vec<bool>` (1 byte/elem vs the
/// 24-byte boxed Literal). Returns `None` if either operand is not dense-complex.
fn eval_same_shape_complex_compare(
    primitive: Primitive,
    lhs: &TensorValue,
    rhs: &TensorValue,
) -> Result<Option<Value>, EvalError> {
    if !matches!(primitive, Primitive::Eq | Primitive::Ne) {
        return Ok(None);
    }
    let (Some(left), Some(right)) = (
        lhs.elements.as_complex_slice(),
        rhs.elements.as_complex_slice(),
    ) else {
        return Ok(None);
    };
    let is_eq = primitive == Primitive::Eq;
    let out: Vec<bool> = left
        .iter()
        .zip(right)
        .map(|(&(lre, lim), &(rre, rim))| {
            let equal = lre == rre && lim == rim;
            if is_eq { equal } else { !equal }
        })
        .collect();
    Ok(Some(Value::Tensor(TensorValue::new_bool_values(
        lhs.shape.clone(),
        out,
    )?)))
}

/// Same-shape F32⊗F32 comparison fast path producing a `DType::Bool` tensor.
/// JAX's default float dtype — `x > thresh`/mask idioms run on f32. Bit-for-bit
/// identical to the generic `compare_literals` path, which for F32 operands widens
/// each to f64 (`as_f64()` = `f64::from(f32)` — exact, NaN preserved) and applies
/// `float_cmp`; this does the same off the packed `as_f32_slice` backing into a
/// dense `Vec<bool>` (1 byte/elem vs 24). Returns `None` if not dense F32.
#[inline]
fn eval_same_shape_f32_compare(
    lhs: &TensorValue,
    rhs: &TensorValue,
    float_cmp: &impl Fn(f64, f64) -> bool,
) -> Result<Option<Value>, EvalError> {
    let (Some(left), Some(right)) = (lhs.elements.as_f32_slice(), rhs.elements.as_f32_slice())
    else {
        return Ok(None);
    };
    let out: Vec<bool> = left
        .iter()
        .zip(right)
        .map(|(&a, &b)| float_cmp(f64::from(a), f64::from(b)))
        .collect();
    Ok(Some(Value::Tensor(TensorValue::new_bool_values(
        lhs.shape.clone(),
        out,
    )?)))
}

/// Same-shape BF16/F16 comparison fast path producing a `DType::Bool` tensor.
/// Half-float is the dominant mixed-precision training dtype and `x > thresh`
/// mask idioms (ReLU/dropout masks) run on it. Bit-for-bit identical to the
/// generic `compare_literals` path, which for half operands (`literal_to_i128`
/// returns `None`) widens each to f64 via `Literal::{BF16,F16}Bits::as_f64()`
/// (exact, order- and NaN-preserving) and applies `float_cmp`; this does the same
/// off the packed `as_half_float_slice` (u16) backing into a dense `Vec<bool>`
/// (1 byte/elem vs 24). Both operands must share the half dtype. Returns `None`
/// otherwise (caller falls through to the generic per-`Literal` loop).
#[inline]
fn eval_same_shape_half_float_compare(
    lhs: &TensorValue,
    rhs: &TensorValue,
    float_cmp: &impl Fn(f64, f64) -> bool,
) -> Result<Option<Value>, EvalError> {
    if lhs.dtype != rhs.dtype {
        return Ok(None);
    }
    let (Some(left), Some(right)) = (
        lhs.elements.as_half_float_slice(),
        rhs.elements.as_half_float_slice(),
    ) else {
        return Ok(None);
    };
    let decode: fn(u16) -> f64 = match lhs.dtype {
        DType::BF16 => |b| Literal::BF16Bits(b).as_f64().unwrap_or(f64::NAN),
        DType::F16 => |b| Literal::F16Bits(b).as_f64().unwrap_or(f64::NAN),
        _ => return Ok(None),
    };
    let out: Vec<bool> = left
        .iter()
        .zip(right)
        .map(|(&a, &b)| float_cmp(decode(a), decode(b)))
        .collect();
    Ok(Some(Value::Tensor(TensorValue::new_bool_values(
        lhs.shape.clone(),
        out,
    )?)))
}

/// Same-shape I64 x I64 comparison fast path producing a `DType::Bool` tensor.
///
/// This is identical to the generic `compare_literals` path for `I64` pairs:
/// that path converts both literals losslessly to `i128` and applies the same
/// integer comparison closure. It only skips per-element enum dispatch and
/// float/complex probes.
#[inline]
fn eval_same_shape_i64_compare(
    lhs: &TensorValue,
    rhs: &TensorValue,
    int_cmp: &impl Fn(i128, i128) -> bool,
) -> Result<Option<Value>, EvalError> {
    // Dense i64 fast path: read both contiguous `i64` backing slices directly,
    // skipping the per-element `Literal::I64` match + 24-byte enum stride.
    // Bit-for-bit identical to the generic loop below — same `int_cmp` applied
    // to `i128::from(value)` in the same element order, same Bool output.
    // `as_i64_slice()` is `Some` only for I64 dense storage. (The Bool output is
    // still `Vec<Literal>` — no dense Bool storage exists — so the win is on the
    // input side.)
    if let (Some(left_values), Some(right_values)) =
        (lhs.elements.as_i64_slice(), rhs.elements.as_i64_slice())
    {
        let out: Vec<bool> = left_values
            .iter()
            .zip(right_values)
            .map(|(&left, &right)| int_cmp(i128::from(left), i128::from(right)))
            .collect();
        return Ok(Some(Value::Tensor(TensorValue::new_bool_values(
            lhs.shape.clone(),
            out,
        )?)));
    }

    let mut out = Vec::with_capacity(lhs.elements.len());
    for (left, right) in lhs.elements.iter().zip(&rhs.elements) {
        let (Literal::I64(left), Literal::I64(right)) = (*left, *right) else {
            return Ok(None);
        };
        out.push(int_cmp(i128::from(left), i128::from(right)));
    }

    Ok(Some(Value::Tensor(TensorValue::new_bool_values(
        lhs.shape.clone(),
        out,
    )?)))
}

/// I64 scalar⊗tensor broadcast comparison fast path producing a `DType::Bool`
/// tensor. `scalar_on_left` preserves operand order for asymmetric ordered
/// comparisons. Bit-for-bit identical to the generic `compare_literals` path for
/// I64 operands (`int_cmp(i128::from(a), i128::from(b))`). Returns `Ok(None)`
/// unless the scalar is `I64` and the tensor is I64 dense storage.
#[inline]
fn eval_i64_scalar_compare(
    scalar: Literal,
    tensor: &TensorValue,
    scalar_on_left: bool,
    int_cmp: &impl Fn(i128, i128) -> bool,
) -> Result<Option<Value>, EvalError> {
    let Literal::I64(scalar) = scalar else {
        return Ok(None);
    };
    // I64 and I32 both store sign-extended i64 values, so `int_cmp(i128::from(..))` is
    // identical; the bool output needs no narrowing. I32 is JAX's default int.
    if !matches!(tensor.dtype, DType::I64 | DType::I32) {
        return Ok(None);
    }
    let Some(values) = tensor.elements.as_i64_slice() else {
        return Ok(None);
    };
    let scalar = i128::from(scalar);
    let out: Vec<bool> = values
        .iter()
        .map(|&v| {
            let value = i128::from(v);
            if scalar_on_left {
                int_cmp(scalar, value)
            } else {
                int_cmp(value, scalar)
            }
        })
        .collect();
    Ok(Some(Value::Tensor(TensorValue::new_bool_values(
        tensor.shape.clone(),
        out,
    )?)))
}

/// Dense same-shape comparison for unsigned (`u32`/`u64`) operands. Reads the
/// packed `as_u32_slice`/`as_u64_slice` backing directly. Bit-for-bit identical
/// to the generic `compare_literals` path: that path routes `U32`/`U64` through
/// `literal_to_i128` (always non-negative for unsigned), so the comparison is
/// `int_cmp(i128::from(left), i128::from(right))` in the same element order with
/// no signed/unsigned ambiguity. Returns `Ok(None)` unless both operands share
/// the same unsigned dense backing.
#[inline]
fn eval_same_shape_unsigned_compare(
    lhs: &TensorValue,
    rhs: &TensorValue,
    int_cmp: &impl Fn(i128, i128) -> bool,
) -> Result<Option<Value>, EvalError> {
    if let (Some(left), Some(right)) = (lhs.elements.as_u32_slice(), rhs.elements.as_u32_slice()) {
        let out: Vec<bool> = left
            .iter()
            .zip(right)
            .map(|(&l, &r)| int_cmp(i128::from(l), i128::from(r)))
            .collect();
        return Ok(Some(Value::Tensor(TensorValue::new_bool_values(
            lhs.shape.clone(),
            out,
        )?)));
    }
    if let (Some(left), Some(right)) = (lhs.elements.as_u64_slice(), rhs.elements.as_u64_slice()) {
        let out: Vec<bool> = left
            .iter()
            .zip(right)
            .map(|(&l, &r)| int_cmp(i128::from(l), i128::from(r)))
            .collect();
        return Ok(Some(Value::Tensor(TensorValue::new_bool_values(
            lhs.shape.clone(),
            out,
        )?)));
    }
    Ok(None)
}

/// Unsigned (`u32`/`u64`) scalar⊗tensor broadcast comparison fast path. Mirror of
/// [`eval_i64_scalar_compare`] for unsigned dense storage; `scalar_on_left`
/// preserves operand order. Bit-for-bit identical to the generic path
/// (`int_cmp(i128::from(..))`, unsigned values always non-negative). Returns
/// `Ok(None)` unless the scalar and tensor share the same unsigned dtype/backing.
#[inline]
fn eval_unsigned_scalar_compare(
    scalar: Literal,
    tensor: &TensorValue,
    scalar_on_left: bool,
    int_cmp: &impl Fn(i128, i128) -> bool,
) -> Result<Option<Value>, EvalError> {
    if let (Literal::U32(scalar), Some(values)) = (scalar, tensor.elements.as_u32_slice()) {
        let scalar = i128::from(scalar);
        let out: Vec<bool> = values
            .iter()
            .map(|&v| {
                let value = i128::from(v);
                if scalar_on_left {
                    int_cmp(scalar, value)
                } else {
                    int_cmp(value, scalar)
                }
            })
            .collect();
        return Ok(Some(Value::Tensor(TensorValue::new_bool_values(
            tensor.shape.clone(),
            out,
        )?)));
    }
    if let (Literal::U64(scalar), Some(values)) = (scalar, tensor.elements.as_u64_slice()) {
        let scalar = i128::from(scalar);
        let out: Vec<bool> = values
            .iter()
            .map(|&v| {
                let value = i128::from(v);
                if scalar_on_left {
                    int_cmp(scalar, value)
                } else {
                    int_cmp(value, scalar)
                }
            })
            .collect();
        return Ok(Some(Value::Tensor(TensorValue::new_bool_values(
            tensor.shape.clone(),
            out,
        )?)));
    }
    Ok(None)
}

/// F64 scalar/tensor broadcast comparison fast path producing a `DType::Bool`
/// tensor. `scalar_on_left` preserves operand order (`Scalar ⊗ Tensor` vs
/// `Tensor ⊗ Scalar`) for the asymmetric ordered comparisons. Bit-for-bit
/// identical to the generic `compare_literals` path for F64 operands. Returns
/// `Ok(None)` for non-F64 scalar/elements so the caller uses the generic path.
#[inline]
fn eval_f64_scalar_compare(
    scalar: Literal,
    tensor: &TensorValue,
    scalar_on_left: bool,
    float_cmp: &impl Fn(f64, f64) -> bool,
) -> Result<Option<Value>, EvalError> {
    let Literal::F64Bits(scalar_bits) = scalar else {
        return Ok(None);
    };
    if tensor.dtype != DType::F64 {
        return Ok(None);
    }
    let scalar = f64::from_bits(scalar_bits);
    // Dense F64 path: contiguous map into a dense Bool backing.
    if let Some(values) = tensor.elements.as_f64_slice() {
        let out: Vec<bool> = values
            .iter()
            .map(|&value| {
                if scalar_on_left {
                    float_cmp(scalar, value)
                } else {
                    float_cmp(value, scalar)
                }
            })
            .collect();
        return Ok(Some(Value::Tensor(TensorValue::new_bool_values(
            tensor.shape.clone(),
            out,
        )?)));
    }

    let mut out = Vec::with_capacity(tensor.elements.len());
    for &elem in &tensor.elements {
        let Literal::F64Bits(bits) = elem else {
            return Ok(None);
        };
        let value = f64::from_bits(bits);
        out.push(if scalar_on_left {
            float_cmp(scalar, value)
        } else {
            float_cmp(value, scalar)
        });
    }

    Ok(Some(Value::Tensor(TensorValue::new_bool_values(
        tensor.shape.clone(),
        out,
    )?)))
}

/// F32-scalar ⊗ dense-F32-tensor comparison fast path (`x > 0.0` relu/threshold
/// masks). Mirrors `eval_f64_scalar_compare`: the generic `compare_literals` path
/// widens both F32 operands to f64 then `float_cmp`, so widening the scalar and
/// each tap with `f64::from` is bit-for-bit identical. Returns `None` unless the
/// scalar is `F32Bits` and the tensor is dense F32.
fn eval_f32_scalar_compare(
    scalar: Literal,
    tensor: &TensorValue,
    scalar_on_left: bool,
    float_cmp: &impl Fn(f64, f64) -> bool,
) -> Result<Option<Value>, EvalError> {
    let Literal::F32Bits(scalar_bits) = scalar else {
        return Ok(None);
    };
    if tensor.dtype != DType::F32 {
        return Ok(None);
    }
    let Some(values) = tensor.elements.as_f32_slice() else {
        return Ok(None);
    };
    let scalar = f64::from(f32::from_bits(scalar_bits));
    let out: Vec<bool> = values
        .iter()
        .map(|&value| {
            let v = f64::from(value);
            if scalar_on_left {
                float_cmp(scalar, v)
            } else {
                float_cmp(v, scalar)
            }
        })
        .collect();
    Ok(Some(Value::Tensor(TensorValue::new_bool_values(
        tensor.shape.clone(),
        out,
    )?)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn f64_compare_fast_paths_bit_identical_to_scalar() {
        // Same-shape and both scalar-broadcast orders, all six comparisons,
        // with NaN / +-inf / signed zero, must match the per-element scalar
        // comparison exactly (Bool output).
        let lhs = [
            1.5,
            -0.0,
            f64::INFINITY,
            f64::NAN,
            7.0,
            -3.25,
            0.0,
            f64::NEG_INFINITY,
        ];
        let rhs = [2.0, 0.0, -4.0, 5.0, 7.0, -3.25, f64::NAN, f64::NEG_INFINITY];
        let scalar = 0.0_f64;
        let params = BTreeMap::new();
        for p in [
            Primitive::Eq,
            Primitive::Ne,
            Primitive::Lt,
            Primitive::Le,
            Primitive::Gt,
            Primitive::Ge,
        ] {
            let fcmp = |a: f64, b: f64| match p {
                Primitive::Eq => a == b,
                Primitive::Ne => a != b,
                Primitive::Lt => a < b,
                Primitive::Le => a <= b,
                Primitive::Gt => a > b,
                Primitive::Ge => a >= b,
                _ => unreachable!(),
            };
            let extract = |v: Value| -> Vec<bool> {
                let Value::Tensor(t) = v else {
                    panic!("expected tensor for {p:?}");
                };
                assert_eq!(t.dtype, DType::Bool);
                t.elements
                    .iter()
                    .map(|e| matches!(e, Literal::Bool(true)))
                    .collect()
            };

            // same-shape
            let got =
                extract(crate::eval_primitive(p, &[v_f64(&lhs), v_f64(&rhs)], &params).unwrap());
            let want: Vec<bool> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(&a, &b)| fcmp(a, b))
                .collect();
            assert_eq!(got, want, "{p:?} same-shape mismatch");

            // scalar on left: Scalar ⊗ Tensor
            let got =
                extract(crate::eval_primitive(p, &[s_f64(scalar), v_f64(&rhs)], &params).unwrap());
            let want: Vec<bool> = rhs.iter().map(|&b| fcmp(scalar, b)).collect();
            assert_eq!(got, want, "{p:?} scalar-left mismatch");

            // scalar on right: Tensor ⊗ Scalar
            let got =
                extract(crate::eval_primitive(p, &[v_f64(&lhs), s_f64(scalar)], &params).unwrap());
            let want: Vec<bool> = lhs.iter().map(|&a| fcmp(a, scalar)).collect();
            assert_eq!(got, want, "{p:?} scalar-right mismatch");
        }
    }

    // i32 (densified) tensor: as_i64_slice is Some → dense compare path.
    fn v_i32_dense(data: &[i64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::I32,
                Shape::vector(data.len() as u32),
                data.iter().map(|&v| Literal::I64(v)).collect(),
            )
            .unwrap(),
        )
    }
    // Boxed i32 tensor (as_i64_slice None) → generic per-Literal path, for A/B + iso.
    fn v_i32_boxed(data: &[i64]) -> Value {
        Value::Tensor(
            TensorValue::new_with_literal_buffer(
                DType::I32,
                Shape::vector(data.len() as u32),
                fj_core::LiteralBuffer::new(data.iter().map(|&v| Literal::I64(v)).collect()),
            )
            .unwrap(),
        )
    }

    #[test]
    fn i32_compare_dense_matches_generic() {
        // i32 same-shape AND scalar compares now take the dense i64 stencil (were on the
        // generic per-Literal path). Dense (densified I32, as_i64_slice Some) must be
        // bit-identical to the generic boxed path, across all six comparisons, both
        // scalar-broadcast orders, incl i32::MIN/MAX boundary values. Bool output.
        let lhs = [
            1i64,
            -5,
            i64::from(i32::MAX),
            i64::from(i32::MIN),
            7,
            0,
            -3,
            42,
        ];
        let rhs = [2i64, -5, 4, i64::from(i32::MIN), 7, -1, -3, 100];
        let (ld, lb) = (v_i32_dense(&lhs), v_i32_boxed(&lhs));
        let (rd, rb) = (v_i32_dense(&rhs), v_i32_boxed(&rhs));
        assert!(ld.as_tensor().unwrap().elements.as_i64_slice().is_some());
        assert!(lb.as_tensor().unwrap().elements.as_i64_slice().is_none());
        let params = BTreeMap::new();
        let s = Value::Scalar(Literal::I64(0));
        for p in [
            Primitive::Eq,
            Primitive::Ne,
            Primitive::Lt,
            Primitive::Le,
            Primitive::Gt,
            Primitive::Ge,
        ] {
            let d = extract_bools(
                &crate::eval_primitive(p, &[ld.clone(), rd.clone()], &params).unwrap(),
            );
            let g = extract_bools(
                &crate::eval_primitive(p, &[lb.clone(), rb.clone()], &params).unwrap(),
            );
            assert_eq!(d, g, "{p:?} same-shape i32 dense!=generic");

            let d = extract_bools(
                &crate::eval_primitive(p, &[ld.clone(), s.clone()], &params).unwrap(),
            );
            let g = extract_bools(
                &crate::eval_primitive(p, &[lb.clone(), s.clone()], &params).unwrap(),
            );
            assert_eq!(d, g, "{p:?} tensor⊗scalar i32 dense!=generic");

            let d = extract_bools(
                &crate::eval_primitive(p, &[s.clone(), rd.clone()], &params).unwrap(),
            );
            let g = extract_bools(
                &crate::eval_primitive(p, &[s.clone(), rb.clone()], &params).unwrap(),
            );
            assert_eq!(d, g, "{p:?} scalar⊗tensor i32 dense!=generic");
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_i32_compare_dense_vs_generic() {
        use std::time::Instant;
        let n = 1_000_000usize;
        let lhs: Vec<i64> = (0..n)
            .map(|i| i64::from((i as i32).wrapping_mul(2_654_435_761u32 as i32)))
            .collect();
        let rhs: Vec<i64> = (0..n)
            .map(|i| i64::from((i as i32).wrapping_mul(40_503)))
            .collect();
        let (ld, lb) = (v_i32_dense(&lhs), v_i32_boxed(&lhs));
        let (rd, rb) = (v_i32_dense(&rhs), v_i32_boxed(&rhs));
        let params = BTreeMap::new();
        let best = |mut f: Box<dyn FnMut() -> usize>| {
            f();
            let mut b = f64::MAX;
            for _ in 0..7 {
                let t = Instant::now();
                std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let (lbc, rbc) = (lb.clone(), rb.clone());
        let generic = best(Box::new(move || {
            crate::eval_primitive(Primitive::Lt, &[lbc.clone(), rbc.clone()], &params)
                .unwrap()
                .as_tensor()
                .unwrap()
                .elements
                .len()
        }));
        let params2 = BTreeMap::new();
        let (ldc, rdc) = (ld.clone(), rd.clone());
        let dense = best(Box::new(move || {
            crate::eval_primitive(Primitive::Lt, &[ldc.clone(), rdc.clone()], &params2)
                .unwrap()
                .as_tensor()
                .unwrap()
                .elements
                .len()
        }));
        println!(
            "BENCH i32 same-shape Lt [1e6]: generic={:.2}ms dense={:.2}ms speedup={:.2}x",
            generic * 1e3,
            dense * 1e3,
            generic / dense
        );
    }

    // u32 (densified) tensor: as_u32_slice is Some → dense unsigned compare path.
    fn v_u32_dense(data: &[u32]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::U32,
                Shape::vector(data.len() as u32),
                data.iter().map(|&v| Literal::U32(v)).collect(),
            )
            .unwrap(),
        )
    }
    // Boxed u32 tensor (as_u32_slice None) → generic per-Literal path, for A/B + iso.
    fn v_u32_boxed(data: &[u32]) -> Value {
        Value::Tensor(
            TensorValue::new_with_literal_buffer(
                DType::U32,
                Shape::vector(data.len() as u32),
                fj_core::LiteralBuffer::new(data.iter().map(|&v| Literal::U32(v)).collect()),
            )
            .unwrap(),
        )
    }
    fn v_u64_dense(data: &[u64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::U64,
                Shape::vector(data.len() as u32),
                data.iter().map(|&v| Literal::U64(v)).collect(),
            )
            .unwrap(),
        )
    }
    fn v_u64_boxed(data: &[u64]) -> Value {
        Value::Tensor(
            TensorValue::new_with_literal_buffer(
                DType::U64,
                Shape::vector(data.len() as u32),
                fj_core::LiteralBuffer::new(data.iter().map(|&v| Literal::U64(v)).collect()),
            )
            .unwrap(),
        )
    }

    #[test]
    fn u32_u64_compare_dense_matches_generic() {
        // u32/u64 same-shape AND scalar compares now take the dense unsigned path
        // (were on the generic per-Literal path). Dense must be bit-identical to the
        // boxed generic path across all six comparisons + both scalar orders, incl
        // high-bit-set values (>i32::MAX / >i64::MAX) to prove UNSIGNED ordering.
        let params = BTreeMap::new();

        // u32: include values above i32::MAX to catch any sign-extension bug.
        let lu32 = [0u32, 1, 7, u32::MAX, 3_000_000_000, 100, u32::MAX - 1, 42];
        let ru32 = [2u32, 1, 4, u32::MAX - 1, 3_000_000_000, 99, u32::MAX, 41];
        let (ld, lb) = (v_u32_dense(&lu32), v_u32_boxed(&lu32));
        let (rd, rb) = (v_u32_dense(&ru32), v_u32_boxed(&ru32));
        assert!(ld.as_tensor().unwrap().elements.as_u32_slice().is_some());
        assert!(lb.as_tensor().unwrap().elements.as_u32_slice().is_none());
        let s32 = Value::Scalar(Literal::U32(3_000_000_000));

        // u64: include values above i64::MAX.
        let lu64 = [
            0u64,
            1,
            u64::MAX,
            u32::MAX as u64 + 1,
            1 << 63,
            7,
            u64::MAX - 1,
            5,
        ];
        let ru64 = [
            2u64,
            1,
            u64::MAX - 1,
            u32::MAX as u64 + 2,
            1 << 63,
            6,
            u64::MAX,
            5,
        ];
        let (ld64, lb64) = (v_u64_dense(&lu64), v_u64_boxed(&lu64));
        let (rd64, rb64) = (v_u64_dense(&ru64), v_u64_boxed(&ru64));
        assert!(ld64.as_tensor().unwrap().elements.as_u64_slice().is_some());
        let s64 = Value::Scalar(Literal::U64(1 << 63));

        for p in [
            Primitive::Eq,
            Primitive::Ne,
            Primitive::Lt,
            Primitive::Le,
            Primitive::Gt,
            Primitive::Ge,
        ] {
            for (ld, lb, rd, rb, s) in [
                (&ld, &lb, &rd, &rb, &s32),
                (&ld64, &lb64, &rd64, &rb64, &s64),
            ] {
                let d = extract_bools(
                    &crate::eval_primitive(p, &[ld.clone(), rd.clone()], &params).unwrap(),
                );
                let g = extract_bools(
                    &crate::eval_primitive(p, &[lb.clone(), rb.clone()], &params).unwrap(),
                );
                assert_eq!(d, g, "{p:?} same-shape unsigned dense!=generic");

                let d = extract_bools(
                    &crate::eval_primitive(p, &[ld.clone(), s.clone()], &params).unwrap(),
                );
                let g = extract_bools(
                    &crate::eval_primitive(p, &[lb.clone(), s.clone()], &params).unwrap(),
                );
                assert_eq!(d, g, "{p:?} tensor⊗scalar unsigned dense!=generic");

                let d = extract_bools(
                    &crate::eval_primitive(p, &[s.clone(), rd.clone()], &params).unwrap(),
                );
                let g = extract_bools(
                    &crate::eval_primitive(p, &[s.clone(), rb.clone()], &params).unwrap(),
                );
                assert_eq!(d, g, "{p:?} scalar⊗tensor unsigned dense!=generic");
            }
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_u32_compare_dense_vs_generic() {
        use std::time::Instant;
        let n = 1_000_000usize;
        let lhs: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        let rhs: Vec<u32> = (0..n).map(|i| (i as u32).wrapping_mul(40_503)).collect();
        let (ld, lb) = (v_u32_dense(&lhs), v_u32_boxed(&lhs));
        let (rd, rb) = (v_u32_dense(&rhs), v_u32_boxed(&rhs));
        let params = BTreeMap::new();
        let best = |mut f: Box<dyn FnMut() -> usize>| {
            f();
            let mut b = f64::MAX;
            for _ in 0..7 {
                let t = Instant::now();
                std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let (lbc, rbc) = (lb.clone(), rb.clone());
        let generic = best(Box::new(move || {
            crate::eval_primitive(Primitive::Lt, &[lbc.clone(), rbc.clone()], &params)
                .unwrap()
                .as_tensor()
                .unwrap()
                .elements
                .len()
        }));
        let params2 = BTreeMap::new();
        let (ldc, rdc) = (ld.clone(), rd.clone());
        let dense = best(Box::new(move || {
            crate::eval_primitive(Primitive::Lt, &[ldc.clone(), rdc.clone()], &params2)
                .unwrap()
                .as_tensor()
                .unwrap()
                .elements
                .len()
        }));
        println!(
            "BENCH u32 same-shape Lt [1e6]: generic={:.2}ms dense={:.2}ms speedup={:.2}x",
            generic * 1e3,
            dense * 1e3,
            generic / dense
        );
    }

    #[test]
    fn f64_compare_word_masks_match_literal_path_at_word_boundaries() {
        let params = BTreeMap::new();
        for len in [0usize, 1, 63, 64, 65, 127, 128, 129] {
            let lhs: Vec<f64> = (0..len)
                .map(|i| match i % 11 {
                    0 => f64::NAN,
                    1 => -0.0,
                    2 => 0.0,
                    3 => f64::INFINITY,
                    4 => f64::NEG_INFINITY,
                    _ => i as f64 * 0.25 - 17.0,
                })
                .collect();
            let rhs: Vec<f64> = (0..len)
                .map(|i| match i % 13 {
                    0 => f64::NAN,
                    1 => 0.0,
                    2 => -0.0,
                    3 => f64::NEG_INFINITY,
                    4 => f64::INFINITY,
                    _ => ((i * 7 + 3) % 97) as f64 * 0.5 - 23.0,
                })
                .collect();
            let shape = Shape::vector(len as u32);
            let dense = |data: &[f64]| {
                Value::Tensor(TensorValue::new_f64_values(shape.clone(), data.to_vec()).unwrap())
            };
            let literal = |data: &[f64]| {
                Value::Tensor(
                    TensorValue::new(
                        DType::F64,
                        shape.clone(),
                        data.iter().copied().map(Literal::from_f64).collect(),
                    )
                    .unwrap(),
                )
            };

            for prim in [
                Primitive::Eq,
                Primitive::Ne,
                Primitive::Lt,
                Primitive::Le,
                Primitive::Gt,
                Primitive::Ge,
            ] {
                let word_backed =
                    crate::eval_primitive(prim, &[dense(&lhs), dense(&rhs)], &params).unwrap();
                assert!(
                    word_backed
                        .as_tensor()
                        .unwrap()
                        .elements
                        .as_bool_words()
                        .is_some(),
                    "{prim:?} len={len} must stay word-backed"
                );
                let boxed =
                    crate::eval_primitive(prim, &[literal(&lhs), literal(&rhs)], &params).unwrap();
                assert_eq!(
                    extract_bools(&word_backed),
                    extract_bools(&boxed),
                    "{prim:?} len={len}"
                );
            }
        }
    }

    #[test]
    fn f32_compare_dense_bit_identical_to_generic() {
        // Dense F32 (as_f32_slice) takes the new fast paths; boxed F32 Literals take
        // the generic compare_literals loop. Bool outputs must match exactly across
        // same-shape / scalar-both-orders / broadcast, all six comparisons, incl
        // NaN/+-inf/+-0 (f32->f64 widening is exact + NaN-preserving).
        let la = [
            1.5f32,
            -0.0,
            f32::INFINITY,
            f32::NAN,
            7.0,
            -3.25,
            0.0,
            f32::NEG_INFINITY,
        ];
        let ra = [
            2.0f32,
            0.0,
            -4.0,
            5.0,
            7.0,
            -3.25,
            f32::NAN,
            f32::NEG_INFINITY,
        ];
        let dims = vec![la.len() as u32];
        let dense = |d: &[f32]| {
            Value::Tensor(
                TensorValue::new_f32_values(Shape { dims: dims.clone() }, d.to_vec()).unwrap(),
            )
        };
        let boxed = |d: &[f32]| {
            Value::Tensor(
                crate::new_boxed(
                    DType::F32,
                    Shape { dims: dims.clone() },
                    d.iter().copied().map(Literal::from_f32).collect(),
                )
                .unwrap(),
            )
        };
        let s = Value::Scalar(Literal::from_f32(0.0));
        // broadcast: [n] vs [1]
        let dense_b = Value::Tensor(
            TensorValue::new_f32_values(Shape { dims: vec![1] }, vec![0.0f32]).unwrap(),
        );
        let boxed_b = Value::Tensor(
            crate::new_boxed(
                DType::F32,
                Shape { dims: vec![1] },
                vec![Literal::from_f32(0.0)],
            )
            .unwrap(),
        );
        assert!(
            dense(&la)
                .as_tensor()
                .unwrap()
                .elements
                .as_f32_slice()
                .is_some()
        );
        assert!(
            boxed(&la)
                .as_tensor()
                .unwrap()
                .elements
                .as_f32_slice()
                .is_none()
        );
        let p = BTreeMap::new();
        for prim in [
            Primitive::Eq,
            Primitive::Ne,
            Primitive::Lt,
            Primitive::Le,
            Primitive::Gt,
            Primitive::Ge,
        ] {
            let run = |a: Value, b: Value| {
                extract_bools(&crate::eval_primitive(prim, &[a, b], &p).unwrap())
            };
            assert_eq!(
                run(dense(&la), dense(&ra)),
                run(boxed(&la), boxed(&ra)),
                "{prim:?} same-shape"
            );
            assert_eq!(
                run(s.clone(), dense(&ra)),
                run(s.clone(), boxed(&ra)),
                "{prim:?} scalar-left"
            );
            assert_eq!(
                run(dense(&la), s.clone()),
                run(boxed(&la), s.clone()),
                "{prim:?} scalar-right"
            );
            assert_eq!(
                run(dense(&la), dense_b.clone()),
                run(boxed(&la), boxed_b.clone()),
                "{prim:?} broadcast"
            );
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_f32_compare_scalar_dense_vs_boxed() {
        use std::time::Instant;
        // `x > 0.0` relu/threshold mask over a large f32 tensor.
        let n = 1usize << 22; // 4M
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 2000.0).collect();
        let dims = vec![n as u32];
        let dense = Value::Tensor(
            TensorValue::new_f32_values(Shape { dims: dims.clone() }, data.clone()).unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape { dims: dims.clone() },
                data.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        let s = Value::Scalar(Literal::from_f32(0.0));
        let p = BTreeMap::new();
        let time = |x: &Value| {
            let _ = crate::eval_primitive(Primitive::Gt, &[x.clone(), s.clone()], &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..20 {
                let t = Instant::now();
                let _ = crate::eval_primitive(Primitive::Gt, &[x.clone(), s.clone()], &p).unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let generic = time(&boxed);
        let dense_t = time(&dense);
        println!(
            "BENCH f32 compare (x>0) n={n}: boxed(per-Literal)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    #[test]
    fn i64_compare_fast_path_matches_integer_closure_order() -> Result<(), EvalError> {
        let lhs = [-5, 0, 7, i64::MIN, i64::MAX, 42];
        let rhs = [-5, 1, -7, i64::MAX, i64::MIN, 42];
        let params = BTreeMap::new();

        for p in [
            Primitive::Eq,
            Primitive::Ne,
            Primitive::Lt,
            Primitive::Le,
            Primitive::Gt,
            Primitive::Ge,
        ] {
            let icmp = |a: i128, b: i128| match p {
                Primitive::Eq => a == b,
                Primitive::Ne => a != b,
                Primitive::Lt => a < b,
                Primitive::Le => a <= b,
                Primitive::Gt => a > b,
                Primitive::Ge => a >= b,
                _ => false,
            };
            let fcmp = |a: f64, b: f64| match p {
                Primitive::Eq => a == b,
                Primitive::Ne => a != b,
                Primitive::Lt => a < b,
                Primitive::Le => a <= b,
                Primitive::Gt => a > b,
                Primitive::Ge => a >= b,
                _ => false,
            };
            let want: Vec<bool> = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(&a, &b)| icmp(i128::from(a), i128::from(b)))
                .collect();

            let dispatch_inputs = [v_i64(&lhs)?, v_i64(&rhs)?];
            let via_dispatch = crate::eval_primitive(p, &dispatch_inputs, &params)?;
            assert_eq!(
                extract_bools(&via_dispatch),
                want,
                "{p:?} dispatch i64 same-shape mismatch"
            );

            let direct_inputs = [v_i64(&lhs)?, v_i64(&rhs)?];
            let direct = eval_comparison(p, &direct_inputs, icmp, fcmp)?;
            assert_eq!(
                extract_bools(&direct),
                want,
                "{p:?} direct i64 same-shape mismatch"
            );
        }

        Ok(())
    }

    /// Dense i64 compare fast paths (same-shape + both scalar-broadcast orders)
    /// must be bit-identical to the Vec<Literal> path across all six comparisons,
    /// including i64::MIN/MAX. Dense input via vector_i64, literal via v_i64.
    #[test]
    fn dense_i64_compare_bit_identical_to_literal_path() {
        let data: Vec<i64> = vec![-5, 0, 7, i64::MIN, i64::MAX, 42, -1, 1000];
        let other: Vec<i64> = vec![-5, 1, -7, i64::MAX, i64::MIN, 42, 1000, -1];
        let params = BTreeMap::new();
        let dense = || Value::vector_i64(&data).unwrap();
        let dense_other = || Value::vector_i64(&other).unwrap();
        let lit = || v_i64(&data).unwrap();
        let lit_other = || v_i64(&other).unwrap();
        let scalar_d = Value::scalar_i64(42);
        let scalar_l = Value::Scalar(Literal::I64(42));

        for p in [
            Primitive::Eq,
            Primitive::Ne,
            Primitive::Lt,
            Primitive::Le,
            Primitive::Gt,
            Primitive::Ge,
        ] {
            // same-shape: assert dense input has dense storage, then compare.
            assert!(
                dense()
                    .as_tensor()
                    .unwrap()
                    .elements
                    .as_i64_slice()
                    .is_some()
            );
            assert!(lit().as_tensor().unwrap().elements.as_i64_slice().is_none());
            let d = extract_bools(
                &crate::eval_primitive(p, &[dense(), dense_other()], &params).unwrap(),
            );
            let l =
                extract_bools(&crate::eval_primitive(p, &[lit(), lit_other()], &params).unwrap());
            assert_eq!(d, l, "{p:?} same-shape dense vs literal");

            // Tensor ⊗ Scalar
            let d = extract_bools(
                &crate::eval_primitive(p, &[dense(), scalar_d.clone()], &params).unwrap(),
            );
            let l = extract_bools(
                &crate::eval_primitive(p, &[lit(), scalar_l.clone()], &params).unwrap(),
            );
            assert_eq!(d, l, "{p:?} tensor⊗scalar dense vs literal");

            // Scalar ⊗ Tensor
            let d = extract_bools(
                &crate::eval_primitive(p, &[scalar_d.clone(), dense()], &params).unwrap(),
            );
            let l = extract_bools(
                &crate::eval_primitive(p, &[scalar_l.clone(), lit()], &params).unwrap(),
            );
            assert_eq!(d, l, "{p:?} scalar⊗tensor dense vs literal");
        }
    }

    /// Bit-exact parity for the dense multi-dim broadcast-compare fast paths
    /// (f64 + i64) via the BroadcastOdometer, across several broadcast shapes and
    /// all six comparisons, vs the Vec<Literal>-backed generic broadcast loop.
    /// f64 includes NaN/+-inf/signed-zero.
    #[test]
    fn dense_broadcast_compare_bit_identical_to_literal_path() {
        let shapes: [(Vec<u32>, Vec<u32>); 4] = [
            (vec![4, 5], vec![5]),
            (vec![4, 5], vec![4, 1]),
            (vec![3, 1], vec![1, 6]),
            (vec![2, 3, 4], vec![4]),
        ];
        let prod = |d: &[u32]| d.iter().map(|&x| x as usize).product::<usize>();
        let params = BTreeMap::new();

        for (ls, rs) in shapes {
            let ln = prod(&ls);
            let rn = prod(&rs);
            let lf: Vec<f64> = (0..ln)
                .map(|i| match i % 5 {
                    0 => f64::NAN,
                    1 => f64::INFINITY,
                    2 => -0.0,
                    3 => f64::NEG_INFINITY,
                    _ => (i as f64 - 3.0) * 0.5,
                })
                .collect();
            let rf: Vec<f64> = (0..rn).map(|i| (i as f64 - 2.0) * 0.5).collect();
            let li: Vec<i64> = (0..ln as i64).map(|i| i - 4).collect();
            let ri: Vec<i64> = (0..rn as i64).map(|i| i - 2).collect();

            let dense_f = |d: &[f64], s: &[u32]| {
                Value::Tensor(
                    TensorValue::new_f64_values(Shape { dims: s.to_vec() }, d.to_vec()).unwrap(),
                )
            };
            let lit_f = |d: &[f64], s: &[u32]| {
                Value::Tensor(
                    TensorValue::new(
                        DType::F64,
                        Shape { dims: s.to_vec() },
                        d.iter().copied().map(Literal::from_f64).collect(),
                    )
                    .unwrap(),
                )
            };
            let dense_i = |d: &[i64], s: &[u32]| {
                Value::Tensor(
                    TensorValue::new_i64_values(Shape { dims: s.to_vec() }, d.to_vec()).unwrap(),
                )
            };
            let lit_i = |d: &[i64], s: &[u32]| {
                Value::Tensor(
                    TensorValue::new(
                        DType::I64,
                        Shape { dims: s.to_vec() },
                        d.iter().copied().map(Literal::I64).collect(),
                    )
                    .unwrap(),
                )
            };

            for p in [
                Primitive::Eq,
                Primitive::Ne,
                Primitive::Lt,
                Primitive::Le,
                Primitive::Gt,
                Primitive::Ge,
            ] {
                let df = extract_bools(
                    &crate::eval_primitive(p, &[dense_f(&lf, &ls), dense_f(&rf, &rs)], &params)
                        .unwrap(),
                );
                let lfr = extract_bools(
                    &crate::eval_primitive(p, &[lit_f(&lf, &ls), lit_f(&rf, &rs)], &params)
                        .unwrap(),
                );
                assert_eq!(df, lfr, "f64 {p:?} {ls:?} {rs:?}");

                let di = extract_bools(
                    &crate::eval_primitive(p, &[dense_i(&li, &ls), dense_i(&ri, &rs)], &params)
                        .unwrap(),
                );
                let lir = extract_bools(
                    &crate::eval_primitive(p, &[lit_i(&li, &ls), lit_i(&ri, &rs)], &params)
                        .unwrap(),
                );
                assert_eq!(di, lir, "i64 {p:?} {ls:?} {rs:?}");
            }
        }
    }

    #[test]
    fn f64_row_broadcast_compare_bit_identical_to_literal_path() {
        let matrix_shape = Shape { dims: vec![4, 5] };
        let row_shape = Shape { dims: vec![5] };
        let matrix: Vec<f64> = (0..20)
            .map(|i| match i % 7 {
                0 => f64::NAN,
                1 => -0.0,
                2 => f64::INFINITY,
                3 => f64::NEG_INFINITY,
                _ => (i as f64 - 8.0) * 0.25,
            })
            .collect();
        let row = vec![f64::NAN, 0.0, f64::NEG_INFINITY, 0.75, f64::INFINITY];
        let dense = |shape: Shape, data: &[f64]| {
            Value::Tensor(TensorValue::new_f64_values(shape, data.to_vec()).unwrap())
        };
        let literal = |shape: Shape, data: &[f64]| {
            Value::Tensor(
                TensorValue::new(
                    DType::F64,
                    shape,
                    data.iter().copied().map(Literal::from_f64).collect(),
                )
                .unwrap(),
            )
        };
        let params = BTreeMap::new();
        let mut golden_rows = Vec::new();

        for (label, dense_lhs, dense_rhs, lit_lhs, lit_rhs) in [
            (
                "matrix_row",
                dense(matrix_shape.clone(), &matrix),
                dense(row_shape.clone(), &row),
                literal(matrix_shape.clone(), &matrix),
                literal(row_shape.clone(), &row),
            ),
            (
                "row_matrix",
                dense(row_shape.clone(), &row),
                dense(matrix_shape.clone(), &matrix),
                literal(row_shape.clone(), &row),
                literal(matrix_shape.clone(), &matrix),
            ),
        ] {
            for prim in [
                Primitive::Eq,
                Primitive::Ne,
                Primitive::Lt,
                Primitive::Le,
                Primitive::Gt,
                Primitive::Ge,
            ] {
                let dense_out = extract_bools(
                    &crate::eval_primitive(prim, &[dense_lhs.clone(), dense_rhs.clone()], &params)
                        .unwrap(),
                );
                let literal_out = extract_bools(
                    &crate::eval_primitive(prim, &[lit_lhs.clone(), lit_rhs.clone()], &params)
                        .unwrap(),
                );
                assert_eq!(dense_out, literal_out, "{label} {prim:?}");
                golden_rows.push(format!("{label}:{prim:?}:{dense_out:?}"));
            }
        }

        let digest = fj_test_utils::fixture_id_from_json(&golden_rows).expect("sha256 digest");
        assert_eq!(
            digest,
            "fd7293300699f850c5fd274a548f8ee215b9cc4f3b75e86cfbea2fa0e02e00c6"
        );
    }

    fn s_f64(v: f64) -> Value {
        Value::Scalar(Literal::from_f64(v))
    }
    fn s_i64(v: i64) -> Value {
        Value::Scalar(Literal::I64(v))
    }
    fn v_f64(data: &[f64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::F64,
                fj_core::Shape {
                    dims: vec![data.len() as u32],
                },
                data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        )
    }
    // Boxed (Vec<Literal>) i64 reference: TensorValue::new now densifies all-I64
    // inputs into dense storage (fj-core i64-densify), so build the boxed buffer
    // explicitly via new_with_literal_buffer to keep exercising the generic path.
    fn v_i64(data: &[i64]) -> Result<Value, EvalError> {
        Ok(Value::Tensor(TensorValue::new_with_literal_buffer(
            DType::I64,
            fj_core::Shape {
                dims: vec![data.len() as u32],
            },
            fj_core::LiteralBuffer::new(data.iter().copied().map(Literal::I64).collect()),
        )?))
    }
    fn extract_bool(val: &Value) -> bool {
        match val {
            Value::Scalar(Literal::Bool(b)) => *b,
            _ => std::panic::panic_any(format!("expected bool scalar, got {val:?}")),
        }
    }
    fn extract_bools(val: &Value) -> Vec<bool> {
        val.as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| match l {
                Literal::Bool(b) => *b,
                _ => std::panic::panic_any("expected bool element"),
            })
            .collect()
    }

    #[test]
    fn eq_scalar_true() {
        let result = eval_comparison(
            Primitive::Eq,
            &[s_f64(5.0), s_f64(5.0)],
            |a, b| a == b,
            |a, b| a == b,
        )
        .unwrap();
        assert!(extract_bool(&result));
    }

    #[test]
    fn eq_scalar_false() {
        let result = eval_comparison(
            Primitive::Eq,
            &[s_f64(5.0), s_f64(3.0)],
            |a, b| a == b,
            |a, b| a == b,
        )
        .unwrap();
        assert!(!extract_bool(&result));
    }

    #[test]
    fn ne_scalar() {
        let result = eval_comparison(
            Primitive::Ne,
            &[s_i64(1), s_i64(2)],
            |a, b| a != b,
            |a, b| a != b,
        )
        .unwrap();
        assert!(extract_bool(&result));
    }

    #[test]
    fn lt_scalar() {
        let result = eval_comparison(
            Primitive::Lt,
            &[s_f64(1.0), s_f64(2.0)],
            |a, b| a < b,
            |a, b| a < b,
        )
        .unwrap();
        assert!(extract_bool(&result));
        let result = eval_comparison(
            Primitive::Lt,
            &[s_f64(2.0), s_f64(1.0)],
            |a, b| a < b,
            |a, b| a < b,
        )
        .unwrap();
        assert!(!extract_bool(&result));
    }

    #[test]
    fn ge_scalar() {
        let result = eval_comparison(
            Primitive::Ge,
            &[s_f64(5.0), s_f64(5.0)],
            |a, b| a >= b,
            |a, b| a >= b,
        )
        .unwrap();
        assert!(extract_bool(&result));
        let result = eval_comparison(
            Primitive::Ge,
            &[s_f64(4.0), s_f64(5.0)],
            |a, b| a >= b,
            |a, b| a >= b,
        )
        .unwrap();
        assert!(!extract_bool(&result));
    }

    #[test]
    fn eq_tensor_elementwise() {
        let a = v_f64(&[1.0, 2.0, 3.0]);
        let b = v_f64(&[1.0, 9.0, 3.0]);
        let result = eval_comparison(Primitive::Eq, &[a, b], |a, b| a == b, |a, b| a == b).unwrap();
        assert_eq!(extract_bools(&result), vec![true, false, true]);
    }

    #[test]
    fn lt_tensor_elementwise() {
        let a = v_f64(&[1.0, 5.0, 3.0]);
        let b = v_f64(&[2.0, 4.0, 3.0]);
        let result = eval_comparison(Primitive::Lt, &[a, b], |a, b| a < b, |a, b| a < b).unwrap();
        assert_eq!(extract_bools(&result), vec![true, false, false]);
    }

    #[test]
    fn comparison_shape_mismatch() {
        let a = v_f64(&[1.0, 2.0]);
        let b = v_f64(&[1.0, 2.0, 3.0]);
        let result = eval_comparison(Primitive::Eq, &[a, b], |a, b| a == b, |a, b| a == b);
        assert!(result.is_err());
    }

    #[test]
    fn comparison_scalar_tensor_broadcast() {
        let a = s_f64(2.0);
        let b = v_f64(&[1.0, 2.0, 3.0]);
        let result = eval_comparison(Primitive::Eq, &[a, b], |a, b| a == b, |a, b| a == b).unwrap();
        assert_eq!(extract_bools(&result), vec![false, true, false]);
    }

    #[test]
    fn comparison_arity_error() {
        let result = eval_comparison(Primitive::Eq, &[s_f64(1.0)], |a, b| a == b, |a, b| a == b);
        assert!(result.is_err());
    }

    fn half_bits(l: Literal) -> u16 {
        match l {
            Literal::BF16Bits(b) | Literal::F16Bits(b) => b,
            o => panic!("expected half literal, got {o:?}"),
        }
    }

    #[test]
    fn half_float_compare_dense_matches_generic() {
        // bf16/f16 same-shape compares now take the dense half-float path; must be
        // bit-identical to the boxed generic path (widen→f64→float_cmp) across all
        // six comparisons, incl NaN/±inf/±0.
        let vals_a = [
            1.5f32,
            -0.0,
            f32::INFINITY,
            f32::NAN,
            7.0,
            -3.25,
            0.0,
            f32::NEG_INFINITY,
        ];
        let vals_b = [
            2.0f32,
            0.0,
            -4.0,
            5.0,
            7.0,
            -3.25,
            f32::NAN,
            f32::NEG_INFINITY,
        ];
        let params = BTreeMap::new();
        for (dt, mk) in [
            (DType::BF16, Literal::from_bf16_f32 as fn(f32) -> Literal),
            (DType::F16, Literal::from_f16_f32 as fn(f32) -> Literal),
        ] {
            let lits_a: Vec<Literal> = vals_a.iter().map(|&v| mk(v)).collect();
            let lits_b: Vec<Literal> = vals_b.iter().map(|&v| mk(v)).collect();
            let abits: Vec<u16> = lits_a.iter().map(|&l| half_bits(l)).collect();
            let bbits: Vec<u16> = lits_b.iter().map(|&l| half_bits(l)).collect();
            let shape = Shape::vector(abits.len() as u32);
            let dense_a = Value::Tensor(
                TensorValue::new_half_float_values(dt, shape.clone(), abits).unwrap(),
            );
            let dense_b = Value::Tensor(
                TensorValue::new_half_float_values(dt, shape.clone(), bbits).unwrap(),
            );
            let boxed_a = Value::Tensor(
                TensorValue::new_with_literal_buffer(
                    dt,
                    shape.clone(),
                    fj_core::LiteralBuffer::new(lits_a.clone()),
                )
                .unwrap(),
            );
            let boxed_b = Value::Tensor(
                TensorValue::new_with_literal_buffer(
                    dt,
                    shape.clone(),
                    fj_core::LiteralBuffer::new(lits_b.clone()),
                )
                .unwrap(),
            );
            assert!(
                dense_a
                    .as_tensor()
                    .unwrap()
                    .elements
                    .as_half_float_slice()
                    .is_some(),
                "dense {dt:?} should expose half slice"
            );
            assert!(
                boxed_a
                    .as_tensor()
                    .unwrap()
                    .elements
                    .as_half_float_slice()
                    .is_none(),
                "boxed {dt:?} should NOT expose half slice"
            );
            for p in [
                Primitive::Eq,
                Primitive::Ne,
                Primitive::Lt,
                Primitive::Le,
                Primitive::Gt,
                Primitive::Ge,
            ] {
                let d = extract_bools(
                    &crate::eval_primitive(p, &[dense_a.clone(), dense_b.clone()], &params)
                        .unwrap(),
                );
                let g = extract_bools(
                    &crate::eval_primitive(p, &[boxed_a.clone(), boxed_b.clone()], &params)
                        .unwrap(),
                );
                assert_eq!(d, g, "{dt:?} {p:?} half compare dense != boxed");
            }
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_bf16_compare_dense_vs_boxed() {
        use std::time::Instant;
        let n = 1_000_000usize;
        let a: Vec<u16> = (0..n)
            .map(|i| half_bits(Literal::from_bf16_f32((i as f32) * 0.01 - 5000.0)))
            .collect();
        let b: Vec<u16> = (0..n)
            .map(|i| half_bits(Literal::from_bf16_f32(((n - i) as f32) * 0.01 - 5000.0)))
            .collect();
        let shape = Shape::vector(n as u32);
        let dense = [
            Value::Tensor(
                TensorValue::new_half_float_values(DType::BF16, shape.clone(), a.clone()).unwrap(),
            ),
            Value::Tensor(
                TensorValue::new_half_float_values(DType::BF16, shape.clone(), b.clone()).unwrap(),
            ),
        ];
        let boxed = [
            Value::Tensor(
                TensorValue::new_with_literal_buffer(
                    DType::BF16,
                    shape.clone(),
                    fj_core::LiteralBuffer::new(a.iter().map(|&x| Literal::BF16Bits(x)).collect()),
                )
                .unwrap(),
            ),
            Value::Tensor(
                TensorValue::new_with_literal_buffer(
                    DType::BF16,
                    shape.clone(),
                    fj_core::LiteralBuffer::new(b.iter().map(|&x| Literal::BF16Bits(x)).collect()),
                )
                .unwrap(),
            ),
        ];
        let params = BTreeMap::new();
        let best = |inputs: &[Value]| {
            let _ = crate::eval_primitive(Primitive::Gt, inputs, &params).unwrap();
            let mut tm = f64::MAX;
            for _ in 0..5 {
                let s = Instant::now();
                let o = crate::eval_primitive(Primitive::Gt, inputs, &params).unwrap();
                std::hint::black_box(o.as_tensor().unwrap().elements.len());
                tm = tm.min(s.elapsed().as_secs_f64());
            }
            tm
        };
        let bx = best(&boxed);
        let dn = best(&dense);
        println!(
            "BENCH bf16 Gt [{n}]: boxed={:.2}ms dense={:.2}ms speedup={:.2}x",
            bx * 1e3,
            dn * 1e3,
            bx / dn
        );
    }

    #[test]
    fn complex_eq_ne_dense_matches_boxed_and_rejects_ordered() {
        let pairs_a: Vec<(f64, f64)> = (0..32)
            .map(|i| (i as f64 * 0.5 - 8.0, -(i as f64) * 0.25))
            .collect();
        let mut pairs_b = pairs_a.clone();
        pairs_b[3] = (pairs_a[3].0 + 1.0, pairs_a[3].1); // differ in real
        pairs_b[7] = (pairs_a[7].0, pairs_a[7].1 - 2.0); // differ in imag
        pairs_b[10] = (f64::NAN, pairs_a[10].1); // NaN -> never equal (Eq false, Ne true)
        let shape = Shape::vector(32);
        let mk = |pairs: &[(f64, f64)], dense: bool| -> Value {
            if dense {
                Value::Tensor(
                    TensorValue::new_complex_values(DType::Complex128, shape.clone(), pairs.to_vec())
                        .unwrap(),
                )
            } else {
                Value::Tensor(
                    TensorValue::new_with_literal_buffer(
                        DType::Complex128,
                        shape.clone(),
                        fj_core::LiteralBuffer::new(
                            pairs
                                .iter()
                                .map(|&(re, im)| Literal::from_complex128(re, im))
                                .collect(),
                        ),
                    )
                    .unwrap(),
                )
            }
        };
        let bools = |v: &Value| -> Vec<bool> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::Bool(x) => *x,
                    other => panic!("expected Bool, got {other:?}"),
                })
                .collect()
        };
        for &prim in &[Primitive::Eq, Primitive::Ne] {
            let int_cmp = move |a: i128, b: i128| if prim == Primitive::Eq { a == b } else { a != b };
            let float_cmp = move |a: f64, b: f64| if prim == Primitive::Eq { a == b } else { a != b };
            let d = eval_comparison(
                prim,
                &[mk(&pairs_a, true), mk(&pairs_b, true)],
                int_cmp,
                float_cmp,
            )
            .unwrap();
            let b = eval_comparison(
                prim,
                &[mk(&pairs_a, false), mk(&pairs_b, false)],
                int_cmp,
                float_cmp,
            )
            .unwrap();
            assert_eq!(bools(&d), bools(&b), "{prim:?} dense vs boxed");
            assert!(
                d.as_tensor().unwrap().elements.as_bool_slice().is_some(),
                "{prim:?} dense bool output"
            );
        }
        // Ordered comparisons on complex must still error (JAX `_ordered` excludes complex).
        for &prim in &[Primitive::Lt, Primitive::Le, Primitive::Gt, Primitive::Ge] {
            let r = eval_comparison(
                prim,
                &[mk(&pairs_a, true), mk(&pairs_b, true)],
                |a: i128, b: i128| a < b,
                |a: f64, b: f64| a < b,
            );
            assert!(r.is_err(), "{prim:?} on complex must error");
        }
    }

    #[test]
    fn complex_eq_ne_broadcast_matches_boxed() {
        // lhs [3,4] complex vs rhs [4] complex (broadcasts over rows).
        let a: Vec<(f64, f64)> = (0..12)
            .map(|i| ((i % 4) as f64 - 1.0, (i / 4) as f64))
            .collect();
        let b: Vec<(f64, f64)> = (0..4).map(|i| (i as f64 - 1.0, 0.0)).collect();
        let sa = Shape {
            dims: vec![3, 4],
        };
        let sb = Shape::vector(4);
        let mk = |pairs: &[(f64, f64)], shape: &Shape, dense: bool| -> Value {
            if dense {
                Value::Tensor(
                    TensorValue::new_complex_values(DType::Complex128, shape.clone(), pairs.to_vec())
                        .unwrap(),
                )
            } else {
                Value::Tensor(
                    TensorValue::new_with_literal_buffer(
                        DType::Complex128,
                        shape.clone(),
                        fj_core::LiteralBuffer::new(
                            pairs
                                .iter()
                                .map(|&(re, im)| Literal::from_complex128(re, im))
                                .collect(),
                        ),
                    )
                    .unwrap(),
                )
            }
        };
        let bools = |v: &Value| -> Vec<bool> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::Bool(x) => *x,
                    other => panic!("expected Bool, got {other:?}"),
                })
                .collect()
        };
        for &prim in &[Primitive::Eq, Primitive::Ne] {
            let int_cmp = move |x: i128, y: i128| if prim == Primitive::Eq { x == y } else { x != y };
            let float_cmp = move |x: f64, y: f64| if prim == Primitive::Eq { x == y } else { x != y };
            let d = eval_comparison(
                prim,
                &[mk(&a, &sa, true), mk(&b, &sb, true)],
                int_cmp,
                float_cmp,
            )
            .unwrap();
            let bx = eval_comparison(
                prim,
                &[mk(&a, &sa, false), mk(&b, &sb, false)],
                int_cmp,
                float_cmp,
            )
            .unwrap();
            assert_eq!(bools(&d), bools(&bx), "{prim:?} broadcast dense vs boxed");
            assert_eq!(d.as_tensor().unwrap().shape.dims, vec![3, 4]);
            assert!(
                d.as_tensor().unwrap().elements.as_bool_slice().is_some(),
                "{prim:?} broadcast dense bool output"
            );
        }
    }

    fn half_lit(dt: DType, v: f32) -> Literal {
        if dt == DType::BF16 {
            Literal::from_bf16_f32(v)
        } else {
            Literal::from_f16_f32(v)
        }
    }
    fn half_dense_sh(dt: DType, dims: &[u32], vals: &[f32]) -> Value {
        let bits: Vec<u16> = vals.iter().map(|&v| half_bits(half_lit(dt, v))).collect();
        Value::Tensor(
            TensorValue::new_half_float_values(
                dt,
                Shape {
                    dims: dims.to_vec(),
                },
                bits,
            )
            .unwrap(),
        )
    }
    fn half_boxed_sh(dt: DType, dims: &[u32], vals: &[f32]) -> Value {
        let lits: Vec<Literal> = vals.iter().map(|&v| half_lit(dt, v)).collect();
        Value::Tensor(
            TensorValue::new_with_literal_buffer(
                dt,
                Shape {
                    dims: dims.to_vec(),
                },
                fj_core::LiteralBuffer::new(lits),
            )
            .unwrap(),
        )
    }

    #[test]
    fn half_float_broadcast_compare_dense_matches_generic() {
        // bf16/f16 broadcast (row/col/rank-1) compares now take the dense half path;
        // must be bit-identical to the boxed generic broadcast loop across all six
        // comparisons, both operand orders, incl NaN/±inf/±0.
        let params = BTreeMap::new();
        let m = [1.5f32, -0.0, f32::INFINITY, f32::NAN, 7.0, -3.25];
        let row = [2.0f32, 0.0, 7.0];
        let col = [3.0f32, -1.0];
        let v = [4.0f32, f32::NAN, 0.0];
        for dt in [DType::BF16, DType::F16] {
            for (rd, rdims) in [
                (&row[..], vec![1u32, 3]),
                (&col[..], vec![2, 1]),
                (&v[..], vec![3]),
            ] {
                for p in [
                    Primitive::Eq,
                    Primitive::Ne,
                    Primitive::Lt,
                    Primitive::Le,
                    Primitive::Gt,
                    Primitive::Ge,
                ] {
                    let d = extract_bools(
                        &crate::eval_primitive(
                            p,
                            &[
                                half_dense_sh(dt, &[2, 3], &m),
                                half_dense_sh(dt, &rdims, rd),
                            ],
                            &params,
                        )
                        .unwrap(),
                    );
                    let g = extract_bools(
                        &crate::eval_primitive(
                            p,
                            &[
                                half_boxed_sh(dt, &[2, 3], &m),
                                half_boxed_sh(dt, &rdims, rd),
                            ],
                            &params,
                        )
                        .unwrap(),
                    );
                    assert_eq!(d, g, "{dt:?} {p:?} bcast {rdims:?}");
                    let d2 = extract_bools(
                        &crate::eval_primitive(
                            p,
                            &[
                                half_dense_sh(dt, &rdims, rd),
                                half_dense_sh(dt, &[2, 3], &m),
                            ],
                            &params,
                        )
                        .unwrap(),
                    );
                    let g2 = extract_bools(
                        &crate::eval_primitive(
                            p,
                            &[
                                half_boxed_sh(dt, &rdims, rd),
                                half_boxed_sh(dt, &[2, 3], &m),
                            ],
                            &params,
                        )
                        .unwrap(),
                    );
                    assert_eq!(d2, g2, "{dt:?} {p:?} bcast-swapped {rdims:?}");
                }
            }
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_bf16_broadcast_compare_dense_vs_boxed() {
        use std::time::Instant;
        let (rows, cols) = (1000usize, 1000usize);
        let n = rows * cols;
        let m: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5000.0).collect();
        let row: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.02 - 1000.0).collect();
        let mat_dims = vec![rows as u32, cols as u32];
        let row_dims = vec![1u32, cols as u32];
        let dense = [
            half_dense_sh(DType::BF16, &mat_dims, &m),
            half_dense_sh(DType::BF16, &row_dims, &row),
        ];
        let boxed = [
            half_boxed_sh(DType::BF16, &mat_dims, &m),
            half_boxed_sh(DType::BF16, &row_dims, &row),
        ];
        let params = BTreeMap::new();
        let best = |inputs: &[Value]| {
            let _ = crate::eval_primitive(Primitive::Gt, inputs, &params).unwrap();
            let mut tm = f64::MAX;
            for _ in 0..5 {
                let s = Instant::now();
                let o = crate::eval_primitive(Primitive::Gt, inputs, &params).unwrap();
                std::hint::black_box(o.as_tensor().unwrap().elements.len());
                tm = tm.min(s.elapsed().as_secs_f64());
            }
            tm
        };
        let bx = best(&boxed);
        let dn = best(&dense);
        println!(
            "BENCH bf16 broadcast Gt [{rows}x{cols} ⊗ 1x{cols}]: boxed={:.2}ms dense={:.2}ms speedup={:.2}x",
            bx * 1e3,
            dn * 1e3,
            bx / dn
        );
    }
}
