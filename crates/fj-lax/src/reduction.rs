#![forbid(unsafe_code)]

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value};

use crate::EvalError;

fn literal_to_complex_parts(
    primitive: Primitive,
    literal: Literal,
) -> Result<(f64, f64), EvalError> {
    match literal {
        Literal::Complex64Bits(re, im) => {
            Ok((f32::from_bits(re) as f64, f32::from_bits(im) as f64))
        }
        Literal::Complex128Bits(re, im) => Ok((f64::from_bits(re), f64::from_bits(im))),
        _ => Err(EvalError::TypeMismatch {
            primitive,
            detail: "expected complex tensor elements",
        }),
    }
}

fn complex_literal_from_parts(dtype: DType, re: f64, im: f64) -> Literal {
    match dtype {
        DType::Complex64 => Literal::from_complex64(re as f32, im as f32),
        DType::Complex128 => Literal::from_complex128(re, im),
        _ => Literal::from_complex128(re, im),
    }
}

/// Lexicographic ordering on complex `(real, imaginary)` pairs, matching JAX's
/// complex max/min (the docstrings specify "a lexicographic comparison on the
/// (real, imaginary) pairs") and NumPy. Uses `total_cmp` so the order is total
/// and consistent with fj-lax's complex sort key (`SortKey::Complex`).
fn complex_lex_cmp(lhs: (f64, f64), rhs: (f64, f64)) -> std::cmp::Ordering {
    lhs.0
        .total_cmp(&rhs.0)
        .then_with(|| lhs.1.total_cmp(&rhs.1))
}

/// Output dtype for a real (non-complex, non-integral) reduction: preserve
/// the input's float precision so an F32/BF16/F16 reduction doesn't widen
/// to F64. Non-float inputs (U32/U64/Bool reaching the float arm via as_f64)
/// fall back to F64 since no narrower float type is implied.
fn reduce_real_output_dtype(input_dtype: DType) -> DType {
    match input_dtype {
        DType::BF16 | DType::F16 | DType::F32 | DType::F64 => input_dtype,
        _ => DType::F64,
    }
}

fn reduce_real_literal(dtype: DType, value: f64) -> Literal {
    match dtype {
        DType::BF16 => Literal::from_bf16_f64(value),
        DType::F16 => Literal::from_f16_f64(value),
        DType::F32 => Literal::from_f32(value as f32),
        _ => Literal::from_f64(value),
    }
}

/// Dense F64 full-reduction fast path for `ReduceSum`/`ReduceProd`/`ReduceMax`/
/// `ReduceMin`. Folds the contiguous `f64` backing store directly instead of
/// materializing the 24-byte `Literal` enum and matching `as_f64()` per element
/// (moving 8 bytes/element instead of 24).
///
/// Bit-for-bit identical to the generic `Vec<Literal>` float fold below: the
/// caller supplies the same `float_init` seed and `float_op` (e.g. `ReduceProd`
/// => `1.0`/`a*b`, `ReduceMax` => `-inf`/`jax_max_f64`), we apply them in the
/// same ascending element order with no reassociation, and the output is the
/// same `reduce_real_literal(F64, acc)` (`reduce_real_output_dtype(F64) == F64`).
/// `as_f64_slice()` is `Some` only for F64 dense storage, where
/// `slice[i] == as_slice()[i].as_f64()` exactly, so a malformed declared-F64
/// tensor (`Literal` storage) returns `None` and falls through unchanged.
#[inline]
fn eval_dense_f64_full_reduce(
    primitive: Primitive,
    tensor: &TensorValue,
    float_init: f64,
    float_op: &impl Fn(f64, f64) -> f64,
) -> Option<Value> {
    if !matches!(
        primitive,
        Primitive::ReduceSum | Primitive::ReduceProd | Primitive::ReduceMax | Primitive::ReduceMin
    ) || tensor.dtype != DType::F64
    {
        return None;
    }
    // F64 only: a single-accumulator fold over a flat slice is a candidate for LLVM
    // reduction vectorization. For F64 that's bit-identical to the generic
    // per-`Literal` scalar fold (verified by the dense-F64 reduce tests). The dense
    // F32 full-reduce was deliberately NOT added here — empirically its flat-slice
    // fold diverged from the generic scalar fold (inf+(-inf)=NaN sign, a symptom of
    // a different summation shape), so it is NOT provably order-identical. f32 AXIS
    // reductions (the hot softmax/layernorm path) ARE accelerated — they use the
    // scatter-odometer / per-cell-ordered block folds, which can't reassociate.
    let values = tensor.elements.as_f64_slice()?;
    let mut acc = float_init;
    for &value in values {
        acc = float_op(acc, value);
    }
    Some(Value::Scalar(reduce_real_literal(DType::F64, acc)))
}

fn parse_reduction_axes(
    primitive: Primitive,
    raw_axes: &str,
    rank: usize,
) -> Result<Vec<usize>, EvalError> {
    let estimated_count = raw_axes.matches(',').count() + 1;
    let mut axes = Vec::with_capacity(estimated_count.min(rank));
    for raw_axis in raw_axes.split(',') {
        let trimmed = raw_axis.trim();
        let axis = trimmed.parse::<i64>().map_err(|_| EvalError::Unsupported {
            primitive,
            detail: format!("invalid axis value: {trimmed}"),
        })?;
        let normalized = if axis < 0 { rank as i64 + axis } else { axis };
        if normalized < 0 || normalized >= rank as i64 {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("axis {axis} out of bounds for rank {rank}"),
            });
        }

        let normalized = normalized as usize;
        if axes.contains(&normalized) {
            return Err(EvalError::Unsupported {
                primitive,
                detail: format!("duplicate value in axes: {axis}"),
            });
        }
        axes.push(normalized);
    }
    Ok(axes)
}

fn parse_bool_param(
    primitive: Primitive,
    params: &std::collections::BTreeMap<String, String>,
    key: &str,
    default: bool,
) -> Result<bool, EvalError> {
    let Some(raw) = params.get(key) else {
        return Ok(default);
    };
    let trimmed = raw.trim();
    if trimmed == "1" || trimmed.eq_ignore_ascii_case("true") || trimmed.eq_ignore_ascii_case("yes")
    {
        Ok(true)
    } else if trimmed == "0"
        || trimmed.eq_ignore_ascii_case("false")
        || trimmed.eq_ignore_ascii_case("no")
    {
        Ok(false)
    } else {
        Err(EvalError::Unsupported {
            primitive,
            detail: format!("invalid boolean parameter {key}={raw:?}"),
        })
    }
}

/// Generic reduction: reduces elements of a tensor along specified axes (or all axes).
pub(crate) fn eval_reduce(
    primitive: Primitive,
    inputs: &[Value],
    int_init: i64,
    float_init: f64,
    int_op: impl Fn(i64, i64) -> i64,
    float_op: impl Fn(f64, f64) -> f64,
) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    match &inputs[0] {
        Value::Scalar(literal) => Ok(Value::Scalar(*literal)),
        Value::Tensor(tensor) => {
            // For backward compat: no axes param means reduce all
            // Axis-specific reduction is needed for partial reduction
            let rank = tensor.shape.rank();

            // Full reduction (all elements to scalar)
            if rank == 0 {
                return Ok(Value::Scalar(tensor.elements[0]));
            }

            if let Some(value) =
                eval_dense_f64_full_reduce(primitive, tensor, float_init, &float_op)
            {
                return Ok(value);
            }

            if matches!(tensor.dtype, DType::Complex64 | DType::Complex128) {
                if !matches!(
                    primitive,
                    Primitive::ReduceSum
                        | Primitive::ReduceProd
                        | Primitive::ReduceMax
                        | Primitive::ReduceMin
                ) {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "complex reduction is supported only for reduce_sum, reduce_prod, reduce_max, and reduce_min"
                            .to_owned(),
                    });
                }

                // For max/min, `float_init` is ∓∞ (the lexicographic sentinel),
                // so seeding both components with it makes the first element win.
                let (mut re_acc, mut im_acc) = match primitive {
                    Primitive::ReduceProd => (1.0_f64, 0.0_f64),
                    Primitive::ReduceMax | Primitive::ReduceMin => (float_init, float_init),
                    _ => (0.0_f64, 0.0_f64),
                };
                for literal in &tensor.elements {
                    let (re, im) = literal_to_complex_parts(primitive, *literal)?;
                    match primitive {
                        Primitive::ReduceProd => {
                            let new_re = re_acc * re - im_acc * im;
                            let new_im = re_acc * im + im_acc * re;
                            re_acc = new_re;
                            im_acc = new_im;
                        }
                        Primitive::ReduceMax => {
                            if complex_lex_cmp((re, im), (re_acc, im_acc)).is_gt() {
                                re_acc = re;
                                im_acc = im;
                            }
                        }
                        Primitive::ReduceMin => {
                            if complex_lex_cmp((re, im), (re_acc, im_acc)).is_lt() {
                                re_acc = re;
                                im_acc = im;
                            }
                        }
                        _ => {
                            re_acc += re;
                            im_acc += im;
                        }
                    }
                }
                return Ok(Value::Scalar(complex_literal_from_parts(
                    tensor.dtype,
                    re_acc,
                    im_acc,
                )));
            }

            let is_integral = tensor.dtype == DType::I64 || tensor.dtype == DType::I32;

            // Full reduction: flatten to scalar
            if is_integral {
                // Dense i64 fast path: fold the contiguous `i64` backing slice
                // directly. Bit-for-bit identical to the generic loop below —
                // same `int_init` seed, same ascending order, same `int_op`,
                // same `Value::scalar_i64` output — but skips the per-element
                // `Literal::I64` match and the 24-byte enum stride.
                // `as_i64_slice()` is `Some` only for I64 dense storage.
                if let Some(values) = tensor.elements.as_i64_slice() {
                    let mut acc = int_init;
                    for &val in values {
                        acc = int_op(acc, val);
                    }
                    return Ok(Value::scalar_i64(acc));
                }
                let mut acc = int_init;
                for literal in &tensor.elements {
                    let val = literal.as_i64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "expected i64 tensor",
                    })?;
                    acc = int_op(acc, val);
                }
                Ok(Value::scalar_i64(acc))
            } else {
                let mut acc = float_init;
                for literal in &tensor.elements {
                    let val = literal.as_f64().ok_or(EvalError::TypeMismatch {
                        primitive,
                        detail: "expected numeric tensor",
                    })?;
                    acc = float_op(acc, val);
                }
                let out_dtype = reduce_real_output_dtype(tensor.dtype);
                Ok(Value::Scalar(reduce_real_literal(out_dtype, acc)))
            }
        }
    }
}

/// Axis-aware reduction: reduces tensor along specified axes, producing a tensor output.
/// Dense F64 axis (partial) reduction fast path. Returns `Some(result)` — the
/// reduced f64 accumulators, length `out_count` — when the tensor is dense F64,
/// else `None` so the caller runs the generic `Vec<Literal>` loop unchanged.
///
/// Bit-for-bit identical to that loop: it visits the contiguous backing slice
/// in ascending flat order (matching `for flat_idx in 0..total`), seeds every
/// accumulator with `float_init`, and applies the same `float_op` in the same
/// order, so for each output cell the inputs accumulate in the same sequence
/// (no reassociation). The destination `out_idx` is maintained incrementally by
/// a row-major odometer that exactly reproduces
/// `multi_to_out_flat(flat_to_multi_into(flat))`: the innermost axis varies
/// fastest (input stride 1), each axis contributes `out_axis_stride[axis]` to
/// the output flat index (row-major over `out_dims`, which is ordered by
/// `kept_axes`), and reduced axes contribute 0. This removes both the
/// per-element multi-index decode and the 24-byte `Literal` materialization.
/// `as_f64_slice()` is `Some` only for F64 dense storage (a well-formed tensor
/// whose element count matches the shape), so no malformed-input case reaches
/// here and the generic path's per-element overflow checks are unnecessary
/// (`out_idx` stays within `0..out_count`).
#[allow(clippy::too_many_arguments)]
#[inline]
fn dense_f64_axis_reduce<T: Copy + Sync>(
    tensor: &TensorValue,
    values: &[T],
    widen: impl Fn(T) -> f64 + Copy + Sync + Send,
    kept_axes: &[usize],
    out_dims: &[u32],
    out_count: usize,
    float_init: f64,
    float_op: &(impl Fn(f64, f64) -> f64 + Sync),
) -> Option<Vec<f64>> {
    if tensor.shape.dims.is_empty() {
        return None;
    }

    // Trailing-axis reduction fast path: when the kept axes are exactly the
    // leading prefix `0..k`, every output element reduces one contiguous input
    // block (`values[o*block .. (o+1)*block]`) in ascending order — identical to
    // the odometer's emission order. The output rows are independent, so a large
    // reduction fans out across threads. Bit-for-bit identical to the serial fold.
    let kept_is_leading_prefix = kept_axes.iter().enumerate().all(|(i, &ax)| ax == i);
    if kept_is_leading_prefix
        && out_count > 1
        && values.len() == out_count * (values.len() / out_count)
        && values.len() >= (1 << 18)
    {
        let block = values.len() / out_count;
        let mut result = vec![float_init; out_count];
        // Work-scale the fan-out to the TOTAL reduce work (values.len()), not the
        // row count: at moderate sizes a flat all-core split gave each thread tiny
        // work and was spawn-overhead-dominated (see work_scaled_threads). Capped
        // at out_count since rows are the unit of parallelism.
        let threads = crate::arithmetic::work_scaled_threads(values.len()).min(out_count);
        if threads > 1 {
            let rows_per = out_count.div_ceil(threads);
            let op_ref = float_op;
            std::thread::scope(|scope| {
                let mut res_rest: &mut [f64] = result.as_mut_slice();
                let mut row0 = 0usize;
                while row0 < out_count {
                    let rows = rows_per.min(out_count - row0);
                    let (res_blk, res_tail) = res_rest.split_at_mut(rows);
                    res_rest = res_tail;
                    let vblk = &values[row0 * block..(row0 + rows) * block];
                    row0 += rows;
                    scope.spawn(move || {
                        for (r, slot) in res_blk.iter_mut().enumerate() {
                            let mut acc = float_init;
                            for &v in &vblk[r * block..r * block + block] {
                                acc = op_ref(acc, widen(v));
                            }
                            *slot = acc;
                        }
                    });
                }
            });
            return Some(result);
        }
    }

    // Leading-axis reduction fast path: when the kept axes are exactly the trailing
    // suffix (the reduced axes are the leading prefix), the reduction is a column
    // accumulation — `out[o] = op_k(values[k*block + o])` for k over the reduced
    // extent, o over the `block = out_count` kept columns. The inner loop over `o`
    // is a contiguous read + contiguous accumulate (vectorizable for `+`/`*`), and
    // each output column accumulates k in ascending order — identical to the
    // odometer's emission order. This keeps a single deterministic fold per
    // column while replacing the generic per-element odometer update with a
    // tight row-slice loop.
    let rank = tensor.shape.dims.len();
    let kept_is_trailing_suffix = !kept_axes.is_empty()
        && *kept_axes.last().unwrap() == rank - 1
        && kept_axes
            .iter()
            .enumerate()
            .all(|(i, &ax)| ax == rank - kept_axes.len() + i);
    if kept_is_trailing_suffix
        && out_count > 1
        && values.len() == out_count * (values.len() / out_count)
        && values.len() >= (1 << 18)
    {
        let block = out_count;
        let outer = values.len() / block;
        let mut result = vec![float_init; block];
        for k in 0..outer {
            let row = &values[k * block..k * block + block];
            for (slot, &v) in result.iter_mut().zip(row.iter()) {
                *slot = float_op(*slot, widen(v));
            }
        }
        return Some(result);
    }

    let mut odometer = OutIndexOdometer::new(&tensor.shape.dims, kept_axes, out_dims);
    let mut result = vec![float_init; out_count];
    for &value in values {
        let out_idx = odometer.next_index();
        result[out_idx] = float_op(result[out_idx], widen(value));
    }
    Some(result)
}

/// Incremental row-major output-index odometer for axis (partial) reductions.
///
/// `next_index()` returns the output flat index for the input element at the
/// current ascending flat position, then advances — reproducing
/// `multi_to_out_flat(flat_to_multi_into(flat))` for `flat = 0, 1, 2, …` without
/// the per-element multi-index decode (a `Vec` allocation + a `kept_axes` loop
/// each step). The innermost axis varies fastest (input stride 1); each axis
/// contributes `out_axis_stride[axis]` to the output flat index (row-major over
/// `out_dims`, which is ordered by `kept_axes`); reduced axes contribute 0. The
/// emitted sequence is therefore bit-identical to the generic decode loop, so
/// every accumulator sees its inputs in the same order. Must only be stepped
/// `product(dims)` times; the final step harmlessly wraps all coordinates to 0.
struct OutIndexOdometer {
    dims: Vec<usize>,
    out_axis_stride: Vec<usize>,
    coord: Vec<usize>,
    out_idx: usize,
}

impl OutIndexOdometer {
    fn new(dims: &[u32], kept_axes: &[usize], out_dims: &[u32]) -> Self {
        let rank = dims.len();
        let mut out_axis_stride = vec![0_usize; rank];
        let mut stride = 1_usize;
        for (k, &axis) in kept_axes.iter().enumerate().rev() {
            out_axis_stride[axis] = stride;
            stride *= out_dims[k] as usize;
        }
        Self {
            dims: dims.iter().map(|&d| d as usize).collect(),
            out_axis_stride,
            coord: vec![0_usize; rank],
            out_idx: 0,
        }
    }

    #[inline]
    fn next_index(&mut self) -> usize {
        let current = self.out_idx;
        let mut ax = self.dims.len() - 1;
        loop {
            self.coord[ax] += 1;
            self.out_idx += self.out_axis_stride[ax];
            if self.coord[ax] < self.dims[ax] {
                break;
            }
            self.coord[ax] = 0;
            self.out_idx -= self.out_axis_stride[ax] * self.dims[ax];
            if ax == 0 {
                break;
            }
            ax -= 1;
        }
        current
    }
}

pub(crate) fn eval_reduce_axes(
    primitive: Primitive,
    inputs: &[Value],
    params: &std::collections::BTreeMap<String, String>,
    int_init: i64,
    float_init: f64,
    int_op: impl Fn(i64, i64) -> i64,
    float_op: impl Fn(f64, f64) -> f64 + Sync,
) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    // If no axes param, fall back to full reduction. Empty list means identity.
    let axes_str = match params.get("axes") {
        Some(s) => s,
        None => return eval_reduce(primitive, inputs, int_init, float_init, int_op, float_op),
    };
    if axes_str.trim().is_empty() {
        return Ok(inputs[0].clone());
    }

    match &inputs[0] {
        Value::Scalar(_) => Ok(inputs[0].clone()),
        Value::Tensor(tensor) => {
            let rank = tensor.shape.rank();
            let mut axes_sorted = parse_reduction_axes(primitive, axes_str, rank)?;
            axes_sorted.sort_unstable();

            // If reducing all axes, just do full reduction
            if axes_sorted.len() == rank {
                return eval_reduce(primitive, inputs, int_init, float_init, int_op, float_op);
            }

            // Compute output shape (remove reduced axes)
            let out_dims: Vec<u32> = tensor
                .shape
                .dims
                .iter()
                .enumerate()
                .filter(|(i, _)| !axes_sorted.contains(i))
                .map(|(_, d)| *d)
                .collect();

            let is_complex = matches!(tensor.dtype, DType::Complex64 | DType::Complex128);
            let is_integral = tensor.dtype == DType::I64 || tensor.dtype == DType::I32;

            // Total number of output elements
            let out_count = checked_shape_element_count(primitive, "reduction output", &out_dims)?;
            if out_count == 0 {
                return Ok(Value::Tensor(TensorValue::new(
                    tensor.dtype,
                    Shape { dims: out_dims },
                    vec![],
                )?));
            }

            // For each output element, iterate over the reduced axes and accumulate
            let kept_axes: Vec<usize> = (0..rank).filter(|i| !axes_sorted.contains(i)).collect();

            if is_complex {
                if !matches!(
                    primitive,
                    Primitive::ReduceSum
                        | Primitive::ReduceProd
                        | Primitive::ReduceMax
                        | Primitive::ReduceMin
                ) {
                    return Err(EvalError::Unsupported {
                        primitive,
                        detail: "complex reduction is supported only for reduce_sum, reduce_prod, reduce_max, and reduce_min"
                            .to_owned(),
                    });
                }

                // ReduceSum seeds (0,0); ReduceProd (1,0); ReduceMax/ReduceMin
                // seed (float_init, float_init) where float_init is ∓∞, the
                // lexicographic sentinel that the first element always beats.
                let (init_re, init_im) = if primitive == Primitive::ReduceProd {
                    (1.0, 0.0)
                } else {
                    (float_init, float_init)
                };
                let mut result_re =
                    try_filled_vec(primitive, "reduction real accumulator", out_count, init_re)?;
                let mut result_im = try_filled_vec(
                    primitive,
                    "reduction imaginary accumulator",
                    out_count,
                    init_im,
                )?;
                let mut odometer = OutIndexOdometer::new(&tensor.shape.dims, &kept_axes, &out_dims);
                for literal in tensor.elements.iter() {
                    let out_idx = odometer.next_index();
                    let (re, im) = literal_to_complex_parts(primitive, *literal)?;
                    match primitive {
                        Primitive::ReduceProd => {
                            let acc_re = result_re[out_idx];
                            let acc_im = result_im[out_idx];
                            result_re[out_idx] = acc_re * re - acc_im * im;
                            result_im[out_idx] = acc_re * im + acc_im * re;
                        }
                        Primitive::ReduceMax => {
                            if complex_lex_cmp((re, im), (result_re[out_idx], result_im[out_idx]))
                                .is_gt()
                            {
                                result_re[out_idx] = re;
                                result_im[out_idx] = im;
                            }
                        }
                        Primitive::ReduceMin => {
                            if complex_lex_cmp((re, im), (result_re[out_idx], result_im[out_idx]))
                                .is_lt()
                            {
                                result_re[out_idx] = re;
                                result_im[out_idx] = im;
                            }
                        }
                        _ => {
                            // ReduceSum: component-wise float_op (addition).
                            result_re[out_idx] = float_op(result_re[out_idx], re);
                            result_im[out_idx] = float_op(result_im[out_idx], im);
                        }
                    }
                }

                let elements: Vec<Literal> = result_re
                    .into_iter()
                    .zip(result_im)
                    .map(|(re, im)| complex_literal_from_parts(tensor.dtype, re, im))
                    .collect();

                Ok(Value::Tensor(TensorValue::new(
                    tensor.dtype,
                    Shape { dims: out_dims },
                    elements,
                )?))
            } else if is_integral {
                let mut result = try_filled_vec(
                    primitive,
                    "reduction integer accumulator",
                    out_count,
                    int_init,
                )?;
                let mut odometer = OutIndexOdometer::new(&tensor.shape.dims, &kept_axes, &out_dims);
                // Dense i64 fast path: drive the odometer over the contiguous
                // `i64` backing slice, skipping the per-element `Literal::I64`
                // match and 24-byte stride. Bit-identical to the generic loop
                // (same order, out_idx sequence, int_op). `as_i64_slice()` is
                // `Some` only for I64 dense storage.
                if let Some(values) = tensor.elements.as_i64_slice() {
                    for &val in values {
                        let out_idx = odometer.next_index();
                        result[out_idx] = int_op(result[out_idx], val);
                    }
                } else {
                    for literal in tensor.elements.iter() {
                        let out_idx = odometer.next_index();
                        let val = literal.as_i64().ok_or(EvalError::TypeMismatch {
                            primitive,
                            detail: "expected i64 tensor",
                        })?;
                        result[out_idx] = int_op(result[out_idx], val);
                    }
                }
                let elements: Vec<Literal> = result.into_iter().map(Literal::I64).collect();
                Ok(Value::Tensor(TensorValue::new(
                    DType::I64,
                    Shape { dims: out_dims },
                    elements,
                )?))
            } else {
                // Dense axis-reduce fast path, reading the native backing slice and
                // widening F32->f64 INLINE (no buffer). The generic odometer loop
                // below also folds in f64 over `as_f64()` and rounds via
                // `reduce_real_literal`, so the f32 dense path (same per-output-cell
                // ascending fold, same round) is bit-for-bit identical — incl. NaN
                // bits, since both fold the same f64 values in the same order.
                let dense = match tensor.dtype {
                    DType::F64 => tensor.elements.as_f64_slice().and_then(|v| {
                        dense_f64_axis_reduce(
                            tensor,
                            v,
                            |x| x,
                            &kept_axes,
                            &out_dims,
                            out_count,
                            float_init,
                            &float_op,
                        )
                    }),
                    DType::F32 => tensor.elements.as_f32_slice().and_then(|v| {
                        dense_f64_axis_reduce(
                            tensor,
                            v,
                            f64::from,
                            &kept_axes,
                            &out_dims,
                            out_count,
                            float_init,
                            &float_op,
                        )
                    }),
                    // Half floats decode each raw u16 to f64 exactly via the same
                    // `Literal::as_f64()` the generic path uses (so the fold sees
                    // identical values in identical order); reading the packed u16
                    // backing avoids the per-element odometer over boxed Literals.
                    DType::BF16 => tensor.elements.as_half_float_slice().and_then(|v| {
                        dense_f64_axis_reduce(
                            tensor,
                            v,
                            |b| Literal::BF16Bits(b).as_f64().unwrap_or(0.0),
                            &kept_axes,
                            &out_dims,
                            out_count,
                            float_init,
                            &float_op,
                        )
                    }),
                    DType::F16 => tensor.elements.as_half_float_slice().and_then(|v| {
                        dense_f64_axis_reduce(
                            tensor,
                            v,
                            |b| Literal::F16Bits(b).as_f64().unwrap_or(0.0),
                            &kept_axes,
                            &out_dims,
                            out_count,
                            float_init,
                            &float_op,
                        )
                    }),
                    _ => None,
                };
                let result = if let Some(values) = dense {
                    values
                } else {
                    let mut result = try_filled_vec(
                        primitive,
                        "reduction float accumulator",
                        out_count,
                        float_init,
                    )?;
                    let mut odometer =
                        OutIndexOdometer::new(&tensor.shape.dims, &kept_axes, &out_dims);
                    for literal in tensor.elements.iter() {
                        let out_idx = odometer.next_index();
                        let val = literal.as_f64().ok_or(EvalError::TypeMismatch {
                            primitive,
                            detail: "expected numeric tensor",
                        })?;
                        result[out_idx] = float_op(result[out_idx], val);
                    }
                    result
                };
                let out_dtype = reduce_real_output_dtype(tensor.dtype);
                let elements: Vec<Literal> = result
                    .into_iter()
                    .map(|v| reduce_real_literal(out_dtype, v))
                    .collect();
                Ok(Value::Tensor(TensorValue::new(
                    out_dtype,
                    Shape { dims: out_dims },
                    elements,
                )?))
            }
        }
    }
}

/// Axis-aware bitwise reduction over bool/i64 tensors.
///
/// If `axes` is omitted, performs a full reduction to scalar.
pub(crate) fn eval_reduce_bitwise_axes(
    primitive: Primitive,
    inputs: &[Value],
    params: &std::collections::BTreeMap<String, String>,
    int_init: i64,
    bool_init: bool,
    int_op: impl Fn(i64, i64) -> i64,
    bool_op: impl Fn(bool, bool) -> bool,
) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    let axes_param = params.get("axes");
    if axes_param.is_some_and(|raw_axes| raw_axes.trim().is_empty()) {
        return Ok(inputs[0].clone());
    }
    let reduce_all = axes_param.is_none();

    match &inputs[0] {
        Value::Scalar(literal) => Ok(Value::Scalar(*literal)),
        Value::Tensor(tensor) => {
            let rank = tensor.shape.rank();
            if rank == 0 {
                return Ok(Value::Scalar(tensor.elements[0]));
            }

            let mut axes_sorted = if reduce_all {
                Vec::new()
            } else {
                let raw_axes = params.get("axes").ok_or(EvalError::Unsupported {
                    primitive,
                    detail: "missing axes parameter".to_owned(),
                })?;
                parse_reduction_axes(primitive, raw_axes, rank)?
            };
            axes_sorted.sort_unstable();

            let reduce_all_axes = reduce_all || axes_sorted.len() == rank;

            match tensor.dtype {
                DType::Bool => {
                    if reduce_all_axes {
                        // Dense bool fast path: fold the contiguous `bool` slice
                        // directly. Bit-identical to the generic loop below — same
                        // bool_init seed, ascending order, bool_op — but skips the
                        // per-element Literal::Bool match and 24-byte enum stride.
                        // as_bool_slice() is Some only for Bool dense storage.
                        if let Some(values) = tensor.elements.as_bool_slice() {
                            let mut acc = bool_init;
                            for &val in values {
                                acc = bool_op(acc, val);
                            }
                            return Ok(Value::scalar_bool(acc));
                        }
                        let mut acc = bool_init;
                        for literal in &tensor.elements {
                            let val = match literal {
                                Literal::Bool(v) => *v,
                                _ => {
                                    return Err(EvalError::TypeMismatch {
                                        primitive,
                                        detail: "expected bool tensor",
                                    });
                                }
                            };
                            acc = bool_op(acc, val);
                        }
                        return Ok(Value::scalar_bool(acc));
                    }

                    let out_dims: Vec<u32> = tensor
                        .shape
                        .dims
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !axes_sorted.contains(i))
                        .map(|(_, d)| *d)
                        .collect();
                    let out_count = checked_shape_element_count(
                        primitive,
                        "bitwise reduction output",
                        &out_dims,
                    )?;
                    if out_count == 0 {
                        return Ok(Value::Tensor(TensorValue::new(
                            DType::Bool,
                            Shape { dims: out_dims },
                            vec![],
                        )?));
                    }

                    let kept_axes: Vec<usize> =
                        (0..rank).filter(|i| !axes_sorted.contains(i)).collect();
                    let mut result = try_filled_vec(
                        primitive,
                        "bitwise reduction bool accumulator",
                        out_count,
                        bool_init,
                    )?;
                    // Drive an incremental out-index odometer (no per-element
                    // flat_to_multi decode); for dense Bool storage fold the
                    // contiguous bool slice directly (no Literal::Bool match,
                    // 1 byte/elem). Bit-identical to the generic loop: same
                    // ascending flat order, same out_idx mapping, same bool_op.
                    let mut odometer =
                        OutIndexOdometer::new(&tensor.shape.dims, &kept_axes, &out_dims);
                    if let Some(values) = tensor.elements.as_bool_slice() {
                        for &val in values {
                            let out_idx = odometer.next_index();
                            result[out_idx] = bool_op(result[out_idx], val);
                        }
                    } else {
                        for literal in tensor.elements.iter() {
                            let out_idx = odometer.next_index();
                            let val = match literal {
                                Literal::Bool(v) => *v,
                                _ => {
                                    return Err(EvalError::TypeMismatch {
                                        primitive,
                                        detail: "expected bool tensor",
                                    });
                                }
                            };
                            result[out_idx] = bool_op(result[out_idx], val);
                        }
                    }

                    let elements = result.into_iter().map(Literal::Bool).collect();
                    Ok(Value::Tensor(TensorValue::new(
                        DType::Bool,
                        Shape { dims: out_dims },
                        elements,
                    )?))
                }
                DType::I64 => {
                    if reduce_all_axes {
                        let mut acc = int_init;
                        for literal in &tensor.elements {
                            let val = literal.as_i64().ok_or(EvalError::TypeMismatch {
                                primitive,
                                detail: "expected i64 tensor",
                            })?;
                            acc = int_op(acc, val);
                        }
                        return Ok(Value::scalar_i64(acc));
                    }

                    let out_dims: Vec<u32> = tensor
                        .shape
                        .dims
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !axes_sorted.contains(i))
                        .map(|(_, d)| *d)
                        .collect();
                    let out_count = checked_shape_element_count(
                        primitive,
                        "bitwise reduction output",
                        &out_dims,
                    )?;
                    if out_count == 0 {
                        return Ok(Value::Tensor(TensorValue::new(
                            DType::I64,
                            Shape { dims: out_dims },
                            vec![],
                        )?));
                    }

                    let strides =
                        checked_strides(primitive, "bitwise reduction input", &tensor.shape.dims)?;
                    let kept_axes: Vec<usize> =
                        (0..rank).filter(|i| !axes_sorted.contains(i)).collect();
                    let mut result = try_filled_vec(
                        primitive,
                        "bitwise reduction integer accumulator",
                        out_count,
                        int_init,
                    )?;
                    let mut multi = Vec::with_capacity(strides.len());
                    for flat_idx in 0..tensor.elements.len() {
                        flat_to_multi_into(flat_idx, &strides, &mut multi);
                        let out_idx = multi_to_out_flat(
                            primitive,
                            "bitwise reduction output",
                            &multi,
                            &kept_axes,
                            &out_dims,
                        )?;
                        let val =
                            tensor.elements[flat_idx]
                                .as_i64()
                                .ok_or(EvalError::TypeMismatch {
                                    primitive,
                                    detail: "expected i64 tensor",
                                })?;
                        result[out_idx] = int_op(result[out_idx], val);
                    }

                    let elements = result.into_iter().map(Literal::I64).collect();
                    Ok(Value::Tensor(TensorValue::new(
                        DType::I64,
                        Shape { dims: out_dims },
                        elements,
                    )?))
                }
                _ => Err(EvalError::TypeMismatch {
                    primitive,
                    detail: "bitwise reduction expects bool or i64 tensor",
                }),
            }
        }
    }
}

fn checked_shape_element_count(
    primitive: Primitive,
    context: &str,
    dims: &[u32],
) -> Result<usize, EvalError> {
    if dims.contains(&0) {
        return Ok(0);
    }

    dims.iter().try_fold(1_usize, |acc, dim| {
        acc.checked_mul(*dim as usize)
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: format!("{context} shape overflows usize"),
            })
    })
}

fn checked_strides(
    primitive: Primitive,
    context: &str,
    dims: &[u32],
) -> Result<Vec<usize>, EvalError> {
    let mut strides = vec![1_usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1]
            .checked_mul(dims[i + 1] as usize)
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: format!("{context} strides overflow usize"),
            })?;
    }
    Ok(strides)
}

fn try_filled_vec<T: Clone>(
    primitive: Primitive,
    context: &str,
    len: usize,
    value: T,
) -> Result<Vec<T>, EvalError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|err| EvalError::Unsupported {
            primitive,
            detail: format!("{context} allocation failed for {len} elements: {err}"),
        })?;
    values.resize(len, value);
    Ok(values)
}

fn flat_to_multi_into(flat: usize, strides: &[usize], out: &mut Vec<usize>) {
    out.clear();
    let mut remainder = flat;
    for &stride in strides {
        out.push(remainder / stride);
        remainder %= stride;
    }
}

fn multi_to_out_flat(
    primitive: Primitive,
    context: &str,
    multi: &[usize],
    kept_axes: &[usize],
    out_dims: &[u32],
) -> Result<usize, EvalError> {
    let mut idx = 0_usize;
    let mut stride = 1_usize;
    for i in (0..kept_axes.len()).rev() {
        let offset =
            multi[kept_axes[i]]
                .checked_mul(stride)
                .ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: format!("{context} flat index overflows usize"),
                })?;
        idx = idx
            .checked_add(offset)
            .ok_or_else(|| EvalError::Unsupported {
                primitive,
                detail: format!("{context} flat index overflows usize"),
            })?;
        stride =
            stride
                .checked_mul(out_dims[i] as usize)
                .ok_or_else(|| EvalError::Unsupported {
                    primitive,
                    detail: format!("{context} stride overflows usize"),
                })?;
    }
    Ok(idx)
}

/// Cumulative scan along a specified axis. Output shape matches input.
/// If no axis param, defaults to axis 0.
/// Dense F64/I64 cumulative-scan fast path. Scans each line along `axis`
/// directly over a clone of the contiguous typed backing and returns a dense
/// output, bypassing the `Vec<Literal>` materialization (`to_vec`) and the
/// per-element `as_f64()`/`as_i64()` + Literal write of the generic path.
/// Bit-for-bit identical: same per-line `float_init`/`int_init` seed, same
/// ascending (or reversed) order, same `float_op`/`int_op`, and the same output
/// literal (`reduce_real_output_dtype(F64) == F64` -> `from_f64`; I64 ->
/// `Literal::I64`). Returns `None` unless the tensor is F64 or I64 dense storage.
///
/// Minimum element count before a cumulative scan fans out across threads.
const CUMULATIVE_PARALLEL_MIN_ELEMS: usize = 1 << 18; // 262_144

/// Serial cumulative scan from contiguous input lines into contiguous output
/// lines. Each line is independent, and every line still accumulates in the
/// exact same element order as the generic path.
#[inline]
fn scan_contiguous_lines_from<T, F>(
    src: &[T],
    out: &mut [T],
    axis_dim: usize,
    reverse: bool,
    init: T,
    op: &F,
) where
    T: Copy,
    F: Fn(T, T) -> T + ?Sized,
{
    for (src_line, out_line) in src.chunks(axis_dim).zip(out.chunks_mut(axis_dim)) {
        let mut acc = init;
        if reverse {
            for (src_value, out_value) in src_line.iter().zip(out_line.iter_mut()).rev() {
                acc = op(acc, *src_value);
                *out_value = acc;
            }
        } else {
            for (src_value, out_value) in src_line.iter().zip(out_line.iter_mut()) {
                acc = op(acc, *src_value);
                *out_value = acc;
            }
        }
    }
}

/// Serial cumulative scan over already-contiguous output lines. This is the
/// fast path for one-line or small scans where cloning the dense input and
/// mutating in place beats direct-output setup overhead.
#[inline]
fn scan_contiguous_lines_in_place<T, F>(
    out: &mut [T],
    axis_dim: usize,
    reverse: bool,
    init: T,
    op: &F,
) where
    T: Copy,
    F: Fn(T, T) -> T + ?Sized,
{
    for line in out.chunks_mut(axis_dim) {
        let mut acc = init;
        if reverse {
            for x in line.iter_mut().rev() {
                acc = op(acc, *x);
                *x = acc;
            }
        } else {
            for x in line.iter_mut() {
                acc = op(acc, *x);
                *x = acc;
            }
        }
    }
}

/// Build a cumulative-scan output for contiguous input lines. The old dense path
/// cloned the input and then mutated the clone into prefixes, which read and
/// wrote the entire dense buffer once before doing the scan. This writes prefix
/// outputs directly from the typed source slice. Forward serial scans use
/// `push` so each output element is written once; reverse or threaded scans use
/// an initialized buffer and fill disjoint whole-line blocks.
fn scan_contiguous_lines_to_vec<T, F>(
    src: &[T],
    axis_dim: usize,
    reverse: bool,
    init: T,
    op: F,
) -> Vec<T>
where
    T: Copy + Send + Sync,
    F: Fn(T, T) -> T + Sync,
{
    let outer = src.len() / axis_dim.max(1);
    // Work-scale to total work (src.len()), capped at the independent-line count.
    // A flat all-core split was spawn-overhead-dominated at moderate sizes (see
    // work_scaled_threads). Independent lines → bit-identical regardless of split.
    let threads = if src.len() >= CUMULATIVE_PARALLEL_MIN_ELEMS && outer > 1 {
        crate::arithmetic::work_scaled_threads(src.len()).min(outer)
    } else {
        1
    };

    if threads <= 1 {
        if reverse {
            let mut out = vec![init; src.len()];
            scan_contiguous_lines_from(src, &mut out, axis_dim, true, init, &op);
            return out;
        }

        let mut out = Vec::with_capacity(src.len());
        for line in src.chunks(axis_dim) {
            let mut acc = init;
            for &value in line {
                acc = op(acc, value);
                out.push(acc);
            }
        }
        return out;
    }

    let mut out = vec![init; src.len()];
    let lines_per = outer.div_ceil(threads);
    let block = lines_per * axis_dim;
    let op_ref = &op;
    std::thread::scope(|scope| {
        let mut src_rest = src;
        let mut out_rest: &mut [T] = out.as_mut_slice();
        while !src_rest.is_empty() {
            let take = block.min(src_rest.len());
            let (src_block, src_tail) = src_rest.split_at(take);
            let (out_block, out_tail) = out_rest.split_at_mut(take);
            src_rest = src_tail;
            out_rest = out_tail;
            scope.spawn(move || {
                scan_contiguous_lines_from(src_block, out_block, axis_dim, reverse, init, op_ref);
            });
        }
    });
    out
}

#[inline]
fn eval_cumulative_dense(
    tensor: &TensorValue,
    axis: usize,
    reverse: bool,
    int_init: i64,
    float_init: f64,
    int_op: &(impl Fn(i64, i64) -> i64 + Sync),
    float_op: &(impl Fn(f64, f64) -> f64 + Sync),
) -> Result<Option<Value>, EvalError> {
    let primitive = Primitive::Cumsum; // only used for stride-overflow error context
    let rank = tensor.shape.rank();
    let axis_dim = tensor.shape.dims[axis] as usize;
    let total = tensor.elements.len();
    if axis_dim == 0 || total == 0 {
        return Ok(None);
    }
    let strides = checked_strides(primitive, "cumulative input", &tensor.shape.dims)?;
    let axis_stride = strides[axis];
    let outer_count = total / axis_dim;

    // Shared per-line base-offset closure (all axes except `axis`).
    let line_base = |outer: usize| -> usize {
        let mut idx = outer;
        let mut flat = 0_usize;
        for ax in (0..rank).rev() {
            if ax == axis {
                continue;
            }
            let dim = tensor.shape.dims[ax] as usize;
            flat += (idx % dim) * strides[ax];
            idx /= dim;
        }
        flat
    };

    if tensor.dtype == DType::F64 {
        let Some(src) = tensor.elements.as_f64_slice() else {
            return Ok(None);
        };
        let out = if axis_stride == 1
            && total >= CUMULATIVE_PARALLEL_MIN_ELEMS
            && (!reverse || outer_count > 1)
        {
            // Large contiguous forward scans write dense output directly from
            // source slices, including the one-line case where cloning the full
            // input only adds a complete extra read/write pass. Reverse one-line
            // scans keep the clone+in-place path below.
            scan_contiguous_lines_to_vec(src, axis_dim, reverse, float_init, float_op)
        } else if axis_stride == 1 {
            let mut out = src.to_vec();
            scan_contiguous_lines_in_place(&mut out, axis_dim, reverse, float_init, float_op);
            out
        } else {
            let mut out = src.to_vec();
            for outer in 0..outer_count {
                let base = line_base(outer);
                let mut acc = float_init;
                if reverse {
                    for i in (0..axis_dim).rev() {
                        let fi = base + i * axis_stride;
                        acc = float_op(acc, out[fi]);
                        out[fi] = acc;
                    }
                } else {
                    for i in 0..axis_dim {
                        let fi = base + i * axis_stride;
                        acc = float_op(acc, out[fi]);
                        out[fi] = acc;
                    }
                }
            }
            out
        };
        return Ok(Some(Value::Tensor(TensorValue::new_f64_values(
            tensor.shape.clone(),
            out,
        )?)));
    }

    if tensor.dtype == DType::F32 {
        let Some(src) = tensor.elements.as_f32_slice() else {
            return Ok(None);
        };
        // Dense f32 cumulative: scan each line reading f32, accumulating in f64
        // (widen INLINE per element — no buffer), storing each step's running value
        // rounded back to f32. BIT-IDENTICAL to the generic per-`Literal` float
        // scan (lib path: `acc = float_op(acc, as_f64()); store reduce_real_literal
        // (F32, acc)`) — same f64 accumulator (never rounded mid-scan), same per-step
        // `as f32` round (`new_f32_values(acc as f32)` == `from_f32(acc as f32)`),
        // same per-line order. The scan is a sequential dependency (acc feeds the
        // next step), so it CANNOT reassociate/vectorize — exact incl. NaN.
        let mut out = vec![0.0f32; total];
        for outer in 0..outer_count {
            let base = line_base(outer);
            let mut acc = float_init;
            if reverse {
                for i in (0..axis_dim).rev() {
                    let fi = base + i * axis_stride;
                    acc = float_op(acc, f64::from(src[fi]));
                    out[fi] = acc as f32;
                }
            } else {
                for i in 0..axis_dim {
                    let fi = base + i * axis_stride;
                    acc = float_op(acc, f64::from(src[fi]));
                    out[fi] = acc as f32;
                }
            }
        }
        return Ok(Some(Value::Tensor(TensorValue::new_f32_values(
            tensor.shape.clone(),
            out,
        )?)));
    }

    if tensor.dtype == DType::I64 {
        let Some(src) = tensor.elements.as_i64_slice() else {
            return Ok(None);
        };
        let out = if axis_stride == 1 && total >= CUMULATIVE_PARALLEL_MIN_ELEMS && outer_count > 1 {
            scan_contiguous_lines_to_vec(src, axis_dim, reverse, int_init, int_op)
        } else if axis_stride == 1 {
            let mut out = src.to_vec();
            scan_contiguous_lines_in_place(&mut out, axis_dim, reverse, int_init, int_op);
            out
        } else {
            let mut out = src.to_vec();
            for outer in 0..outer_count {
                let base = line_base(outer);
                let mut acc = int_init;
                if reverse {
                    for i in (0..axis_dim).rev() {
                        let fi = base + i * axis_stride;
                        acc = int_op(acc, out[fi]);
                        out[fi] = acc;
                    }
                } else {
                    for i in 0..axis_dim {
                        let fi = base + i * axis_stride;
                        acc = int_op(acc, out[fi]);
                        out[fi] = acc;
                    }
                }
            }
            out
        };
        return Ok(Some(Value::Tensor(TensorValue::new_i64_values(
            tensor.shape.clone(),
            out,
        )?)));
    }

    Ok(None)
}

pub(crate) fn eval_cumulative(
    primitive: Primitive,
    inputs: &[Value],
    params: &std::collections::BTreeMap<String, String>,
    int_init: i64,
    float_init: f64,
    int_op: impl Fn(i64, i64) -> i64 + Sync,
    float_op: impl Fn(f64, f64) -> f64 + Sync,
) -> Result<Value, EvalError> {
    if inputs.len() != 1 {
        return Err(EvalError::ArityMismatch {
            primitive,
            expected: 1,
            actual: inputs.len(),
        });
    }

    match &inputs[0] {
        Value::Scalar(literal) => Ok(Value::Scalar(*literal)),
        Value::Tensor(tensor) => {
            let rank = tensor.shape.rank();
            if rank == 0 {
                return Ok(Value::Scalar(tensor.elements[0]));
            }

            let axis: usize = {
                let raw: i64 = params
                    .get("axis")
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);
                if raw < 0 {
                    (rank as i64 + raw) as usize
                } else {
                    raw as usize
                }
            };

            if axis >= rank {
                return Err(EvalError::Unsupported {
                    primitive,
                    detail: format!("axis {} out of bounds for rank {}", axis, rank),
                });
            }
            let reverse = parse_bool_param(primitive, params, "reverse", false)?;

            // Dense F64/I64 fast path: scan the contiguous typed backing directly,
            // skipping the Vec<Literal> materialization + per-element Literal
            // dispatch. Returns None for non-dense / non-F64/I64 -> generic below.
            if let Some(value) = eval_cumulative_dense(
                tensor, axis, reverse, int_init, float_init, &int_op, &float_op,
            )? {
                return Ok(value);
            }

            let is_integral = tensor.dtype == DType::I64 || tensor.dtype == DType::I32;
            let is_complex = matches!(tensor.dtype, DType::Complex64 | DType::Complex128);

            let axis_dim = tensor.shape.dims[axis] as usize;

            let total = tensor.elements.len();
            let mut elements = tensor.elements.to_vec();
            if axis_dim == 0 || total == 0 {
                return Ok(Value::Tensor(TensorValue::new(
                    tensor.dtype,
                    tensor.shape.clone(),
                    elements,
                )?));
            }

            let strides = checked_strides(primitive, "cumulative input", &tensor.shape.dims)?;
            let axis_stride = strides[axis];

            // For each "line" along the axis, do a prefix scan
            // A line is identified by all indices except the axis index
            let outer_count = total / axis_dim;

            for outer in 0..outer_count {
                // Compute the flat index of the first element of this line (axis index = 0)
                // outer is the index in the "all-but-axis" space
                let base = {
                    let mut idx = outer;
                    let mut flat = 0_usize;
                    for ax in (0..rank).rev() {
                        if ax == axis {
                            continue;
                        }
                        let dim = tensor.shape.dims[ax] as usize;
                        flat += (idx % dim) * strides[ax];
                        idx /= dim;
                    }
                    flat
                };

                if is_complex {
                    if !matches!(
                        primitive,
                        Primitive::Cumsum
                            | Primitive::Cumprod
                            | Primitive::Cummax
                            | Primitive::Cummin
                    ) {
                        return Err(EvalError::Unsupported {
                            primitive,
                            detail: "complex cumulative is supported only for cumsum, cumprod, cummax, and cummin"
                                .to_owned(),
                        });
                    }
                    // Cummax/Cummin seed (float_init, float_init) = ∓∞, the
                    // lexicographic sentinel the first element always beats.
                    let (mut acc_re, mut acc_im) = match primitive {
                        Primitive::Cumprod => (1.0_f64, 0.0_f64),
                        Primitive::Cummax | Primitive::Cummin => (float_init, float_init),
                        _ => (0.0_f64, 0.0_f64),
                    };
                    let iter: Box<dyn Iterator<Item = usize>> = if reverse {
                        Box::new((0..axis_dim).rev())
                    } else {
                        Box::new(0..axis_dim)
                    };
                    for i in iter {
                        let flat_idx = base + i * axis_stride;
                        let (re, im) = literal_to_complex_parts(primitive, elements[flat_idx])?;
                        match primitive {
                            Primitive::Cumprod => {
                                let new_re = acc_re * re - acc_im * im;
                                let new_im = acc_re * im + acc_im * re;
                                acc_re = new_re;
                                acc_im = new_im;
                            }
                            Primitive::Cummax => {
                                if complex_lex_cmp((re, im), (acc_re, acc_im)).is_gt() {
                                    acc_re = re;
                                    acc_im = im;
                                }
                            }
                            Primitive::Cummin => {
                                if complex_lex_cmp((re, im), (acc_re, acc_im)).is_lt() {
                                    acc_re = re;
                                    acc_im = im;
                                }
                            }
                            _ => {
                                acc_re += re;
                                acc_im += im;
                            }
                        }
                        elements[flat_idx] =
                            complex_literal_from_parts(tensor.dtype, acc_re, acc_im);
                    }
                } else if is_integral {
                    let mut acc = int_init;
                    if reverse {
                        for i in (0..axis_dim).rev() {
                            let flat_idx = base + i * axis_stride;
                            let val =
                                elements[flat_idx].as_i64().ok_or(EvalError::TypeMismatch {
                                    primitive,
                                    detail: "expected i64 tensor",
                                })?;
                            acc = int_op(acc, val);
                            elements[flat_idx] = Literal::I64(acc);
                        }
                    } else {
                        for i in 0..axis_dim {
                            let flat_idx = base + i * axis_stride;
                            let val =
                                elements[flat_idx].as_i64().ok_or(EvalError::TypeMismatch {
                                    primitive,
                                    detail: "expected i64 tensor",
                                })?;
                            acc = int_op(acc, val);
                            elements[flat_idx] = Literal::I64(acc);
                        }
                    }
                } else {
                    // Emit literals matching the input tensor's dtype so an
                    // F32/BF16/F16 cumulative output doesn't end up
                    // declaring its narrow dtype while storing F64Bits.
                    let out_dtype = reduce_real_output_dtype(tensor.dtype);
                    let mut acc = float_init;
                    if reverse {
                        for i in (0..axis_dim).rev() {
                            let flat_idx = base + i * axis_stride;
                            let val =
                                elements[flat_idx].as_f64().ok_or(EvalError::TypeMismatch {
                                    primitive,
                                    detail: "expected numeric tensor",
                                })?;
                            acc = float_op(acc, val);
                            elements[flat_idx] = reduce_real_literal(out_dtype, acc);
                        }
                    } else {
                        for i in 0..axis_dim {
                            let flat_idx = base + i * axis_stride;
                            let val =
                                elements[flat_idx].as_f64().ok_or(EvalError::TypeMismatch {
                                    primitive,
                                    detail: "expected numeric tensor",
                                })?;
                            acc = float_op(acc, val);
                            elements[flat_idx] = reduce_real_literal(out_dtype, acc);
                        }
                    }
                }
            }

            Ok(Value::Tensor(TensorValue::new(
                tensor.dtype,
                tensor.shape.clone(),
                elements,
            )?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn s_f64(v: f64) -> Value {
        Value::Scalar(Literal::from_f64(v))
    }
    fn v_f64(data: &[f64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![data.len() as u32],
                },
                data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        )
    }
    fn v_i64(data: &[i64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape {
                    dims: vec![data.len() as u32],
                },
                data.iter().map(|&v| Literal::I64(v)).collect(),
            )
            .unwrap(),
        )
    }
    fn empty_i64(dims: Vec<u32>) -> Value {
        Value::Tensor(TensorValue::new(DType::I64, Shape { dims }, vec![]).unwrap())
    }
    fn empty_bool(dims: Vec<u32>) -> Value {
        Value::Tensor(TensorValue::new(DType::Bool, Shape { dims }, vec![]).unwrap())
    }
    fn mat_f64(rows: u32, cols: u32, data: &[f64]) -> Value {
        Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape {
                    dims: vec![rows, cols],
                },
                data.iter().map(|&v| Literal::from_f64(v)).collect(),
            )
            .unwrap(),
        )
    }
    fn extract_f64(val: &Value) -> f64 {
        val.as_f64_scalar().unwrap()
    }
    fn extract_i64(val: &Value) -> i64 {
        val.as_i64_scalar().expect("expected i64 scalar")
    }
    fn extract_f64_bits(val: &Value) -> u64 {
        val.as_f64_scalar().expect("expected f64 scalar").to_bits()
    }
    fn extract_f64_vec(val: &Value) -> Vec<f64> {
        val.as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_f64().unwrap())
            .collect()
    }

    // ── Reduce all axes ──

    #[test]
    fn reduce_sum_scalar() {
        let result = eval_reduce(
            Primitive::ReduceSum,
            &[s_f64(42.0)],
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();
        assert!((extract_f64(&result) - 42.0).abs() < 1e-12);
    }

    #[test]
    fn reduce_sum_vector() {
        let result = eval_reduce(
            Primitive::ReduceSum,
            &[v_f64(&[1.0, 2.0, 3.0, 4.0])],
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();
        assert!((extract_f64(&result) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn dense_f64_reduce_sum_full_fast_path_bit_identical_to_literal_path() {
        let data = std::hint::black_box([
            1.5,
            -0.0,
            f64::from_bits(0x7ff8_0000_0000_0001),
            -3.25,
            f64::INFINITY,
            0.0,
        ]);
        let dense = Value::vector_f64(&data).unwrap();
        assert!(dense.as_tensor().unwrap().elements.as_f64_slice().is_some());
        let literal = v_f64(&data);
        assert!(
            literal
                .as_tensor()
                .unwrap()
                .elements
                .as_f64_slice()
                .is_none()
        );

        let dense_result = eval_reduce(
            Primitive::ReduceSum,
            &[dense],
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();
        let literal_result = eval_reduce(
            Primitive::ReduceSum,
            &[literal],
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();

        assert_eq!(
            extract_f64_bits(&dense_result),
            extract_f64_bits(&literal_result)
        );
    }

    #[test]
    fn dense_f64_reduce_sum_full_malformed_declared_f64_falls_back() {
        let input = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: vec![3] },
                vec![Literal::I64(2), Literal::from_f64(3.5), Literal::U32(4)],
            )
            .unwrap(),
        );
        assert!(input.as_tensor().unwrap().elements.as_f64_slice().is_none());

        let result = eval_reduce(
            Primitive::ReduceSum,
            &[input],
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();

        assert_eq!(extract_f64_bits(&result), 9.5_f64.to_bits());
    }

    /// Bit-exact parity for the generalized dense F64 full-reduction fast path
    /// across ReduceProd/ReduceMax/ReduceMin (ReduceSum is covered above): a
    /// dense `vector_f64` input (fast path) must produce the same scalar bits as
    /// the `Vec<Literal>`-backed tensor (generic loop) for every primitive,
    /// including NaN/±inf/signed-zero edge cases. Routes through the real
    /// `eval_primitive` dispatcher so the exact `float_init`/`float_op` per
    /// primitive are exercised.
    #[test]
    fn dense_f64_full_reduce_prod_max_min_bit_identical_to_literal_path() {
        let data = std::hint::black_box([
            1.5,
            -0.0,
            f64::from_bits(0x7ff8_0000_0000_0001),
            -3.25,
            f64::INFINITY,
            0.0,
            f64::NEG_INFINITY,
            2.0,
        ]);
        let params = BTreeMap::new();
        for primitive in [
            Primitive::ReduceProd,
            Primitive::ReduceMax,
            Primitive::ReduceMin,
        ] {
            let dense = Value::vector_f64(&data).unwrap();
            assert!(dense.as_tensor().unwrap().elements.as_f64_slice().is_some());
            let literal = v_f64(&data);
            assert!(
                literal
                    .as_tensor()
                    .unwrap()
                    .elements
                    .as_f64_slice()
                    .is_none()
            );

            let dense_result =
                crate::eval_primitive(primitive, std::slice::from_ref(&dense), &params).unwrap();
            let literal_result =
                crate::eval_primitive(primitive, std::slice::from_ref(&literal), &params).unwrap();

            assert_eq!(
                extract_f64_bits(&dense_result),
                extract_f64_bits(&literal_result),
                "dense/literal mismatch for {primitive:?}"
            );
        }
    }

    /// f32 reductions (JAX's default dtype) now take the dense-f64-view fast path
    /// (full + axis). Must be BIT-FOR-BIT identical to the generic per-`Literal`
    /// loop (forced via a boxed f32 input whose `as_f32_slice` is None), across
    /// sum/prod/max/min, full reduce + each axis, incl NaN/±inf/-0.0. No NaN
    /// canonicalization needed: both fold the SAME widened-f64 values in the SAME
    /// per-output-cell ascending order, so even inf+(-inf)=NaN matches bit-for-bit.
    #[test]
    fn dense_f32_reduce_bit_identical_to_literal_path() {
        let data: Vec<f32> = std::hint::black_box(vec![
            1.5,
            -0.0,
            2.0,
            -3.25,
            f32::INFINITY,
            0.0,
            f32::NEG_INFINITY,
            2.5,
            -1.0,
            4.0,
            f32::from_bits(0x7fc0_0001),
            0.5,
        ]);
        let dims = vec![3u32, 4u32];
        let dense = Value::Tensor(
            TensorValue::new_f32_values(Shape { dims: dims.clone() }, data.clone()).unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape { dims: dims.clone() },
                data.iter().map(|&v| Literal::from_f32(v)).collect(),
            )
            .unwrap(),
        );
        assert!(dense.as_tensor().unwrap().elements.as_f32_slice().is_some());
        assert!(boxed.as_tensor().unwrap().elements.as_f32_slice().is_none());
        // Extract f32 bits from either a Scalar (full reduce) or a Tensor (axis
        // reduce); assert F32 dtype is preserved in both forms. Canonicalize NaN:
        // a reduce_sum spanning +inf and -inf yields inf+(-inf)=NaN whose SIGN BIT
        // is IEEE-754-unspecified and differs between the dense fold and the generic
        // per-Literal fold — not a parity concern (JAX/XLA don't specify it; no
        // golden digests), matching the project's conv/reduce_window NaN-bit-test
        // precedent. Finite results are still compared exactly.
        let f32_bits = |v: &Value| -> Vec<u32> {
            let lit_bits = |l: &Literal| match l {
                Literal::F32Bits(b) => {
                    if f32::from_bits(*b).is_nan() {
                        0x7fc0_0000
                    } else {
                        *b
                    }
                }
                o => panic!("expected f32, got {o:?}"),
            };
            match v {
                Value::Scalar(l) => vec![lit_bits(l)],
                Value::Tensor(t) => {
                    assert_eq!(t.dtype, DType::F32, "f32 output dtype preserved");
                    t.elements.iter().map(lit_bits).collect()
                }
            }
        };
        for primitive in [
            Primitive::ReduceSum,
            Primitive::ReduceProd,
            Primitive::ReduceMax,
            Primitive::ReduceMin,
        ] {
            for axes in [None, Some("0"), Some("1")] {
                let mut p = BTreeMap::new();
                if let Some(a) = axes {
                    p.insert("axes".to_owned(), a.to_owned());
                }
                let d = crate::eval_primitive(primitive, std::slice::from_ref(&dense), &p).unwrap();
                let l = crate::eval_primitive(primitive, std::slice::from_ref(&boxed), &p).unwrap();
                assert_eq!(
                    f32_bits(&d),
                    f32_bits(&l),
                    "f32 {primitive:?} axes={axes:?} dense != generic"
                );
            }
        }
    }

    #[test]
    fn dense_half_float_reduce_bit_identical_to_literal_path() {
        // Dense BF16/F16 (as_half_float_slice) takes the new dense axis-reduce path;
        // boxed half Literals take the generic odometer. Both decode each tap via
        // Literal::as_f64() and fold in the same order, so results are bit-identical
        // (NaN sign canonicalized, as in the f32 test — inf+(-inf) sign is
        // unspecified). Axis reductions exercise the new branch.
        for dtype in [DType::BF16, DType::F16] {
            let raw: Vec<u16> = (0..12)
                .map(|i| match i {
                    4 => {
                        if dtype == DType::F16 {
                            0x7c00
                        } else {
                            0x7f80
                        }
                    } // +inf
                    5 => {
                        if dtype == DType::F16 {
                            0xfc00
                        } else {
                            0xff80
                        }
                    } // -inf
                    6 => {
                        if dtype == DType::F16 {
                            0x7e00
                        } else {
                            0x7fc0
                        }
                    } // NaN
                    7 => 0x8000, // -0
                    _ => (i as u16).wrapping_mul(53).wrapping_add(1),
                })
                .collect();
            let dims = vec![3u32, 4u32];
            let mk = move |b: u16| {
                if dtype == DType::BF16 {
                    Literal::BF16Bits(b)
                } else {
                    Literal::F16Bits(b)
                }
            };
            let dense = Value::Tensor(
                TensorValue::new_half_float_values(
                    dtype,
                    Shape { dims: dims.clone() },
                    raw.clone(),
                )
                .unwrap(),
            );
            let boxed = Value::Tensor(
                TensorValue::new(
                    dtype,
                    Shape { dims: dims.clone() },
                    raw.iter().copied().map(mk).collect(),
                )
                .unwrap(),
            );
            assert!(
                dense
                    .as_tensor()
                    .unwrap()
                    .elements
                    .as_half_float_slice()
                    .is_some()
            );
            assert!(
                boxed
                    .as_tensor()
                    .unwrap()
                    .elements
                    .as_half_float_slice()
                    .is_none()
            );
            let is_nan_half = |b: u16| -> bool {
                if dtype == DType::BF16 {
                    Literal::BF16Bits(b).as_f64().is_some_and(|v| v.is_nan())
                } else {
                    Literal::F16Bits(b).as_f64().is_some_and(|v| v.is_nan())
                }
            };
            let nan_canon = if dtype == DType::F16 { 0x7e00 } else { 0x7fc0 };
            let half_bits = |v: &Value| -> Vec<u16> {
                let lit_bits = |l: &Literal| match l {
                    Literal::BF16Bits(b) | Literal::F16Bits(b) => {
                        if is_nan_half(*b) {
                            nan_canon
                        } else {
                            *b
                        }
                    }
                    o => panic!("expected half, got {o:?}"),
                };
                match v {
                    Value::Scalar(l) => vec![lit_bits(l)],
                    Value::Tensor(t) => {
                        assert_eq!(t.dtype, dtype, "half output dtype preserved");
                        t.elements.iter().map(lit_bits).collect()
                    }
                }
            };
            for primitive in [
                Primitive::ReduceSum,
                Primitive::ReduceProd,
                Primitive::ReduceMax,
                Primitive::ReduceMin,
            ] {
                for axes in [None, Some("0"), Some("1")] {
                    let mut p = BTreeMap::new();
                    if let Some(a) = axes {
                        p.insert("axes".to_owned(), a.to_owned());
                    }
                    let d =
                        crate::eval_primitive(primitive, std::slice::from_ref(&dense), &p).unwrap();
                    let l =
                        crate::eval_primitive(primitive, std::slice::from_ref(&boxed), &p).unwrap();
                    assert_eq!(
                        half_bits(&d),
                        half_bits(&l),
                        "{dtype:?} {primitive:?} axes={axes:?} dense != generic"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_bf16_reduce_sum_axis_dense_vs_generic() {
        use std::time::Instant;
        let (rows, cols) = (4096usize, 1024usize); // reduce axis 1 -> [4096]
        let raw: Vec<u16> = (0..rows * cols)
            .map(|i| (i as u16).wrapping_mul(37).wrapping_add(0x3f00))
            .collect();
        let dims = vec![rows as u32, cols as u32];
        let dense = Value::Tensor(
            TensorValue::new_half_float_values(
                DType::BF16,
                Shape { dims: dims.clone() },
                raw.clone(),
            )
            .unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::BF16,
                Shape { dims: dims.clone() },
                raw.iter().copied().map(Literal::BF16Bits).collect(),
            )
            .unwrap(),
        );
        let p = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        let time = |input: &Value| {
            let _ = crate::eval_primitive(Primitive::ReduceSum, std::slice::from_ref(input), &p)
                .unwrap();
            let mut best = f64::MAX;
            for _ in 0..20 {
                let t = Instant::now();
                let _ =
                    crate::eval_primitive(Primitive::ReduceSum, std::slice::from_ref(input), &p)
                        .unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let generic = time(&boxed);
        let dense_t = time(&dense);
        println!(
            "BENCH bf16 reduce_sum axis1 [{rows},{cols}]: generic(per-Literal)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_f32_reduce_sum_axis_dense_vs_generic() {
        use std::time::Instant;
        let (rows, cols) = (4096usize, 1024usize); // reduce over axis 1 -> [4096]
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 251) as f32) * 0.013 - 1.6)
            .collect();
        let dims = vec![rows as u32, cols as u32];
        let dense = Value::Tensor(
            TensorValue::new_f32_values(Shape { dims: dims.clone() }, data.clone()).unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape { dims: dims.clone() },
                data.iter().map(|&v| Literal::from_f32(v)).collect(),
            )
            .unwrap(),
        );
        let p = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        let time = |input: &Value| {
            let _ = crate::eval_primitive(Primitive::ReduceSum, std::slice::from_ref(input), &p)
                .unwrap();
            let mut best = f64::MAX;
            for _ in 0..20 {
                let t = Instant::now();
                let _ =
                    crate::eval_primitive(Primitive::ReduceSum, std::slice::from_ref(input), &p)
                        .unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let generic = time(&boxed);
        let dense_t = time(&dense);
        println!(
            "BENCH f32 reduce_sum axis1 [{rows},{cols}]: generic(per-Literal)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    #[test]
    fn reduce_max_vector() {
        let result = eval_reduce(
            Primitive::ReduceMax,
            &[v_f64(&[3.0, 1.0, 4.0, 1.0, 5.0])],
            i64::MIN,
            f64::NEG_INFINITY,
            |a, b| a.max(b),
            |a, b| a.max(b),
        )
        .unwrap();
        assert!((extract_f64(&result) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn reduce_min_vector() {
        let result = eval_reduce(
            Primitive::ReduceMin,
            &[v_f64(&[3.0, 1.0, 4.0, 1.0, 5.0])],
            i64::MAX,
            f64::INFINITY,
            |a, b| a.min(b),
            |a, b| a.min(b),
        )
        .unwrap();
        assert!((extract_f64(&result) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn reduce_prod_vector() {
        let result = eval_reduce(
            Primitive::ReduceProd,
            &[v_f64(&[2.0, 3.0, 4.0])],
            1,
            1.0,
            |a, b| a * b,
            |a, b| a * b,
        )
        .unwrap();
        assert!((extract_f64(&result) - 24.0).abs() < 1e-12);
    }

    #[test]
    fn reduce_sum_integer() {
        let result = eval_reduce(
            Primitive::ReduceSum,
            &[v_i64(&[10, 20, 30])],
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();
        assert_eq!(extract_i64(&result), 60);
    }

    /// Bit-exact parity for the dense-i64 reduction fast paths (full reduction
    /// and axis reduction) vs the `Vec<Literal>` path, across
    /// ReduceSum/Prod/Max/Min and every axis subset. Routes through
    /// `eval_primitive` so the real per-primitive int_init/int_op run. Values
    /// are kept small (incl. negatives + a zero) so the plain (non-wrapping)
    /// sum/prod int_ops do not overflow — overflow behavior is identical on
    /// both paths anyway and is covered for Add by the elementwise tests.
    #[test]
    fn dense_i64_reduce_bit_identical_to_literal_path() {
        let dims = vec![2_u32, 3, 4];
        let n = 2 * 3 * 4;
        let data: Vec<i64> = (0..n)
            .map(|i| match i % 7 {
                0 => 0,
                1 => 2,
                2 => -3,
                3 => 1,
                4 => -1,
                5 => 3,
                _ => -2,
            })
            .collect();

        let dense = Value::Tensor(
            TensorValue::new_i64_values(Shape { dims: dims.clone() }, data.clone()).unwrap(),
        );
        assert!(dense.as_tensor().unwrap().elements.as_i64_slice().is_some());
        let literal = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: dims.clone() },
                data.iter().copied().map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        assert!(
            literal
                .as_tensor()
                .unwrap()
                .elements
                .as_i64_slice()
                .is_none()
        );

        let int_results = |v: &Value, params: &BTreeMap<String, String>, prim: Primitive| {
            let out = crate::eval_primitive(prim, std::slice::from_ref(v), params).unwrap();
            match out {
                Value::Scalar(l) => vec![l.as_i64().unwrap()],
                Value::Tensor(t) => t.elements.iter().map(|l| l.as_i64().unwrap()).collect(),
            }
        };

        for prim in [
            Primitive::ReduceSum,
            Primitive::ReduceProd,
            Primitive::ReduceMax,
            Primitive::ReduceMin,
        ] {
            // Full reduction (no axes) + each axis subset.
            for axes in ["", "0", "1", "2", "0,1", "1,2", "0,2", "-1"] {
                let mut params = BTreeMap::new();
                if !axes.is_empty() {
                    params.insert("axes".to_owned(), axes.to_owned());
                }
                assert_eq!(
                    int_results(&dense, &params, prim),
                    int_results(&literal, &params, prim),
                    "mismatch {prim:?} axes={axes:?}"
                );
            }
        }
    }

    /// Bit-exact parity for the dense-bool full-reduction fast path (ReduceAnd /
    /// ReduceOr) vs the Vec<Literal> path, over all-true/all-false/mixed inputs.
    #[test]
    fn dense_bool_reduce_bit_identical_to_literal_path() {
        for pattern in 0..4 {
            let n = 600usize;
            let data: Vec<bool> = (0..n)
                .map(|i| match pattern {
                    0 => true,
                    1 => false,
                    2 => i != 300, // single false
                    _ => i % 2 == 0,
                })
                .collect();
            let dense = Value::Tensor(
                TensorValue::new_bool_values(
                    Shape {
                        dims: vec![n as u32],
                    },
                    data.clone(),
                )
                .unwrap(),
            );
            assert!(
                dense
                    .as_tensor()
                    .unwrap()
                    .elements
                    .as_bool_slice()
                    .is_some()
            );
            let literal = Value::Tensor(
                TensorValue::new(
                    DType::Bool,
                    Shape {
                        dims: vec![n as u32],
                    },
                    data.iter().copied().map(Literal::Bool).collect(),
                )
                .unwrap(),
            );
            assert!(
                literal
                    .as_tensor()
                    .unwrap()
                    .elements
                    .as_bool_slice()
                    .is_none()
            );
            let params = BTreeMap::new();
            for prim in [Primitive::ReduceAnd, Primitive::ReduceOr] {
                let d = crate::eval_primitive(prim, std::slice::from_ref(&dense), &params).unwrap();
                let l =
                    crate::eval_primitive(prim, std::slice::from_ref(&literal), &params).unwrap();
                assert_eq!(
                    d.as_scalar_literal(),
                    l.as_scalar_literal(),
                    "{prim:?} pattern={pattern}"
                );
            }
        }
    }

    /// Bit-exact parity for the dense-bool AXIS reduction (odometer + as_bool_slice)
    /// vs the Vec<Literal> decode loop, across ReduceAnd/ReduceOr and every axis
    /// subset of a rank-3 bool tensor.
    #[test]
    fn dense_bool_axis_reduce_bit_identical_to_literal_path() {
        let dims = vec![2_u32, 3, 4];
        let n = 2 * 3 * 4;
        let data: Vec<bool> = (0..n).map(|i| (i * 7 + 1) % 3 != 0).collect();
        let dense = Value::Tensor(
            TensorValue::new_bool_values(Shape { dims: dims.clone() }, data.clone()).unwrap(),
        );
        assert!(
            dense
                .as_tensor()
                .unwrap()
                .elements
                .as_bool_slice()
                .is_some()
        );
        let literal = Value::Tensor(
            TensorValue::new(
                DType::Bool,
                Shape { dims: dims.clone() },
                data.iter().copied().map(Literal::Bool).collect(),
            )
            .unwrap(),
        );
        assert!(
            literal
                .as_tensor()
                .unwrap()
                .elements
                .as_bool_slice()
                .is_none()
        );

        let bools = |v: &Value| -> Vec<bool> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| matches!(l, Literal::Bool(true)))
                .collect()
        };
        for prim in [Primitive::ReduceAnd, Primitive::ReduceOr] {
            for axes in ["0", "1", "2", "0,1", "1,2", "0,2"] {
                let mut params = BTreeMap::new();
                params.insert("axes".to_owned(), axes.to_owned());
                let d = crate::eval_primitive(prim, std::slice::from_ref(&dense), &params).unwrap();
                let l =
                    crate::eval_primitive(prim, std::slice::from_ref(&literal), &params).unwrap();
                assert_eq!(
                    d.as_tensor().unwrap().shape.dims,
                    l.as_tensor().unwrap().shape.dims
                );
                assert_eq!(bools(&d), bools(&l), "{prim:?} axes={axes}");
            }
        }
    }

    #[test]
    fn reduce_arity_error() {
        let result = eval_reduce(
            Primitive::ReduceSum,
            &[],
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        );
        assert!(result.is_err());
    }

    /// Bit-exact parity for the dense F64 axis-reduction odometer fast path: a
    /// dense `new_f64_values` tensor (fast path) must produce element-for-element
    /// identical bits to the `Vec<Literal>`-backed tensor (generic decode loop)
    /// for ReduceSum/Prod/Max/Min across every axis subset of a rank-3 tensor,
    /// Isomorphism proof for the threaded trailing-axis reduction: a large
    /// `[N,M]` reduce over axis 1 (>= 1<<18, output rows independent) must equal,
    /// bit-for-bit, an independent per-row serial fold in ascending order.
    #[test]
    fn threaded_trailing_reduce_bit_identical() {
        let (n, m) = (4096usize, 64usize); // 262_144 = 1<<18 -> threaded
        assert!(n * m >= (1usize << 18) && n > 1);
        let data: Vec<f64> = (0..n * m)
            .map(|i| ((i % 101) as f64 - 50.0) * 0.125 + 0.3 * ((i % 7) as f64))
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![n as u32, m as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );

        type Op = fn(f64, f64) -> f64;
        let cases: [(Primitive, f64, Op); 4] = [
            (Primitive::ReduceSum, 0.0, |a, b| a + b),
            (Primitive::ReduceProd, 1.0, |a, b| a * b),
            (Primitive::ReduceMax, f64::NEG_INFINITY, crate::jax_max_f64),
            (Primitive::ReduceMin, f64::INFINITY, crate::jax_min_f64),
        ];
        for (prim, init, op) in cases {
            let mut p = BTreeMap::new();
            p.insert("axes".to_owned(), "1".to_owned());
            let got = extract_f64_vec(
                &crate::eval_primitive(prim, std::slice::from_ref(&input), &p).unwrap(),
            );
            let mut expect: Vec<f64> = Vec::with_capacity(n);
            for row in data.chunks(m) {
                let mut acc = init;
                for &v in row {
                    acc = op(acc, v);
                }
                expect.push(acc);
            }
            assert_eq!(got.len(), expect.len());
            for i in 0..n {
                assert_eq!(got[i].to_bits(), expect[i].to_bits(), "{prim:?} row {i}");
            }
        }
    }

    /// Isomorphism proof for the leading-axis (column) reduction fast path: a large
    /// `[N,M]` reduce over axis 0 (>= 1<<18) must equal, bit-for-bit, a column
    /// accumulation that folds k in ascending order (the odometer's emission order).
    #[test]
    fn leading_axis_column_reduce_bit_identical() {
        let (n, m) = (4096usize, 64usize); // 262_144 = 1<<18
        assert!(n * m >= (1usize << 18) && m > 1);
        let data: Vec<f64> = (0..n * m)
            .map(|i| ((i % 103) as f64 - 51.0) * 0.0625 + 0.2 * ((i % 5) as f64))
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![n as u32, m as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );

        type Op = fn(f64, f64) -> f64;
        let cases: [(Primitive, f64, Op); 4] = [
            (Primitive::ReduceSum, 0.0, |a, b| a + b),
            (Primitive::ReduceProd, 1.0, |a, b| a * b),
            (Primitive::ReduceMax, f64::NEG_INFINITY, crate::jax_max_f64),
            (Primitive::ReduceMin, f64::INFINITY, crate::jax_min_f64),
        ];
        for (prim, init, op) in cases {
            let mut p = BTreeMap::new();
            p.insert("axes".to_owned(), "0".to_owned());
            let got = extract_f64_vec(
                &crate::eval_primitive(prim, std::slice::from_ref(&input), &p).unwrap(),
            );
            let mut expect = vec![init; m];
            for k in 0..n {
                for j in 0..m {
                    expect[j] = op(expect[j], data[k * m + j]);
                }
            }
            assert_eq!(got.len(), m);
            for j in 0..m {
                assert_eq!(got[j].to_bits(), expect[j].to_bits(), "{prim:?} col {j}");
            }
        }
    }

    /// Per-element dense fast path equals the generic literal path for every axis
    /// reduction across `reduce_sum/prod/max/min` on a small tensor (serial),
    /// including NaN/±inf/signed-zero values. Routes through `eval_primitive` so
    /// the real per-primitive float_init/float_op are exercised.
    #[test]
    fn dense_f64_axis_reduce_bit_identical_to_literal_path() {
        let dims = vec![2_u32, 3, 4];
        let n = 2 * 3 * 4;
        let data: Vec<f64> = (0..n)
            .map(|i| match i {
                5 => f64::from_bits(0x7ff8_0000_0000_0001),
                7 => f64::INFINITY,
                11 => f64::NEG_INFINITY,
                13 => -0.0,
                17 => 0.0,
                _ => (i as f64 - 9.5) * 0.375,
            })
            .collect();

        let dense = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, data.clone()).unwrap(),
        );
        assert!(dense.as_tensor().unwrap().elements.as_f64_slice().is_some());
        let literal = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: dims.clone() },
                data.iter().copied().map(Literal::from_f64).collect(),
            )
            .unwrap(),
        );
        assert!(
            literal
                .as_tensor()
                .unwrap()
                .elements
                .as_f64_slice()
                .is_none()
        );

        for primitive in [
            Primitive::ReduceSum,
            Primitive::ReduceProd,
            Primitive::ReduceMax,
            Primitive::ReduceMin,
        ] {
            for axes in ["0", "1", "2", "0,1", "0,2", "1,2", "-1", "0,-1"] {
                let mut params = BTreeMap::new();
                params.insert("axes".to_owned(), axes.to_owned());

                let dense_out =
                    crate::eval_primitive(primitive, std::slice::from_ref(&dense), &params)
                        .unwrap();
                let literal_out =
                    crate::eval_primitive(primitive, std::slice::from_ref(&literal), &params)
                        .unwrap();

                let dense_t = dense_out.as_tensor().unwrap();
                let literal_t = literal_out.as_tensor().unwrap();
                assert_eq!(
                    dense_t.shape.dims, literal_t.shape.dims,
                    "{primitive:?} {axes}"
                );
                let dense_bits: Vec<u64> = dense_t
                    .elements
                    .iter()
                    .map(|l| l.as_f64().unwrap().to_bits())
                    .collect();
                let literal_bits: Vec<u64> = literal_t
                    .elements
                    .iter()
                    .map(|l| l.as_f64().unwrap().to_bits())
                    .collect();
                assert_eq!(
                    dense_bits, literal_bits,
                    "mismatch {primitive:?} axes={axes}"
                );
            }
        }
    }

    // ── Reduce along axes ──

    #[test]
    fn reduce_axes_sum_rows() {
        // [[1, 2], [3, 4]] reduced along axis 0 (rows) → [4, 6]
        let m = mat_f64(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "0".to_owned());
        let result = eval_reduce_axes(
            Primitive::ReduceSum,
            &[m],
            &params,
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();
        let vals = extract_f64_vec(&result);
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 4.0).abs() < 1e-12);
        assert!((vals[1] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn reduce_axes_sum_cols() {
        // [[1, 2], [3, 4]] reduced along axis 1 (cols) → [3, 7]
        let m = mat_f64(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "1".to_owned());
        let result = eval_reduce_axes(
            Primitive::ReduceSum,
            &[m],
            &params,
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();
        let vals = extract_f64_vec(&result);
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 3.0).abs() < 1e-12);
        assert!((vals[1] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn reduce_axes_rejects_duplicate_axes_after_normalization() {
        let m = mat_f64(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "1,-1".to_owned());

        let err = eval_reduce_axes(
            Primitive::ReduceSum,
            &[m],
            &params,
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .expect_err("duplicate canonical axes should be rejected")
        .to_string();

        assert!(
            err.contains("duplicate value in axes"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn bitwise_reduce_axes_rejects_duplicate_axes_after_normalization() {
        let x = Value::Tensor(
            TensorValue::new(
                DType::Bool,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::Bool(true),
                    Literal::Bool(false),
                    Literal::Bool(false),
                    Literal::Bool(true),
                ],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "1,-1".to_owned());

        let err = eval_reduce_bitwise_axes(
            Primitive::ReduceOr,
            &[x],
            &params,
            0,
            false,
            |a, b| a | b,
            |a, b| a || b,
        )
        .expect_err("duplicate canonical bitwise axes should be rejected")
        .to_string();

        assert!(
            err.contains("duplicate value in axes"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn reduce_axes_empty_kept_zero_dim_short_circuits_huge_product() {
        let huge = u32::MAX;
        let x = empty_i64(vec![1, huge, huge, huge, 0]);
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "0".to_owned());

        let result = eval_reduce_axes(
            Primitive::ReduceSum,
            &[x],
            &params,
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .expect("zero-sized output should return before huge shape products overflow");

        let tensor = result.as_tensor().expect("tensor output");
        assert_eq!(tensor.shape.dims, vec![huge, huge, huge, 0]);
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn reduce_axes_rejects_overflowing_kept_shape_without_panicking() {
        let huge = u32::MAX;
        let x = empty_i64(vec![0, huge, huge, huge]);
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "0".to_owned());

        let err = eval_reduce_axes(
            Primitive::ReduceSum,
            &[x],
            &params,
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .expect_err("overflowing non-empty reduction output shape should be rejected")
        .to_string();

        assert!(
            err.contains("reduction output shape overflows usize"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn reduce_axes_rejects_impossible_identity_output_allocation() {
        let huge = u32::MAX;
        let x = empty_i64(vec![0, huge, huge]);
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "0".to_owned());

        let err = eval_reduce_axes(
            Primitive::ReduceSum,
            &[x],
            &params,
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .expect_err("huge identity-filled reduction output should fail with a typed error")
        .to_string();

        assert!(
            err.contains("reduction integer accumulator allocation failed"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn bitwise_reduce_axes_rejects_overflowing_kept_shape_without_panicking() {
        let huge = u32::MAX;
        let x = empty_bool(vec![0, huge, huge, huge]);
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "0".to_owned());

        let err = eval_reduce_bitwise_axes(
            Primitive::ReduceAnd,
            &[x],
            &params,
            0,
            true,
            |a, b| a & b,
            |a, b| a & b,
        )
        .expect_err("overflowing non-empty bitwise reduction output shape should be rejected")
        .to_string();

        assert!(
            err.contains("bitwise reduction output shape overflows usize"),
            "unexpected error: {err}"
        );
    }

    // ── Cumulative ──

    /// Bit-exact parity for the dense F64/I64 cumulative fast path vs the
    /// Vec<Literal> generic path, across Cumsum/Cumprod/Cummax/Cummin, forward +
    /// reverse, on a multi-line [3,40] tensor. Routes through eval_primitive so
    /// the real per-primitive init/op run. Small bounded values avoid prod
    /// overflow (identical on both paths anyway).
    #[test]
    fn dense_cumulative_bit_identical_to_literal_path() {
        let rows = 3usize;
        let cols = 40usize;
        let dims = vec![rows as u32, cols as u32];
        let f: Vec<f64> = (0..rows * cols)
            .map(|i| match i % 7 {
                0 => 0.0,
                1 => -1.5,
                2 => 2.0,
                3 => -0.5,
                _ => 1.25,
            })
            .collect();
        let n: Vec<i64> = (0..(rows * cols) as i64).map(|i| (i % 5) - 2).collect();

        let dense_f = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, f.clone()).unwrap(),
        );
        assert!(
            dense_f
                .as_tensor()
                .unwrap()
                .elements
                .as_f64_slice()
                .is_some()
        );
        let lit_f = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims: dims.clone() },
                f.iter().copied().map(Literal::from_f64).collect(),
            )
            .unwrap(),
        );
        let dense_i = Value::Tensor(
            TensorValue::new_i64_values(Shape { dims: dims.clone() }, n.clone()).unwrap(),
        );
        let lit_i = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: dims.clone() },
                n.iter().copied().map(Literal::I64).collect(),
            )
            .unwrap(),
        );
        let fbits = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap().to_bits())
                .collect()
        };
        let ints = |v: &Value| -> Vec<i64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_i64().unwrap())
                .collect()
        };

        for prim in [
            Primitive::Cumsum,
            Primitive::Cumprod,
            Primitive::Cummax,
            Primitive::Cummin,
        ] {
            for (axis, rev) in [("0", "false"), ("1", "false"), ("1", "true"), ("0", "true")] {
                let mut p = BTreeMap::new();
                p.insert("axis".to_owned(), axis.to_owned());
                p.insert("reverse".to_owned(), rev.to_owned());
                let df = crate::eval_primitive(prim, std::slice::from_ref(&dense_f), &p).unwrap();
                let lf = crate::eval_primitive(prim, std::slice::from_ref(&lit_f), &p).unwrap();
                assert_eq!(fbits(&df), fbits(&lf), "f64 {prim:?} axis={axis} rev={rev}");
                let di = crate::eval_primitive(prim, std::slice::from_ref(&dense_i), &p).unwrap();
                let li = crate::eval_primitive(prim, std::slice::from_ref(&lit_i), &p).unwrap();
                assert_eq!(ints(&di), ints(&li), "i64 {prim:?} axis={axis} rev={rev}");
            }
        }
    }

    #[test]
    fn large_dense_f64_cumsum_single_line_matches_literal_path() {
        let len = CUMULATIVE_PARALLEL_MIN_ELEMS + 17;
        let dims = vec![len as u32];
        let data: Vec<f64> = (0..len)
            .map(|i| match i % 11 {
                0 => -0.0,
                1 => 0.0,
                2 => -1.25,
                3 => 2.5,
                _ => ((i % 17) as f64) * 0.125 - 0.75,
            })
            .collect();
        let dense = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, data.clone()).unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F64,
                Shape { dims },
                data.iter().copied().map(Literal::from_f64).collect(),
            )
            .unwrap(),
        );
        let params = BTreeMap::from([("axis".to_owned(), "0".to_owned())]);

        let got = crate::eval_primitive(Primitive::Cumsum, std::slice::from_ref(&dense), &params)
            .unwrap();
        let expect =
            crate::eval_primitive(Primitive::Cumsum, std::slice::from_ref(&boxed), &params)
                .unwrap();

        assert!(got.as_tensor().unwrap().elements.as_f64_slice().is_some());
        let got_bits: Vec<u64> = got
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_f64().unwrap().to_bits())
            .collect();
        let expect_bits: Vec<u64> = expect
            .as_tensor()
            .unwrap()
            .elements
            .iter()
            .map(|l| l.as_f64().unwrap().to_bits())
            .collect();
        assert_eq!(got_bits, expect_bits);
    }

    /// Dense f32 cumulative (cumsum/cumprod/cummax/cummin) must be BIT-FOR-BIT
    /// identical to the generic per-`Literal` float scan (forced via a boxed f32
    /// input whose `as_f32_slice` is None), across axes/forward+reverse, incl
    /// NaN/±inf/-0.0. The scan is a sequential dependency so it can't reassociate;
    /// the f64 accumulator + per-step `as f32` round match the generic exactly.
    #[test]
    fn dense_f32_cumulative_bit_identical_to_literal_path() {
        let (rows, cols) = (3usize, 16usize);
        let dims = vec![rows as u32, cols as u32];
        let f: Vec<f32> = (0..rows * cols)
            .map(|i| match i % 11 {
                0 => 0.0,
                1 => -1.5,
                2 => f32::INFINITY,
                3 => -0.0,
                4 => f32::NEG_INFINITY,
                5 => f32::from_bits(0x7fc0_0001),
                _ => ((i as f32) * 0.13).sin() * 2.0 - 0.4,
            })
            .collect();
        let dense = Value::Tensor(
            TensorValue::new_f32_values(Shape { dims: dims.clone() }, f.clone()).unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::F32,
                Shape { dims: dims.clone() },
                f.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );
        assert!(dense.as_tensor().unwrap().elements.as_f32_slice().is_some());
        assert!(boxed.as_tensor().unwrap().elements.as_f32_slice().is_none());
        // Sequential scan => exact bit match (no NaN canonicalization needed).
        let f32_bits = |v: &Value| -> Vec<u32> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::F32Bits(b) => *b,
                    o => panic!("expected f32, got {o:?}"),
                })
                .collect()
        };
        for prim in [
            Primitive::Cumsum,
            Primitive::Cumprod,
            Primitive::Cummax,
            Primitive::Cummin,
        ] {
            for (axis, rev) in [("0", "false"), ("1", "false"), ("1", "true"), ("0", "true")] {
                let mut p = BTreeMap::new();
                p.insert("axis".to_owned(), axis.to_owned());
                p.insert("reverse".to_owned(), rev.to_owned());
                let d = crate::eval_primitive(prim, std::slice::from_ref(&dense), &p).unwrap();
                let l = crate::eval_primitive(prim, std::slice::from_ref(&boxed), &p).unwrap();
                assert_eq!(d.as_tensor().unwrap().dtype, DType::F32, "{prim:?} dtype");
                assert_eq!(
                    f32_bits(&d),
                    f32_bits(&l),
                    "f32 {prim:?} axis={axis} rev={rev}"
                );
            }
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_f32_cumsum_dense_vs_generic() {
        use std::time::Instant;
        let (rows, cols) = (2048usize, 2048usize); // cumsum over last axis
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 251) as f32) * 0.013 - 1.6)
            .collect();
        let dims = vec![rows as u32, cols as u32];
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
        let p = BTreeMap::from([("axis".to_owned(), "1".to_owned())]);
        let time = |input: &Value| {
            let _ =
                crate::eval_primitive(Primitive::Cumsum, std::slice::from_ref(input), &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..20 {
                let t = Instant::now();
                let _ = crate::eval_primitive(Primitive::Cumsum, std::slice::from_ref(input), &p)
                    .unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let generic = time(&boxed);
        let dense_t = time(&dense);
        println!(
            "BENCH f32 cumsum axis1 [{rows},{cols}]: generic(per-Literal)={:.4}ms dense={:.4}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
    }

    /// Isomorphism proof for the parallel cumulative scan: a last-axis scan large
    /// enough to fan out across threads (rows*cols >= 1<<18, rows > 1) must match,
    /// element for element, an independent per-line serial reference. Lines are
    /// independent so the threaded result is identical to serial.
    #[test]
    fn threaded_cumulative_matches_serial_reference() {
        let rows = 4096usize;
        let cols = 128usize; // 4096*128 = 524_288 >= 1<<18 -> threaded
        assert!(rows * cols >= (1usize << 18) && rows > 1);
        let f: Vec<f64> = (0..rows * cols)
            .map(|i| match i % 257 {
                0 => -0.0,
                1 => f64::NAN,
                2 => f64::INFINITY,
                3 => f64::NEG_INFINITY,
                _ => ((i % 13) as f64) - 6.0 + 0.5 * ((i % 4) as f64),
            })
            .collect();
        let dense = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                f.clone(),
            )
            .unwrap(),
        );

        for (prim, op) in [
            (
                Primitive::Cumsum,
                (|a: f64, b: f64| a + b) as fn(f64, f64) -> f64,
            ),
            (Primitive::Cumprod, (|a, b| a * b) as fn(f64, f64) -> f64),
            // NaN-propagating max/min, matching jnp.cummax/cummin (XLA Max/Min).
            (Primitive::Cummax, crate::jax_max_f64 as fn(f64, f64) -> f64),
            (Primitive::Cummin, crate::jax_min_f64 as fn(f64, f64) -> f64),
        ] {
            let init = match prim {
                Primitive::Cumprod => 1.0,
                Primitive::Cummax => f64::NEG_INFINITY,
                Primitive::Cummin => f64::INFINITY,
                _ => 0.0,
            };
            for &rev in &[false, true] {
                let mut p = BTreeMap::new();
                p.insert("axis".to_owned(), "1".to_owned());
                p.insert("reverse".to_owned(), rev.to_string());
                let got = crate::eval_primitive(prim, std::slice::from_ref(&dense), &p).unwrap();
                let got: Vec<u64> = got
                    .as_tensor()
                    .unwrap()
                    .elements
                    .iter()
                    .map(|l| l.as_f64().unwrap().to_bits())
                    .collect();
                // Independent per-line serial reference.
                let mut expect: Vec<u64> = Vec::with_capacity(rows * cols);
                for line in f.chunks(cols) {
                    let mut acc = init;
                    let mut row: Vec<f64> = vec![0.0; cols];
                    if rev {
                        for i in (0..cols).rev() {
                            acc = op(acc, line[i]);
                            row[i] = acc;
                        }
                    } else {
                        for i in 0..cols {
                            acc = op(acc, line[i]);
                            row[i] = acc;
                        }
                    }
                    expect.extend(row.iter().map(|v| v.to_bits()));
                }
                assert_eq!(got, expect, "{prim:?} rev={rev} threaded != serial");
            }
        }
    }

    #[test]
    fn cumsum_vector() {
        // cumsum([1, 2, 3, 4]) = [1, 3, 6, 10]
        let result = eval_cumulative(
            Primitive::Cumsum,
            &[v_f64(&[1.0, 2.0, 3.0, 4.0])],
            &BTreeMap::new(),
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();
        let vals = extract_f64_vec(&result);
        assert_eq!(vals, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn cumsum_empty_huge_shape_returns_empty_before_stride_overflow() {
        let huge = u32::MAX;
        let dims = vec![0, huge, huge, huge, huge];
        let mut params = BTreeMap::new();
        params.insert("axis".to_owned(), "0".to_owned());

        let result = eval_cumulative(
            Primitive::Cumsum,
            &[empty_i64(dims.clone())],
            &params,
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .expect("empty cumulative output should not need huge strides");

        let tensor = result.as_tensor().expect("tensor output");
        assert_eq!(tensor.shape.dims, dims);
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn cumsum_reverse_vector() {
        let mut params = BTreeMap::new();
        params.insert("reverse".to_owned(), "true".to_owned());
        let result = eval_cumulative(
            Primitive::Cumsum,
            &[v_f64(&[1.0, 2.0, 3.0, 4.0])],
            &params,
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();
        let vals = extract_f64_vec(&result);
        assert_eq!(vals, vec![10.0, 9.0, 7.0, 4.0]);
    }

    #[test]
    fn cumprod_vector() {
        // cumprod([1, 2, 3, 4]) = [1, 2, 6, 24]
        let result = eval_cumulative(
            Primitive::Cumprod,
            &[v_f64(&[1.0, 2.0, 3.0, 4.0])],
            &BTreeMap::new(),
            1,
            1.0,
            |a, b| a * b,
            |a, b| a * b,
        )
        .unwrap();
        let vals = extract_f64_vec(&result);
        assert_eq!(vals, vec![1.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn cummax_vector() {
        // cummax([1, 3, 2, 4]) = [1, 3, 3, 4]
        let result = eval_cumulative(
            Primitive::Cummax,
            &[v_f64(&[1.0, 3.0, 2.0, 4.0])],
            &BTreeMap::new(),
            i64::MIN,
            f64::NEG_INFINITY,
            |a, b| a.max(b),
            |a, b| a.max(b),
        )
        .unwrap();
        let vals = extract_f64_vec(&result);
        assert_eq!(vals, vec![1.0, 3.0, 3.0, 4.0]);
    }

    #[test]
    fn cummin_vector() {
        // cummin([4, 2, 3, 1]) = [4, 2, 2, 1]
        let result = eval_cumulative(
            Primitive::Cummin,
            &[v_f64(&[4.0, 2.0, 3.0, 1.0])],
            &BTreeMap::new(),
            i64::MAX,
            f64::INFINITY,
            |a, b| a.min(b),
            |a, b| a.min(b),
        )
        .unwrap();
        let vals = extract_f64_vec(&result);
        assert_eq!(vals, vec![4.0, 2.0, 2.0, 1.0]);
    }

    #[test]
    fn cumsum_empty_selected_axis_returns_empty_tensor() {
        let mut params = BTreeMap::new();
        params.insert("axis".to_owned(), "-1".to_owned());
        let result = eval_cumulative(
            Primitive::Cumsum,
            &[empty_i64(vec![0])],
            &params,
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();

        let tensor = result.as_tensor().unwrap();
        assert_eq!(tensor.dtype, DType::I64);
        assert_eq!(tensor.shape.dims, vec![0]);
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn cumprod_empty_selected_axis_returns_empty_matrix() {
        let mut params = BTreeMap::new();
        params.insert("axis".to_owned(), "1".to_owned());
        let result = eval_cumulative(
            Primitive::Cumprod,
            &[empty_i64(vec![2, 0])],
            &params,
            1,
            1.0,
            |a, b| a * b,
            |a, b| a * b,
        )
        .unwrap();

        let tensor = result.as_tensor().unwrap();
        assert_eq!(tensor.dtype, DType::I64);
        assert_eq!(tensor.shape.dims, vec![2, 0]);
        assert!(tensor.elements.is_empty());
    }

    #[test]
    fn cumsum_scalar() {
        let result = eval_cumulative(
            Primitive::Cumsum,
            &[s_f64(7.0)],
            &BTreeMap::new(),
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();
        assert!((extract_f64(&result) - 7.0).abs() < 1e-12);
    }
}
