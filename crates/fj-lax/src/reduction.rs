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
#[inline]
fn dense_f64_axis_reduce(
    tensor: &TensorValue,
    kept_axes: &[usize],
    out_dims: &[u32],
    out_count: usize,
    float_init: f64,
    float_op: &impl Fn(f64, f64) -> f64,
) -> Option<Vec<f64>> {
    let values = tensor.elements.as_f64_slice()?;
    if tensor.shape.dims.is_empty() {
        return None;
    }
    let mut odometer = OutIndexOdometer::new(&tensor.shape.dims, kept_axes, out_dims);
    let mut result = vec![float_init; out_count];
    for &value in values {
        let out_idx = odometer.next_index();
        result[out_idx] = float_op(result[out_idx], value);
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
    float_op: impl Fn(f64, f64) -> f64,
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
                let result = if let Some(values) = dense_f64_axis_reduce(
                    tensor, &kept_axes, &out_dims, out_count, float_init, &float_op,
                ) {
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
pub(crate) fn eval_cumulative(
    primitive: Primitive,
    inputs: &[Value],
    params: &std::collections::BTreeMap<String, String>,
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
