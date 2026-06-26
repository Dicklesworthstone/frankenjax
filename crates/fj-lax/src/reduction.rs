#![forbid(unsafe_code)]

use fj_core::{DType, Literal, Primitive, Shape, TensorValue, Value, ValueError};

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

fn bool_word_bit(words: &[u64], index: usize) -> bool {
    let word = words[index / u64::BITS as usize];
    ((word >> (index % u64::BITS as usize)) & 1) != 0
}

fn reduce_bool_words_scalar(
    primitive: Primitive,
    words: &[u64],
    len: usize,
    bool_init: bool,
) -> Option<bool> {
    if len == 0 {
        return Some(bool_init);
    }

    match primitive {
        Primitive::ReduceAnd => {
            let full_words = len / u64::BITS as usize;
            let tail_bits = len % u64::BITS as usize;
            let full_ok = words[..full_words].iter().all(|&word| word == u64::MAX);
            let tail_ok = if tail_bits == 0 {
                true
            } else {
                let tail_mask = (1_u64 << tail_bits) - 1;
                words[full_words] == tail_mask
            };
            Some(full_ok && tail_ok)
        }
        Primitive::ReduceOr => Some(words.iter().any(|&word| word != 0)),
        Primitive::ReduceXor => Some(
            words
                .iter()
                .fold(false, |acc, word| acc ^ (word.count_ones() % 2 == 1)),
        ),
        _ => None,
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
/// SIMD full-reduce max/min over a dense `f64` slice, bit-identical to the scalar
/// `jax_max_f64`/`jax_min_f64` fold but vectorized. The scalar fold can't autovectorize
/// because the per-step NaN branch breaks LLVM's reduction recognition.
///
/// `jax_max`/`jax_min` are associative AND commutative on the VALUE (any NaN ⇒ canonical
/// NaN; else `f64::max`/`min`), so a lane-parallel reduction equals the sequential fold
/// on the value. Two specials need care:
///   • NaN — `simd_max`/`simd_min` IGNORE NaN, so any-NaN is tracked separately and the
///     result is overridden to canonical `f64::NAN` (matching the scalar fold, which
///     collapses every NaN payload to `f64::NAN`).
///   • signed zero — `f64::max(+0,-0)` returns the SECOND operand (x86 `maxsd`/LLVM
///     `maxnum`), so the scalar fold's ±0 SIGN is order-dependent (last-zero-wins) and a
///     lane-parallel reduce can't reproduce it. So when the SIMD value is exactly ±0
///     (the only ambiguous case, and rare — it means every element is ≤0 for max / ≥0
///     for min) fall back to the exact scalar fold, which is bit-identical by definition.
fn simd_reduce_minmax_f64(values: &[f64], is_max: bool) -> f64 {
    use std::simd::{Simd, num::SimdFloat};
    const LANES: usize = 8;
    let init = if is_max {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };

    let mut vacc = Simd::<f64, LANES>::splat(init);
    // SIMD-mask NaN accumulate (one `.any()` at the end) — see simd_reduce_minmax_f32. Bit-identical.
    let mut nan_acc = vacc.is_nan();
    let chunks = values.chunks_exact(LANES);
    let tail = chunks.remainder();
    for chunk in chunks {
        let v = Simd::<f64, LANES>::from_slice(chunk);
        nan_acc |= v.is_nan();
        vacc = if is_max {
            vacc.simd_max(v)
        } else {
            vacc.simd_min(v)
        };
    }
    // Horizontal combine with scalar f64::max/min (vacc holds no NaN: simd_max/min
    // ignore them).
    let mut any_nan = nan_acc.any();
    let mut m = init;
    for &lane in vacc.to_array().iter() {
        m = if is_max { m.max(lane) } else { m.min(lane) };
    }
    for &v in tail {
        if v.is_nan() {
            any_nan = true;
        } else {
            m = if is_max { m.max(v) } else { m.min(v) };
        }
    }

    if any_nan {
        return f64::NAN;
    }
    if m == 0.0 {
        // ±0 sign is order-dependent under f64::max/min; defer to the exact fold (no
        // NaN here, so the fold is just f64::max/min over the slice).
        let mut acc = init;
        for &v in values {
            acc = if is_max { acc.max(v) } else { acc.min(v) };
        }
        return acc;
    }
    m
}

/// f32 sibling of [`simd_reduce_minmax_f64`]. The scalar f32 fold widens each f32 to
/// f64, applies `jax_max`/`jax_min`, and rounds the result back to f32 — but for
/// max/min the result is always one of the inputs, so widen→op→round equals the f32
/// max/min directly. Same NaN / ±0 handling on f32 patterns.
fn simd_reduce_minmax_f32(values: &[f32], is_max: bool) -> f32 {
    use std::simd::{Simd, num::SimdFloat};
    const LANES: usize = 16;
    let init = if is_max {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };

    let mut vacc = Simd::<f32, LANES>::splat(init);
    // Accumulate NaN detection as a SIMD MASK (one `.any()` at the end) instead of a horizontal
    // `.any()` per chunk — the per-chunk reduction was ~half the per-row cost. Bit-identical.
    let mut nan_acc = vacc.is_nan(); // all-false (init is finite)
    let chunks = values.chunks_exact(LANES);
    let tail = chunks.remainder();
    for chunk in chunks {
        let v = Simd::<f32, LANES>::from_slice(chunk);
        nan_acc |= v.is_nan();
        vacc = if is_max {
            vacc.simd_max(v)
        } else {
            vacc.simd_min(v)
        };
    }
    let mut any_nan = nan_acc.any();
    let mut m = init;
    for &lane in vacc.to_array().iter() {
        m = if is_max { m.max(lane) } else { m.min(lane) };
    }
    for &v in tail {
        if v.is_nan() {
            any_nan = true;
        } else {
            m = if is_max { m.max(v) } else { m.min(v) };
        }
    }

    if any_nan {
        return f32::NAN;
    }
    if m == 0.0 {
        // Match the scalar f32 fold exactly: accumulate in f64, round to f32.
        let mut acc = if is_max {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        for &v in values {
            let vv = f64::from(v);
            acc = if is_max { acc.max(vv) } else { acc.min(vv) };
        }
        return acc as f32;
    }
    m
}

/// Threaded full f64 max/min reduce: split the input into chunks, SIMD-reduce each on a
/// scoped thread, then combine the partials. Bit-for-bit identical to
/// [`simd_reduce_minmax_f64`] over the whole slice — max/min are associative+commutative,
/// each partial already resolves ±0 sign and NaN exactly (per-chunk), and the combine
/// preserves both (NaN if any partial is NaN -> canonical `f64::NAN`; `f64::max`/`min`'s
/// maxNum/minNum ±0 handling is order-independent so fold-of-folds == full fold). Only
/// engages once the input is DRAM-bound (see [`CHEAP_BINARY_PARALLEL_MIN`]); the serial
/// SIMD reduce wins while it fits in cache.
fn threaded_reduce_minmax_f64(values: &[f64], is_max: bool) -> f64 {
    let n = values.len();
    if n < crate::arithmetic::CHEAP_BINARY_PARALLEL_MIN {
        return simd_reduce_minmax_f64(values, is_max);
    }
    let threads = crate::arithmetic::work_scaled_threads(n);
    if threads <= 1 {
        return simd_reduce_minmax_f64(values, is_max);
    }
    let chunk = n.div_ceil(threads);
    let partials: Vec<f64> = std::thread::scope(|scope| {
        let handles: Vec<_> = values
            .chunks(chunk)
            .map(|c| scope.spawn(move || simd_reduce_minmax_f64(c, is_max)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    if partials.iter().any(|p| p.is_nan()) {
        return f64::NAN;
    }
    let mut m = if is_max {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    for &p in &partials {
        m = if is_max { m.max(p) } else { m.min(p) };
    }
    m
}

fn threaded_tree_reduce_sum_f64(values: &[f64]) -> f64 {
    let n = values.len();
    let threads = crate::arithmetic::work_scaled_threads(n).min(8);
    if n < crate::arithmetic::CHEAP_BINARY_PARALLEL_MIN || threads <= 1 {
        let mut acc = 0.0;
        for &value in values {
            acc += value;
        }
        return acc;
    }

    let chunk = n.div_ceil(threads);
    let partials: Vec<f64> = std::thread::scope(|scope| {
        let handles: Vec<_> = values
            .chunks(chunk)
            .map(|chunk_values| {
                scope.spawn(move || {
                    let mut acc = 0.0;
                    for &value in chunk_values {
                        acc += value;
                    }
                    acc
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| match handle.join() {
                Ok(partial) => partial,
                Err(payload) => std::panic::resume_unwind(payload),
            })
            .collect()
    });

    let mut acc = 0.0;
    for partial in partials {
        acc += partial;
    }
    acc
}

/// f32 sibling of [`threaded_reduce_minmax_f64`].
fn threaded_reduce_minmax_f32(values: &[f32], is_max: bool) -> f32 {
    let n = values.len();
    if n < crate::arithmetic::CHEAP_BINARY_PARALLEL_MIN {
        return simd_reduce_minmax_f32(values, is_max);
    }
    let threads = crate::arithmetic::work_scaled_threads(n);
    if threads <= 1 {
        return simd_reduce_minmax_f32(values, is_max);
    }
    let chunk = n.div_ceil(threads);
    let partials: Vec<f32> = std::thread::scope(|scope| {
        let handles: Vec<_> = values
            .chunks(chunk)
            .map(|c| scope.spawn(move || simd_reduce_minmax_f32(c, is_max)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    if partials.iter().any(|p| p.is_nan()) {
        return f32::NAN;
    }
    let mut m = if is_max {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    for &p in &partials {
        m = if is_max { m.max(p) } else { m.min(p) };
    }
    m
}

/// SIMD axis-reduce max/min for the trailing-axes (`inner == 1`) contiguous-block
/// layout — i.e. `jnp.max(x, axis=-1)` / softmax/attention stability, the dominant
/// case. Each of the `outer` output cells reduces one contiguous run of `reduce`
/// elements, so apply [`simd_reduce_minmax_f64`] (bit-identical to the scalar
/// `jax_max`/`jax_min` fold) per cell. Rows are independent, so large reductions
/// fan out across threads (same gate/sizing as `dense_f64_axis_reduce`). The Vec<f64>
/// result feeds the caller's existing `reduce_real_literal` wrap unchanged.
fn simd_minmax_axis_reduce_f64(
    values: &[f64],
    is_max: bool,
    outer: usize,
    reduce: usize,
) -> Vec<f64> {
    let mut result = vec![0.0f64; outer];
    let threads = if values.len() >= (1 << 18) {
        crate::arithmetic::work_scaled_threads(values.len()).min(outer)
    } else {
        1
    };
    if threads > 1 {
        let rows_per = outer.div_ceil(threads);
        std::thread::scope(|scope| {
            let mut res_rest: &mut [f64] = result.as_mut_slice();
            let mut row0 = 0usize;
            while row0 < outer {
                let rows = rows_per.min(outer - row0);
                let (blk, tail) = res_rest.split_at_mut(rows);
                res_rest = tail;
                let vblk = &values[row0 * reduce..(row0 + rows) * reduce];
                row0 += rows;
                scope.spawn(move || {
                    for (r, slot) in blk.iter_mut().enumerate() {
                        *slot =
                            simd_reduce_minmax_f64(&vblk[r * reduce..r * reduce + reduce], is_max);
                    }
                });
            }
        });
    } else {
        for (o, slot) in result.iter_mut().enumerate() {
            *slot = simd_reduce_minmax_f64(&values[o * reduce..o * reduce + reduce], is_max);
        }
    }
    result
}

/// f32 sibling — each cell's per-`reduce` max/min via [`simd_reduce_minmax_f32`],
/// stored as f64 (exact: the result is an input f32 value). The caller rounds back
/// to f32 via `reduce_real_literal(F32, …)`, which round-trips exactly.
fn simd_minmax_axis_reduce_f32(
    values: &[f32],
    is_max: bool,
    outer: usize,
    reduce: usize,
) -> Vec<f64> {
    let mut result = vec![0.0f64; outer];
    let threads = if values.len() >= (1 << 18) {
        crate::arithmetic::work_scaled_threads(values.len()).min(outer)
    } else {
        1
    };
    if threads > 1 {
        let rows_per = outer.div_ceil(threads);
        std::thread::scope(|scope| {
            let mut res_rest: &mut [f64] = result.as_mut_slice();
            let mut row0 = 0usize;
            while row0 < outer {
                let rows = rows_per.min(outer - row0);
                let (blk, tail) = res_rest.split_at_mut(rows);
                res_rest = tail;
                let vblk = &values[row0 * reduce..(row0 + rows) * reduce];
                row0 += rows;
                scope.spawn(move || {
                    for (r, slot) in blk.iter_mut().enumerate() {
                        *slot = f64::from(simd_reduce_minmax_f32(
                            &vblk[r * reduce..r * reduce + reduce],
                            is_max,
                        ));
                    }
                });
            }
        });
    } else {
        for (o, slot) in result.iter_mut().enumerate() {
            *slot = f64::from(simd_reduce_minmax_f32(
                &values[o * reduce..o * reduce + reduce],
                is_max,
            ));
        }
    }
    result
}

/// BF16 sibling — each cell's per-`reduce` max/min via [`simd_reduce_minmax_bf16`]
/// (widen u16→f32, simd_max). `bf16 max(x, axis=-1)` is a real training path
/// (attention-score max for stability). Returns Vec<f64>; the caller rounds back to
/// BF16 via `reduce_real_literal(BF16, …)`, which round-trips an exact bf16 value.
fn simd_minmax_axis_reduce_bf16(
    values: &[u16],
    is_max: bool,
    outer: usize,
    reduce: usize,
) -> Vec<f64> {
    let mut result = vec![0.0f64; outer];
    let threads = if values.len() >= (1 << 18) {
        crate::arithmetic::work_scaled_threads(values.len()).min(outer)
    } else {
        1
    };
    if threads > 1 {
        let rows_per = outer.div_ceil(threads);
        std::thread::scope(|scope| {
            let mut res_rest: &mut [f64] = result.as_mut_slice();
            let mut row0 = 0usize;
            while row0 < outer {
                let rows = rows_per.min(outer - row0);
                let (blk, tail) = res_rest.split_at_mut(rows);
                res_rest = tail;
                let vblk = &values[row0 * reduce..(row0 + rows) * reduce];
                row0 += rows;
                scope.spawn(move || {
                    for (r, slot) in blk.iter_mut().enumerate() {
                        *slot =
                            simd_reduce_minmax_bf16(&vblk[r * reduce..r * reduce + reduce], is_max);
                    }
                });
            }
        });
    } else {
        for (o, slot) in result.iter_mut().enumerate() {
            *slot = simd_reduce_minmax_bf16(&values[o * reduce..o * reduce + reduce], is_max);
        }
    }
    result
}

/// SIMD full-reduce max/min over a dense BF16 slice (the dominant TRAINING dtype),
/// bit-identical to the scalar `jax_max`/`jax_min` fold over `BF16Bits.as_f64()`.
/// bf16→f32 is the exact top-16-bits widen (`f32::from_bits((b as u32) << 16)`), so
/// the f32 max/min equals the f64 fold's value and `reduce_real_literal(BF16, …)`
/// rounds it back exactly. Same NaN / ±0 handling as [`simd_reduce_minmax_f64`]
/// (any-NaN → canonical NaN; ±0 → scalar-fold fallback for the order-dependent sign).
fn simd_reduce_minmax_bf16(values: &[u16], is_max: bool) -> f64 {
    use std::simd::{
        Simd,
        num::{SimdFloat, SimdUint},
    };
    const L: usize = 16;
    let init = if is_max {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    let widen = |b: u16| f32::from_bits((b as u32) << 16);

    let mut vacc = Simd::<f32, L>::splat(init);
    let mut any_nan = false;
    let chunks = values.chunks_exact(L);
    let tail = chunks.remainder();
    for chunk in chunks {
        let u = Simd::<u16, L>::from_slice(chunk);
        let f = Simd::<f32, L>::from_bits(u.cast::<u32>() << Simd::splat(16u32));
        any_nan |= f.is_nan().any();
        vacc = if is_max {
            vacc.simd_max(f)
        } else {
            vacc.simd_min(f)
        };
    }
    let mut m = init;
    for &lane in vacc.to_array().iter() {
        m = if is_max { m.max(lane) } else { m.min(lane) };
    }
    for &b in tail {
        let v = widen(b);
        if v.is_nan() {
            any_nan = true;
        } else {
            m = if is_max { m.max(v) } else { m.min(v) };
        }
    }
    if any_nan {
        return f64::NAN;
    }
    if m == 0.0 {
        let mut acc = init;
        for &b in values {
            let v = widen(b);
            acc = if is_max { acc.max(v) } else { acc.min(v) };
        }
        return f64::from(acc);
    }
    f64::from(m)
}

/// F16 sibling of [`simd_minmax_axis_reduce_bf16`]. F16 `max(x, axis=-1)` is a real
/// inference path. The current F16 path ([`fold_f16_axis_block`]) SIMD-DECODES but folds
/// with a per-element scalar `jax_max`/`jax_min` (two NaN branches each); this folds the
/// decoded lanes with a single `simd_max`/`simd_min`, bit-identical.
fn simd_minmax_axis_reduce_f16(
    values: &[u16],
    is_max: bool,
    outer: usize,
    reduce: usize,
) -> Vec<f64> {
    let mut result = vec![0.0f64; outer];
    let threads = if values.len() >= (1 << 18) {
        crate::arithmetic::work_scaled_threads(values.len()).min(outer)
    } else {
        1
    };
    if threads > 1 {
        let rows_per = outer.div_ceil(threads);
        std::thread::scope(|scope| {
            let mut res_rest: &mut [f64] = result.as_mut_slice();
            let mut row0 = 0usize;
            while row0 < outer {
                let rows = rows_per.min(outer - row0);
                let (blk, tail) = res_rest.split_at_mut(rows);
                res_rest = tail;
                let vblk = &values[row0 * reduce..(row0 + rows) * reduce];
                row0 += rows;
                scope.spawn(move || {
                    for (r, slot) in blk.iter_mut().enumerate() {
                        *slot =
                            simd_reduce_minmax_f16(&vblk[r * reduce..r * reduce + reduce], is_max);
                    }
                });
            }
        });
    } else {
        for (o, slot) in result.iter_mut().enumerate() {
            *slot = simd_reduce_minmax_f16(&values[o * reduce..o * reduce + reduce], is_max);
        }
    }
    result
}

/// SIMD full-reduce max/min over a dense F16 slice. Bit-identical to the scalar
/// [`fold_f16_axis_block`] fold (jax_max/jax_min over `F16Bits.as_f64()`): 8-lane chunks
/// decode through [`crate::arithmetic::f16_widen8_full_f32`] without a per-chunk scalar
/// split; NaN is tracked separately because `simd_max`/`simd_min` drop NaN. A `0.0`
/// result still triggers a scalar re-scan for order-dependent +/-0 signs.
fn simd_reduce_minmax_f16(values: &[u16], is_max: bool) -> f64 {
    use std::simd::{Mask, Simd, num::SimdFloat};
    const L: usize = 8; // matches f16_widen8_full_f32 lane count
    // FULLY BRANCHLESS f32x8: f16_widen8_full_f32 decodes every f16 (normal/subnormal/±0/inf/NaN)
    // with no per-chunk fast/slow split. simd_max/min drops NaN, so NaN is tracked in a SIMD mask
    // (one `.any()` at the end). f16 max/min is exact in f32; bit-identical to the scalar fold.
    let init = if is_max {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    let decode_f16 = |b: u16| Literal::F16Bits(b).as_f64().unwrap_or(0.0) as f32;

    let mut vacc = Simd::<f32, L>::splat(init);
    let mut nan_acc = Mask::<i32, L>::splat(false);
    let chunks = values.chunks_exact(L);
    let tail = chunks.remainder();
    for chunk in chunks {
        let f = crate::arithmetic::f16_widen8_full_f32(Simd::<u16, L>::from_slice(chunk));
        nan_acc |= f.is_nan();
        vacc = if is_max {
            vacc.simd_max(f)
        } else {
            vacc.simd_min(f)
        };
    }
    let mut m = init;
    for &lane in vacc.to_array().iter() {
        m = if is_max { m.max(lane) } else { m.min(lane) };
    }
    let mut any_nan = nan_acc.any();
    for &b in tail {
        let v = decode_f16(b);
        if v.is_nan() {
            any_nan = true;
        } else {
            m = if is_max { m.max(v) } else { m.min(v) };
        }
    }

    if any_nan {
        return f64::NAN;
    }
    if m == 0.0 {
        // ±0 sign is order-dependent under simd reduce — re-fold scalar (no NaN here).
        let mut s = init;
        for &b in values {
            let v = decode_f16(b);
            s = if is_max { s.max(v) } else { s.min(v) };
        }
        return f64::from(s);
    }
    f64::from(m)
}

/// One row-wise max/min accumulate step `out[i] = jax_max/min(out[i], inp[i])`, SIMD
/// over a CONTIGUOUS row with per-lane NaN propagation (mask-select). No ±0 ambiguity
/// here (each cell accumulates deterministically — no horizontal reduce), so it is
/// bit-identical to the scalar `jax_max`/`jax_min` step. Used for the inner>1
/// (leading/middle-axis) reduction where the trailing-axis SIMD path does not apply.
#[inline]
fn simd_minmax_row_acc_f64(out: &mut [f64], inp: &[f64], is_max: bool) {
    use std::simd::{Select, Simd, num::SimdFloat};
    const L: usize = 4;
    let mut oc = out.chunks_exact_mut(L);
    let mut ic = inp.chunks_exact(L);
    for (o, i) in oc.by_ref().zip(ic.by_ref()) {
        let a = Simd::<f64, L>::from_slice(o);
        let b = Simd::<f64, L>::from_slice(i);
        let m = if is_max { a.simd_max(b) } else { a.simd_min(b) };
        let nan = a.is_nan() | b.is_nan();
        o.copy_from_slice(&nan.select(Simd::splat(f64::NAN), m).to_array());
    }
    for (o, &v) in oc.into_remainder().iter_mut().zip(ic.remainder()) {
        *o = jax_minmax_scalar(*o, v, is_max);
    }
}

/// f32-NATIVE row-accumulate (f32x8, no f64 widen): `out[c] = minmax(out[c], inp[c])` over the
/// contiguous inner row, NaN-propagating to canonical `f32::NAN`. Max/min of f32 is exact, so the
/// resulting VALUE is identical to a f64-widened accumulate (and the f32 output's NaN is canonical
/// regardless). 8-wide + no widen replaces the prior 4-wide f64-widening accumulate.
#[inline]
fn simd_minmax_row_acc_f32_native(out: &mut [f32], inp: &[f32], is_max: bool) {
    use std::simd::{Select, Simd, num::SimdFloat};
    const L: usize = 8;
    let mut oc = out.chunks_exact_mut(L);
    let mut ic = inp.chunks_exact(L);
    for (o, i) in oc.by_ref().zip(ic.by_ref()) {
        let a = Simd::<f32, L>::from_slice(o);
        let b = Simd::<f32, L>::from_slice(i);
        let m = if is_max { a.simd_max(b) } else { a.simd_min(b) };
        let nan = a.is_nan() | b.is_nan();
        o.copy_from_slice(&nan.select(Simd::splat(f32::NAN), m).to_array());
    }
    for (o, &v) in oc.into_remainder().iter_mut().zip(ic.remainder()) {
        let cur = *o;
        *o = if cur.is_nan() || v.is_nan() {
            f32::NAN
        } else if is_max {
            cur.max(v)
        } else {
            cur.min(v)
        };
    }
}

/// Accumulate the f32-native running min/max for output cells `[0, out.len())` over `reduce` rows
/// of stride `inner`, starting at `values[base..]`, where `out` is the (contiguous) output column
/// block at column offset `col0` within each row. Used by both the serial and threaded paths.
#[inline]
fn simd_minmax_f32_accumulate_block(
    out: &mut [f32],
    values: &[f32],
    base: usize,
    reduce: usize,
    inner: usize,
    col0: usize,
    is_max: bool,
) {
    let w = out.len();
    for r in 0..reduce {
        let start = base + r * inner + col0;
        simd_minmax_row_acc_f32_native(out, &values[start..start + w], is_max);
    }
}

#[inline]
fn jax_minmax_scalar(a: f64, b: f64, is_max: bool) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if is_max {
        a.max(b)
    } else {
        a.min(b)
    }
}

/// SIMD inner>1 (leading/middle-axis) max/min: for each `outer` cell, fold the
/// `reduce` contiguous inner rows into the output row with [`simd_minmax_row_acc_f64`]
/// in ascending-r order — bit-identical to the scalar block-fold (`jnp.max(x, axis=0)`
/// / a middle axis). Returns the f64 accumulators (caller rounds to the dtype).
fn simd_minmax_inner_axis_reduce_f64(
    values: &[f64],
    is_max: bool,
    outer: usize,
    reduce: usize,
    inner: usize,
) -> Vec<f64> {
    let init = if is_max {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    let mut result = vec![init; outer * inner];
    for o in 0..outer {
        let out_row = &mut result[o * inner..(o + 1) * inner];
        let base = o * reduce * inner;
        for r in 0..reduce {
            simd_minmax_row_acc_f64(
                out_row,
                &values[base + r * inner..base + r * inner + inner],
                is_max,
            );
        }
    }
    result
}

fn simd_minmax_inner_axis_reduce_f32(
    values: &[f32],
    is_max: bool,
    outer: usize,
    reduce: usize,
    inner: usize,
) -> Vec<f64> {
    let init = if is_max {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    // f32-NATIVE accumulator (no f64 widen — max/min of f32 is exact); widen to f64 once at the end.
    let mut acc = vec![init; outer * inner];
    let total = values.len();
    let want_threads = if total >= crate::arithmetic::CHEAP_BINARY_PARALLEL_MIN {
        crate::arithmetic::work_scaled_threads(total).min(16)
    } else {
        1
    };
    if want_threads <= 1 {
        for o in 0..outer {
            let out_row = &mut acc[o * inner..(o + 1) * inner];
            simd_minmax_f32_accumulate_block(
                out_row,
                values,
                o * reduce * inner,
                reduce,
                inner,
                0,
                is_max,
            );
        }
    } else if outer >= 2 {
        // Thread over the independent OUTER groups (each writes a contiguous `inner` block).
        let threads = want_threads.min(outer);
        let per = outer.div_ceil(threads);
        std::thread::scope(|scope| {
            let mut rest: &mut [f32] = acc.as_mut_slice();
            let mut o0 = 0usize;
            while o0 < outer {
                let cnt = per.min(outer - o0);
                let (blk, tail) = rest.split_at_mut(cnt * inner);
                rest = tail;
                let base0 = o0;
                o0 += cnt;
                scope.spawn(move || {
                    for oi in 0..cnt {
                        let o = base0 + oi;
                        let out_row = &mut blk[oi * inner..(oi + 1) * inner];
                        simd_minmax_f32_accumulate_block(
                            out_row,
                            values,
                            o * reduce * inner,
                            reduce,
                            inner,
                            0,
                            is_max,
                        );
                    }
                });
            }
        });
    } else {
        // outer == 1 (e.g. leading-axis max over [rows, cols]): thread over independent INNER
        // column blocks (each a contiguous output sub-slice; reads contiguous sub-rows).
        let threads = want_threads.min(inner.max(1));
        let per = inner.div_ceil(threads);
        std::thread::scope(|scope| {
            let mut rest: &mut [f32] = acc.as_mut_slice();
            let mut c0 = 0usize;
            while c0 < inner {
                let w = per.min(inner - c0);
                let (blk, tail) = rest.split_at_mut(w);
                rest = tail;
                let col0 = c0;
                c0 += w;
                scope.spawn(move || {
                    simd_minmax_f32_accumulate_block(blk, values, 0, reduce, inner, col0, is_max);
                });
            }
        });
    }
    acc.iter().map(|&v| f64::from(v)).collect()
}

/// BF16 row-accumulate into an f32 running min/max: widen bf16→f32 via the exact top-16-bit shift
/// (`(u16 as u32) << 16`), f32x8 max/min, NaN→canonical `f32::NAN`. bf16 max/min is EXACT in f32
/// (the result is one of the bf16 inputs), so this is bit-identical to a f64-widening accumulate for
/// the resulting value; f32x8 + a cheap shift replaces the prior f64x4 + f32→f64 cast (2× width, no
/// f64 widen).
#[inline]
fn simd_minmax_row_acc_bf16_f32(out: &mut [f32], inp: &[u16], is_max: bool) {
    use std::simd::{Select, Simd, num::SimdFloat, num::SimdUint};
    const L: usize = 8;
    let mut oc = out.chunks_exact_mut(L);
    let mut ic = inp.chunks_exact(L);
    for (o, i) in oc.by_ref().zip(ic.by_ref()) {
        let u = Simd::<u16, L>::from_slice(i);
        let b = Simd::<f32, L>::from_bits(u.cast::<u32>() << Simd::splat(16u32));
        let a = Simd::<f32, L>::from_slice(o);
        let m = if is_max { a.simd_max(b) } else { a.simd_min(b) };
        let nan = a.is_nan() | b.is_nan();
        o.copy_from_slice(&nan.select(Simd::splat(f32::NAN), m).to_array());
    }
    for (o, &v) in oc.into_remainder().iter_mut().zip(ic.remainder()) {
        let bv = f32::from_bits((v as u32) << 16);
        let cur = *o;
        *o = if cur.is_nan() || bv.is_nan() {
            f32::NAN
        } else if is_max {
            cur.max(bv)
        } else {
            cur.min(bv)
        };
    }
}

#[inline]
fn simd_minmax_bf16_accumulate_block(
    out: &mut [f32],
    values: &[u16],
    base: usize,
    reduce: usize,
    inner: usize,
    col0: usize,
    is_max: bool,
) {
    let w = out.len();
    for r in 0..reduce {
        let start = base + r * inner + col0;
        simd_minmax_row_acc_bf16_f32(out, &values[start..start + w], is_max);
    }
}

fn simd_minmax_inner_axis_reduce_bf16(
    values: &[u16],
    is_max: bool,
    outer: usize,
    reduce: usize,
    inner: usize,
) -> Vec<f64> {
    let init = if is_max {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    // f32 accumulator (no f64 widen) + threaded — mirrors simd_minmax_inner_axis_reduce_f32.
    let mut acc = vec![init; outer * inner];
    let total = values.len();
    let want_threads = if total >= crate::arithmetic::CHEAP_BINARY_PARALLEL_MIN {
        crate::arithmetic::work_scaled_threads(total).min(16)
    } else {
        1
    };
    if want_threads <= 1 {
        for o in 0..outer {
            let out_row = &mut acc[o * inner..(o + 1) * inner];
            simd_minmax_bf16_accumulate_block(
                out_row,
                values,
                o * reduce * inner,
                reduce,
                inner,
                0,
                is_max,
            );
        }
    } else if outer >= 2 {
        let threads = want_threads.min(outer);
        let per = outer.div_ceil(threads);
        std::thread::scope(|scope| {
            let mut rest: &mut [f32] = acc.as_mut_slice();
            let mut o0 = 0usize;
            while o0 < outer {
                let cnt = per.min(outer - o0);
                let (blk, tail) = rest.split_at_mut(cnt * inner);
                rest = tail;
                let base0 = o0;
                o0 += cnt;
                scope.spawn(move || {
                    for oi in 0..cnt {
                        let o = base0 + oi;
                        let out_row = &mut blk[oi * inner..(oi + 1) * inner];
                        simd_minmax_bf16_accumulate_block(
                            out_row,
                            values,
                            o * reduce * inner,
                            reduce,
                            inner,
                            0,
                            is_max,
                        );
                    }
                });
            }
        });
    } else {
        let threads = want_threads.min(inner.max(1));
        let per = inner.div_ceil(threads);
        std::thread::scope(|scope| {
            let mut rest: &mut [f32] = acc.as_mut_slice();
            let mut c0 = 0usize;
            while c0 < inner {
                let w = per.min(inner - c0);
                let (blk, tail) = rest.split_at_mut(w);
                rest = tail;
                let col0 = c0;
                c0 += w;
                scope.spawn(move || {
                    simd_minmax_bf16_accumulate_block(blk, values, 0, reduce, inner, col0, is_max);
                });
            }
        });
    }
    acc.iter().map(|&v| f64::from(v)).collect()
}

/// F16 row-accumulate into an f32 running min/max — FULLY BRANCHLESS via `f16_widen8_full_f32`
/// (decodes normal/subnormal/±0/inf/NaN with no per-chunk fast/slow split), then the same
/// NaN-propagating `select`. f16 max/min is exact in f32. Bit-identical to the prior
/// `f16_input_needs_scalar`-gated path (verified: the branchless decode matches the scalar decode
/// for all 65536 patterns, and the `nan` select propagates NaN that `simd_max` would otherwise drop).
#[inline]
fn simd_minmax_row_acc_f16_f32(out: &mut [f32], inp: &[u16], is_max: bool) {
    use std::simd::{Select, Simd, num::SimdFloat};
    const L: usize = 8; // matches f16_widen8_full_f32 lane count
    let mut oc = out.chunks_exact_mut(L);
    let mut ic = inp.chunks_exact(L);
    for (o, i) in oc.by_ref().zip(ic.by_ref()) {
        let b = crate::arithmetic::f16_widen8_full_f32(Simd::<u16, L>::from_slice(i));
        let a = Simd::<f32, L>::from_slice(o);
        let m = if is_max { a.simd_max(b) } else { a.simd_min(b) };
        let nan = a.is_nan() | b.is_nan();
        o.copy_from_slice(&nan.select(Simd::splat(f32::NAN), m).to_array());
    }
    for (o, &v) in oc.into_remainder().iter_mut().zip(ic.remainder()) {
        let bv = Literal::F16Bits(v).as_f64().unwrap_or(0.0) as f32;
        *o = if o.is_nan() || bv.is_nan() {
            f32::NAN
        } else if is_max {
            o.max(bv)
        } else {
            o.min(bv)
        };
    }
}

#[inline]
fn simd_minmax_f16_accumulate_block(
    out: &mut [f32],
    values: &[u16],
    base: usize,
    reduce: usize,
    inner: usize,
    col0: usize,
    is_max: bool,
) {
    let w = out.len();
    for r in 0..reduce {
        let start = base + r * inner + col0;
        simd_minmax_row_acc_f16_f32(out, &values[start..start + w], is_max);
    }
}

fn simd_minmax_inner_axis_reduce_f16(
    values: &[u16],
    is_max: bool,
    outer: usize,
    reduce: usize,
    inner: usize,
) -> Vec<f64> {
    let init = if is_max {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    // f32 accumulator (no f64 widen) + threaded — mirrors simd_minmax_inner_axis_reduce_bf16.
    let mut acc = vec![init; outer * inner];
    let total = values.len();
    let want_threads = if total >= crate::arithmetic::CHEAP_BINARY_PARALLEL_MIN {
        crate::arithmetic::work_scaled_threads(total).min(16)
    } else {
        1
    };
    if want_threads <= 1 {
        for o in 0..outer {
            let out_row = &mut acc[o * inner..(o + 1) * inner];
            simd_minmax_f16_accumulate_block(
                out_row,
                values,
                o * reduce * inner,
                reduce,
                inner,
                0,
                is_max,
            );
        }
    } else if outer >= 2 {
        let threads = want_threads.min(outer);
        let per = outer.div_ceil(threads);
        std::thread::scope(|scope| {
            let mut rest: &mut [f32] = acc.as_mut_slice();
            let mut o0 = 0usize;
            while o0 < outer {
                let cnt = per.min(outer - o0);
                let (blk, tail) = rest.split_at_mut(cnt * inner);
                rest = tail;
                let base0 = o0;
                o0 += cnt;
                scope.spawn(move || {
                    for oi in 0..cnt {
                        let o = base0 + oi;
                        let out_row = &mut blk[oi * inner..(oi + 1) * inner];
                        simd_minmax_f16_accumulate_block(
                            out_row,
                            values,
                            o * reduce * inner,
                            reduce,
                            inner,
                            0,
                            is_max,
                        );
                    }
                });
            }
        });
    } else {
        let threads = want_threads.min(inner.max(1));
        let per = inner.div_ceil(threads);
        std::thread::scope(|scope| {
            let mut rest: &mut [f32] = acc.as_mut_slice();
            let mut c0 = 0usize;
            while c0 < inner {
                let w = per.min(inner - c0);
                let (blk, tail) = rest.split_at_mut(w);
                rest = tail;
                let col0 = c0;
                c0 += w;
                scope.spawn(move || {
                    simd_minmax_f16_accumulate_block(blk, values, 0, reduce, inner, col0, is_max);
                });
            }
        });
    }
    acc.iter().map(|&v| f64::from(v)).collect()
}

// ── inner>1 (leading/middle-axis) SUM/PROD for half floats ──────────────────
// The column accumulate `out[c] OP= decode(x[k,c])` vectorizes ACROSS cells while
// keeping each cell's k-fold order — so the result is bit-identical to the scalar fold
// even though +/* are non-associative (no reassociation WITHIN a cell). And IEEE +/*
// propagate NaN/±inf naturally, so no NaN mask is needed (unlike min/max). The
// per-`Literal` decode is the compute SIMD wins (LLVM does not autovectorize it).

#[inline]
fn simd_sumprod_row_acc_bf16(out: &mut [f64], inp: &[u16], is_sum: bool) {
    use std::simd::{Simd, num::SimdFloat, num::SimdUint};
    const L: usize = 4;
    let mut oc = out.chunks_exact_mut(L);
    let mut ic = inp.chunks_exact(L);
    for (o, i) in oc.by_ref().zip(ic.by_ref()) {
        let u = Simd::<u16, L>::from_slice(i);
        let b: Simd<f64, L> =
            Simd::<f32, L>::from_bits(u.cast::<u32>() << Simd::splat(16u32)).cast();
        let a = Simd::<f64, L>::from_slice(o);
        o.copy_from_slice(&(if is_sum { a + b } else { a * b }).to_array());
    }
    for (o, &v) in oc.into_remainder().iter_mut().zip(ic.remainder()) {
        let d = Literal::BF16Bits(v).as_f64().unwrap_or(0.0);
        *o = if is_sum { *o + d } else { *o * d };
    }
}

#[inline]
fn simd_sumprod_row_acc_f16(out: &mut [f64], inp: &[u16], is_sum: bool) {
    use std::simd::Simd;
    const L: usize = 8;
    let mut oc = out.chunks_exact_mut(L);
    let mut ic = inp.chunks_exact(L);
    for (o, i) in oc.by_ref().zip(ic.by_ref()) {
        let u = Simd::<u16, L>::from_slice(i);
        if crate::arithmetic::f16_input_needs_scalar(u) {
            for (slot, &bits) in o.iter_mut().zip(i) {
                let d = Literal::F16Bits(bits).as_f64().unwrap_or(0.0);
                *slot = if is_sum { *slot + d } else { *slot * d };
            }
        } else {
            let b = crate::arithmetic::f16_widen8(u);
            let a = Simd::<f64, L>::from_slice(o);
            o.copy_from_slice(&(if is_sum { a + b } else { a * b }).to_array());
        }
    }
    for (o, &v) in oc.into_remainder().iter_mut().zip(ic.remainder()) {
        let d = Literal::F16Bits(v).as_f64().unwrap_or(0.0);
        *o = if is_sum { *o + d } else { *o * d };
    }
}

fn simd_sumprod_inner_axis_reduce_bf16(
    values: &[u16],
    is_sum: bool,
    outer: usize,
    reduce: usize,
    inner: usize,
    init: f64,
) -> Vec<f64> {
    let mut result = vec![init; outer * inner];
    for o in 0..outer {
        let out_row = &mut result[o * inner..(o + 1) * inner];
        let base = o * reduce * inner;
        for r in 0..reduce {
            simd_sumprod_row_acc_bf16(
                out_row,
                &values[base + r * inner..base + r * inner + inner],
                is_sum,
            );
        }
    }
    result
}

fn simd_sumprod_inner_axis_reduce_f16(
    values: &[u16],
    is_sum: bool,
    outer: usize,
    reduce: usize,
    inner: usize,
    init: f64,
) -> Vec<f64> {
    let mut result = vec![init; outer * inner];
    for o in 0..outer {
        let out_row = &mut result[o * inner..(o + 1) * inner];
        let base = o * reduce * inner;
        for r in 0..reduce {
            simd_sumprod_row_acc_f16(
                out_row,
                &values[base + r * inner..base + r * inner + inner],
                is_sum,
            );
        }
    }
    result
}

#[inline]
fn eval_dense_float_full_reduce(
    primitive: Primitive,
    tensor: &TensorValue,
    float_init: f64,
    float_op: &impl Fn(f64, f64) -> f64,
) -> Option<Value> {
    if !matches!(
        primitive,
        Primitive::ReduceSum | Primitive::ReduceProd | Primitive::ReduceMax | Primitive::ReduceMin
    ) {
        return None;
    }
    // A single-accumulator f64 fold over the flat backing slice, in strictly ascending
    // order. This is the EXACT shape of the generic per-`Literal` float full-reduce
    // (the `else` branch below): `acc: f64 = float_op(acc, literal.as_f64())`, then
    // round to the input's float dtype via `reduce_real_literal`. The only change is
    // reading a 4/8-byte contiguous scalar instead of a 24-byte boxed `Literal` with an
    // enum match. No reassociation (the `float_op` fn-pointer fold is not LLVM
    // reduction-vectorized at default opt), so it is BIT-IDENTICAL including ±inf/NaN
    // and signed-zero — verified by the dense full-reduce tests.
    //
    // F32 (JAX's DEFAULT float) accumulates in f64 too: `v as f64` for an `f32` equals
    // `Literal::F32Bits(b).as_f64()` exactly (lossless widening), and the result rounds
    // back to f32 at output — so the f32 path is just as order-identical as f64. (An
    // earlier attempt that accumulated NATIVELY in f32 diverged on inf+(-inf) NaN sign;
    // accumulating in f64 mirrors the generic fold and cannot diverge.)
    // ReduceMax/ReduceMin SIMD path: the scalar `jax_max`/`jax_min` fold above can't
    // autovectorize (per-step NaN branch), so vectorize it explicitly. max/min are
    // associative+commutative, so this is bit-identical (see simd_reduce_minmax_f64).
    // Prod stays on the scalar fold. Large f64 ReduceSum follows the JAX/XLA
    // tolerance contract instead of this module's older internal bit-order lock.
    let is_minmax = matches!(primitive, Primitive::ReduceMax | Primitive::ReduceMin);
    if is_minmax {
        let is_max = primitive == Primitive::ReduceMax;
        match tensor.dtype {
            DType::F64 => {
                if let Some(values) = tensor.elements.as_f64_slice() {
                    return Some(Value::Scalar(reduce_real_literal(
                        DType::F64,
                        threaded_reduce_minmax_f64(values, is_max),
                    )));
                }
            }
            DType::F32 => {
                if let Some(values) = tensor.elements.as_f32_slice() {
                    let m = threaded_reduce_minmax_f32(values, is_max);
                    return Some(Value::Scalar(Literal::from_f32(m)));
                }
            }
            DType::BF16 => {
                if let Some(values) = tensor.elements.as_half_float_slice() {
                    return Some(Value::Scalar(reduce_real_literal(
                        DType::BF16,
                        simd_reduce_minmax_bf16(values, is_max),
                    )));
                }
            }
            _ => {}
        }
    }

    match tensor.dtype {
        DType::F64 => {
            let values = tensor.elements.as_f64_slice()?;
            let acc = if primitive == Primitive::ReduceSum {
                threaded_tree_reduce_sum_f64(values)
            } else {
                let mut acc = float_init;
                for &value in values {
                    acc = float_op(acc, value);
                }
                acc
            };
            Some(Value::Scalar(reduce_real_literal(DType::F64, acc)))
        }
        DType::F32 => {
            let values = tensor.elements.as_f32_slice()?;
            let mut acc = float_init;
            for &value in values {
                acc = float_op(acc, value as f64);
            }
            Some(Value::Scalar(reduce_real_literal(DType::F32, acc)))
        }
        // BF16 is the dominant TRAINING dtype; loss / grad-norm full reductions over it
        // were boxed. bf16->f32 is EXACTLY `f32::from_bits((bits as u32) << 16)` (bf16 is
        // the truncated top 16 bits of f32, exact for all values incl ±inf/NaN/subnormal —
        // identical to the half crate's `f32::from(bf16)` that `Literal::BF16Bits.as_f64()`
        // uses). That widen is the bottleneck (unlike f32's free `as f64`), so we VECTORIZE
        // it 8 lanes at a time: u16 -> u32 (zero-extend) -> <<16 -> reinterpret f32 -> widen
        // f64. The fold itself stays a SINGLE scalar f64 accumulator over the lanes in
        // ascending order — no reassociation — so it is BIT-IDENTICAL to the generic
        // per-Literal fold (verified incl ±inf/NaN/signed-zero). Tail handled scalar.
        DType::BF16 => {
            use std::simd::{
                Simd,
                num::{SimdFloat, SimdUint},
            };
            const LANES: usize = 8;
            let values = tensor.elements.as_half_float_slice()?;
            let mut acc = float_init;
            let chunks = values.chunks_exact(LANES);
            let tail = chunks.remainder();
            for chunk in chunks {
                let u16v = Simd::<u16, LANES>::from_slice(chunk);
                let u32v = u16v.cast::<u32>() << Simd::splat(16u32);
                let f64v = Simd::<f32, LANES>::from_bits(u32v).cast::<f64>();
                for &v in f64v.to_array().iter() {
                    acc = float_op(acc, v);
                }
            }
            for &bits in tail {
                acc = float_op(acc, f64::from(f32::from_bits((bits as u32) << 16)));
            }
            Some(Value::Scalar(reduce_real_literal(DType::BF16, acc)))
        }
        // F16 needs the IEEE 5-bit-exponent decode (not a shift), and that widen is the
        // compute floor of the fold. Vectorize it 8 lanes via `f16_widen8` for chunks whose
        // inputs are all NORMAL/±0 (the common case); any chunk with a subnormal/inf/NaN f16
        // and the tail decode each value scalar via `Literal::F16Bits.as_f64()`. The fold is
        // a SINGLE scalar f64 accumulator over the lanes in ascending order (no reassociation,
        // and the SIMD widen equals `as_f64` exactly for normal/±0), so it is BIT-IDENTICAL
        // to the per-element scalar fold incl ±inf/NaN/subnormal/signed-zero.
        DType::F16 => {
            use std::simd::Simd;
            const LANES: usize = 8;
            let values = tensor.elements.as_half_float_slice()?;
            let mut acc = float_init;
            let chunks = values.chunks_exact(LANES);
            let tail = chunks.remainder();
            for chunk in chunks {
                let u = Simd::<u16, LANES>::from_slice(chunk);
                if crate::arithmetic::f16_input_needs_scalar(u) {
                    for &bits in chunk {
                        acc = float_op(acc, Literal::F16Bits(bits).as_f64().unwrap_or(0.0));
                    }
                } else {
                    for &v in crate::arithmetic::f16_widen8(u).to_array().iter() {
                        acc = float_op(acc, v);
                    }
                }
            }
            for &bits in tail {
                acc = float_op(acc, Literal::F16Bits(bits).as_f64().unwrap_or(0.0));
            }
            Some(Value::Scalar(reduce_real_literal(DType::F16, acc)))
        }
        _ => None,
    }
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
/// Thread a complex trailing-axis (inner==1) reduce over its INDEPENDENT output
/// rows: row `o` folds the contiguous run `values[o*reduce..(o+1)*reduce]` via
/// `fold_row` into `(result_re[o], result_im[o])`. Each row's fold is a
/// sequential complex-multiply / lexicographic-compare dependency CHAIN
/// (non-associative — its ascending-r order is preserved WITHIN a row), so the
/// win is parallelizing the independent rows' latency, NOT reassociating inside a
/// row; the result is BIT-IDENTICAL to the serial loop for any row partition
/// (each thread owns disjoint `result_re`/`result_im` sub-slices). Mirrors the
/// real-f64 trailing-axis threaded path (`dense_f64_axis_reduce`); gated on the
/// same `1<<18` total-work threshold so small reductions stay serial.
fn complex_inner1_reduce_rows(
    result_re: &mut [f64],
    result_im: &mut [f64],
    values: &[(f64, f64)],
    outer: usize,
    reduce: usize,
    fold_row: impl Fn(&[(f64, f64)]) -> (f64, f64) + Sync,
) {
    let total = outer.saturating_mul(reduce);
    let threads = if total >= (1 << 18) && outer > 1 {
        crate::arithmetic::work_scaled_threads(total).min(outer)
    } else {
        1
    };
    if threads <= 1 {
        for o in 0..outer {
            let (re, im) = fold_row(&values[o * reduce..o * reduce + reduce]);
            result_re[o] = re;
            result_im[o] = im;
        }
        return;
    }
    let rows_per = outer.div_ceil(threads);
    let fold_row = &fold_row;
    std::thread::scope(|scope| {
        let mut re_rest: &mut [f64] = result_re;
        let mut im_rest: &mut [f64] = result_im;
        let mut row0 = 0usize;
        while row0 < outer {
            let rows = rows_per.min(outer - row0);
            let (re_blk, re_tail) = re_rest.split_at_mut(rows);
            let (im_blk, im_tail) = im_rest.split_at_mut(rows);
            re_rest = re_tail;
            im_rest = im_tail;
            let vblk = &values[row0 * reduce..(row0 + rows) * reduce];
            row0 += rows;
            scope.spawn(move || {
                for r in 0..rows {
                    let (re, im) = fold_row(&vblk[r * reduce..r * reduce + reduce]);
                    re_blk[r] = re;
                    im_blk[r] = im;
                }
            });
        }
    });
}

/// Thread a complex INNER>1 (leading/middle-axis) reduce over its independent
/// output rows `o`. Output cell `(o, i)` (`i` in `0..inner`) folds the `reduce`
/// taps `values[(o*reduce + r)*inner + i]` for `r` ascending via `step`, seeded
/// with `init`. Each row `o` writes the disjoint block `[o*inner, (o+1)*inner)`
/// of `result_re`/`result_im`, so rows are independent and the threaded result is
/// BIT-IDENTICAL to the serial accumulate for any partition: the per-cell fold
/// visits `r` in the SAME ascending order as the serial `r`-outer/`i`-inner loop
/// (only the iteration order over distinct cells changes, never a cell's own fold
/// sequence). Prod/Max/Min route here (no `float_op`); ReduceSum stays serial.
/// Gated on the `1<<18` total-work threshold like the trailing-axis paths.
#[allow(clippy::too_many_arguments)]
fn complex_inner_axis_reduce_rows(
    result_re: &mut [f64],
    result_im: &mut [f64],
    values: &[(f64, f64)],
    outer: usize,
    reduce: usize,
    inner: usize,
    init: (f64, f64),
    step: impl Fn((f64, f64), (f64, f64)) -> (f64, f64) + Sync,
) {
    let total = outer.saturating_mul(reduce).saturating_mul(inner);
    let threads = if total >= (1 << 18) && outer > 1 {
        crate::arithmetic::work_scaled_threads(total).min(outer)
    } else {
        1
    };
    let compute = |o_base: usize, re_blk: &mut [f64], im_blk: &mut [f64]| {
        let n_o = re_blk.len() / inner;
        for lo in 0..n_o {
            let o = o_base + lo;
            let row_base = o * reduce * inner;
            for i in 0..inner {
                let mut acc = init;
                let mut idx = row_base + i;
                for _ in 0..reduce {
                    acc = step(acc, values[idx]);
                    idx += inner;
                }
                re_blk[lo * inner + i] = acc.0;
                im_blk[lo * inner + i] = acc.1;
            }
        }
    };
    if threads <= 1 {
        compute(0, result_re, result_im);
        return;
    }
    let rows_per = outer.div_ceil(threads);
    let compute = &compute;
    std::thread::scope(|scope| {
        let mut re_rest: &mut [f64] = result_re;
        let mut im_rest: &mut [f64] = result_im;
        let mut o0 = 0usize;
        while o0 < outer {
            let n_o = rows_per.min(outer - o0);
            let (re_blk, re_tail) = re_rest.split_at_mut(n_o * inner);
            let (im_blk, im_tail) = im_rest.split_at_mut(n_o * inner);
            re_rest = re_tail;
            im_rest = im_tail;
            let o_base = o0;
            scope.spawn(move || compute(o_base, re_blk, im_blk));
            o0 += n_o;
        }
    });
}

pub(crate) fn eval_reduce(
    primitive: Primitive,
    inputs: &[Value],
    int_init: i64,
    float_init: f64,
    int_op: impl Fn(i64, i64) -> i64 + Sync,
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
                let literal = tensor.elements[0];
                if tensor.dtype == DType::I32 {
                    let value = literal.as_i64().ok_or(EvalError::InvalidTensor(
                        ValueError::ElementDTypeMismatch {
                            index: 0,
                            declared: tensor.dtype,
                            literal,
                        },
                    ))?;
                    return Ok(Value::scalar_i32(value as i32));
                }
                return Ok(Value::Scalar(literal));
            }

            if let Some(value) =
                eval_dense_float_full_reduce(primitive, tensor, float_init, &float_op)
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
                // Per-element fold step shared by the dense and boxed paths.
                let fold = |re_acc: &mut f64, im_acc: &mut f64, re: f64, im: f64| match primitive {
                    Primitive::ReduceProd => {
                        let new_re = *re_acc * re - *im_acc * im;
                        let new_im = *re_acc * im + *im_acc * re;
                        *re_acc = new_re;
                        *im_acc = new_im;
                    }
                    Primitive::ReduceMax => {
                        if complex_lex_cmp((re, im), (*re_acc, *im_acc)).is_gt() {
                            *re_acc = re;
                            *im_acc = im;
                        }
                    }
                    Primitive::ReduceMin => {
                        if complex_lex_cmp((re, im), (*re_acc, *im_acc)).is_lt() {
                            *re_acc = re;
                            *im_acc = im;
                        }
                    }
                    _ => {
                        *re_acc += re;
                        *im_acc += im;
                    }
                };
                // Dense fast path: fold the packed (re, im) backing directly — bit-
                // identical to the boxed loop (as_complex_slice yields exactly what
                // literal_to_complex_parts returns for dense complex), skipping the
                // per-element 24-byte Literal reconstruction.
                if let Some(src) = tensor.elements.as_complex_slice() {
                    for &(re, im) in src {
                        fold(&mut re_acc, &mut im_acc, re, im);
                    }
                } else {
                    for literal in &tensor.elements {
                        let (re, im) = literal_to_complex_parts(primitive, *literal)?;
                        fold(&mut re_acc, &mut im_acc, re, im);
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
                // same `int_init` seed, same ascending order, same `int_op` —
                // but skips the per-element `Literal::I64` match and the 24-byte
                // enum stride. `as_i64_slice()` is `Some` for I64 dense storage
                // (which also backs I32-dtype tensors), so both dtypes use it.
                let acc = if let Some(values) = tensor.elements.as_i64_slice() {
                    let n = values.len();
                    // Integer reduce ops (sum/prod/and/or/xor/max/min) are ASSOCIATIVE
                    // / order-invariant and `int_init` is the op's identity, so chunked
                    // partial folds combined with `int_op` are BIT-IDENTICAL to the
                    // sequential fold (mod 2^64 / mod 2^32 is a +,* homomorphism; max/min
                    // order-free). A full reduce is a pure sequential read (BW-bound):
                    // one core cannot saturate multi-channel DRAM, so split the read
                    // across cores. Float can't do this (non-associative); integers can.
                    if n >= crate::arithmetic::CHEAP_BINARY_PARALLEL_MIN
                        && crate::arithmetic::work_scaled_threads(n) > 1
                    {
                        // A pure sequential read saturates multi-channel DRAM at ~8
                        // cores; more threads just add spawn + sequential-combine
                        // overhead (each reads a smaller chunk). Cap accordingly.
                        let threads = crate::arithmetic::work_scaled_threads(n).min(8);
                        let chunk = n.div_ceil(threads);
                        let int_op_ref = &int_op;
                        let partials: Vec<i64> = std::thread::scope(|scope| {
                            let handles: Vec<_> = values
                                .chunks(chunk)
                                .map(|c| {
                                    scope.spawn(move || {
                                        let mut a = int_init;
                                        for &v in c {
                                            a = int_op_ref(a, v);
                                        }
                                        a
                                    })
                                })
                                .collect();
                            handles
                                .into_iter()
                                .map(|h| h.join().expect("reduce partial-fold thread"))
                                .collect()
                        });
                        let mut acc = int_init;
                        for p in partials {
                            acc = int_op(acc, p);
                        }
                        acc
                    } else {
                        let mut acc = int_init;
                        for &val in values {
                            acc = int_op(acc, val);
                        }
                        acc
                    }
                } else {
                    let mut acc = int_init;
                    for literal in &tensor.elements {
                        let val = literal.as_i64().ok_or(EvalError::TypeMismatch {
                            primitive,
                            detail: "expected i64 tensor",
                        })?;
                        acc = int_op(acc, val);
                    }
                    acc
                };
                // b6w3l gap (2), full-reduce-to-scalar half: an int32 sum/prod must
                // wrap two's-complement to int32 (JAX/XLA keep int32 width). The
                // scalar result carries NO dtype tag (no Literal::I32), so it can't
                // be re-narrowed downstream by narrow_i32_tensor_result — fix the
                // value here where tensor.dtype is still known. Wrapping only the
                // final accumulator equals per-step int32 wrapping because mod 2^32
                // is a ring homomorphism for + and *. Gated to sum/prod: max/min of
                // valid int32 inputs always stay in range, so they need no wrap (and
                // this avoids touching their empty-input init sentinels).
                let acc = if tensor.dtype == DType::I32
                    && matches!(primitive, Primitive::ReduceSum | Primitive::ReduceProd)
                {
                    i64::from(acc as i32)
                } else {
                    acc
                };
                if tensor.dtype == DType::I32 {
                    Ok(Value::scalar_i32(acc as i32))
                } else {
                    Ok(Value::scalar_i64(acc))
                }
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
        // Column-accumulation (sum/mean over the LEADING axis — batch reduction): each
        // output column folds its `outer` rows in ascending order. The columns are
        // INDEPENDENT, so split the OUTPUT COLUMNS across threads — every column keeps
        // its exact ascending fold, so this is BIT-IDENTICAL to the serial fold even
        // for non-associative float (we never reassociate a single column's sum; we
        // only assign different columns to different threads). A read-bound pass; one
        // core can't saturate multi-channel DRAM, so cap at ~8 cores (more REGRESS).
        let threads = crate::arithmetic::work_scaled_threads(values.len())
            .min(block)
            .min(8);
        if threads > 1 {
            let cols_per = block.div_ceil(threads);
            let op_ref = float_op;
            std::thread::scope(|scope| {
                let mut res_rest: &mut [f64] = result.as_mut_slice();
                let mut c0 = 0usize;
                while c0 < block {
                    let w = cols_per.min(block - c0);
                    let (res_blk, res_tail) = res_rest.split_at_mut(w);
                    res_rest = res_tail;
                    let c_start = c0;
                    c0 += w;
                    scope.spawn(move || {
                        for k in 0..outer {
                            let row = &values[k * block + c_start..k * block + c_start + w];
                            for (slot, &v) in res_blk.iter_mut().zip(row.iter()) {
                                *slot = op_ref(*slot, widen(v));
                            }
                        }
                    });
                }
            });
            return Some(result);
        }
        for k in 0..outer {
            let row = &values[k * block..k * block + block];
            for (slot, &v) in result.iter_mut().zip(row.iter()) {
                *slot = float_op(*slot, widen(v));
            }
        }
        return Some(result);
    }

    // General contiguous-block serial path: the two threaded fast paths above
    // cover only LARGE leading-prefix / trailing-suffix reductions. Any other
    // contiguous-block reduction — a middle block, or a small tensor below the
    // threading gate — still beats the per-element odometer by a wide margin
    // with a plain hoistable fold. Factor [outer, reduce, inner]; inner==1 folds
    // a contiguous run into a scalar accumulator, inner>1 does per-row folds.
    // Each cell accumulates ascending-r — identical to the odometer's emission
    // order (and to the dense_f64 round below), so bit-for-bit identical incl.
    // NaN bits and non-associative sum order. Non-contiguous axis sets fall
    // through to the odometer.
    let reduced_axes: Vec<usize> = (0..rank).filter(|i| !kept_axes.contains(i)).collect();
    if let Some((outer, reduce, inner)) = contiguous_reduce_block(&tensor.shape.dims, &reduced_axes)
    {
        let mut result = vec![float_init; out_count];
        if inner == 1 {
            for (o, slot) in result.iter_mut().enumerate() {
                let base = o * reduce;
                let mut acc = float_init;
                for &v in &values[base..base + reduce] {
                    acc = float_op(acc, widen(v));
                }
                *slot = acc;
            }
        } else {
            // Reduce over a MIDDLE block keeping a contiguous trailing `inner` (e.g. global
            // avg/sum pool: sum H*W keeping channel C). Each `outer` slice is independent and
            // its `reduce` rows are folded in ascending order, so threading across `outer` is
            // bit-identical (per-outer order preserved; max/min order-independent). The inner
            // `out_row[c] op= widen(in_row[c])` loop autovectorizes per monomorphization.
            let work = outer.saturating_mul(reduce).saturating_mul(inner);
            let threads = if outer >= 2 && work >= (1 << 18) {
                std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(1)
                    .min(16)
                    .min(outer)
            } else {
                1
            };
            let fold_outer = |base: usize, chunk: &mut [f64]| {
                let cnt = chunk.len() / inner;
                for oi in 0..cnt {
                    let o = base + oi;
                    let out_row = &mut chunk[oi * inner..(oi + 1) * inner];
                    for r in 0..reduce {
                        let in_row = &values[(o * reduce + r) * inner..][..inner];
                        for (slot, &v) in out_row.iter_mut().zip(in_row) {
                            *slot = float_op(*slot, widen(v));
                        }
                    }
                }
            };
            if threads <= 1 {
                fold_outer(0, result.as_mut_slice());
            } else {
                let per = outer.div_ceil(threads);
                std::thread::scope(|scope| {
                    let mut o0 = 0usize;
                    let mut rest: &mut [f64] = result.as_mut_slice();
                    while o0 < outer {
                        let cnt = per.min(outer - o0);
                        let (chunk, tail) = rest.split_at_mut(cnt * inner);
                        rest = tail;
                        let base = o0;
                        let fold_ref = &fold_outer;
                        scope.spawn(move || fold_ref(base, chunk));
                        o0 += cnt;
                    }
                });
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

#[inline]
fn fold_f16_axis_block(
    values: &[u16],
    float_init: f64,
    float_op: &(impl Fn(f64, f64) -> f64 + Sync),
) -> f64 {
    use std::simd::Simd;
    const LANES: usize = 8;

    let mut acc = float_init;
    let chunks = values.chunks_exact(LANES);
    let tail = chunks.remainder();
    for chunk in chunks {
        let u = Simd::<u16, LANES>::from_slice(chunk);
        if crate::arithmetic::f16_input_needs_scalar(u) {
            for &bits in chunk {
                acc = float_op(acc, Literal::F16Bits(bits).as_f64().unwrap_or(0.0));
            }
        } else {
            for &value in crate::arithmetic::f16_widen8(u).to_array().iter() {
                acc = float_op(acc, value);
            }
        }
    }
    for &bits in tail {
        acc = float_op(acc, Literal::F16Bits(bits).as_f64().unwrap_or(0.0));
    }
    acc
}

#[inline]
fn dense_f16_trailing_axis_reduce(
    tensor: &TensorValue,
    values: &[u16],
    kept_axes: &[usize],
    out_count: usize,
    float_init: f64,
    float_op: &(impl Fn(f64, f64) -> f64 + Sync),
) -> Option<Vec<f64>> {
    if tensor.shape.dims.is_empty() || out_count == 0 {
        return None;
    }
    let kept_is_leading_prefix = kept_axes.iter().enumerate().all(|(i, &ax)| ax == i);
    if !kept_is_leading_prefix || values.len() != out_count * (values.len() / out_count) {
        return None;
    }

    let reduce = values.len() / out_count;
    let mut result = vec![float_init; out_count];
    let threads = if values.len() >= (1 << 18) {
        crate::arithmetic::work_scaled_threads(values.len()).min(out_count)
    } else {
        1
    };
    if threads > 1 {
        let rows_per = out_count.div_ceil(threads);
        std::thread::scope(|scope| {
            let mut res_rest: &mut [f64] = result.as_mut_slice();
            let mut row0 = 0usize;
            while row0 < out_count {
                let rows = rows_per.min(out_count - row0);
                let (res_blk, res_tail) = res_rest.split_at_mut(rows);
                res_rest = res_tail;
                let vblk = &values[row0 * reduce..(row0 + rows) * reduce];
                row0 += rows;
                scope.spawn(move || {
                    for (r, slot) in res_blk.iter_mut().enumerate() {
                        *slot = fold_f16_axis_block(
                            &vblk[r * reduce..r * reduce + reduce],
                            float_init,
                            float_op,
                        );
                    }
                });
            }
        });
    } else {
        for (row, slot) in result.iter_mut().enumerate() {
            let base = row * reduce;
            *slot = fold_f16_axis_block(&values[base..base + reduce], float_init, float_op);
        }
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

/// If the sorted reduced axes form a single contiguous block `[lo..=hi]` of the
/// shape, factor the row-major layout as `[outer, reduce, inner]` — the product
/// of axes strictly before the block, within it, and strictly after it. A flat
/// element at `((o * reduce) + r) * inner + i` reduces into output cell
/// `o * inner + i`, folding `r` in ascending order — exactly the visitation
/// order of the generic `OutIndexOdometer`, so the result is bit-identical while
/// the inner `i` (or, when `inner == 1`, the `r`) loop is a contiguous, hoistable
/// fold that autovectorizes. Returns `None` for a non-contiguous axis set (e.g.
/// `{0, 2}` of a rank-3 tensor), which keeps the generic odometer.
fn contiguous_reduce_block(dims: &[u32], axes_sorted: &[usize]) -> Option<(usize, usize, usize)> {
    let &lo = axes_sorted.first()?;
    let &hi = axes_sorted.last()?;
    if hi - lo + 1 != axes_sorted.len() {
        return None;
    }
    let prod = |slice: &[u32]| slice.iter().map(|&d| d as usize).product::<usize>();
    let outer = prod(&dims[..lo]);
    let reduce = prod(&dims[lo..=hi]);
    let inner = prod(&dims[hi + 1..]);
    Some((outer, reduce, inner))
}

pub(crate) fn eval_reduce_axes(
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

    // Unsigned (u32/u64) reductions: JAX keeps the uint dtype and wraps sum/prod mod
    // 2^N with UNSIGNED max/min, but the integral dense path below gates on I64|I32 —
    // so u32/u64 used to fall to the FLOAT arm (f64 accumulation, F64 output): wrong
    // dtype, no wrap, precision loss above 2^53. Fix by mapping the unsigned values to
    // i64 with an ORDER- and RING-faithful transform, reusing the fully-tested i64
    // reduce machinery (full + partial), then mapping the i64 result back. Only
    // sum/prod/max/min reach here (and/or/xor route through eval_reduce_bitwise_axes).
    //
    // Transforms (each provably bit-exact vs JAX):
    //   u32 (all ops):   fwd v→i64::from(v) (exact, non-negative ⇒ signed cmp == unsigned);
    //                    inv i64→`v as u32` (sum/prod: mod 2^64 then mod 2^32 == mod 2^32).
    //   u64 sum/prod:    fwd `v as i64` (bit reinterpret); i64 wrapping fold is mod 2^64
    //                    == u64 wrapping; inv `i64 as u64`.
    //   u64 max/min:     fwd `(v ^ 2^63) as i64` — the order-preserving sign-flip maps the
    //                    unsigned order bijectively onto the signed i64 order (so signed
    //                    max/min == unsigned, incl values > i64::MAX, and the i64::MIN/MAX
    //                    empty-init sentinels invert to the correct 0 / u64::MAX identity);
    //                    inv `(i64 as u64) ^ 2^63`.
    if let Value::Tensor(t) = &inputs[0]
        && matches!(t.dtype, DType::U32 | DType::U64)
        && matches!(
            primitive,
            Primitive::ReduceSum
                | Primitive::ReduceProd
                | Primitive::ReduceMax
                | Primitive::ReduceMin
        )
    {
        const FLIP: u64 = 1u64 << 63;
        let is_u32 = t.dtype == DType::U32;
        let is_minmax = matches!(primitive, Primitive::ReduceMax | Primitive::ReduceMin);
        let fwd = |v: u64| -> i64 {
            if is_u32 {
                v as i64 // u32 value (low 32 bits set) -> non-negative i64
            } else if is_minmax {
                (v ^ FLIP) as i64
            } else {
                v as i64
            }
        };
        let inv = |v: i64| -> u64 {
            if is_u32 {
                u64::from(v as u32)
            } else if is_minmax {
                (v as u64) ^ FLIP
            } else {
                v as u64
            }
        };
        let widened: Vec<i64> = match t.elements.as_u32_slice() {
            Some(s) => s.iter().map(|&v| fwd(u64::from(v))).collect(),
            None => match t.elements.as_u64_slice() {
                Some(s) => s.iter().map(|&v| fwd(v)).collect(),
                None => t
                    .elements
                    .iter()
                    .map(|l| {
                        l.as_u64().map(fwd).ok_or(EvalError::TypeMismatch {
                            primitive,
                            detail: "expected unsigned tensor",
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            },
        };
        let widened_input = vec![Value::Tensor(TensorValue::new_i64_values(
            t.shape.clone(),
            widened,
        )?)];
        let result = eval_reduce_axes(
            primitive,
            &widened_input,
            params,
            int_init,
            float_init,
            int_op,
            float_op,
        )?;
        let build = |shape: Shape, vals: Vec<u64>| -> Result<Value, EvalError> {
            Ok(Value::Tensor(if is_u32 {
                TensorValue::new_u32_values(shape, vals.iter().map(|&v| v as u32).collect())?
            } else {
                TensorValue::new_u64_values(shape, vals)?
            }))
        };
        return Ok(match result {
            Value::Scalar(Literal::I64(v)) => {
                let u = inv(v);
                if is_u32 {
                    Value::Scalar(Literal::U32(u as u32))
                } else {
                    Value::Scalar(Literal::U64(u))
                }
            }
            Value::Scalar(other) => Value::Scalar(other),
            Value::Tensor(rt) => {
                let vals: Vec<u64> = match rt.elements.as_i64_slice() {
                    Some(s) => s.iter().map(|&v| inv(v)).collect(),
                    None => rt
                        .elements
                        .iter()
                        .map(|l| inv(l.as_i64().unwrap_or(0)))
                        .collect(),
                };
                build(rt.shape.clone(), vals)?
            }
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
                // One accumulation step into output cell `idx` for value `(re, im)`,
                // shared by the dense contiguous-block fast path and the generic
                // odometer fallback so they stay bit-identical.
                let accumulate = |result_re: &mut [f64],
                                  result_im: &mut [f64],
                                  idx: usize,
                                  re: f64,
                                  im: f64| {
                    match primitive {
                        Primitive::ReduceProd => {
                            let acc_re = result_re[idx];
                            let acc_im = result_im[idx];
                            result_re[idx] = acc_re * re - acc_im * im;
                            result_im[idx] = acc_re * im + acc_im * re;
                        }
                        Primitive::ReduceMax => {
                            if complex_lex_cmp((re, im), (result_re[idx], result_im[idx])).is_gt() {
                                result_re[idx] = re;
                                result_im[idx] = im;
                            }
                        }
                        Primitive::ReduceMin => {
                            if complex_lex_cmp((re, im), (result_re[idx], result_im[idx])).is_lt() {
                                result_re[idx] = re;
                                result_im[idx] = im;
                            }
                        }
                        _ => {
                            // ReduceSum: component-wise float_op (addition).
                            result_re[idx] = float_op(result_re[idx], re);
                            result_im[idx] = float_op(result_im[idx], im);
                        }
                    }
                };

                // Dense complex contiguous-block fast path: when the reduced axes
                // form one contiguous block, factor [outer, reduce, inner] and fold
                // each output cell over a contiguous run — no per-element odometer
                // carry. `as_complex_slice()` yields the same (re, im) pairs as
                // `literal_to_complex_parts` (Complex64 storage is f32-exact), and
                // each cell accumulates ascending-r — exactly the odometer's order —
                // so the result is bit-identical (incl. non-associative sum order and
                // lexicographic max/min). Non-contiguous axis sets keep the odometer.
                let block = tensor
                    .elements
                    .as_complex_slice()
                    .zip(contiguous_reduce_block(&tensor.shape.dims, &axes_sorted));
                if let Some((values, (outer, reduce, inner))) = block {
                    // Hoist the per-op match OUT of the element loop. For the
                    // dominant inner==1 (reduce trailing axes) case, fold the
                    // contiguous run into scalar (re, im) accumulators — no closure
                    // call, no per-element match, no indexed write — then store once.
                    // inner>1 reuses the shared `accumulate` step (still ascending-r,
                    // bit-identical). ReduceSum keeps `float_op` so its addition
                    // semantics stay identical to the generic path.
                    if inner == 1 {
                        match primitive {
                            // ReduceSum float_op is component-wise addition; inlining
                            // `+` is bit-identical (proven by the trailing-axis test)
                            // and keeps the row-fold `Sync`, so sum threads too.
                            Primitive::ReduceSum => complex_inner1_reduce_rows(
                                &mut result_re,
                                &mut result_im,
                                values,
                                outer,
                                reduce,
                                |row| {
                                    let mut acc_re = init_re;
                                    let mut acc_im = init_im;
                                    for &(re, im) in row {
                                        acc_re += re;
                                        acc_im += im;
                                    }
                                    (acc_re, acc_im)
                                },
                            ),
                            // Prod/Max/Min are per-row complex-multiply /
                            // lexicographic-compare dependency chains (no `float_op`):
                            // thread the independent rows (bit-identical, latency-bound).
                            Primitive::ReduceProd => complex_inner1_reduce_rows(
                                &mut result_re,
                                &mut result_im,
                                values,
                                outer,
                                reduce,
                                |row| {
                                    let mut acc_re = init_re;
                                    let mut acc_im = init_im;
                                    for &(re, im) in row {
                                        let nr = acc_re * re - acc_im * im;
                                        let ni = acc_re * im + acc_im * re;
                                        acc_re = nr;
                                        acc_im = ni;
                                    }
                                    (acc_re, acc_im)
                                },
                            ),
                            Primitive::ReduceMax => complex_inner1_reduce_rows(
                                &mut result_re,
                                &mut result_im,
                                values,
                                outer,
                                reduce,
                                |row| {
                                    let mut best = (init_re, init_im);
                                    for &(re, im) in row {
                                        if complex_lex_cmp((re, im), best).is_gt() {
                                            best = (re, im);
                                        }
                                    }
                                    best
                                },
                            ),
                            _ => complex_inner1_reduce_rows(
                                // ReduceMin
                                &mut result_re,
                                &mut result_im,
                                values,
                                outer,
                                reduce,
                                |row| {
                                    let mut best = (init_re, init_im);
                                    for &(re, im) in row {
                                        if complex_lex_cmp((re, im), best).is_lt() {
                                            best = (re, im);
                                        }
                                    }
                                    best
                                },
                            ),
                        }
                    } else {
                        // inner>1 (leading/middle-axis). All four route through the
                        // threaded row driver: prod/max/min are per-cell chains;
                        // ReduceSum's float_op is component-wise `+`, inlined here
                        // (bit-identical, keeps the step `Sync`).
                        match primitive {
                            Primitive::ReduceSum => complex_inner_axis_reduce_rows(
                                &mut result_re,
                                &mut result_im,
                                values,
                                outer,
                                reduce,
                                inner,
                                (init_re, init_im),
                                |acc, v| (acc.0 + v.0, acc.1 + v.1),
                            ),
                            Primitive::ReduceProd => complex_inner_axis_reduce_rows(
                                &mut result_re,
                                &mut result_im,
                                values,
                                outer,
                                reduce,
                                inner,
                                (init_re, init_im),
                                |acc, v| (acc.0 * v.0 - acc.1 * v.1, acc.0 * v.1 + acc.1 * v.0),
                            ),
                            Primitive::ReduceMax => complex_inner_axis_reduce_rows(
                                &mut result_re,
                                &mut result_im,
                                values,
                                outer,
                                reduce,
                                inner,
                                (init_re, init_im),
                                |acc, v| {
                                    if complex_lex_cmp(v, acc).is_gt() {
                                        v
                                    } else {
                                        acc
                                    }
                                },
                            ),
                            Primitive::ReduceMin => complex_inner_axis_reduce_rows(
                                &mut result_re,
                                &mut result_im,
                                values,
                                outer,
                                reduce,
                                inner,
                                (init_re, init_im),
                                |acc, v| {
                                    if complex_lex_cmp(v, acc).is_lt() {
                                        v
                                    } else {
                                        acc
                                    }
                                },
                            ),
                            _ => {
                                for o in 0..outer {
                                    for r in 0..reduce {
                                        let in_base = (o * reduce + r) * inner;
                                        let out_base = o * inner;
                                        for i in 0..inner {
                                            let (re, im) = values[in_base + i];
                                            accumulate(
                                                &mut result_re,
                                                &mut result_im,
                                                out_base + i,
                                                re,
                                                im,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    let mut odometer =
                        OutIndexOdometer::new(&tensor.shape.dims, &kept_axes, &out_dims);
                    for literal in tensor.elements.iter() {
                        let out_idx = odometer.next_index();
                        let (re, im) = literal_to_complex_parts(primitive, *literal)?;
                        accumulate(&mut result_re, &mut result_im, out_idx, re, im);
                    }
                }

                // De-box: wrap the dense (re, im) pairs directly via new_complex_values
                // (dense Complex storage) instead of mapping through
                // complex_literal_from_parts into a boxed Vec<Literal>. Bit-identical:
                // new_complex_values(Complex64, (re,im)) rounds re/im to f32 exactly as
                // from_complex64(re as f32, im as f32), and Complex128 stores f64 as-is —
                // the same values complex_literal_from_parts produced. Keeps complex reduce
                // output on the dense Complex fast path (FFT post-processing chains).
                let out: Vec<(f64, f64)> = result_re.into_iter().zip(result_im).collect();
                Ok(Value::Tensor(TensorValue::new_complex_values(
                    tensor.dtype,
                    Shape { dims: out_dims },
                    out,
                )?))
            } else if is_integral {
                let mut result = try_filled_vec(
                    primitive,
                    "reduction integer accumulator",
                    out_count,
                    int_init,
                )?;
                // Dense i64 contiguous-block fast path: when the reduced axes form
                // one contiguous block, the layout factors as [outer, reduce, inner]
                // and each output cell is a hoistable, autovectorizing fold over a
                // contiguous run — no per-element odometer carry. Bit-identical:
                // each cell accumulates ascending-r, exactly the odometer's order
                // (int_op is associative for sum/prod; max/min are order-invariant).
                let dense_i64 = tensor.elements.as_i64_slice();
                let block =
                    dense_i64.zip(contiguous_reduce_block(&tensor.shape.dims, &axes_sorted));
                if let Some((values, (outer, reduce, inner))) = block {
                    let par = values.len() >= crate::arithmetic::CHEAP_BINARY_PARALLEL_MIN
                        && crate::arithmetic::work_scaled_threads(values.len()) > 1;
                    if inner == 1 {
                        // Trailing-axis integer reduce: each output element folds one
                        // contiguous `reduce`-run; the outputs are INDEPENDENT, so thread
                        // over them (contiguous reads, bit-identical — each block keeps
                        // its ascending fold). A read-bound pass saturates DRAM at ~8 cores.
                        let outer_n = result.len();
                        if par && outer_n > 1 {
                            let threads = crate::arithmetic::work_scaled_threads(values.len())
                                .min(outer_n)
                                .min(8);
                            let rows_per = outer_n.div_ceil(threads);
                            let op_ref = &int_op;
                            std::thread::scope(|scope| {
                                let mut res_rest: &mut [i64] = result.as_mut_slice();
                                let mut o0 = 0usize;
                                while o0 < outer_n {
                                    let rows = rows_per.min(outer_n - o0);
                                    let (res_blk, res_tail) = res_rest.split_at_mut(rows);
                                    res_rest = res_tail;
                                    let vblk = &values[o0 * reduce..(o0 + rows) * reduce];
                                    o0 += rows;
                                    scope.spawn(move || {
                                        for (r, slot) in res_blk.iter_mut().enumerate() {
                                            let mut acc = int_init;
                                            for &v in &vblk[r * reduce..r * reduce + reduce] {
                                                acc = op_ref(acc, v);
                                            }
                                            *slot = acc;
                                        }
                                    });
                                }
                            });
                        } else {
                            for (o, slot) in result.iter_mut().enumerate() {
                                let base = o * reduce;
                                let mut acc = int_init;
                                for &v in &values[base..base + reduce] {
                                    acc = int_op(acc, v);
                                }
                                *slot = acc;
                            }
                        }
                    } else if outer == 1 && par && reduce > 1 {
                        // Leading-axis integer reduce (sum over axis 0): column
                        // accumulation over `reduce` rows. Integer add/etc. is ASSOCIATIVE,
                        // so unlike the float column reduce we can thread the REDUCE
                        // dimension with CONTIGUOUS reads: each thread folds a row-band into
                        // a local `inner`-wide partial, then combine partials in chunk order
                        // (== ascending). Bit-identical (associativity); contiguous beats the
                        // strided column threading the float path is limited to.
                        let threads = crate::arithmetic::work_scaled_threads(values.len())
                            .min(reduce)
                            .min(8);
                        let rows_per = reduce.div_ceil(threads);
                        let op_ref = &int_op;
                        let partials: Vec<Vec<i64>> = std::thread::scope(|scope| {
                            let mut handles = Vec::new();
                            let mut r0 = 0usize;
                            while r0 < reduce {
                                let r1 = (r0 + rows_per).min(reduce);
                                let vchunk = &values[r0 * inner..r1 * inner];
                                let rows = r1 - r0;
                                r0 = r1;
                                handles.push(scope.spawn(move || {
                                    let mut local = vec![int_init; inner];
                                    for r in 0..rows {
                                        let in_row = &vchunk[r * inner..r * inner + inner];
                                        for (slot, &v) in local.iter_mut().zip(in_row) {
                                            *slot = op_ref(*slot, v);
                                        }
                                    }
                                    local
                                }));
                            }
                            handles
                                .into_iter()
                                .map(|h| h.join().expect("reduce partial thread"))
                                .collect()
                        });
                        for partial in &partials {
                            for (slot, &v) in result.iter_mut().zip(partial.iter()) {
                                *slot = int_op(*slot, v);
                            }
                        }
                    } else {
                        for o in 0..outer {
                            let out_row = &mut result[o * inner..(o + 1) * inner];
                            for r in 0..reduce {
                                let in_row = &values[(o * reduce + r) * inner..][..inner];
                                for (slot, &v) in out_row.iter_mut().zip(in_row) {
                                    *slot = int_op(*slot, v);
                                }
                            }
                        }
                    }
                } else {
                    let mut odometer =
                        OutIndexOdometer::new(&tensor.shape.dims, &kept_axes, &out_dims);
                    // Dense i64 fast path: drive the odometer over the contiguous
                    // `i64` backing slice, skipping the per-element `Literal::I64`
                    // match and 24-byte stride. Bit-identical to the generic loop
                    // (same order, out_idx sequence, int_op). `as_i64_slice()` is
                    // `Some` only for I64 dense storage.
                    if let Some(values) = dense_i64 {
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
                }
                // Preserve the input integer dtype: a JAX/XLA int32 reduction
                // stays int32 (it does NOT widen to int64). Accumulation runs in
                // i64 (matching XLA's widened accumulator), then an int32 result
                // wraps back to two's-complement int32 — same semantics as the
                // elementwise int32 ops and the `narrow_i32_tensor_result`
                // chokepoint, applied here so the reduce is self-correct
                // regardless of caller. int64 reductions are unchanged.
                let out_dtype = tensor.dtype;
                // De-box the dense `Vec<i64> result` directly (8 vs 24-byte boxed Literal
                // stride + per-element TensorValue::new validation; the boxing dominated
                // large integer-reduce outputs). Bit-identical: new_i64_values stores
                // Literal::I64(v); for I32, new_i32_values stores the i32-wrapped value
                // exactly as the boxed Literal::I64(i64::from(v as i32)) did (same ctor
                // convert_element_type uses for f->i32). i32 is JAX's default int.
                match out_dtype {
                    DType::I64 => Ok(Value::Tensor(TensorValue::new_i64_values(
                        Shape { dims: out_dims },
                        result,
                    )?)),
                    DType::I32 => Ok(Value::Tensor(TensorValue::new_i32_values(
                        Shape { dims: out_dims },
                        result.into_iter().map(|v| i64::from(v as i32)).collect(),
                    )?)),
                    _ => {
                        let elements: Vec<Literal> = result.into_iter().map(Literal::I64).collect();
                        Ok(Value::Tensor(TensorValue::new(
                            out_dtype,
                            Shape { dims: out_dims },
                            elements,
                        )?))
                    }
                }
            } else {
                // SIMD ReduceMax/ReduceMin axis path (F64/F32, trailing-axes `inner==1`
                // contiguous block — the `jnp.max(x, axis=-1)` case): each output cell
                // folds a contiguous run via the vectorized min/max reduce, bit-identical
                // to the scalar `jax_max`/`jax_min` per-cell fold. Falls through to the
                // generic dense path for inner>1 / non-contiguous axis sets / Sum/Prod.
                let simd_minmax =
                    if matches!(primitive, Primitive::ReduceMax | Primitive::ReduceMin) {
                        let is_max = primitive == Primitive::ReduceMax;
                        match (
                            tensor.dtype,
                            contiguous_reduce_block(&tensor.shape.dims, &axes_sorted),
                        ) {
                            (DType::F64, Some((outer, reduce, 1))) => tensor
                                .elements
                                .as_f64_slice()
                                .map(|v| simd_minmax_axis_reduce_f64(v, is_max, outer, reduce)),
                            (DType::F32, Some((outer, reduce, 1))) => tensor
                                .elements
                                .as_f32_slice()
                                .map(|v| simd_minmax_axis_reduce_f32(v, is_max, outer, reduce)),
                            (DType::BF16, Some((outer, reduce, 1))) => tensor
                                .elements
                                .as_half_float_slice()
                                .map(|v| simd_minmax_axis_reduce_bf16(v, is_max, outer, reduce)),
                            (DType::F16, Some((outer, reduce, 1))) => tensor
                                .elements
                                .as_half_float_slice()
                                .map(|v| simd_minmax_axis_reduce_f16(v, is_max, outer, reduce)),
                            // inner>1: leading/middle-axis reduction (e.g. max(x, axis=0)).
                            // Each output cell folds a contiguous inner ROW with SIMD
                            // max/min + per-lane NaN mask (no ±0 fallback needed — no
                            // horizontal reduce). f64/f32 only (half decode TBD).
                            (DType::F64, Some((outer, reduce, inner))) if inner > 1 => {
                                tensor.elements.as_f64_slice().map(|v| {
                                    simd_minmax_inner_axis_reduce_f64(
                                        v, is_max, outer, reduce, inner,
                                    )
                                })
                            }
                            (DType::F32, Some((outer, reduce, inner))) if inner > 1 => {
                                tensor.elements.as_f32_slice().map(|v| {
                                    simd_minmax_inner_axis_reduce_f32(
                                        v, is_max, outer, reduce, inner,
                                    )
                                })
                            }
                            // bf16 inner>1: the per-element decode is the compute SIMD
                            // wins (the native f64/f32 fold is already autovectorized).
                            (DType::BF16, Some((outer, reduce, inner))) if inner > 1 => {
                                tensor.elements.as_half_float_slice().map(|v| {
                                    simd_minmax_inner_axis_reduce_bf16(
                                        v, is_max, outer, reduce, inner,
                                    )
                                })
                            }
                            (DType::F16, Some((outer, reduce, inner))) if inner > 1 => {
                                tensor.elements.as_half_float_slice().map(|v| {
                                    simd_minmax_inner_axis_reduce_f16(
                                        v, is_max, outer, reduce, inner,
                                    )
                                })
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };

                // Half-float inner>1 SUM/PROD SIMD path: the column accumulate folds the
                // contiguous inner ROW with a SIMD decode + add/mul, preserving each
                // cell's k-order (bit-identical though +/* are non-associative). The
                // decode is the win (native f64/f32 sum/prod is already autovectorized;
                // the trailing-axis half sum is handled by the existing decode paths).
                // BOTH ReduceSum and ReduceProd route here: the per-cell column fold keeps
                // the exact k-order, the decode (`f16_widen8`/bf16 shift) is identical to
                // the boxed `Literal::as_f64()` the generic path uses, and `float_init` is
                // the primitive's identity (0.0 for sum, 1.0 for prod). So `out[c] OP= dec`
                // is bit-for-bit the generic odometer for both `+` and `*` — pinned by
                // dense_half_float_reduce_bit_identical_to_literal_path and the [5,4,17]
                // inner>1 prod fuzz test. (An earlier note suspected a `*` divergence; it
                // was a stale pre-`float_init`/pre-`needs_scalar` kernel — re-verified clean.)
                let is_sum = primitive == Primitive::ReduceSum;
                let simd_sumprod =
                    if matches!(primitive, Primitive::ReduceSum | Primitive::ReduceProd) {
                        match (
                            tensor.dtype,
                            contiguous_reduce_block(&tensor.shape.dims, &axes_sorted),
                        ) {
                            (DType::BF16, Some((outer, reduce, inner))) if inner > 1 => {
                                tensor.elements.as_half_float_slice().map(|v| {
                                    simd_sumprod_inner_axis_reduce_bf16(
                                        v, is_sum, outer, reduce, inner, float_init,
                                    )
                                })
                            }
                            (DType::F16, Some((outer, reduce, inner))) if inner > 1 => {
                                tensor.elements.as_half_float_slice().map(|v| {
                                    simd_sumprod_inner_axis_reduce_f16(
                                        v, is_sum, outer, reduce, inner, float_init,
                                    )
                                })
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };

                // Dense axis-reduce fast path, reading the native backing slice and
                // widening F32->f64 INLINE (no buffer). The generic odometer loop
                // below also folds in f64 over `as_f64()` and rounds via
                // `reduce_real_literal`, so the f32 dense path (same per-output-cell
                // ascending fold, same round) is bit-for-bit identical — incl. NaN
                // bits, since both fold the same f64 values in the same order.
                let dense = simd_minmax.or(simd_sumprod).or_else(|| match tensor.dtype {
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
                        dense_f16_trailing_axis_reduce(
                            tensor, v, &kept_axes, out_count, float_init, &float_op,
                        )
                        .or_else(|| {
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
                        })
                    }),
                    _ => None,
                });
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
                // De-box the dense `Vec<f64> result` directly into dense storage instead
                // of mapping it through `reduce_real_literal` into a boxed `Vec<Literal>`
                // (16 B/elem + enum tag, per-element construction). Bit-identical:
                // reduce_real_literal(F64, v) == F64Bits(v) and (F32, v) == F32Bits(v as
                // f32), exactly what new_f64_values / new_f32_values store. Keeps the
                // reduced output on the dense fast path so downstream ops (e.g. softmax's
                // divide after its sum) avoid re-boxing. BF16/F16 stay on the boxed map
                // (niche + already decode-dominated).
                match out_dtype {
                    DType::F64 => Ok(Value::Tensor(TensorValue::new_f64_values(
                        Shape { dims: out_dims },
                        result,
                    )?)),
                    DType::F32 => {
                        let f32s: Vec<f32> = result.iter().map(|&v| v as f32).collect();
                        Ok(Value::Tensor(TensorValue::new_f32_values(
                            Shape { dims: out_dims },
                            f32s,
                        )?))
                    }
                    DType::BF16 | DType::F16 => {
                        // Half-float reduce output: de-box to dense u16 storage. For a large
                        // output the boxed Literal construction dominates here too (the fold
                        // is cheap), so this is the same ~14x class — and bf16/f16 reductions
                        // are common in mixed-precision ML. Bits come straight from
                        // reduce_real_literal so the value is bit-identical by construction
                        // (from_bf16_f64/from_f16_f64), just stored densely.
                        let bits: Vec<u16> = result
                            .into_iter()
                            .map(|v| match reduce_real_literal(out_dtype, v) {
                                Literal::BF16Bits(b) | Literal::F16Bits(b) => b,
                                _ => unreachable!("half-float reduce produced non-half literal"),
                            })
                            .collect();
                        Ok(Value::Tensor(TensorValue::new_half_float_values(
                            out_dtype,
                            Shape { dims: out_dims },
                            bits,
                        )?))
                    }
                    _ => {
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
                        if let Some((words, len)) = tensor.elements.as_bool_words()
                            && let Some(acc) =
                                reduce_bool_words_scalar(primitive, words, len, bool_init)
                        {
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
                    // Contiguous-block fast path: when the reduced axes form one
                    // contiguous block, the layout factors as [outer, reduce, inner]
                    // and each output cell is a hoistable, autovectorizing fold over
                    // a contiguous run — no per-element odometer carry. Covers the
                    // dominant `any/all(mask, axis=-1)` (inner == 1) and
                    // `any/all(mask, axis=0)` (outer == 1) idioms. Bit-identical:
                    // same ascending-`r` accumulation order per cell as the odometer.
                    if let (Some(values), Some((outer, reduce, inner))) = (
                        tensor.elements.as_bool_slice(),
                        contiguous_reduce_block(&tensor.shape.dims, &axes_sorted),
                    ) {
                        if inner == 1 {
                            for (o, slot) in result.iter_mut().enumerate() {
                                let base = o * reduce;
                                let mut acc = bool_init;
                                for &v in &values[base..base + reduce] {
                                    acc = bool_op(acc, v);
                                }
                                *slot = acc;
                            }
                        } else {
                            for o in 0..outer {
                                let out_row = &mut result[o * inner..(o + 1) * inner];
                                for r in 0..reduce {
                                    let in_row = &values[(o * reduce + r) * inner..][..inner];
                                    for (slot, &v) in out_row.iter_mut().zip(in_row) {
                                        *slot = bool_op(*slot, v);
                                    }
                                }
                            }
                        }
                        // Emit dense 1-byte bool storage (not a boxed 24-byte
                        // Vec<Literal::Bool>) — bit-identical to the odometer
                        // fallback below, which builds the SAME `result: Vec<bool>`
                        // and wraps it via `new_bool_values`. This contiguous-block
                        // path is the HOT case (any/all(mask, axis=-1|0)); boxing
                        // its already-dense result wasted 24 B/elem + per-element
                        // TensorValue::new dtype validation.
                        return Ok(Value::Tensor(TensorValue::new_bool_values(
                            Shape { dims: out_dims },
                            result,
                        )?));
                    }

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
                    } else if let Some((words, len)) = tensor.elements.as_bool_words() {
                        for index in 0..len {
                            let out_idx = odometer.next_index();
                            result[out_idx] = bool_op(result[out_idx], bool_word_bit(words, index));
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

                    Ok(Value::Tensor(TensorValue::new_bool_values(
                        Shape { dims: out_dims },
                        result,
                    )?))
                }
                // I32 (JAX's default int) is dense i64-backed, so it shares the i64
                // reduce path: as_i64 reads the sign-extended slot, and bitwise
                // and/or/xor preserve the sign-extension invariant (the i64 result
                // is exactly the sign-extension of the i32 result for all three
                // ops), so new_i32_values stores the correct value. Matches the
                // non-bitwise integer reduce (which also routes I64|I32 here) and
                // the full-reduce scalar convention (scalar_i64 even for i32, as
                // sum/prod/max do — there is no i32 scalar repr). Previously i32
                // hit the `_` arm and erred despite the comment below claiming it
                // took this branch.
                DType::I64 | DType::I32 => {
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
                            tensor.dtype,
                            Shape { dims: out_dims },
                            vec![],
                        )?));
                    }

                    let kept_axes: Vec<usize> =
                        (0..rank).filter(|i| !axes_sorted.contains(i)).collect();
                    let mut result = try_filled_vec(
                        primitive,
                        "bitwise reduction integer accumulator",
                        out_count,
                        int_init,
                    )?;
                    // Drive an incremental out-index odometer (no per-element
                    // flat_to_multi decode); for dense i64 storage fold the
                    // contiguous i64 slice directly (no Literal materialization /
                    // per-element as_i64 match). Bit-identical to the prior
                    // flat_to_multi loop: same ascending flat order, same out_idx
                    // mapping (the odometer carries the kept-axis coordinate), same
                    // int_op. Mirrors the Bool path above. (i32 tensors are dense
                    // i64-backed, so they take the dense branch too.)
                    let mut odometer =
                        OutIndexOdometer::new(&tensor.shape.dims, &kept_axes, &out_dims);
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

                    // Emit dense integer storage (not a boxed 24-byte
                    // Vec<Literal::I64>) — the odometer above already built the
                    // dense `result`. I32 output is tagged via new_i32_values (the
                    // i64-backed result is the sign-extension of the i32 bitwise
                    // result); I64 via new_i64_values. Mirrors the Bool arm's
                    // new_bool_values wrap and the non-bitwise integer reduce's
                    // I64/I32 dtype split.
                    let shape = Shape { dims: out_dims };
                    let tv = if tensor.dtype == DType::I32 {
                        TensorValue::new_i32_values(shape, result)
                    } else {
                        TensorValue::new_i64_values(shape, result)
                    };
                    Ok(Value::Tensor(tv?))
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

/// Per-block forward/reverse cumulative scan for dense F32 lines, accumulating in
/// f64 and rounding each running value back to f32 — BIT-IDENTICAL to the serial
/// `acc = float_op(acc, f64::from(src)); out = acc as f32` loop (same f64
/// accumulator, never rounded mid-scan; same per-step `as f32`; same per-line
/// order). Each line is independent, so any contiguous block of whole lines may be
/// scanned in isolation.
fn scan_contiguous_f32_lines_from<F>(
    src: &[f32],
    out: &mut [f32],
    axis_dim: usize,
    reverse: bool,
    float_init: f64,
    float_op: &F,
) where
    F: Fn(f64, f64) -> f64 + ?Sized,
{
    for (src_line, out_line) in src.chunks(axis_dim).zip(out.chunks_mut(axis_dim)) {
        let mut acc = float_init;
        if reverse {
            for (sv, ov) in src_line.iter().zip(out_line.iter_mut()).rev() {
                acc = float_op(acc, f64::from(*sv));
                *ov = acc as f32;
            }
        } else {
            for (sv, ov) in src_line.iter().zip(out_line.iter_mut()) {
                acc = float_op(acc, f64::from(*sv));
                *ov = acc as f32;
            }
        }
    }
}

/// Dense F32 cumulative scan over contiguous independent lines, threaded over the
/// outer (line) dimension once total work crosses `CUMULATIVE_PARALLEL_MIN_ELEMS`.
/// Mirrors `scan_contiguous_lines_to_vec` but keeps the f64 accumulator (F32 cum
/// widens per element and rounds back per step, so the generic `T = f32` scanner
/// can't be reused). Lines are independent and each line's f64-accumulate order is
/// preserved inside its block, so the result is bit-identical to the serial scan
/// for ANY partition (incl. reverse and the one-line `threads == 1` case).
fn scan_contiguous_f32_lines_to_vec<F>(
    src: &[f32],
    axis_dim: usize,
    reverse: bool,
    float_init: f64,
    float_op: &F,
) -> Vec<f32>
where
    F: Fn(f64, f64) -> f64 + Sync,
{
    let outer = src.len() / axis_dim.max(1);
    let threads = if src.len() >= CUMULATIVE_PARALLEL_MIN_ELEMS && outer > 1 {
        crate::arithmetic::work_scaled_threads(src.len()).min(outer)
    } else {
        1
    };
    let mut out = vec![0.0f32; src.len()];
    if threads <= 1 {
        scan_contiguous_f32_lines_from(src, &mut out, axis_dim, reverse, float_init, float_op);
        return out;
    }
    let lines_per = outer.div_ceil(threads);
    let block = lines_per * axis_dim;
    std::thread::scope(|scope| {
        let mut src_rest = src;
        let mut out_rest: &mut [f32] = out.as_mut_slice();
        while !src_rest.is_empty() {
            let take = block.min(src_rest.len());
            let (src_block, src_tail) = src_rest.split_at(take);
            let (out_block, out_tail) = out_rest.split_at_mut(take);
            src_rest = src_tail;
            out_rest = out_tail;
            scope.spawn(move || {
                scan_contiguous_f32_lines_from(
                    src_block, out_block, axis_dim, reverse, float_init, float_op,
                );
            });
        }
    });
    out
}

/// Streaming leading-axis (`axis == 0`) cumulative scan of a `rows × cols` row-major
/// tensor: scan DOWN each column (the leading axis) keeping a per-column f64
/// accumulator (cols-wide, L1-resident) while reading/writing k-outer / column-inner
/// so every access is CONTIGUOUS. The serial strided path instead re-reads each
/// column at stride `cols` (cache-hostile — 3-4x off bandwidth). Each column folds
/// `k` in the SAME ascending (or reverse) order as that serial scan, so the result is
/// BIT-IDENTICAL. The f64 accumulator + per-step `narrow` matches both the F64
/// (`narrow` = identity) and F32 (`narrow` = `as f32`, never rounded mid-scan) serial
/// contracts.
#[allow(clippy::too_many_arguments)]
fn scan_leading_axis_to_vec<S: Copy, T: Copy>(
    src: &[S],
    rows: usize,
    cols: usize,
    reverse: bool,
    init: f64,
    widen: impl Fn(S) -> f64,
    narrow: impl Fn(f64) -> T,
    fill: T,
    op: impl Fn(f64, f64) -> f64,
) -> Vec<T> {
    let mut out = vec![fill; rows * cols];
    let mut acc = vec![init; cols];
    let scan_row = |k: usize, out: &mut [T], acc: &mut [f64]| {
        let base = k * cols;
        let srow = &src[base..base + cols];
        let orow = &mut out[base..base + cols];
        // Iterator zip elides per-element bounds checks (the non-aliasing acc / srow /
        // orow update across the independent columns is the hot inner loop).
        for ((a, &s), o) in acc.iter_mut().zip(srow.iter()).zip(orow.iter_mut()) {
            *a = op(*a, widen(s));
            *o = narrow(*a);
        }
    };
    if reverse {
        for k in (0..rows).rev() {
            scan_row(k, &mut out, &mut acc);
        }
    } else {
        for k in 0..rows {
            scan_row(k, &mut out, &mut acc);
        }
    }
    out
}

// Below this a single-line scan stays sequential — the 2-pass blocked overhead (init pass + two
// thread fan-outs) isn't worth it.
const CUMSUM_BLOCKED_MIN_ELEMS: usize = 1 << 20; // 1,048,576

// Parallel associative prefix-scan for ONE long f64 cummax/cummin line (forward). max/min are
// associative + commutative INCLUDING NaN (`jax_minmax_scalar` returns NaN if either operand is NaN —
// propagates identically under any grouping), so 2-pass chunking is BIT-IDENTICAL to the sequential fold.
// Calls `jax_minmax_scalar` DIRECTLY (inlines) at ALL cores — beats the generic op-closure blocked scan
// (non-inlined op + 16-thread cap) ~2.5x. Profiled 16M: 28ms vs JAX 68ms = ~2.4x WIN.
fn parallel_cummax_f64(src: &[f64], init: f64, is_max: bool, reverse: bool) -> Vec<f64> {
    let n = src.len();
    let cores = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(8);
    let threads = (n / (1 << 18)).clamp(2, cores);
    let block = n.div_ceil(threads);
    let mut out = vec![0.0f64; n];
    // Pass 1: per-chunk local DIRECTIONAL cummax; return each chunk's extremum (direction-independent).
    let ext: Vec<f64> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut s_rest = src;
        let mut o_rest: &mut [f64] = out.as_mut_slice();
        while !s_rest.is_empty() {
            let take = block.min(s_rest.len());
            let (s, st) = s_rest.split_at(take);
            let (o, ot) = o_rest.split_at_mut(take);
            s_rest = st;
            o_rest = ot;
            handles.push(scope.spawn(move || {
                let mut acc = init;
                if reverse {
                    for (slot, &v) in o.iter_mut().rev().zip(s.iter().rev()) {
                        acc = jax_minmax_scalar(acc, v, is_max);
                        *slot = acc;
                    }
                } else {
                    for (slot, &v) in o.iter_mut().zip(s) {
                        acc = jax_minmax_scalar(acc, v, is_max);
                        *slot = acc;
                    }
                }
                acc
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    // Pass 2: directional exclusive prefix of the chunk extrema -> per-chunk carry (forward = earlier
    // chunks, reverse = later chunks).
    let mut carries = vec![init; ext.len()];
    let mut acc = init;
    if reverse {
        for k in (0..ext.len()).rev() {
            carries[k] = acc;
            acc = jax_minmax_scalar(acc, ext[k], is_max);
        }
    } else {
        for (k, &e) in ext.iter().enumerate() {
            carries[k] = acc;
            acc = jax_minmax_scalar(acc, e, is_max);
        }
    }
    // Pass 3: fold each chunk's carry into its outputs (carry == init is the identity -> skip).
    std::thread::scope(|scope| {
        let mut o_rest: &mut [f64] = out.as_mut_slice();
        let mut k = 0usize;
        while !o_rest.is_empty() {
            let take = block.min(o_rest.len());
            let (o, ot) = o_rest.split_at_mut(take);
            o_rest = ot;
            let carry = carries[k];
            if carry != init {
                scope.spawn(move || {
                    for slot in o.iter_mut() {
                        *slot = jax_minmax_scalar(carry, *slot, is_max);
                    }
                });
            }
            k += 1;
        }
    });
    out
}

// f64 sibling of `parallel_assoc_scan_f32` for reverse f64 cumsum/cumprod single-chain (forward f64 cumsum
// keeps the shipped `blocked_prefix_scan_to_vec`). Same 3-pass rescan + directional carries; only the
// inter-chunk carry is reassociated (TOLERANCE-legal, same policy as the blocked scan).
fn parallel_assoc_scan_f64<F>(src: &[f64], init: f64, op: &F, reverse: bool) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64 + Sync,
{
    let n = src.len();
    let cores = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(8);
    let threads = (n / (1 << 18)).clamp(2, cores);
    let block = n.div_ceil(threads);
    let totals: Vec<f64> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut s_rest = src;
        while !s_rest.is_empty() {
            let take = block.min(s_rest.len());
            let (s, st) = s_rest.split_at(take);
            s_rest = st;
            handles.push(scope.spawn(move || {
                let mut acc = init;
                for &v in s {
                    acc = op(acc, v);
                }
                acc
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut carries = vec![init; totals.len()];
    let mut acc = init;
    if reverse {
        for k in (0..totals.len()).rev() {
            carries[k] = acc;
            acc = op(acc, totals[k]);
        }
    } else {
        for (k, &t) in totals.iter().enumerate() {
            carries[k] = acc;
            acc = op(acc, t);
        }
    }
    let mut out = vec![0.0f64; n];
    std::thread::scope(|scope| {
        let mut s_rest = src;
        let mut o_rest: &mut [f64] = out.as_mut_slice();
        let mut k = 0usize;
        while !s_rest.is_empty() {
            let take = block.min(s_rest.len());
            let (s, st) = s_rest.split_at(take);
            let (o, ot) = o_rest.split_at_mut(take);
            s_rest = st;
            o_rest = ot;
            let carry = carries[k];
            scope.spawn(move || {
                let mut acc = carry;
                if reverse {
                    for (slot, &v) in o.iter_mut().rev().zip(s.iter().rev()) {
                        acc = op(acc, v);
                        *slot = acc;
                    }
                } else {
                    for (slot, &v) in o.iter_mut().zip(s) {
                        acc = op(acc, v);
                        *slot = acc;
                    }
                }
            });
            k += 1;
        }
    });
    out
}

// f32 sibling of `parallel_cummax_f64` (f32 is JAX's default dtype). Matches the sequential f32 contract
// (`scan_contiguous_f32_lines_from`): accumulate in f64 (widen each element), store each step rounded to
// f32. BIT-IDENTICAL under chunking because max/min of f32 values is EXACT (the f64 running extremum is a
// widened f32; round-trip f32→f64→f32 is lossless), incl NaN propagation. Profiled 16M: 42→~27ms vs JAX 38.
fn parallel_cummax_f32(src: &[f32], init: f64, is_max: bool, reverse: bool) -> Vec<f32> {
    let n = src.len();
    let cores = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(8);
    let threads = (n / (1 << 18)).clamp(2, cores);
    let block = n.div_ceil(threads);
    let mut out = vec![0.0f32; n];
    // Pass 1: per-chunk local DIRECTIONAL cummax; return each chunk's extremum (direction-independent).
    let ext: Vec<f64> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut s_rest = src;
        let mut o_rest: &mut [f32] = out.as_mut_slice();
        while !s_rest.is_empty() {
            let take = block.min(s_rest.len());
            let (s, st) = s_rest.split_at(take);
            let (o, ot) = o_rest.split_at_mut(take);
            s_rest = st;
            o_rest = ot;
            handles.push(scope.spawn(move || {
                let mut acc = init;
                if reverse {
                    for (slot, &v) in o.iter_mut().rev().zip(s.iter().rev()) {
                        acc = jax_minmax_scalar(acc, f64::from(v), is_max);
                        *slot = acc as f32;
                    }
                } else {
                    for (slot, &v) in o.iter_mut().zip(s) {
                        acc = jax_minmax_scalar(acc, f64::from(v), is_max);
                        *slot = acc as f32;
                    }
                }
                acc
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    // Pass 2: directional exclusive prefix of the chunk extrema -> per-chunk carry. Forward = max of
    // EARLIER chunks; reverse = max of LATER chunks.
    let mut carries = vec![init; ext.len()];
    let mut acc = init;
    if reverse {
        for k in (0..ext.len()).rev() {
            carries[k] = acc;
            acc = jax_minmax_scalar(acc, ext[k], is_max);
        }
    } else {
        for (k, &e) in ext.iter().enumerate() {
            carries[k] = acc;
            acc = jax_minmax_scalar(acc, e, is_max);
        }
    }
    // Pass 3: fold each chunk's carry into its (already locally-scanned) outputs. carry == init is the op
    // identity -> a no-op, so skip it (the boundary chunk: first for forward, last for reverse).
    std::thread::scope(|scope| {
        let mut o_rest: &mut [f32] = out.as_mut_slice();
        let mut k = 0usize;
        while !o_rest.is_empty() {
            let take = block.min(o_rest.len());
            let (o, ot) = o_rest.split_at_mut(take);
            o_rest = ot;
            let carry = carries[k];
            if carry != init {
                scope.spawn(move || {
                    for slot in o.iter_mut() {
                        *slot = jax_minmax_scalar(carry, f64::from(*slot), is_max) as f32;
                    }
                });
            }
            k += 1;
        }
    });
    out
}

// Parallel prefix scan for ONE long f32 cumsum/cumprod line (forward). cumsum/cumprod are NON-associative,
// but their reassociation is TOLERANCE-legal — the SAME accepted policy as the f64 `blocked_prefix_scan_to_vec`
// (the cumsum/cumprod oracle is abs<1e-10, not bit-exact). 3-pass rescan keeps a full f64 accumulator per
// chunk (single round per output, like the sequential f32 contract): pass1 per-chunk op-reduction, pass2
// exclusive op-prefix of the chunk totals (the carries), pass3 re-scan each chunk from its carry. Only the
// inter-chunk carry is reassociated (chunk-grouped vs element-sequential) — identical structure to the f64
// blocked scan. Profiled 16M: 49.8→~15ms vs JAX 39ms.
fn parallel_assoc_scan_f32<F>(src: &[f32], init: f64, op: &F, reverse: bool) -> Vec<f32>
where
    F: Fn(f64, f64) -> f64 + Sync,
{
    let n = src.len();
    let cores = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(8);
    let threads = (n / (1 << 18)).clamp(2, cores);
    let block = n.div_ceil(threads);
    // Pass 1: per-chunk op-reduction from init (the chunk's total — direction-independent), in parallel.
    let totals: Vec<f64> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut s_rest = src;
        while !s_rest.is_empty() {
            let take = block.min(s_rest.len());
            let (s, st) = s_rest.split_at(take);
            s_rest = st;
            handles.push(scope.spawn(move || {
                let mut acc = init;
                for &v in s {
                    acc = op(acc, f64::from(v));
                }
                acc
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    // Pass 2: directional exclusive op-prefix of the chunk totals -> per-chunk carry. Forward = op of
    // EARLIER chunks; reverse = op of LATER chunks.
    let mut carries = vec![init; totals.len()];
    let mut acc = init;
    if reverse {
        for k in (0..totals.len()).rev() {
            carries[k] = acc;
            acc = op(acc, totals[k]);
        }
    } else {
        for (k, &t) in totals.iter().enumerate() {
            carries[k] = acc;
            acc = op(acc, t);
        }
    }
    // Pass 3: re-scan each chunk DIRECTIONALLY from its carry (full f64 acc, single round per output).
    let mut out = vec![0.0f32; n];
    std::thread::scope(|scope| {
        let mut s_rest = src;
        let mut o_rest: &mut [f32] = out.as_mut_slice();
        let mut k = 0usize;
        while !s_rest.is_empty() {
            let take = block.min(s_rest.len());
            let (s, st) = s_rest.split_at(take);
            let (o, ot) = o_rest.split_at_mut(take);
            s_rest = st;
            o_rest = ot;
            let carry = carries[k];
            scope.spawn(move || {
                let mut acc = carry;
                if reverse {
                    for (slot, &v) in o.iter_mut().rev().zip(s.iter().rev()) {
                        acc = op(acc, f64::from(v));
                        *slot = acc as f32;
                    }
                } else {
                    for (slot, &v) in o.iter_mut().zip(s) {
                        acc = op(acc, f64::from(v));
                        *slot = acc as f32;
                    }
                }
            });
            k += 1;
        }
    });
    out
}

// Blocked parallel prefix-scan for ONE long contiguous f64 line (outer_count == 1, forward).
// The sequential single-line scan is f64 dependency-chain-bound (~30ms at 4M; JAX reassociates to
// ~14ms). Since the cumsum/cumprod oracle is TOLERANCE (abs<1e-10, NOT bit-exact) and `init` is the
// op identity, split into per-thread blocks: (A) each block scans locally from `init` in parallel,
// (B) exclusive-prefix the block totals, (C) each block applies its offset via `op` in parallel.
// Reassociation here uses SHORTER chains than the single long scan, so it is at least as accurate
// (more JAX-faithful — JAX reassociates too); for associative max/min/int ops it is bit-exact.
fn blocked_prefix_scan_to_vec<F>(src: &[f64], init: f64, op: F) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64 + Sync,
{
    let n = src.len();
    let cores = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(8);
    let threads = (n / (1 << 18)).clamp(2, cores.min(16));
    let block = n.div_ceil(threads);
    let mut out = vec![init; n];
    let op_ref = &op;
    // Pass A: parallel local scans from `init`.
    std::thread::scope(|scope| {
        let mut s_rest = src;
        let mut o_rest: &mut [f64] = out.as_mut_slice();
        while !s_rest.is_empty() {
            let take = block.min(s_rest.len());
            let (s, st) = s_rest.split_at(take);
            let (o, ot) = o_rest.split_at_mut(take);
            s_rest = st;
            o_rest = ot;
            scope.spawn(move || {
                let mut acc = init;
                for (slot, &v) in o.iter_mut().zip(s) {
                    acc = op_ref(acc, v);
                    *slot = acc;
                }
            });
        }
    });
    // Exclusive prefix of block totals (each block total = its last local-scan element).
    let mut offsets: Vec<f64> = Vec::with_capacity(n.div_ceil(block));
    let mut acc = init;
    let mut idx = 0;
    while idx < n {
        offsets.push(acc);
        let last = (idx + block).min(n) - 1;
        acc = op_ref(acc, out[last]);
        idx += block;
    }
    // Pass B: parallel apply each block's offset (block 0's offset == init == identity no-op).
    std::thread::scope(|scope| {
        let mut o_rest: &mut [f64] = out.as_mut_slice();
        let mut b = 0;
        while !o_rest.is_empty() {
            let take = block.min(o_rest.len());
            let (o, ot) = o_rest.split_at_mut(take);
            o_rest = ot;
            let offset = offsets[b];
            b += 1;
            scope.spawn(move || {
                for slot in o.iter_mut() {
                    *slot = op_ref(offset, *slot);
                }
            });
        }
    });
    out
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn eval_cumulative_dense(
    tensor: &TensorValue,
    cum_primitive: Primitive,
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
            && (!reverse || outer_count > 1 || total >= CUMSUM_BLOCKED_MIN_ELEMS)
        {
            // Large contiguous scans write dense output directly from source slices.
            if outer_count == 1 && total >= CUMSUM_BLOCKED_MIN_ELEMS {
                // Single long line: the per-line scan is single-thread + dependency-bound. cummax/cummin
                // are ASSOCIATIVE (incl NaN) → a dedicated all-cores parallel prefix scan with DIRECT
                // (inlined) jax_minmax_scalar is BIT-IDENTICAL, forward OR reverse. cumsum/cumprod are
                // non-associative but TOLERANCE-legal: forward keeps the shipped blocked scan; reverse uses
                // the parallel rescan. NOTE: match on `cum_primitive` — `primitive` is shadowed to Cumsum.
                if matches!(cum_primitive, Primitive::Cummax | Primitive::Cummin) {
                    parallel_cummax_f64(
                        src,
                        float_init,
                        matches!(cum_primitive, Primitive::Cummax),
                        reverse,
                    )
                } else if reverse {
                    parallel_assoc_scan_f64(src, float_init, float_op, true)
                } else {
                    blocked_prefix_scan_to_vec(src, float_init, float_op)
                }
            } else {
                scan_contiguous_lines_to_vec(src, axis_dim, reverse, float_init, float_op)
            }
        } else if axis_stride == 1 {
            let mut out = src.to_vec();
            scan_contiguous_lines_in_place(&mut out, axis_dim, reverse, float_init, float_op);
            out
        } else if axis == 0 && outer_count > 1 {
            // Leading-axis (cumsum/cumprod/cummax/cummin DOWN the columns): the serial branch
            // below re-reads each column at stride `axis_stride == outer_count`
            // (cache-hostile). Stream k-outer/column-inner instead (contiguous
            // reads+writes, per-column f64 acc in L1) — bit-identical.
            scan_leading_axis_to_vec(
                src,
                axis_dim,
                outer_count,
                reverse,
                float_init,
                |x| x,
                |a| a,
                float_init,
                float_op,
            )
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
        // same per-line order. The scan is a sequential dependency WITHIN a line
        // (acc feeds the next step), so it CANNOT reassociate/vectorize — but the
        // lines are INDEPENDENT, so contiguous (`axis_stride == 1`) large inputs
        // thread the scan over the outer/line dimension bit-identically. Strided
        // inputs keep the serial per-line gather below.
        let out = if axis_stride == 1 {
            // Single long f32 line: scan_contiguous_f32_lines_to_vec stays sequential (it only threads
            // when outer>1). cummax/cummin are associative (max/min of f32 is exact → bit-identical
            // parallel scan). cumsum/cumprod are non-associative but their reassociation is TOLERANCE-legal
            // (same accepted policy as the f64 blocked_prefix_scan_to_vec) → parallel rescan.
            if outer_count == 1 && total >= CUMSUM_BLOCKED_MIN_ELEMS {
                if matches!(cum_primitive, Primitive::Cummax | Primitive::Cummin) {
                    parallel_cummax_f32(
                        src,
                        float_init,
                        matches!(cum_primitive, Primitive::Cummax),
                        reverse,
                    )
                } else {
                    parallel_assoc_scan_f32(src, float_init, float_op, reverse)
                }
            } else {
                scan_contiguous_f32_lines_to_vec(src, axis_dim, reverse, float_init, float_op)
            }
        } else if axis == 0 && outer_count > 1 {
            // Leading-axis f32 (cumsum DOWN the columns): stream k-outer/column-inner
            // (contiguous), f64 acc rounded to f32 per step — bit-identical to the
            // serial strided scan below.
            scan_leading_axis_to_vec(
                src,
                axis_dim,
                outer_count,
                reverse,
                float_init,
                f64::from,
                |a| a as f32,
                0.0f32,
                float_op,
            )
        } else {
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
            out
        };
        return Ok(Some(Value::Tensor(TensorValue::new_f32_values(
            tensor.shape.clone(),
            out,
        )?)));
    }

    // I64 AND I32 (JAX's default int) share the dense i64 backing. i32 is sign-extended,
    // so `int_op` (wrapping) over i64 is correct; cumsum/cumprod prefixes may exceed i32
    // range but the I32-tagged output is wrapped mod 2^32 by the eval_primitive chokepoint
    // (homomorphism: wrap-of-i64-prefix == per-step i32 wrap), and cummax/cummin select an
    // in-range input so no wrap occurs. i32 otherwise ran the generic per-Literal scan.
    if matches!(tensor.dtype, DType::I64 | DType::I32) {
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
        let shape = tensor.shape.clone();
        let tv = if tensor.dtype == DType::I32 {
            TensorValue::new_i32_values(shape, out)
        } else {
            TensorValue::new_i64_values(shape, out)
        };
        return Ok(Some(Value::Tensor(tv?)));
    }

    if matches!(tensor.dtype, DType::BF16 | DType::F16) {
        let Some(src) = tensor.elements.as_half_float_slice() else {
            return Ok(None);
        };
        // Dense half-float cumulative: scan each line reading the raw u16 backing,
        // accumulating in f64 (widen INLINE per element via the SAME `as_f64`
        // conversion the generic per-`Literal` path uses), storing each step's
        // running value rounded back to the half dtype via the SAME
        // `reduce_real_literal` rounding. BIT-IDENTICAL to the generic float scan
        // (lib path: `acc = float_op(acc, lit.as_f64()); store reduce_real_literal
        // (dtype, acc)`) — same f64 accumulator (never rounded mid-scan), same
        // per-step round, same per-line order. The scan is a sequential dependency
        // (acc feeds the next step), so it CANNOT reassociate/vectorize — exact
        // incl. NaN. Mirrors the dense F32 path; the win is reading the 2-byte u16
        // backing + emitting dense half storage instead of materializing the
        // 24-byte `Literal` per element on both input and output.
        let dt = tensor.dtype;
        let widen = |u: u16| -> f64 {
            match dt {
                DType::BF16 => Literal::BF16Bits(u).as_f64(),
                _ => Literal::F16Bits(u).as_f64(),
            }
            .unwrap_or(0.0)
        };
        let round = |acc: f64| -> u16 {
            match reduce_real_literal(dt, acc) {
                Literal::BF16Bits(b) | Literal::F16Bits(b) => b,
                _ => 0,
            }
        };
        let mut out = vec![0u16; total];
        // One contiguous line's scan (sequential acc dependency WITHIN the line; lines independent).
        let scan_line = |src_line: &[u16], out_line: &mut [u16]| {
            let mut acc = float_init;
            if reverse {
                for i in (0..axis_dim).rev() {
                    acc = float_op(acc, widen(src_line[i]));
                    out_line[i] = round(acc);
                }
            } else {
                for i in 0..axis_dim {
                    acc = float_op(acc, widen(src_line[i]));
                    out_line[i] = round(acc);
                }
            }
        };
        if axis_stride == 1 {
            // Trailing/contiguous axis: lines are out[outer*axis_dim..]; thread over the independent
            // lines (bit-identical — each line's scan order is unchanged). Matches the f64/i64 path.
            let threads = if total >= CUMULATIVE_PARALLEL_MIN_ELEMS && outer_count > 1 {
                crate::arithmetic::work_scaled_threads(total).min(outer_count)
            } else {
                1
            };
            if threads <= 1 {
                for (sl, ol) in src
                    .chunks_exact(axis_dim)
                    .zip(out.chunks_exact_mut(axis_dim))
                {
                    scan_line(sl, ol);
                }
            } else {
                let per = outer_count.div_ceil(threads);
                std::thread::scope(|scope| {
                    let mut rest: &mut [u16] = out.as_mut_slice();
                    let mut o0 = 0usize;
                    while o0 < outer_count {
                        let cnt = per.min(outer_count - o0);
                        let (blk, tail) = rest.split_at_mut(cnt * axis_dim);
                        rest = tail;
                        let s = &src[o0 * axis_dim..(o0 + cnt) * axis_dim];
                        let f = &scan_line;
                        o0 += cnt;
                        scope.spawn(move || {
                            for (sl, ol) in
                                s.chunks_exact(axis_dim).zip(blk.chunks_exact_mut(axis_dim))
                            {
                                f(sl, ol);
                            }
                        });
                    }
                });
            }
        } else {
            for outer in 0..outer_count {
                let base = line_base(outer);
                let mut acc = float_init;
                if reverse {
                    for i in (0..axis_dim).rev() {
                        let fi = base + i * axis_stride;
                        acc = float_op(acc, widen(src[fi]));
                        out[fi] = round(acc);
                    }
                } else {
                    for i in 0..axis_dim {
                        let fi = base + i * axis_stride;
                        acc = float_op(acc, widen(src[fi]));
                        out[fi] = round(acc);
                    }
                }
            }
        }
        return Ok(Some(Value::Tensor(TensorValue::new_half_float_values(
            dt,
            tensor.shape.clone(),
            out,
        )?)));
    }

    // Dense complex cumulative (cumsum/cumprod/cummax/cummin). Reads the contiguous
    // (re, im) backing and scans each line, BIT-IDENTICAL to the generic per-`Literal`
    // complex path: same seeds (cumprod=(1,0), cummax/cummin=(float_init,float_init),
    // cumsum=(0,0)), same complex-multiply / lexicographic complex_lex_cmp / component
    // add in the same per-line order, and `new_complex_values` rounds Complex64 to f32
    // exactly as `complex_literal_from_parts` does (== the generic store). Reading
    // `out[fi]` is the original input: each position is read once before its own write.
    if matches!(tensor.dtype, DType::Complex64 | DType::Complex128) {
        if !matches!(
            cum_primitive,
            Primitive::Cumsum | Primitive::Cumprod | Primitive::Cummax | Primitive::Cummin
        ) {
            return Ok(None);
        }
        let Some(src) = tensor.elements.as_complex_slice() else {
            return Ok(None);
        };
        let mut out: Vec<(f64, f64)> = src.to_vec();
        // Single-thread: complex cumsum is MEMORY-bound (16-byte (f64,f64), cheap add), so threading
        // REGRESSES (measured 0.95x at [512,4096]) — overhead with no BW headroom, unlike the
        // compute-bound half-float scan (4.47x). Kept serial (see ledger 2026-06-25).
        for outer in 0..outer_count {
            let base = line_base(outer);
            let (mut acc_re, mut acc_im) = match cum_primitive {
                Primitive::Cumprod => (1.0_f64, 0.0_f64),
                Primitive::Cummax | Primitive::Cummin => (float_init, float_init),
                _ => (0.0_f64, 0.0_f64),
            };
            let mut step = |fi: usize, out: &mut [(f64, f64)]| {
                let (re, im) = out[fi];
                match cum_primitive {
                    Primitive::Cumprod => {
                        let nr = acc_re * re - acc_im * im;
                        let ni = acc_re * im + acc_im * re;
                        acc_re = nr;
                        acc_im = ni;
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
                out[fi] = (acc_re, acc_im);
            };
            if reverse {
                for i in (0..axis_dim).rev() {
                    step(base + i * axis_stride, &mut out);
                }
            } else {
                for i in 0..axis_dim {
                    step(base + i * axis_stride, &mut out);
                }
            }
        }
        return Ok(Some(Value::Tensor(TensorValue::new_complex_values(
            tensor.dtype,
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

            // Unsigned (u32/u64) cumulative scans: same bug + fix as eval_reduce_axes
            // (bead 2iotb). u32/u64 used to fall to the float arm (F64 output, no wrap).
            // Map unsigned→i64 with the order/ring-faithful transform, reuse the tested
            // i64 cumulative machinery (same-shape output), then map each element back.
            // cumsum/cumprod wrap mod 2^N (per-step == map-back per element, ring
            // homomorphism); cummax/cummin use the sign-flip so signed cmp == unsigned.
            if matches!(tensor.dtype, DType::U32 | DType::U64)
                && matches!(
                    primitive,
                    Primitive::Cumsum | Primitive::Cumprod | Primitive::Cummax | Primitive::Cummin
                )
            {
                const FLIP: u64 = 1u64 << 63;
                let is_u32 = tensor.dtype == DType::U32;
                let is_minmax = matches!(primitive, Primitive::Cummax | Primitive::Cummin);
                let fwd = |v: u64| -> i64 {
                    if !is_u32 && is_minmax {
                        (v ^ FLIP) as i64
                    } else {
                        v as i64
                    }
                };
                let inv = |v: i64| -> u64 {
                    if is_u32 {
                        u64::from(v as u32)
                    } else if is_minmax {
                        (v as u64) ^ FLIP
                    } else {
                        v as u64
                    }
                };
                let widened: Vec<i64> = match tensor.elements.as_u32_slice() {
                    Some(s) => s.iter().map(|&v| fwd(u64::from(v))).collect(),
                    None => match tensor.elements.as_u64_slice() {
                        Some(s) => s.iter().map(|&v| fwd(v)).collect(),
                        None => tensor
                            .elements
                            .iter()
                            .map(|l| {
                                l.as_u64().map(fwd).ok_or(EvalError::TypeMismatch {
                                    primitive,
                                    detail: "expected unsigned tensor",
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    },
                };
                let widened_input = vec![Value::Tensor(TensorValue::new_i64_values(
                    tensor.shape.clone(),
                    widened,
                )?)];
                let result = eval_cumulative(
                    primitive,
                    &widened_input,
                    params,
                    int_init,
                    float_init,
                    int_op,
                    float_op,
                )?;
                let Value::Tensor(rt) = result else {
                    return Ok(result);
                };
                let vals: Vec<u64> = match rt.elements.as_i64_slice() {
                    Some(s) => s.iter().map(|&v| inv(v)).collect(),
                    None => rt
                        .elements
                        .iter()
                        .map(|l| inv(l.as_i64().unwrap_or(0)))
                        .collect(),
                };
                return Ok(Value::Tensor(if is_u32 {
                    TensorValue::new_u32_values(
                        rt.shape.clone(),
                        vals.iter().map(|&v| v as u32).collect(),
                    )?
                } else {
                    TensorValue::new_u64_values(rt.shape.clone(), vals)?
                }));
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
                tensor, primitive, axis, reverse, int_init, float_init, &int_op, &float_op,
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
    // Test-only `[(Primitive, f64, fn(f64, f64) -> f64); N]` case tables.
    #![allow(clippy::type_complexity)]
    use super::*;
    use fj_core::LiteralBuffer;
    use std::collections::BTreeMap;

    // PROFILE the cummax 29<->73 puzzle (bead frankenjax-parallel-cummax-scan): time the 2-pass parallel
    // associative scan DIRECTLY on the tensor's f64 slice vs the full eval_primitive(Cummax) path, on the
    // SAME 16M data, to localize the ~44ms overhead (scan itself vs eval pipeline).
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_cummax_profile_scan_vs_eval() {
        use std::time::Instant;
        let n = 16_000_000usize;
        let data: Vec<f64> = (0..n)
            .map(|i| ((i.wrapping_mul(2654435761) % 100003) as f64) * 0.01 - 500.0)
            .collect();
        let x = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![n as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        let slice: &[f64] = match &x {
            Value::Tensor(t) => t.elements.as_f64_slice().unwrap(),
            _ => unreachable!(),
        };
        let init = f64::NEG_INFINITY;
        let par = |src: &[f64]| -> Vec<f64> {
            let cores = std::thread::available_parallelism()
                .map(|c| c.get())
                .unwrap_or(8);
            let threads = (src.len() / (1 << 18)).clamp(2, cores);
            let block = src.len().div_ceil(threads);
            let mut out = vec![0.0f64; src.len()];
            let ext: Vec<f64> = std::thread::scope(|s| {
                let mut hs = Vec::new();
                let mut sr = src;
                let mut orr: &mut [f64] = out.as_mut_slice();
                while !sr.is_empty() {
                    let take = block.min(sr.len());
                    let (sc, st) = sr.split_at(take);
                    let (oc, ot) = orr.split_at_mut(take);
                    sr = st;
                    orr = ot;
                    hs.push(s.spawn(move || {
                        let mut acc = init;
                        for (o, &v) in oc.iter_mut().zip(sc) {
                            acc = super::jax_minmax_scalar(acc, v, true);
                            *o = acc;
                        }
                        acc
                    }));
                }
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let mut carries = vec![init; ext.len()];
            let mut acc = init;
            for (k, &e) in ext.iter().enumerate() {
                carries[k] = acc;
                acc = super::jax_minmax_scalar(acc, e, true);
            }
            std::thread::scope(|s| {
                let mut orr: &mut [f64] = out.as_mut_slice();
                let mut k = 0usize;
                while !orr.is_empty() {
                    let take = block.min(orr.len());
                    let (oc, ot) = orr.split_at_mut(take);
                    orr = ot;
                    let carry = carries[k];
                    if k > 0 {
                        s.spawn(move || {
                            for o in oc.iter_mut() {
                                *o = super::jax_minmax_scalar(carry, *o, true);
                            }
                        });
                    }
                    k += 1;
                }
            });
            out
        };
        let bench = |label: &str, f: &dyn Fn()| {
            f();
            let mut b = f64::MAX;
            for _ in 0..6 {
                let t = Instant::now();
                f();
                b = b.min(t.elapsed().as_secs_f64());
            }
            println!("cummax-profile {label}: {:.3}ms", b * 1e3);
        };
        bench("par-scan-direct-on-tensor-slice", &|| {
            std::hint::black_box(par(slice));
        });
        let p = BTreeMap::from([("axis".to_owned(), "0".to_owned())]);
        bench("eval_primitive(Cummax)", &|| {
            std::hint::black_box(
                crate::eval_primitive(Primitive::Cummax, std::slice::from_ref(&x), &p).unwrap(),
            );
        });
    }

    // BOLD-VERIFY: 1-D cummax/cummin vs JAX (16M f64, measured JAX lax.cummax 68.1ms / cummin 72.1ms).
    // max/min are associative (a parallel scan would be bit-identical) — but JAX doesn't fast-scan them
    // on CPU, so fj-lax's sequential fold should still win.
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_cummax1d_vs_jax() {
        use std::time::Instant;
        let n = 16_000_000usize;
        let data: Vec<f64> = (0..n)
            .map(|i| ((i.wrapping_mul(2654435761) % 100003) as f64) * 0.01 - 500.0)
            .collect();
        let x = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![n as u32],
                },
                data,
            )
            .unwrap(),
        );
        let p = BTreeMap::from([("axis".to_owned(), "0".to_owned())]);
        let bench = |label: &str, prim: Primitive| {
            let f = || {
                std::hint::black_box(
                    crate::eval_primitive(prim, std::slice::from_ref(&x), &p).unwrap(),
                );
            };
            f();
            let mut b = f64::MAX;
            for _ in 0..6 {
                let s = Instant::now();
                f();
                b = b.min(s.elapsed().as_secs_f64());
            }
            println!("fj-lax {label} f64 16M: {:.3}ms", b * 1e3);
        };
        bench("cummax1d", Primitive::Cummax);
        bench("cummin1d", Primitive::Cummin);
    }

    // f32 is JAX's DEFAULT dtype, so f32 cummax/cummin is the common case (measured JAX 38.1ms each). The
    // f32 cumulative path is separate (widen-to-f64 accumulate + round); check whether it needs the same
    // parallel associative scan.
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_cummax1d_f32_vs_jax() {
        use std::time::Instant;
        let n = 16_000_000usize;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i.wrapping_mul(2654435761) % 100003) as f32) * 0.01 - 500.0)
            .collect();
        let x = Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: vec![n as u32],
                },
                data,
            )
            .unwrap(),
        );
        let p = BTreeMap::from([("axis".to_owned(), "0".to_owned())]);
        let bench = |label: &str, prim: Primitive| {
            let f = || {
                std::hint::black_box(
                    crate::eval_primitive(prim, std::slice::from_ref(&x), &p).unwrap(),
                );
            };
            f();
            let mut b = f64::MAX;
            for _ in 0..6 {
                let s = Instant::now();
                f();
                b = b.min(s.elapsed().as_secs_f64());
            }
            println!("fj-lax {label} f32 16M: {:.3}ms | JAX=38.1ms", b * 1e3);
        };
        bench("cummax1d", Primitive::Cummax);
        bench("cummin1d", Primitive::Cummin);
    }

    // 2-D cummax along each axis (measured JAX f32 [4096,4096]: axis0=49.3ms, axis1=31.0ms). axis1 (rows)
    // threads bit-identically; axis0 (columns) uses the streaming leading-axis path — check if it needs
    // column-parallelism.
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_cummax2d_vs_jax() {
        use std::time::Instant;
        let (rows, cols) = (4096usize, 4096usize);
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i.wrapping_mul(2654435761) % 100003) as f32) * 0.01 - 500.0)
            .collect();
        let x = Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data,
            )
            .unwrap(),
        );
        let bench = |axis: &str, jax: f64| {
            let p = BTreeMap::from([("axis".to_owned(), axis.to_owned())]);
            let f = || {
                std::hint::black_box(
                    crate::eval_primitive(Primitive::Cummax, std::slice::from_ref(&x), &p).unwrap(),
                );
            };
            f();
            let mut b = f64::MAX;
            for _ in 0..6 {
                let s = Instant::now();
                f();
                b = b.min(s.elapsed().as_secs_f64());
            }
            println!(
                "fj-lax cummax f32 [4096,4096] axis{axis}: {:.3}ms | JAX={jax}ms",
                b * 1e3
            );
        };
        bench("0", 49.3);
        bench("1", 31.0);
    }

    // BOLD-VERIFY: 1-D cumsum/cumprod vs JAX (16M f64, measured JAX cumsum 95.0ms / cumprod 94.2ms — XLA
    // has no fast single-chain scan). fj-lax does a sequential latency-bound fold; this checks the ratio.
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_cumsum1d_vs_jax() {
        use std::time::Instant;
        let n = 16_000_000usize;
        let data: Vec<f64> = (0..n).map(|i| 0.5 + (i % 9973) as f64 * 1e-5).collect();
        let x = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![n as u32],
                },
                data,
            )
            .unwrap(),
        );
        let p = BTreeMap::from([("axis".to_owned(), "0".to_owned())]);
        let bench = |label: &str, prim: Primitive| {
            let f = || {
                std::hint::black_box(
                    crate::eval_primitive(prim, std::slice::from_ref(&x), &p).unwrap(),
                );
            };
            f();
            let mut b = f64::MAX;
            for _ in 0..6 {
                let s = Instant::now();
                f();
                b = b.min(s.elapsed().as_secs_f64());
            }
            println!("fj-lax {label} f64 16M: {:.3}ms", b * 1e3);
        };
        bench("cumsum1d", Primitive::Cumsum);
        bench("cumprod1d", Primitive::Cumprod);
    }

    // Parity gate for the parallel f32 cumsum reassociation (parallel_assoc_scan_f32): at 16M (>= the
    // blocked threshold) the parallel scan must stay within tolerance of the sequential f64-accumulate
    // reference. The reassociation is only inter-chunk (chunk-grouped sums) — same structure/policy as the
    // shipped f64 blocked scan. Asserts max relative error is tiny (well under any cumsum oracle tolerance).
    #[test]
    fn parallel_f32_cumsum_within_tolerance() {
        let n = 4_000_000usize; // > CUMSUM_BLOCKED_MIN_ELEMS (1<<20) -> exercises the parallel path
        let data: Vec<f32> = (0..n).map(|i| 0.25 + (i % 9973) as f32 * 1e-5).collect();
        let x = Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: vec![n as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        let p = BTreeMap::from([("axis".to_owned(), "0".to_owned())]);
        let Value::Tensor(out) =
            crate::eval_primitive(Primitive::Cumsum, std::slice::from_ref(&x), &p).unwrap()
        else {
            panic!("expected tensor");
        };
        let got = out.elements.as_f32_slice().expect("dense f32");
        // Sequential f64-accumulate reference (the contract scan_contiguous_f32_lines_from implements).
        let mut acc = 0.0f64;
        let mut max_rel = 0.0f64;
        for (i, &v) in data.iter().enumerate() {
            acc += f64::from(v);
            let reference = acc as f32;
            let g = got[i];
            let denom = (f64::from(reference)).abs().max(1.0);
            max_rel = max_rel.max((f64::from(g) - f64::from(reference)).abs() / denom);
        }
        assert!(
            max_rel < 1e-5,
            "parallel f32 cumsum reassociation drifted: max relative error {max_rel:e}"
        );
    }

    // Parity gate for the REVERSE parallel f32 scans (parallel_cummax_f32 / parallel_assoc_scan_f32 with
    // reverse=true), at 4M (> the blocked threshold). Reverse cummax is associative → must be BIT-IDENTICAL
    // to the sequential reverse cummax; reverse cumsum is tolerance-legal → within 1e-5 of the sequential
    // reverse f64-accumulate reference.
    #[test]
    fn parallel_f32_reverse_scans_match_sequential() {
        let n = 4_000_000usize;
        let data: Vec<f32> = (0..n).map(|i| ((i % 9973) as f32) * 1e-3 - 5.0).collect();
        let x = Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: vec![n as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        let p = BTreeMap::from([
            ("axis".to_owned(), "0".to_owned()),
            ("reverse".to_owned(), "true".to_owned()),
        ]);
        // Reverse cummax: bit-identical (max of f32 is exact under any grouping).
        let Value::Tensor(cmax) =
            crate::eval_primitive(Primitive::Cummax, std::slice::from_ref(&x), &p).unwrap()
        else {
            panic!("tensor");
        };
        let got_max = cmax.elements.as_f32_slice().expect("dense f32");
        let mut ref_max = vec![0.0f32; n];
        let mut acc = f64::NEG_INFINITY;
        for i in (0..n).rev() {
            acc = jax_minmax_scalar(acc, f64::from(data[i]), true);
            ref_max[i] = acc as f32;
        }
        let max_mismatches = got_max
            .iter()
            .zip(&ref_max)
            .filter(|(g, r)| g.to_bits() != r.to_bits())
            .count();
        assert_eq!(max_mismatches, 0, "reverse cummax not bit-identical");
        // Reverse cumsum: tolerance.
        let Value::Tensor(csum) =
            crate::eval_primitive(Primitive::Cumsum, std::slice::from_ref(&x), &p).unwrap()
        else {
            panic!("tensor");
        };
        let got_sum = csum.elements.as_f32_slice().expect("dense f32");
        let mut acc = 0.0f64;
        let mut max_rel = 0.0f64;
        for i in (0..n).rev() {
            acc += f64::from(data[i]);
            let reference = acc as f32;
            let denom = f64::from(reference).abs().max(1.0);
            max_rel = max_rel.max((f64::from(got_sum[i]) - f64::from(reference)).abs() / denom);
        }
        assert!(
            max_rel < 1e-5,
            "reverse cumsum drifted: max relative error {max_rel:e}"
        );
    }

    // f64 reverse parity gate (parallel_cummax_f64 / parallel_assoc_scan_f64 with reverse): reverse cummax
    // bit-identical to the sequential reverse scan; reverse cumsum within 1e-5.
    #[test]
    fn parallel_f64_reverse_scans_match_sequential() {
        let n = 4_000_000usize;
        let data: Vec<f64> = (0..n).map(|i| ((i % 9973) as f64) * 1e-3 - 5.0).collect();
        let x = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![n as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        let p = BTreeMap::from([
            ("axis".to_owned(), "0".to_owned()),
            ("reverse".to_owned(), "true".to_owned()),
        ]);
        let Value::Tensor(cmax) =
            crate::eval_primitive(Primitive::Cummax, std::slice::from_ref(&x), &p).unwrap()
        else {
            panic!("tensor");
        };
        let got_max = cmax.elements.as_f64_slice().expect("dense f64");
        let mut acc = f64::NEG_INFINITY;
        let mut max_mismatches = 0usize;
        let mut ref_rev = vec![0.0f64; n];
        for i in (0..n).rev() {
            acc = jax_minmax_scalar(acc, data[i], true);
            ref_rev[i] = acc;
        }
        for (g, r) in got_max.iter().zip(&ref_rev) {
            if g.to_bits() != r.to_bits() {
                max_mismatches += 1;
            }
        }
        assert_eq!(max_mismatches, 0, "f64 reverse cummax not bit-identical");
        let Value::Tensor(csum) =
            crate::eval_primitive(Primitive::Cumsum, std::slice::from_ref(&x), &p).unwrap()
        else {
            panic!("tensor");
        };
        let got_sum = csum.elements.as_f64_slice().expect("dense f64");
        let mut acc = 0.0f64;
        let mut max_rel = 0.0f64;
        for i in (0..n).rev() {
            acc += data[i];
            let denom = acc.abs().max(1.0);
            max_rel = max_rel.max((got_sum[i] - acc).abs() / denom);
        }
        assert!(max_rel < 1e-5, "f64 reverse cumsum drifted: {max_rel:e}");
    }

    // f32 cumsum/cumprod 1-D (JAX's default dtype; measured JAX cumsum 39.0ms / cumprod 35.9ms). f64 1-D
    // cumsum uses the blocked parallel prefix scan (~21ms); does the f32 path get the same parallelism?
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_cumsum1d_f32_vs_jax() {
        use std::time::Instant;
        let n = 16_000_000usize;
        let data: Vec<f32> = (0..n).map(|i| 0.5 + (i % 9973) as f32 * 1e-5).collect();
        let x = Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: vec![n as u32],
                },
                data,
            )
            .unwrap(),
        );
        let p = BTreeMap::from([("axis".to_owned(), "0".to_owned())]);
        let bench = |label: &str, prim: Primitive, jax: f64| {
            let f = || {
                std::hint::black_box(
                    crate::eval_primitive(prim, std::slice::from_ref(&x), &p).unwrap(),
                );
            };
            f();
            let mut b = f64::MAX;
            for _ in 0..6 {
                let s = Instant::now();
                f();
                b = b.min(s.elapsed().as_secs_f64());
            }
            println!("fj-lax {label} f32 16M: {:.3}ms | JAX={jax}ms", b * 1e3);
        };
        bench("cumsum1d", Primitive::Cumsum, 39.0);
        bench("cumprod1d", Primitive::Cumprod, 35.9);
    }

    // REVERSE 1-D cumsum/cummax (common: attention backward, reverse scans). Measured JAX f32 cumsum_rev
    // 28.4ms / cummax_rev 30.0ms. The parallel scans gate on !reverse, so fj-lax reverse single-chain is
    // sequential — check the gap.
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_cumulative1d_rev_f32_vs_jax() {
        use std::time::Instant;
        let n = 16_000_000usize;
        let data: Vec<f32> = (0..n).map(|i| 0.5 + (i % 9973) as f32 * 1e-5).collect();
        let x = Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: vec![n as u32],
                },
                data,
            )
            .unwrap(),
        );
        let p = BTreeMap::from([
            ("axis".to_owned(), "0".to_owned()),
            ("reverse".to_owned(), "true".to_owned()),
        ]);
        let bench = |label: &str, prim: Primitive, jax: f64| {
            let f = || {
                std::hint::black_box(
                    crate::eval_primitive(prim, std::slice::from_ref(&x), &p).unwrap(),
                );
            };
            f();
            let mut b = f64::MAX;
            for _ in 0..6 {
                let s = Instant::now();
                f();
                b = b.min(s.elapsed().as_secs_f64());
            }
            println!("fj-lax {label} f32 16M: {:.3}ms | JAX={jax}ms", b * 1e3);
        };
        bench("cumsum1d_rev", Primitive::Cumsum, 28.4);
        bench("cummax1d_rev", Primitive::Cummax, 30.0);
    }

    // REVERSE 1-D cumsum/cummax f64 (measured JAX cumsum_rev 57.9ms / cummax_rev 59.4ms). f64 reverse is
    // gated out of the parallel path -> sequential; check the gap.
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_cumulative1d_rev_f64_vs_jax() {
        use std::time::Instant;
        let n = 16_000_000usize;
        let data: Vec<f64> = (0..n).map(|i| 0.5 + (i % 9973) as f64 * 1e-5).collect();
        let x = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![n as u32],
                },
                data,
            )
            .unwrap(),
        );
        let p = BTreeMap::from([
            ("axis".to_owned(), "0".to_owned()),
            ("reverse".to_owned(), "true".to_owned()),
        ]);
        let bench = |label: &str, prim: Primitive, jax: f64| {
            let f = || {
                std::hint::black_box(
                    crate::eval_primitive(prim, std::slice::from_ref(&x), &p).unwrap(),
                );
            };
            f();
            let mut b = f64::MAX;
            for _ in 0..6 {
                let s = Instant::now();
                f();
                b = b.min(s.elapsed().as_secs_f64());
            }
            println!("fj-lax {label} f64 16M: {:.3}ms | JAX={jax}ms", b * 1e3);
        };
        bench("cumsum1d_rev", Primitive::Cumsum, 57.9);
        bench("cummax1d_rev", Primitive::Cummax, 59.4);
    }

    // BOLD-VERIFY: cumsum [4096,1024] vs JAX (slow: f32 ax0 6.93 ax1 2.96ms; f64 ax0 20.85 ax1 18.28ms).
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_cumsum2d() {
        use std::time::Instant;
        let (rows, cols) = (4096usize, 1024usize);
        let total = rows * cols;
        let d64: Vec<f64> = (0..total)
            .map(|i| ((i % 9973) as f64) * 0.013 - 50.0)
            .collect();
        let d32: Vec<f32> = d64.iter().map(|&v| v as f32).collect();
        let shape = Shape {
            dims: vec![rows as u32, cols as u32],
        };
        let t64 = Value::Tensor(TensorValue::new_f64_values(shape.clone(), d64).unwrap());
        let t32 = Value::Tensor(TensorValue::new_f32_values(shape, d32).unwrap());
        for (prim, pname) in [
            (Primitive::Cumsum, "cumsum"),
            (Primitive::Cummax, "cummax"),
            (Primitive::Cumprod, "cumprod"),
        ] {
            for (name, t) in [("f64", &t64), ("f32", &t32)] {
                for ax in [0usize, 1usize] {
                    let p = BTreeMap::from([("axis".to_owned(), ax.to_string())]);
                    let run = || crate::eval_primitive(prim, std::slice::from_ref(t), &p).unwrap();
                    let _ = run();
                    let mut best = f64::MAX;
                    for _ in 0..15 {
                        let s = Instant::now();
                        let r = run();
                        best = best.min(s.elapsed().as_secs_f64());
                        std::hint::black_box(&r);
                    }
                    println!(
                        "BENCH {pname} [4096,1024] {name} axis={ax}: fj-lax={:.4}ms",
                        best * 1e3
                    );
                }
            }
        }
    }

    // Same-binary A/B: f16 per-row trailing reduce, OLD f64x8 widen vs NEW f32x8 (production).
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_f16_trailing_reduce_ab() {
        use std::time::Instant;
        // OLD path: f64x8 (f16_widen8 → f64), the pre-fix version.
        fn old_f16_reduce_f64x8(values: &[u16], is_max: bool) -> f64 {
            use std::simd::{Simd, num::SimdFloat};
            const L: usize = 8;
            let init = if is_max {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
            let decode_f16 = |b: u16| Literal::F16Bits(b).as_f64().unwrap_or(0.0);
            let mut vacc = Simd::<f64, L>::splat(init);
            let mut acc = init;
            let mut any_nan = false;
            let mut used = false;
            let chunks = values.chunks_exact(L);
            let tail = chunks.remainder();
            for chunk in chunks {
                let u = Simd::<u16, L>::from_slice(chunk);
                if crate::arithmetic::f16_input_needs_scalar(u) {
                    for &b in chunk {
                        let v = decode_f16(b);
                        if v.is_nan() {
                            any_nan = true;
                        } else {
                            acc = if is_max { acc.max(v) } else { acc.min(v) };
                        }
                    }
                } else {
                    let f = crate::arithmetic::f16_widen8(u);
                    vacc = if is_max {
                        vacc.simd_max(f)
                    } else {
                        vacc.simd_min(f)
                    };
                    used = true;
                }
            }
            for &b in tail {
                let v = decode_f16(b);
                if v.is_nan() {
                    any_nan = true;
                } else {
                    acc = if is_max { acc.max(v) } else { acc.min(v) };
                }
            }
            if any_nan {
                return f64::NAN;
            }
            let mut m = acc;
            if used {
                for &lane in vacc.to_array().iter() {
                    m = if is_max { m.max(lane) } else { m.min(lane) };
                }
            }
            m
        }
        let (rows, cols) = (4096usize, 4096usize);
        let f16_bits: Vec<u16> = (0..rows * cols)
            .map(
                |i| match Literal::from_f16_f64(((i % 9973) as f64) * 0.01 - 40.0) {
                    Literal::F16Bits(b) => b,
                    _ => 0,
                },
            )
            .collect();
        // PRIOR version: f32x8 with the per-chunk f16_input_needs_scalar fast/slow split.
        fn ns_f16_reduce_f32x8(values: &[u16], is_max: bool) -> f64 {
            use std::simd::{Simd, num::SimdFloat};
            const L: usize = 8;
            let init = if is_max {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            };
            let decode_f16 = |b: u16| Literal::F16Bits(b).as_f64().unwrap_or(0.0) as f32;
            let mut vacc = Simd::<f32, L>::splat(init);
            let mut acc = init;
            let mut any_nan = false;
            let mut used = false;
            let chunks = values.chunks_exact(L);
            let tail = chunks.remainder();
            for chunk in chunks {
                let u = Simd::<u16, L>::from_slice(chunk);
                if crate::arithmetic::f16_input_needs_scalar(u) {
                    for &b in chunk {
                        let v = decode_f16(b);
                        if v.is_nan() {
                            any_nan = true;
                        } else {
                            acc = if is_max { acc.max(v) } else { acc.min(v) };
                        }
                    }
                } else {
                    // mimic the old normal-only widen (shift+select) then f32 max
                    let h32 = u.cast::<u32>();
                    use std::simd::cmp::SimdPartialEq;
                    use std::simd::{Select, num::SimdUint};
                    let sign = (h32 & Simd::splat(0x8000u32)) << Simd::splat(16u32);
                    let he = (h32 & Simd::splat(0x7C00u32)) >> Simd::splat(10u32);
                    let fe = (he + Simd::splat(112u32)) << Simd::splat(23u32);
                    let fm = (h32 & Simd::splat(0x03FFu32)) << Simd::splat(13u32);
                    let nb = sign | fe | fm;
                    let iz = (h32 & Simd::splat(0x7FFFu32)).simd_eq(Simd::splat(0u32));
                    let f = Simd::<f32, L>::from_bits(iz.select(sign, nb));
                    vacc = if is_max {
                        vacc.simd_max(f)
                    } else {
                        vacc.simd_min(f)
                    };
                    used = true;
                }
            }
            for &b in tail {
                let v = decode_f16(b);
                if v.is_nan() {
                    any_nan = true;
                } else {
                    acc = if is_max { acc.max(v) } else { acc.min(v) };
                }
            }
            if any_nan {
                return f64::NAN;
            }
            let mut m = acc;
            if used {
                for &lane in vacc.to_array().iter() {
                    m = if is_max { m.max(lane) } else { m.min(lane) };
                }
            }
            f64::from(m)
        }
        let ab = |mode: u8| {
            let mut b = f64::MAX;
            for _ in 0..20 {
                let s = Instant::now();
                let mut acc = 0.0f64;
                for r in 0..rows {
                    let row = &f16_bits[r * cols..r * cols + cols];
                    let m = match mode {
                        0 => old_f16_reduce_f64x8(row, true),
                        1 => ns_f16_reduce_f32x8(row, true),
                        _ => super::simd_reduce_minmax_f16(row, true),
                    };
                    acc += m;
                }
                std::hint::black_box(acc);
                b = b.min(s.elapsed().as_secs_f64());
            }
            b * 1e3
        };
        let f64x8 = ab(0);
        let ns = ab(1);
        let branchless = ab(2);
        println!(
            "AB f16 trailing reduce [4096,4096] single-thread: f64x8={f64x8:.4}ms needs_scalar-f32x8={ns:.4}ms branchless-f32x8={branchless:.4}ms | branchless vs ns = {:.2}x",
            ns / branchless
        );
    }

    // BOLD-VERIFY: any(x>0) vs JAX (16M f64, measured JAX 5.95ms). fj-lax composes Gt (128MB read ->
    // packed-bool) + ReduceOr (reads only the 2MB packed words, short-circuit) — comparison-read-bound.
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_any_gt_vs_jax() {
        use std::time::Instant;
        let n = 16_000_000usize;
        let data: Vec<f64> = (0..n).map(|i| (i % 9973) as f64 * 0.01 - 1.0).collect();
        let x = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![n as u32],
                },
                data,
            )
            .unwrap(),
        );
        let zero = Value::Scalar(Literal::F64Bits(0.0f64.to_bits()));
        let p = BTreeMap::from([("axes".to_owned(), "0".to_owned())]);
        let f = || {
            let mask =
                crate::eval_primitive(Primitive::Gt, &[x.clone(), zero.clone()], &BTreeMap::new())
                    .unwrap();
            std::hint::black_box(
                crate::eval_primitive(Primitive::ReduceOr, std::slice::from_ref(&mask), &p)
                    .unwrap(),
            );
        };
        f();
        let mut b = f64::MAX;
        for _ in 0..8 {
            let s = Instant::now();
            f();
            b = b.min(s.elapsed().as_secs_f64());
        }
        println!("fj-lax any(x>0) f64 16M: {:.3}ms | JAX=5.95ms", b * 1e3);
    }

    // BOLD-VERIFY: full reductions vs JAX (16M f64, measured): sum 6.5ms / max 6.7ms / argmax 25.2ms.
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_full_reduce_vs_jax() {
        use std::time::Instant;
        let n = 16_000_000usize;
        let data: Vec<f64> = (0..n).map(|i| 0.5 + (i % 9973) as f64 * 1e-4).collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![n as u32],
                },
                data,
            )
            .unwrap(),
        );
        let p = BTreeMap::from([("axes".to_owned(), "0".to_owned())]);
        let bench = |label: &str, prim: Primitive| {
            let f = || {
                std::hint::black_box(
                    crate::eval_primitive(prim, std::slice::from_ref(&input), &p).unwrap(),
                );
            };
            f();
            let mut b = f64::MAX;
            for _ in 0..8 {
                let s = Instant::now();
                f();
                b = b.min(s.elapsed().as_secs_f64());
            }
            println!("fj-lax {label} f64 16M: {:.3}ms", b * 1e3);
        };
        bench("sum", Primitive::ReduceSum);
        bench("max", Primitive::ReduceMax);
        bench("prod", Primitive::ReduceProd);
        bench("argmax", Primitive::Argmax);
    }

    // BOLD-VERIFY: max-reduce 2D vs JAX (f32 ax0 1.67 ax1 0.65ms; f64 ax0 10.36 ax1 3.13ms).
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_maxreduce2d() {
        use std::time::Instant;
        let (rows, cols) = (4096usize, 4096usize);
        let total = rows * cols;
        let d64: Vec<f64> = (0..total)
            .map(|i| ((i % 99991) as f64) * 0.001 - 50.0)
            .collect();
        let d32: Vec<f32> = d64.iter().map(|&v| v as f32).collect();
        let bf16_bits: Vec<u16> = d64
            .iter()
            .map(|&v| match Literal::from_bf16_f64(v) {
                Literal::BF16Bits(b) => b,
                _ => 0,
            })
            .collect();
        let f16_bits: Vec<u16> = d64
            .iter()
            .map(|&v| match Literal::from_f16_f64(v) {
                Literal::F16Bits(b) => b,
                _ => 0,
            })
            .collect();
        let shape = Shape {
            dims: vec![rows as u32, cols as u32],
        };
        let t64 = Value::Tensor(TensorValue::new_f64_values(shape.clone(), d64).unwrap());
        let t32 = Value::Tensor(TensorValue::new_f32_values(shape.clone(), d32).unwrap());
        let tbf16 = Value::Tensor(
            TensorValue::new_half_float_values(DType::BF16, shape.clone(), bf16_bits).unwrap(),
        );
        let tf16 =
            Value::Tensor(TensorValue::new_half_float_values(DType::F16, shape, f16_bits).unwrap());
        for (name, t) in [
            ("f64", &t64),
            ("f32", &t32),
            ("bf16", &tbf16),
            ("f16", &tf16),
        ] {
            for ax in [0usize, 1usize] {
                let p = BTreeMap::from([("axes".to_owned(), ax.to_string())]);
                let run = || {
                    crate::eval_primitive(Primitive::ReduceMax, std::slice::from_ref(t), &p)
                        .unwrap()
                };
                let _ = run();
                let mut best = f64::MAX;
                for _ in 0..15 {
                    let s = Instant::now();
                    let r = run();
                    best = best.min(s.elapsed().as_secs_f64());
                    std::hint::black_box(&r);
                }
                println!(
                    "BENCH maxreduce [4096,4096] {name} axis={ax}: fj-lax={:.4}ms",
                    best * 1e3
                );
            }
        }
    }

    // BOLD-VERIFY: global-avg-pool = sum over spatial axes {1,2} of NHWC, keeping the contiguous
    // channel C (inner!=1 reduce path). Currently a scalar out_row[c]+=in_row[c] loop (reduction.rs
    // ~1656). Measure vs JAX (f32 0.14ms, f64 0.42ms).
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_global_avg_pool_reduce() {
        use std::time::Instant;
        let (nb, hw, c) = (8usize, 112usize, 64usize);
        let total = nb * hw * hw * c;
        let d64: Vec<f64> = (0..total)
            .map(|i| ((i % 9973) as f64) * 0.013 - 50.0)
            .collect();
        let d32: Vec<f32> = d64.iter().map(|&v| v as f32).collect();
        let shape = Shape {
            dims: vec![nb as u32, hw as u32, hw as u32, c as u32],
        };
        let t64 = Value::Tensor(TensorValue::new_f64_values(shape.clone(), d64).unwrap());
        let t32 = Value::Tensor(TensorValue::new_f32_values(shape, d32).unwrap());
        let mut p = BTreeMap::new();
        p.insert("axes".to_string(), "1,2".to_string());
        // Serial reference (the exact inner!=1 fold, single-thread) for a SAME-BINARY A/B vs the
        // now-threaded eval_primitive path.
        let (outer, reduce, inner) = (nb, hw * hw, c);
        let vals64: Vec<f64> = (0..total)
            .map(|i| ((i % 9973) as f64) * 0.013 - 50.0)
            .collect();
        let serial_f64 = |v: &[f64]| {
            let mut out = vec![0.0f64; outer * inner];
            for o in 0..outer {
                let out_row = &mut out[o * inner..(o + 1) * inner];
                for r in 0..reduce {
                    let in_row = &v[(o * reduce + r) * inner..][..inner];
                    for (slot, &x) in out_row.iter_mut().zip(in_row) {
                        *slot += x;
                    }
                }
            }
            out
        };
        // Explicit-SIMD fold (single-thread) — does it beat the closure (i.e. does the closure
        // fail to autovectorize)? Decides bead 5y9jg.
        let simd_f64 = |v: &[f64]| {
            use std::simd::Simd;
            type F = Simd<f64, 8>;
            let mut out = vec![0.0f64; outer * inner];
            let c8 = inner - inner % 8;
            for o in 0..outer {
                let out_row = &mut out[o * inner..(o + 1) * inner];
                for r in 0..reduce {
                    let in_row = &v[(o * reduce + r) * inner..][..inner];
                    let mut c = 0;
                    while c < c8 {
                        let a =
                            F::from_slice(&out_row[c..c + 8]) + F::from_slice(&in_row[c..c + 8]);
                        a.copy_to_slice(&mut out_row[c..c + 8]);
                        c += 8;
                    }
                    while c < inner {
                        out_row[c] += in_row[c];
                        c += 1;
                    }
                }
            }
            out
        };
        let bench = |f: &dyn Fn()| {
            f();
            let mut b = f64::MAX;
            for _ in 0..20 {
                let s = Instant::now();
                f();
                b = b.min(s.elapsed().as_secs_f64());
            }
            b * 1e3
        };
        let sref = bench(&|| {
            std::hint::black_box(serial_f64(&vals64));
        });
        let simdref = bench(&|| {
            std::hint::black_box(simd_f64(&vals64));
        });
        // sanity: same result
        assert_eq!(
            serial_f64(&vals64)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            simd_f64(&vals64)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let thr = bench(&|| {
            std::hint::black_box(
                crate::eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&t64), &p)
                    .unwrap(),
            );
        });
        let thr32 = bench(&|| {
            std::hint::black_box(
                crate::eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&t32), &p)
                    .unwrap(),
            );
        });
        println!(
            "AB global-avg-pool axes12 [8,112,112,64]: f64 serial-closure={sref:.4} serial-SIMD={simdref:.4} (closure/simd={:.2}x) threaded={thr:.4} | f32 threaded={thr32:.4} (JAX f64 0.42 f32 0.14)",
            sref / simdref
        );
    }

    #[test]
    fn threaded_leading_axis_reduce_bit_identical_to_serial() {
        // Reduce over axis 0 of a large [rows, cols] f64 tensor (the column-accumulation
        // / leading-axis path). The threaded path splits OUTPUT COLUMNS across threads;
        // each column folds its rows in ascending order on exactly one thread, so the
        // result MUST be bit-identical to the serial column fold — even for
        // non-associative float (we never reassociate a single column's sum). rows*cols
        // exceeds the 1<<18 gate so the threaded path engages.
        let (rows, cols) = (512usize, 600usize); // 307_200 > 262_144 gate
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| ((i as f64) * 0.000_173).sin() * 1e6 - (i as f64) * 0.5)
            .collect();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        let mut p = BTreeMap::new();
        p.insert("axes".to_string(), "0".to_string());
        let got =
            crate::eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p).unwrap();
        let got: Vec<u64> = match got {
            Value::Tensor(t) => t
                .elements
                .iter()
                .map(|l| l.as_f64().unwrap().to_bits())
                .collect(),
            other => panic!("expected tensor, got {other:?}"),
        };
        // Serial reference: each column folded over rows in ascending order.
        let mut want = vec![0.0f64; cols];
        for r in 0..rows {
            for c in 0..cols {
                want[c] += data[r * cols + c];
            }
        }
        let want: Vec<u64> = want.iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            got, want,
            "threaded leading-axis reduce != serial column fold"
        );
    }

    #[test]
    fn threaded_integer_axis_reduce_bit_identical_to_serial() {
        // Large 2D i64 reduce over axis 0 (leading -> reduce-dim chunking) and axis 1
        // (trailing -> output threading). Integer reduce is associative, so both threaded
        // paths MUST equal the serial column/block fold exactly (incl. i64 wraparound).
        // rows*cols exceeds the 1<<23 gate so the threaded paths engage.
        let (rows, cols) = (8192usize, 1100usize); // 9_011_200 > 8_388_608 gate
        let data: Vec<i64> = (0..(rows * cols) as i64)
            .map(|i| i.wrapping_mul(6_364_136_223_846_793_005) ^ (i << 7))
            .collect();
        let input = Value::Tensor(
            TensorValue::new_i64_values(
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        // axis 0 (leading): out[c] = sum_r data[r*cols + c]
        let mut p0 = BTreeMap::new();
        p0.insert("axes".to_string(), "0".to_string());
        let got0 =
            crate::eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p0).unwrap();
        let got0: Vec<i64> = match got0 {
            Value::Tensor(t) => t.elements.iter().map(|l| l.as_i64().unwrap()).collect(),
            _ => panic!(),
        };
        let mut want0 = vec![0i64; cols];
        for r in 0..rows {
            for c in 0..cols {
                want0[c] = want0[c].wrapping_add(data[r * cols + c]);
            }
        }
        assert_eq!(got0, want0, "axis0 threaded leading reduce != serial");
        // axis 1 (trailing): out[r] = sum_c data[r*cols + c]
        let mut p1 = BTreeMap::new();
        p1.insert("axes".to_string(), "1".to_string());
        let got1 =
            crate::eval_primitive(Primitive::ReduceSum, std::slice::from_ref(&input), &p1).unwrap();
        let got1: Vec<i64> = match got1 {
            Value::Tensor(t) => t.elements.iter().map(|l| l.as_i64().unwrap()).collect(),
            _ => panic!(),
        };
        let want1: Vec<i64> = (0..rows)
            .map(|r| {
                data[r * cols..r * cols + cols]
                    .iter()
                    .fold(0i64, |a, &v| a.wrapping_add(v))
            })
            .collect();
        assert_eq!(got1, want1, "axis1 threaded trailing reduce != serial");
    }

    #[test]
    fn threaded_integer_full_reduce_bit_identical_to_serial() {
        // >= gate so the threaded integer full-reduce engages. Integer
        // sum/prod/and/or/xor/max/min are associative / order-invariant, so the
        // chunked partial-fold MUST equal the sequential fold EXACTLY (incl. i64
        // wraparound), since `int_init` is each op's identity.
        let n = crate::arithmetic::CHEAP_BINARY_PARALLEL_MIN + 4321;
        let data: Vec<i64> = (0..n as i64)
            .map(|i| i.wrapping_mul(2_654_435_761) ^ (i << 11))
            .collect();
        let input = Value::Tensor(
            TensorValue::new_i64_values(
                Shape {
                    dims: vec![n as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        let mut p = BTreeMap::new();
        p.insert("axes".to_string(), "0".to_string());

        let cases: [(Primitive, i64, fn(i64, i64) -> i64); 6] = [
            (Primitive::ReduceSum, 0, i64::wrapping_add),
            (Primitive::ReduceProd, 1, i64::wrapping_mul),
            (Primitive::ReduceMax, i64::MIN, std::cmp::max),
            (Primitive::ReduceMin, i64::MAX, std::cmp::min),
            (Primitive::ReduceAnd, -1, |a, b| a & b),
            (Primitive::ReduceXor, 0, |a, b| a ^ b),
        ];
        for (prim, init, op) in cases {
            let got =
                eval_reduce(prim, std::slice::from_ref(&input), init, 0.0, op, |a, _| a).unwrap();
            let got = match got {
                Value::Scalar(l) => l.as_i64().expect("i64 scalar"),
                other => panic!("expected scalar, got {other:?}"),
            };
            let want = data.iter().fold(init, |a, &v| op(a, v));
            assert_eq!(got, want, "{prim:?} threaded full-reduce != sequential");
        }
    }

    #[test]
    fn threaded_reduce_minmax_bit_identical_to_serial() {
        // >= gate so the threaded path engages; seed NaN / +-inf / +-0 at varied positions
        // (incl. near chunk boundaries) and compare bit-for-bit to the serial SIMD reduce.
        let n = crate::arithmetic::CHEAP_BINARY_PARALLEL_MIN + 1234;
        let base: Vec<f64> = (0..n).map(|i| ((i % 4096) as f64) * 0.5 - 1000.0).collect();
        let make = |specials: &[(usize, f64)]| {
            let mut v = base.clone();
            for &(pos, val) in specials {
                v[pos % n] = val;
            }
            v
        };
        let cases: Vec<Vec<f64>> = vec![
            base.clone(),
            make(&[(0, 0.0), (n - 1, -0.0), (n / 2, 0.0)]), // ±0 only -> exercises the ±0 fold
            make(&[(7, f64::INFINITY), (n - 3, f64::NEG_INFINITY)]),
            make(&[(n / 3, f64::NAN)]),
            make(&[(n - 1, f64::NAN), (0, f64::NAN)]),
            make(&[(5, 0.0), (6, -0.0), (n - 2, -0.0)]),
        ];
        for v in &cases {
            for is_max in [true, false] {
                let serial = simd_reduce_minmax_f64(v, is_max);
                let threaded = threaded_reduce_minmax_f64(v, is_max);
                assert_eq!(
                    serial.to_bits(),
                    threaded.to_bits(),
                    "f64 threaded minmax (is_max={is_max}) != serial"
                );
                let vf: Vec<f32> = v.iter().map(|&x| x as f32).collect();
                let serial32 = simd_reduce_minmax_f32(&vf, is_max);
                let threaded32 = threaded_reduce_minmax_f32(&vf, is_max);
                assert_eq!(
                    serial32.to_bits(),
                    threaded32.to_bits(),
                    "f32 threaded minmax (is_max={is_max}) != serial"
                );
            }
        }
    }

    #[test]
    fn axis_reduce_output_is_dense_and_bit_identical() {
        // The float axis-reduce output must be dense (as_*_slice Some, not boxed Literals)
        // and byte-identical to the reduce_real_literal reference. De-boxing it is ~14x for
        // large outputs (the per-element Literal construction + TensorValue::new validation
        // dominated; A/B [2,2M] sum axis0: boxed 28ms -> dense 2ms).
        let (f, n) = (3usize, 5usize);
        let data: Vec<f64> = (0..f * n).map(|i| (i as f64) - 6.0).collect();
        let params = BTreeMap::from([("axes".to_owned(), "0".to_owned())]);
        // F64 sum.
        let va = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![f as u32, n as u32],
                },
                data.clone(),
            )
            .unwrap(),
        );
        let out = crate::eval_primitive(Primitive::ReduceSum, &[va], &params).unwrap();
        let t = out.as_tensor().unwrap();
        assert_eq!(t.dtype, DType::F64);
        assert!(
            t.elements.as_f64_slice().is_some(),
            "f64 reduce output must be dense"
        );
        let got = t.elements.as_f64_slice().unwrap();
        for (j, &g) in got.iter().enumerate() {
            let expect: f64 = (0..f).map(|r| data[r * n + j]).sum();
            assert_eq!(g.to_bits(), expect.to_bits(), "f64 reduce[{j}]");
        }
        // F32 sum — output stays dense f32, rounded.
        let vf32 = Value::Tensor(
            TensorValue::new_f32_values(
                Shape {
                    dims: vec![f as u32, n as u32],
                },
                data.iter().map(|&v| v as f32).collect(),
            )
            .unwrap(),
        );
        let out = crate::eval_primitive(Primitive::ReduceSum, &[vf32], &params).unwrap();
        let t = out.as_tensor().unwrap();
        assert_eq!(t.dtype, DType::F32);
        assert!(
            t.elements.as_f32_slice().is_some(),
            "f32 reduce output must be dense"
        );
        // F64 max — dense path covers all real reducers.
        let vmax = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![f as u32, n as u32],
                },
                data,
            )
            .unwrap(),
        );
        let out = crate::eval_primitive(Primitive::ReduceMax, &[vmax], &params).unwrap();
        assert!(
            out.as_tensor().unwrap().elements.as_f64_slice().is_some(),
            "f64 max reduce output must be dense"
        );
        // I64 sum — integer axis-reduce output must also be dense + correct.
        let idata: Vec<i64> = (0..f * n).map(|i| i as i64 - 6).collect();
        let vi = Value::Tensor(
            TensorValue::new_i64_values(
                Shape {
                    dims: vec![f as u32, n as u32],
                },
                idata.clone(),
            )
            .unwrap(),
        );
        let out = crate::eval_primitive(Primitive::ReduceSum, &[vi], &params).unwrap();
        let t = out.as_tensor().unwrap();
        assert_eq!(t.dtype, DType::I64);
        assert!(
            t.elements.as_i64_slice().is_some(),
            "i64 reduce output must be dense"
        );
        let got = t.elements.as_i64_slice().unwrap();
        for (j, &g) in got.iter().enumerate() {
            let expect: i64 = (0..f).map(|r| idata[r * n + j]).sum();
            assert_eq!(g, expect, "i64 reduce[{j}]");
        }
        // BF16 sum — half-float axis-reduce output must also be dense (u16 backing).
        let bf16data: Vec<u16> = (0..f * n)
            .map(|i| match Literal::from_bf16_f64((i as f64) * 0.5) {
                Literal::BF16Bits(b) => b,
                _ => unreachable!(),
            })
            .collect();
        let vbf = Value::Tensor(
            TensorValue::new_half_float_values(
                DType::BF16,
                Shape {
                    dims: vec![f as u32, n as u32],
                },
                bf16data,
            )
            .unwrap(),
        );
        let out = crate::eval_primitive(Primitive::ReduceSum, &[vbf], &params).unwrap();
        let t = out.as_tensor().unwrap();
        assert_eq!(t.dtype, DType::BF16);
        assert!(
            t.elements.as_half_float_slice().is_some(),
            "bf16 reduce output must be dense u16"
        );
    }

    fn s_f64(v: f64) -> Value {
        Value::Scalar(Literal::from_f64(v))
    }
    fn v_f64(data: &[f64]) -> Value {
        // Boxed (Literal-backed) reference for dense-vs-literal guard tests.
        // `TensorValue::new` densifies homogeneous F64 (cbea72b3), which would
        // turn this reference dense; `crate::new_boxed` keeps it boxed.
        Value::Tensor(
            crate::new_boxed(
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
    fn dense_f64_reduce_sum_large_tree_matches_sequential_with_tolerance() {
        let n = crate::arithmetic::CHEAP_BINARY_PARALLEL_MIN + 1024;
        let data: Vec<f64> = (0..n).map(|i| ((i as f64) * 1.1e-7).sin() * 3.0).collect();
        let expected: f64 = data.iter().copied().sum();
        let input = Value::Tensor(
            TensorValue::new_f64_values(
                Shape {
                    dims: vec![n as u32],
                },
                data,
            )
            .unwrap(),
        );

        let result = eval_reduce(
            Primitive::ReduceSum,
            &[input],
            0,
            0.0,
            |a, b| a + b,
            |a, b| a + b,
        )
        .unwrap();
        let got = extract_f64(&result);
        let tolerance = expected.abs() * 1e-12 + 1e-9;

        assert!(
            (got - expected).abs() <= tolerance,
            "large tree reduce_sum got {got}, sequential {expected}, diff {} > {tolerance}",
            (got - expected).abs()
        );
    }

    #[test]
    fn dense_f32_full_reduce_bit_identical_to_literal_path() {
        // The dense F32 full-reduce (eval_dense_float_full_reduce) must be BIT-FOR-BIT
        // identical to the generic boxed-Literal path for sum/prod/max/min, including
        // ±inf, NaN payload/sign, and signed zero — both promote each f32 to f64, fold a
        // single f64 accumulator in ascending order, then round to f32 at output.
        let data: Vec<f32> = std::hint::black_box(vec![
            1.5_f32,
            -0.0,
            f32::from_bits(0x7fc0_0001), // NaN payload
            -3.25,
            f32::INFINITY,
            0.0,
            1e30,
            -1e30,
            f32::NEG_INFINITY,
            2.5,
        ]);
        let dims = vec![data.len() as u32];

        // Dense F32 storage (fast path) vs boxed-Literal F32 (generic path).
        let dense = Value::Tensor(
            TensorValue::new_f32_values(Shape { dims: dims.clone() }, data.clone()).unwrap(),
        );
        assert!(dense.as_tensor().unwrap().elements.as_f32_slice().is_some());
        let boxed = Value::Tensor(
            crate::new_boxed(
                DType::F32,
                Shape { dims },
                data.iter().map(|&v| Literal::from_f32(v)).collect(),
            )
            .unwrap(),
        );
        assert!(boxed.as_tensor().unwrap().elements.as_f32_slice().is_none());

        let extract_f32_bits = |val: &Value| -> u32 {
            match val {
                Value::Scalar(Literal::F32Bits(b)) => *b,
                other => panic!("expected f32 scalar, got {other:?}"),
            }
        };

        // (primitive, float_init, float_op) mirroring lib.rs reduce wiring.
        let cases: [(Primitive, f64, fn(f64, f64) -> f64); 4] = [
            (Primitive::ReduceSum, 0.0, |a, b| a + b),
            (Primitive::ReduceProd, 1.0, |a, b| a * b),
            (Primitive::ReduceMax, f64::NEG_INFINITY, crate::jax_max_f64),
            (Primitive::ReduceMin, f64::INFINITY, crate::jax_min_f64),
        ];
        for (prim, finit, fop) in cases {
            let dense_result =
                eval_reduce(prim, std::slice::from_ref(&dense), 0, finit, |a, _| a, fop).unwrap();
            let boxed_result =
                eval_reduce(prim, std::slice::from_ref(&boxed), 0, finit, |a, _| a, fop).unwrap();
            assert_eq!(
                extract_f32_bits(&dense_result),
                extract_f32_bits(&boxed_result),
                "{prim:?} dense vs boxed f32 full-reduce must be bit-identical"
            );
        }
    }

    #[test]
    fn dense_half_float_full_reduce_bit_identical_to_literal_path() {
        // BF16/F16 full-reduce: dense packed-u16 storage (fast path) must be BIT-FOR-BIT
        // identical to the generic boxed-Literal path for sum/prod/max/min — both decode
        // each u16 to f64 via Literal::{BF16,F16}Bits.as_f64(), fold a single f64
        // accumulator ascending, then round back to the half dtype at output. Cover
        // ±inf, NaN, signed zero, and large magnitudes.
        let src: Vec<f64> = std::hint::black_box(vec![
            1.5,
            -0.0,
            f64::NAN,
            -3.25,
            f64::INFINITY,
            0.0,
            240.0,
            -240.0,
            f64::NEG_INFINITY,
            2.5,
        ]);
        let cases: [(Primitive, f64, fn(f64, f64) -> f64); 4] = [
            (Primitive::ReduceSum, 0.0, |a, b| a + b),
            (Primitive::ReduceProd, 1.0, |a, b| a * b),
            (Primitive::ReduceMax, f64::NEG_INFINITY, crate::jax_max_f64),
            (Primitive::ReduceMin, f64::INFINITY, crate::jax_min_f64),
        ];
        for (dtype, to_lit) in [
            (DType::BF16, Literal::from_bf16_f64 as fn(f64) -> Literal),
            (DType::F16, Literal::from_f16_f64 as fn(f64) -> Literal),
        ] {
            // Build matching dense + boxed tensors from the SAME half-rounded literals.
            let lits: Vec<Literal> = src.iter().map(|&v| to_lit(v)).collect();
            let bits: Vec<u16> = lits
                .iter()
                .map(|l| match l {
                    Literal::BF16Bits(b) | Literal::F16Bits(b) => *b,
                    other => panic!("expected half-float literal, got {other:?}"),
                })
                .collect();
            let dims = vec![src.len() as u32];
            let dense = Value::Tensor(
                TensorValue::new_half_float_values(dtype, Shape { dims: dims.clone() }, bits)
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
            let boxed = Value::Tensor(crate::new_boxed(dtype, Shape { dims }, lits).unwrap());
            assert!(
                boxed
                    .as_tensor()
                    .unwrap()
                    .elements
                    .as_half_float_slice()
                    .is_none()
            );

            let extract_half_bits = |val: &Value| -> u16 {
                match val {
                    Value::Scalar(Literal::BF16Bits(b) | Literal::F16Bits(b)) => *b,
                    other => panic!("expected half-float scalar, got {other:?}"),
                }
            };
            for (prim, finit, fop) in cases {
                let d = eval_reduce(prim, std::slice::from_ref(&dense), 0, finit, |a, _| a, fop)
                    .unwrap();
                let b = eval_reduce(prim, std::slice::from_ref(&boxed), 0, finit, |a, _| a, fop)
                    .unwrap();
                assert_eq!(
                    extract_half_bits(&d),
                    extract_half_bits(&b),
                    "{dtype:?} {prim:?} dense vs boxed half-float full-reduce must be bit-identical"
                );
            }
        }
    }

    #[test]
    fn simd_bf16_minmax_zero_fallback_bit_identical() {
        // BF16 full-reduce max/min where the result is ±0 (every value ≤ 0 for max,
        // with ±0 present) — exercises the SIMD ±0 scalar-fold fallback. >=16 elems
        // to span SIMD chunks + tail. dense (SIMD) vs boxed (scalar) bit-for-bit.
        let src: Vec<f64> = vec![
            -1.0, -0.0, -2.5, -3.0, -0.0, -1.5, -4.0, -2.0, 0.0, -1.0, -5.0, -0.0, -2.0, -3.5,
            -1.0, -6.0, 0.0, -2.0, -0.0,
        ];
        let lits: Vec<Literal> = src.iter().map(|&v| Literal::from_bf16_f64(v)).collect();
        let bits: Vec<u16> = lits
            .iter()
            .map(|l| match l {
                Literal::BF16Bits(b) => *b,
                other => panic!("expected bf16, got {other:?}"),
            })
            .collect();
        let dims = vec![src.len() as u32];
        let dense = Value::Tensor(
            TensorValue::new_half_float_values(DType::BF16, Shape { dims: dims.clone() }, bits)
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
        let boxed = Value::Tensor(
            crate::new_boxed(DType::BF16, Shape { dims: dims.clone() }, lits).unwrap(),
        );
        assert!(
            boxed
                .as_tensor()
                .unwrap()
                .elements
                .as_half_float_slice()
                .is_none()
        );
        let params = BTreeMap::new();
        for primitive in [Primitive::ReduceMax, Primitive::ReduceMin] {
            let d =
                crate::eval_primitive(primitive, std::slice::from_ref(&dense), &params).unwrap();
            let b =
                crate::eval_primitive(primitive, std::slice::from_ref(&boxed), &params).unwrap();
            let bits_of = |v: &Value| match v {
                Value::Scalar(Literal::BF16Bits(x)) => *x,
                other => panic!("expected bf16 scalar, got {other:?}"),
            };
            assert_eq!(
                bits_of(&d),
                bits_of(&b),
                "bf16 {primitive:?} ±0 fallback mismatch"
            );
        }
    }

    #[test]
    fn simd_bf16_axis_minmax_bit_identical() {
        // BF16 `max/min(x, axis=-1)` (e.g. attention-score max for stability): the
        // dense SIMD per-cell path must equal the boxed scalar fold over NaN / ±0
        // (per-cell ±0 fallback) / general data. cols=19: spans bf16 SIMD chunks+tail.
        let rows = 5u32;
        let cols = 19u32;
        let n = (rows * cols) as usize;
        let nan = f64::NAN;
        let src: Vec<f64> = (0..n)
            .map(|i| {
                let r = i / cols as usize;
                let c = i % cols as usize;
                match r {
                    0 => ((i as f64) * 0.3).sin() * 6.0 + 0.25, // general, NaN-free
                    1 => {
                        if c == cols as usize - 1 {
                            nan
                        } else {
                            (c as f64) - 9.0
                        }
                    } // NaN in row
                    2 => {
                        if c.is_multiple_of(2) {
                            -0.0
                        } else {
                            -(1.0 + c as f64)
                        }
                    } // all ≤0 ⇒ max ±0
                    3 => {
                        if c.is_multiple_of(2) {
                            0.0
                        } else {
                            1.0 + c as f64
                        }
                    } // all ≥0 ⇒ min ±0
                    _ => (c as f64) * 0.5 - 4.0,
                }
            })
            .collect();
        let lits: Vec<Literal> = src.iter().map(|&v| Literal::from_bf16_f64(v)).collect();
        let bits: Vec<u16> = lits
            .iter()
            .map(|l| match l {
                Literal::BF16Bits(b) => *b,
                o => panic!("expected bf16, got {o:?}"),
            })
            .collect();
        let dims = vec![rows, cols];
        let dense = Value::Tensor(
            TensorValue::new_half_float_values(DType::BF16, Shape { dims: dims.clone() }, bits)
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
        let boxed = Value::Tensor(
            crate::new_boxed(DType::BF16, Shape { dims: dims.clone() }, lits).unwrap(),
        );
        assert!(
            boxed
                .as_tensor()
                .unwrap()
                .elements
                .as_half_float_slice()
                .is_none()
        );
        let mut p = BTreeMap::new();
        p.insert("axes".to_owned(), "1".to_owned());
        for primitive in [Primitive::ReduceMax, Primitive::ReduceMin] {
            let d = crate::eval_primitive(primitive, std::slice::from_ref(&dense), &p).unwrap();
            let b = crate::eval_primitive(primitive, std::slice::from_ref(&boxed), &p).unwrap();
            let bits_of = |v: &Value| -> Vec<u16> {
                v.as_tensor()
                    .unwrap()
                    .elements
                    .iter()
                    .map(|l| match l {
                        Literal::BF16Bits(x) => *x,
                        o => panic!("expected bf16, got {o:?}"),
                    })
                    .collect()
            };
            assert_eq!(
                bits_of(&d),
                bits_of(&b),
                "bf16 axis {primitive:?} SIMD != scalar"
            );
        }
    }

    #[test]
    fn simd_inner_axis_reduce_bit_identical() {
        // The inner>1 (leading/middle-axis) SIMD reduce paths (dense storage) must equal
        // the boxed scalar odometer fold bit-for-bit — for max/min (f64/f32/bf16/f16) AND
        // sum/prod (bf16/f16, half decode + SIMD add/mul keeping per-cell order) — across
        // NaN / ±0 / inf / subnormal and chunk+tail widths, for 2D axis=0 and 3D axis=1.
        let nan = f64::NAN;
        let inf = f64::INFINITY;
        // cell value generator with edge values sprinkled in.
        let cell = |i: usize, n: usize| -> f64 {
            match i % 7 {
                0 if i.is_multiple_of(13) => nan,
                1 => -0.0,
                2 => 0.0,
                3 if i.is_multiple_of(11) => inf,
                4 if i.is_multiple_of(11) => -inf,
                _ => ((i as f64) * 0.37).sin() * 5.0 - ((i % n) as f64) * 0.25,
            }
        };
        // (rank dims, axis) cases. cols=13/17 span SIMD chunks (L=4) + tail.
        let cases: Vec<(Vec<u32>, &str)> = vec![
            (vec![6, 13], "0"),    // 2D leading axis -> inner=13
            (vec![5, 4, 17], "1"), // 3D middle axis -> outer=5, reduce=4, inner=17
            (vec![3, 7, 5], "1"),  // small (below thread gate)
        ];
        for (dims, axes) in &cases {
            let n: usize = dims.iter().map(|&d| d as usize).product();
            let src: Vec<f64> = (0..n)
                .map(|i| cell(i, *dims.last().unwrap() as usize))
                .collect();
            for dt in [DType::F64, DType::F32, DType::BF16, DType::F16] {
                let to_lit = |v: f64| -> Literal {
                    match dt {
                        DType::F64 => Literal::from_f64(v),
                        DType::F32 => Literal::from_f32(v as f32),
                        DType::F16 => Literal::from_f16_f64(v),
                        _ => Literal::from_bf16_f64(v),
                    }
                };
                let lits: Vec<Literal> = src.iter().map(|&v| to_lit(v)).collect();
                let half_bits = |l: &Literal| -> u16 {
                    match l {
                        Literal::BF16Bits(b) | Literal::F16Bits(b) => *b,
                        o => panic!("expected half, got {o:?}"),
                    }
                };
                let dense = match dt {
                    DType::F64 => Value::Tensor(
                        TensorValue::new_f64_values(Shape { dims: dims.clone() }, src.clone())
                            .unwrap(),
                    ),
                    DType::F32 => Value::Tensor(
                        TensorValue::new_f32_values(
                            Shape { dims: dims.clone() },
                            src.iter().map(|&v| v as f32).collect(),
                        )
                        .unwrap(),
                    ),
                    _ => Value::Tensor(
                        TensorValue::new_half_float_values(
                            dt,
                            Shape { dims: dims.clone() },
                            lits.iter().map(&half_bits).collect(),
                        )
                        .unwrap(),
                    ),
                };
                let boxed = Value::Tensor(
                    TensorValue::new(dt, Shape { dims: dims.clone() }, lits).unwrap(),
                );
                let mut p = BTreeMap::new();
                p.insert("axes".to_owned(), (*axes).to_owned());
                let raw = |v: &Value| -> Vec<u64> {
                    v.as_tensor()
                        .unwrap()
                        .elements
                        .iter()
                        .map(|l| l.as_f64().unwrap().to_bits())
                        .collect()
                };
                for primitive in [
                    Primitive::ReduceMax,
                    Primitive::ReduceMin,
                    Primitive::ReduceSum,
                ] {
                    let d =
                        crate::eval_primitive(primitive, std::slice::from_ref(&dense), &p).unwrap();
                    let b =
                        crate::eval_primitive(primitive, std::slice::from_ref(&boxed), &p).unwrap();
                    assert_eq!(
                        raw(&d),
                        raw(&b),
                        "{dt:?} {primitive:?} dims={dims:?} axes={axes} dense != scalar"
                    );
                }
            }
        }
    }

    #[test]
    fn simd_f16_axis_minmax_bit_identical() {
        // The new F16 SIMD min/max axis path (dense half_float_slice storage) must equal
        // the scalar fold (boxed Literals storage) bit-for-bit — across NaN, ±0
        // (order-dependent sign fallback), subnormal / inf (needs_scalar fallback), and
        // general data. cols=19 spans an 8-lane SIMD chunk + tail.
        let rows = 6u32;
        let cols = 19u32;
        let n = (rows * cols) as usize;
        let nan = f64::NAN;
        let src: Vec<f64> = (0..n)
            .map(|i| {
                let r = i / cols as usize;
                let c = i % cols as usize;
                match r {
                    0 => ((i as f64) * 0.3).sin() * 6.0 + 0.25, // general, NaN-free
                    1 => {
                        if c == cols as usize - 1 {
                            nan
                        } else {
                            (c as f64) - 9.0
                        }
                    } // NaN in row
                    2 => {
                        if c.is_multiple_of(2) {
                            -0.0
                        } else {
                            -(1.0 + c as f64)
                        }
                    } // all ≤0 ⇒ max ±0
                    3 => {
                        if c.is_multiple_of(2) {
                            0.0
                        } else {
                            1.0 + c as f64
                        }
                    } // all ≥0 ⇒ min ±0
                    4 => {
                        if c == 3 {
                            1e-6 // f16 subnormal -> needs_scalar fallback
                        } else if c == 7 {
                            f64::INFINITY
                        } else {
                            (c as f64) * 0.25 - 2.0
                        }
                    }
                    _ => (c as f64) * 0.5 - 4.0,
                }
            })
            .collect();
        let lits: Vec<Literal> = src.iter().map(|&v| Literal::from_f16_f64(v)).collect();
        let bits: Vec<u16> = lits
            .iter()
            .map(|l| match l {
                Literal::F16Bits(b) => *b,
                o => panic!("expected f16, got {o:?}"),
            })
            .collect();
        let dims = vec![rows, cols];
        let dense = Value::Tensor(
            TensorValue::new_half_float_values(DType::F16, Shape { dims: dims.clone() }, bits)
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
        let boxed = Value::Tensor(
            crate::new_boxed(DType::F16, Shape { dims: dims.clone() }, lits).unwrap(),
        );
        assert!(
            boxed
                .as_tensor()
                .unwrap()
                .elements
                .as_half_float_slice()
                .is_none()
        );
        let mut p = BTreeMap::new();
        p.insert("axes".to_owned(), "1".to_owned());
        for primitive in [Primitive::ReduceMax, Primitive::ReduceMin] {
            let d = crate::eval_primitive(primitive, std::slice::from_ref(&dense), &p).unwrap();
            let b = crate::eval_primitive(primitive, std::slice::from_ref(&boxed), &p).unwrap();
            let bits_of = |v: &Value| -> Vec<u16> {
                v.as_tensor()
                    .unwrap()
                    .elements
                    .iter()
                    .map(|l| match l {
                        Literal::F16Bits(x) => *x,
                        o => panic!("expected f16, got {o:?}"),
                    })
                    .collect()
            };
            assert_eq!(
                bits_of(&d),
                bits_of(&b),
                "f16 axis {primitive:?} SIMD != scalar"
            );
        }
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

    /// SIMD ReduceMax/ReduceMin full-reduce must be bit-identical to the boxed
    /// per-`Literal` scalar fold across the SIMD hazards: the tail (length not a
    /// multiple of the lane count), NaN propagation (incl. payload collapse), and —
    /// critically — ±0 where `simd_max`/`simd_min` corrupt a lane but the ±0 fixup
    /// must still recover the exact bit (e.g. +0.0 and -0.0 landing in the SAME lane
    /// across chunks). The boxed input (`as_f64_slice` None) forces the scalar
    /// reference path; the dense input takes the SIMD path.
    #[test]
    fn simd_reduce_minmax_bit_identical_to_scalar_fold() {
        let nan_payload = f64::from_bits(0x7ff8_0000_0000_0001);
        let snan = f64::from_bits(0x7ff0_0000_0000_0007);
        let cases: Vec<Vec<f64>> = vec![
            // +0 (lane 0, chunk 0) and -0 (lane 0, chunk 1) collide in one lane;
            // everything else negative so the reduced value is ±0. (17 elems: tail.)
            {
                let mut a = vec![-1.0f64; 17];
                a[0] = 0.0;
                a[8] = -0.0;
                a[16] = -5.0;
                a
            },
            // Only -0.0 zeros present (max ⇒ -0.0, min picks the smallest negative).
            {
                let mut a = vec![-2.0f64; 20];
                a[3] = -0.0;
                a[11] = -0.0;
                a
            },
            // +0 and -0 with everything else positive (min ⇒ -0.0, max ⇒ the max pos).
            {
                let mut a = vec![3.0f64; 19];
                a[1] = 0.0;
                a[9] = -0.0;
                a
            },
            // NaN payload in the body — result must collapse to canonical NaN.
            {
                let mut a: Vec<f64> = (0..23).map(|i| i as f64 * 0.5 - 5.0).collect();
                a[7] = nan_payload;
                a
            },
            // Signaling-NaN only in the tail.
            {
                let mut a: Vec<f64> = (0..13).map(|i| i as f64 - 6.0).collect();
                a[12] = snan;
                a
            },
            // ±inf mixed, longer, no zeros.
            {
                let mut a: Vec<f64> = (0..71).map(|i| (i as f64 * 0.37).sin() * 10.0).collect();
                a[5] = f64::INFINITY;
                a[40] = f64::NEG_INFINITY;
                a
            },
            vec![-0.0f64; 8], // all -0, exactly one chunk
            vec![0.0f64; 9],  // all +0, chunk + tail
        ];
        // Reference = the exact scalar fold the SIMD path REPLACES (jax_max/jax_min
        // over the dense slice, in ascending order) — NOT the boxed per-`Literal`
        // path, which has a separate pre-existing ±0 quirk (it can yield -0.0 where
        // f64::max yields +0.0). The SIMD path must equal the dense scalar fold.
        let jmax = |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        };
        let jmin = |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.min(b)
            }
        };
        let params = BTreeMap::new();
        for (ci, data) in cases.iter().enumerate() {
            for primitive in [Primitive::ReduceMax, Primitive::ReduceMin] {
                let is_max = primitive == Primitive::ReduceMax;
                let mut acc = if is_max {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                };
                for &v in data {
                    acc = if is_max { jmax(acc, v) } else { jmin(acc, v) };
                }
                let want64 = acc.to_bits();

                let dense = Value::vector_f64(data).unwrap();
                assert!(dense.as_tensor().unwrap().elements.as_f64_slice().is_some());
                let d = crate::eval_primitive(primitive, std::slice::from_ref(&dense), &params)
                    .unwrap();
                assert_eq!(
                    extract_f64_bits(&d),
                    want64,
                    "case {ci} {primitive:?}: SIMD f64 != dense scalar fold"
                );

                // f32 sibling (LANES=16): reference folds widened f64 then rounds to f32.
                let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
                let mut acc32 = if is_max {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                };
                for &v in &f32_data {
                    let vv = f64::from(v);
                    acc32 = if is_max {
                        jmax(acc32, vv)
                    } else {
                        jmin(acc32, vv)
                    };
                }
                let want32 = (acc32 as f32).to_bits();
                let dense32 = Value::Tensor(
                    TensorValue::new_f32_values(
                        Shape {
                            dims: vec![f32_data.len() as u32],
                        },
                        f32_data.clone(),
                    )
                    .unwrap(),
                );
                let d32 = crate::eval_primitive(primitive, std::slice::from_ref(&dense32), &params)
                    .unwrap();
                let bits32 = match &d32 {
                    Value::Scalar(Literal::F32Bits(b)) => *b,
                    other => panic!("expected f32 scalar, got {other:?}"),
                };
                assert_eq!(
                    bits32, want32,
                    "case {ci} {primitive:?}: SIMD f32 != dense scalar fold"
                );
            }
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
            crate::new_boxed(
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
                crate::new_boxed(
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
    fn simd_inner_axis_prod_half_bit_identical_to_generic_fuzz() {
        // The [5,4,17] axis=1 inner>1 PROD case the SIMD sum/prod kernel was once
        // (wrongly) suspected to diverge on. The column fold `out[c] *= decode(x[k,c])`
        // keeps each cell's exact k-order, and the decode matches the boxed
        // Literal::as_f64() — so the dense SIMD path must be bit-for-bit the generic
        // odometer for BOTH sum and prod. Fuzz a spread of normal/subnormal/inf/NaN/±0
        // half patterns across many seeds to make any `*` divergence visible.
        let nan_canon = |dtype: DType, b: u16| -> u16 {
            let v = if dtype == DType::BF16 {
                Literal::BF16Bits(b).as_f64()
            } else {
                Literal::F16Bits(b).as_f64()
            };
            if v.is_some_and(|x| x.is_nan()) {
                if dtype == DType::F16 { 0x7e00 } else { 0x7fc0 }
            } else {
                b
            }
        };
        // A pool of "interesting" half bit patterns (dtype-agnostic raw u16): the SIMD
        // kernels reinterpret raw bits per dtype, so the same pool exercises both.
        let specials: [u16; 8] = [
            0x0000, 0x8000, // +0, -0
            0x7c00, 0xfc00, // f16 +/-inf (bf16 reads these as finite — also fine)
            0x7e00, // f16 NaN
            0x0001, // subnormal
            0x3c00, 0xbc00, // +/-1.0 in f16
        ];
        for dtype in [DType::BF16, DType::F16] {
            for shape in [vec![5u32, 4, 17], vec![3, 9, 8], vec![2, 7, 33]] {
                let total: usize = shape.iter().map(|&d| d as usize).product();
                for seed in 0u64..24 {
                    // xorshift-style deterministic PRNG (no Instant/rand dependency).
                    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
                    let mut next = || {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        s
                    };
                    let raw: Vec<u16> = (0..total)
                        .map(|_| {
                            let r = next();
                            if r & 0x7 == 0 {
                                specials[(r >> 8) as usize % specials.len()]
                            } else {
                                // Keep magnitudes modest so products stay in range
                                // (still hits inf/NaN occasionally via specials above).
                                ((r >> 16) as u16) & 0x3fff
                            }
                        })
                        .collect();
                    let dense = Value::Tensor(
                        TensorValue::new_half_float_values(
                            dtype,
                            Shape {
                                dims: shape.clone(),
                            },
                            raw.clone(),
                        )
                        .unwrap(),
                    );
                    let boxed = Value::Tensor(
                        TensorValue::new(
                            dtype,
                            Shape {
                                dims: shape.clone(),
                            },
                            raw.iter()
                                .copied()
                                .map(|b| {
                                    if dtype == DType::BF16 {
                                        Literal::BF16Bits(b)
                                    } else {
                                        Literal::F16Bits(b)
                                    }
                                })
                                .collect(),
                        )
                        .unwrap(),
                    );
                    let bits = |v: &Value| -> Vec<u16> {
                        let t = v.as_tensor().unwrap();
                        t.elements
                            .iter()
                            .map(|l| match l {
                                Literal::BF16Bits(b) | Literal::F16Bits(b) => nan_canon(dtype, *b),
                                o => panic!("expected half, got {o:?}"),
                            })
                            .collect()
                    };
                    for primitive in [Primitive::ReduceSum, Primitive::ReduceProd] {
                        // axis=1 -> outer=shape[0], reduce=shape[1], inner=shape[2] (>1).
                        let mut p = BTreeMap::new();
                        p.insert("axes".to_owned(), "1".to_owned());
                        let d = crate::eval_primitive(primitive, std::slice::from_ref(&dense), &p)
                            .unwrap();
                        let g = crate::eval_primitive(primitive, std::slice::from_ref(&boxed), &p)
                            .unwrap();
                        assert_eq!(
                            bits(&d),
                            bits(&g),
                            "{dtype:?} {primitive:?} shape={shape:?} seed={seed} SIMD != generic"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn dense_f16_trailing_axis_reduce_simd_decode_matches_boxed_edge_rows() {
        // Axis=1 is the F16 trailing-axis SIMD decode path. Compare raw output
        // bits with the boxed Literal::F16Bits odometer path, without NaN
        // canonicalization in the assertion, across normal chunks, scalar-fallback
        // chunks, and scalar tail elements.
        let cols = 18u32;
        let raw: Vec<u16> = vec![
            // Row 0: all normal/zero chunks plus a scalar tail.
            0x3c00, 0xbc00, 0x4000, 0xc000, 0x3555, 0xb555, 0x0000, 0x8000, 0x4200, 0xc200, 0x3a00,
            0xba00, 0x2c00, 0xac00, 0x1000, 0x9000, 0x3400, 0xb400,
            // Row 1: first chunk vectorizes; second chunk falls back for special values.
            0x3c00, 0x3800, 0x4000, 0xbc00, 0x3555, 0xb555, 0x0400, 0x8400, 0x0001, 0x03ff, 0x7c00,
            0xfc00, 0x7e01, 0x7d55, 0xfe00, 0x8001, 0x3000, 0xb000,
            // Row 2: first chunk falls back; second chunk vectorizes; tail has a NaN.
            0x7c00, 0xfc00, 0x7e00, 0x0001, 0x8001, 0x0000, 0x8000, 0x3c00, 0x4200, 0xc200, 0x3a00,
            0xba00, 0x2c00, 0xac00, 0x1000, 0x9000, 0x7e11, 0x3555,
        ];
        let rows = (raw.len() as u32) / cols;
        let dims = vec![rows, cols];
        let dense = Value::Tensor(
            TensorValue::new_half_float_values(
                DType::F16,
                Shape { dims: dims.clone() },
                raw.clone(),
            )
            .unwrap(),
        );
        let boxed = Value::Tensor(
            crate::new_boxed(
                DType::F16,
                Shape { dims: dims.clone() },
                raw.iter().copied().map(Literal::F16Bits).collect(),
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

        let output_bits = |value: &Value| -> Vec<u16> {
            let tensor = value.as_tensor().unwrap();
            assert_eq!(tensor.dtype, DType::F16);
            assert_eq!(tensor.shape.dims, vec![rows]);
            tensor
                .elements
                .iter()
                .map(|literal| match literal {
                    Literal::F16Bits(bits) => *bits,
                    other => panic!("expected F16Bits, got {other:?}"),
                })
                .collect()
        };
        let cases: [(Primitive, f64, fn(f64, f64) -> f64); 4] = [
            (Primitive::ReduceSum, 0.0, |a, b| a + b),
            (Primitive::ReduceProd, 1.0, |a, b| a * b),
            (Primitive::ReduceMax, f64::NEG_INFINITY, crate::jax_max_f64),
            (Primitive::ReduceMin, f64::INFINITY, crate::jax_min_f64),
        ];
        let params = BTreeMap::from([("axes".to_owned(), "1".to_owned())]);
        for (primitive, float_init, float_op) in cases {
            let dense_result = eval_reduce_axes(
                primitive,
                std::slice::from_ref(&dense),
                &params,
                0,
                float_init,
                |a, _| a,
                float_op,
            )
            .unwrap();
            let boxed_result = eval_reduce_axes(
                primitive,
                std::slice::from_ref(&boxed),
                &params,
                0,
                float_init,
                |a, _| a,
                float_op,
            )
            .unwrap();
            assert_eq!(
                output_bits(&dense_result),
                output_bits(&boxed_result),
                "{primitive:?} dense F16 trailing-axis SIMD decode diverged from boxed path"
            );
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_bf16_inner_axis_sum_simd_vs_scalar() {
        use std::time::Instant;
        // bf16 sum(x[K,N], axis=0): per-element BF16Bits.as_f64() decode + scalar add
        // vs the SIMD decode + simd add (per-cell k-order preserved -> bit-identical).
        let (k, n) = (4096usize, 1024usize);
        let values: Vec<u16> = (0..k * n)
            .map(
                |i| match Literal::from_bf16_f64(((i % 251) as f64) * 0.03125 - 3.75) {
                    Literal::BF16Bits(b) => b,
                    other => panic!("expected bf16, got {other:?}"),
                },
            )
            .collect();
        let best = |mut f: Box<dyn FnMut() -> f64>| {
            f();
            let mut b = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let vs = values.clone();
        let t_scalar = best(Box::new(move || {
            let mut res = vec![0.0f64; n];
            for kk in 0..k {
                let row = &vs[kk * n..kk * n + n];
                for (slot, &v) in res.iter_mut().zip(row) {
                    *slot += Literal::BF16Bits(v).as_f64().unwrap_or(0.0);
                }
            }
            res.iter().sum()
        }));
        let vv = values.clone();
        let t_simd = best(Box::new(move || {
            simd_sumprod_inner_axis_reduce_bf16(&vv, true, 1, k, n, 0.0)
                .iter()
                .sum()
        }));
        let golden = {
            let g = simd_sumprod_inner_axis_reduce_bf16(&values, true, 1, k, n, 0.0);
            fj_test_utils::fixture_id_from_json(&(
                "inner-axis-sum-bf16",
                g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ))
            .unwrap_or_default()
        };
        println!(
            "BENCH bf16 inner-axis sum(x[{k},{n}],axis=0): scalar={:.4}ms simd={:.4}ms speedup={:.2}x sha256={golden}",
            t_scalar * 1e3,
            t_simd * 1e3,
            t_scalar / t_simd,
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_half_inner_axis_prod_simd_vs_scalar() {
        use std::time::Instant;
        // Half inner>1 PROD (prod(x[K,N], axis=0)): per-element decode + scalar `*`
        // vs the SIMD decode + simd `*` (per-cell k-order preserved -> bit-identical;
        // see simd_inner_axis_prod_half_bit_identical_to_generic_fuzz). Same kernel as
        // the SUM bench; the half DECODE is the SIMD win (the `*` vectorizes too).
        // Inputs kept very close to 1.0 so the column product stays finite over K=4096.
        let (k, n) = (4096usize, 1024usize);
        let best = |mut f: Box<dyn FnMut() -> f64>| {
            f();
            let mut b = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        for dtype in [DType::BF16, DType::F16] {
            let values: Vec<u16> = (0..k * n)
                .map(|i| {
                    let x = 1.0 + (((i % 251) as f64) * 0.0008 - 0.1);
                    match reduce_real_literal(dtype, x) {
                        Literal::BF16Bits(b) | Literal::F16Bits(b) => b,
                        other => panic!("expected half, got {other:?}"),
                    }
                })
                .collect();
            let vs = values.clone();
            let dt = dtype;
            let t_scalar = best(Box::new(move || {
                let mut res = vec![1.0f64; n];
                for kk in 0..k {
                    let row = &vs[kk * n..kk * n + n];
                    for (slot, &v) in res.iter_mut().zip(row) {
                        let d = if dt == DType::BF16 {
                            Literal::BF16Bits(v).as_f64().unwrap_or(0.0)
                        } else {
                            Literal::F16Bits(v).as_f64().unwrap_or(0.0)
                        };
                        *slot *= d;
                    }
                }
                res.iter().sum()
            }));
            let vv = values.clone();
            let dt2 = dtype;
            let t_simd = best(Box::new(move || {
                if dt2 == DType::BF16 {
                    simd_sumprod_inner_axis_reduce_bf16(&vv, false, 1, k, n, 1.0)
                } else {
                    simd_sumprod_inner_axis_reduce_f16(&vv, false, 1, k, n, 1.0)
                }
                .iter()
                .sum()
            }));
            let golden = {
                let g = if dtype == DType::BF16 {
                    simd_sumprod_inner_axis_reduce_bf16(&values, false, 1, k, n, 1.0)
                } else {
                    simd_sumprod_inner_axis_reduce_f16(&values, false, 1, k, n, 1.0)
                };
                fj_test_utils::fixture_id_from_json(&(
                    "inner-axis-prod-half",
                    g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ))
                .unwrap_or_default()
            };
            println!(
                "BENCH {dtype:?} inner-axis prod(x[{k},{n}],axis=0): scalar={:.4}ms simd={:.4}ms speedup={:.2}x sha256={golden}",
                t_scalar * 1e3,
                t_simd * 1e3,
                t_scalar / t_simd,
            );
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_f16_inner_axis_max_simd_vs_scalar() {
        use std::time::Instant;
        // f16 max(x[K,N], axis=0): per-element F16Bits.as_f64() decode + scalar jax_max
        // vs the SIMD f16_widen8 decode + simd_max (needs_scalar fallback per chunk).
        let (k, n) = (4096usize, 1024usize);
        let values: Vec<u16> = (0..k * n)
            .map(
                |i| match Literal::from_f16_f64(((i % 251) as f64) * 0.03125 - 3.75) {
                    Literal::F16Bits(b) => b,
                    other => panic!("expected f16, got {other:?}"),
                },
            )
            .collect();
        let best = |mut f: Box<dyn FnMut() -> f64>| {
            f();
            let mut b = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let vs = values.clone();
        let t_scalar = best(Box::new(move || {
            let mut res = vec![f64::NEG_INFINITY; n];
            for kk in 0..k {
                let row = &vs[kk * n..kk * n + n];
                for (slot, &v) in res.iter_mut().zip(row) {
                    let dv = Literal::F16Bits(v).as_f64().unwrap_or(0.0);
                    *slot = if slot.is_nan() || dv.is_nan() {
                        f64::NAN
                    } else {
                        slot.max(dv)
                    };
                }
            }
            res.iter().sum()
        }));
        let vv = values.clone();
        let t_simd = best(Box::new(move || {
            simd_minmax_inner_axis_reduce_f16(&vv, true, 1, k, n)
                .iter()
                .sum()
        }));
        let golden = {
            let g = simd_minmax_inner_axis_reduce_f16(&values, true, 1, k, n);
            fj_test_utils::fixture_id_from_json(&(
                "inner-axis-max-f16",
                g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ))
            .unwrap_or_default()
        };
        println!(
            "BENCH f16 inner-axis max(x[{k},{n}],axis=0): scalar={:.4}ms simd={:.4}ms speedup={:.2}x sha256={golden}",
            t_scalar * 1e3,
            t_simd * 1e3,
            t_scalar / t_simd,
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_inner_axis_max_simd_vs_scalar() {
        use std::time::Instant;
        // bf16 max(x[K,N], axis=0): outer=1, reduce=K, inner=N. The per-element bf16
        // DECODE is the compute SIMD wins (the native f64/f32 column fold is already
        // LLVM-autovectorized, so this lever targets the half-float decode path).
        let (k, n) = (4096usize, 1024usize);
        let values: Vec<u16> = (0..k * n)
            .map(
                |i| match Literal::from_bf16_f64(((i % 251) as f64) * 0.03125 - 3.75) {
                    Literal::BF16Bits(b) => b,
                    other => panic!("expected bf16, got {other:?}"),
                },
            )
            .collect();
        let best = |mut f: Box<dyn FnMut() -> f64>| {
            f();
            let mut b = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        // prior path: per-element Literal::BF16Bits.as_f64() decode + scalar jax_max.
        let vs = values.clone();
        let t_scalar = best(Box::new(move || {
            let mut res = vec![f64::NEG_INFINITY; n];
            for kk in 0..k {
                let row = &vs[kk * n..kk * n + n];
                for (slot, &v) in res.iter_mut().zip(row) {
                    let dv = Literal::BF16Bits(v).as_f64().unwrap_or(0.0);
                    *slot = if slot.is_nan() || dv.is_nan() {
                        f64::NAN
                    } else {
                        slot.max(dv)
                    };
                }
            }
            res.iter().sum()
        }));
        let vv = values.clone();
        let t_simd = best(Box::new(move || {
            simd_minmax_inner_axis_reduce_bf16(&vv, true, 1, k, n)
                .iter()
                .sum()
        }));
        let golden = {
            let g = simd_minmax_inner_axis_reduce_bf16(&values, true, 1, k, n);
            fj_test_utils::fixture_id_from_json(&(
                "inner-axis-max-bf16",
                g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ))
            .unwrap_or_default()
        };
        println!(
            "BENCH bf16 inner-axis max(x[{k},{n}],axis=0): scalar={:.4}ms simd={:.4}ms speedup={:.2}x sha256={golden}",
            t_scalar * 1e3,
            t_simd * 1e3,
            t_scalar / t_simd,
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_f16_axis_max_simd_vs_scalar_fold() {
        use std::time::Instant;
        // Compare the new full-SIMD F16 max (simd_minmax_axis_reduce_f16) against the
        // prior SIMD-decode + SCALAR-fold path (fold_f16_axis_block with jax-max).
        let (rows, cols) = (4096usize, 1024usize);
        let raw: Vec<u16> = (0..rows * cols)
            .map(
                |i| match Literal::from_f16_f64(((i % 251) as f64) * 0.03125 - 3.75) {
                    Literal::F16Bits(bits) => bits,
                    other => panic!("expected f16, got {other:?}"),
                },
            )
            .collect();
        let jax_max = |a: f64, b: f64| {
            if a.is_nan() || b.is_nan() {
                f64::NAN
            } else {
                a.max(b)
            }
        };
        let best = |mut f: Box<dyn FnMut() -> f64>| {
            f();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                let r = f();
                std::hint::black_box(r);
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let raw_s = raw.clone();
        let t_scalar = best(Box::new(move || {
            let mut acc = 0.0;
            for o in 0..rows {
                acc += fold_f16_axis_block(
                    &raw_s[o * cols..o * cols + cols],
                    f64::NEG_INFINITY,
                    &jax_max,
                );
            }
            acc
        }));
        let raw_v = raw.clone();
        let t_simd = best(Box::new(move || {
            simd_minmax_axis_reduce_f16(&raw_v, true, rows, cols)
                .iter()
                .sum()
        }));
        let golden = {
            let g = simd_minmax_axis_reduce_f16(&raw, true, rows, cols);
            fj_test_utils::fixture_id_from_json(&(
                "f16-axis-max",
                g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ))
            .unwrap_or_default()
        };
        println!(
            "BENCH f16 axis-max [{rows},{cols}]: scalar_fold={:.4}ms full_simd={:.4}ms speedup={:.2}x sha256={golden}",
            t_scalar * 1e3,
            t_simd * 1e3,
            t_scalar / t_simd,
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_f16_reduce_sum_axis_simd_decode_vs_scalar_dense() {
        use std::time::Instant;

        let (rows, cols) = (4096usize, 1024usize);
        let raw: Vec<u16> = (0..rows * cols)
            .map(|i| {
                let x = ((i % 251) as f64) * 0.03125 - 3.75;
                match Literal::from_f16_f64(x) {
                    Literal::F16Bits(bits) => bits,
                    other => panic!("expected f16 literal, got {other:?}"),
                }
            })
            .collect();
        let tensor = TensorValue::new_half_float_values(
            DType::F16,
            Shape {
                dims: vec![rows as u32, cols as u32],
            },
            raw,
        )
        .unwrap();
        let values = tensor.elements.as_half_float_slice().unwrap();
        let kept_axes = [0usize];
        let out_dims = [rows as u32];
        let out_count = rows;
        let add = |a: f64, b: f64| a + b;

        let scalar_reference = dense_f64_axis_reduce(
            &tensor,
            values,
            |b| Literal::F16Bits(b).as_f64().unwrap_or(0.0),
            &kept_axes,
            &out_dims,
            out_count,
            0.0,
            &add,
        )
        .unwrap();
        let simd_reference =
            dense_f16_trailing_axis_reduce(&tensor, values, &kept_axes, out_count, 0.0, &add)
                .unwrap();
        assert_eq!(
            scalar_reference
                .iter()
                .map(|v| Literal::from_f16_f64(*v))
                .collect::<Vec<_>>(),
            simd_reference
                .iter()
                .map(|v| Literal::from_f16_f64(*v))
                .collect::<Vec<_>>(),
            "SIMD F16 axis reduce must preserve the old scalar dense output"
        );

        let time_scalar = || {
            let mut best = f64::MAX;
            for _ in 0..12 {
                let start = Instant::now();
                let out = dense_f64_axis_reduce(
                    &tensor,
                    values,
                    |b| Literal::F16Bits(b).as_f64().unwrap_or(0.0),
                    &kept_axes,
                    &out_dims,
                    out_count,
                    0.0,
                    &add,
                )
                .unwrap();
                std::hint::black_box(out);
                best = best.min(start.elapsed().as_secs_f64());
            }
            best
        };
        let time_simd = || {
            let mut best = f64::MAX;
            for _ in 0..12 {
                let start = Instant::now();
                let out = dense_f16_trailing_axis_reduce(
                    &tensor, values, &kept_axes, out_count, 0.0, &add,
                )
                .unwrap();
                std::hint::black_box(out);
                best = best.min(start.elapsed().as_secs_f64());
            }
            best
        };

        let scalar = time_scalar();
        let simd = time_simd();
        let golden_bits: Vec<u16> = simd_reference
            .iter()
            .map(|v| Literal::from_f16_f64(*v))
            .map(|lit| {
                let Literal::F16Bits(bits) = lit else {
                    unreachable!("f16 reduce emits f16");
                };
                bits
            })
            .collect();
        let golden = fj_test_utils::fixture_id_from_json(&golden_bits)
            .expect("F16 axis reduce golden digest should build");
        assert_eq!(
            golden, "a321e41484cba20e931270f7c083710b843a9a031cb6c90830e82ab112739b4e",
            "F16 trailing-axis reduce golden output changed"
        );
        println!(
            "BENCH f16 reduce_sum axis1 [{rows},{cols}]: scalar_dense={:.4}ms simd_decode={:.4}ms speedup={:.2}x sha256={golden}",
            scalar * 1e3,
            simd * 1e3,
            scalar / simd
        );
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
        // Boxed (Vec<Literal>) reference: TensorValue::new now densifies all-I64 inputs
        // (fj-core i64-densify), so build the boxed buffer explicitly to keep exercising
        // the generic path against the dense reduce fast path.
        let literal = Value::Tensor(
            TensorValue::new_with_literal_buffer(
                DType::I64,
                Shape { dims: dims.clone() },
                fj_core::LiteralBuffer::new(data.iter().copied().map(Literal::I64).collect()),
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
                crate::new_boxed(
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
            crate::new_boxed(
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
    fn bool_word_reduce_bit_identical_to_literal_path() {
        let word_tensor = |dims: Vec<u32>, data: &[bool]| -> Value {
            let mut words = vec![0_u64; data.len().div_ceil(u64::BITS as usize)];
            for (index, &flag) in data.iter().enumerate() {
                if flag {
                    words[index / u64::BITS as usize] |= 1_u64 << (index % u64::BITS as usize);
                }
            }
            Value::Tensor(
                TensorValue::new_with_literal_buffer(
                    DType::Bool,
                    Shape { dims },
                    LiteralBuffer::from_bool_words(words, data.len()).unwrap(),
                )
                .unwrap(),
            )
        };
        let literal_tensor = |dims: Vec<u32>, data: &[bool]| -> Value {
            Value::Tensor(
                TensorValue::new(
                    DType::Bool,
                    Shape { dims },
                    data.iter().copied().map(Literal::Bool).collect(),
                )
                .unwrap(),
            )
        };
        let bools = |v: &Value| -> Vec<bool> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|literal| matches!(literal, Literal::Bool(true)))
                .collect()
        };

        for len in [0usize, 1, 63, 64, 65, 127, 128, 129] {
            let data: Vec<bool> = (0..len).map(|i| i % 5 != 2 && i % 11 != 7).collect();
            let word = word_tensor(vec![len as u32], &data);
            assert!(word.as_tensor().unwrap().elements.as_bool_words().is_some());
            let literal = literal_tensor(vec![len as u32], &data);
            let params = BTreeMap::new();
            for prim in [
                Primitive::ReduceAnd,
                Primitive::ReduceOr,
                Primitive::ReduceXor,
            ] {
                let w = crate::eval_primitive(prim, std::slice::from_ref(&word), &params).unwrap();
                let l =
                    crate::eval_primitive(prim, std::slice::from_ref(&literal), &params).unwrap();
                assert_eq!(
                    w.as_scalar_literal(),
                    l.as_scalar_literal(),
                    "{prim:?} len={len}"
                );
            }
        }

        let dims = vec![3_u32, 43];
        let data: Vec<bool> = (0..129).map(|i| (i * 7 + 3) % 17 < 9).collect();
        let word = word_tensor(dims.clone(), &data);
        let literal = literal_tensor(dims, &data);
        for prim in [
            Primitive::ReduceAnd,
            Primitive::ReduceOr,
            Primitive::ReduceXor,
        ] {
            for axes in ["0", "1"] {
                let mut params = BTreeMap::new();
                params.insert("axes".to_owned(), axes.to_owned());
                let w = crate::eval_primitive(prim, std::slice::from_ref(&word), &params).unwrap();
                let l =
                    crate::eval_primitive(prim, std::slice::from_ref(&literal), &params).unwrap();
                assert_eq!(bools(&w), bools(&l), "{prim:?} axes={axes}");
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
            crate::new_boxed(
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

    /// SIMD axis-reduce max/min (`axis=-1`, inner==1) must be bit-identical to the
    /// boxed scalar fold across the SIMD hazards in a PER-CELL setting: a tail
    /// (reduce length not a multiple of the lane count), and — critically — cells
    /// whose reduced value is exactly ±0 (so the per-cell SIMD ±0 fallback fires).
    #[test]
    fn simd_axis_minmax_bit_identical_to_scalar_fold() {
        let rows = 6u32;
        let cols = 11u32; // not a multiple of 8 (f64) or 16 (f32): exercises tails
        let n = (rows * cols) as usize;
        let nan_payload = f64::from_bits(0x7ff8_0000_0000_0001);
        let mut data = vec![0.0f64; n];
        let c = cols as usize;
        for r in 0..rows as usize {
            for j in 0..c {
                let idx = r * c + j;
                data[idx] = match r {
                    // all negative except a trailing -0.0 ⇒ max cell result is ±0.
                    0 => {
                        if j == c - 1 {
                            -0.0
                        } else {
                            -1.0 - j as f64
                        }
                    }
                    // +0.0 then a later -0.0, all else negative ⇒ max result ±0 (last-zero).
                    1 => {
                        if j == 2 {
                            0.0
                        } else if j == c - 1 {
                            -0.0
                        } else {
                            -2.0
                        }
                    }
                    // all positive except -0.0 then +0.0 ⇒ min result ±0 (last-zero).
                    2 => {
                        if j == 4 {
                            -0.0
                        } else if j == c - 2 {
                            0.0
                        } else {
                            3.0
                        }
                    }
                    // NaN payload in the row ⇒ canonical NaN.
                    3 => {
                        if j == 5 {
                            nan_payload
                        } else {
                            j as f64 - 4.0
                        }
                    }
                    // ±inf mixed.
                    4 => match j {
                        1 => f64::INFINITY,
                        8 => f64::NEG_INFINITY,
                        _ => (j as f64 - 5.0) * 0.5,
                    },
                    // ordinary values.
                    _ => (j as f64 - 5.0) * 1.25 + 0.3,
                };
            }
        }
        let dims = vec![rows, cols];
        let dense = Value::Tensor(
            TensorValue::new_f64_values(Shape { dims: dims.clone() }, data.clone()).unwrap(),
        );
        assert!(dense.as_tensor().unwrap().elements.as_f64_slice().is_some());
        let boxed = Value::Tensor(
            crate::new_boxed(
                DType::F64,
                Shape { dims: dims.clone() },
                data.iter().copied().map(Literal::from_f64).collect(),
            )
            .unwrap(),
        );
        assert!(boxed.as_tensor().unwrap().elements.as_f64_slice().is_none());

        let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        let dense32 = Value::Tensor(
            TensorValue::new_f32_values(Shape { dims: dims.clone() }, f32_data.clone()).unwrap(),
        );
        let boxed32 = Value::Tensor(
            crate::new_boxed(
                DType::F32,
                Shape { dims: dims.clone() },
                f32_data.iter().copied().map(Literal::from_f32).collect(),
            )
            .unwrap(),
        );

        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "1".to_owned());
        for primitive in [Primitive::ReduceMax, Primitive::ReduceMin] {
            let d =
                crate::eval_primitive(primitive, std::slice::from_ref(&dense), &params).unwrap();
            let b =
                crate::eval_primitive(primitive, std::slice::from_ref(&boxed), &params).unwrap();
            assert_eq!(
                extract_f64_vec(&d)
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                extract_f64_vec(&b)
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                "f64 axis {primitive:?}: SIMD != scalar"
            );

            let d32 =
                crate::eval_primitive(primitive, std::slice::from_ref(&dense32), &params).unwrap();
            let b32 =
                crate::eval_primitive(primitive, std::slice::from_ref(&boxed32), &params).unwrap();
            let bits32 = |v: &Value| -> Vec<u32> {
                v.as_tensor()
                    .unwrap()
                    .elements
                    .iter()
                    .map(|l| match l {
                        Literal::F32Bits(b) => *b,
                        other => panic!("expected f32, got {other:?}"),
                    })
                    .collect()
            };
            assert_eq!(
                bits32(&d32),
                bits32(&b32),
                "f32 axis {primitive:?}: SIMD != scalar"
            );
        }
    }

    // ── Reduce along axes ──

    #[test]
    fn threaded_complex_trailing_axis_reduce_bit_identical_to_serial() {
        // Complex128 trailing-axis (inner==1) prod/max/min over a tensor large
        // enough to fan rows across threads (outer*reduce >= 1<<18, outer > 1).
        // Each output row is an independent dependency chain, so the threaded
        // result must be bit-identical to a hand-written serial fold for any row
        // partition. (Sum stays serial — verified to match the same reference.)
        let outer = 2048usize;
        let reduce = 128usize; // outer*reduce = 262144 = 1<<18.
        let n = outer * reduce;
        // Magnitudes near 1 so the 128-fold product stays finite and varied.
        let cplx: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let a = 1.0 + (((i * 2_654_435_761) % 97) as f64 - 48.0) * 1e-3;
                let b = (((i * 40_503) % 89) as f64 - 44.0) * 1e-3;
                (a, b)
            })
            .collect();
        let tensor = Value::Tensor(
            TensorValue::new_complex_values(
                DType::Complex128,
                Shape {
                    dims: vec![outer as u32, reduce as u32],
                },
                cplx.clone(),
            )
            .unwrap(),
        );
        assert!(
            tensor
                .as_tensor()
                .unwrap()
                .elements
                .as_complex_slice()
                .is_some(),
            "test must exercise the dense-complex (threaded) path"
        );
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "1".to_owned());

        let cases: [(Primitive, f64, fn(f64, f64) -> f64); 4] = [
            (Primitive::ReduceSum, 0.0, |a, b| a + b),
            (Primitive::ReduceProd, 1.0, |a, b| a * b),
            (Primitive::ReduceMax, f64::NEG_INFINITY, crate::jax_max_f64),
            (Primitive::ReduceMin, f64::INFINITY, crate::jax_min_f64),
        ];
        for (prim, finit, fop) in cases {
            let got = eval_reduce_axes(
                prim,
                std::slice::from_ref(&tensor),
                &params,
                0,
                finit,
                |a, _| a,
                fop,
            )
            .unwrap();
            let got_t = got.as_tensor().unwrap();
            assert_eq!(got_t.elements.len(), outer, "{prim:?} output row count");

            // Independent serial reference, mirroring the production fold exactly.
            let (init_re, init_im) = if prim == Primitive::ReduceProd {
                (1.0, 0.0)
            } else {
                (finit, finit)
            };
            for o in 0..outer {
                let row = &cplx[o * reduce..(o + 1) * reduce];
                let (want_re, want_im) = match prim {
                    Primitive::ReduceProd => {
                        let mut acc = (init_re, init_im);
                        for &(re, im) in row {
                            acc = (acc.0 * re - acc.1 * im, acc.0 * im + acc.1 * re);
                        }
                        acc
                    }
                    Primitive::ReduceMax => {
                        let mut best = (init_re, init_im);
                        for &(re, im) in row {
                            if super::complex_lex_cmp((re, im), best).is_gt() {
                                best = (re, im);
                            }
                        }
                        best
                    }
                    Primitive::ReduceMin => {
                        let mut best = (init_re, init_im);
                        for &(re, im) in row {
                            if super::complex_lex_cmp((re, im), best).is_lt() {
                                best = (re, im);
                            }
                        }
                        best
                    }
                    _ => {
                        let mut acc = (init_re, init_im);
                        for &(re, im) in row {
                            acc = (acc.0 + re, acc.1 + im);
                        }
                        acc
                    }
                };
                let (got_re, got_im) = match got_t.elements.get(o) {
                    Some(Literal::Complex128Bits(rb, ib)) => {
                        (f64::from_bits(*rb), f64::from_bits(*ib))
                    }
                    other => panic!("expected complex128, got {other:?}"),
                };
                assert_eq!(
                    (got_re.to_bits(), got_im.to_bits()),
                    (want_re.to_bits(), want_im.to_bits()),
                    "{prim:?} row {o} threaded != serial reference"
                );
            }
        }
    }

    #[test]
    fn complex_cumulative_dense_matches_boxed() {
        // Dense complex cumulative (new_complex_values storage → dense path) must
        // equal the boxed-Literal construction, for cumsum/cumprod/cummax/cummin,
        // forward and reverse, on a 2-D [lines, axis] tensor.
        let (lines, axdim) = (5usize, 9usize);
        let cplx: Vec<(f64, f64)> = (0..lines * axdim)
            .map(|k| {
                let a = (((k * 2_654_435_761) % 41) as f64 - 20.0) * 0.05;
                let b = (((k * 40_503) % 37) as f64 - 18.0) * 0.05;
                (a, b)
            })
            .collect();
        let dims = vec![lines as u32, axdim as u32];
        let dense = Value::Tensor(
            TensorValue::new_complex_values(
                DType::Complex128,
                Shape { dims: dims.clone() },
                cplx.clone(),
            )
            .unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::Complex128,
                Shape { dims: dims.clone() },
                cplx.iter()
                    .map(|&(re, im)| Literal::from_complex128(re, im))
                    .collect(),
            )
            .unwrap(),
        );
        let bits = |v: &Value| -> Vec<(u64, u64)> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::Complex128Bits(re, im) => (*re, *im),
                    o => panic!("expected Complex128Bits, got {o:?}"),
                })
                .collect()
        };
        for prim in ["cumsum", "cumprod", "cummax", "cummin"] {
            for rev in ["false", "true"] {
                let p = BTreeMap::from([
                    ("axis".to_owned(), "1".to_owned()),
                    ("reverse".to_owned(), rev.to_owned()),
                ]);
                let d = crate::eval_primitive(cum_prim(prim), std::slice::from_ref(&dense), &p)
                    .unwrap();
                let g = crate::eval_primitive(cum_prim(prim), std::slice::from_ref(&boxed), &p)
                    .unwrap();
                assert_eq!(
                    bits(&d),
                    bits(&g),
                    "complex {prim} rev={rev}: dense != boxed"
                );
            }
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_complex_cumsum_dense_vs_generic() {
        use std::time::Instant;
        let (lines, axdim) = (4096usize, 1024usize);
        let cplx: Vec<(f64, f64)> = (0..lines * axdim)
            .map(|k| ((k % 251) as f64 * 0.01, (k % 97) as f64 * 0.02))
            .collect();
        let dims = vec![lines as u32, axdim as u32];
        let dense = Value::Tensor(
            TensorValue::new_complex_values(
                DType::Complex128,
                Shape { dims: dims.clone() },
                cplx.clone(),
            )
            .unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new(
                DType::Complex128,
                Shape { dims: dims.clone() },
                cplx.iter()
                    .map(|&(re, im)| Literal::from_complex128(re, im))
                    .collect(),
            )
            .unwrap(),
        );
        let p = BTreeMap::from([("axis".to_owned(), "1".to_owned())]);
        let csum = |v: Value| -> u64 {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .fold(0u64, |a, l| match l {
                    Literal::Complex128Bits(re, _) => a.wrapping_add(*re),
                    _ => a,
                })
        };
        let time = |input: &Value| {
            let _ =
                crate::eval_primitive(Primitive::Cumsum, std::slice::from_ref(input), &p).unwrap();
            let mut best = f64::MAX;
            for _ in 0..10 {
                let t = Instant::now();
                let _ = crate::eval_primitive(Primitive::Cumsum, std::slice::from_ref(input), &p)
                    .unwrap();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        assert_eq!(
            csum(
                crate::eval_primitive(Primitive::Cumsum, std::slice::from_ref(&dense), &p).unwrap()
            ),
            csum(
                crate::eval_primitive(Primitive::Cumsum, std::slice::from_ref(&boxed), &p).unwrap()
            ),
            "parity"
        );
        let g = time(&boxed);
        let d = time(&dense);
        println!(
            "BENCH complex128 cumsum axis1 [{lines},{axdim}]: generic={:.4}ms dense={:.4}ms speedup={:.2}x",
            g * 1e3,
            d * 1e3,
            g / d
        );
    }

    fn cum_prim(name: &str) -> Primitive {
        match name {
            "cumprod" => Primitive::Cumprod,
            "cummax" => Primitive::Cummax,
            "cummin" => Primitive::Cummin,
            _ => Primitive::Cumsum,
        }
    }

    #[test]
    fn threaded_complex_inner_axis_reduce_bit_identical_to_serial() {
        // Complex128 leading/middle-axis (inner>1) prod/max/min over a 3-D tensor
        // [O, R, I] reducing the MIDDLE axis (axis=1) -> output [O, I], inner=I>1.
        // Large enough to thread (O*R*I >= 1<<18, O>1). Each output cell (o,i) folds
        // the R taps in ascending r; the threaded result must equal a per-cell serial
        // reference. (Sum stays serial — verified to match the same reference.)
        let (o_dim, r_dim, i_dim) = (512usize, 16usize, 32usize); // 262144 = 1<<18
        let n = o_dim * r_dim * i_dim;
        let cplx: Vec<(f64, f64)> = (0..n)
            .map(|k| {
                let a = 1.0 + (((k * 2_654_435_761) % 97) as f64 - 48.0) * 1e-3;
                let b = (((k * 40_503) % 89) as f64 - 44.0) * 1e-3;
                (a, b)
            })
            .collect();
        let tensor = Value::Tensor(
            TensorValue::new_complex_values(
                DType::Complex128,
                Shape {
                    dims: vec![o_dim as u32, r_dim as u32, i_dim as u32],
                },
                cplx.clone(),
            )
            .unwrap(),
        );
        assert!(
            tensor
                .as_tensor()
                .unwrap()
                .elements
                .as_complex_slice()
                .is_some(),
            "must exercise the dense-complex (threaded) path"
        );
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "1".to_owned());

        let cases: [(Primitive, f64, fn(f64, f64) -> f64); 4] = [
            (Primitive::ReduceSum, 0.0, |a, b| a + b),
            (Primitive::ReduceProd, 1.0, |a, b| a * b),
            (Primitive::ReduceMax, f64::NEG_INFINITY, crate::jax_max_f64),
            (Primitive::ReduceMin, f64::INFINITY, crate::jax_min_f64),
        ];
        for (prim, finit, fop) in cases {
            let got = eval_reduce_axes(
                prim,
                std::slice::from_ref(&tensor),
                &params,
                0,
                finit,
                |a, _| a,
                fop,
            )
            .unwrap();
            let got_t = got.as_tensor().unwrap();
            assert_eq!(got_t.elements.len(), o_dim * i_dim, "{prim:?} output count");

            let (init_re, init_im) = if prim == Primitive::ReduceProd {
                (1.0, 0.0)
            } else {
                (finit, finit)
            };
            for o in 0..o_dim {
                for i in 0..i_dim {
                    let mut acc = (init_re, init_im);
                    for r in 0..r_dim {
                        let (re, im) = cplx[(o * r_dim + r) * i_dim + i];
                        acc = match prim {
                            Primitive::ReduceProd => {
                                (acc.0 * re - acc.1 * im, acc.0 * im + acc.1 * re)
                            }
                            Primitive::ReduceMax => {
                                if super::complex_lex_cmp((re, im), acc).is_gt() {
                                    (re, im)
                                } else {
                                    acc
                                }
                            }
                            Primitive::ReduceMin => {
                                if super::complex_lex_cmp((re, im), acc).is_lt() {
                                    (re, im)
                                } else {
                                    acc
                                }
                            }
                            _ => (acc.0 + re, acc.1 + im),
                        };
                    }
                    let out_idx = o * i_dim + i;
                    let (got_re, got_im) = match got_t.elements.get(out_idx) {
                        Some(Literal::Complex128Bits(rb, ib)) => {
                            (f64::from_bits(*rb), f64::from_bits(*ib))
                        }
                        other => panic!("expected complex128, got {other:?}"),
                    };
                    assert_eq!(
                        (got_re.to_bits(), got_im.to_bits()),
                        (acc.0.to_bits(), acc.1.to_bits()),
                        "{prim:?} cell ({o},{i}) threaded != serial reference"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_threaded_complex_inner_axis_prod() {
        use std::time::Instant;
        // Complex128 prod over the MIDDLE axis of [O,R,I] (inner=I>1): threaded
        // eval_reduce_axes vs the identical single-threaded per-cell complex-multiply
        // chain. Bit-identical; digested zero-copy.
        let (o_dim, r_dim, i_dim) = (4096usize, 64usize, 64usize);
        let n = o_dim * r_dim * i_dim;
        let cplx: Vec<(f64, f64)> = (0..n)
            .map(|k| {
                let a = 1.0 + (((k * 2_654_435_761) % 97) as f64 - 48.0) * 1e-4;
                let b = (((k * 40_503) % 89) as f64 - 44.0) * 1e-4;
                (a, b)
            })
            .collect();
        let tensor = Value::Tensor(
            TensorValue::new_complex_values(
                DType::Complex128,
                Shape {
                    dims: vec![o_dim as u32, r_dim as u32, i_dim as u32],
                },
                cplx.clone(),
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "1".to_owned());
        let best = |mut f: Box<dyn FnMut() -> u64>| {
            f();
            let mut b = f64::MAX;
            let mut d = 0u64;
            for _ in 0..5 {
                let t = Instant::now();
                d = std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            (b, d)
        };

        let cs = cplx.clone();
        let (t_serial, d_serial) = best(Box::new(move || {
            let mut acc = 0u64;
            for o in 0..o_dim {
                for i in 0..i_dim {
                    let mut a = (1.0_f64, 0.0_f64);
                    for r in 0..r_dim {
                        let (re, im) = cs[(o * r_dim + r) * i_dim + i];
                        a = (a.0 * re - a.1 * im, a.0 * im + a.1 * re);
                    }
                    acc ^= a.0.to_bits() ^ a.1.to_bits();
                }
            }
            acc
        }));

        let (t_threaded, d_threaded) = best(Box::new(move || {
            let out = eval_reduce_axes(
                Primitive::ReduceProd,
                std::slice::from_ref(&tensor),
                &params,
                0,
                1.0,
                |a, _| a,
                |a, b| a * b,
            )
            .unwrap();
            out.as_tensor()
                .unwrap()
                .elements
                .iter()
                .fold(0u64, |acc, l| match l {
                    Literal::Complex128Bits(rb, ib) => acc ^ rb ^ ib,
                    _ => acc,
                })
        }));

        assert_eq!(
            d_serial, d_threaded,
            "threaded inner-axis prod digest must match serial"
        );
        println!(
            "BENCH complex128 prod(x[{o_dim},{r_dim},{i_dim}],axis=1): serial={:.4}ms threaded={:.4}ms speedup={:.2}x",
            t_serial * 1e3,
            t_threaded * 1e3,
            t_serial / t_threaded,
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_threaded_complex_trailing_axis_prod() {
        use std::time::Instant;
        // Complex128 prod over the trailing axis: threaded eval_reduce_axes vs the
        // identical single-threaded per-row complex-multiply chain. Bit-identical;
        // digested zero-copy.
        let (outer, reduce) = (16384usize, 256usize);
        let n = outer * reduce;
        let cplx: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let a = 1.0 + (((i * 2_654_435_761) % 97) as f64 - 48.0) * 1e-4;
                let b = (((i * 40_503) % 89) as f64 - 44.0) * 1e-4;
                (a, b)
            })
            .collect();
        let tensor = Value::Tensor(
            TensorValue::new_complex_values(
                DType::Complex128,
                Shape {
                    dims: vec![outer as u32, reduce as u32],
                },
                cplx.clone(),
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "1".to_owned());
        let best = |mut f: Box<dyn FnMut() -> u64>| {
            f();
            let mut b = f64::MAX;
            let mut d = 0u64;
            for _ in 0..5 {
                let t = Instant::now();
                d = std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            (b, d)
        };

        let cs = cplx.clone();
        let (t_serial, d_serial) = best(Box::new(move || {
            let mut acc = 0u64;
            for o in 0..outer {
                let row = &cs[o * reduce..(o + 1) * reduce];
                let mut a = (1.0_f64, 0.0_f64);
                for &(re, im) in row {
                    a = (a.0 * re - a.1 * im, a.0 * im + a.1 * re);
                }
                acc ^= a.0.to_bits() ^ a.1.to_bits();
            }
            acc
        }));

        let (t_threaded, d_threaded) = best(Box::new(move || {
            let out = eval_reduce_axes(
                Primitive::ReduceProd,
                std::slice::from_ref(&tensor),
                &params,
                0,
                1.0,
                |a, _| a,
                |a, b| a * b,
            )
            .unwrap();
            out.as_tensor()
                .unwrap()
                .elements
                .iter()
                .fold(0u64, |acc, l| match l {
                    Literal::Complex128Bits(rb, ib) => acc ^ rb ^ ib,
                    _ => acc,
                })
        }));

        assert_eq!(
            d_serial, d_threaded,
            "threaded complex prod digest must match serial"
        );
        println!(
            "BENCH complex128 prod(x[{outer},{reduce}],axis=-1): serial={:.4}ms threaded={:.4}ms speedup={:.2}x",
            t_serial * 1e3,
            t_threaded * 1e3,
            t_serial / t_threaded,
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_threaded_complex_trailing_axis_sum() {
        use std::time::Instant;
        // Complex128 SUM over the trailing axis (the lightest reducer: 2 adds/elem,
        // 16 bytes/elem) — checks whether threading the independent rows still pays
        // off or whether sum is memory-bandwidth-bound.
        let (outer, reduce) = (16384usize, 256usize);
        let n = outer * reduce;
        let cplx: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let a = (((i * 2_654_435_761) % 997) as f64 - 498.0) * 1e-3;
                let b = (((i * 40_503) % 991) as f64 - 495.0) * 1e-3;
                (a, b)
            })
            .collect();
        let tensor = Value::Tensor(
            TensorValue::new_complex_values(
                DType::Complex128,
                Shape {
                    dims: vec![outer as u32, reduce as u32],
                },
                cplx.clone(),
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "1".to_owned());
        let best = |mut f: Box<dyn FnMut() -> u64>| {
            f();
            let mut b = f64::MAX;
            let mut d = 0u64;
            for _ in 0..5 {
                let t = Instant::now();
                d = std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            (b, d)
        };
        let cs = cplx.clone();
        let (t_serial, d_serial) = best(Box::new(move || {
            let mut acc = 0u64;
            for o in 0..outer {
                let row = &cs[o * reduce..(o + 1) * reduce];
                let mut a = (0.0_f64, 0.0_f64);
                for &(re, im) in row {
                    a = (a.0 + re, a.1 + im);
                }
                acc ^= a.0.to_bits() ^ a.1.to_bits();
            }
            acc
        }));
        let (t_threaded, d_threaded) = best(Box::new(move || {
            let out = eval_reduce_axes(
                Primitive::ReduceSum,
                std::slice::from_ref(&tensor),
                &params,
                0,
                0.0,
                |a, _| a,
                |a, b| a + b,
            )
            .unwrap();
            out.as_tensor()
                .unwrap()
                .elements
                .iter()
                .fold(0u64, |acc, l| match l {
                    Literal::Complex128Bits(rb, ib) => acc ^ rb ^ ib,
                    _ => acc,
                })
        }));
        assert_eq!(
            d_serial, d_threaded,
            "threaded complex sum digest must match serial"
        );
        println!(
            "BENCH complex128 sum(x[{outer},{reduce}],axis=-1): serial={:.4}ms threaded={:.4}ms speedup={:.2}x",
            t_serial * 1e3,
            t_threaded * 1e3,
            t_serial / t_threaded,
        );
    }

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
    fn reduce_axes_i32_preserves_dtype_and_wraps() {
        // b6w3l gap (1): an int32 partial-axis reduction must STAY int32 (JAX/XLA
        // does not widen to int64) and overflow must wrap two's-complement.
        // [[2^30, 1], [2^30, 2]] reduced along axis 0 → [2^31, 3]; 2^31 overflows
        // int32 and wraps to i32::MIN = -2147483648. The kept column [1,2] sums to
        // 3 (in range, unchanged).
        let big = 1_i64 << 30; // 1073741824, a valid int32
        let m = Value::Tensor(
            TensorValue::new(
                DType::I32,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::I64(big),
                    Literal::I64(1),
                    Literal::I64(big),
                    Literal::I64(2),
                ],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "0".to_owned());
        let result = eval_reduce_axes(
            Primitive::ReduceSum,
            &[m],
            &params,
            0,
            0.0,
            i64::wrapping_add,
            |a, b| a + b,
        )
        .unwrap();

        let Value::Tensor(t) = &result else {
            panic!("expected tensor result, got {result:?}");
        };
        assert_eq!(t.dtype, DType::I32, "int32 reduction must stay int32");
        let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
        assert_eq!(
            vals,
            vec![i64::from(i32::MIN), 3],
            "2^31 must wrap to i32::MIN; in-range column unchanged"
        );
    }

    #[test]
    fn u32_reductions_keep_dtype_and_wrap_unsigned() {
        // BUG FIX (bead 2iotb): u32 reductions used to fall to the float arm (f64
        // accumulation, F64 output). JAX keeps uint32 and wraps sum/prod mod 2^32,
        // with UNSIGNED max/min. Verified via the real dispatch (eval_primitive).
        let mk = |dims: Vec<u32>, data: &[u32]| {
            Value::Tensor(
                TensorValue::new(
                    DType::U32,
                    Shape { dims },
                    data.iter().map(|&v| Literal::U32(v)).collect(),
                )
                .unwrap(),
            )
        };
        let two31 = 1u32 << 31; // 2147483648 > i32::MAX
        let p = BTreeMap::new();

        // Full sum: 2^31 + 2^31 + 1 = 2^32 + 1 ≡ 1 (mod 2^32).
        let r = crate::eval_primitive(Primitive::ReduceSum, &[mk(vec![3], &[two31, two31, 1])], &p)
            .unwrap();
        assert_eq!(
            r,
            Value::Scalar(Literal::U32(1)),
            "u32 sum must wrap mod 2^32"
        );

        // Full prod: 65536 * 65536 = 2^32 ≡ 0 (mod 2^32).
        let r = crate::eval_primitive(Primitive::ReduceProd, &[mk(vec![2], &[65536, 65536])], &p)
            .unwrap();
        assert_eq!(
            r,
            Value::Scalar(Literal::U32(0)),
            "u32 prod must wrap mod 2^32"
        );

        // Full max/min must be UNSIGNED (3e9 > i32::MAX must win the max, not look negative).
        let big = 3_000_000_000u32;
        let r =
            crate::eval_primitive(Primitive::ReduceMax, &[mk(vec![3], &[5, big, 7])], &p).unwrap();
        assert_eq!(
            r,
            Value::Scalar(Literal::U32(big)),
            "u32 max must be unsigned"
        );
        let r =
            crate::eval_primitive(Primitive::ReduceMin, &[mk(vec![3], &[5, big, 7])], &p).unwrap();
        assert_eq!(
            r,
            Value::Scalar(Literal::U32(5)),
            "u32 min must be unsigned"
        );

        // Partial axis-0 sum of [[2^31,1],[2^31,2]] → [2^32 mod 2^32 = 0, 3], stays U32.
        let mut pa = BTreeMap::new();
        pa.insert("axes".to_owned(), "0".to_owned());
        let r = crate::eval_primitive(
            Primitive::ReduceSum,
            &[mk(vec![2, 2], &[two31, 1, two31, 2])],
            &pa,
        )
        .unwrap();
        let Value::Tensor(t) = &r else {
            panic!("expected tensor, got {r:?}")
        };
        assert_eq!(t.dtype, DType::U32, "partial u32 sum stays u32");
        let vals: Vec<u32> = t
            .elements
            .iter()
            .map(|l| match l {
                Literal::U32(v) => *v,
                o => panic!("expected U32, got {o:?}"),
            })
            .collect();
        assert_eq!(vals, vec![0, 3], "partial u32 sum wraps mod 2^32 per-cell");
    }

    #[test]
    fn u64_reductions_keep_dtype_and_wrap_unsigned() {
        // u64 half of bead 2iotb: sum/prod wrap mod 2^64; max/min UNSIGNED (values
        // above i64::MAX must compare as the larger). Reuses the i64 machinery via
        // bit-reinterpret (sum/prod) and the order-preserving sign-flip (max/min).
        let mk = |dims: Vec<u32>, data: &[u64]| {
            Value::Tensor(
                TensorValue::new(
                    DType::U64,
                    Shape { dims },
                    data.iter().map(|&v| Literal::U64(v)).collect(),
                )
                .unwrap(),
            )
        };
        let two63 = 1u64 << 63; // 9223372036854775808 > i64::MAX
        let p = BTreeMap::new();

        // Full sum: 2^63 + 2^63 + 1 = 2^64 + 1 ≡ 1 (mod 2^64).
        let r = crate::eval_primitive(Primitive::ReduceSum, &[mk(vec![3], &[two63, two63, 1])], &p)
            .unwrap();
        assert_eq!(r, Value::Scalar(Literal::U64(1)), "u64 sum wraps mod 2^64");

        // Full prod: 2^32 * 2^32 = 2^64 ≡ 0 (mod 2^64).
        let r = crate::eval_primitive(
            Primitive::ReduceProd,
            &[mk(vec![2], &[1u64 << 32, 1u64 << 32])],
            &p,
        )
        .unwrap();
        assert_eq!(r, Value::Scalar(Literal::U64(0)), "u64 prod wraps mod 2^64");

        // Unsigned max/min: 2^63+100 and u64::MAX exceed i64::MAX, must compare largest.
        let big = two63 + 100;
        let r = crate::eval_primitive(
            Primitive::ReduceMax,
            &[mk(vec![4], &[5, big, u64::MAX, 7])],
            &p,
        )
        .unwrap();
        assert_eq!(r, Value::Scalar(Literal::U64(u64::MAX)), "u64 max unsigned");
        let r = crate::eval_primitive(
            Primitive::ReduceMin,
            &[mk(vec![4], &[big, 5, u64::MAX, 7])],
            &p,
        )
        .unwrap();
        assert_eq!(r, Value::Scalar(Literal::U64(5)), "u64 min unsigned");

        // Partial axis-0 max of [[5, u64::MAX],[big, 7]] → [big, u64::MAX], stays U64.
        let mut pa = BTreeMap::new();
        pa.insert("axes".to_owned(), "0".to_owned());
        let r = crate::eval_primitive(
            Primitive::ReduceMax,
            &[mk(vec![2, 2], &[5, u64::MAX, big, 7])],
            &pa,
        )
        .unwrap();
        let Value::Tensor(t) = &r else {
            panic!("expected tensor, got {r:?}")
        };
        assert_eq!(t.dtype, DType::U64, "partial u64 max stays u64");
        let vals: Vec<u64> = t
            .elements
            .iter()
            .map(|l| match l {
                Literal::U64(v) => *v,
                o => panic!("expected U64, got {o:?}"),
            })
            .collect();
        assert_eq!(
            vals,
            vec![big, u64::MAX],
            "partial u64 max unsigned per-cell"
        );
    }

    #[test]
    fn u32_u64_cumulative_keep_dtype_and_wrap_unsigned() {
        // Cumulative scans had the same float-arm bug as reductions (bead 2iotb).
        // cumsum/cumprod wrap mod 2^N, cummax/cummin are UNSIGNED, dtype preserved.
        let p = BTreeMap::new();
        let u32t = |data: &[u32]| {
            Value::Tensor(
                TensorValue::new(
                    DType::U32,
                    Shape::vector(data.len() as u32),
                    data.iter().map(|&v| Literal::U32(v)).collect(),
                )
                .unwrap(),
            )
        };
        let u64t = |data: &[u64]| {
            Value::Tensor(
                TensorValue::new(
                    DType::U64,
                    Shape::vector(data.len() as u32),
                    data.iter().map(|&v| Literal::U64(v)).collect(),
                )
                .unwrap(),
            )
        };
        let getu32 = |v: &Value| -> Vec<u32> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::U32(x) => *x,
                    o => panic!("expected U32, got {o:?}"),
                })
                .collect()
        };
        let getu64 = |v: &Value| -> Vec<u64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::U64(x) => *x,
                    o => panic!("expected U64, got {o:?}"),
                })
                .collect()
        };

        // u32 cumsum [2^31, 2^31, 1] → [2^31, 0 (2^32 wraps), 1].
        let two31 = 1u32 << 31;
        let r = crate::eval_primitive(Primitive::Cumsum, &[u32t(&[two31, two31, 1])], &p).unwrap();
        assert_eq!(r.as_tensor().unwrap().dtype, DType::U32);
        assert_eq!(getu32(&r), vec![two31, 0, 1], "u32 cumsum wraps mod 2^32");

        // u32 cummax [5, 3e9, 7] → [5, 3e9, 3e9] (unsigned; 3e9 > i32::MAX).
        let big = 3_000_000_000u32;
        let r = crate::eval_primitive(Primitive::Cummax, &[u32t(&[5, big, 7])], &p).unwrap();
        assert_eq!(getu32(&r), vec![5, big, big], "u32 cummax unsigned");

        // u64 cumsum [2^63, 2^63] → [2^63, 0 (2^64 wraps)].
        let two63 = 1u64 << 63;
        let r = crate::eval_primitive(Primitive::Cumsum, &[u64t(&[two63, two63])], &p).unwrap();
        assert_eq!(r.as_tensor().unwrap().dtype, DType::U64);
        assert_eq!(getu64(&r), vec![two63, 0], "u64 cumsum wraps mod 2^64");

        // u64 cummax [5, 2^63+100, u64::MAX, 7] → running unsigned max.
        let r = crate::eval_primitive(
            Primitive::Cummax,
            &[u64t(&[5, two63 + 100, u64::MAX, 7])],
            &p,
        )
        .unwrap();
        assert_eq!(
            getu64(&r),
            vec![5, two63 + 100, u64::MAX, u64::MAX],
            "u64 cummax unsigned"
        );
    }

    #[test]
    fn full_reduce_i32_sum_wraps_to_scalar() {
        // b6w3l gap (2), full-reduce-to-scalar: jnp.sum over a whole int32 array
        // wraps two's-complement (int32 stays int32 in JAX/XLA). [2^30, 2^30,
        // 2^30, 2^30] sums to 2^32 == 0 mod 2^32 → scalar 0. (No axes param ⇒ full
        // reduction; the result is a dtype-less Value::Scalar, but the VALUE must
        // already be the wrapped int32.)
        let big = 1_i64 << 30; // valid int32
        let t = Value::Tensor(
            TensorValue::new(
                DType::I32,
                Shape { dims: vec![4] },
                vec![
                    Literal::I64(big),
                    Literal::I64(big),
                    Literal::I64(big),
                    Literal::I64(big),
                ],
            )
            .unwrap(),
        );
        let result = eval_reduce(
            Primitive::ReduceSum,
            &[t],
            0,
            0.0,
            i64::wrapping_add,
            |a, b| a + b,
        )
        .unwrap();
        let Value::Scalar(lit) = &result else {
            panic!("expected scalar result, got {result:?}");
        };
        assert_eq!(
            lit.as_i64().unwrap(),
            0,
            "2^32 must wrap to 0 at int32 width"
        );
    }

    #[test]
    fn full_reduce_i64_sum_does_not_wrap() {
        // Regression guard: int64 full reductions keep full width (no int32 wrap).
        let big = 1_i64 << 30;
        let t = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![4] },
                vec![
                    Literal::I64(big),
                    Literal::I64(big),
                    Literal::I64(big),
                    Literal::I64(big),
                ],
            )
            .unwrap(),
        );
        let result = eval_reduce(
            Primitive::ReduceSum,
            &[t],
            0,
            0.0,
            i64::wrapping_add,
            |a, b| a + b,
        )
        .unwrap();
        let Value::Scalar(lit) = &result else {
            panic!("expected scalar result, got {result:?}");
        };
        assert_eq!(
            lit.as_i64().unwrap(),
            4 * big,
            "int64 sum keeps full width (2^32)"
        );
    }

    #[test]
    fn reduce_axes_i64_still_widens_unchanged() {
        // Regression guard: int64 reductions keep int64 dtype and full-width sums.
        let big = 1_i64 << 40; // 1099511627776, far past int32 range
        let m = Value::Tensor(
            TensorValue::new(
                DType::I64,
                Shape { dims: vec![2, 2] },
                vec![
                    Literal::I64(big),
                    Literal::I64(1),
                    Literal::I64(big),
                    Literal::I64(2),
                ],
            )
            .unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "0".to_owned());
        let result = eval_reduce_axes(
            Primitive::ReduceSum,
            &[m],
            &params,
            0,
            0.0,
            i64::wrapping_add,
            |a, b| a + b,
        )
        .unwrap();

        let Value::Tensor(t) = &result else {
            panic!("expected tensor result, got {result:?}");
        };
        assert_eq!(t.dtype, DType::I64, "int64 reduction stays int64");
        let vals: Vec<i64> = t.elements.iter().map(|l| l.as_i64().unwrap()).collect();
        assert_eq!(vals, vec![2 * big, 3], "int64 does not wrap at int32 width");
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

    /// The dense-i64 bitwise axis-reduction fast path (OutIndexOdometer over the
    /// contiguous `as_i64_slice`) must be bit-for-bit identical to the boxed
    /// Literal path (forced via `new_with_literal_buffer`, `as_i64_slice` None),
    /// across ReduceAnd/ReduceOr/ReduceXor and every axis subset of a rank-3
    /// tensor — including negative values / high bits / duplicates.
    #[test]
    fn dense_i64_bitwise_axis_reduce_bit_identical_to_literal_path() {
        let dims = vec![4_u32, 5, 6];
        let n = 4 * 5 * 6;
        let data: Vec<i64> = (0..n as i64)
            .map(|i| (i.wrapping_mul(2_654_435_761) ^ (i << 17)).wrapping_sub(123))
            .collect();
        let dense = Value::Tensor(
            TensorValue::new_i64_values(Shape { dims: dims.clone() }, data.clone()).unwrap(),
        );
        assert!(
            dense.as_tensor().unwrap().elements.as_i64_slice().is_some(),
            "dense i64 operand"
        );
        let boxed = Value::Tensor(
            TensorValue::new_with_literal_buffer(
                DType::I64,
                Shape { dims: dims.clone() },
                LiteralBuffer::new(data.iter().copied().map(Literal::I64).collect()),
            )
            .unwrap(),
        );
        assert!(
            boxed.as_tensor().unwrap().elements.as_i64_slice().is_none(),
            "boxed i64 operand must not be dense"
        );

        let i64s = |v: &Value| -> Vec<i64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_i64().unwrap())
                .collect()
        };
        for prim in [
            Primitive::ReduceAnd,
            Primitive::ReduceOr,
            Primitive::ReduceXor,
        ] {
            // Partial-axis subsets only (the all-axes reduce returns a Scalar via a
            // separate branch this fast path doesn't touch).
            for axes in ["0", "1", "2", "0,1", "1,2", "0,2"] {
                let mut params = BTreeMap::new();
                params.insert("axes".to_owned(), axes.to_owned());
                let d = crate::eval_primitive(prim, std::slice::from_ref(&dense), &params).unwrap();
                let l = crate::eval_primitive(prim, std::slice::from_ref(&boxed), &params).unwrap();
                assert!(
                    d.as_tensor().unwrap().elements.as_i64_slice().is_some(),
                    "{prim:?} axes={axes}: output must be dense i64 storage, not boxed"
                );
                assert_eq!(
                    d.as_tensor().unwrap().shape.dims,
                    l.as_tensor().unwrap().shape.dims,
                    "{prim:?} axes={axes} shape"
                );
                assert_eq!(i64s(&d), i64s(&l), "{prim:?} axes={axes} values");
            }
        }
    }

    #[test]
    fn i32_bitwise_axis_reduce_works_and_matches_i64_path() {
        // i32 (JAX's default int) bitwise axis-reduce previously hit the `_` arm
        // and erred ("expects bool or i64 tensor") despite the code comment
        // claiming i32 took the i64 branch. It must now (a) succeed, (b) tag its
        // output I32 (dense), and (c) produce the same values as the i64 path on
        // the same data — bitwise and/or/xor preserve the sign-extension invariant,
        // so an i64-backed i32 reduces identically. Include negatives to exercise
        // sign extension.
        let dims = vec![3_u32, 4];
        let data_i32: Vec<i32> = vec![
            -1,
            0x0f0f,
            5,
            -8,
            0x7fff_ffff,
            -16,
            12,
            0,
            i32::MIN,
            255,
            -3,
            9,
        ];
        let i32_tensor = Value::Tensor(
            TensorValue::new_i32_values(
                Shape { dims: dims.clone() },
                data_i32.iter().map(|&v| i64::from(v)).collect(),
            )
            .unwrap(),
        );
        let i64_tensor = Value::Tensor(
            TensorValue::new_i64_values(
                Shape { dims: dims.clone() },
                data_i32.iter().map(|&v| i64::from(v)).collect(),
            )
            .unwrap(),
        );
        let vals = |v: &Value| -> Vec<i64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| l.as_i64().unwrap())
                .collect()
        };
        for prim in [
            Primitive::ReduceAnd,
            Primitive::ReduceOr,
            Primitive::ReduceXor,
        ] {
            for axes in ["0", "1"] {
                let mut params = BTreeMap::new();
                params.insert("axes".to_owned(), axes.to_owned());
                let out_i32 =
                    crate::eval_primitive(prim, std::slice::from_ref(&i32_tensor), &params)
                        .unwrap_or_else(|e| {
                            panic!("{prim:?} axes={axes} i32 must not error: {e:?}")
                        });
                let out_i64 =
                    crate::eval_primitive(prim, std::slice::from_ref(&i64_tensor), &params)
                        .unwrap();
                let t = out_i32.as_tensor().unwrap();
                assert_eq!(
                    t.dtype,
                    DType::I32,
                    "{prim:?} axes={axes}: output dtype I32"
                );
                assert!(
                    t.elements.as_i64_slice().is_some(),
                    "{prim:?} axes={axes}: i32 output must be dense"
                );
                // Values equal the i64 path (sign-extension preserved across &/|/^).
                assert_eq!(
                    vals(&out_i32),
                    vals(&out_i64),
                    "{prim:?} axes={axes}: i32 values match i64 path"
                );
            }
        }
        // Full-reduce (all axes) also succeeds, returning an i64-backed scalar
        // (the established integer full-reduce convention).
        let mut all = BTreeMap::new();
        all.insert("axes".to_owned(), "0,1".to_owned());
        let s = crate::eval_primitive(Primitive::ReduceOr, std::slice::from_ref(&i32_tensor), &all)
            .expect("i32 full bitwise-reduce must not error");
        assert!(
            matches!(s, Value::Scalar(Literal::I64(_))),
            "scalar i64 result"
        );
    }

    #[test]
    fn dense_bool_bitwise_axis_reduce_is_dense_and_bit_identical() {
        // The contiguous-block bool fast path (any/all over a contiguous axis
        // block, the dominant `any/all(mask, axis=-1|0)` idiom) must emit dense
        // 1-byte bool storage and match the boxed Vec<Literal::Bool> path bit for
        // bit. Regression for the de-box: that path used to `map(Literal::Bool)`
        // into a 24 B/elem boxed buffer despite already holding a dense Vec<bool>.
        let dims = vec![4_u32, 5, 6];
        let n = (4 * 5 * 6) as usize;
        let data: Vec<bool> = (0..n)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 3) & 1 == 0)
            .collect();
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
                || dense
                    .as_tensor()
                    .unwrap()
                    .elements
                    .as_bool_words()
                    .is_some(),
            "dense bool operand"
        );
        let boxed = Value::Tensor(
            TensorValue::new_with_literal_buffer(
                DType::Bool,
                Shape { dims: dims.clone() },
                LiteralBuffer::new(data.iter().copied().map(Literal::Bool).collect()),
            )
            .unwrap(),
        );
        assert!(
            boxed
                .as_tensor()
                .unwrap()
                .elements
                .as_bool_slice()
                .is_none(),
            "boxed bool operand must not be dense"
        );
        let bools = |v: &Value| -> Vec<bool> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| matches!(l, Literal::Bool(true)))
                .collect()
        };
        for prim in [
            Primitive::ReduceAnd,
            Primitive::ReduceOr,
            Primitive::ReduceXor,
        ] {
            // "2" -> inner==1, "0" -> outer==1, "1"/"0,1"/"1,2" contiguous blocks
            // (fast path); "0,2" non-contiguous (odometer fallback). All output
            // dtype Bool, so the dense ctor applies on every path.
            for axes in ["0", "1", "2", "0,1", "1,2", "0,2"] {
                let mut params = BTreeMap::new();
                params.insert("axes".to_owned(), axes.to_owned());
                let d = crate::eval_primitive(prim, std::slice::from_ref(&dense), &params).unwrap();
                let l = crate::eval_primitive(prim, std::slice::from_ref(&boxed), &params).unwrap();
                let dt = d.as_tensor().unwrap();
                assert!(
                    dt.elements.as_bool_slice().is_some() || dt.elements.as_bool_words().is_some(),
                    "{prim:?} axes={axes}: output must be dense bool storage, not boxed"
                );
                assert_eq!(
                    dt.shape.dims,
                    l.as_tensor().unwrap().shape.dims,
                    "{prim:?} axes={axes} shape"
                );
                assert_eq!(bools(&d), bools(&l), "{prim:?} axes={axes} values");
            }
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_i64_bitwise_axis_reduce_dense_vs_boxed() {
        use std::time::Instant;
        // True before/after for this commit: OLD algorithm (per-element
        // flat_to_multi divmod decode + multi_to_out_flat + Vec<Literal>
        // materialization, reconstructed inline) vs NEW (eval_primitive ->
        // OutIndexOdometer over as_i64_slice). Rank-3 reduce over the middle axis.
        let (d0, d1, d2) = (256usize, 256usize, 256usize);
        let n = d0 * d1 * d2;
        let data: Vec<i64> = (0..n as i64)
            .map(|i| i.wrapping_mul(6_364_136_223_846_793_005) ^ (i << 13))
            .collect();
        let dims = vec![d0 as u32, d1 as u32, d2 as u32];
        let dense = Value::Tensor(
            TensorValue::new_i64_values(Shape { dims: dims.clone() }, data.clone()).unwrap(),
        );
        let mut params = BTreeMap::new();
        params.insert("axes".to_owned(), "1".to_owned());

        // Reduce over axis 1: input strides [d1*d2, d2, 1]; output [d0, d2].
        let strides = [d1 * d2, d2, 1usize];
        let out_count = d0 * d2;
        let timed = |mut f: Box<dyn FnMut() -> i64>| -> (f64, i64) {
            f();
            let mut b = f64::MAX;
            let mut digest = 0i64;
            for _ in 0..10 {
                let t = Instant::now();
                digest = std::hint::black_box(f());
                b = b.min(t.elapsed().as_secs_f64());
            }
            (b, digest)
        };

        // OLD: materialize Vec<Literal> (the dense Index path forced this) + per
        // element flat_to_multi (3 divmods) + multi_to_out_flat (kept axes 0,2).
        let lits: Vec<Literal> = data.iter().copied().map(Literal::I64).collect();
        let (t_old, d_old) = timed(Box::new(move || {
            let mut result = vec![0i64; out_count];
            let mut multi = [0usize; 3];
            for (flat_idx, lit) in lits.iter().enumerate() {
                let mut rem = flat_idx;
                for ax in 0..3 {
                    multi[ax] = rem / strides[ax];
                    rem %= strides[ax];
                }
                let out_idx = multi[0] * d2 + multi[2];
                let val = lit.as_i64().unwrap();
                result[out_idx] ^= val;
            }
            result.iter().fold(0i64, |a, &v| a ^ v)
        }));

        // NEW: the production path.
        let (t_new, d_new) = timed(Box::new(move || {
            let out =
                crate::eval_primitive(Primitive::ReduceXor, std::slice::from_ref(&dense), &params)
                    .unwrap();
            out.as_tensor()
                .unwrap()
                .elements
                .iter()
                .fold(0i64, |a, l| a ^ l.as_i64().unwrap())
        }));
        assert_eq!(d_old, d_new, "old vs new bitwise-reduce digest");
        println!(
            "BENCH i64 ReduceXor [{d0},{d1},{d2}] axis1: old(flat_to_multi+materialize)={:.4}ms new(odometer+slice)={:.4}ms speedup={:.2}x digest={d_old:016x}",
            t_old * 1e3,
            t_new * 1e3,
            t_old / t_new,
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
    fn dense_i32_cumulative_matches_generic_and_preserves_dtype() {
        // i32 (JAX's default int) cumsum/cumprod/cummax/cummin now use the dense i64 scan
        // (were on the generic per-Literal scan). i32 shares the i64 backing and
        // sign-extends; cumsum/cumprod prefixes wrap mod 2^32 via the eval_primitive
        // chokepoint (ring homomorphism), cummax/cummin select an in-range input. Dense
        // (densified) must match the boxed generic path AND stay I32.
        let n = 600usize;
        let data: Vec<i64> = (0..n)
            .map(|i| i64::from((((i as i64) * 715_827_883) - 1_000_000_000) as i32))
            .collect();
        let dense = Value::Tensor(
            TensorValue::new(
                DType::I32,
                Shape {
                    dims: vec![n as u32],
                },
                data.iter().map(|&v| Literal::I64(v)).collect(),
            )
            .unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new_with_literal_buffer(
                DType::I32,
                Shape {
                    dims: vec![n as u32],
                },
                fj_core::LiteralBuffer::new(data.iter().map(|&v| Literal::I64(v)).collect()),
            )
            .unwrap(),
        );
        assert!(dense.as_tensor().unwrap().elements.as_i64_slice().is_some());
        assert!(boxed.as_tensor().unwrap().elements.as_i64_slice().is_none());
        let geti = |v: &Value| -> Vec<i64> {
            v.as_tensor()
                .unwrap()
                .elements
                .iter()
                .map(|l| match l {
                    Literal::I64(x) => *x,
                    o => panic!("expected I64-backed i32, got {o:?}"),
                })
                .collect()
        };
        for prim in [
            Primitive::Cumsum,
            Primitive::Cumprod,
            Primitive::Cummax,
            Primitive::Cummin,
        ] {
            for rev in ["false", "true"] {
                let p = BTreeMap::from([
                    ("axis".to_owned(), "0".to_owned()),
                    ("reverse".to_owned(), rev.to_owned()),
                ]);
                let d = crate::eval_primitive(prim, std::slice::from_ref(&dense), &p).unwrap();
                let b = crate::eval_primitive(prim, std::slice::from_ref(&boxed), &p).unwrap();
                assert_eq!(
                    d.as_tensor().unwrap().dtype,
                    DType::I32,
                    "{prim:?} rev={rev}: dtype must stay I32"
                );
                assert_eq!(geti(&d), geti(&b), "{prim:?} rev={rev}: dense != generic");
            }
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_i32_cumsum_dense_vs_generic() {
        use std::time::Instant;
        let (rows, cols) = (512usize, 2048usize);
        let data: Vec<i64> = (0..rows * cols)
            .map(|i| i64::from(((i as i64).wrapping_mul(2_654_435_761) ^ 0x1234) as i32))
            .collect();
        let dense = Value::Tensor(
            TensorValue::new(
                DType::I32,
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                data.iter().map(|&v| Literal::I64(v)).collect(),
            )
            .unwrap(),
        );
        let boxed = Value::Tensor(
            TensorValue::new_with_literal_buffer(
                DType::I32,
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                fj_core::LiteralBuffer::new(data.iter().map(|&v| Literal::I64(v)).collect()),
            )
            .unwrap(),
        );
        let p = BTreeMap::from([("axis".to_owned(), "1".to_owned())]);
        let best = |v: &Value| {
            let _ = crate::eval_primitive(Primitive::Cumsum, std::slice::from_ref(v), &p).unwrap();
            let mut b = f64::MAX;
            for _ in 0..5 {
                let t = Instant::now();
                let o =
                    crate::eval_primitive(Primitive::Cumsum, std::slice::from_ref(v), &p).unwrap();
                std::hint::black_box(o.as_tensor().unwrap().elements.len());
                b = b.min(t.elapsed().as_secs_f64());
            }
            b
        };
        let generic = best(&boxed);
        let dense_t = best(&dense);
        println!(
            "BENCH i32 cumsum [{rows}x{cols}] axis1: generic={:.2}ms dense={:.2}ms speedup={:.2}x",
            generic * 1e3,
            dense_t * 1e3,
            generic / dense_t
        );
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

    #[test]
    fn blocked_dense_f64_single_line_cumulative_matches_serial_reference() {
        let len = CUMSUM_BLOCKED_MIN_ELEMS + 257;
        let dims = vec![len as u32];
        let data: Vec<f64> = (0..len)
            .map(|i| {
                let centered = (i % 29) as f64 - 14.0;
                let wobble = (i % 7) as f64 - 3.0;
                1.0 + centered * 1e-8 + wobble * 1e-12
            })
            .collect();
        let dense =
            Value::Tensor(TensorValue::new_f64_values(Shape { dims }, data.clone()).unwrap());
        let params = BTreeMap::from([("axis".to_owned(), "0".to_owned())]);

        for (prim, init, op, tolerance) in [
            (
                Primitive::Cumsum,
                0.0,
                (|a: f64, b: f64| a + b) as fn(f64, f64) -> f64,
                1e-12,
            ),
            (Primitive::Cumprod, 1.0, (|a, b| a * b), 4e-12),
            (
                Primitive::Cummax,
                f64::NEG_INFINITY,
                crate::jax_max_f64 as fn(f64, f64) -> f64,
                0.0,
            ),
            (
                Primitive::Cummin,
                f64::INFINITY,
                crate::jax_min_f64 as fn(f64, f64) -> f64,
                0.0,
            ),
        ] {
            let got = crate::eval_primitive(prim, std::slice::from_ref(&dense), &params).unwrap();
            let got = got
                .as_tensor()
                .unwrap()
                .elements
                .as_f64_slice()
                .expect("blocked f64 cumulative should keep dense storage");

            let mut acc = init;
            let mut max_abs = 0.0_f64;
            let mut max_allowed: f64 = tolerance;
            for (&actual, &input) in got.iter().zip(data.iter()) {
                acc = op(acc, input);
                max_abs = max_abs.max((actual - acc).abs());
                max_allowed = max_allowed.max(tolerance * acc.abs().max(actual.abs()).max(1.0));
            }
            assert!(
                max_abs <= max_allowed,
                "{prim:?} blocked single-line scan drift {max_abs:e} > {max_allowed:e}"
            );
        }
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
            crate::new_boxed(
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

    // A/B: bf16 trailing-axis cumsum — single-thread per-line scan vs the THREADED dense path
    // (eval_primitive). Isolates the threading win (both compute the same per-line scan).
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_half_cumsum_threaded_ab() {
        use std::time::Instant;
        let (rows, cols) = (512usize, 4096usize);
        let bits: Vec<u16> = (0..rows * cols)
            .map(|i| match Literal::from_bf16_f64(((i % 97) as f64) * 0.01) {
                Literal::BF16Bits(b) => b,
                _ => 0,
            })
            .collect();
        let t = Value::Tensor(
            TensorValue::new_half_float_values(
                DType::BF16,
                Shape {
                    dims: vec![rows as u32, cols as u32],
                },
                bits.clone(),
            )
            .unwrap(),
        );
        let p = BTreeMap::from([("axis".to_owned(), "1".to_owned())]);
        let threaded = || {
            std::hint::black_box(
                crate::eval_primitive(Primitive::Cumsum, std::slice::from_ref(&t), &p).unwrap(),
            );
        };
        // Single-thread reference: same per-line widen→f64-accumulate→round-to-bf16 scan.
        let serial = || {
            let mut out = vec![0u16; rows * cols];
            for r in 0..rows {
                let mut acc = 0.0f64;
                for c in 0..cols {
                    acc += Literal::BF16Bits(bits[r * cols + c])
                        .as_f64()
                        .unwrap_or(0.0);
                    out[r * cols + c] = match reduce_real_literal(DType::BF16, acc) {
                        Literal::BF16Bits(b) | Literal::F16Bits(b) => b,
                        _ => 0,
                    };
                }
            }
            std::hint::black_box(&out);
        };
        let bench = |f: &dyn Fn()| {
            f();
            let mut b = f64::MAX;
            for _ in 0..12 {
                let ti = Instant::now();
                f();
                b = b.min(ti.elapsed().as_secs_f64());
            }
            b * 1e3
        };
        let s = bench(&serial);
        let d = bench(&threaded);
        println!(
            "AB bf16 cumsum [512,4096] axis1: serial={s:.4}ms threaded={d:.4}ms ({:.2}x)",
            s / d
        );
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_half_cumsum_dense_vs_generic() {
        use std::time::Instant;
        // Dense half-float (BF16/F16) cumsum vs the generic per-`Literal` scan.
        // Dense HalfFloat storage (`as_half_float_slice`) takes the new dense path;
        // a boxed `Literal::BF16Bits`/`F16Bits` tensor falls to the generic loop.
        // Also a parity proof: the half bits must match element-for-element.
        let (rows, cols) = (2048usize, 2048usize); // cumsum over last axis (finite data, no NaN)
        let p = BTreeMap::from([("axis".to_owned(), "1".to_owned())]);
        for dt in [DType::BF16, DType::F16] {
            let bits = |v: f64| -> u16 {
                match reduce_real_literal(dt, v) {
                    Literal::BF16Bits(b) | Literal::F16Bits(b) => b,
                    _ => 0,
                }
            };
            let lit = |b: u16| -> Literal {
                match dt {
                    DType::BF16 => Literal::BF16Bits(b),
                    _ => Literal::F16Bits(b),
                }
            };
            let raw: Vec<u16> = (0..rows * cols)
                .map(|i| bits(((i % 251) as f64) * 0.013 - 1.6))
                .collect();
            let dims = vec![rows as u32, cols as u32];
            let dense = Value::Tensor(
                TensorValue::new_half_float_values(dt, Shape { dims: dims.clone() }, raw.clone())
                    .unwrap(),
            );
            let boxed = Value::Tensor(
                TensorValue::new(
                    dt,
                    Shape { dims: dims.clone() },
                    raw.iter().copied().map(lit).collect(),
                )
                .unwrap(),
            );
            let half_bits = |v: &Value| -> Vec<u16> {
                v.as_tensor()
                    .unwrap()
                    .elements
                    .iter()
                    .map(|l| match l {
                        Literal::BF16Bits(b) | Literal::F16Bits(b) => *b,
                        _ => 0,
                    })
                    .collect()
            };
            let d_out =
                crate::eval_primitive(Primitive::Cumsum, std::slice::from_ref(&dense), &p).unwrap();
            let g_out =
                crate::eval_primitive(Primitive::Cumsum, std::slice::from_ref(&boxed), &p).unwrap();
            assert_eq!(
                half_bits(&d_out),
                half_bits(&g_out),
                "{dt:?} dense cumsum half bits must match generic per-Literal scan"
            );

            let time = |input: &Value| {
                let _ = crate::eval_primitive(Primitive::Cumsum, std::slice::from_ref(input), &p)
                    .unwrap();
                let mut best = f64::MAX;
                for _ in 0..20 {
                    let t = Instant::now();
                    let _ =
                        crate::eval_primitive(Primitive::Cumsum, std::slice::from_ref(input), &p)
                            .unwrap();
                    best = best.min(t.elapsed().as_secs_f64());
                }
                best
            };
            let generic = time(&boxed);
            let dense_t = time(&dense);
            println!(
                "BENCH {dt:?} cumsum axis1 [{rows},{cols}]: generic(per-Literal)={:.4}ms dense={:.4}ms speedup={:.2}x",
                generic * 1e3,
                dense_t * 1e3,
                generic / dense_t
            );
        }
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
    fn leading_axis_cumulative_matches_serial_reference() {
        // Exercises the streaming leading-axis (axis 0) path (scan_leading_axis_to_vec)
        // for F64 and F32 against an independent per-column serial reference, all four
        // ops + reverse, with special values (-0.0/NaN/±inf) to pin tie/NaN behavior.
        let rows = 512usize;
        let cols = 384usize;
        let raw: Vec<f64> = (0..rows * cols)
            .map(|i| match i % 263 {
                0 => -0.0,
                1 => f64::NAN,
                2 => f64::INFINITY,
                3 => f64::NEG_INFINITY,
                _ => ((i % 13) as f64) - 6.0 + 0.5 * ((i % 4) as f64),
            })
            .collect();
        let shape = Shape {
            dims: vec![rows as u32, cols as u32],
        };
        let dense_f64 =
            Value::Tensor(TensorValue::new_f64_values(shape.clone(), raw.clone()).unwrap());
        let f32_raw: Vec<f32> = raw.iter().map(|&v| v as f32).collect();
        let dense_f32 = Value::Tensor(TensorValue::new_f32_values(shape, f32_raw.clone()).unwrap());

        let ops: [(Primitive, fn(f64, f64) -> f64, f64); 4] = [
            (Primitive::Cumsum, |a, b| a + b, 0.0),
            (Primitive::Cumprod, |a, b| a * b, 1.0),
            (Primitive::Cummax, crate::jax_max_f64, f64::NEG_INFINITY),
            (Primitive::Cummin, crate::jax_min_f64, f64::INFINITY),
        ];
        for (prim, op, init) in ops {
            for &rev in &[false, true] {
                let mut p = BTreeMap::new();
                p.insert("axis".to_owned(), "0".to_owned());
                p.insert("reverse".to_owned(), rev.to_string());

                // F64
                let got64: Vec<u64> =
                    crate::eval_primitive(prim, std::slice::from_ref(&dense_f64), &p)
                        .unwrap()
                        .as_tensor()
                        .unwrap()
                        .elements
                        .iter()
                        .map(|l| l.as_f64().unwrap().to_bits())
                        .collect();
                // F32 — read the RAW stored f32 bits (not via as_f64, which would
                // re-canonicalize NaN payloads through an f32->f64->f32 round-trip).
                let res32 =
                    crate::eval_primitive(prim, std::slice::from_ref(&dense_f32), &p).unwrap();
                let res32_t = res32.as_tensor().unwrap();
                let got32: Vec<u32> = res32_t
                    .elements
                    .as_f32_slice()
                    .expect("dense f32 cumulative output")
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();

                // Independent per-column serial reference (down the leading axis).
                let mut exp64 = vec![0u64; rows * cols];
                let mut exp32 = vec![0u32; rows * cols];
                for c in 0..cols {
                    let order: Vec<usize> = if rev {
                        (0..rows).rev().collect()
                    } else {
                        (0..rows).collect()
                    };
                    let mut acc = init;
                    let mut acc32 = init;
                    for &k in &order {
                        let idx = k * cols + c;
                        acc = op(acc, raw[idx]);
                        exp64[idx] = acc.to_bits();
                        acc32 = op(acc32, f64::from(f32_raw[idx]));
                        exp32[idx] = (acc32 as f32).to_bits();
                    }
                }
                // Canonicalize NaN before the bit comparison: NaN payload/sign is not
                // contractual (the production scan and this hand-rolled reference can
                // pick different NaN signs for the same value), so compare NaNs as
                // equal while keeping every finite/±inf result bit-exact.
                let canon64 = |b: u64| {
                    if f64::from_bits(b).is_nan() {
                        0x7ff8_0000_0000_0000
                    } else {
                        b
                    }
                };
                let canon32 = |b: u32| {
                    if f32::from_bits(b).is_nan() {
                        0x7fc0_0000
                    } else {
                        b
                    }
                };
                let got64: Vec<u64> = got64.iter().map(|&b| canon64(b)).collect();
                let exp64: Vec<u64> = exp64.iter().map(|&b| canon64(b)).collect();
                let got32: Vec<u32> = got32.iter().map(|&b| canon32(b)).collect();
                let exp32: Vec<u32> = exp32.iter().map(|&b| canon32(b)).collect();
                assert_eq!(got64, exp64, "{prim:?} rev={rev} f64 leading-axis");
                assert_eq!(got32, exp32, "{prim:?} rev={rev} f32 leading-axis");
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
