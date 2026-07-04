//! Neural network activation functions matching JAX's jax.nn module.
//!
//! These are standalone functions that operate on f64 slices, matching JAX semantics.

use std::f64::consts::PI;

const SOFTMAX_2D_PARALLEL_MIN: usize = 1 << 18;

/// Threaded elementwise `f64` map: applies `f` over `x` across work-scaled scoped threads,
/// BIT-IDENTICAL to the sequential `x.iter().map(f).collect()` (each element is independent,
/// so no value or order changes). Below the work threshold `work_scaled_threads` returns 1
/// and the sequential map is used. Used by the COMPUTE-bound (`exp`/`tanh`) activations, which
/// were single-threaded eager maps — a needless serial bottleneck on a many-core host (cf. the
/// tanh FMA-gate fix). Cheap BW-bound activations (relu/clamp) instead use
/// [`threaded_f64_map_bw`], which only threads past L3 to avoid the below-L3 regress trap.
#[inline]
fn threaded_f64_map(x: &[f64], f: impl Fn(f64) -> f64 + Sync) -> Vec<f64> {
    let n = x.len();
    let threads = crate::arithmetic::work_scaled_threads(n);
    if threads <= 1 {
        return x.iter().map(|&v| f(v)).collect();
    }
    let mut out = vec![0.0f64; n];
    let chunk = n.div_ceil(threads);
    let f = &f;
    std::thread::scope(|scope| {
        let mut rest: &mut [f64] = out.as_mut_slice();
        let mut start = 0usize;
        while start < n {
            let len = chunk.min(n - start);
            let (blk, tail) = rest.split_at_mut(len);
            rest = tail;
            let src = &x[start..start + len];
            scope.spawn(move || {
                for (o, &v) in blk.iter_mut().zip(src) {
                    *o = f(v);
                }
            });
            start += len;
        }
    });
    out
}

/// Like [`threaded_f64_map`] but for CHEAP, bandwidth-bound elementwise ops (relu/clamp):
/// only threads past L3 (`>= 1<<23` elems), where aggregate DRAM bandwidth across cores
/// wins ~1.7-2x; below that a single core already saturates L3 and threading a memory-bound
/// op REGRESSES (the documented L3-scoped DO-NOT). Bit-identical to the sequential map.
#[inline]
fn threaded_f64_map_bw(x: &[f64], f: impl Fn(f64) -> f64 + Sync) -> Vec<f64> {
    if x.len() < (1 << 23) {
        return x.iter().map(|&v| f(v)).collect();
    }
    threaded_f64_map(x, f)
}

/// ReLU: max(x, 0)
///
/// Matches `jax.nn.relu(x)`.
#[must_use]
pub fn relu(x: &[f64]) -> Vec<f64> {
    threaded_f64_map_bw(x, |v| v.max(0.0))
}

/// Leaky ReLU: x if x >= 0 else negative_slope * x
///
/// Matches `jax.nn.leaky_relu(x, negative_slope)`.
#[must_use]
pub fn leaky_relu(x: &[f64], negative_slope: f64) -> Vec<f64> {
    threaded_f64_map_bw(x, |v| if v >= 0.0 { v } else { negative_slope * v })
}

/// ReLU6: min(max(x, 0), 6)
///
/// Matches `jax.nn.relu6(x)`.
#[must_use]
pub fn relu6(x: &[f64]) -> Vec<f64> {
    threaded_f64_map_bw(x, |v| v.clamp(0.0, 6.0))
}

/// Sigmoid: 1 / (1 + exp(-x))
///
/// Matches `jax.nn.sigmoid(x)` and `jax.lax.logistic(x)`.
#[must_use]
pub fn sigmoid(x: &[f64]) -> Vec<f64> {
    threaded_f64_map(x, |v| 1.0 / (1.0 + (-v).exp()))
}

/// Hard sigmoid: clip((x + 3) / 6, 0, 1)
///
/// Matches `jax.nn.hard_sigmoid(x)`.
#[must_use]
pub fn hard_sigmoid(x: &[f64]) -> Vec<f64> {
    threaded_f64_map_bw(x, |v| ((v + 3.0) / 6.0).clamp(0.0, 1.0))
}

/// Hard tanh: clip(x, -1, 1)
///
/// Matches `jax.nn.hard_tanh(x)`.
#[must_use]
pub fn hard_tanh(x: &[f64]) -> Vec<f64> {
    threaded_f64_map_bw(x, |v| v.clamp(-1.0, 1.0))
}

/// SiLU / Swish: x * sigmoid(x)
///
/// Matches `jax.nn.silu(x)` and `jax.nn.swish(x)`.
#[must_use]
pub fn silu(x: &[f64]) -> Vec<f64> {
    threaded_f64_map(x, |v| v / (1.0 + (-v).exp()))
}

/// Alias for silu
#[must_use]
pub fn swish(x: &[f64]) -> Vec<f64> {
    silu(x)
}

/// Hard SiLU / Hard Swish: x * hard_sigmoid(x)
///
/// Matches `jax.nn.hard_silu(x)` and `jax.nn.hard_swish(x)`.
#[must_use]
pub fn hard_silu(x: &[f64]) -> Vec<f64> {
    threaded_f64_map_bw(x, |v| v * ((v + 3.0) / 6.0).clamp(0.0, 1.0))
}

/// Alias for hard_silu
#[must_use]
pub fn hard_swish(x: &[f64]) -> Vec<f64> {
    hard_silu(x)
}

/// GELU (Gaussian Error Linear Unit): x * Phi(x)
///
/// Uses the approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// Matches `jax.nn.gelu(x, approximate=True)`.
#[must_use]
pub fn gelu(x: &[f64]) -> Vec<f64> {
    let sqrt_2_over_pi = (2.0 / PI).sqrt();
    threaded_f64_map(x, |v| {
        let inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
        0.5 * v * (1.0 + inner.tanh())
    })
}

/// Exact GELU `0.5·x·(1 + erf(x/√2))` (`jax.nn.gelu(approximate=False)`), computed in the
/// operation grouping of the decomposed graph: `(x·0.5) · (1 + erf(x / √2))`. Calls the SAME
/// `arithmetic::erf_approx` that the `Erf` primitive dispatches to (its SIMD `erf_f64x8` is
/// bit-identical to `erf_approx`), so this is bit-for-bit identical to the decomposed
/// `Div → Erf → Add → Mul → Mul` jaxpr. Elementwise (order-independent) → threads flat. Used by
/// the interpreter GELU superinstruction.
#[must_use]
pub fn gelu_erf(x: &[f64]) -> Vec<f64> {
    let sqrt2 = 2.0_f64.sqrt();
    threaded_f64_map(x, move |v| {
        let e = crate::arithmetic::erf_approx(v / sqrt2);
        (v * 0.5) * (1.0 + e)
    })
}

/// ELU (Exponential Linear Unit): x if x > 0 else alpha * (exp(x) - 1)
///
/// Matches `jax.nn.elu(x, alpha)`, which uses `expm1` (not `exp(x) - 1`) so the
/// negative branch stays accurate near `x = 0` — `exp(x) - 1` loses ~`|x|`
/// relative precision there from the cancellation, while `expm1(x)` is exact.
#[must_use]
pub fn elu(x: &[f64], alpha: f64) -> Vec<f64> {
    threaded_f64_map(x, |v| if v > 0.0 { v } else { alpha * v.exp_m1() })
}

/// CELU (Continuously-differentiable ELU): max(x, 0) + min(0, alpha * (exp(x/alpha) - 1))
///
/// Matches `jax.nn.celu(x, alpha)`: `max(x, 0) + alpha * expm1(min(x, 0) / alpha)`
/// (JAX uses `expm1`, accurate near `x = 0`, where `exp(x/alpha) - 1` would lose
/// precision to cancellation).
#[must_use]
pub fn celu(x: &[f64], alpha: f64) -> Vec<f64> {
    threaded_f64_map(x, |v| v.max(0.0) + alpha * (v.min(0.0) / alpha).exp_m1())
}

/// SELU (Scaled ELU): scale * (max(x, 0) + alpha * min(0, exp(x) - 1))
///
/// Uses the self-normalizing constants:
/// - alpha ≈ 1.6732632423543772
/// - scale ≈ 1.0507009873554805
///
/// Matches `jax.nn.selu(x) = scale * elu(x, alpha)`. Uses `expm1` for the
/// negative branch (as JAX's `elu` does), accurate near `x = 0`.
#[must_use]
pub fn selu(x: &[f64]) -> Vec<f64> {
    const ALPHA: f64 = 1.6732632423543772;
    const SCALE: f64 = 1.0507009873554805;
    threaded_f64_map(x, |v| SCALE * if v > 0.0 { v } else { ALPHA * v.exp_m1() })
}

/// Numerically-stable scalar softplus matching `jax.nn.softplus(x) =
/// jnp.logaddexp(x, 0)`.
///
/// Uses the exact identity `softplus(x) = max(x, 0) + log1p(exp(-|x|))`, which
/// is finite for every `x` (the `exp` argument is ≤ 0, so no overflow) and
/// reproduces JAX's `logaddexp` to f64 precision with no discontinuity. The
/// previous implementation switched to the approximations `x` (for `x > 20`)
/// and `exp(x)` (for `x < -20`), dropping the `log1p(exp(-|x|))` correction
/// term — e.g. `softplus(21.0)` returned `21.0` instead of JAX's
/// `21.0 + 7.58e-10`. `NaN`/`±inf` propagate as JAX does.
#[inline]
#[must_use]
fn softplus_scalar(v: f64) -> f64 {
    v.max(0.0) + (-v.abs()).exp().ln_1p()
}

/// Softplus: log(1 + exp(x)) = logaddexp(x, 0)
///
/// Numerically stable for all inputs (no overflow). Matches `jax.nn.softplus(x)`.
#[must_use]
pub fn softplus(x: &[f64]) -> Vec<f64> {
    threaded_f64_map(x, softplus_scalar)
}

/// Softsign: x / (1 + |x|)
///
/// Matches `jax.nn.soft_sign(x)`.
#[must_use]
pub fn softsign(x: &[f64]) -> Vec<f64> {
    threaded_f64_map_bw(x, |v| v / (1.0 + v.abs()))
}

/// Mish: x * tanh(softplus(x))
///
/// Matches `jax.nn.mish(x)`.
#[must_use]
pub fn mish(x: &[f64]) -> Vec<f64> {
    threaded_f64_map(x, |v| v * softplus_scalar(v).tanh())
}

/// Log sigmoid: log(sigmoid(x)) = -softplus(-x)
///
/// Uses numerically stable formulation.
/// Matches `jax.nn.log_sigmoid(x)`.
#[must_use]
pub fn log_sigmoid(x: &[f64]) -> Vec<f64> {
    // log(sigmoid(x)) = -softplus(-x); reuse the exact stable softplus.
    threaded_f64_map(x, |v| -softplus_scalar(-v))
}

/// Log-sigmoid computed in the DIRECT grouping `-log(1 + exp(-x))` (the naive
/// `Neg → Exp → Add(1) → Log → Neg` graph), NOT the stable `-softplus(-x)` grouping of
/// [`log_sigmoid`] (which differs in the last bit). Uses only `exp`/`ln`, both bit-identical
/// to the `Exp`/`Log` primitives (as proven by the softmax/log-softmax superinstructions), so
/// this matches the decomposed graph bit-for-bit. Used by the interpreter log-sigmoid
/// superinstruction. Elementwise → threads flat.
#[must_use]
pub fn log_sigmoid_direct(x: &[f64]) -> Vec<f64> {
    threaded_f64_map(x, |v| -((1.0 + (-v).exp()).ln()))
}

/// Softplus computed in the DIRECT grouping `log(1 + exp(x))` (the naive `Exp → Add(1) → Log`
/// graph), NOT the stable `max(x,0) + log1p(exp(-|x|))` grouping of [`softplus`] (which differs
/// in the last bit). Uses only `exp`/`ln`, both bit-identical to the `Exp`/`Log` primitives (as
/// proven by the softmax/log-softmax superinstructions), so this matches the decomposed graph
/// bit-for-bit. Used by the interpreter softplus superinstruction. Elementwise → threads flat.
#[must_use]
pub fn softplus_direct(x: &[f64]) -> Vec<f64> {
    threaded_f64_map(x, |v| (1.0 + v.exp()).ln())
}

/// Compute log-sum-exp in a numerically stable way.
///
/// Matches `jax.scipy.special.logsumexp(x)`.
/// Uses the identity: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
#[must_use]
pub fn logsumexp(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val.is_infinite() {
        return max_val;
    }
    // Large 1D logsumexp (e.g. a language-model softmax over a big vocab) is dominated by
    // the per-element `exp`. Thread it into a buffer, then sum SERIALLY in index order —
    // BIT-IDENTICAL to the fused `.map(exp).sum()` (same values, same left-fold order). For
    // small inputs keep the fused single pass (no buffer / no thread-spawn overhead). (Refines
    // a convergent unconditional-threading commit that paid a buffer+extra-pass cost on small
    // inputs — the common axis-reduction case.)
    const LOGSUMEXP_THREAD_MIN: usize = 1 << 21;
    let sum_exp: f64 = if x.len() >= LOGSUMEXP_THREAD_MIN {
        let exp_buf = threaded_f64_map(x, |v| (v - max_val).exp());
        exp_buf.iter().sum()
    } else {
        x.iter().map(|&v| (v - max_val).exp()).sum()
    };
    max_val + sum_exp.ln()
}

/// Compute log-sum-exp along the last axis of a 2D array.
///
/// Returns a vector of logsumexp values, one per row. ROW-PARALLEL: rows are independent, and
/// each row still goes through the same [`logsumexp`] (fold-max, index-order exp-sum), so this is
/// BIT-IDENTICAL to the serial per-row map — only the outer row loop is distributed across
/// threads (above the `softmax_2d_thread_count` work gate). Empty rows (`cols == 0`) keep their
/// `logsumexp(&[]) == -inf` semantics on the serial path. Used by the interpreter logsumexp
/// superinstruction; serial fused measured only ~1.9x, row-parallel lifts it well past 2x.
#[must_use]
pub fn logsumexp_2d(x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let _len = checked_2d_row_major_len("2D logsumexp", x, rows, cols);
    let mut result = vec![0.0; rows];
    // `cols == 0` → empty rows are trivial (-inf); never worth threading. Otherwise gate on total
    // work exactly like the softmax/cross-entropy row-parallel kernels.
    let threads = if cols == 0 {
        1
    } else {
        softmax_2d_thread_count(rows, rows * cols)
    };
    if threads <= 1 {
        for (i, slot) in result.iter_mut().enumerate() {
            let start = i * cols;
            *slot = logsumexp(&x[start..start + cols]);
        }
        return result;
    }

    let rows_per = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut out_rest: &mut [f64] = &mut result;
        let mut row0 = 0usize;
        while row0 < rows {
            let row_count = rows_per.min(rows - row0);
            let (out_block, out_tail) = out_rest.split_at_mut(row_count);
            out_rest = out_tail;
            let x_block = &x[row0 * cols..(row0 + row_count) * cols];
            row0 += row_count;
            scope.spawn(move || {
                for (k, out_slot) in out_block.iter_mut().enumerate() {
                    let start = k * cols;
                    *out_slot = logsumexp(&x_block[start..start + cols]);
                }
            });
        }
    });
    result
}

/// Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
///
/// Matches `jax.nn.softmax(x)` for a 1D array.
/// Uses numerically stable computation via log-sum-exp.
#[must_use]
pub fn softmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return Vec::new();
    }
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // Thread the compute-bound `exp` map (≈20 flop/elem dominates the cheap O(1) max/sum
    // reductions). `exp_shifted` is built in index order, so the serial `sum` below reads it
    // in the same order — BIT-IDENTICAL. The cheap divide stays sequential (BW-bound).
    let exp_shifted = threaded_f64_map(x, |v| (v - max_val).exp());
    let sum_exp: f64 = exp_shifted.iter().sum();
    exp_shifted.iter().map(|&e| e / sum_exp).collect()
}

#[inline]
fn softmax_row_into(src: &[f64], dst: &mut [f64]) {
    let max_val = src.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum_exp = 0.0;
    for (d, &v) in dst.iter_mut().zip(src.iter()) {
        let e = (v - max_val).exp();
        *d = e;
        sum_exp += e;
    }
    for d in dst.iter_mut() {
        *d /= sum_exp;
    }
}

#[inline]
fn softmax_2d_thread_count(rows: usize, total: usize) -> usize {
    if rows <= 1 || total < SOFTMAX_2D_PARALLEL_MIN {
        1
    } else {
        crate::arithmetic::work_scaled_threads(total).min(rows)
    }
}

#[inline]
fn checked_2d_row_major_len(context: &str, x: &[f64], rows: usize, cols: usize) -> usize {
    let Some(len) = rows.checked_mul(cols) else {
        panic!("{context} output shape overflow");
    };
    assert!(
        x.len() >= len,
        "{context} input length {} is shorter than declared shape {}x{}={}",
        x.len(),
        rows,
        cols,
        len
    );
    len
}

fn fill_softmax_rows_parallel(
    x: &[f64],
    result: &mut [f64],
    rows: usize,
    cols: usize,
    row_fn: fn(&[f64], &mut [f64]),
) {
    let threads = softmax_2d_thread_count(rows, result.len());
    if threads <= 1 {
        for i in 0..rows {
            let start = i * cols;
            row_fn(&x[start..start + cols], &mut result[start..start + cols]);
        }
        return;
    }

    let rows_per = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut dst_rest: &mut [f64] = result;
        let mut row0 = 0usize;
        while row0 < rows {
            let row_count = rows_per.min(rows - row0);
            let elem_count = row_count * cols;
            let (dst_block, dst_tail) = dst_rest.split_at_mut(elem_count);
            dst_rest = dst_tail;
            let src_block = &x[row0 * cols..(row0 + row_count) * cols];
            row0 += row_count;
            scope.spawn(move || {
                for (src_row, dst_row) in src_block
                    .chunks_exact(cols)
                    .zip(dst_block.chunks_exact_mut(cols))
                {
                    row_fn(src_row, dst_row);
                }
            });
        }
    });
}

/// Softmax along the last axis of a 2D array.
///
/// Matches `jax.nn.softmax(x, axis=-1)` for a 2D array.
/// Returns a flattened result with the same shape as input.
///
/// Each row is computed in place directly into the output buffer — no per-row
/// `Vec` allocation or copy — using the SAME operations in the SAME order as the
/// 1D [`softmax`] (`exp(x-max)` written to the output slot, summed, then divided
/// in place), so the result is bit-for-bit identical to mapping [`softmax`] over
/// the rows. The per-row heap allocation [`softmax`] performs dominated the
/// many-rows/small-cols batched regime; this removes it.
#[must_use]
pub fn softmax_2d(x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let len = checked_2d_row_major_len("2D softmax", x, rows, cols);
    let mut result = vec![0.0; len];
    if cols == 0 {
        return result;
    }
    fill_softmax_rows_parallel(x, &mut result, rows, cols, softmax_row_into);
    result
}

/// Log-softmax: log(softmax(x)) = x - logsumexp(x)
///
/// Matches `jax.nn.log_softmax(x)` for a 1D array.
/// More numerically stable than computing log(softmax(x)) directly.
#[must_use]
pub fn log_softmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return Vec::new();
    }
    let lse = logsumexp(x);
    threaded_f64_map_bw(x, move |v| v - lse)
}

#[inline]
fn log_softmax_row_into(src: &[f64], dst: &mut [f64]) {
    let max_val = src.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lse = if max_val.is_infinite() {
        max_val
    } else {
        let sum_exp: f64 = src.iter().map(|&v| (v - max_val).exp()).sum();
        max_val + sum_exp.ln()
    };
    for (d, &v) in dst.iter_mut().zip(src.iter()) {
        *d = v - lse;
    }
}

/// Log-softmax along the last axis of a 2D array.
///
/// Matches `jax.nn.log_softmax(x, axis=-1)` for a 2D array.
///
/// Each row is computed in place directly into the output buffer — no per-row
/// `Vec` allocation or copy — using the SAME operations in the SAME order as the
/// 1D [`log_softmax`] (`lse = max + ln(sum(exp(x-max)))`, then `x - lse`), so the
/// result is bit-for-bit identical to mapping [`log_softmax`] over the rows.
#[must_use]
pub fn log_softmax_2d(x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let len = checked_2d_row_major_len("2D softmax", x, rows, cols);
    let mut result = vec![0.0; len];
    if cols == 0 {
        return result;
    }
    fill_softmax_rows_parallel(x, &mut result, rows, cols, log_softmax_row_into);
    result
}

#[inline]
fn log_softmax_row_shifted_into(src: &[f64], dst: &mut [f64]) {
    // Log-softmax computed in the SAME operation grouping as the decomposed XLA/JAX
    // graph: `shifted = x - rowmax`, then `shifted - log(sum(exp(shifted)))`. This
    // differs from `log_softmax_row_into`'s `x - (rowmax + log(sum))` grouping by float
    // (non-)associativity of the final subtract, so the two are NOT interchangeable for a
    // bit-exact superinstruction. The row-max fold and index-order exp-sum are identical to
    // `softmax_row_into`, so this equals the ReduceMax/Sub/Exp/ReduceSum/Log/Sub jaxpr
    // bit-for-bit. Superinstruction callers guarantee finite input, so `rowmax` is finite.
    let max_val = src.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum_exp = 0.0;
    for (d, &v) in dst.iter_mut().zip(src.iter()) {
        let shifted = v - max_val;
        *d = shifted;
        sum_exp += shifted.exp();
    }
    let log_val = sum_exp.ln();
    for d in dst.iter_mut() {
        *d -= log_val;
    }
}

/// Log-softmax along the last axis of a 2D array, computed in the operation grouping
/// of the decomposed XLA/JAX graph `shifted - log(sum(exp(shifted)))` (`shifted = x -
/// rowmax`). Used by the interpreter log-softmax superinstruction so the fused result is
/// bit-for-bit identical to the generic `ReduceMax → Sub → Exp → ReduceSum → Log → Sub`
/// path. (Contrast [`log_softmax_2d`], which uses the 1D `x - (rowmax + log(sum))`
/// grouping and so can differ in the last bit.)
#[must_use]
pub fn log_softmax_2d_shifted(x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let len = checked_2d_row_major_len("2D log_softmax", x, rows, cols);
    let mut result = vec![0.0; len];
    if cols == 0 {
        return result;
    }
    fill_softmax_rows_parallel(x, &mut result, rows, cols, log_softmax_row_shifted_into);
    result
}

#[inline]
fn rms_norm_row_into(src: &[f64], dst: &mut [f64], cols_f64: f64, epsilon: f64) {
    // RMSNorm row: `x * rsqrt(mean(x²) + eps)`, no mean subtraction. Bit-identical to the
    // decomposed Square → ReduceSum → Div → Add → Rsqrt → Mul jaxpr: `x*x` Square, index-order
    // sum, `1/√·` Rsqrt (matching the layer-norm superinstruction's op forms).
    let mut sum_sq = 0.0;
    for &value in src {
        sum_sq += value * value;
    }
    let inv_rms = 1.0 / ((sum_sq / cols_f64) + epsilon).sqrt();
    for (d, &value) in dst.iter_mut().zip(src.iter()) {
        *d = value * inv_rms;
    }
}

/// RMS-norm along the last axis of a 2D array: `x * rsqrt(mean(x²) + eps)` (no learned scale —
/// the modern-transformer normalization). Row-parallel (rows are independent), bit-for-bit
/// identical to the decomposed jaxpr and to the serial per-row fold. Used by the interpreter
/// RMS-norm superinstruction. `epsilon` is a runtime value, so this cannot reuse the
/// fn-pointer [`fill_softmax_rows_parallel`] and threads via a captured closure.
#[must_use]
pub fn rms_norm_2d(x: &[f64], rows: usize, cols: usize, epsilon: f64) -> Vec<f64> {
    let len = checked_2d_row_major_len("2D rms_norm", x, rows, cols);
    let mut result = vec![0.0; len];
    if cols == 0 {
        return result;
    }
    let cols_f64 = cols as f64;
    let threads = softmax_2d_thread_count(rows, len);
    if threads <= 1 {
        for i in 0..rows {
            let start = i * cols;
            rms_norm_row_into(
                &x[start..start + cols],
                &mut result[start..start + cols],
                cols_f64,
                epsilon,
            );
        }
        return result;
    }

    let rows_per = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut dst_rest: &mut [f64] = &mut result;
        let mut row0 = 0usize;
        while row0 < rows {
            let row_count = rows_per.min(rows - row0);
            let elem_count = row_count * cols;
            let (dst_block, dst_tail) = dst_rest.split_at_mut(elem_count);
            dst_rest = dst_tail;
            let src_block = &x[row0 * cols..(row0 + row_count) * cols];
            row0 += row_count;
            scope.spawn(move || {
                for (src_row, dst_row) in src_block
                    .chunks_exact(cols)
                    .zip(dst_block.chunks_exact_mut(cols))
                {
                    rms_norm_row_into(src_row, dst_row, cols_f64, epsilon);
                }
            });
        }
    });
    result
}

#[inline]
fn softmax_cross_entropy_row(logits: &[f64], labels: &[f64]) -> f64 {
    // `-Σ_j labels[j] · log_softmax(logits)[j]`, bit-identical to the decomposed graph:
    // log_softmax[j] = `(logits[j] - rowmax) - log(Σ exp(shifted))` (the log-softmax
    // superinstruction's `shifted - log` grouping), then `Mul(labels, ls)` per element, an
    // index-order `ReduceSum`, and a final `Neg`. Callers guarantee finite input → rowmax finite.
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum_exp = 0.0;
    for &value in logits {
        sum_exp += (value - max_val).exp();
    }
    let log_val = sum_exp.ln();
    let mut s = 0.0;
    for (&value, &label) in logits.iter().zip(labels.iter()) {
        s += label * ((value - max_val) - log_val);
    }
    -s
}

/// Softmax cross-entropy loss `-Σ labels · log_softmax(logits)` along the last axis of a 2D
/// array, one scalar per row (`optax.softmax_cross_entropy`). Row-parallel; bit-for-bit
/// identical to the decomposed log-softmax → Mul → ReduceSum → Neg jaxpr (same `shifted - log`
/// grouping + index-order reductions as the log-softmax superinstruction). Used by the
/// interpreter cross-entropy superinstruction.
#[must_use]
pub fn softmax_cross_entropy_2d(logits: &[f64], labels: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let _ = checked_2d_row_major_len("2D softmax_cross_entropy logits", logits, rows, cols);
    let _ = checked_2d_row_major_len("2D softmax_cross_entropy labels", labels, rows, cols);
    let mut result = vec![0.0; rows];
    if cols == 0 || rows == 0 {
        return result;
    }
    let threads = softmax_2d_thread_count(rows, rows * cols);
    if threads <= 1 {
        for (i, slot) in result.iter_mut().enumerate() {
            let start = i * cols;
            *slot = softmax_cross_entropy_row(
                &logits[start..start + cols],
                &labels[start..start + cols],
            );
        }
        return result;
    }

    let rows_per = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut out_rest: &mut [f64] = &mut result;
        let mut row0 = 0usize;
        while row0 < rows {
            let row_count = rows_per.min(rows - row0);
            let (out_block, out_tail) = out_rest.split_at_mut(row_count);
            out_rest = out_tail;
            let logits_block = &logits[row0 * cols..(row0 + row_count) * cols];
            let labels_block = &labels[row0 * cols..(row0 + row_count) * cols];
            row0 += row_count;
            scope.spawn(move || {
                for (k, out_slot) in out_block.iter_mut().enumerate() {
                    let start = k * cols;
                    *out_slot = softmax_cross_entropy_row(
                        &logits_block[start..start + cols],
                        &labels_block[start..start + cols],
                    );
                }
            });
        }
    });
    result
}

/// Normalize input to have zero mean and unit variance.
///
/// Matches `jax.nn.standardize(x)` for a 1D array.
#[must_use]
pub fn standardize(x: &[f64], epsilon: f64) -> Vec<f64> {
    if x.is_empty() {
        return Vec::new();
    }
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = (var + epsilon).sqrt();
    x.iter().map(|&v| (v - mean) / std).collect()
}

/// Soft-sign: x / (|x| + 1)
///
/// JAX-named alias of [`softsign`] (`jax.nn` exposes this as `soft_sign`).
#[must_use]
pub fn soft_sign(x: &[f64]) -> Vec<f64> {
    softsign(x)
}

/// Identity: returns the input unmodified.
///
/// Matches `jax.nn.identity(x)`.
#[must_use]
pub fn identity(x: &[f64]) -> Vec<f64> {
    x.to_vec()
}

/// Squareplus: (x + sqrt(x^2 + b)) / 2
///
/// A smooth approximation of ReLU. `b` is the smoothness parameter (JAX default
/// 4). Matches `jax.nn.squareplus(x, b)`.
#[must_use]
pub fn squareplus(x: &[f64], b: f64) -> Vec<f64> {
    x.iter().map(|&v| (v + (v * v + b).sqrt()) / 2.0).collect()
}

/// Sparse plus: 0 for x <= -1, (x+1)^2/4 for -1 < x < 1, x for x >= 1.
///
/// The twin of softplus (zero output below -1, linear above 1, smooth/convex
/// between). Matches `jax.nn.sparse_plus(x)`:
/// `where(x <= -1, 0, where(x >= 1, x, (x+1)^2/4))`.
#[must_use]
pub fn sparse_plus(x: &[f64]) -> Vec<f64> {
    x.iter()
        .map(|&v| {
            if v <= -1.0 {
                0.0
            } else if v >= 1.0 {
                v
            } else {
                (v + 1.0).powi(2) / 4.0
            }
        })
        .collect()
}

/// Sparse sigmoid: 0.5 * clip(x + 1, 0, 2).
///
/// The derivative of [`sparse_plus`] and the twin of sigmoid (0 below -1, 1
/// above 1, linear between). Matches `jax.nn.sparse_sigmoid(x)`.
#[must_use]
pub fn sparse_sigmoid(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| 0.5 * (v + 1.0).clamp(0.0, 2.0)).collect()
}

/// Numerically stable `log(1 - exp(-x))`, undefined for x < 0.
///
/// Matches `jax.nn.log1mexp(x)`: with `c = ln 2`, uses `ln(-expm1(-x))` for
/// `x < c` and `ln1p(-exp(-x))` otherwise — the standard Mächler split that
/// keeps both branches accurate near 0 and near +inf.
#[must_use]
pub fn log1mexp(x: &[f64]) -> Vec<f64> {
    let c = 2.0_f64.ln();
    x.iter()
        .map(|&v| {
            if v < c {
                // -expm1(-v) = 1 - exp(-v), accurate for small v.
                (-(-v).exp_m1()).ln()
            } else {
                // ln1p(-exp(-v)) = ln(1 - exp(-v)), accurate for large v.
                (-(-v).exp()).ln_1p()
            }
        })
        .collect()
}

/// Gated linear unit: splits the input in half and gates the first half by the
/// sigmoid of the second.
///
/// Matches `jax.nn.glu(x, axis=-1)` for a 1-D input: with `n = len`, returns
/// `x[0:n/2] * sigmoid(x[n/2:n])` (length `n/2`). The axis size must be even;
/// an odd middle element (ill-defined per JAX, which asserts) is ignored.
#[must_use]
pub fn glu(x: &[f64]) -> Vec<f64> {
    let half = x.len() / 2;
    let (a, b) = (&x[..half], &x[half..half + half]);
    a.iter()
        .zip(b.iter())
        .map(|(&g, &v)| g * (1.0 / (1.0 + (-v).exp())))
        .collect()
}

/// Log-mean-exp: `logsumexp(x) - ln(n)`.
///
/// The log of the mean of the exponentials, a numerically stable companion to
/// [`logsumexp`]. Matches `jax.nn.logmeanexp(x)` reduced over all elements.
/// Empty input returns -inf (consistent with [`logsumexp`]).
#[must_use]
pub fn logmeanexp(x: &[f64]) -> f64 {
    if x.is_empty() {
        return f64::NEG_INFINITY;
    }
    logsumexp(x) - (x.len() as f64).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// Per-row reference: map the 1D helper over rows (the prior `softmax_2d`
    /// body). The fused in-place kernel must be BIT-for-BIT identical to this.
    fn softmax_2d_rowmap_ref(x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        let mut result = vec![0.0; rows * cols];
        for i in 0..rows {
            let row = &x[i * cols..(i + 1) * cols];
            let sm = softmax(row);
            result[i * cols..(i + 1) * cols].copy_from_slice(&sm);
        }
        result
    }

    fn log_softmax_2d_rowmap_ref(x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        let mut result = vec![0.0; rows * cols];
        for i in 0..rows {
            let row = &x[i * cols..(i + 1) * cols];
            let lsm = log_softmax(row);
            result[i * cols..(i + 1) * cols].copy_from_slice(&lsm);
        }
        result
    }

    fn softmax_2d_fused_serial_ref(x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        let mut result = vec![0.0; rows * cols];
        for i in 0..rows {
            let start = i * cols;
            softmax_row_into(&x[start..start + cols], &mut result[start..start + cols]);
        }
        result
    }

    fn log_softmax_2d_fused_serial_ref(x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
        let mut result = vec![0.0; rows * cols];
        for i in 0..rows {
            let start = i * cols;
            log_softmax_row_into(&x[start..start + cols], &mut result[start..start + cols]);
        }
        result
    }

    #[test]
    fn softmax_2d_fused_bit_identical_to_rowmap() {
        // Mixed magnitudes, signs, duplicates, and a row with a large value to
        // exercise the max-subtraction; assert raw bit equality (not approx).
        let rows = 7;
        let cols = 5;
        let x: Vec<f64> = (0..rows * cols)
            .map(|k| ((k as f64) * 0.37).sin() * 1000.0 - (k as f64) * 1.5)
            .collect();
        let fused = softmax_2d(&x, rows, cols);
        let reference = softmax_2d_rowmap_ref(&x, rows, cols);
        assert_eq!(fused.len(), reference.len());
        for (a, b) in fused.iter().zip(reference.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "softmax_2d diverged: {a} vs {b}");
        }
    }

    #[test]
    fn log_softmax_2d_fused_bit_identical_to_rowmap() {
        let rows = 6;
        let cols = 8;
        let x: Vec<f64> = (0..rows * cols)
            .map(|k| ((k as f64) * 0.91).cos() * 50.0 + (k as f64))
            .collect();
        let fused = log_softmax_2d(&x, rows, cols);
        let reference = log_softmax_2d_rowmap_ref(&x, rows, cols);
        assert_eq!(fused.len(), reference.len());
        for (a, b) in fused.iter().zip(reference.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "log_softmax_2d diverged: {a} vs {b}"
            );
        }
    }

    fn softmax_parallel_fixture(rows: usize, cols: usize) -> Vec<f64> {
        let mut x: Vec<f64> = (0..rows * cols)
            .map(|k| ((k as f64) * 0.017).sin() * 40.0 - ((k % cols) as f64) * 0.125)
            .collect();
        if rows >= 4 && cols >= 5 {
            x[cols] = f64::INFINITY;
            x[cols + 1] = 17.0;
            x[2 * cols] = f64::NEG_INFINITY;
            x[2 * cols + 1] = -300.0;
            x[3 * cols] = f64::NAN;
            x[3 * cols + 1] = 3.5;
            x[3 * cols + 2] = -0.0;
            x[3 * cols + 3] = 0.0;
            x[3 * cols + 4] = f64::INFINITY;
        }
        x
    }

    #[test]
    fn softmax_2d_parallel_bit_identical_to_serial_fused() -> Result<(), Box<dyn std::error::Error>>
    {
        let cols = 17;
        let rows = (SOFTMAX_2D_PARALLEL_MIN / cols) + 64;
        let x = softmax_parallel_fixture(rows, cols);
        if std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
            > 1
        {
            assert!(softmax_2d_thread_count(rows, rows * cols) > 1);
        }

        let parallel = softmax_2d(&x, rows, cols);
        let reference = softmax_2d_fused_serial_ref(&x, rows, cols);
        assert_eq!(parallel.len(), reference.len());
        for (idx, (a, b)) in parallel.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "parallel softmax_2d diverged at {idx}: {a} vs {b}"
            );
        }

        let output_bits: Vec<u64> = parallel.iter().map(|v| v.to_bits()).collect();
        let digest = fj_test_utils::fixture_id_from_json(&output_bits)?;
        assert_eq!(
            digest, "804d2a4abc52612601a766311fbe9a8e230766c65110b667b35676a7803bc6dd",
            "softmax_2d parallel golden output digest changed"
        );
        Ok(())
    }

    #[test]
    fn log_softmax_2d_parallel_bit_identical_to_serial_fused()
    -> Result<(), Box<dyn std::error::Error>> {
        let cols = 17;
        let rows = (SOFTMAX_2D_PARALLEL_MIN / cols) + 64;
        let x = softmax_parallel_fixture(rows, cols);
        if std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
            > 1
        {
            assert!(softmax_2d_thread_count(rows, rows * cols) > 1);
        }

        let parallel = log_softmax_2d(&x, rows, cols);
        let reference = log_softmax_2d_fused_serial_ref(&x, rows, cols);
        assert_eq!(parallel.len(), reference.len());
        for (idx, (a, b)) in parallel.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "parallel log_softmax_2d diverged at {idx}: {a} vs {b}"
            );
        }

        let output_bits: Vec<u64> = parallel.iter().map(|v| v.to_bits()).collect();
        let digest = fj_test_utils::fixture_id_from_json(&output_bits)?;
        assert_eq!(
            digest, "34e1e76bbcfde76e7bb49161efd7e6b7a8225967e35aa16a6c2ab41d96b8e2d2",
            "log_softmax_2d parallel golden output digest changed"
        );
        Ok(())
    }

    #[test]
    fn softmax_2d_log_softmax_2d_handle_zero_cols() {
        assert!(softmax_2d(&[], 0, 0).is_empty());
        assert!(log_softmax_2d(&[], 3, 0).is_empty());
    }

    #[test]
    #[should_panic(expected = "2D logsumexp output shape overflow")]
    fn logsumexp_2d_rejects_row_shape_overflow_before_indexing() {
        let _ = logsumexp_2d(&[], usize::MAX, 2);
    }

    #[test]
    #[should_panic(expected = "2D logsumexp input length 0 is shorter than declared shape")]
    fn logsumexp_2d_rejects_oversized_declared_shape_before_indexing() {
        let _ = logsumexp_2d(&[], usize::MAX, 1);
    }

    #[test]
    #[should_panic(expected = "2D softmax output shape overflow")]
    fn softmax_2d_rejects_output_shape_overflow_before_allocation() {
        let _ = softmax_2d(&[], usize::MAX, 2);
    }

    #[test]
    #[should_panic(expected = "2D softmax output shape overflow")]
    fn log_softmax_2d_rejects_output_shape_overflow_before_allocation() {
        let _ = log_softmax_2d(&[], usize::MAX, 2);
    }

    #[test]
    #[should_panic(expected = "2D softmax input length 0 is shorter than declared shape")]
    fn softmax_2d_rejects_oversized_declared_shape_before_allocation() {
        let _ = softmax_2d(&[], usize::MAX, 1);
    }

    #[test]
    fn test_relu_basic() {
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let y = relu(&x);
        assert_eq!(y, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_soft_sign_aliases_softsign() {
        let x = vec![-2.0, -0.5, 0.0, 0.5, 3.0];
        assert_eq!(soft_sign(&x), softsign(&x));
        // x / (|x| + 1)
        assert!(approx_eq(soft_sign(&x)[0], -2.0 / 3.0, 1e-12));
        assert!(approx_eq(soft_sign(&x)[3], 0.5 / 1.5, 1e-12));
    }

    #[test]
    fn test_identity_returns_input() {
        let x = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        assert_eq!(identity(&x), x);
    }

    #[test]
    fn test_squareplus_matches_formula() {
        // squareplus(x, b) = (x + sqrt(x^2 + b)) / 2; default b = 4.
        let x = vec![-1.0, 0.0, 2.0];
        let y = squareplus(&x, 4.0);
        assert!(approx_eq(y[0], (-1.0 + 5.0_f64.sqrt()) / 2.0, 1e-12));
        assert!(approx_eq(y[1], 1.0, 1e-12)); // (0 + 2)/2
        assert!(approx_eq(y[2], (2.0 + 8.0_f64.sqrt()) / 2.0, 1e-12));
        // x=0, b=0 collapses to relu-like 0; large x ~ x.
        assert!(approx_eq(squareplus(&[100.0], 4.0)[0], 100.0, 1e-2));
    }

    #[test]
    fn test_sparse_plus_piecewise() {
        // 0 for x<=-1, (x+1)^2/4 for -1<x<1, x for x>=1.
        let x = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 3.0];
        let y = sparse_plus(&x);
        assert!(approx_eq(y[0], 0.0, 1e-12));
        assert!(approx_eq(y[1], 0.0, 1e-12)); // x = -1 -> 0
        assert!(approx_eq(y[2], 0.25, 1e-12)); // (0+1)^2/4
        assert!(approx_eq(y[3], (1.5_f64).powi(2) / 4.0, 1e-12));
        assert!(approx_eq(y[4], 1.0, 1e-12)); // x = 1 -> x
        assert!(approx_eq(y[5], 3.0, 1e-12));
    }

    #[test]
    fn test_sparse_sigmoid_piecewise() {
        // 0.5 * clip(x+1, 0, 2): 0 for x<=-1, (x+1)/2 between, 1 for x>=1.
        let x = vec![-2.0, -1.0, 0.0, 1.0, 5.0];
        let y = sparse_sigmoid(&x);
        assert!(approx_eq(y[0], 0.0, 1e-12));
        assert!(approx_eq(y[1], 0.0, 1e-12));
        assert!(approx_eq(y[2], 0.5, 1e-12)); // (0+1)/2
        assert!(approx_eq(y[3], 1.0, 1e-12));
        assert!(approx_eq(y[4], 1.0, 1e-12));
    }

    #[test]
    fn test_log1mexp_stable_both_branches() {
        // log(1 - exp(-x)). Small-x branch (x < ln2) and large-x branch.
        let small = 0.1_f64;
        assert!(approx_eq(
            log1mexp(&[small])[0],
            (1.0 - (-small).exp()).ln(),
            1e-9
        ));
        let large = 5.0_f64;
        assert!(approx_eq(
            log1mexp(&[large])[0],
            (1.0 - (-large).exp()).ln(),
            1e-12
        ));
        // As x -> inf, 1 - exp(-x) -> 1, so log -> 0^-.
        assert!(log1mexp(&[40.0])[0].abs() < 1e-15);
    }

    #[test]
    fn test_glu_gates_first_half_by_sigmoid_of_second() {
        // x = [a0, a1, b0, b1] -> [a0*sigmoid(b0), a1*sigmoid(b1)].
        let x = vec![1.0, 2.0, 0.0, 100.0];
        let y = glu(&x);
        assert_eq!(y.len(), 2);
        assert!(approx_eq(y[0], 1.0 * 0.5, 1e-12)); // sigmoid(0) = 0.5
        assert!(approx_eq(y[1], 2.0 * 1.0, 1e-9)); // sigmoid(100) ~ 1
    }

    #[test]
    fn test_logmeanexp_is_logsumexp_minus_log_n() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        assert!(approx_eq(
            logmeanexp(&x),
            logsumexp(&x) - (4.0_f64).ln(),
            1e-12
        ));
        // mean of equal values v is v: logmeanexp([v;n]) == v.
        assert!(approx_eq(logmeanexp(&[2.5, 2.5, 2.5]), 2.5, 1e-12));
        assert_eq!(logmeanexp(&[]), f64::NEG_INFINITY);
    }

    #[test]
    fn test_leaky_relu_basic() {
        let x = vec![-2.0, 0.0, 2.0];
        let y = leaky_relu(&x, 0.1);
        assert!(approx_eq(y[0], -0.2, 1e-10));
        assert!(approx_eq(y[1], 0.0, 1e-10));
        assert!(approx_eq(y[2], 2.0, 1e-10));
    }

    #[test]
    fn test_relu6_clips() {
        let x = vec![-1.0, 0.0, 3.0, 6.0, 10.0];
        let y = relu6(&x);
        assert_eq!(y, vec![0.0, 0.0, 3.0, 6.0, 6.0]);
    }

    #[test]
    fn test_sigmoid_bounds() {
        let x = vec![-100.0, 0.0, 100.0];
        let y = sigmoid(&x);
        assert!(y[0] < 0.001);
        assert!(approx_eq(y[1], 0.5, 1e-10));
        assert!(y[2] > 0.999);
    }

    #[test]
    fn test_hard_sigmoid_piecewise() {
        let x = vec![-10.0, -3.0, 0.0, 3.0, 10.0];
        let y = hard_sigmoid(&x);
        assert!(approx_eq(y[0], 0.0, 1e-10));
        assert!(approx_eq(y[1], 0.0, 1e-10));
        assert!(approx_eq(y[2], 0.5, 1e-10));
        assert!(approx_eq(y[3], 1.0, 1e-10));
        assert!(approx_eq(y[4], 1.0, 1e-10));
    }

    #[test]
    fn test_hard_tanh_clips() {
        let x = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
        let y = hard_tanh(&x);
        assert_eq!(y, vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_silu_at_zero() {
        let x = vec![0.0];
        let y = silu(&x);
        assert!(approx_eq(y[0], 0.0, 1e-10));
    }

    #[test]
    fn test_swish_is_silu() {
        let x = vec![-1.0, 0.0, 1.0, 2.0];
        assert_eq!(silu(&x), swish(&x));
    }

    #[test]
    fn test_gelu_symmetry() {
        let x = vec![-1.0, 1.0];
        let y = gelu(&x);
        // GELU is not symmetric, but gelu(-x) + gelu(x) ≈ x for small x
        // Just check it computes something reasonable
        assert!(y[0] < 0.0);
        assert!(y[1] > 0.0);
    }

    #[test]
    fn test_elu_at_zero() {
        let x = vec![0.0];
        let y = elu(&x, 1.0);
        assert!(approx_eq(y[0], 0.0, 1e-10));
    }

    #[test]
    fn test_elu_negative() {
        let x = vec![-1.0];
        let y = elu(&x, 1.0);
        // elu(-1) = exp(-1) - 1 ≈ -0.632
        assert!(approx_eq(y[0], (-1.0_f64).exp() - 1.0, 1e-10));
    }

    #[test]
    fn test_elu_celu_selu_use_expm1_for_negative_branch() {
        // JAX's elu/celu/selu use expm1 in the negative branch so it stays
        // accurate near x=0 (exp(x)-1 loses ~|x| relative precision to
        // cancellation). Guard bit-identity to the expm1 form across magnitudes;
        // a revert to exp(x)-1 would differ for small |x|.
        const ALPHA: f64 = 1.6732632423543772;
        const SCALE: f64 = 1.0507009873554805;
        for &x in &[-1e-12, -1e-8, -1e-4, -0.5, -3.0] {
            assert_eq!(
                elu(&[x], 1.0)[0].to_bits(),
                x.exp_m1().to_bits(),
                "elu({x})"
            );
            assert_eq!(
                elu(&[x], 2.0)[0].to_bits(),
                (2.0 * x.exp_m1()).to_bits(),
                "elu({x}, alpha=2)"
            );
            assert_eq!(
                selu(&[x])[0].to_bits(),
                (SCALE * (ALPHA * x.exp_m1())).to_bits(),
                "selu({x})"
            );
            assert_eq!(
                celu(&[x], 1.0)[0].to_bits(),
                x.exp_m1().to_bits(),
                "celu({x})"
            );
        }
        // And the expm1 form is strictly more accurate than exp(x)-1 near 0:
        // elu(-1e-10) should be ~-1e-10; the naive form is off by ~1e-16.
        let x = -1e-10_f64;
        let elu_val = elu(&[x], 1.0)[0];
        let naive = x.exp() - 1.0;
        assert!(
            (elu_val - x).abs() < (naive - x).abs(),
            "expm1 form must be more accurate than exp-1: expm1={elu_val}, naive={naive}"
        );
    }

    #[test]
    fn test_selu_self_normalizing_constants() {
        // SELU should preserve mean 0 and variance 1 for standard normal inputs
        // Just check the constants are applied
        let x = vec![1.0];
        let y = selu(&x);
        assert!(approx_eq(y[0], 1.0507009873554805, 1e-10));
    }

    #[test]
    fn test_softplus_positive() {
        let x = vec![-2.0, 0.0, 2.0];
        let y = softplus(&x);
        assert!(y.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_softplus_large_input() {
        let x = vec![100.0];
        let y = softplus(&x);
        assert!(approx_eq(y[0], 100.0, 1e-10));
    }

    #[test]
    fn test_softsign_bounds() {
        let x = vec![-100.0, 0.0, 100.0];
        let y = softsign(&x);
        assert!(y[0] > -1.0 && y[0] < -0.99);
        assert!(approx_eq(y[1], 0.0, 1e-10));
        assert!(y[2] < 1.0 && y[2] > 0.99);
    }

    #[test]
    fn test_mish_at_zero() {
        let x = vec![0.0];
        let y = mish(&x);
        assert!(approx_eq(y[0], 0.0, 1e-10));
    }

    #[test]
    fn test_log_sigmoid_bounds() {
        let x = vec![-100.0, 0.0, 100.0];
        let y = log_sigmoid(&x);
        assert!(approx_eq(y[0], -100.0, 1.0)); // approximately -100
        assert!(approx_eq(y[1], -std::f64::consts::LN_2, 1e-5)); // ln(0.5)
        assert!(approx_eq(y[2], 0.0, 1e-10));
    }

    #[test]
    fn test_softplus_exact_logaddexp_no_threshold() {
        // softplus(x) == logaddexp(x, 0) == max(x,0) + log1p(exp(-|x|)) for ALL x.
        // The previous ±20 threshold approximation dropped the log1p correction
        // term outside [-20, 20]; e.g. softplus(21) was 21.0 exactly. Pin the
        // exact value and prove the correction term is retained.
        let xs: [f64; 10] = [-50.0, -25.0, -21.0, -5.0, 0.0, 5.0, 21.0, 25.0, 50.0, 100.0];
        for &x in &xs {
            let logaddexp = x.max(0.0) + (-x.abs()).exp().ln_1p();
            let got = softplus(&[x])[0];
            assert_eq!(
                got.to_bits(),
                logaddexp.to_bits(),
                "softplus({x}) bit mismatch"
            );
            assert!(got.is_finite(), "softplus({x}) must be finite");
        }
        // The correction term is retained just past the old +20 cut, where the
        // previous threshold approximation returned exactly 21.0 (the loop above
        // already pins the full bit-exact value against logaddexp at x = 21).
        assert!(
            softplus(&[21.0])[0] > 21.0,
            "softplus(21) must keep the log1p term"
        );

        // mish and log_sigmoid are defined via the same exact softplus.
        for &x in &xs {
            let m = mish(&[x])[0];
            assert_eq!(m.to_bits(), (x * softplus(&[x])[0].tanh()).to_bits());
            let ls = log_sigmoid(&[x])[0];
            assert_eq!(ls.to_bits(), (-softplus(&[-x])[0]).to_bits());
        }

        // ±inf / NaN propagate as JAX's logaddexp does.
        assert_eq!(softplus(&[f64::INFINITY])[0], f64::INFINITY);
        assert_eq!(softplus(&[f64::NEG_INFINITY])[0], 0.0);
        assert!(softplus(&[f64::NAN])[0].is_nan());
    }

    #[test]
    fn test_celu_continuous() {
        let x = vec![-0.01, 0.0, 0.01];
        let y = celu(&x, 1.0);
        // Should be continuous at 0
        assert!(approx_eq(y[1], 0.0, 1e-10));
        assert!((y[2] - y[0]).abs() < 0.03);
    }

    #[test]
    fn test_hard_silu_at_boundaries() {
        let x = vec![-3.0, 0.0, 3.0];
        let y = hard_silu(&x);
        assert!(approx_eq(y[0], 0.0, 1e-10)); // -3 * 0 = 0
        assert!(approx_eq(y[1], 0.0, 1e-10)); // 0 * 0.5 = 0
        assert!(approx_eq(y[2], 3.0, 1e-10)); // 3 * 1 = 3
    }

    #[test]
    fn test_hard_swish_is_hard_silu() {
        let x = vec![-1.0, 0.0, 1.0, 2.0];
        assert_eq!(hard_silu(&x), hard_swish(&x));
    }

    #[test]
    fn test_logsumexp_basic() {
        let x = vec![0.0, 0.0];
        let lse = logsumexp(&x);
        assert!(approx_eq(lse, 2.0_f64.ln(), 1e-10));
    }

    #[test]
    fn test_logsumexp_large_values() {
        let x = vec![1000.0, 1000.0];
        let lse = logsumexp(&x);
        assert!(approx_eq(lse, 1000.0 + 2.0_f64.ln(), 1e-10));
    }

    #[test]
    fn test_logsumexp_small_values() {
        let x = vec![-1000.0, -1000.0];
        let lse = logsumexp(&x);
        assert!(approx_eq(lse, -1000.0 + 2.0_f64.ln(), 1e-10));
    }

    #[test]
    fn test_logsumexp_empty() {
        let x: Vec<f64> = vec![];
        let lse = logsumexp(&x);
        assert!(lse.is_infinite() && lse < 0.0);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let x = vec![1.0, 2.0, 3.0];
        let sm = softmax(&x);
        let sum: f64 = sm.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-10));
    }

    #[test]
    fn test_softmax_uniform() {
        let x = vec![1.0, 1.0, 1.0];
        let sm = softmax(&x);
        for &v in &sm {
            assert!(approx_eq(v, 1.0 / 3.0, 1e-10));
        }
    }

    #[test]
    fn test_softmax_preserves_order() {
        let x = vec![1.0, 2.0, 3.0];
        let sm = softmax(&x);
        assert!(sm[0] < sm[1] && sm[1] < sm[2]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let x = vec![1000.0, 1001.0, 1002.0];
        let sm = softmax(&x);
        let sum: f64 = sm.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-10));
        assert!(sm.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_log_softmax_basic() {
        let x = vec![1.0, 2.0, 3.0];
        let lsm = log_softmax(&x);
        let sm = softmax(&x);
        for (l, s) in lsm.iter().zip(sm.iter()) {
            assert!(approx_eq(*l, s.ln(), 1e-10));
        }
    }

    #[test]
    fn test_log_softmax_sums_to_zero_exp() {
        let x = vec![1.0, 2.0, 3.0];
        let lsm = log_softmax(&x);
        let sum_exp: f64 = lsm.iter().map(|&v| v.exp()).sum();
        assert!(approx_eq(sum_exp, 1.0, 1e-10));
    }

    #[test]
    fn test_log_softmax_numerical_stability() {
        let x = vec![1000.0, 1001.0, 1002.0];
        let lsm = log_softmax(&x);
        assert!(lsm.iter().all(|&v| v.is_finite()));
        let sum_exp: f64 = lsm.iter().map(|&v| v.exp()).sum();
        assert!(approx_eq(sum_exp, 1.0, 1e-10));
    }

    #[test]
    fn test_softmax_2d() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let sm = softmax_2d(&x, 2, 3);
        let row1_sum: f64 = sm[0..3].iter().sum();
        let row2_sum: f64 = sm[3..6].iter().sum();
        assert!(approx_eq(row1_sum, 1.0, 1e-10));
        assert!(approx_eq(row2_sum, 1.0, 1e-10));
    }

    #[test]
    fn test_log_softmax_2d() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let lsm = log_softmax_2d(&x, 2, 3);
        let row1_sum_exp: f64 = lsm[0..3].iter().map(|&v| v.exp()).sum();
        let row2_sum_exp: f64 = lsm[3..6].iter().map(|&v| v.exp()).sum();
        assert!(approx_eq(row1_sum_exp, 1.0, 1e-10));
        assert!(approx_eq(row2_sum_exp, 1.0, 1e-10));
    }

    #[test]
    fn test_standardize_zero_mean() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = standardize(&x, 1e-10);
        let mean: f64 = s.iter().sum::<f64>() / s.len() as f64;
        assert!(approx_eq(mean, 0.0, 1e-10));
    }

    #[test]
    fn test_standardize_unit_variance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = standardize(&x, 1e-10);
        let n = s.len() as f64;
        let mean = s.iter().sum::<f64>() / n;
        let var = s.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
        assert!(approx_eq(var, 1.0, 1e-10));
    }

    #[test]
    fn differential_activations_match_raw_f64_definitions() {
        // Differential checks vs references computed with raw std f64 ops (independent
        // of nn.rs internals) across a range — catches formula/precision drift the
        // existing point/property tests can miss. Range avoids exp overflow.
        let xs = [-5.0_f64, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0];
        for &x in &xs {
            // silu(x) = x * sigmoid(x) = x / (1 + e^-x)
            let sigmoid_ref = 1.0 / (1.0 + (-x).exp());
            assert!(
                approx_eq(silu(&[x])[0], x * sigmoid_ref, 1e-12),
                "silu({x}) vs x*sigmoid(x)"
            );
            // mish(x) = x * tanh(softplus(x)), softplus(x) = ln(1 + e^x)
            let mish_ref = x * x.exp().ln_1p().tanh();
            assert!(
                approx_eq(mish(&[x])[0], mish_ref, 1e-9),
                "mish({x}) vs x*tanh(softplus(x))"
            );
            // log_sigmoid(x) = ln(sigmoid(x)) = -ln(1 + e^-x)
            let log_sigmoid_ref = -(-x).exp().ln_1p();
            assert!(
                approx_eq(log_sigmoid(&[x])[0], log_sigmoid_ref, 1e-12),
                "log_sigmoid({x}) vs -ln(1+e^-x)"
            );
            // softplus(x) = ln(1 + e^x)
            assert!(
                approx_eq(softplus(&[x])[0], x.exp().ln_1p(), 1e-12),
                "softplus({x}) vs ln(1+e^x)"
            );
        }
    }

    #[test]
    fn differential_piecewise_activations_match_raw_f64_definitions() {
        // Piecewise / closed-form activations vs raw-f64 references (non-circular),
        // across a range. Distinct from the silu/mish batch.
        let xs = [-5.0_f64, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0, 7.0];
        for &x in &xs {
            // relu6(x) = min(max(x,0), 6)
            assert!(
                approx_eq(relu6(&[x])[0], x.clamp(0.0, 6.0), 1e-12),
                "relu6({x})"
            );
            // leaky_relu(x, s) = x if x>=0 else s*x
            let s = 0.1;
            let lr_ref = if x >= 0.0 { x } else { s * x };
            assert!(
                approx_eq(leaky_relu(&[x], s)[0], lr_ref, 1e-12),
                "leaky_relu({x})"
            );
            // elu(x, a) = x if x>0 else a*expm1(x)  (nn.rs uses expm1 on the neg branch)
            let a = 1.0;
            let elu_ref = if x > 0.0 { x } else { a * x.exp_m1() };
            assert!(approx_eq(elu(&[x], a)[0], elu_ref, 1e-12), "elu({x})");
            // hard_tanh(x) = clamp(x, -1, 1)
            assert!(
                approx_eq(hard_tanh(&[x])[0], x.clamp(-1.0, 1.0), 1e-12),
                "hard_tanh({x})"
            );
            // softsign(x) = x / (1 + |x|)
            assert!(
                approx_eq(softsign(&[x])[0], x / (1.0 + x.abs()), 1e-12),
                "softsign({x})"
            );
            // squareplus(x, b) = (x + sqrt(x^2 + b)) / 2
            let b = 4.0;
            let sp_ref = (x + (x * x + b).sqrt()) / 2.0;
            assert!(
                approx_eq(squareplus(&[x], b)[0], sp_ref, 1e-12),
                "squareplus({x})"
            );
        }
    }

    #[test]
    fn differential_logsumexp_matches_naive_definition_in_stable_regime() {
        // logsumexp(x) == ln(sum(exp(x))) for moderate values where the naive sum
        // doesn't overflow — cross-validates the numerically-stabilized impl against
        // the raw mathematical definition (non-circular, raw f64).
        let cases: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 2.0, 3.0],
            vec![-2.0, -1.0, 0.5, 1.5],
            vec![5.0, -3.0, 2.0],
            vec![-10.0, -10.0, -10.0],
        ];
        for x in &cases {
            let naive = x.iter().map(|&v| v.exp()).sum::<f64>().ln();
            assert!(
                approx_eq(logsumexp(x), naive, 1e-10),
                "logsumexp({x:?}) vs ln(sum(exp(x)))"
            );
        }
    }

    #[test]
    fn threaded_activations_bit_identical_to_sequential() {
        // The threaded compute-bound activations must equal the sequential map bit-for-bit
        // (pure elementwise; threading never reorders). Size > work threshold to engage it.
        let n = 5_000_000usize;
        let x: Vec<f64> = (0..n)
            .map(|i| ((i % 4001) as f64 - 2000.0) * 0.0017)
            .collect();
        let sqrt_2_over_pi = (2.0 / PI).sqrt();
        let gelu_seq: Vec<f64> = x
            .iter()
            .map(|&v| {
                let inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
                0.5 * v * (1.0 + inner.tanh())
            })
            .collect();
        let silu_seq: Vec<f64> = x.iter().map(|&v| v / (1.0 + (-v).exp())).collect();
        let softplus_seq: Vec<f64> = x.iter().map(|&v| softplus_scalar(v)).collect();
        let mish_seq: Vec<f64> = x.iter().map(|&v| v * softplus_scalar(v).tanh()).collect();
        for (a, b) in gelu(&x).iter().zip(&gelu_seq) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (a, b) in silu(&x).iter().zip(&silu_seq) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (a, b) in softplus(&x).iter().zip(&softplus_seq) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (a, b) in mish(&x).iter().zip(&mish_seq) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        // elu/celu/selu/log_sigmoid (expm1-bound) also threaded.
        let alpha = 1.3;
        let elu_seq: Vec<f64> = x
            .iter()
            .map(|&v| if v > 0.0 { v } else { alpha * v.exp_m1() })
            .collect();
        let celu_seq: Vec<f64> = x
            .iter()
            .map(|&v| v.max(0.0) + alpha * (v.min(0.0) / alpha).exp_m1())
            .collect();
        let selu_seq: Vec<f64> = x
            .iter()
            .map(|&v| {
                const A: f64 = 1.6732632423543772;
                const S: f64 = 1.0507009873554805;
                S * if v > 0.0 { v } else { A * v.exp_m1() }
            })
            .collect();
        let lsig_seq: Vec<f64> = x.iter().map(|&v| -softplus_scalar(-v)).collect();
        for (a, b) in elu(&x, alpha).iter().zip(&elu_seq) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (a, b) in celu(&x, alpha).iter().zip(&celu_seq) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (a, b) in selu(&x).iter().zip(&selu_seq) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (a, b) in log_sigmoid(&x).iter().zip(&lsig_seq) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        // softmax: threaded exp map must match the sequential reference bit-for-bit.
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_seq: Vec<f64> = x.iter().map(|&v| (v - max_val).exp()).collect();
        let sum_seq: f64 = exp_seq.iter().sum();
        let sm_seq: Vec<f64> = exp_seq.iter().map(|&e| e / sum_seq).collect();
        for (a, b) in softmax(&x).iter().zip(&sm_seq) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        let sigmoid_seq: Vec<f64> = x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
        for (a, b) in sigmoid(&x).iter().zip(&sigmoid_seq) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        // logsumexp/log_softmax: exp map may thread, but the sum remains in index order.
        let lse_seq = max_val + sum_seq.ln();
        assert_eq!(logsumexp(&x).to_bits(), lse_seq.to_bits());
        let lsm_seq: Vec<f64> = x.iter().map(|&v| v - lse_seq).collect();
        for (a, b) in log_softmax(&x).iter().zip(&lsm_seq) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_softmax_1d_threaded_vs_sequential() {
        use std::time::Instant;
        let n = 16_000_000usize;
        let x: Vec<f64> = (0..n)
            .map(|i| ((i % 4001) as f64 - 2000.0) * 0.001)
            .collect();
        let bench = |label: &str, f: &dyn Fn()| {
            f();
            let mut b = f64::MAX;
            for _ in 0..5 {
                let s = Instant::now();
                f();
                b = b.min(s.elapsed().as_secs_f64());
            }
            println!("softmax_1d f64 16M [{label}]: {:.3}ms", b * 1e3);
        };
        bench("sequential", &|| {
            let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_shifted: Vec<f64> = x.iter().map(|&v| (v - max_val).exp()).collect();
            let sum_exp: f64 = exp_shifted.iter().sum();
            std::hint::black_box(
                exp_shifted
                    .iter()
                    .map(|&e| e / sum_exp)
                    .collect::<Vec<f64>>(),
            );
        });
        bench("threaded", &|| {
            std::hint::black_box(softmax(&x));
        });
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_sigmoid_threaded_vs_sequential() {
        use std::time::Instant;
        let n = 16_000_000usize;
        let x: Vec<f64> = (0..n)
            .map(|i| ((i % 4001) as f64 - 2000.0) * 0.001)
            .collect();
        let bench = |label: &str, f: &dyn Fn()| {
            f();
            let mut b = f64::MAX;
            for _ in 0..5 {
                let s = Instant::now();
                f();
                b = b.min(s.elapsed().as_secs_f64());
            }
            println!("sigmoid f64 16M [{label}]: {:.3}ms", b * 1e3);
        };
        bench("sequential", &|| {
            let out: Vec<f64> = x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
            std::hint::black_box(out);
        });
        bench("threaded", &|| {
            std::hint::black_box(sigmoid(&x));
        });
    }

    #[test]
    fn threaded_cheap_activations_bit_identical_to_sequential() {
        // Cheap BW-bound activations only thread past the L3 gate (>= 1<<23), so test at a
        // size that engages the threaded path; must equal the sequential map bit-for-bit.
        let n = 9_000_000usize;
        let x: Vec<f64> = (0..n)
            .map(|i| ((i % 4001) as f64 - 2000.0) * 0.0017)
            .collect();
        let relu_seq: Vec<f64> = x.iter().map(|&v| v.max(0.0)).collect();
        let relu6_seq: Vec<f64> = x.iter().map(|&v| v.clamp(0.0, 6.0)).collect();
        let lrelu_seq: Vec<f64> = x
            .iter()
            .map(|&v| if v >= 0.0 { v } else { 0.01 * v })
            .collect();
        let hsig_seq: Vec<f64> = x
            .iter()
            .map(|&v| ((v + 3.0) / 6.0).clamp(0.0, 1.0))
            .collect();
        let htanh_seq: Vec<f64> = x.iter().map(|&v| v.clamp(-1.0, 1.0)).collect();
        let hsilu_seq: Vec<f64> = x
            .iter()
            .map(|&v| v * ((v + 3.0) / 6.0).clamp(0.0, 1.0))
            .collect();
        let ssign_seq: Vec<f64> = x.iter().map(|&v| v / (1.0 + v.abs())).collect();
        let check = |got: &[f64], want: &[f64], name: &str| {
            for (a, b) in got.iter().zip(want) {
                assert_eq!(a.to_bits(), b.to_bits(), "{name}");
            }
        };
        check(&relu(&x), &relu_seq, "relu");
        check(&relu6(&x), &relu6_seq, "relu6");
        check(&leaky_relu(&x, 0.01), &lrelu_seq, "leaky_relu");
        check(&hard_sigmoid(&x), &hsig_seq, "hard_sigmoid");
        check(&hard_tanh(&x), &htanh_seq, "hard_tanh");
        check(&hard_silu(&x), &hsilu_seq, "hard_silu");
        check(&softsign(&x), &ssign_seq, "softsign");
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_relu_threaded_vs_sequential() {
        use std::time::Instant;
        let n = 16_000_000usize;
        let x: Vec<f64> = (0..n)
            .map(|i| ((i % 4001) as f64 - 2000.0) * 0.001)
            .collect();
        let bench = |label: &str, f: &dyn Fn()| {
            f();
            let mut b = f64::MAX;
            for _ in 0..7 {
                let s = Instant::now();
                f();
                b = b.min(s.elapsed().as_secs_f64());
            }
            println!("relu f64 16M [{label}]: {:.3}ms", b * 1e3);
        };
        bench("sequential", &|| {
            let out: Vec<f64> = x.iter().map(|&v| v.max(0.0)).collect();
            std::hint::black_box(out);
        });
        bench("threaded", &|| {
            std::hint::black_box(relu(&x));
        });
    }

    #[test]
    fn logsumexp_log_softmax_threaded_bit_identical() {
        // At >= the threading gate, logsumexp/log_softmax take the threaded-exp path; they
        // must equal the fused single-pass reference bit-for-bit (same index-order sum).
        let n = 3_000_000usize;
        let x: Vec<f64> = (0..n)
            .map(|i| ((i % 9973) as f64 - 5000.0) * 0.0009)
            .collect();
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum_ref: f64 = x.iter().map(|&v| (v - max_val).exp()).sum();
        let lse_ref = max_val + sum_ref.ln();
        assert_eq!(logsumexp(&x).to_bits(), lse_ref.to_bits());
        let ls_ref: Vec<f64> = x.iter().map(|&v| v - lse_ref).collect();
        for (a, b) in log_softmax(&x).iter().zip(&ls_ref) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_logsumexp_threaded_vs_fused() {
        use std::time::Instant;
        let n = 16_000_000usize;
        let x: Vec<f64> = (0..n)
            .map(|i| ((i % 4001) as f64 - 2000.0) * 0.001)
            .collect();
        let bench = |label: &str, f: &dyn Fn()| {
            f();
            let mut b = f64::MAX;
            for _ in 0..5 {
                let s = Instant::now();
                f();
                b = b.min(s.elapsed().as_secs_f64());
            }
            println!("logsumexp f64 16M [{label}]: {:.3}ms", b * 1e3);
        };
        bench("fused_sequential", &|| {
            let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = x.iter().map(|&v| (v - max_val).exp()).sum();
            std::hint::black_box(max_val + sum_exp.ln());
        });
        bench("threaded", &|| {
            std::hint::black_box(logsumexp(&x));
        });
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_gelu_threaded_vs_sequential() {
        use std::time::Instant;
        let n = 16_000_000usize;
        let x: Vec<f64> = (0..n)
            .map(|i| ((i % 4001) as f64 - 2000.0) * 0.001)
            .collect();
        let sqrt_2_over_pi = (2.0 / PI).sqrt();
        let bench = |label: &str, f: &dyn Fn()| {
            f();
            let mut b = f64::MAX;
            for _ in 0..5 {
                let s = Instant::now();
                f();
                b = b.min(s.elapsed().as_secs_f64());
            }
            println!("gelu f64 16M [{label}]: {:.3}ms", b * 1e3);
        };
        bench("sequential", &|| {
            let out: Vec<f64> = x
                .iter()
                .map(|&v| {
                    let inner = sqrt_2_over_pi * (v + 0.044715 * v * v * v);
                    0.5 * v * (1.0 + inner.tanh())
                })
                .collect();
            std::hint::black_box(out);
        });
        bench("threaded", &|| {
            std::hint::black_box(gelu(&x));
        });
    }
}
