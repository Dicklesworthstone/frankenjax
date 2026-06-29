//! Einstein summation convention matching JAX's jnp.einsum.
//!
//! Supports subscript notation like "ij,jk->ik" for matrix multiplication.

use std::collections::{HashMap, HashSet};

/// Error type for einsum parsing and execution.
#[derive(Debug, Clone, PartialEq)]
pub enum EinsumError {
    InvalidSubscript(String),
    ShapeMismatch {
        index: char,
        expected: usize,
        got: usize,
    },
    MissingOperand {
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for EinsumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSubscript(s) => write!(f, "invalid einsum subscript: {s}"),
            Self::ShapeMismatch {
                index,
                expected,
                got,
            } => {
                write!(
                    f,
                    "dimension mismatch for index '{index}': expected {expected}, got {got}"
                )
            }
            Self::MissingOperand { expected, got } => {
                write!(f, "expected {expected} operands, got {got}")
            }
        }
    }
}

impl std::error::Error for EinsumError {}

/// Parse an einsum subscript string into input subscripts and output subscript.
///
/// Examples:
/// - "ij,jk->ik" -> (["ij", "jk"], "ik")
/// - "ij,jk" -> (["ij", "jk"], None) - implicit output
fn parse_subscripts(subscripts: &str) -> Result<(Vec<String>, Option<String>), EinsumError> {
    let subscripts = subscripts.replace(' ', "");

    let (inputs_str, output_str) = if let Some(idx) = subscripts.find("->") {
        (&subscripts[..idx], Some(&subscripts[idx + 2..]))
    } else {
        (subscripts.as_str(), None)
    };

    let input_subs: Vec<String> = inputs_str.split(',').map(|s| s.to_string()).collect();

    // Validate subscripts contain only lowercase letters
    for sub in &input_subs {
        if !sub.chars().all(|c| c.is_ascii_lowercase() || c == '.') {
            return Err(EinsumError::InvalidSubscript(format!(
                "subscript '{sub}' contains invalid characters"
            )));
        }
    }

    if let Some(out) = output_str
        && !out.chars().all(|c| c.is_ascii_lowercase() || c == '.')
    {
        return Err(EinsumError::InvalidSubscript(format!(
            "output subscript '{out}' contains invalid characters"
        )));
    }

    Ok((input_subs, output_str.map(|s| s.to_string())))
}

/// Compute implicit output subscript (indices that appear exactly once).
fn compute_implicit_output(input_subs: &[String]) -> String {
    let mut counts: HashMap<char, usize> = HashMap::new();
    for sub in input_subs {
        for c in sub.chars() {
            *counts.entry(c).or_insert(0) += 1;
        }
    }

    // Output indices are those that appear exactly once, in alphabetical order
    let mut output_chars: Vec<char> = counts
        .iter()
        .filter(|&(_, &count)| count == 1)
        .map(|(&c, _)| c)
        .collect();
    output_chars.sort();
    output_chars.into_iter().collect()
}

/// Execute einsum with two operands (most common case).
///
/// Matches `jnp.einsum(subscripts, a, b)`.
pub fn einsum2(
    subscripts: &str,
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
) -> Result<(Vec<f64>, Vec<usize>), EinsumError> {
    let (input_subs, output_sub) = parse_subscripts(subscripts)?;

    if input_subs.len() != 2 {
        return Err(EinsumError::MissingOperand {
            expected: 2,
            got: input_subs.len(),
        });
    }

    let sub_a = &input_subs[0];
    let sub_b = &input_subs[1];
    let sub_out = output_sub.unwrap_or_else(|| compute_implicit_output(&input_subs));

    // Validate shapes match subscripts
    if sub_a.len() != a_shape.len() {
        return Err(EinsumError::InvalidSubscript(format!(
            "subscript '{}' has {} indices but operand has {} dimensions",
            sub_a,
            sub_a.len(),
            a_shape.len()
        )));
    }
    if sub_b.len() != b_shape.len() {
        return Err(EinsumError::InvalidSubscript(format!(
            "subscript '{}' has {} indices but operand has {} dimensions",
            sub_b,
            sub_b.len(),
            b_shape.len()
        )));
    }

    // Build index->dimension mapping
    let mut index_dims: HashMap<char, usize> = HashMap::new();
    for (i, c) in sub_a.chars().enumerate() {
        index_dims.insert(c, a_shape[i]);
    }
    for (i, c) in sub_b.chars().enumerate() {
        if let Some(&existing) = index_dims.get(&c) {
            if existing != b_shape[i] {
                return Err(EinsumError::ShapeMismatch {
                    index: c,
                    expected: existing,
                    got: b_shape[i],
                });
            }
        } else {
            index_dims.insert(c, b_shape[i]);
        }
    }

    // Fast path: the standard (optionally batched) matrix product
    // "B…XY,B…YZ->B…XZ" (distinct X,Y,Z). The generic path below allocates a
    // HashMap and decodes coordinates *per (output, contracted) pair* — O(out·sum)
    // with per-iteration heap traffic — whereas each batch slice is exactly
    // `matmul_2d`. Both sum the contracted index Y in ascending order into
    // result[(batch)·M·N + X·N + Z], so the f64 output is bit-for-bit identical
    // (see einsum2_matmul_fast_path_bit_identical and
    // einsum2_batched_matmul_fast_path_bit_identical). Degenerate (zero-size)
    // dims fall through to the generic path, which is correct and cheap there.
    if let Some(fast) = try_einsum2_matmul(sub_a, sub_b, &sub_out, a, a_shape, b, b_shape) {
        return Ok(fast);
    }
    // Fast path: transposed-rhs matrix product "B…XY,B…ZY->B…XZ" (A·Bᵀ — attention
    // QKᵀ / linear-layer x·Wᵀ). Same BLAS-class routing as above after an exact
    // per-slice [n,k]->[k,n] transpose; bit-identical to the generic odometer.
    if let Some(fast) = try_einsum2_matmul_bt(sub_a, sub_b, &sub_out, a, a_shape, b, b_shape) {
        return Ok(fast);
    }
    // Fast path: lhs-transposed matrix products "B…YX,B…YZ->B…XZ" (Aᵀ·B, backprop
    // weight gradient) and "B…YX,B…ZY->B…XZ" (Aᵀ·Bᵀ). Transpose the lhs (and rhs
    // when needed) then route to matmul_2d; bit-identical to the generic odometer.
    if let Some(fast) = try_einsum2_matmul_at(sub_a, sub_b, &sub_out, a, a_shape, b, b_shape) {
        return Ok(fast);
    }
    // Fast path: pre-aligned GROUPED contraction "FA…K…,K…FB…->FA…FB…" — a matrix
    // product whose free (M) / contracted (K) / free (N) roles are each a group of
    // already-aligned axes (e.g. "abcd,cdef->abef"). A is [M,K] and B is [K,N]
    // contiguous, so reshape is a no-op; route to matmul_2d.
    if let Some(fast) = try_einsum2_matmul_grouped(sub_a, sub_b, &sub_out, a, a_shape, b, b_shape) {
        return Ok(fast);
    }
    // General fast path: any remaining distinct-label contraction (interleaved
    // batch/free/contracted axes — e.g. attention `bqhd,bkhd->bhqk`, or tensordot
    // over non-adjacent axes) is permuted into `[batch, M, K]`/`[batch, K, N]` and
    // run as one cache-blocked GEMM per batch slice, replacing the O(out·sum)
    // strided odometer below. Returns `None` (→ odometer) for diagonal/trace
    // (repeated labels) or single-operand reductions.
    if let Some(fast) = try_einsum2_matmul_general(sub_a, sub_b, &sub_out, a, a_shape, b, b_shape) {
        return Ok(fast);
    }

    // Compute output shape
    let out_shape: Vec<usize> = sub_out.chars().map(|c| index_dims[&c]).collect();

    // Find summed (contracted) indices
    let all_indices: HashSet<char> = sub_a.chars().chain(sub_b.chars()).collect();
    let out_indices: HashSet<char> = sub_out.chars().collect();
    let sum_indices: Vec<char> = all_indices.difference(&out_indices).copied().collect();

    // Compute sum dimensions
    let sum_dims: Vec<usize> = sum_indices.iter().map(|&c| index_dims[&c]).collect();
    let sum_size: usize = sum_dims.iter().product();

    // Output size
    let out_size: usize = out_shape.iter().product();
    if out_size == 0 {
        return Ok((vec![], out_shape));
    }

    let mut result = vec![0.0; out_size.max(1)];

    // Allocation-free generic contraction. The previous interpreter rebuilt a
    // `HashMap<char, usize>` and re-decoded both operand subscripts *per
    // (output, contracted) pair* — O(out·sum) heap traffic. Instead precompute,
    // for each operand, a per-axis (stride, coordinate-source) plan plus a
    // per-summed-axis stride contribution, then walk the output once and the
    // contracted axes via a row-major odometer that maintains the operand flat
    // indices incrementally. The summation visits the contracted multi-index in
    // the exact same row-major ascending order as before, so every output's f64
    // accumulation is bit-for-bit identical (see
    // einsum2_generic_allocation_free_bit_identical).
    let out_strides = row_major_strides(&out_shape);
    let a_strides = row_major_strides(a_shape);
    let b_strides = row_major_strides(b_shape);

    // label -> position in the output coord list / summed-index list.
    let out_pos: HashMap<char, usize> = sub_out.chars().enumerate().map(|(i, c)| (c, i)).collect();
    let sum_pos: HashMap<char, usize> = sum_indices
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();

    // Per-axis plan entries: (stride, is_summed, pos) where `pos` indexes
    // out_coords for free axes or the summed-axis list for contracted axes.
    let build_plan = |sub: &str, strides: &[usize]| -> Vec<(usize, bool, usize)> {
        sub.chars()
            .enumerate()
            .map(|(i, c)| {
                if let Some(&p) = out_pos.get(&c) {
                    (strides[i], false, p)
                } else {
                    (strides[i], true, sum_pos[&c])
                }
            })
            .collect()
    };
    let a_plan = build_plan(sub_a, &a_strides);
    let b_plan = build_plan(sub_b, &b_strides);

    let num_sum = sum_dims.len();
    let mut a_sum_stride = vec![0usize; num_sum];
    let mut b_sum_stride = vec![0usize; num_sum];
    for &(stride, is_sum, pos) in &a_plan {
        if is_sum {
            a_sum_stride[pos] += stride;
        }
    }
    for &(stride, is_sum, pos) in &b_plan {
        if is_sum {
            b_sum_stride[pos] += stride;
        }
    }

    let mut out_coords = vec![0usize; out_shape.len()];
    let mut sum_coords = vec![0usize; num_sum];
    // `out_size > 0` here (the empty case returned early), so every out_shape
    // extent and every row-major stride is >= 1 — the decode divisions are safe.
    for (out_idx, slot) in result.iter_mut().enumerate() {
        // Decode the output multi-index (row-major) into the reusable buffer.
        for ax in 0..out_shape.len() {
            out_coords[ax] = (out_idx / out_strides[ax]) % out_shape[ax];
        }
        // Operand base indices from the free (output-driven) axes.
        let mut a_base = 0usize;
        for &(stride, is_sum, pos) in &a_plan {
            if !is_sum {
                a_base += stride * out_coords[pos];
            }
        }
        let mut b_base = 0usize;
        for &(stride, is_sum, pos) in &b_plan {
            if !is_sum {
                b_base += stride * out_coords[pos];
            }
        }

        // Sum over the contracted axes via a row-major odometer, maintaining
        // a_idx / b_idx incrementally (ascending order, matching the prior loop).
        for c in sum_coords.iter_mut() {
            *c = 0;
        }
        let mut a_idx = a_base;
        let mut b_idx = b_base;
        let mut sum = 0.0;
        for _ in 0..sum_size.max(1) {
            if a_idx < a.len() && b_idx < b.len() {
                sum += a[a_idx] * b[b_idx];
            }
            for j in (0..num_sum).rev() {
                sum_coords[j] += 1;
                a_idx += a_sum_stride[j];
                b_idx += b_sum_stride[j];
                if sum_coords[j] < sum_dims[j] {
                    break;
                }
                sum_coords[j] = 0;
                a_idx -= a_sum_stride[j] * sum_dims[j];
                b_idx -= b_sum_stride[j] * sum_dims[j];
            }
        }
        *slot = sum;
    }

    Ok((result, out_shape))
}

/// Row-major (C-order) strides for `shape`: `strides[i] = prod(shape[i+1..])`.
/// Matches the stride convention used throughout einsum index math.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    let mut strides = vec![1usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Execute einsum with a single operand (trace, diagonal, transpose, etc.).
pub fn einsum1(
    subscripts: &str,
    a: &[f64],
    a_shape: &[usize],
) -> Result<(Vec<f64>, Vec<usize>), EinsumError> {
    let (input_subs, output_sub) = parse_subscripts(subscripts)?;

    if input_subs.len() != 1 {
        return Err(EinsumError::MissingOperand {
            expected: 1,
            got: input_subs.len(),
        });
    }

    let sub_a = &input_subs[0];
    let sub_out = output_sub.unwrap_or_else(|| {
        // For single operand, implicit output keeps indices appearing once
        let mut counts: HashMap<char, usize> = HashMap::new();
        for c in sub_a.chars() {
            *counts.entry(c).or_insert(0) += 1;
        }
        let mut output_chars: Vec<char> = counts
            .iter()
            .filter(|&(_, &count)| count == 1)
            .map(|(&c, _)| c)
            .collect();
        output_chars.sort();
        output_chars.into_iter().collect()
    });

    if sub_a.len() != a_shape.len() {
        return Err(EinsumError::InvalidSubscript(format!(
            "subscript '{}' has {} indices but operand has {} dimensions",
            sub_a,
            sub_a.len(),
            a_shape.len()
        )));
    }

    // Build index->dimension mapping
    let mut index_dims: HashMap<char, usize> = HashMap::new();
    for (i, c) in sub_a.chars().enumerate() {
        if let Some(&existing) = index_dims.get(&c) {
            if existing != a_shape[i] {
                return Err(EinsumError::ShapeMismatch {
                    index: c,
                    expected: existing,
                    got: a_shape[i],
                });
            }
        } else {
            index_dims.insert(c, a_shape[i]);
        }
    }

    // Compute output shape
    let out_shape: Vec<usize> = sub_out.chars().map(|c| index_dims[&c]).collect();

    // Find summed indices (those not in output)
    let out_indices: HashSet<char> = sub_out.chars().collect();
    let sum_indices: Vec<char> = sub_a.chars().filter(|c| !out_indices.contains(c)).collect();
    let unique_sum: HashSet<char> = sum_indices.iter().copied().collect();
    let sum_dims: Vec<usize> = unique_sum.iter().map(|&c| index_dims[&c]).collect();
    let sum_size: usize = sum_dims.iter().product();

    let out_size: usize = out_shape.iter().product();
    if out_size == 0 {
        return Ok((vec![], out_shape));
    }

    let mut result = vec![0.0; out_size.max(1)];

    // Allocation-free single-operand contraction (same technique as einsum2's
    // generic loop): precompute a per-axis (stride, coordinate-source) plan plus
    // a per-summed-axis stride contribution, then walk the output once and the
    // summed axes via a row-major odometer that maintains the operand flat index
    // incrementally. Eliminates the per-(output, summed)-pair `HashMap` rebuild +
    // subscript re-decode. Repeated labels (trace "ii->", diagonal "ii->i") fold
    // into summed strides — `subscript_to_flat_idx` summed each occurrence's
    // stride×coord, so do we; for a single contracted axis the row-major
    // ascending order is unambiguous, hence bit-for-bit identical (see
    // einsum1_allocation_free_bit_identical).
    let out_strides = row_major_strides(&out_shape);
    let a_strides = row_major_strides(a_shape);
    let out_pos: HashMap<char, usize> = sub_out.chars().enumerate().map(|(i, c)| (c, i)).collect();
    // Unique summed labels + their extents, captured from one iteration so the
    // odometer axes and dims stay aligned.
    let sum_labels: Vec<char> = unique_sum.iter().copied().collect();
    let sum_extent: Vec<usize> = sum_labels.iter().map(|&c| index_dims[&c]).collect();
    let sum_pos: HashMap<char, usize> = sum_labels
        .iter()
        .enumerate()
        .map(|(j, &c)| (c, j))
        .collect();

    // Per-axis plan: (stride, is_summed, pos) where pos indexes out_coords (free)
    // or the summed-axis list (summed).
    let a_plan: Vec<(usize, bool, usize)> = sub_a
        .chars()
        .enumerate()
        .map(|(i, c)| {
            if let Some(&p) = out_pos.get(&c) {
                (a_strides[i], false, p)
            } else {
                (a_strides[i], true, sum_pos[&c])
            }
        })
        .collect();
    let num_sum = sum_extent.len();
    let mut a_sum_stride = vec![0usize; num_sum];
    for &(stride, is_sum, pos) in &a_plan {
        if is_sum {
            a_sum_stride[pos] += stride;
        }
    }

    let mut out_coords = vec![0usize; out_shape.len()];
    let mut sum_coords = vec![0usize; num_sum];
    // out_size > 0 here, so every out extent / stride is >= 1 (decode is safe).
    for (out_idx, slot) in result.iter_mut().enumerate() {
        for ax in 0..out_shape.len() {
            out_coords[ax] = (out_idx / out_strides[ax]) % out_shape[ax];
        }
        let mut a_base = 0usize;
        for &(stride, is_sum, pos) in &a_plan {
            if !is_sum {
                a_base += stride * out_coords[pos];
            }
        }
        for c in sum_coords.iter_mut() {
            *c = 0;
        }
        let mut a_idx = a_base;
        let mut sum = 0.0;
        for _ in 0..sum_size.max(1) {
            if a_idx < a.len() {
                sum += a[a_idx];
            }
            for j in (0..num_sum).rev() {
                sum_coords[j] += 1;
                a_idx += a_sum_stride[j];
                if sum_coords[j] < sum_extent[j] {
                    break;
                }
                sum_coords[j] = 0;
                a_idx -= a_sum_stride[j] * sum_extent[j];
            }
        }
        *slot = sum;
    }

    Ok((result, out_shape))
}

/// Detect the standard (optionally **batched**) single-contraction matrix
/// product "B…XY,B…YZ->B…XZ" — an identical leading batch-label prefix `B…`
/// (possibly empty) shared by both operands and the output, followed by the
/// matmul triple `X` (lhs free), `Y` (contracted), `Z` (rhs free) with X, Y, Z
/// three distinct labels not appearing in the batch prefix — and evaluate each
/// batch slice via the contiguous `matmul_2d` kernel instead of the generic
/// per-(output, contracted)-pair loop (which allocates a `HashMap` per
/// iteration).
///
/// Batched single-contraction matmul over `batch_size` independent `[m,k]·[k,n]` slices, writing
/// the `[batch, m, n]` row-major result. Each slice is dispatched to the cache-blocked
/// [`matmul_2d_into`]; when `transpose_rhs` is set the rhs slice is stored `[n,k]` (the `A·Bᵀ`
/// attention-score layout) and is transposed to `[k,n]` per slice first.
///
/// The slices are INDEPENDENT, so for the attention regime — many small cache-resident slices, each
/// below `matmul_2d`'s own ~8M-op fanout floor (so it runs single-threaded) — the batch loop is
/// fanned across scoped threads, leaving no cores idle. Gated so it only threads when each slice is
/// below that floor (no oversubscription with `matmul_2d`'s internal threading) yet the whole batch
/// is substantial. Bit-for-bit identical to the serial loop: each slice is computed by the same
/// kernel and written to the same disjoint output range, regardless of which thread runs it.
fn batched_matmul_slices_into(
    a: &[f64],
    b: &[f64],
    batch_size: usize,
    m: usize,
    k: usize,
    n: usize,
    lhs_stride: usize,
    rhs_stride: usize,
    out_stride: usize,
    transpose_rhs: bool,
) -> Vec<f64> {
    let mut out = vec![0.0f64; batch_size * out_stride];
    let per_slice_ops = m.saturating_mul(k).saturating_mul(n);
    let total_ops = batch_size.saturating_mul(per_slice_ops);
    // matmul_2d self-threads at >= 1<<23 ops; below that a slice is serial, so batch-level
    // threading is the only parallelism and won't oversubscribe.
    const SLICE_SELF_THREADS_OPS: usize = 1 << 23;
    const BATCH_PAR_MIN_OPS: usize = 1 << 21;
    let threads = if batch_size >= 2
        && per_slice_ops < SLICE_SELF_THREADS_OPS
        && total_ops >= BATCH_PAR_MIN_OPS
    {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
            .min(batch_size)
    } else {
        1
    };
    let compute = |bt: usize, dst: &mut [f64], buf: &mut Vec<f64>| {
        let a_slice = &a[bt * lhs_stride..bt * lhs_stride + lhs_stride];
        let b_slice = &b[bt * rhs_stride..bt * rhs_stride + rhs_stride];
        if transpose_rhs {
            // b_slice is [n, k]; transpose to [k, n] (ascending, exact data movement).
            buf.clear();
            buf.resize(k * n, 0.0);
            for r in 0..n {
                let row = &b_slice[r * k..r * k + k];
                for (c, &v) in row.iter().enumerate() {
                    buf[c * n + r] = v;
                }
            }
            crate::tensor_contraction::matmul_2d_into(a_slice, m, k, buf, n, dst);
        } else {
            crate::tensor_contraction::matmul_2d_into(a_slice, m, k, b_slice, n, dst);
        }
    };
    if threads <= 1 {
        let mut buf = Vec::new();
        for bt in 0..batch_size {
            let s = bt * out_stride;
            compute(bt, &mut out[s..s + out_stride], &mut buf);
        }
        return out;
    }
    let per = batch_size.div_ceil(threads);
    let compute_ref = &compute;
    std::thread::scope(|scope| {
        let mut rest: &mut [f64] = out.as_mut_slice();
        let mut bt0 = 0usize;
        while bt0 < batch_size {
            let cnt = per.min(batch_size - bt0);
            let (blk, tail) = rest.split_at_mut(cnt * out_stride);
            rest = tail;
            let start = bt0;
            bt0 += cnt;
            scope.spawn(move || {
                let mut buf = Vec::new();
                for j in 0..cnt {
                    compute_ref(
                        start + j,
                        &mut blk[j * out_stride..(j + 1) * out_stride],
                        &mut buf,
                    );
                }
            });
        }
    });
    out
}

/// Returns `None` for any other pattern (repeated/diagonal/transposed indices,
/// non-leading or repeated batch labels, a zero-size extent), leaving the
/// generic path to handle it. Each batch slice (`M·K` / `K·N`) is contiguous in
/// row-major layout, and `matmul_2d` accumulates `Y` in ascending order into
/// `result[(batch)·M·N + X·N + Z]` — identical summation order and batch-major
/// output layout to the generic loop — so the result is bit-for-bit identical.
fn try_einsum2_matmul(
    sub_a: &str,
    sub_b: &str,
    sub_out: &str,
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
) -> Option<(Vec<f64>, Vec<usize>)> {
    let sa: Vec<char> = sub_a.chars().collect();
    let sb: Vec<char> = sub_b.chars().collect();
    let so: Vec<char> = sub_out.chars().collect();
    let len = sa.len();
    // All three subscripts share the same rank = nb (batch) + 2 (matmul axes).
    if len < 2 || sb.len() != len || so.len() != len {
        return None;
    }
    let nb = len - 2;

    // Leading `nb` labels must be an identical batch prefix across a, b, out,
    // and those labels must be distinct (a repeated batch label would not map to
    // a simple contiguous batch stride).
    for i in 0..nb {
        if sa[i] != sb[i] || sa[i] != so[i] {
            return None;
        }
        for j in (i + 1)..nb {
            if sa[i] == sa[j] {
                return None;
            }
        }
    }

    let x = sa[nb];
    let y = sa[nb + 1];
    let (y2, z) = (sb[nb], sb[nb + 1]);
    // "…XY,…YZ->…XZ": trailing lhs index is the contracted one, shared with the
    // leading rhs matmul index; output's trailing two are the free indices.
    if y != y2 || so[nb] != x || so[nb + 1] != z {
        return None;
    }
    // X, Y, Z distinct, and none reused as a batch label.
    if x == y || y == z || x == z {
        return None;
    }
    for &c in &sa[..nb] {
        if c == x || c == y || c == z {
            return None;
        }
    }

    let m = a_shape[nb];
    let k = a_shape[nb + 1];
    let n = b_shape[nb + 1];
    if b_shape[nb] != k || m == 0 || k == 0 || n == 0 {
        return None;
    }
    // Batch extents must agree between operands; build the batch size + output shape.
    let mut batch_size = 1usize;
    for i in 0..nb {
        if a_shape[i] != b_shape[i] || a_shape[i] == 0 {
            return None;
        }
        batch_size = batch_size.checked_mul(a_shape[i])?;
    }
    let lhs_stride = m.checked_mul(k)?;
    let rhs_stride = k.checked_mul(n)?;
    let out_stride = m.checked_mul(n)?;
    // Defensive bounds check (the caller validated dims vs subscripts already).
    if a.len() < batch_size.checked_mul(lhs_stride)?
        || b.len() < batch_size.checked_mul(rhs_stride)?
    {
        return None;
    }

    // Defensive: out_stride product must not overflow before the helper allocs.
    batch_size.checked_mul(out_stride)?;
    let out = batched_matmul_slices_into(
        a, b, batch_size, m, k, n, lhs_stride, rhs_stride, out_stride, false,
    );

    let mut out_shape: Vec<usize> = a_shape[..nb].to_vec();
    out_shape.push(m);
    out_shape.push(n);
    Some((out, out_shape))
}

/// Detect the **transposed-rhs** single-contraction matrix product
/// "B…XY,B…ZY->B…XZ" — i.e. `A · Bᵀ`, the dominant attention-score (`QKᵀ`) and
/// linear-layer-weight (`x·Wᵀ`) layout — where the contracted label `Y` is the
/// **trailing** axis of *both* operands (so the rhs is stored `[…, Z, Y]`).
/// This pattern is NOT caught by [`try_einsum2_matmul`] (which needs the rhs as
/// `[…, Y, Z]`) and otherwise falls to the generic O(out·sum) odometer. Each rhs
/// batch slice `[n, k]` is transposed to `[k, n]` — O(n·k), cheap against the
/// O(m·n·k) product — then routed through the cache-blocked [`matmul_2d`] kernel.
///
/// Bit-for-bit identical to the generic path: the transpose is exact data
/// movement, and `matmul_2d` accumulates the contracted index `Y` in ascending
/// order into `result[(batch)·M·N + X·N + Z]` — the same order and output layout
/// the generic odometer uses (verified by `matmul_2d`'s own ascending-sum
/// bit-identity, see [`try_einsum2_matmul`]). Returns `None` for any other
/// pattern, leaving the generic path to handle it.
fn try_einsum2_matmul_bt(
    sub_a: &str,
    sub_b: &str,
    sub_out: &str,
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
) -> Option<(Vec<f64>, Vec<usize>)> {
    let sa: Vec<char> = sub_a.chars().collect();
    let sb: Vec<char> = sub_b.chars().collect();
    let so: Vec<char> = sub_out.chars().collect();
    let len = sa.len();
    if len < 2 || sb.len() != len || so.len() != len {
        return None;
    }
    let nb = len - 2;

    // Identical, distinct leading batch prefix across a, b, out.
    for i in 0..nb {
        if sa[i] != sb[i] || sa[i] != so[i] {
            return None;
        }
        for j in (i + 1)..nb {
            if sa[i] == sa[j] {
                return None;
            }
        }
    }

    let x = sa[nb];
    let y = sa[nb + 1];
    // Transposed rhs "…ZY": Z leads, the contracted Y trails (vs "…YZ" canonical).
    let (z, y2) = (sb[nb], sb[nb + 1]);
    if y != y2 || so[nb] != x || so[nb + 1] != z {
        return None;
    }
    if x == y || y == z || x == z {
        return None;
    }
    for &c in &sa[..nb] {
        if c == x || c == y || c == z {
            return None;
        }
    }

    let m = a_shape[nb];
    let k = a_shape[nb + 1];
    let n = b_shape[nb]; // rhs slice is [n, k]
    if b_shape[nb + 1] != k || m == 0 || k == 0 || n == 0 {
        return None;
    }
    let mut batch_size = 1usize;
    for i in 0..nb {
        if a_shape[i] != b_shape[i] || a_shape[i] == 0 {
            return None;
        }
        batch_size = batch_size.checked_mul(a_shape[i])?;
    }
    let lhs_stride = m.checked_mul(k)?;
    let rhs_stride = n.checked_mul(k)?;
    let out_stride = m.checked_mul(n)?;
    if a.len() < batch_size.checked_mul(lhs_stride)?
        || b.len() < batch_size.checked_mul(rhs_stride)?
    {
        return None;
    }

    batch_size.checked_mul(out_stride)?;
    // rhs slice stored [n, k] (transposed); the helper transposes each to [k, n) per slice
    // and threads independent slices for the attention-score regime.
    let out = batched_matmul_slices_into(
        a, b, batch_size, m, k, n, lhs_stride, rhs_stride, out_stride, true,
    );

    let mut out_shape: Vec<usize> = a_shape[..nb].to_vec();
    out_shape.push(m);
    out_shape.push(n);
    Some((out, out_shape))
}

/// Detect the **lhs-transposed** single-contraction matrix products
/// "B…YX,B…YZ->B…XZ" (`Aᵀ·B`, the backprop `Xᵀ·dY` weight-gradient layout) and
/// "B…YX,B…ZY->B…XZ" (`Aᵀ·Bᵀ`) — where the contracted label `Y` is the **leading**
/// matmul axis of the lhs (so `A` is stored `[…, Y, X]`). The lhs contracted axis
/// is column-strided in this layout, so the generic odometer's inner contraction
/// is cache-unfriendly; transposing each lhs slice `[k, m] -> [m, k]` (and the rhs
/// `[n, k] -> [k, n]` for the `Bᵀ` case) — O(m·k)/O(n·k), cheap vs the O(m·n·k)
/// product — then routing through the cache-blocked [`matmul_2d`] is a clear win.
///
/// Bit-for-bit identical to the generic path: the transposes are exact data
/// movement and `matmul_2d` accumulates `Y` in ascending order, the same order
/// the odometer uses. Returns `None` for any other pattern.
fn try_einsum2_matmul_at(
    sub_a: &str,
    sub_b: &str,
    sub_out: &str,
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
) -> Option<(Vec<f64>, Vec<usize>)> {
    let sa: Vec<char> = sub_a.chars().collect();
    let sb: Vec<char> = sub_b.chars().collect();
    let so: Vec<char> = sub_out.chars().collect();
    let len = sa.len();
    if len < 2 || sb.len() != len || so.len() != len {
        return None;
    }
    let nb = len - 2;
    for i in 0..nb {
        if sa[i] != sb[i] || sa[i] != so[i] {
            return None;
        }
        for j in (i + 1)..nb {
            if sa[i] == sa[j] {
                return None;
            }
        }
    }

    // lhs is "…YX": leading label Y is contracted, trailing X is free.
    let y = sa[nb];
    let x = sa[nb + 1];
    // rhs holds Y either leading ("…YZ", canonical B) or trailing ("…ZY", Bᵀ).
    let (z, b_transposed) = if sb[nb] == y {
        (sb[nb + 1], false)
    } else if sb[nb + 1] == y {
        (sb[nb], true)
    } else {
        return None;
    };
    if so[nb] != x || so[nb + 1] != z {
        return None;
    }
    if x == y || y == z || x == z {
        return None;
    }
    for &c in &sa[..nb] {
        if c == x || c == y || c == z {
            return None;
        }
    }

    // A is [k, m] (Y=k leading, X=m trailing).
    let k = a_shape[nb];
    let m = a_shape[nb + 1];
    let n = if b_transposed {
        b_shape[nb]
    } else {
        b_shape[nb + 1]
    };
    let rhs_k = if b_transposed {
        b_shape[nb + 1]
    } else {
        b_shape[nb]
    };
    if rhs_k != k || m == 0 || k == 0 || n == 0 {
        return None;
    }
    let mut batch_size = 1usize;
    for i in 0..nb {
        if a_shape[i] != b_shape[i] || a_shape[i] == 0 {
            return None;
        }
        batch_size = batch_size.checked_mul(a_shape[i])?;
    }
    let lhs_stride = k.checked_mul(m)?;
    let rhs_stride = n.checked_mul(k)?;
    let out_stride = m.checked_mul(n)?;
    if a.len() < batch_size.checked_mul(lhs_stride)?
        || b.len() < batch_size.checked_mul(rhs_stride)?
    {
        return None;
    }

    let mut out = Vec::with_capacity(batch_size.checked_mul(out_stride)?);
    let mut at_buf = vec![0.0f64; lhs_stride]; // [m, k] transposed lhs scratch
    let mut bt_buf = vec![0.0f64; rhs_stride]; // [k, n] transposed rhs scratch (Bᵀ case)
    for bt in 0..batch_size {
        let a_slice = &a[bt * lhs_stride..bt * lhs_stride + lhs_stride]; // [k, m]
        for r in 0..k {
            let row = &a_slice[r * m..r * m + m];
            for (c, &v) in row.iter().enumerate() {
                at_buf[c * k + r] = v; // [m, k]
            }
        }
        let b_slice = &b[bt * rhs_stride..bt * rhs_stride + rhs_stride];
        let b_kn: &[f64] = if b_transposed {
            // rhs is [n, k] -> transpose to [k, n].
            for r in 0..n {
                let row = &b_slice[r * k..r * k + k];
                for (c, &v) in row.iter().enumerate() {
                    bt_buf[c * n + r] = v;
                }
            }
            &bt_buf
        } else {
            // rhs already [k, n].
            b_slice
        };
        out.extend_from_slice(&crate::tensor_contraction::matmul_2d(
            &at_buf, m, k, b_kn, n,
        ));
    }

    let mut out_shape: Vec<usize> = a_shape[..nb].to_vec();
    out_shape.push(m);
    out_shape.push(n);
    Some((out, out_shape))
}

/// Detect a pre-aligned **grouped** matrix product "[FA][K],[K][FB]->[FA][FB]"
/// where the free-lhs (M), contracted (K), and free-rhs (N) roles are each a
/// CONTIGUOUS, already-aligned group of labels — e.g. "abcd,cdef->abef" (contract
/// c,d) or "abc,cde->abde" (grouped free axes, single contraction). Because A is
/// laid out `[FA…, K…]` and B `[K…, FB…]` in row-major, the [M,K]/[K,N] reshape is
/// a no-op view, so each is routed straight through the cache-blocked `matmul_2d`
/// (B is otherwise strided in the generic odometer's inner contraction).
///
/// Gated to a non-empty free-lhs AND non-empty free-rhs, so it only fires for a
/// genuine M×N matrix product (never matrix-vector or Frobenius, whose tested
/// behavior is left untouched); the plain single-axis i/j/k products are already
/// claimed by the earlier fast paths. Bit-for-bit identical to a textbook
/// ascending-K reference (matmul_2d's accumulation order); for ≥2 contracted axes
/// that order is one valid ordering of the generic path's nondeterministic
/// `sum_indices` (HashSet) order — i.e. tolerance-equal, not necessarily
/// bit-equal, to the generic loop (cf. the Frobenius test). Returns `None`
/// otherwise.
fn try_einsum2_matmul_grouped(
    sub_a: &str,
    sub_b: &str,
    sub_out: &str,
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
) -> Option<(Vec<f64>, Vec<usize>)> {
    let sa: Vec<char> = sub_a.chars().collect();
    let sb: Vec<char> = sub_b.chars().collect();
    let so: Vec<char> = sub_out.chars().collect();

    // No repeated labels within any subscript (excludes diagonal/trace patterns).
    let distinct = |v: &[char]| {
        for i in 0..v.len() {
            for j in (i + 1)..v.len() {
                if v[i] == v[j] {
                    return false;
                }
            }
        }
        true
    };
    if !distinct(&sa) || !distinct(&sb) || !distinct(&so) {
        return None;
    }

    let in_a = |c: char| sa.contains(&c);
    let in_b = |c: char| sb.contains(&c);
    let in_o = |c: char| so.contains(&c);

    // Every A label is exactly free (in output) XOR contracted (in B, not output).
    for &c in &sa {
        if in_o(c) == in_b(c) {
            return None;
        }
    }
    // Every B label is exactly free (in output) XOR contracted (in A).
    for &c in &sb {
        if in_o(c) == in_a(c) {
            return None;
        }
    }
    // Every output label comes from exactly one operand (and is not contracted).
    for &c in &so {
        if in_a(c) == in_b(c) {
            return None;
        }
    }

    // A must be [free_A…, contracted…]; contracted labels are A's labels that are in B.
    let n_k = sa.iter().filter(|&&c| in_b(c)).count();
    if n_k == 0 {
        return None;
    }
    let split_a = sa.len() - n_k; // free_A count
    let free_b_len = sb.len() - n_k;
    // Require a genuine M×N product: non-empty free-lhs and free-rhs.
    if split_a == 0 || free_b_len == 0 {
        return None;
    }
    for (i, &c) in sa.iter().enumerate() {
        if (i >= split_a) != in_b(c) {
            return None; // free_A prefix, contracted suffix
        }
    }
    // B must be [contracted…, free_B…] with contracted in the SAME order as A's suffix.
    for i in 0..n_k {
        if sb[i] != sa[split_a + i] {
            return None;
        }
    }
    for &c in &sb[n_k..] {
        if in_a(c) {
            return None; // free_B must not appear in A
        }
    }
    // Output must be exactly [free_A…, free_B…] in order.
    if so.len() != split_a + free_b_len {
        return None;
    }
    for i in 0..split_a {
        if so[i] != sa[i] {
            return None;
        }
    }
    for i in 0..free_b_len {
        if so[split_a + i] != sb[n_k + i] {
            return None;
        }
    }

    // Contracted extents must agree per-axis (A suffix vs B prefix).
    for i in 0..n_k {
        if a_shape[split_a + i] != b_shape[i] {
            return None;
        }
    }
    let m: usize = a_shape[..split_a].iter().product();
    let k: usize = a_shape[split_a..].iter().product();
    let n: usize = b_shape[n_k..].iter().product();
    if m == 0 || k == 0 || n == 0 {
        return None;
    }
    if a.len() < m.checked_mul(k)? || b.len() < k.checked_mul(n)? {
        return None;
    }

    // A is contiguous [M, K], B is contiguous [K, N] — feed matmul_2d directly.
    let out = crate::tensor_contraction::matmul_2d(a, m, k, b, n);
    let mut out_shape: Vec<usize> = a_shape[..split_a].to_vec();
    out_shape.extend_from_slice(&b_shape[n_k..]);
    Some((out, out_shape))
}

/// Materialize a contiguous row-major copy of `src` (shape `src_shape`) whose axis
/// `i` is `src`'s axis `perm[i]` (output shape = `src_shape` gathered by `perm`).
/// Used to bring an einsum operand into the canonical `[batch…, free…, contract…]`
/// layout the batched [`matmul_2d`] kernel expects.
fn permute_copy_f64(src: &[f64], src_shape: &[usize], perm: &[usize]) -> Vec<f64> {
    let rank = src_shape.len();
    let src_strides = row_major_strides(src_shape);
    let out_shape: Vec<usize> = perm.iter().map(|&p| src_shape[p]).collect();
    let perm_strides: Vec<usize> = perm.iter().map(|&p| src_strides[p]).collect();
    let total: usize = out_shape.iter().product();

    // Trailing run of output axes that is contiguous in src (perm fixes an
    // identity-mapped suffix — perm_strides match the inner output row-major
    // stride). That run copies verbatim as src[base..base+block_len]; replicate via
    // extend_from_slice and decode only the outer axes. Bit-identical; strict
    // generalization (block_len==1 when the innermost axis moves = per-element walk).
    let mut block_len = 1usize;
    let mut block_start = rank;
    let mut expected = 1usize;
    for ax in (0..rank).rev() {
        if perm_strides[ax] != expected {
            break;
        }
        block_len *= out_shape[ax];
        expected *= out_shape[ax];
        block_start = ax;
    }
    let block_len = block_len.max(1);

    let mut out = Vec::with_capacity(total);
    let outer_total = total / block_len;
    let mut coord = vec![0usize; block_start];
    let mut idx = 0usize;
    for _ in 0..outer_total {
        out.extend_from_slice(&src[idx..idx + block_len]);
        for ax in (0..block_start).rev() {
            coord[ax] += 1;
            idx += perm_strides[ax];
            if coord[ax] < out_shape[ax] {
                break;
            }
            coord[ax] = 0;
            idx -= perm_strides[ax] * out_shape[ax];
        }
    }
    out
}

/// General two-operand einsum → permute + batched cache-blocked GEMM.
///
/// Subsumes every distinct-label (no diagonal/trace) two-operand contraction the
/// four pattern-specific fast paths above DON'T already catch — in particular
/// contractions whose batch / free / contracted axes are INTERLEAVED and so need a
/// physical permute before they become a matrix product (e.g. `bqhd,bkhd->bhqk`,
/// attention scores with the head axis in the middle; or any `tensordot` over
/// non-adjacent axes). The generic fallback handles these with an O(out·sum)
/// strided per-element odometer whose contracted-multi-index summation order is
/// derived from a `HashSet` (non-deterministic across runs); this path instead
/// permutes each operand into `[batch…, free…, K…]` / `[batch…, K…, free…]`,
/// flattens the groups to `[bsz, M, K]` / `[bsz, K, N]`, and runs one
/// [`matmul_2d`] per batch slice.
///
/// PARITY: the contracted axes are flattened in a fixed row-major order (K-labels
/// in their order of appearance in A), and `matmul_2d` accumulates that flattened
/// `K` in ascending order — so the result is bit-for-bit identical to a
/// deterministic ascending-K naive contraction (see
/// `einsum2_general_matmul_bit_identical_to_naive`), and within f64 tolerance of
/// the (order-nondeterministic) generic odometer it replaces.
#[allow(clippy::too_many_arguments)]
fn try_einsum2_matmul_general(
    sub_a: &str,
    sub_b: &str,
    sub_out: &str,
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
) -> Option<(Vec<f64>, Vec<usize>)> {
    let sa: Vec<char> = sub_a.chars().collect();
    let sb: Vec<char> = sub_b.chars().collect();
    let so: Vec<char> = sub_out.chars().collect();
    if sa.len() != a_shape.len() || sb.len() != b_shape.len() {
        return None;
    }

    let distinct = |v: &[char]| {
        for i in 0..v.len() {
            for j in (i + 1)..v.len() {
                if v[i] == v[j] {
                    return None;
                }
            }
        }
        Some(())
    };
    distinct(&sa)?;
    distinct(&sb)?;
    distinct(&so)?;

    let in_a = |c: char| sa.contains(&c);
    let in_b = |c: char| sb.contains(&c);
    let in_o = |c: char| so.contains(&c);

    // Every A label must be batch (in B & out), contracted (in B, not out), or
    // free-A (out, not B). An A-only label (not in B, not in out) is a single-
    // operand reduction this matrix-product path does not model — bail.
    for &c in &sa {
        if !in_b(c) && !in_o(c) {
            return None;
        }
    }
    for &c in &sb {
        if !in_a(c) && !in_o(c) {
            return None;
        }
    }
    // No output label may be absent from both operands (no broadcast new-axis).
    for &c in &so {
        if !in_a(c) && !in_b(c) {
            return None;
        }
    }

    // Per-label extent, checking the shared dims agree between operands.
    let mut ext: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
    for (i, &c) in sa.iter().enumerate() {
        ext.insert(c, a_shape[i]);
    }
    for (i, &c) in sb.iter().enumerate() {
        if let Some(&e) = ext.get(&c)
            && e != b_shape[i]
        {
            return None;
        }
        ext.insert(c, b_shape[i]);
    }

    // Label groups. Batch / free-A / free-B follow their order of appearance in the
    // OUTPUT (so the canonical [batch, M, N] result aligns with `sub_out`'s grouping
    // when not interleaved, and a single final permute fixes interleaving). The
    // contracted labels follow their order of appearance in A (fixing the K layout).
    let batch: Vec<char> = so.iter().copied().filter(|&c| in_a(c) && in_b(c)).collect();
    let m_labels: Vec<char> = so
        .iter()
        .copied()
        .filter(|&c| in_a(c) && !in_b(c))
        .collect();
    let n_labels: Vec<char> = so
        .iter()
        .copied()
        .filter(|&c| !in_a(c) && in_b(c))
        .collect();
    let k_labels: Vec<char> = sa
        .iter()
        .copied()
        .filter(|&c| in_b(c) && !in_o(c))
        .collect();
    if k_labels.is_empty() {
        return None; // no contraction → outer product, not a GEMM
    }

    let prod = |labels: &[char]| -> usize { labels.iter().map(|&c| ext[&c]).product() };
    let bsz = prod(&batch);
    let m = prod(&m_labels);
    let k = prod(&k_labels);
    let n = prod(&n_labels);
    if bsz == 0 || m == 0 || k == 0 || n == 0 {
        return None;
    }

    // A → [batch…, M…, K…]; B → [batch…, K…, N…]. perm[i] = position of the target
    // label within the source subscript.
    let pos_in = |src: &[char], c: char| -> Option<usize> { src.iter().position(|&x| x == c) };
    let mut perm_a = Vec::with_capacity(sa.len());
    for &c in batch.iter().chain(&m_labels).chain(&k_labels) {
        perm_a.push(pos_in(&sa, c)?);
    }
    if perm_a.len() != sa.len() {
        return None;
    }
    let mut perm_b = Vec::with_capacity(sb.len());
    for &c in batch.iter().chain(&k_labels).chain(&n_labels) {
        perm_b.push(pos_in(&sb, c)?);
    }
    if perm_b.len() != sb.len() {
        return None;
    }
    if a.len() < bsz.checked_mul(m)?.checked_mul(k)?
        || b.len() < bsz.checked_mul(k)?.checked_mul(n)?
    {
        return None;
    }

    let a_perm = permute_copy_f64(a, a_shape, &perm_a); // [bsz, M, K] row-major
    let b_perm = permute_copy_f64(b, b_shape, &perm_b); // [bsz, K, N] row-major

    // One batched register-blocked GEMM over [bsz, M, K] × [bsz, K, N].
    // This preserves the same ascending-K accumulation as the previous
    // per-batch matmul loop, but lets the existing batched kernel split work
    // across batch×rows and avoids a temporary Vec per batch slice.
    let canon = crate::tensor_contraction::batched_matmul_2d(&a_perm, bsz, m, k, &b_perm, n);

    // Canonical result layout = [batch…, M…, N…]; permute to `sub_out` if needed.
    let canon_labels: Vec<char> = batch
        .iter()
        .chain(&m_labels)
        .chain(&n_labels)
        .copied()
        .collect();
    let canon_shape: Vec<usize> = canon_labels.iter().map(|&c| ext[&c]).collect();
    let final_shape: Vec<usize> = so.iter().map(|&c| ext[&c]).collect();
    if canon_labels == so {
        return Some((canon, final_shape));
    }
    let mut out_perm = Vec::with_capacity(so.len());
    for &c in &so {
        out_perm.push(pos_in(&canon_labels, c)?);
    }
    let out = permute_copy_f64(&canon, &canon_shape, &out_perm);
    Some((out, final_shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    // A/B: serial vs batch-threaded batched matmul on the attention regime (many small
    // cache-resident slices, each below matmul_2d's own 8M-op fanout floor → serial today).
    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_batched_matmul_serial_vs_threaded() {
        use std::time::Instant;
        // [B*H=256, M=128, K=64] · [256, 64, 128] -> attention-score-like batched GEMM.
        for (batch, m, k, n) in [(256usize, 128usize, 64usize, 128usize), (512, 64, 64, 64)] {
            let a: Vec<f64> = (0..batch * m * k)
                .map(|i| (i as f64 * 0.0007).sin())
                .collect();
            let b: Vec<f64> = (0..batch * k * n)
                .map(|i| (i as f64 * 0.0009).cos())
                .collect();
            let (ls, rs, os) = (m * k, k * n, m * n);
            let best = |f: &dyn Fn() -> f64| -> u64 {
                let mut bb = u64::MAX;
                for _ in 0..5 {
                    let t0 = Instant::now();
                    let acc = f();
                    std::hint::black_box(acc);
                    bb = bb.min(t0.elapsed().as_nanos() as u64);
                }
                bb
            };
            let serial = best(&|| {
                let mut out = vec![0.0f64; batch * os];
                for bt in 0..batch {
                    crate::tensor_contraction::matmul_2d_into(
                        &a[bt * ls..bt * ls + ls],
                        m,
                        k,
                        &b[bt * rs..bt * rs + rs],
                        n,
                        &mut out[bt * os..bt * os + os],
                    );
                }
                out[0] + out[out.len() / 2]
            });
            let threaded = best(&|| {
                let mut out = vec![0.0f64; batch * os];
                let threads = std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(1)
                    .min(batch);
                let per = batch.div_ceil(threads);
                std::thread::scope(|scope| {
                    let mut rest: &mut [f64] = out.as_mut_slice();
                    let mut bt0 = 0usize;
                    while bt0 < batch {
                        let cnt = per.min(batch - bt0);
                        let (blk, tail) = rest.split_at_mut(cnt * os);
                        rest = tail;
                        let (a, b) = (&a, &b);
                        let start = bt0;
                        bt0 += cnt;
                        scope.spawn(move || {
                            for j in 0..cnt {
                                let bt = start + j;
                                crate::tensor_contraction::matmul_2d_into(
                                    &a[bt * ls..bt * ls + ls],
                                    m,
                                    k,
                                    &b[bt * rs..bt * rs + rs],
                                    n,
                                    &mut blk[j * os..j * os + os],
                                );
                            }
                        });
                    }
                });
                out[0] + out[out.len() / 2]
            });
            eprintln!(
                "[batched_matmul B={batch} {m}x{k}x{n}] serial={:.3}ms threaded={:.3}ms ratio={:.2}x (min of 5)",
                serial as f64 / 1e6,
                threaded as f64 / 1e6,
                serial as f64 / threaded as f64,
            );
        }
    }

    #[test]
    fn einsum2_matmul_fast_path_bit_identical() {
        // The matmul fast path must equal the textbook ascending-contracted-index
        // reference bit-for-bit, at a size that exercises real summation rounding.
        // Also covers the alternate labelling "ik,kj->ij" (same positional form).
        let (m, k, n) = (12usize, 31usize, 17usize);
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.073).sin() * 2.5).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.041).cos() * 1.7).collect();

        let mut want = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for l in 0..k {
                    s += a[i * k + l] * b[l * n + j];
                }
                want[i * n + j] = s;
            }
        }

        for subs in ["ij,jk->ik", "ik,kj->ij", "ab,bc->ac"] {
            let (got, shape) = einsum2(subs, &a, &[m, k], &b, &[k, n]).unwrap();
            assert_eq!(shape, vec![m, n], "shape for {subs}");
            for idx in 0..m * n {
                assert_eq!(
                    got[idx].to_bits(),
                    want[idx].to_bits(),
                    "{subs} mismatch at {idx}"
                );
            }
        }
    }

    #[test]
    fn einsum2_generic_allocation_free_bit_identical() {
        // Non-matmul 2-operand patterns (which bypass the matmul fast path and
        // exercise the rewritten generic loop) must equal explicit textbook
        // references bit-for-bit. Covers matrix-vector, a@b^T, Frobenius inner
        // product, batched dot, and outer product.
        let (m, k, n) = (6usize, 5usize, 4usize);
        let am: Vec<f64> = (0..m * k)
            .map(|i| (i as f64 * 0.053).sin() * 2.0 - 0.3)
            .collect();

        // matrix-vector "ij,j->i": want[i] = sum_j a[i,j]*v[j]
        let v: Vec<f64> = (0..k).map(|j| (j as f64 * 0.21).cos() * 1.1).collect();
        let mut mv = vec![0.0f64; m];
        for i in 0..m {
            let mut s = 0.0;
            for j in 0..k {
                s += am[i * k + j] * v[j];
            }
            mv[i] = s;
        }
        let (got, sh) = einsum2("ij,j->i", &am, &[m, k], &v, &[k]).unwrap();
        assert_eq!(sh, vec![m]);
        assert!(
            got.iter().zip(&mv).all(|(g, w)| g.to_bits() == w.to_bits()),
            "matvec"
        );

        // a @ b^T "ij,kj->ik": want[i,k2] = sum_j a[i,j]*b[k2,j]
        let bk: Vec<f64> = (0..n * k).map(|i| (i as f64 * 0.037).cos() * 1.3).collect();
        let mut abt = vec![0.0f64; m * n];
        for i in 0..m {
            for k2 in 0..n {
                let mut s = 0.0;
                for j in 0..k {
                    s += am[i * k + j] * bk[k2 * k + j];
                }
                abt[i * n + k2] = s;
            }
        }
        let (got, sh) = einsum2("ij,kj->ik", &am, &[m, k], &bk, &[n, k]).unwrap();
        assert_eq!(sh, vec![m, n]);
        assert!(
            got.iter()
                .zip(&abt)
                .all(|(g, w)| g.to_bits() == w.to_bits()),
            "a@b^T"
        );

        // Frobenius "ij,ij->": want = sum_ij a[i,j]*c[i,j]
        let c2: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.067).sin() + 0.5).collect();
        let mut fro = 0.0f64;
        for i in 0..m {
            for j in 0..k {
                fro += am[i * k + j] * c2[i * k + j];
            }
        }
        let (got, sh) = einsum2("ij,ij->", &am, &[m, k], &c2, &[m, k]).unwrap();
        assert!(sh.is_empty());
        // Frobenius sums over TWO contracted indices, whose visitation order is
        // implementation-defined (the `sum_indices` set order) and unchanged by
        // this rewrite, so assert value equality within tolerance rather than
        // exact bits against a fixed-order reference.
        assert!(
            (got[0] - fro).abs() < 1e-9,
            "frobenius: {} vs {fro}",
            got[0]
        );

        // batched dot "bn,bn->b": want[b] = sum_n a[b,n]*c[b,n]
        let mut bd = vec![0.0f64; m];
        for bb in 0..m {
            let mut s = 0.0;
            for j in 0..k {
                s += am[bb * k + j] * c2[bb * k + j];
            }
            bd[bb] = s;
        }
        let (got, sh) = einsum2("bn,bn->b", &am, &[m, k], &c2, &[m, k]).unwrap();
        assert_eq!(sh, vec![m]);
        assert!(
            got.iter().zip(&bd).all(|(g, w)| g.to_bits() == w.to_bits()),
            "batched dot"
        );

        // outer "i,j->ij": want[i,j] = u[i]*w[j]
        let u: Vec<f64> = (0..m).map(|i| (i as f64 + 1.0) * 0.7).collect();
        let w2: Vec<f64> = (0..n).map(|j| (j as f64 + 1.0) * -0.4).collect();
        let mut outp = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                outp[i * n + j] = u[i] * w2[j];
            }
        }
        let (got, sh) = einsum2("i,j->ij", &u, &[m], &w2, &[n]).unwrap();
        assert_eq!(sh, vec![m, n]);
        assert!(
            got.iter()
                .zip(&outp)
                .all(|(g, w)| g.to_bits() == w.to_bits()),
            "outer"
        );
    }

    #[test]
    fn einsum1_allocation_free_bit_identical() {
        // Single-operand patterns (transpose, diagonal, trace, single-axis sum)
        // must equal explicit textbook references bit-for-bit; multi-axis sum is
        // checked within tolerance (its visitation order is set-defined).
        let (r, c) = (5usize, 7usize);
        let m: Vec<f64> = (0..r * c)
            .map(|i| (i as f64 * 0.041).sin() * 2.0 - 0.4)
            .collect();

        // transpose "ij->ji": want[j,i] = m[i,j]
        let mut tr = vec![0.0f64; c * r];
        for i in 0..r {
            for j in 0..c {
                tr[j * r + i] = m[i * c + j];
            }
        }
        let (got, sh) = einsum1("ij->ji", &m, &[r, c]).unwrap();
        assert_eq!(sh, vec![c, r]);
        assert!(
            got.iter().zip(&tr).all(|(g, w)| g.to_bits() == w.to_bits()),
            "transpose"
        );

        // single-axis reduction "ij->i": want[i] = sum_j m[i,j]
        let mut rs = vec![0.0f64; r];
        for i in 0..r {
            let mut s = 0.0;
            for j in 0..c {
                s += m[i * c + j];
            }
            rs[i] = s;
        }
        let (got, sh) = einsum1("ij->i", &m, &[r, c]).unwrap();
        assert_eq!(sh, vec![r]);
        assert!(
            got.iter().zip(&rs).all(|(g, w)| g.to_bits() == w.to_bits()),
            "row sum"
        );

        // square cases: diagonal "ii->i" and trace "ii->"
        let n = 6usize;
        let sq: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.07).cos() * 1.5).collect();
        let mut diag = vec![0.0f64; n];
        let mut trace = 0.0f64;
        for i in 0..n {
            diag[i] = sq[i * n + i];
            trace += sq[i * n + i];
        }
        let (got, sh) = einsum1("ii->i", &sq, &[n, n]).unwrap();
        assert_eq!(sh, vec![n]);
        assert!(
            got.iter()
                .zip(&diag)
                .all(|(g, w)| g.to_bits() == w.to_bits()),
            "diagonal"
        );
        let (got, sh) = einsum1("ii->", &sq, &[n, n]).unwrap();
        assert!(sh.is_empty());
        assert_eq!(got[0].to_bits(), trace.to_bits(), "trace");

        // multi-axis sum "ijk->i": value within tolerance (sum order set-defined).
        let (d0, d1, d2) = (4usize, 3usize, 5usize);
        let t3: Vec<f64> = (0..d0 * d1 * d2)
            .map(|i| (i as f64 * 0.013).sin())
            .collect();
        let mut want3 = vec![0.0f64; d0];
        for i in 0..d0 {
            let mut s = 0.0;
            for j in 0..d1 {
                for k in 0..d2 {
                    s += t3[(i * d1 + j) * d2 + k];
                }
            }
            want3[i] = s;
        }
        let (got, sh) = einsum1("ijk->i", &t3, &[d0, d1, d2]).unwrap();
        assert_eq!(sh, vec![d0]);
        assert!(
            got.iter().zip(&want3).all(|(g, w)| (g - w).abs() < 1e-9),
            "3d sum"
        );
    }

    #[test]
    fn einsum2_batched_matmul_fast_path_bit_identical() {
        // Batched matmul "B…XY,B…YZ->B…XZ" (1- and 2-batch-dim forms) must equal
        // the textbook per-batch ascending-contracted-index reference bit-for-bit.
        let (b0, b1, m, k, n) = (3usize, 2usize, 5usize, 7usize, 4usize);
        let bsz = b0 * b1;
        let a: Vec<f64> = (0..bsz * m * k)
            .map(|i| (i as f64 * 0.029).sin() * 2.0)
            .collect();
        let b: Vec<f64> = (0..bsz * k * n)
            .map(|i| (i as f64 * 0.037).cos() * 1.3)
            .collect();

        let mut want = vec![0.0f64; bsz * m * n];
        for bt in 0..bsz {
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0;
                    for l in 0..k {
                        s += a[bt * m * k + i * k + l] * b[bt * k * n + l * n + j];
                    }
                    want[bt * m * n + i * n + j] = s;
                }
            }
        }

        // 1 batch dim (bsz flattened) and 2 batch dims [b0,b1].
        type BatchedMatmulCase = (&'static str, Vec<usize>, Vec<usize>, Vec<usize>);
        let cases: [BatchedMatmulCase; 2] = [
            (
                "bij,bjk->bik",
                vec![bsz, m, k],
                vec![bsz, k, n],
                vec![bsz, m, n],
            ),
            (
                "cdij,cdjk->cdik",
                vec![b0, b1, m, k],
                vec![b0, b1, k, n],
                vec![b0, b1, m, n],
            ),
        ];
        for (subs, ash, bsh, want_shape) in cases {
            let (got, shape) = einsum2(subs, &a, &ash, &b, &bsh).unwrap();
            assert_eq!(shape, want_shape, "shape for {subs}");
            for idx in 0..bsz * m * n {
                assert_eq!(
                    got[idx].to_bits(),
                    want[idx].to_bits(),
                    "{subs} mismatch at {idx}"
                );
            }
        }
    }

    #[test]
    fn einsum_matrix_mul() {
        // "ij,jk->ik" is matrix multiplication
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = [5.0, 6.0, 7.0, 8.0]; // 2x2
        let (result, shape) = einsum2("ij,jk->ik", &a, &[2, 2], &b, &[2, 2]).unwrap();
        assert_eq!(shape, vec![2, 2]);
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        assert!((result[0] - 19.0).abs() < 1e-10);
        assert!((result[1] - 22.0).abs() < 1e-10);
        assert!((result[2] - 43.0).abs() < 1e-10);
        assert!((result[3] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn einsum_dot_product() {
        // "i,i->" is dot product
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let (result, shape) = einsum2("i,i->", &a, &[3], &b, &[3]).unwrap();
        assert!(shape.is_empty());
        assert!((result[0] - 32.0).abs() < 1e-10);
    }

    #[test]
    fn einsum_outer_product() {
        // "i,j->ij" is outer product
        let a = [1.0, 2.0];
        let b = [3.0, 4.0, 5.0];
        let (result, shape) = einsum2("i,j->ij", &a, &[2], &b, &[3]).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(result, vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn einsum_trace() {
        // "ii->" is trace
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let (result, shape) = einsum1("ii->", &a, &[2, 2]).unwrap();
        assert!(shape.is_empty());
        assert!((result[0] - 5.0).abs() < 1e-10); // 1 + 4 = 5
    }

    #[test]
    fn einsum_transpose() {
        // "ij->ji" is transpose
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let (result, shape) = einsum1("ij->ji", &a, &[2, 3]).unwrap();
        assert_eq!(shape, vec![3, 2]);
        assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn einsum_diagonal() {
        // "ii->i" extracts diagonal
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let (result, shape) = einsum1("ii->i", &a, &[2, 2]).unwrap();
        assert_eq!(shape, vec![2]);
        assert_eq!(result, vec![1.0, 4.0]);
    }

    #[test]
    fn einsum_batch_matmul() {
        // "bij,bjk->bik" is batched matrix multiplication
        // 2 batches of 2x2 matrices
        let a = [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]; // shape [2, 2, 2]
        let b = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]; // shape [2, 2, 2]
        let (result, shape) = einsum2("bij,bjk->bik", &a, &[2, 2, 2], &b, &[2, 2, 2]).unwrap();
        assert_eq!(shape, vec![2, 2, 2]);
        // First batch: I @ [[1,2],[3,4]] = [[1,2],[3,4]]
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
        assert!((result[2] - 3.0).abs() < 1e-10);
        assert!((result[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn einsum_invalid_subscript() {
        let a = [1.0, 2.0];
        let result = einsum1("AB->", &a, &[2]);
        assert!(result.is_err());
    }

    #[test]
    fn einsum_shape_mismatch() {
        let a = [1.0, 2.0, 3.0]; // 3 elements
        let b = [1.0, 2.0]; // 2 elements
        let result = einsum2("i,i->", &a, &[3], &b, &[2]);
        assert!(matches!(result, Err(EinsumError::ShapeMismatch { .. })));
    }

    #[test]
    fn einsum2_transposed_matmul_batched_bit_identical() {
        // Batched A·Bᵀ "bij,bkj->bik" must hit try_einsum2_matmul_bt and equal the
        // textbook ascending-contracted reference bit-for-bit.
        let (bt, m, k, n) = (3usize, 9usize, 13usize, 7usize);
        let a: Vec<f64> = (0..bt * m * k)
            .map(|i| (i as f64 * 0.029).sin() * 1.9)
            .collect();
        let b: Vec<f64> = (0..bt * n * k)
            .map(|i| (i as f64 * 0.047).cos() * 2.3)
            .collect();
        let mut want = vec![0.0f64; bt * m * n];
        for bb in 0..bt {
            for i in 0..m {
                for z in 0..n {
                    let mut s = 0.0;
                    for y in 0..k {
                        s += a[(bb * m + i) * k + y] * b[(bb * n + z) * k + y];
                    }
                    want[(bb * m + i) * n + z] = s;
                }
            }
        }
        let (got, shape) = einsum2("bij,bkj->bik", &a, &[bt, m, k], &b, &[bt, n, k]).unwrap();
        assert_eq!(shape, vec![bt, m, n]);
        for idx in 0..bt * m * n {
            assert_eq!(
                got[idx].to_bits(),
                want[idx].to_bits(),
                "batched a@bT at {idx}"
            );
        }
    }

    #[test]
    fn einsum2_lhs_transposed_matmul_bit_identical() {
        // Aᵀ·B "ji,jk->ik" and Aᵀ·Bᵀ "ji,kj->ik" must hit try_einsum2_matmul_at and
        // equal the textbook ascending-contracted reference bit-for-bit.
        let (m, k, n) = (11usize, 13usize, 9usize);
        let a: Vec<f64> = (0..k * m).map(|i| (i as f64 * 0.031).sin() * 1.7).collect(); // [k, m]

        // Aᵀ·B "ji,jk->ik": rhs canonical [k, n]. want[x,z] = sum_y a[y,x]*bkn[y,z].
        let bkn: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.043).cos() * 2.1).collect();
        let mut want_atb = vec![0.0f64; m * n];
        for x in 0..m {
            for z in 0..n {
                let mut s = 0.0;
                for y in 0..k {
                    s += a[y * m + x] * bkn[y * n + z];
                }
                want_atb[x * n + z] = s;
            }
        }
        let (got, sh) = einsum2("ji,jk->ik", &a, &[k, m], &bkn, &[k, n]).unwrap();
        assert_eq!(sh, vec![m, n]);
        for idx in 0..m * n {
            assert_eq!(got[idx].to_bits(), want_atb[idx].to_bits(), "AtB at {idx}");
        }

        // Aᵀ·Bᵀ "ji,kj->ik": rhs [n, k]. want[x,z] = sum_y a[y,x]*bnk[z,y].
        let bnk: Vec<f64> = (0..n * k).map(|i| (i as f64 * 0.037).cos() * 1.9).collect();
        let mut want_atbt = vec![0.0f64; m * n];
        for x in 0..m {
            for z in 0..n {
                let mut s = 0.0;
                for y in 0..k {
                    s += a[y * m + x] * bnk[z * k + y];
                }
                want_atbt[x * n + z] = s;
            }
        }
        let (got2, sh2) = einsum2("ji,kj->ik", &a, &[k, m], &bnk, &[n, k]).unwrap();
        assert_eq!(sh2, vec![m, n]);
        for idx in 0..m * n {
            assert_eq!(
                got2[idx].to_bits(),
                want_atbt[idx].to_bits(),
                "AtBt at {idx}"
            );
        }
    }

    #[test]
    fn einsum2_grouped_matmul_bit_identical() {
        // Grouped free axes, single contraction "abc,cde->abde": M=[a,b], K=[c],
        // N=[d,e]. Bit-identical to the ascending-K textbook reference.
        let (da, db, dc, dd, de) = (3usize, 4, 5, 2, 6);
        let a: Vec<f64> = (0..da * db * dc)
            .map(|i| (i as f64 * 0.021).sin() * 1.3)
            .collect();
        let b: Vec<f64> = (0..dc * dd * de)
            .map(|i| (i as f64 * 0.033).cos() * 1.7)
            .collect();
        let (mm, kk, nn) = (da * db, dc, dd * de);
        let mut want = vec![0.0f64; mm * nn];
        for mi in 0..mm {
            for ni in 0..nn {
                let mut s = 0.0;
                for ki in 0..kk {
                    s += a[mi * kk + ki] * b[ki * nn + ni];
                }
                want[mi * nn + ni] = s;
            }
        }
        let (got, sh) = einsum2("abc,cde->abde", &a, &[da, db, dc], &b, &[dc, dd, de]).unwrap();
        assert_eq!(sh, vec![da, db, dd, de]);
        for idx in 0..mm * nn {
            assert_eq!(
                got[idx].to_bits(),
                want[idx].to_bits(),
                "abc,cde->abde at {idx}"
            );
        }

        // Grouped contraction "abcd,cdef->abef": M=[a,b], K=[c,d], N=[e,f]. The
        // collapsed-K ascending order matches matmul_2d (one valid ordering of the
        // generic's nondeterministic multi-axis sum).
        let (a2, b2, c2, d2, e2, f2) = (2usize, 3, 2, 3, 2, 4);
        let av: Vec<f64> = (0..a2 * b2 * c2 * d2)
            .map(|i| (i as f64 * 0.017).sin())
            .collect();
        let bv: Vec<f64> = (0..c2 * d2 * e2 * f2)
            .map(|i| (i as f64 * 0.029).cos())
            .collect();
        let (mm2, kk2, nn2) = (a2 * b2, c2 * d2, e2 * f2);
        let mut want2 = vec![0.0f64; mm2 * nn2];
        for mi in 0..mm2 {
            for ni in 0..nn2 {
                let mut s = 0.0;
                for ki in 0..kk2 {
                    s += av[mi * kk2 + ki] * bv[ki * nn2 + ni];
                }
                want2[mi * nn2 + ni] = s;
            }
        }
        let (got2, sh2) = einsum2(
            "abcd,cdef->abef",
            &av,
            &[a2, b2, c2, d2],
            &bv,
            &[c2, d2, e2, f2],
        )
        .unwrap();
        assert_eq!(sh2, vec![a2, b2, e2, f2]);
        for idx in 0..mm2 * nn2 {
            assert_eq!(
                got2[idx].to_bits(),
                want2[idx].to_bits(),
                "abcd,cdef->abef at {idx}"
            );
        }
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_einsum2_grouped_vs_odometer() {
        use std::time::Instant;
        let (a, b, c, d, e, f) = (32usize, 32, 32, 32, 32, 32); // M=K=N=1024
        let (mm, kk, nn) = (a * b, c * d, e * f);
        let av: Vec<f64> = (0..mm * kk).map(|i| ((i as f64) * 0.0007).sin()).collect();
        let bv: Vec<f64> = (0..kk * nn).map(|i| ((i as f64) * 0.0009).cos()).collect();
        // "Before": the generic odometer's effective inner contraction — b strided
        // by N over the contracted index (the cache-pathological rhs orientation).
        let odometer = || {
            let mut out = vec![0.0f64; mm * nn];
            for mi in 0..mm {
                for ni in 0..nn {
                    let mut s = 0.0;
                    for ki in 0..kk {
                        s += av[mi * kk + ki] * bv[ki * nn + ni];
                    }
                    out[mi * nn + ni] = s;
                }
            }
            out
        };
        let fast = || {
            einsum2("abcd,cdef->abef", &av, &[a, b, c, d], &bv, &[c, d, e, f])
                .unwrap()
                .0
        };
        let best = |g: &dyn Fn() -> Vec<f64>| {
            let _ = g();
            let mut t = f64::MAX;
            for _ in 0..3 {
                let s = Instant::now();
                std::hint::black_box(g());
                t = t.min(s.elapsed().as_secs_f64());
            }
            t
        };
        let base = best(&odometer);
        let fst = best(&fast);
        println!(
            "BENCH einsum2 grouped [{mm}x{kk}·{kk}x{nn}]: odometer={:.2}ms matmul2d={:.2}ms speedup={:.2}x",
            base * 1e3,
            fst * 1e3,
            base / fst
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_einsum2_lhs_transposed_vs_odometer() {
        use std::time::Instant;
        let (m, k, n) = (1024usize, 1024usize, 1024usize);
        let a: Vec<f64> = (0..k * m).map(|i| ((i as f64) * 0.0007).sin()).collect(); // [k, m]
        let b: Vec<f64> = (0..k * n).map(|i| ((i as f64) * 0.0009).cos()).collect(); // [k, n]
        // "Before": the odometer's inner contraction for Aᵀ·B — the lhs is accessed
        // column-strided (a[y*m+x]), the cache-UNfriendly orientation.
        let odometer = || {
            let mut out = vec![0.0f64; m * n];
            for x in 0..m {
                for z in 0..n {
                    let mut s = 0.0;
                    for y in 0..k {
                        s += a[y * m + x] * b[y * n + z];
                    }
                    out[x * n + z] = s;
                }
            }
            out
        };
        let fast = || einsum2("ji,jk->ik", &a, &[k, m], &b, &[k, n]).unwrap().0;
        let best = |f: &dyn Fn() -> Vec<f64>| {
            let _ = f();
            let mut t = f64::MAX;
            for _ in 0..3 {
                let s = Instant::now();
                std::hint::black_box(f());
                t = t.min(s.elapsed().as_secs_f64());
            }
            t
        };
        let base = best(&odometer);
        let fst = best(&fast);
        println!(
            "BENCH einsum2 Aᵀ·B [{k}x{m}·{k}x{n}]: odometer={:.2}ms matmul2d={:.2}ms speedup={:.2}x",
            base * 1e3,
            fst * 1e3,
            base / fst
        );
    }

    #[test]
    #[ignore = "benchmark: run with --ignored --nocapture"]
    fn bench_einsum2_transposed_vs_dot_loop() {
        use std::time::Instant;
        let (m, k, n) = (1024usize, 1024usize, 1024usize);
        let a: Vec<f64> = (0..m * k).map(|i| ((i as f64) * 0.0007).sin()).collect();
        let b: Vec<f64> = (0..n * k).map(|i| ((i as f64) * 0.0009).cos()).collect();
        // "Before": the contiguous-contiguous dot loop the generic odometer reduces
        // to for A·Bᵀ (both contracted strides == 1) — the cache-friendly baseline.
        let dot_loop = || {
            let mut out = vec![0.0f64; m * n];
            for i in 0..m {
                let arow = &a[i * k..i * k + k];
                for z in 0..n {
                    let brow = &b[z * k..z * k + k];
                    let mut s = 0.0;
                    for y in 0..k {
                        s += arow[y] * brow[y];
                    }
                    out[i * n + z] = s;
                }
            }
            out
        };
        let fast = || einsum2("ij,kj->ik", &a, &[m, k], &b, &[n, k]).unwrap().0;
        let best = |f: &dyn Fn() -> Vec<f64>| {
            let _ = f();
            let mut t = f64::MAX;
            for _ in 0..3 {
                let s = Instant::now();
                std::hint::black_box(f());
                t = t.min(s.elapsed().as_secs_f64());
            }
            t
        };
        let base = best(&dot_loop);
        let fst = best(&fast);
        println!(
            "BENCH einsum2 A·Bᵀ [{m}x{k}·{n}x{k}]: dot_loop={:.2}ms matmul2d={:.2}ms speedup={:.2}x",
            base * 1e3,
            fst * 1e3,
            base / fst
        );
    }

    /// Deterministic ascending-K naive contraction reference: contracted labels are
    /// those in A∩B not in out, taken in A's order; summed row-major. Mirrors the
    /// general GEMM path's K flattening exactly, so the two must agree bit-for-bit.
    fn naive_einsum2(
        sub_a: &str,
        sub_b: &str,
        sub_out: &str,
        a: &[f64],
        a_shape: &[usize],
        b: &[f64],
        b_shape: &[usize],
    ) -> (Vec<f64>, Vec<usize>) {
        let sa: Vec<char> = sub_a.chars().collect();
        let sb: Vec<char> = sub_b.chars().collect();
        let so: Vec<char> = sub_out.chars().collect();
        let mut ext = std::collections::HashMap::new();
        for (i, &c) in sa.iter().enumerate() {
            ext.insert(c, a_shape[i]);
        }
        for (i, &c) in sb.iter().enumerate() {
            ext.insert(c, b_shape[i]);
        }
        let k_labels: Vec<char> = sa
            .iter()
            .copied()
            .filter(|&c| sb.contains(&c) && !so.contains(&c))
            .collect();
        let out_shape: Vec<usize> = so.iter().map(|&c| ext[&c]).collect();
        let a_str = row_major_strides(a_shape);
        let b_str = row_major_strides(b_shape);
        let o_str = row_major_strides(&out_shape);
        let out_total: usize = out_shape.iter().product();
        let k_dims: Vec<usize> = k_labels.iter().map(|&c| ext[&c]).collect();
        let k_total: usize = k_dims.iter().product::<usize>().max(1);
        let mut out = vec![0.0f64; out_total];
        let mut coord = std::collections::HashMap::new();
        for (out_idx, slot) in out.iter_mut().enumerate() {
            for (ax, &c) in so.iter().enumerate() {
                coord.insert(c, (out_idx / o_str[ax]) % out_shape[ax]);
            }
            let mut sum = 0.0f64;
            let mut kc = vec![0usize; k_labels.len()];
            for _ in 0..k_total {
                for (j, &c) in k_labels.iter().enumerate() {
                    coord.insert(c, kc[j]);
                }
                let mut ai = 0usize;
                for (ax, &c) in sa.iter().enumerate() {
                    ai += coord[&c] * a_str[ax];
                }
                let mut bi = 0usize;
                for (ax, &c) in sb.iter().enumerate() {
                    bi += coord[&c] * b_str[ax];
                }
                sum += a[ai] * b[bi];
                for j in (0..k_labels.len()).rev() {
                    kc[j] += 1;
                    if kc[j] < k_dims[j] {
                        break;
                    }
                    kc[j] = 0;
                }
            }
            *slot = sum;
        }
        (out, out_shape)
    }

    #[test]
    fn einsum2_general_matmul_bit_identical_to_naive() {
        // Interleaved batch/free/contracted patterns the four pattern fast paths do
        // NOT catch (so the general permute+GEMM path runs), checked bit-for-bit
        // against the deterministic ascending-K reference. Single- and multi-axis K,
        // and an interleaved output order.
        struct Case {
            subs: &'static str, // "sa,sb->so"
            a_shape: Vec<usize>,
            b_shape: Vec<usize>,
        }
        let cases = [
            // attention scores, head axis in the middle (batch={b,h}, M={q}, N={k}, K={d})
            Case {
                subs: "bqhd,bkhd->bhqk",
                a_shape: vec![2, 5, 3, 4],
                b_shape: vec![2, 6, 3, 4],
            },
            // two contracted axes (d,e), interleaved batch — exercises multi-K order
            Case {
                subs: "bqhde,bkhde->bhqk",
                a_shape: vec![2, 5, 3, 4, 2],
                b_shape: vec![2, 6, 3, 4, 2],
            },
            // interleaved OUTPUT order (canonical [b,h,q,k] permuted to b,q,h,k)
            Case {
                subs: "bqhd,bkhd->bqhk",
                a_shape: vec![2, 5, 3, 4],
                b_shape: vec![2, 6, 3, 4],
            },
            // plain tensordot over non-adjacent axes: contract b,d
            Case {
                subs: "abcd,bedf->acef",
                a_shape: vec![3, 4, 2, 5],
                b_shape: vec![4, 3, 5, 2],
            },
        ];
        for case in &cases {
            let (sub_a, rest) = case.subs.split_once(',').unwrap();
            let (sub_b, sub_out) = rest.split_once("->").unwrap();
            let a_n: usize = case.a_shape.iter().product();
            let b_n: usize = case.b_shape.iter().product();
            let a: Vec<f64> = (0..a_n).map(|i| (i as f64 * 0.0137).sin() * 3.1).collect();
            let b: Vec<f64> = (0..b_n).map(|i| (i as f64 * 0.0091).cos() * 2.3).collect();

            // The general path must accept these (not fall through to None).
            let got = super::try_einsum2_matmul_general(
                sub_a,
                sub_b,
                sub_out,
                &a,
                &case.a_shape,
                &b,
                &case.b_shape,
            )
            .unwrap_or_else(|| panic!("general path rejected {}", case.subs));
            let want = naive_einsum2(sub_a, sub_b, sub_out, &a, &case.a_shape, &b, &case.b_shape);
            assert_eq!(got.1, want.1, "shape for {}", case.subs);
            for (i, (&g, &w)) in got.0.iter().zip(want.0.iter()).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "value[{i}] mismatch for {} (got {g}, want {w})",
                    case.subs
                );
            }
            // And the full einsum2 dispatch agrees.
            let full = einsum2(case.subs, &a, &case.a_shape, &b, &case.b_shape).unwrap();
            assert_eq!(full.0, got.0, "dispatch mismatch for {}", case.subs);
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_einsum2_general_vs_odometer() {
        use std::time::Instant;
        // Attention-scores einsum with the head axis in the middle (bqhd,bkhd->bhqk).
        // NEW general permute+GEMM path vs the OLD strided odometer (reconstructed via
        // naive_einsum2, which is the same O(out·sum) gather the generic path runs).
        let (bsz, q, h, d, kk) = (8usize, 128usize, 8usize, 64usize, 128usize);
        let a_shape = vec![bsz, q, h, d];
        let b_shape = vec![bsz, kk, h, d];
        let a: Vec<f64> = (0..bsz * q * h * d)
            .map(|i| (i as f64 * 0.0011).sin())
            .collect();
        let b: Vec<f64> = (0..bsz * kk * h * d)
            .map(|i| (i as f64 * 0.0009).cos())
            .collect();
        let best = |f: &dyn Fn() -> Vec<f64>| {
            let _ = f();
            let mut t = f64::MAX;
            let mut digest = 0u64;
            for _ in 0..5 {
                let s = Instant::now();
                let v = f();
                t = t.min(s.elapsed().as_secs_f64());
                digest = v.iter().fold(0u64, |acc, &x| acc ^ x.to_bits());
            }
            (t, digest)
        };
        let (t_new, d_new) = best(&|| {
            super::try_einsum2_matmul_general("bqhd", "bkhd", "bhqk", &a, &a_shape, &b, &b_shape)
                .unwrap()
                .0
        });
        let (t_old, d_old) =
            best(&|| naive_einsum2("bqhd", "bkhd", "bhqk", &a, &a_shape, &b, &b_shape).0);
        assert_eq!(d_new, d_old, "general vs odometer digest");
        println!(
            "BENCH einsum2 bqhd,bkhd->bhqk [{bsz},{q},{h},{d}/{kk}]: odometer={:.2}ms general(permute+GEMM)={:.2}ms speedup={:.2}x digest={d_old:016x}",
            t_old * 1e3,
            t_new * 1e3,
            t_old / t_new
        );
    }
}
