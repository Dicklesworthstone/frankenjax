//! Tensor contraction operations matching JAX's jnp module.
//!
//! Provides tensordot and related operations.

#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

/// Compute tensor dot product along specified axes.
///
/// Matches `jnp.tensordot(a, b, axes)`.
///
/// # Arguments
/// * `a` - First tensor as flat f64 array
/// * `a_shape` - Shape of first tensor
/// * `b` - Second tensor as flat f64 array
/// * `b_shape` - Shape of second tensor
/// * `axes_a` - Axes of `a` to contract
/// * `axes_b` - Axes of `b` to contract (must have same length as axes_a)
///
/// # Returns
/// Result tensor and its shape
pub fn tensordot(
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
    axes_a: &[usize],
    axes_b: &[usize],
) -> (Vec<f64>, Vec<usize>) {
    assert_eq!(
        axes_a.len(),
        axes_b.len(),
        "axes_a and axes_b must have same length"
    );

    // Validate contraction dimensions match
    for (&ax_a, &ax_b) in axes_a.iter().zip(axes_b.iter()) {
        assert_eq!(
            a_shape[ax_a], b_shape[ax_b],
            "contracted dimensions must match"
        );
    }

    // Compute output shape: non-contracted axes of a, then non-contracted axes of b
    let mut out_shape = Vec::new();
    for (i, &dim) in a_shape.iter().enumerate() {
        if !axes_a.contains(&i) {
            out_shape.push(dim);
        }
    }
    for (i, &dim) in b_shape.iter().enumerate() {
        if !axes_b.contains(&i) {
            out_shape.push(dim);
        }
    }

    if out_shape.is_empty() {
        // Scalar result (full contraction)
        let mut sum = 0.0;
        for (i, &av) in a.iter().enumerate() {
            sum += av * b[i];
        }
        return (vec![sum], vec![]);
    }

    // Compute strides for a and b
    let a_strides = compute_strides(a_shape);
    let b_strides = compute_strides(b_shape);
    let out_strides = compute_strides(&out_shape);

    // Get non-contracted axes
    let free_axes_a: Vec<usize> = (0..a_shape.len()).filter(|i| !axes_a.contains(i)).collect();
    let _free_axes_b: Vec<usize> = (0..b_shape.len()).filter(|i| !axes_b.contains(i)).collect();

    // Contracted dimension size
    let contracted_size: usize = axes_a.iter().map(|&ax| a_shape[ax]).product();

    // Output size
    let out_size: usize = out_shape.iter().product();
    let mut result = vec![0.0; out_size];

    // For each output element
    for out_idx in 0..out_size {
        // Decode output index into coordinates
        let out_coords = index_to_coords(out_idx, &out_strides, out_shape.len());

        // Split output coords: first part for a's free axes, second for b's free axes
        let a_free_coords = &out_coords[..free_axes_a.len()];
        let b_free_coords = &out_coords[free_axes_a.len()..];

        // Sum over contracted dimensions
        let mut sum = 0.0;
        for c_idx in 0..contracted_size {
            // Decode contracted index into coordinates
            let c_coords = index_to_contracted_coords(c_idx, axes_a, a_shape);

            // Build full a index
            let mut a_flat_idx = 0;
            let mut free_i = 0;
            let mut contr_i = 0;
            for ax in 0..a_shape.len() {
                let coord = if axes_a.contains(&ax) {
                    let c = c_coords[contr_i];
                    contr_i += 1;
                    c
                } else {
                    let c = a_free_coords[free_i];
                    free_i += 1;
                    c
                };
                a_flat_idx += coord * a_strides[ax];
            }

            // Build full b index
            let mut b_flat_idx = 0;
            let mut free_i = 0;
            let mut contr_i = 0;
            for ax in 0..b_shape.len() {
                let coord = if axes_b.contains(&ax) {
                    let c = c_coords[contr_i];
                    contr_i += 1;
                    c
                } else {
                    let c = b_free_coords[free_i];
                    free_i += 1;
                    c
                };
                b_flat_idx += coord * b_strides[ax];
            }

            sum += a[a_flat_idx] * b[b_flat_idx];
        }
        result[out_idx] = sum;
    }

    (result, out_shape)
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[allow(clippy::manual_checked_ops)]
fn index_to_coords(mut idx: usize, strides: &[usize], ndim: usize) -> Vec<usize> {
    let mut coords = vec![0; ndim];
    for i in 0..ndim {
        if strides[i] > 0 {
            coords[i] = idx / strides[i];
            idx %= strides[i];
        }
    }
    coords
}

fn index_to_contracted_coords(idx: usize, axes: &[usize], shape: &[usize]) -> Vec<usize> {
    let contracted_shape: Vec<usize> = axes.iter().map(|&ax| shape[ax]).collect();
    let contracted_strides = compute_strides(&contracted_shape);
    index_to_coords(idx, &contracted_strides, axes.len())
}

/// Matrix multiplication as a special case of tensordot.
///
/// Matches `jnp.matmul(a, b)` for 2D arrays.
pub fn matmul_2d(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    // Parallelize large products across disjoint output ROW-BLOCKS (scoped
    // threads, no external dependency, 100% safe). Threading is gated to large
    // matmuls: at 256³ (~16.7M FMAs) the serial i-k-j is already L2-served and
    // thread/bandwidth overhead REGRESSES it (~0.74×), but at 512³ (~134M FMAs)
    // row-block threading is ~3.7× (worker-corrected, RCH). The threshold keeps
    // small/medium matmuls on the zero-overhead serial path.
    //
    // (pass110: single-thread NBxKB cache-blocking REGRESSED ~0.60× and 4-row M
    // register-tiling gave ~1.09× — the serial kernel is already L2-served, so
    // the remaining axis is parallelism.)
    //
    // (pass119, REJECTED: a k-blocked threaded kernel — kk outer so each KB×n B
    // panel is reused L2-resident across a thread's rows — was hypothesized to
    // rescue threaded 256³ by cutting shared-L3 B traffic. Measured 5.82ms vs
    // 4.6ms serial = still ~0.79× on the ~48-core RCH worker. The 256³ regression
    // is NOT B-bandwidth: it is too little work per core (16.7M FMAs / ~48 cores
    // ≈ 0.35M FMAs/thread) to amortize the thread fan-out, which blocking can't
    // change. So 256³ correctly stays serial; the threshold below is well-placed.
    // The next real single-thread lever is a packed register MR×NR microkernel —
    // see artifacts/performance/evidence/fj_lax_gemm_blocked_threaded_rejected_pass119.)
    const PARALLEL_MIN_OPS: usize = 1 << 26; // ~67M FMAs (~406³); 256³ stays serial
    let ops = m.saturating_mul(k).saturating_mul(n);
    let threads = if ops >= PARALLEL_MIN_OPS {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
            .min(m.max(1))
    } else {
        1
    };
    matmul_2d_with_threads(a, m, k, b, n, threads)
}

/// `matmul_2d` driver with an explicit thread count (1 = serial). Splitting the
/// output into disjoint contiguous row-blocks across scoped threads is bit-for-bit
/// identical to the serial kernel: every output row is computed by exactly one
/// thread accumulating in ascending-`l` order (see matmul_2d_threaded_bit_identical
/// and dot_rank2_matmul_f64_matches_row_major_ijk_bits).
fn matmul_2d_with_threads(
    a: &[f64],
    m: usize,
    k: usize,
    b: &[f64],
    n: usize,
    threads: usize,
) -> Vec<f64> {
    let mut result = vec![0.0; m * n];
    if m == 0 || n == 0 || k == 0 {
        return result;
    }
    if threads <= 1 {
        matmul_2d_row_block(a, k, b, n, 0, &mut result);
        return result;
    }

    let rows_per = m.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut rest: &mut [f64] = result.as_mut_slice();
        let mut row_start = 0usize;
        while row_start < m {
            let chunk_rows = rows_per.min(m - row_start);
            let (block, tail) = rest.split_at_mut(chunk_rows * n);
            rest = tail;
            let rs = row_start;
            scope.spawn(move || matmul_2d_row_block(a, k, b, n, rs, block));
            row_start += chunk_rows;
        }
    });
    result
}

/// Compute a contiguous block of output rows (starting at `row_start`,
/// `block.len() / n` rows) of the m×n product into `block`, via the i-k-j
/// kernel. Each output element accumulates `a[i][l]*b[l][j]` in ascending-`l`
/// order — bit-for-bit identical to the serial whole-matrix kernel.
fn matmul_2d_row_block(a: &[f64], k: usize, b: &[f64], n: usize, row_start: usize, block: &mut [f64]) {
    for (ri, c_row) in block.chunks_mut(n).enumerate() {
        let a_row = (row_start + ri) * k;
        for l in 0..k {
            let a_il = a[a_row + l];
            let src = &b[l * n..l * n + n];
            for j in 0..n {
                c_row[j] += a_il * src[j];
            }
        }
    }
}

/// Batched matrix product: `a` is `[batch, m, k]`, `b` is `[batch, k, n]`, both
/// row-major flat; returns `[batch, m, n]` flat. Each batch slice is an
/// independent GEMM. Parallelizes over the flattened (batch × output-row) space
/// with the same i-k-j kernel, so it uses the fast contiguous kernel *and*
/// saturates many cores even when the batch count is small — and every output
/// element accumulates in ascending-`l` order, bit-for-bit identical to a serial
/// per-batch matmul / the generic dot_general loop (see
/// batched_matmul_2d_bit_identical).
pub fn batched_matmul_2d(a: &[f64], batch: usize, m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0; batch * m * n];
    if batch == 0 || m == 0 || n == 0 || k == 0 {
        return result;
    }
    let total_rows = batch * m;
    const PARALLEL_MIN_OPS: usize = 1 << 26; // ~67M FMAs
    let ops = total_rows.saturating_mul(k).saturating_mul(n);
    let threads = if ops >= PARALLEL_MIN_OPS {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
            .min(total_rows)
    } else {
        1
    };

    if threads <= 1 {
        batched_matmul_row_block(a, b, m, k, n, 0, &mut result);
        return result;
    }

    let rows_per = total_rows.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut rest: &mut [f64] = result.as_mut_slice();
        let mut g_start = 0usize;
        while g_start < total_rows {
            let chunk_rows = rows_per.min(total_rows - g_start);
            let (block, tail) = rest.split_at_mut(chunk_rows * n);
            rest = tail;
            let gs = g_start;
            scope.spawn(move || batched_matmul_row_block(a, b, m, k, n, gs, block));
            g_start += chunk_rows;
        }
    });
    result
}

/// Compute global output rows `[g_start, g_start + block.len()/n)` of a batched
/// matmul into `block`. Global row `g` is batch `g / m`, row `g % m`: its A row
/// is `a[g*k .. g*k+k]` (A is `[batch,m,k]` so `(bt*m+i)*k == g*k`) and its B
/// panel is batch `g/m`'s `[k,n]` block. Ascending-`l` accumulation.
fn batched_matmul_row_block(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
    g_start: usize,
    block: &mut [f64],
) {
    for (ri, c_row) in block.chunks_mut(n).enumerate() {
        let g = g_start + ri;
        let bt = g / m;
        let a_off = g * k;
        let b_off = bt * k * n;
        for l in 0..k {
            let a_il = a[a_off + l];
            let src = &b[b_off + l * n..b_off + l * n + n];
            for j in 0..n {
                c_row[j] += a_il * src[j];
            }
        }
    }
}

/// Outer product of two vectors.
///
/// Matches `jnp.outer(a, b)`.
pub fn outer(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; a.len() * b.len()];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            result[i * b.len() + j] = av * bv;
        }
    }
    result
}

/// Inner product (dot product) of two vectors.
///
/// Matches `jnp.inner(a, b)` for 1D arrays.
pub fn inner(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "vectors must have same length");
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Kronecker product of two matrices.
///
/// Matches `jnp.kron(a, b)`.
pub fn kron(a: &[f64], a_m: usize, a_n: usize, b: &[f64], b_m: usize, b_n: usize) -> Vec<f64> {
    let out_m = a_m * b_m;
    let out_n = a_n * b_n;
    let mut result = vec![0.0; out_m * out_n];

    for i in 0..a_m {
        for j in 0..a_n {
            let a_val = a[i * a_n + j];
            for k in 0..b_m {
                for l in 0..b_n {
                    let out_row = i * b_m + k;
                    let out_col = j * b_n + l;
                    result[out_row * out_n + out_col] = a_val * b[k * b_n + l];
                }
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensordot_matrix_mul() {
        // 2x3 @ 3x2 = 2x2 via tensordot
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (result, shape) = tensordot(&a, &[2, 3], &b, &[3, 2], &[1], &[0]);
        assert_eq!(shape, vec![2, 2]);
        // [1,2,3] @ [1,3,5]^T = 1+6+15 = 22
        // [1,2,3] @ [2,4,6]^T = 2+8+18 = 28
        // [4,5,6] @ [1,3,5]^T = 4+15+30 = 49
        // [4,5,6] @ [2,4,6]^T = 8+20+36 = 64
        assert!((result[0] - 22.0).abs() < 1e-10);
        assert!((result[1] - 28.0).abs() < 1e-10);
        assert!((result[2] - 49.0).abs() < 1e-10);
        assert!((result[3] - 64.0).abs() < 1e-10);
    }

    #[test]
    fn tensordot_vector_dot() {
        // Vector dot product: sum(a * b)
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let (result, shape) = tensordot(&a, &[3], &b, &[3], &[0], &[0]);
        assert!(shape.is_empty()); // Scalar result
        assert!((result[0] - 32.0).abs() < 1e-10); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn tensordot_outer_product() {
        // Outer product: tensordot with no axes contracted
        let a = [1.0, 2.0];
        let b = [3.0, 4.0, 5.0];
        let (result, shape) = tensordot(&a, &[2], &b, &[3], &[], &[]);
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(result, vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn matmul_2d_basic() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = [5.0, 6.0, 7.0, 8.0]; // 2x2
        let result = matmul_2d(&a, 2, 2, &b, 2);
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        assert!((result[0] - 19.0).abs() < 1e-10);
        assert!((result[1] - 22.0).abs() < 1e-10);
        assert!((result[2] - 43.0).abs() < 1e-10);
        assert!((result[3] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn matmul_2d_ikj_bit_identical_to_ijk() {
        // The i-k-j kernel must equal the textbook i-j-k accumulation
        // (ascending-l order) bit-for-bit, including rounding.
        let (m, k, n) = (17usize, 23usize, 19usize);
        let a: Vec<f64> = (0..m * k)
            .map(|i| (i as f64 * 0.123_45).sin() * 3.0)
            .collect();
        let b: Vec<f64> = (0..k * n)
            .map(|i| (i as f64 * 0.067_89).cos() * 2.0)
            .collect();

        let got = matmul_2d(&a, m, k, &b, n);

        // Reference: original i-j-k order.
        let mut want = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                want[i * n + j] = sum;
            }
        }
        for idx in 0..m * n {
            assert_eq!(got[idx].to_bits(), want[idx].to_bits(), "mismatch at {idx}");
        }
    }

    #[test]
    fn batched_matmul_2d_bit_identical() {
        // Batched matmul kernel must equal the textbook per-batch ascending-l
        // reference bit-for-bit.
        let (bt, m, k, n) = (3usize, 5usize, 7usize, 4usize);
        let a: Vec<f64> = (0..bt * m * k).map(|i| (i as f64 * 0.021).sin() * 2.0 - 0.5).collect();
        let b: Vec<f64> = (0..bt * k * n).map(|i| (i as f64 * 0.029).cos() * 1.6 + 0.3).collect();
        let got = batched_matmul_2d(&a, bt, m, k, &b, n);
        let mut want = vec![0.0f64; bt * m * n];
        for batch in 0..bt {
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0;
                    for l in 0..k {
                        s += a[(batch * m + i) * k + l] * b[(batch * k + l) * n + j];
                    }
                    want[(batch * m + i) * n + j] = s;
                }
            }
        }
        for idx in 0..bt * m * n {
            assert_eq!(got[idx].to_bits(), want[idx].to_bits(), "mismatch at {idx}");
        }
    }

    #[test]
    fn matmul_2d_threaded_bit_identical() {
        // The multi-threaded row-block driver must equal the serial kernel
        // bit-for-bit, including a thread count that exceeds the row count (so
        // some threads get empty/partial blocks) and a partial last block.
        let (m, k, n) = (13usize, 17usize, 11usize);
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64 * 0.019).sin() * 3.0 - 0.7).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64 * 0.023).cos() * 1.9 + 0.2).collect();
        let serial = super::matmul_2d_with_threads(&a, m, k, &b, n, 1);
        for threads in [2usize, 3, 4, 8, 16, 32] {
            let parallel = super::matmul_2d_with_threads(&a, m, k, &b, n, threads);
            assert_eq!(serial.len(), parallel.len());
            for idx in 0..serial.len() {
                assert_eq!(
                    serial[idx].to_bits(),
                    parallel[idx].to_bits(),
                    "threads={threads} mismatch at {idx}"
                );
            }
        }
    }

    #[test]
    fn matmul_2d_large_bit_identical_to_ijk() {
        // Large, non-power-of-two dims (m=130, k=300, n=290) — guards the
        // matmul_2d kernel's bit-exact ascending-l accumulation at sizes well
        // past L1/L2, where future blocking/tiling/SIMD reworks are most
        // tempting. Must equal the textbook i-j-k accumulation bit-for-bit.
        let (m, k, n) = (130usize, 300usize, 290usize);
        let a: Vec<f64> = (0..m * k)
            .map(|i| (i as f64 * 0.013_57).sin() * 3.0 - 1.0)
            .collect();
        let b: Vec<f64> = (0..k * n)
            .map(|i| (i as f64 * 0.004_21).cos() * 2.0 + 0.5)
            .collect();

        let got = matmul_2d(&a, m, k, &b, n);

        let mut want = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                want[i * n + j] = sum;
            }
        }
        for idx in 0..m * n {
            assert_eq!(got[idx].to_bits(), want[idx].to_bits(), "mismatch at {idx}");
        }
    }

    #[test]
    fn outer_basic() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        let result = outer(&a, &b);
        assert_eq!(result, vec![3.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn inner_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = inner(&a, &b);
        assert!((result - 32.0).abs() < 1e-10);
    }

    #[test]
    fn kron_2x2() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = [0.0, 5.0, 6.0, 7.0]; // 2x2
        let result = kron(&a, 2, 2, &b, 2, 2);
        // Result is 4x4
        assert_eq!(result.len(), 16);
        // First 2x2 block is 1*b
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 5.0).abs() < 1e-10);
    }
}
