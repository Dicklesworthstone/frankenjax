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

/// Register-tile dimensions for the GEMM microkernel: `MR` output rows × `NR`
/// output columns are accumulated together, streamed over `k`.
const MR: usize = 4;
const NR: usize = 8;
type F64xN = std::simd::Simd<f64, NR>;
const F32_NR: usize = 16;
type F32xN = std::simd::Simd<f32, F32_NR>;

#[inline]
fn f64xnr_from_slice(values: &[f64]) -> F64xN {
    F64xN::from_array([
        values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
    ])
}

/// k-dimension block for the cache-blocked macro-kernel. At KC=256 an [MR×KC] A
/// tile and a [KC×NR] B panel are ~8 KB each — both stay L1-resident across a
/// pc-block, so B is reused across all row-tiles and A across all column-panels
/// from L1 instead of re-streaming B from L3 once per MR-row-tile. The cost is
/// re-touching C once per pc-block (k/KC passes over the m×n output); KC is sized
/// so that handful of C passes is far cheaper than the m/MR B re-streams it
/// replaces.
const KC: usize = 256;

/// Superpanel dimensions for the row-contiguous blocked GEMM traversal. The
/// kernel still computes one output tile with the same ascending `l` loop, but
/// walks adjacent NR panels for one MR row tile before moving to the next row
/// tile, keeping C accesses row-contiguous while bounding B-panel rereads.
const ROW_SUPERPANEL_TILES: usize = 8;
const COL_SUPERPANEL_PANELS: usize = 8;

/// Pack B's full `NR`-wide column panels into panel-major order: panel `jp`
/// (columns `jp*NR .. jp*NR+NR`) is stored as a contiguous `[k][NR]` block at
/// `bpack[jp*k*NR ..]`, so `bpack[jp*k*NR + l*NR + jj] == b[l*n + jp*NR + jj]`.
///
/// The microkernel reads one panel's k-stream sequentially (stride `NR`) instead
/// of striding B by `n` (one fresh cache line, half-used, per k-step) — that is
/// the binding constraint once `k·n` spills L2. Packing reorders *where* each B
/// value is read from, never the per-element accumulation order, so the product
/// stays bit-for-bit identical to the strided kernel. Only full panels are packed;
/// the `n % NR` column remainder is read from `b` directly. O(k·n) one-time cost,
/// amortized across the O(m·k·n) multiply.
fn pack_b_panel_range(b: &[f64], k: usize, n: usize, start_panel: usize, out: &mut [f64]) {
    let panel_elems = k * NR;
    for (offset, panel) in out.chunks_exact_mut(panel_elems).enumerate() {
        let jbase = (start_panel + offset) * NR;
        for l in 0..k {
            let src = l * n + jbase;
            panel[l * NR..l * NR + NR].copy_from_slice(&b[src..src + NR]);
        }
    }
}

fn pack_b_panels(b: &[f64], k: usize, n: usize, threads: usize) -> Vec<f64> {
    let npanels = n / NR;
    let mut bpack = vec![0.0f64; npanels * k * NR];
    if threads <= 1 || npanels <= 1 {
        pack_b_panel_range(b, k, n, 0, &mut bpack);
        return bpack;
    }

    let workers = threads.min(npanels);
    let panel_elems = k * NR;
    let panels_per_worker = npanels.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut rest = bpack.as_mut_slice();
        let mut start_panel = 0usize;
        while start_panel < npanels {
            let panels = panels_per_worker.min(npanels - start_panel);
            let elems = panels * panel_elems;
            let (chunk, tail) = rest.split_at_mut(elems);
            rest = tail;
            scope.spawn(move || pack_b_panel_range(b, k, n, start_panel, chunk));
            start_panel += panels;
        }
    });
    bpack
}

fn pack_b_pc_panel_block(b: &[f64], k: usize, n: usize, pc: usize, out: &mut [f64]) {
    let npanels = n / NR;
    let kc = KC.min(k - pc);
    let panel_elems = kc * NR;
    debug_assert_eq!(out.len(), npanels * panel_elems);
    for l in 0..kc {
        let row = &b[(pc + l) * n..(pc + l + 1) * n];
        for jp in 0..npanels {
            let src = jp * NR;
            let dst = jp * panel_elems + l * NR;
            out[dst..dst + NR].copy_from_slice(&row[src..src + NR]);
        }
    }
}

/// Pack B by KC-sized pc blocks, then NR column panels inside each block:
/// `bpack[pc*npanels*NR + jp*kc*NR + l*NR + jj] == b[(pc+l)*n + jp*NR + jj]`.
///
/// This is only for the cache-blocked macro-kernel. It keeps each pc slab's
/// column panels adjacent, cutting the stride between panels from `k*NR` to
/// `kc*NR` while preserving every output element's ascending-k recurrence.
fn pack_b_pc_panels(b: &[f64], k: usize, n: usize, threads: usize) -> Vec<f64> {
    let npanels = n / NR;
    let mut bpack = vec![0.0f64; npanels * k * NR];
    if threads <= 1 || k <= KC {
        let mut pc = 0;
        while pc < k {
            let kc = KC.min(k - pc);
            let elems = npanels * kc * NR;
            let start = pc * npanels * NR;
            pack_b_pc_panel_block(b, k, n, pc, &mut bpack[start..start + elems]);
            pc += KC;
        }
        return bpack;
    }

    let blocks = k.div_ceil(KC);
    let workers = threads.min(blocks);
    let blocks_per_worker = blocks.div_ceil(workers);
    std::thread::scope(|scope| {
        let mut rest = bpack.as_mut_slice();
        let mut block_start = 0usize;
        while block_start < blocks {
            let block_end = (block_start + blocks_per_worker).min(blocks);
            let pc_start = block_start * KC;
            let pc_end = (block_end * KC).min(k);
            let elems = (pc_end - pc_start) * npanels * NR;
            let (chunk, tail) = rest.split_at_mut(elems);
            rest = tail;
            scope.spawn(move || {
                let mut local = chunk;
                let mut pc = pc_start;
                while pc < pc_end {
                    let kc = KC.min(k - pc);
                    let elems = npanels * kc * NR;
                    let (block, tail) = local.split_at_mut(elems);
                    local = tail;
                    pack_b_pc_panel_block(b, k, n, pc, block);
                    pc += KC;
                }
            });
            block_start = block_end;
        }
    });
    bpack
}

/// `k·n` (B element count) above which B is packed into panel-major order before
/// the multiply: once B spills L2 (~1 MB f64 = 131072 elems) the strided read in
/// the microkernel misses on every k-step. Below it B is L2-resident and the pack
/// (plus its scratch allocation) does not pay — keeps the small/medium GEMM path
/// byte-for-byte and perf-for-perf unchanged.
const PACK_B_MIN_KN: usize = 1 << 17;
/// `k·n` threshold for the KC-blocked macro-kernel. Below this, same-worker
/// evidence showed the extra C read/write passes cost more than the B-panel
/// reuse saves; keep those sizes on the flat packed path.
const BLOCKED_B_MIN_KN: usize = 1 << 22;

#[derive(Clone, Copy)]
struct MatmulPlan {
    threads: usize,
    pack_b: bool,
    block_k: bool,
}

#[derive(Clone, Copy)]
struct MatmulRhs<'a> {
    b: &'a [f64],
    bpack: Option<&'a [f64]>,
    n: usize,
    block_k: bool,
}

/// Matrix multiplication as a special case of tensordot.
///
/// Matches `jnp.matmul(a, b)` for 2D arrays.
pub fn matmul_2d(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Vec<f64> {
    // Parallelize across disjoint output ROW-BLOCKS (scoped threads, no external
    // dependency, 100% safe). Thread count is budgeted by work instead of using
    // every core as soon as a fixed threshold trips: at 256³ the all-core path
    // regresses because each worker gets too little arithmetic, while a limited
    // fanout still has enough FMAs per row chunk to amortize spawn overhead.
    //
    // (pass110: single-thread NBxKB cache-blocking REGRESSED ~0.60× and 4-row M
    // register-tiling gave ~1.09× — the serial kernel is already L2-served, so
    // the remaining axis is parallelism.)
    //
    // (pass119, REJECTED: k-blocked all-core threading measured 5.82ms vs 4.6ms
    // serial at 256³; the missing primitive was not B-panel reuse, it was
    // right-sizing fanout so each worker owns enough row work.)
    let ops = m.saturating_mul(k).saturating_mul(n);
    let threads = matmul_thread_count(ops, m);
    let b_elems = k.saturating_mul(n);
    let plan = MatmulPlan {
        threads,
        pack_b: b_elems >= PACK_B_MIN_KN,
        block_k: b_elems >= BLOCKED_B_MIN_KN,
    };
    matmul_2d_with_threads(a, m, k, b, n, plan)
}

pub(crate) fn matmul_2d_into(
    a: &[f64],
    m: usize,
    k: usize,
    b: &[f64],
    n: usize,
    result: &mut [f64],
) {
    let ops = m.saturating_mul(k).saturating_mul(n);
    let threads = matmul_thread_count(ops, m);
    let b_elems = k.saturating_mul(n);
    let plan = MatmulPlan {
        threads,
        pack_b: b_elems >= PACK_B_MIN_KN,
        block_k: b_elems >= BLOCKED_B_MIN_KN,
    };
    matmul_2d_with_threads_into(a, m, k, b, n, plan, result);
}

/// Benchmark-only entry point: identical to [`matmul_2d`] but with the B-panel
/// packing forced on/off, so an A/B harness can compare both strategies in one
/// binary on one worker (the only trustworthy way to measure the pack win — cross
/// `rch` invocations vary 20-60%). Not for production callers; `matmul_2d` picks
/// `do_pack` by the `PACK_B_MIN_KN` gate.
#[doc(hidden)]
pub fn matmul_2d_with_pack(
    a: &[f64],
    m: usize,
    k: usize,
    b: &[f64],
    n: usize,
    do_pack: bool,
) -> Vec<f64> {
    let ops = m.saturating_mul(k).saturating_mul(n);
    let threads = matmul_thread_count(ops, m);
    let plan = MatmulPlan {
        threads,
        pack_b: do_pack,
        block_k: k.saturating_mul(n) >= BLOCKED_B_MIN_KN,
    };
    matmul_2d_with_threads(a, m, k, b, n, plan)
}

#[inline]
fn matmul_thread_count(ops: usize, rows: usize) -> usize {
    const MIN_PARALLEL_OPS: usize = 1 << 23; // ~8M FMAs
    const OPS_PER_THREAD: usize = 1 << 21; // ~2M FMAs/thread
    const MAX_MATMUL_THREADS: usize = 16;
    if rows <= 1 || ops < MIN_PARALLEL_OPS {
        return 1;
    }
    let available = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    let by_work = (ops / OPS_PER_THREAD).max(1);
    available
        .min(MAX_MATMUL_THREADS)
        .min(rows)
        .min(by_work)
        .max(1)
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
    plan: MatmulPlan,
) -> Vec<f64> {
    let mut result = vec![0.0; m * n];
    matmul_2d_with_threads_into(a, m, k, b, n, plan, &mut result);
    result
}

fn matmul_2d_with_threads_into(
    a: &[f64],
    m: usize,
    k: usize,
    b: &[f64],
    n: usize,
    plan: MatmulPlan,
    result: &mut [f64],
) {
    assert_eq!(result.len(), m * n);
    if m == 0 || n == 0 || k == 0 {
        result.fill(0.0);
        return;
    }
    // Pack B's column panels once (shared, read-only across all output rows) when
    // B spills L2. Bit-identical to the strided read; just panel-contiguous.
    let bpack: Option<Vec<f64>> = if plan.pack_b && n >= NR {
        Some(if plan.block_k && k > KC {
            pack_b_pc_panels(b, k, n, plan.threads)
        } else {
            pack_b_panels(b, k, n, plan.threads)
        })
    } else {
        None
    };
    let rhs = MatmulRhs {
        b,
        bpack: bpack.as_deref(),
        n,
        block_k: plan.block_k,
    };
    if plan.threads <= 1 {
        matmul_2d_dispatch(a, k, rhs, 0, result);
        return;
    }

    let rows_per = m.div_ceil(plan.threads);
    std::thread::scope(|scope| {
        let mut rest: &mut [f64] = result;
        let mut row_start = 0usize;
        while row_start < m {
            let chunk_rows = rows_per.min(m - row_start);
            let (block, tail) = rest.split_at_mut(chunk_rows * n);
            rest = tail;
            let rs = row_start;
            scope.spawn(move || matmul_2d_dispatch(a, k, rhs, rs, block));
            row_start += chunk_rows;
        }
    });
}

/// Pick the row-block kernel for one thread's slice. With B packed AND `k` deep
/// enough for the k-blocking to pay (`k > KC`, so C is re-touched only a few
/// times), use the cache-blocked macro-kernel — it keeps each [MR×KC] A-tile and
/// [KC×NR] B-panel L1-resident and reused, instead of re-streaming all of B from
/// L3 per MR-row-tile. Otherwise fall back to the flat packed/strided kernel.
#[inline]
fn matmul_2d_dispatch(
    a: &[f64],
    k: usize,
    rhs: MatmulRhs<'_>,
    row_start: usize,
    block: &mut [f64],
) {
    match rhs.bpack {
        Some(bp) if rhs.block_k && k > KC => {
            matmul_2d_blocked_row_block(a, k, rhs.b, bp, rhs.n, row_start, block);
        }
        _ => matmul_2d_row_block(a, k, rhs.b, rhs.bpack, rhs.n, row_start, block),
    }
}

/// Compute a contiguous block of output rows (starting at `row_start`,
/// `block.len() / n` rows) of the m×n product into `block`, via a
/// register-blocked MR×NR microkernel.
///
/// Each output element `C[i][j]` accumulates `a[i][l]*b[l][j]` over `l` in
/// strictly ascending order into a single scalar accumulator — bit-for-bit
/// identical to the textbook i-j-k / serial i-k-j kernels (Rust does not
/// contract separate `*`/`+=` into a fused FMA at the default release flags, so
/// the per-element rounding is exactly the same). Register tiling only reorders
/// work *across* the (i,j) tile grid; it never reassociates a single element's
/// k-sum, so `matmul_2d_ikj_bit_identical_to_ijk` and the large/threaded
/// bit-identity tests still hold.
///
/// Why this beats the previous i-k-j kernel: that kernel re-streamed the whole
/// B matrix once per output row (m passes over the k·n B reads) and re-read each
/// A element n times from memory. The MR×NR tile keeps its MR·NR C accumulators
/// in registers across the entire k-stream, reuses each loaded B element across
/// MR rows and each loaded A element across NR columns, and writes every C
/// element exactly once. That cuts B traffic ≈MR× and A traffic ≈NR×, which is
/// the binding constraint at GEMM sizes where B (k·n) spills past L2 and streams
/// from L3 on every row.
fn matmul_2d_row_block(
    a: &[f64],
    k: usize,
    b: &[f64],
    bpack: Option<&[f64]>,
    n: usize,
    row_start: usize,
    block: &mut [f64],
) {
    let rows = block.len() / n;
    let full_rows = rows / MR * MR;
    let full_cols = n / NR * NR;

    // Full MR×NR tiles. Process one B panel across every row tile in this row
    // block before advancing to the next panel. Each output element still sees
    // the identical ascending-`l` accumulation order; the only change is that a
    // packed KC×NR B stream is reused across row tiles while it is cache-hot.
    let mut j = 0;
    while j < full_cols {
        let (bsrc, rstride): (&[f64], usize) = match bpack {
            Some(bp) => (&bp[(j / NR) * k * NR..], NR),
            None => (&b[j..], n),
        };
        let mut i = 0;
        while i < full_rows {
            let ar0 = (row_start + i) * k;
            let ar1 = ar0 + k;
            let ar2 = ar1 + k;
            let ar3 = ar2 + k;
            let mut c0 = F64xN::splat(0.0);
            let mut c1 = F64xN::splat(0.0);
            let mut c2 = F64xN::splat(0.0);
            let mut c3 = F64xN::splat(0.0);
            for l in 0..k {
                let brow = &bsrc[l * rstride..l * rstride + NR];
                let bv = F64xN::from_array([
                    brow[0], brow[1], brow[2], brow[3], brow[4], brow[5], brow[6], brow[7],
                ]);
                let a0 = a[ar0 + l];
                let a1 = a[ar1 + l];
                let a2 = a[ar2 + l];
                let a3 = a[ar3 + l];
                c0 += F64xN::splat(a0) * bv;
                c1 += F64xN::splat(a1) * bv;
                c2 += F64xN::splat(a2) * bv;
                c3 += F64xN::splat(a3) * bv;
            }
            block[i * n + j..i * n + j + NR].copy_from_slice(c0.as_array());
            block[(i + 1) * n + j..(i + 1) * n + j + NR].copy_from_slice(c1.as_array());
            block[(i + 2) * n + j..(i + 2) * n + j + NR].copy_from_slice(c2.as_array());
            block[(i + 3) * n + j..(i + 3) * n + j + NR].copy_from_slice(c3.as_array());
            i += MR;
        }
        j += NR;
    }

    // Column remainder (n not a multiple of NR): MR scalar accumulators, same
    // ascending-`l` order.
    let mut i = 0;
    while i < full_rows {
        let ar0 = (row_start + i) * k;
        let ar1 = ar0 + k;
        let ar2 = ar1 + k;
        let ar3 = ar2 + k;
        let mut j = full_cols;
        while j < n {
            let mut s0 = 0.0_f64;
            let mut s1 = 0.0_f64;
            let mut s2 = 0.0_f64;
            let mut s3 = 0.0_f64;
            for l in 0..k {
                let bv = b[l * n + j];
                s0 += a[ar0 + l] * bv;
                s1 += a[ar1 + l] * bv;
                s2 += a[ar2 + l] * bv;
                s3 += a[ar3 + l] * bv;
            }
            block[i * n + j] = s0;
            block[(i + 1) * n + j] = s1;
            block[(i + 2) * n + j] = s2;
            block[(i + 3) * n + j] = s3;
            j += 1;
        }
        i += MR;
    }

    // Row remainder (rows not a multiple of MR): original i-k-j per row, which
    // is itself bit-identical ascending-`l`.
    while i < rows {
        let a_row = (row_start + i) * k;
        let c_row = &mut block[i * n..i * n + n];
        c_row.fill(0.0);
        for l in 0..k {
            let a_il = a[a_row + l];
            let src = &b[l * n..l * n + n];
            for j in 0..n {
                c_row[j] += a_il * src[j];
            }
        }
        i += 1;
    }
}

/// Cache-blocked GEMM macro-kernel for one thread's contiguous row-block, with B
/// already packed into pc-major panel order (`bpack`, see [`pack_b_pc_panels`]).
///
/// Loop order: pc (k in KC chunks) → jp (NR-col panel) → it (MR-row tile). The
/// packed B panel for `jp`'s pc-block lives at
/// `bpack[pc*npanels*NR + jp*kc*NR ..]` and is reused across every row-tile while
/// L1-resident; the [MR×KC] A tile is reused across panels. C is
/// read-modified-written once per pc-block: each pc-block
/// LOADS the running C value (or starts at 0 on the first block) and accumulates
/// its KC products in ascending `l`, so every output element is summed in the
/// exact ascending-`l` order `(((0 + p0) + p1) + …)` of the textbook kernel — the
/// pc-blocking never regroups a partial sum, so the product is bit-for-bit
/// identical to `matmul_2d_row_block` and the i-j-k reference.
///
/// The MR/NR remainder border (rows past the last full MR-tile, columns past the
/// last full NR-panel) is computed once with a single full-`k` ascending sweep —
/// also bit-identical and negligible in size — so it never enters the pc-loop.
fn matmul_2d_blocked_row_block(
    a: &[f64],
    k: usize,
    b: &[f64],
    bpack: &[f64],
    n: usize,
    row_start: usize,
    block: &mut [f64],
) {
    let rows = block.len() / n;
    let full_rows = rows - rows % MR; // rows covered by whole MR-tiles
    let full_cols = n - n % NR; // cols covered by whole NR-panels
    let row_tiles = full_rows / MR;
    let mut apack = vec![0.0_f64; row_tiles * KC * MR];

    // Full MR×NR tile region, k-blocked with C carried across pc-blocks.
    let mut pc = 0;
    while pc < k {
        let kc = KC.min(k - pc);
        let first = pc == 0;

        // Pack this thread's full-row A slab for the pc-block once, in
        // [MR-row-tile][k][row] order. The multiply below still consumes l in
        // ascending order for each output; packing only changes where A is read
        // from after the one-time copy.
        for tile in 0..row_tiles {
            let i = tile * MR;
            let ar0 = (row_start + i) * k + pc;
            let ar1 = ar0 + k;
            let ar2 = ar1 + k;
            let ar3 = ar2 + k;
            let dst = tile * KC * MR;
            for l in 0..kc {
                let base = dst + l * MR;
                apack[base] = a[ar0 + l];
                apack[base + 1] = a[ar1 + l];
                apack[base + 2] = a[ar2 + l];
                apack[base + 3] = a[ar3 + l];
            }
        }

        let col_panels = full_cols / NR;
        let mut row_super = 0;
        while row_super < row_tiles {
            let row_super_end = (row_super + ROW_SUPERPANEL_TILES).min(row_tiles);
            let mut col_super = 0;
            while col_super < col_panels {
                let col_super_end = (col_super + COL_SUPERPANEL_PANELS).min(col_panels);
                let pc_base = pc * col_panels * NR;
                let mut tile = row_super;
                while tile < row_super_end {
                    let i = tile * MR;
                    let abase = tile * KC * MR;
                    let mut jp = col_super;
                    while jp < col_super_end {
                        let j = jp * NR;
                        // This panel's pc-block: kc rows of NR, contiguous from the pack.
                        let panel = &bpack[pc_base + jp * kc * NR..];
                        // Seed accumulators from the running C (0 on the first pc-block),
                        // so the kc products continue the ascending sweep in place.
                        let mut c0 = if first {
                            F64xN::splat(0.0)
                        } else {
                            f64xnr_from_slice(&block[i * n + j..i * n + j + NR])
                        };
                        let mut c1 = if first {
                            F64xN::splat(0.0)
                        } else {
                            f64xnr_from_slice(&block[(i + 1) * n + j..(i + 1) * n + j + NR])
                        };
                        let mut c2 = if first {
                            F64xN::splat(0.0)
                        } else {
                            f64xnr_from_slice(&block[(i + 2) * n + j..(i + 2) * n + j + NR])
                        };
                        let mut c3 = if first {
                            F64xN::splat(0.0)
                        } else {
                            f64xnr_from_slice(&block[(i + 3) * n + j..(i + 3) * n + j + NR])
                        };
                        for l in 0..kc {
                            let brow = &panel[l * NR..l * NR + NR];
                            let bv = f64xnr_from_slice(brow);
                            let ap = abase + l * MR;
                            let a0 = apack[ap];
                            let a1 = apack[ap + 1];
                            let a2 = apack[ap + 2];
                            let a3 = apack[ap + 3];
                            c0 += F64xN::splat(a0) * bv;
                            c1 += F64xN::splat(a1) * bv;
                            c2 += F64xN::splat(a2) * bv;
                            c3 += F64xN::splat(a3) * bv;
                        }
                        block[i * n + j..i * n + j + NR].copy_from_slice(c0.as_array());
                        block[(i + 1) * n + j..(i + 1) * n + j + NR].copy_from_slice(c1.as_array());
                        block[(i + 2) * n + j..(i + 2) * n + j + NR].copy_from_slice(c2.as_array());
                        block[(i + 3) * n + j..(i + 3) * n + j + NR].copy_from_slice(c3.as_array());
                        jp += 1;
                    }
                    tile += 1;
                }
                col_super = col_super_end;
            }
            row_super = row_super_end;
        }
        pc += KC;
    }

    // Border, single full-k ascending sweep (bit-identical, tiny region):
    //   bottom strip = remainder rows × all cols; right strip = full rows ×
    //   remainder cols. Disjoint, together exactly the non-full-tile elements.
    for i in full_rows..rows {
        let a_row = (row_start + i) * k;
        for j in 0..n {
            let mut s = 0.0_f64;
            for l in 0..k {
                s += a[a_row + l] * b[l * n + j];
            }
            block[i * n + j] = s;
        }
    }
    for i in 0..full_rows {
        let a_row = (row_start + i) * k;
        for j in full_cols..n {
            let mut s = 0.0_f64;
            for l in 0..k {
                s += a[a_row + l] * b[l * n + j];
            }
            block[i * n + j] = s;
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
/// Contiguous i-k-j i64 matmul `[m,k]@[k,n]` with wrapping arithmetic, threaded over
/// disjoint output row-blocks. Each output element accumulates `a[i,l]*b[l,j]` over `l` in
/// strictly ascending order with `wrapping_mul`/`wrapping_add` — BIT-FOR-BIT identical to
/// dot_general's generic integer reduction (same ascending-`l` wrapping fold), but the
/// contiguous inner `j`-loop autovectorizes and replaces the generic loop's per-`l`
/// multi-index stride decode. For canonical I64 `[m,k]@[k,n]` matmul, which otherwise has
/// NO fast path and falls to that strided per-element loop.
pub fn rank2_i64_matmul(a: &[i64], m: usize, k: usize, b: &[i64], n: usize) -> Vec<i64> {
    let mut result = vec![0i64; m * n];
    if m == 0 || n == 0 || k == 0 {
        return result;
    }
    let ops = m.saturating_mul(k).saturating_mul(n);
    let threads = matmul_thread_count(ops, m);
    if threads <= 1 {
        rank2_i64_row_block(a, b, k, n, 0, &mut result);
        return result;
    }
    let rows_per = m.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut rest: &mut [i64] = result.as_mut_slice();
        let mut row_start = 0usize;
        while row_start < m {
            let chunk_rows = rows_per.min(m - row_start);
            let (block, tail) = rest.split_at_mut(chunk_rows * n);
            rest = tail;
            let rs = row_start;
            scope.spawn(move || rank2_i64_row_block(a, b, k, n, rs, block));
            row_start += chunk_rows;
        }
    });
    result
}

/// i-k-j row-block for [`rank2_i64_matmul`]: rows `[row_start, row_start+block.len()/n)`.
fn rank2_i64_row_block(
    a: &[i64],
    b: &[i64],
    k: usize,
    n: usize,
    row_start: usize,
    block: &mut [i64],
) {
    let rows = block.len() / n;
    let full = rows - rows % 4; // rows covered by whole 4-row register tiles
    let (blocked, tail) = block.split_at_mut(full * n);

    // 4-row register-blocked region: four output rows share ONE `b_row` load per
    // `l`, so B is streamed `rows/4` times instead of `rows` times (4x less
    // C-vs-B cache traffic) and the four independent wrapping MACs per b load give
    // the integer pipeline 4-way ILP. Each output still accumulates its products
    // over `l` in strictly ascending order with `wrapping_mul`/`wrapping_add`;
    // since `Z/2^64` is a commutative ring this is BIT-FOR-BIT identical to the
    // single-row loop below / the generic integer reduction — the register
    // blocking only interleaves four independent output rows, never regrouping any
    // one output's partial sum. (Mirrors the f64 `matmul_2d_blocked_row_block`
    // MR-row reuse, minus the float-reassociation concern integers don't have.)
    for (g, four) in blocked.chunks_mut(4 * n).enumerate() {
        let (c0, rest) = four.split_at_mut(n);
        let (c1, rest) = rest.split_at_mut(n);
        let (c2, c3) = rest.split_at_mut(n);
        let base = (row_start + g * 4) * k;
        let (a0o, a1o, a2o, a3o) = (base, base + k, base + 2 * k, base + 3 * k);
        for l in 0..k {
            let a0 = a[a0o + l];
            let a1 = a[a1o + l];
            let a2 = a[a2o + l];
            let a3 = a[a3o + l];
            let b_row = &b[l * n..l * n + n];
            for ((((e0, e1), e2), e3), &bv) in c0
                .iter_mut()
                .zip(c1.iter_mut())
                .zip(c2.iter_mut())
                .zip(c3.iter_mut())
                .zip(b_row)
            {
                *e0 = e0.wrapping_add(a0.wrapping_mul(bv));
                *e1 = e1.wrapping_add(a1.wrapping_mul(bv));
                *e2 = e2.wrapping_add(a2.wrapping_mul(bv));
                *e3 = e3.wrapping_add(a3.wrapping_mul(bv));
            }
        }
    }

    // Remainder rows (`rows % 4`): the original single-row i-k-j loop, unchanged.
    for (ri_rem, c_row) in tail.chunks_mut(n).enumerate() {
        let a_off = (row_start + full + ri_rem) * k;
        for l in 0..k {
            let a_il = a[a_off + l];
            let b_row = &b[l * n..l * n + n];
            for (c, &bv) in c_row.iter_mut().zip(b_row) {
                *c = c.wrapping_add(a_il.wrapping_mul(bv));
            }
        }
    }
}

/// Canonical complex `[m,k]@[k,n]` matmul over dense `(re, im)` f64 pairs. Accumulates each
/// output's `(re, im)` over `l` in strictly ascending order using the SAME arithmetic as
/// dot_general's generic complex reduction — `re += ar*br - ai*bi; im += ar*bi + ai*br`
/// (i.e. `complex_mul` followed by separate real/imag adds) — so it is BIT-FOR-BIT identical
/// while replacing the generic loop's per-element multi-index stride decode with a contiguous
/// i-k-j inner loop. Complex matmul otherwise has NO fast path (`general_real_tensordot`
/// returns `None` for complex) and falls to that strided per-element reduction. Threaded over
/// disjoint output row-blocks.
pub fn rank2_complex_matmul(
    a: &[(f64, f64)],
    m: usize,
    k: usize,
    b: &[(f64, f64)],
    n: usize,
) -> Vec<(f64, f64)> {
    let mut result = vec![(0.0f64, 0.0f64); m * n];
    if m == 0 || n == 0 || k == 0 {
        return result;
    }
    // Complex MAC is ~4 real muls + 4 real adds per step — count it as ~4x the real ops so
    // the threading threshold matches the extra arithmetic intensity.
    let ops = m.saturating_mul(k).saturating_mul(n).saturating_mul(4);
    let threads = matmul_thread_count(ops, m);
    if threads <= 1 {
        rank2_complex_row_block(a, b, k, n, 0, &mut result);
        return result;
    }
    let rows_per = m.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut rest: &mut [(f64, f64)] = result.as_mut_slice();
        let mut row_start = 0usize;
        while row_start < m {
            let chunk_rows = rows_per.min(m - row_start);
            let (block, tail) = rest.split_at_mut(chunk_rows * n);
            rest = tail;
            let rs = row_start;
            scope.spawn(move || rank2_complex_row_block(a, b, k, n, rs, block));
            row_start += chunk_rows;
        }
    });
    result
}

/// i-k-j row-block for [`rank2_complex_matmul`]: rows `[row_start, row_start+block.len()/n)`.
fn rank2_complex_row_block(
    a: &[(f64, f64)],
    b: &[(f64, f64)],
    k: usize,
    n: usize,
    row_start: usize,
    block: &mut [(f64, f64)],
) {
    for (ri, c_row) in block.chunks_mut(n).enumerate() {
        let a_off = (row_start + ri) * k;
        for l in 0..k {
            let (ar, ai) = a[a_off + l];
            let b_row = &b[l * n..l * n + n];
            for (c, &(br, bi)) in c_row.iter_mut().zip(b_row) {
                c.0 += ar * br - ai * bi;
                c.1 += ar * bi + ai * br;
            }
        }
    }
}

pub fn batched_matmul_2d(
    a: &[f64],
    batch: usize,
    m: usize,
    k: usize,
    b: &[f64],
    n: usize,
) -> Vec<f64> {
    let mut result = vec![0.0; batch * m * n];
    if batch == 0 || m == 0 || n == 0 || k == 0 {
        return result;
    }
    let total_rows = batch * m;
    let ops = total_rows.saturating_mul(k).saturating_mul(n);
    let threads = matmul_thread_count(ops, total_rows);

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

/// Batched canonical i64 matmul `[batch,m,k]@[batch,k,n]` -> `[batch,m,n]`, threaded
/// over the flattened batch×output-row space (same structure as [`batched_matmul_2d`]).
/// Each output accumulates `a[i,l]*b[l,j]` over `l` in strictly ascending order with
/// `wrapping_mul`/`wrapping_add` — BIT-FOR-BIT identical to the generic integer
/// dot_general reduction (same ascending-`l` wrapping fold per batch slice). Batched
/// i64 matmul otherwise has NO fast path and falls to the generic strided per-element
/// loop. See `batched_rank2_i64_matmul_matches_generic`.
pub fn batched_rank2_i64_matmul(
    a: &[i64],
    batch: usize,
    m: usize,
    k: usize,
    b: &[i64],
    n: usize,
) -> Vec<i64> {
    let mut result = vec![0i64; batch * m * n];
    if batch == 0 || m == 0 || n == 0 || k == 0 {
        return result;
    }
    let total_rows = batch * m;
    let ops = total_rows.saturating_mul(k).saturating_mul(n);
    let threads = matmul_thread_count(ops, total_rows);
    if threads <= 1 {
        batched_i64_row_block(a, b, m, k, n, 0, &mut result);
        return result;
    }
    let rows_per = total_rows.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut rest: &mut [i64] = result.as_mut_slice();
        let mut g_start = 0usize;
        while g_start < total_rows {
            let chunk_rows = rows_per.min(total_rows - g_start);
            let (block, tail) = rest.split_at_mut(chunk_rows * n);
            rest = tail;
            let gs = g_start;
            scope.spawn(move || batched_i64_row_block(a, b, m, k, n, gs, block));
            g_start += chunk_rows;
        }
    });
    result
}

/// i64 row-block for [`batched_rank2_i64_matmul`]: global output rows
/// `[g_start, g_start + block.len()/n)`. Global row `g` is batch `g / m`, row `g % m`.
fn batched_i64_row_block(
    a: &[i64],
    b: &[i64],
    m: usize,
    k: usize,
    n: usize,
    g_start: usize,
    block: &mut [i64],
) {
    for (ri, c_row) in block.chunks_mut(n).enumerate() {
        let g = g_start + ri;
        let bt = g / m;
        let a_off = g * k;
        let b_off = bt * k * n;
        for l in 0..k {
            let a_il = a[a_off + l];
            let src = &b[b_off + l * n..b_off + l * n + n];
            for (c, &bv) in c_row.iter_mut().zip(src) {
                *c = c.wrapping_add(a_il.wrapping_mul(bv));
            }
        }
    }
}

/// Batched canonical complex `[batch,m,k]@[batch,k,n]` -> `[batch,m,n]` over dense
/// `(re, im)` f64 pairs, threaded over the flattened batch×output-row space (same
/// structure as [`batched_matmul_2d`]). Each output accumulates its `(re, im)` over
/// `l` in strictly ascending order with the SAME `re += ar*br - ai*bi; im += ar*bi +
/// ai*br` as the generic complex reduction — BIT-FOR-BIT identical. Batched complex
/// matmul otherwise has NO fast path. See `batched_rank2_complex_matmul_matches_generic`.
pub fn batched_rank2_complex_matmul(
    a: &[(f64, f64)],
    batch: usize,
    m: usize,
    k: usize,
    b: &[(f64, f64)],
    n: usize,
) -> Vec<(f64, f64)> {
    let mut result = vec![(0.0f64, 0.0f64); batch * m * n];
    if batch == 0 || m == 0 || n == 0 || k == 0 {
        return result;
    }
    let total_rows = batch * m;
    // Complex MAC is ~4 real muls + 4 real adds; weight ops ~4x for the thread threshold.
    let ops = total_rows
        .saturating_mul(k)
        .saturating_mul(n)
        .saturating_mul(4);
    let threads = matmul_thread_count(ops, total_rows);
    if threads <= 1 {
        batched_complex_row_block(a, b, m, k, n, 0, &mut result);
        return result;
    }
    let rows_per = total_rows.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut rest: &mut [(f64, f64)] = result.as_mut_slice();
        let mut g_start = 0usize;
        while g_start < total_rows {
            let chunk_rows = rows_per.min(total_rows - g_start);
            let (block, tail) = rest.split_at_mut(chunk_rows * n);
            rest = tail;
            let gs = g_start;
            scope.spawn(move || batched_complex_row_block(a, b, m, k, n, gs, block));
            g_start += chunk_rows;
        }
    });
    result
}

/// complex row-block for [`batched_rank2_complex_matmul`]: global output rows
/// `[g_start, g_start + block.len()/n)`. Global row `g` is batch `g / m`, row `g % m`.
fn batched_complex_row_block(
    a: &[(f64, f64)],
    b: &[(f64, f64)],
    m: usize,
    k: usize,
    n: usize,
    g_start: usize,
    block: &mut [(f64, f64)],
) {
    for (ri, c_row) in block.chunks_mut(n).enumerate() {
        let g = g_start + ri;
        let bt = g / m;
        let a_off = g * k;
        let b_off = bt * k * n;
        for l in 0..k {
            let (ar, ai) = a[a_off + l];
            let src = &b[b_off + l * n..b_off + l * n + n];
            for (c, &(br, bi)) in c_row.iter_mut().zip(src) {
                c.0 += ar * br - ai * bi;
                c.1 += ar * bi + ai * br;
            }
        }
    }
}

/// Native-f32 batched matmul: f32 inputs, **f32 accumulation**, f32 output.
///
/// Reads the `f32` operands directly and keeps each scalar product in the output
/// dtype, matching the XLA/JAX f32 dot contract tracked by `frankenjax-cz0g0`.
/// Each output element still folds `l` in strictly ascending order; the only
/// intentional behavior change is accumulator precision.
pub fn batched_matmul_2d_f32_in(
    a: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
) -> Vec<f32> {
    let mut result = vec![0.0f32; batch * m * n];
    if batch == 0 || m == 0 || n == 0 || k == 0 {
        return result;
    }
    let total_rows = batch * m;
    let ops = total_rows.saturating_mul(k).saturating_mul(n);
    let threads = matmul_thread_count(ops, total_rows);

    if threads <= 1 {
        batched_matmul_row_block_f32_in(a, b, m, k, n, 0, &mut result);
        return result;
    }

    let rows_per = total_rows.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut rest: &mut [f32] = result.as_mut_slice();
        let mut g_start = 0usize;
        while g_start < total_rows {
            let chunk_rows = rows_per.min(total_rows - g_start);
            let (block, tail) = rest.split_at_mut(chunk_rows * n);
            rest = tail;
            let gs = g_start;
            scope.spawn(move || batched_matmul_row_block_f32_in(a, b, m, k, n, gs, block));
            g_start += chunk_rows;
        }
    });
    result
}

/// f32-input row-block kernel: accumulates each output row in an `f32` scratch
/// (ascending-`l`) and writes it directly into the output. The `acc` scratch is
/// reused across the block's rows to avoid per-row allocation.
fn batched_matmul_row_block_f32_in(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    g_start: usize,
    block: &mut [f32],
) {
    let full_cols = n / F32_NR;
    let tail_cols = n - full_cols * F32_NR;
    let mut acc_chunks = vec![F32xN::splat(0.0); full_cols];
    let mut acc_tail = vec![0.0f32; tail_cols];
    for (ri, c_row) in block.chunks_mut(n).enumerate() {
        let g = g_start + ri;
        let bt = g / m;
        let a_off = g * k;
        let b_off = bt * k * n;
        acc_chunks
            .iter_mut()
            .for_each(|acc| *acc = F32xN::splat(0.0));
        acc_tail.iter_mut().for_each(|x| *x = 0.0);
        for l in 0..k {
            let a_scalar = a[a_off + l];
            let a_il = F32xN::splat(a_scalar);
            let src = &b[b_off + l * n..b_off + l * n + n];
            for (chunk_idx, acc) in acc_chunks.iter_mut().enumerate() {
                let j = chunk_idx * F32_NR;
                *acc += a_il * F32xN::from_slice(&src[j..j + F32_NR]);
            }
            let tail_start = full_cols * F32_NR;
            for j in 0..tail_cols {
                acc_tail[j] += a_scalar * src[tail_start + j];
            }
        }
        for (chunk_idx, acc) in acc_chunks.iter().enumerate() {
            let j = chunk_idx * F32_NR;
            c_row[j..j + F32_NR].copy_from_slice(acc.as_array());
        }
        let tail_start = full_cols * F32_NR;
        c_row[tail_start..].copy_from_slice(&acc_tail);
    }
}

/// Mixed-precision batched matmul: **BF16 inputs, native f32 accumulation, BF16
/// output** — the BF16 sibling of [`batched_matmul_2d_f32_in`], matching XLA's bf16
/// dot (which accumulates in f32, NOT f64).
///
/// BF16 is the high 16 bits of an f32 (low mantissa bits zero), so widening a BF16 bit
/// pattern to f32 is the exact `f32::from_bits((bits as u32) << 16)` for EVERY value
/// including inf/NaN/subnormals. Reading BF16 directly avoids the promote buffer AND
/// streams B at 2 bytes/elem; accumulating in f32 (`F32_NR == 16` SIMD lanes, 2x the
/// f64 kernel) further halves the accumulator bytes and matches XLA's precision. Each
/// output rounds f32->BF16 via `round_f32_to_bf16` (round-to-nearest-even), exactly
/// XLA's bf16 dot rounding.
///
/// BIT-FOR-BIT identical to the scalar XLA-parity reference "widen BF16->f32,
/// accumulate each output ascending-`l` in f32, round f32->BF16": same widened values,
/// same accumulation order, same final round. Proven by `bf16_in_matches_f32_accum`.
/// (This restores XLA bf16 parity; the prior f64 accumulation was MORE precise than
/// the reference — the BF16 analog of the cz0g0 f32 native-accum parity fix.)
pub fn batched_matmul_2d_bf16_in(
    a: &[u16],
    batch: usize,
    m: usize,
    k: usize,
    b: &[u16],
    n: usize,
) -> Vec<u16> {
    let mut result = vec![0u16; batch * m * n];
    if batch == 0 || m == 0 || n == 0 || k == 0 {
        return result;
    }
    let total_rows = batch * m;
    let ops = total_rows.saturating_mul(k).saturating_mul(n);
    let threads = matmul_thread_count(ops, total_rows);

    if threads <= 1 {
        batched_matmul_row_block_bf16_in(a, b, m, k, n, 0, &mut result);
        return result;
    }

    let rows_per = total_rows.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut rest: &mut [u16] = result.as_mut_slice();
        let mut g_start = 0usize;
        while g_start < total_rows {
            let chunk_rows = rows_per.min(total_rows - g_start);
            let (block, tail) = rest.split_at_mut(chunk_rows * n);
            rest = tail;
            let gs = g_start;
            scope.spawn(move || batched_matmul_row_block_bf16_in(a, b, m, k, n, gs, block));
            g_start += chunk_rows;
        }
    });
    result
}

/// Widen a BF16 bit pattern to f64, identically to `Literal::BF16Bits(bits).as_f64()`.
/// BF16 occupies the high 16 bits of an f32 with the low 16 mantissa bits zero, so the
/// shift is exact for finite, subnormal, inf and NaN inputs alike.
#[inline]
fn bf16_bits_to_f64(bits: u16) -> f64 {
    f64::from(f32::from_bits((bits as u32) << 16))
}

/// Scalar reference row-block (pre-SIMD): the exact loop the SIMD kernel replaces,
/// kept only so a same-binary A/B bench can isolate the SIMD speedup.
#[doc(hidden)]
fn batched_matmul_row_block_bf16_in_scalar(
    a: &[u16],
    b: &[u16],
    m: usize,
    k: usize,
    n: usize,
    g_start: usize,
    block: &mut [u16],
) {
    let mut acc = vec![0.0f64; n];
    for (ri, c_row) in block.chunks_mut(n).enumerate() {
        let g = g_start + ri;
        let bt = g / m;
        let a_off = g * k;
        let b_off = bt * k * n;
        acc.iter_mut().for_each(|x| *x = 0.0);
        for l in 0..k {
            let a_il = bf16_bits_to_f64(a[a_off + l]);
            let src = &b[b_off + l * n..b_off + l * n + n];
            for j in 0..n {
                acc[j] += a_il * bf16_bits_to_f64(src[j]);
            }
        }
        for (cj, &av) in c_row.iter_mut().zip(acc.iter()) {
            *cj = match fj_core::Literal::from_bf16_f64(av) {
                fj_core::Literal::BF16Bits(bits) => bits,
                _ => 0,
            };
        }
    }
}

/// Bench-only single-thread BF16 matmul over one batch (`batch == 1`). `mode`
/// selects the kernel so a same-invocation A/B can isolate each lever:
///   "f32simd" — native-f32-accum SIMD (production), "f64simd" — prior f64-accum
///   SIMD (a71f4c78), "f64scalar" — original scalar f64-accum reference.
#[doc(hidden)]
pub fn bf16_matmul_bench(a: &[u16], m: usize, k: usize, b: &[u16], n: usize, mode: &str) -> Vec<u16> {
    let mut out = vec![0u16; m * n];
    match mode {
        "f64simd" => batched_matmul_row_block_bf16_in_f64acc(a, b, m, k, n, 0, &mut out),
        "f64scalar" => batched_matmul_row_block_bf16_in_scalar(a, b, m, k, n, 0, &mut out),
        _ => batched_matmul_row_block_bf16_in(a, b, m, k, n, 0, &mut out),
    }
    out
}

/// Bench-only single-batch F16 matmul A/B. `native=true` runs the production
/// native-f32-accum path ([`batched_matmul_2d_f16_in`]); `native=false` runs the
/// prior promote path (decode F16->f64, f64 [`batched_matmul_2d`], round F16) so the
/// native-f32 lever's speedup is measurable in one process.
#[doc(hidden)]
pub fn f16_matmul_bench(a: &[u16], m: usize, k: usize, b: &[u16], n: usize, native: bool) -> Vec<u16> {
    if native {
        return batched_matmul_2d_f16_in(a, 1, m, k, b, n);
    }
    let a64: Vec<f64> = a
        .iter()
        .map(|&x| fj_core::Literal::F16Bits(x).as_f64().unwrap_or(0.0))
        .collect();
    let b64: Vec<f64> = b
        .iter()
        .map(|&x| fj_core::Literal::F16Bits(x).as_f64().unwrap_or(0.0))
        .collect();
    let out64 = batched_matmul_2d(&a64, 1, m, k, &b64, n);
    out64
        .iter()
        .map(|&v| match fj_core::Literal::from_f16_f64(v) {
            fj_core::Literal::F16Bits(bits) => bits,
            _ => 0,
        })
        .collect()
}

/// BF16-input row-block kernel: accumulates each output row in an `f64` scratch
/// (ascending-`l`, widening BF16->f64 per element) then rounds to BF16. See
/// [`batched_matmul_2d_bf16_in`] for the bit-identity argument.
/// Widen `NR` contiguous BF16 bit patterns to an `F64xN` lane vector, identically
/// to `bf16_bits_to_f64` per lane (u16 -> high-16-bits-of-f32 via `<< 16` -> f64).
/// Pure integer/float SIMD casts — no comparison/mask traits (whose import paths
/// drift across nightlies; see the RNG/SIMD note).
#[inline]
fn bf16_chunk_to_f64xn(src: &[u16]) -> F64xN {
    use std::simd::{
        Simd,
        num::{SimdFloat, SimdUint},
    };
    let u = Simd::<u16, NR>::from_slice(src);
    let f32v = Simd::<f32, NR>::from_bits(u.cast::<u32>() << Simd::splat(16u32));
    f32v.cast::<f64>()
}

/// Widen a BF16 bit pattern to f32 — BF16 is the high 16 bits of an f32 (low 16
/// mantissa bits zero), so the `<< 16` shift is the EXACT decoded value for finite,
/// subnormal, inf and NaN alike (no rounding). Identically `f32::from(bf16)`.
#[inline]
fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Widen `F32_NR` contiguous BF16 patterns to an `F32xN` lane vector — a pure
/// integer SIMD shift (`u16 -> u32 << 16 -> f32::from_bits`), no f32->f64 cast and
/// no comparison/mask traits (whose import paths drift across nightlies).
#[inline]
fn bf16_chunk_to_f32xn(src: &[u16]) -> F32xN {
    use std::simd::{
        Simd,
        num::{SimdFloat, SimdUint},
    };
    let u = Simd::<u16, F32_NR>::from_slice(src);
    Simd::<f32, F32_NR>::from_bits(u.cast::<u32>() << Simd::splat(16u32))
}

/// Round an f32 BF16-matmul accumulator to BF16 bits, identically to the XLA
/// rounding `from_bf16_f64(f64::from(acc))`: `f64::from(f32)` is exact and the
/// round-to-odd f64->f32 step inside `from_bf16_f64` is then the identity, so this
/// collapses to a single round-to-nearest-even f32->BF16 (matching XLA's bf16 dot).
#[inline]
fn round_f32_to_bf16(acc: f32) -> u16 {
    match fj_core::Literal::from_bf16_f64(f64::from(acc)) {
        fj_core::Literal::BF16Bits(bits) => bits,
        _ => 0,
    }
}

/// Round an f32 F16-matmul accumulator to F16 bits, identically to the XLA
/// rounding `from_f16_f64(f64::from(acc))`: `f64::from(f32)` is exact and the
/// round-to-odd f64->f32 step inside `from_f16_f64` is then the identity, so this
/// collapses to a single round-to-nearest-even f32->F16 (matching XLA's f16 dot).
#[inline]
fn round_f32_to_f16(acc: f32) -> u16 {
    match fj_core::Literal::from_f16_f64(f64::from(acc)) {
        fj_core::Literal::F16Bits(bits) => bits,
        _ => 0,
    }
}

/// Mixed-precision batched matmul: **F16 inputs, native f32 accumulation, F16
/// output** — the F16 sibling of [`batched_matmul_2d_bf16_in`], matching XLA's f16
/// dot (which accumulates in f32, NOT f64; fj's f64-promote path was MORE precise
/// than the reference).
///
/// Unlike BF16, an F16 value is NOT the high bits of an f32 (5-bit vs 8-bit
/// exponent), so the widen is the canonical `f16 -> f32` decode (`as_f16_f32` =
/// `f32::from(f16::from_bits(_))`). We decode both operands into f32 buffers ONCE
/// (O(mk + kn), negligible beside the O(batch·m·k·n) GEMM) and reuse the optimized
/// native-f32 [`batched_matmul_2d_f32_in`] (16-lane SIMD, ascending-`l` f32
/// accumulation), then round each output f32 -> F16.
///
/// BIT-FOR-BIT identical to the scalar XLA-parity reference "decode F16->f32,
/// accumulate each output ascending-`l` in f32, round f32->F16": `batched_matmul_2d_f32_in`
/// is itself bit-identical to that scalar f32 fold (the cz0g0 contract), and the
/// decode/round are elementwise. Proven by `f16_in_matches_f32_accum`.
pub fn batched_matmul_2d_f16_in(
    a: &[u16],
    batch: usize,
    m: usize,
    k: usize,
    b: &[u16],
    n: usize,
) -> Vec<u16> {
    let decode = |bits: u16| -> f32 {
        fj_core::Literal::F16Bits(bits)
            .as_f16_f32()
            .unwrap_or(0.0)
    };
    let a32: Vec<f32> = a.iter().map(|&x| decode(x)).collect();
    let b32: Vec<f32> = b.iter().map(|&x| decode(x)).collect();
    let out32 = batched_matmul_2d_f32_in(&a32, batch, m, k, &b32, n);
    out32.iter().map(|&v| round_f32_to_f16(v)).collect()
}

/// Native-f32-accumulation BF16 GEMM row-block (XLA parity: XLA accumulates bf16
/// matmul in f32, NOT f64 — fj's earlier f64 accumulation diverged by being MORE
/// precise than the reference). SIMD across OUTPUT COLUMNS (lanes = independent
/// outputs), each lane folding its own ascending-`l` f32 multiply-add — bit-identical
/// to the scalar f32-accum reference (same per-output order, same f32 mul/add, same
/// `round_f32_to_bf16`). `F32_NR == 16` lanes (2x the f64-accum kernel) and the
/// bf16->f32 widen is a bare shift (no f32->f64 cast), so this is the analog of the
/// approved cz0g0 native-f32 dot_general lever. See `bf16_in_matches_f32_accum`.
fn batched_matmul_row_block_bf16_in(
    a: &[u16],
    b: &[u16],
    m: usize,
    k: usize,
    n: usize,
    g_start: usize,
    block: &mut [u16],
) {
    let full_cols = n / F32_NR;
    let tail_cols = n - full_cols * F32_NR;
    let mut acc_chunks = vec![F32xN::splat(0.0); full_cols];
    let mut acc_tail = vec![0.0f32; tail_cols];
    for (ri, c_row) in block.chunks_mut(n).enumerate() {
        let g = g_start + ri;
        let bt = g / m;
        let a_off = g * k;
        let b_off = bt * k * n;
        acc_chunks.iter_mut().for_each(|acc| *acc = F32xN::splat(0.0));
        acc_tail.iter_mut().for_each(|x| *x = 0.0);
        for l in 0..k {
            let a_scalar = bf16_bits_to_f32(a[a_off + l]);
            let a_il = F32xN::splat(a_scalar);
            let src = &b[b_off + l * n..b_off + l * n + n];
            for (chunk_idx, acc) in acc_chunks.iter_mut().enumerate() {
                let j = chunk_idx * F32_NR;
                *acc += a_il * bf16_chunk_to_f32xn(&src[j..j + F32_NR]);
            }
            let tail_start = full_cols * F32_NR;
            for j in 0..tail_cols {
                acc_tail[j] += a_scalar * bf16_bits_to_f32(src[tail_start + j]);
            }
        }
        for (chunk_idx, acc) in acc_chunks.iter().enumerate() {
            let j = chunk_idx * F32_NR;
            for (lane, &av) in acc.as_array().iter().enumerate() {
                c_row[j + lane] = round_f32_to_bf16(av);
            }
        }
        let tail_start = full_cols * F32_NR;
        for (j, &av) in acc_tail.iter().enumerate() {
            c_row[tail_start + j] = round_f32_to_bf16(av);
        }
    }
}

/// Prior f64-accumulation SIMD row-block (the a71f4c78 kernel), kept only as a
/// same-binary A/B reference so the native-f32 lever's speedup is measurable.
#[doc(hidden)]
fn batched_matmul_row_block_bf16_in_f64acc(
    a: &[u16],
    b: &[u16],
    m: usize,
    k: usize,
    n: usize,
    g_start: usize,
    block: &mut [u16],
) {
    let full_cols = n / NR;
    let tail_cols = n - full_cols * NR;
    let mut acc_chunks = vec![F64xN::splat(0.0); full_cols];
    let mut acc_tail = vec![0.0f64; tail_cols];
    for (ri, c_row) in block.chunks_mut(n).enumerate() {
        let g = g_start + ri;
        let bt = g / m;
        let a_off = g * k;
        let b_off = bt * k * n;
        acc_chunks.iter_mut().for_each(|acc| *acc = F64xN::splat(0.0));
        acc_tail.iter_mut().for_each(|x| *x = 0.0);
        for l in 0..k {
            let a_scalar = bf16_bits_to_f64(a[a_off + l]);
            let a_il = F64xN::splat(a_scalar);
            let src = &b[b_off + l * n..b_off + l * n + n];
            for (chunk_idx, acc) in acc_chunks.iter_mut().enumerate() {
                let j = chunk_idx * NR;
                *acc += a_il * bf16_chunk_to_f64xn(&src[j..j + NR]);
            }
            let tail_start = full_cols * NR;
            for j in 0..tail_cols {
                acc_tail[j] += a_scalar * bf16_bits_to_f64(src[tail_start + j]);
            }
        }
        let round = |av: f64| -> u16 {
            match fj_core::Literal::from_bf16_f64(av) {
                fj_core::Literal::BF16Bits(bits) => bits,
                _ => 0,
            }
        };
        for (chunk_idx, acc) in acc_chunks.iter().enumerate() {
            let j = chunk_idx * NR;
            for (lane, &av) in acc.as_array().iter().enumerate() {
                c_row[j + lane] = round(av);
            }
        }
        let tail_start = full_cols * NR;
        for (j, &av) in acc_tail.iter().enumerate() {
            c_row[tail_start + j] = round(av);
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
    fn matmul_2d_into_matches_allocating_path() {
        let (m, k, n) = (11usize, 17usize, 13usize);
        let a: Vec<f64> = (0..m * k)
            .map(|i| (i as f64 * 0.019_31).sin() * 1.7 - 0.2)
            .collect();
        let b: Vec<f64> = (0..k * n)
            .map(|i| (i as f64 * 0.027_73).cos() * 2.1 + 0.4)
            .collect();

        let want = matmul_2d(&a, m, k, &b, n);
        let mut got = vec![f64::NAN; m * n];
        matmul_2d_into(&a, m, k, &b, n, &mut got);

        assert_eq!(got.len(), want.len());
        for idx in 0..got.len() {
            assert_eq!(got[idx].to_bits(), want[idx].to_bits(), "mismatch at {idx}");
        }
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
    fn matmul_2d_special_values_bit_identical_to_ijk() {
        // Diagnostic for the conv dense-vs-literal regression: matmul_2d must equal
        // the textbook ascending-l accumulation bit-for-bit even with -0.0/±Inf/NaN
        // present (the conv tests inject these). m=8,k=12,n=8 exercises one full
        // MR×NR SIMD panel with no column remainder and k<KC unblocked.
        let (m, k, n) = (8usize, 12usize, 8usize);
        let special = |i: usize| -> f64 {
            match i % 11 {
                0 => -0.0,
                1 => 0.0,
                2 => f64::INFINITY,
                3 => f64::NEG_INFINITY,
                4 => 1.0e308,
                _ => (i as f64 * 0.013).sin() * 2.0,
            }
        };
        let a: Vec<f64> = (0..m * k).map(special).collect();
        let b: Vec<f64> = (0..k * n).map(|i| special(i + 5)).collect();

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
            assert_eq!(
                got[idx].to_bits(),
                want[idx].to_bits(),
                "mismatch at {idx}: got {} want {}",
                got[idx],
                want[idx]
            );
        }
    }

    #[test]
    fn batched_matmul_2d_bit_identical() {
        // Batched matmul kernel must equal the textbook per-batch ascending-l
        // reference bit-for-bit.
        let (bt, m, k, n) = (3usize, 5usize, 7usize, 4usize);
        let a: Vec<f64> = (0..bt * m * k)
            .map(|i| (i as f64 * 0.021).sin() * 2.0 - 0.5)
            .collect();
        let b: Vec<f64> = (0..bt * k * n)
            .map(|i| (i as f64 * 0.029).cos() * 1.6 + 0.3)
            .collect();
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

    /// The f32-input GEMM must be BIT-FOR-BIT identical to a native-f32
    /// ascending-`l` reference. Sized large enough to exercise the threaded path.
    #[test]
    fn batched_matmul_2d_f32_in_matches_native_f32_bits() {
        let (bt, m, k, n) = (2usize, 40usize, 33usize, 17usize);
        let af: Vec<f32> = (0..bt * m * k)
            .map(|i| (i as f32 * 0.013).sin() * 2.0 - 0.5)
            .collect();
        let bf: Vec<f32> = (0..bt * k * n)
            .map(|i| (i as f32 * 0.019).cos() * 1.6 + 0.3)
            .collect();
        let got = batched_matmul_2d_f32_in(&af, bt, m, k, &bf, n);
        assert_eq!(got.len(), bt * m * n);
        for batch in 0..bt {
            for i in 0..m {
                for j in 0..n {
                    let mut want = 0.0f32;
                    for l in 0..k {
                        want += af[(batch * m + i) * k + l] * bf[(batch * k + l) * n + j];
                    }
                    let idx = (batch * m + i) * n + j;
                    assert_eq!(
                        got[idx].to_bits(),
                        want.to_bits(),
                        "mismatch at batch={batch}, row={i}, col={j}"
                    );
                }
            }
        }
    }

    /// The native-f32 path intentionally gives up the old f64-accum self-golden,
    /// but the delta stays inside the documented f32 matmul tolerance envelope.
    #[test]
    fn batched_matmul_2d_f32_in_delta_from_f64_accum_is_bounded() {
        let (bt, m, k, n) = (2usize, 4usize, 257usize, 8usize);
        let af: Vec<f32> = (0..bt * m * k)
            .map(|i| (i as f32 * 0.000_7).sin() * 1.3 - 0.2)
            .collect();
        let bf: Vec<f32> = (0..bt * k * n)
            .map(|i| (i as f32 * 0.001_1).cos() * 0.9 + 0.1)
            .collect();
        let got = batched_matmul_2d_f32_in(&af, bt, m, k, &bf, n);
        let mut any_differ = false;
        let mut max_rel = 0.0f64;
        for batch in 0..bt {
            for i in 0..m {
                for j in 0..n {
                    let mut f64_sum = 0.0f64;
                    for l in 0..k {
                        f64_sum += f64::from(af[(batch * m + i) * k + l])
                            * f64::from(bf[(batch * k + l) * n + j]);
                    }
                    let idx = (batch * m + i) * n + j;
                    let old = f64_sum as f32;
                    any_differ |= got[idx].to_bits() != old.to_bits();
                    if f64_sum.abs() > 1e-6 {
                        max_rel = max_rel
                            .max((f64::from(got[idx]) - f64::from(old)).abs() / f64_sum.abs());
                    }
                }
            }
        }
        assert!(
            any_differ,
            "K=257 should expose f32 vs f64 accumulation drift"
        );
        assert!(
            max_rel < 1e-3,
            "f32 vs f64 accumulation relative delta {max_rel:e}"
        );
    }

    #[test]
    fn batched_matmul_2d_f32_native_golden_digest() -> Result<(), Box<dyn std::error::Error>> {
        let (bt, m, k, n) = (2usize, 7usize, 19usize, 5usize);
        let af: Vec<f32> = (0..bt * m * k)
            .map(|i| (i as f32 * 0.013).sin() * 1.7 - 0.3)
            .collect();
        let bf: Vec<f32> = (0..bt * k * n)
            .map(|i| (i as f32 * 0.019).cos() * 1.3 + 0.2)
            .collect();
        let got = batched_matmul_2d_f32_in(&af, bt, m, k, &bf, n);
        let bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
        let digest = fj_test_utils::fixture_id_from_json(&bits)?;
        assert_eq!(
            digest, "02399fb13d6e0643dc9d8ade2c1dd2ce7cb985e38dcd41513cf80e438c0e54c8",
            "native-f32 matmul golden output digest changed"
        );
        Ok(())
    }

    /// Native BF16-input GEMM must be bit-for-bit identical to the XLA-parity
    /// reference: promote both operands BF16->f32 (exact `<< 16` widen), accumulate
    /// each output in f32 ascending-`l`, then round f32->BF16 (round-to-nearest-even).
    /// XLA accumulates bf16 dot in f32 — the kernel's native-f32 accumulation matches
    /// that exactly (the prior f64 accumulation was MORE precise than XLA). Sized large
    /// enough to exercise the threaded path.
    #[test]
    fn bf16_in_matches_f32_accum() {
        let (bt, m, k, n) = (2usize, 40usize, 33usize, 17usize);
        let to_bf16 = |v: f64| -> u16 {
            match fj_core::Literal::from_bf16_f64(v) {
                fj_core::Literal::BF16Bits(b) => b,
                _ => 0,
            }
        };
        let a16: Vec<u16> = (0..bt * m * k)
            .map(|i| to_bf16((i as f64 * 0.013).sin() * 2.0 - 0.5))
            .collect();
        let b16: Vec<u16> = (0..bt * k * n)
            .map(|i| to_bf16((i as f64 * 0.019).cos() * 1.6 + 0.3))
            .collect();
        let got = batched_matmul_2d_bf16_in(&a16, bt, m, k, &b16, n);
        // Reference: BF16->f32 widen, ascending-`l` f32 accumulation per output, round
        // f32->BF16 (exactly what XLA's bf16 dot does).
        let mut want = vec![0u16; bt * m * n];
        for batch in 0..bt {
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for l in 0..k {
                        let av = bf16_bits_to_f32(a16[batch * m * k + i * k + l]);
                        let bv = bf16_bits_to_f32(b16[batch * k * n + l * n + j]);
                        acc += av * bv;
                    }
                    want[batch * m * n + i * n + j] = round_f32_to_bf16(acc);
                }
            }
        }
        assert_eq!(got.len(), want.len());
        for idx in 0..got.len() {
            assert_eq!(got[idx], want[idx], "mismatch at {idx}");
        }
    }

    /// The native-f32 accumulation must stay within bf16 tolerance of the old
    /// f64-accumulation path (i.e. the precision given up to match XLA is bounded —
    /// ~sqrt(K)*eps_f32, far inside bf16's ~1/256 resolution).
    #[test]
    fn bf16_f32_accum_within_tolerance_of_f64_accum() {
        let (m, k, n) = (24usize, 50usize, 24usize);
        let to_bf16 = |v: f64| -> u16 {
            match fj_core::Literal::from_bf16_f64(v) {
                fj_core::Literal::BF16Bits(b) => b,
                _ => 0,
            }
        };
        let a16: Vec<u16> = (0..m * k)
            .map(|i| to_bf16((i as f64 * 0.017).sin()))
            .collect();
        let b16: Vec<u16> = (0..k * n)
            .map(|i| to_bf16((i as f64 * 0.023).cos()))
            .collect();
        let f32acc = bf16_matmul_bench(&a16, m, k, &b16, n, "f32simd");
        let f64acc = bf16_matmul_bench(&a16, m, k, &b16, n, "f64scalar");
        for (x, y) in f32acc.iter().zip(f64acc.iter()) {
            let xv = bf16_bits_to_f32(*x);
            let yv = bf16_bits_to_f32(*y);
            let diff = (xv - yv).abs();
            // bf16 has a ~7-bit mantissa; allow up to a couple of bf16 ULP-scale steps.
            assert!(diff <= 0.06 * yv.abs().max(1.0), "f32 vs f64 accum diverged: {xv} vs {yv}");
        }
    }

    /// Native F16-input GEMM must be bit-for-bit identical to the XLA-parity
    /// reference: decode both operands F16->f32, accumulate each output ascending-`l`
    /// in f32, then round f32->F16 (round-to-nearest-even). XLA accumulates f16 dot in
    /// f32 — the kernel's native-f32 accumulation matches that (the old f64-promote
    /// path was MORE precise than the reference).
    #[test]
    fn f16_in_matches_f32_accum() {
        let (bt, m, k, n) = (2usize, 40usize, 33usize, 17usize);
        let to_f16 = |v: f64| -> u16 {
            match fj_core::Literal::from_f16_f64(v) {
                fj_core::Literal::F16Bits(b) => b,
                _ => 0,
            }
        };
        let decode = |bits: u16| -> f32 { fj_core::Literal::F16Bits(bits).as_f16_f32().unwrap() };
        let a16: Vec<u16> = (0..bt * m * k)
            .map(|i| to_f16((i as f64 * 0.013).sin() * 2.0 - 0.5))
            .collect();
        let b16: Vec<u16> = (0..bt * k * n)
            .map(|i| to_f16((i as f64 * 0.019).cos() * 1.6 + 0.3))
            .collect();
        let got = batched_matmul_2d_f16_in(&a16, bt, m, k, &b16, n);
        // Reference: F16->f32 decode, ascending-`l` f32 accumulation per output, round
        // f32->F16 (exactly what XLA's f16 dot does).
        let mut want = vec![0u16; bt * m * n];
        for batch in 0..bt {
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for l in 0..k {
                        let av = decode(a16[batch * m * k + i * k + l]);
                        let bv = decode(b16[batch * k * n + l * n + j]);
                        acc += av * bv;
                    }
                    want[batch * m * n + i * n + j] = round_f32_to_f16(acc);
                }
            }
        }
        assert_eq!(got.len(), want.len());
        for idx in 0..got.len() {
            assert_eq!(got[idx], want[idx], "mismatch at {idx}");
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_f32_gemm_native_vs_promote() {
        use std::time::Instant;
        let time = |f: &dyn Fn()| {
            f();
            let mut best = f64::MAX;
            for _ in 0..6 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        // Sweep L3-resident -> RAM-bound B: the naive row-block kernel re-streams
        // all of B once per output row, so once B (k·n) spills cache the f32 path's
        // HALF-bytes-of-B advantage grows.
        for &(m, k, n) in &[
            (4096usize, 512usize, 512usize), // B=1MB f32 / 2MB f64 (L3-resident)
            (2048, 2048, 2048),              // B=16MB f32 / 32MB f64 (spills L3)
        ] {
            let af: Vec<f32> = (0..m * k).map(|i| (i % 100) as f32 * 0.01 - 0.5).collect();
            let bf: Vec<f32> = (0..k * n).map(|i| (i % 77) as f32 * 0.01).collect();
            // promote path: f32->f64 alloc+copy, f64 GEMM, round output as f32.
            let promote = time(&|| {
                let a64: Vec<f64> = af.iter().map(|&v| v as f64).collect();
                let b64: Vec<f64> = bf.iter().map(|&v| v as f64).collect();
                let out = batched_matmul_2d(&a64, 1, m, k, &b64, n);
                let _: Vec<f32> = out.iter().map(|&v| v as f32).collect();
            });
            let native = time(&|| {
                let _ = batched_matmul_2d_f32_in(&af, 1, m, k, &bf, n);
            });
            let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
            println!(
                "BENCH f32 GEMM [{m},{k}]@[{k},{n}]: promote+f64+round={:.3}ms ({:.1} GFLOP/s) native-f32-in={:.3}ms ({:.1} GFLOP/s) speedup={:.2}x",
                promote * 1e3,
                gflop / promote,
                native * 1e3,
                gflop / native,
                promote / native
            );
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_bf16_gemm_native_vs_promote() {
        use std::time::Instant;
        let time = |f: &dyn Fn()| {
            f();
            let mut best = f64::MAX;
            for _ in 0..6 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let to_bf16 = |v: f64| -> u16 {
            match fj_core::Literal::from_bf16_f64(v) {
                fj_core::Literal::BF16Bits(b) => b,
                _ => 0,
            }
        };
        // BF16 B streams at a QUARTER of f64's bytes, so the advantage grows once B
        // (k·n) spills cache (the row-block kernel re-streams all of B per output row).
        for &(m, k, n) in &[
            (4096usize, 512usize, 512usize), // B=0.5MB bf16 / 2MB f64
            (2048, 2048, 2048),              // B=8MB bf16 / 32MB f64 (spills L3)
        ] {
            let a16: Vec<u16> = (0..m * k)
                .map(|i| to_bf16((i % 100) as f64 * 0.01 - 0.5))
                .collect();
            let b16: Vec<u16> = (0..k * n)
                .map(|i| to_bf16((i % 77) as f64 * 0.01))
                .collect();
            // promote path: BF16->f64 alloc+copy, f64 GEMM, round each output to BF16.
            let promote = time(&|| {
                let a64: Vec<f64> = a16
                    .iter()
                    .map(|&b| fj_core::Literal::BF16Bits(b).as_f64().unwrap())
                    .collect();
                let b64: Vec<f64> = b16
                    .iter()
                    .map(|&b| fj_core::Literal::BF16Bits(b).as_f64().unwrap())
                    .collect();
                let out = batched_matmul_2d(&a64, 1, m, k, &b64, n);
                let _: Vec<u16> = out.iter().map(|&v| to_bf16(v)).collect();
            });
            let native = time(&|| {
                let _ = batched_matmul_2d_bf16_in(&a16, 1, m, k, &b16, n);
            });
            let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
            println!(
                "BENCH bf16 GEMM [{m},{k}]@[{k},{n}]: promote+f64+round={:.3}ms ({:.1} GFLOP/s) native-bf16-in={:.3}ms ({:.1} GFLOP/s) speedup={:.2}x",
                promote * 1e3,
                gflop / promote,
                native * 1e3,
                gflop / native,
                promote / native
            );
        }
    }

    #[test]
    fn rank2_i64_matmul_matches_generic() {
        // The contiguous i64 kernel must equal a direct ascending-`l` wrapping reference
        // (the same fold dot_general's generic integer reduction does), including overflow
        // wrapping and MR/NR remainder dims.
        // Shapes deliberately hit every `rows % 4` remainder against the 4-row
        // register-blocked kernel: 13→rem1, 64→rem0, 1→all-remainder, 40→rem0,
        // 6→rem2, 7→rem3.
        for &(m, k, n) in &[
            (13usize, 17usize, 11usize),
            (64, 48, 40),
            (1, 33, 1),
            (40, 1, 7),
            (6, 5, 9),
            (7, 9, 5),
        ] {
            let a: Vec<i64> = (0..m * k)
                .map(|i| (i as i64).wrapping_mul(2_654_435_761).wrapping_sub(7))
                .collect();
            let b: Vec<i64> = (0..k * n)
                .map(|i| (i as i64).wrapping_mul(40_503).wrapping_add(3))
                .collect();
            let got = rank2_i64_matmul(&a, m, k, &b, n);
            let mut want = vec![0i64; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0i64;
                    for l in 0..k {
                        s = s.wrapping_add(a[i * k + l].wrapping_mul(b[l * n + j]));
                    }
                    want[i * n + j] = s;
                }
            }
            assert_eq!(got, want, "[{m},{k}]@[{k},{n}]");
        }
    }

    #[test]
    fn rank2_complex_matmul_matches_generic() {
        // The contiguous complex kernel must be BIT-FOR-BIT identical to the generic complex
        // reduction: per output, ascending-`l` `complex_mul` ((ar*br-ai*bi, ar*bi+ai*br)) with
        // separate real/imag adds. Compare bit patterns (to_bits) so NaN/sign/rounding can't
        // hide a divergence. Includes MR/NR remainder dims.
        for &(m, k, n) in &[
            (13usize, 17usize, 11usize),
            (64, 48, 40),
            (1, 33, 1),
            (40, 1, 7),
        ] {
            let a: Vec<(f64, f64)> = (0..m * k)
                .map(|i| ((i as f64) * 0.5 - 3.0, (i as f64) * -0.25 + 1.0))
                .collect();
            let b: Vec<(f64, f64)> = (0..k * n)
                .map(|i| ((i as f64) * -0.125 + 2.0, (i as f64) * 0.375 - 1.5))
                .collect();
            let got = rank2_complex_matmul(&a, m, k, &b, n);
            let mut want = vec![(0.0f64, 0.0f64); m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut re = 0.0f64;
                    let mut im = 0.0f64;
                    for l in 0..k {
                        let (ar, ai) = a[i * k + l];
                        let (br, bi) = b[l * n + j];
                        re += ar * br - ai * bi;
                        im += ar * bi + ai * br;
                    }
                    want[i * n + j] = (re, im);
                }
            }
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(
                    (g.0.to_bits(), g.1.to_bits()),
                    (w.0.to_bits(), w.1.to_bits()),
                    "[{m},{k}]@[{k},{n}]"
                );
            }
        }
    }

    #[test]
    fn batched_rank2_i64_matmul_matches_generic() {
        // The batched i64 kernel must equal a direct per-batch ascending-`l` wrapping
        // reference (the same fold the generic integer dot_general reduction does),
        // including overflow wrapping, across batch counts that exercise the threaded
        // and serial paths.
        for &(bt, m, k, n) in &[
            (1usize, 13usize, 17usize, 11usize),
            (4, 9, 7, 5),
            (256, 6, 8, 6),
            (3, 1, 33, 1),
        ] {
            let a: Vec<i64> = (0..bt * m * k)
                .map(|i| (i as i64).wrapping_mul(2_654_435_761).wrapping_sub(7))
                .collect();
            let b: Vec<i64> = (0..bt * k * n)
                .map(|i| (i as i64).wrapping_mul(40_503).wrapping_add(3))
                .collect();
            let got = batched_rank2_i64_matmul(&a, bt, m, k, &b, n);
            let mut want = vec![0i64; bt * m * n];
            for t in 0..bt {
                for i in 0..m {
                    for j in 0..n {
                        let mut s = 0i64;
                        for l in 0..k {
                            s = s.wrapping_add(
                                a[(t * m + i) * k + l].wrapping_mul(b[(t * k + l) * n + j]),
                            );
                        }
                        want[(t * m + i) * n + j] = s;
                    }
                }
            }
            assert_eq!(got, want, "batch={bt} [{m},{k}]@[{k},{n}]");
        }
    }

    #[test]
    fn batched_rank2_complex_matmul_matches_generic() {
        // The batched complex kernel must be BIT-FOR-BIT identical to a direct per-batch
        // ascending-`l` complex reference (complex_mul + separate real/imag adds), across
        // batch counts exercising the threaded and serial paths. Compare bit patterns.
        for &(bt, m, k, n) in &[
            (1usize, 13usize, 17usize, 11usize),
            (4, 9, 7, 5),
            (256, 6, 8, 6),
            (3, 1, 33, 1),
        ] {
            let a: Vec<(f64, f64)> = (0..bt * m * k)
                .map(|i| ((i as f64) * 0.5 - 3.0, (i as f64) * -0.25 + 1.0))
                .collect();
            let b: Vec<(f64, f64)> = (0..bt * k * n)
                .map(|i| ((i as f64) * -0.125 + 2.0, (i as f64) * 0.375 - 1.5))
                .collect();
            let got = batched_rank2_complex_matmul(&a, bt, m, k, &b, n);
            let mut want = vec![(0.0f64, 0.0f64); bt * m * n];
            for t in 0..bt {
                for i in 0..m {
                    for j in 0..n {
                        let mut re = 0.0f64;
                        let mut im = 0.0f64;
                        for l in 0..k {
                            let (ar, ai) = a[(t * m + i) * k + l];
                            let (br, bi) = b[(t * k + l) * n + j];
                            re += ar * br - ai * bi;
                            im += ar * bi + ai * br;
                        }
                        want[(t * m + i) * n + j] = (re, im);
                    }
                }
            }
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(
                    (g.0.to_bits(), g.1.to_bits()),
                    (w.0.to_bits(), w.1.to_bits()),
                    "batch={bt} [{m},{k}]@[{k},{n}]"
                );
            }
        }
    }

    #[test]
    fn batched_matmul_2d_batch1_matches_matmul_2d() {
        // dot_general routes batch==1 f64 contractions to the packed matmul_2d; it must be
        // bit-for-bit identical to the naive batched_matmul_2d(batch=1) it replaces (both
        // are i-j-k-order references). Includes MR/NR remainder dims.
        for &(m, k, n) in &[(13usize, 17usize, 11usize), (64, 48, 40), (33, 31, 9)] {
            let a: Vec<f64> = (0..m * k)
                .map(|i| (i as f64 * 0.019).sin() * 3.0 - 0.7)
                .collect();
            let b: Vec<f64> = (0..k * n)
                .map(|i| (i as f64 * 0.023).cos() * 1.9 + 0.2)
                .collect();
            let packed = matmul_2d(&a, m, k, &b, n);
            let naive = batched_matmul_2d(&a, 1, m, k, &b, n);
            assert_eq!(packed.len(), naive.len());
            for idx in 0..packed.len() {
                assert_eq!(
                    packed[idx].to_bits(),
                    naive[idx].to_bits(),
                    "mismatch at {idx} for [{m},{k}]@[{k},{n}]"
                );
            }
        }
    }

    #[test]
    #[ignore = "perf benchmark; run explicitly"]
    fn bench_f64_batch1_packed_vs_naive() {
        use std::time::Instant;
        let time = |f: &dyn Fn()| {
            f();
            let mut best = f64::MAX;
            for _ in 0..6 {
                let t = Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        // batch==1 f64 contraction (e.g. transposed/non-canonical matmul reaching
        // general_real_tensordot): packed matmul_2d vs the naive row-block it replaces. The
        // packed win grows once B (k·n) spills cache.
        for &(m, k, n) in &[
            (4096usize, 512usize, 512usize), // B=2MB f64 (L3-resident)
            (2048, 2048, 2048),              // B=32MB f64 (at L3 edge)
            (3072, 3072, 3072),              // B=72MB f64 (clearly RAM-bound)
        ] {
            let a: Vec<f64> = (0..m * k).map(|i| (i % 100) as f64 * 0.01 - 0.5).collect();
            let b: Vec<f64> = (0..k * n).map(|i| (i % 77) as f64 * 0.01).collect();
            let naive = time(&|| {
                let _ = batched_matmul_2d(&a, 1, m, k, &b, n);
            });
            let packed = time(&|| {
                let _ = matmul_2d(&a, m, k, &b, n);
            });
            let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
            println!(
                "BENCH f64 batch1 GEMM [{m},{k}]@[{k},{n}]: naive-row-block={:.3}ms ({:.1} GFLOP/s) packed={:.3}ms ({:.1} GFLOP/s) speedup={:.2}x",
                naive * 1e3,
                gflop / naive,
                packed * 1e3,
                gflop / packed,
                naive / packed
            );
        }
    }

    #[test]
    fn matmul_2d_threaded_bit_identical() {
        // The multi-threaded row-block driver must equal the serial kernel
        // bit-for-bit, including a thread count that exceeds the row count (so
        // some threads get empty/partial blocks) and a partial last block.
        let (m, k, n) = (13usize, 17usize, 11usize);
        let a: Vec<f64> = (0..m * k)
            .map(|i| (i as f64 * 0.019).sin() * 3.0 - 0.7)
            .collect();
        let b: Vec<f64> = (0..k * n)
            .map(|i| (i as f64 * 0.023).cos() * 1.9 + 0.2)
            .collect();
        let serial = super::matmul_2d_with_threads(
            &a,
            m,
            k,
            &b,
            n,
            super::MatmulPlan {
                threads: 1,
                pack_b: false,
                block_k: false,
            },
        );
        // Equal across every thread count AND both pack strategies (packing and
        // row-block threading are each bit-identity-preserving, so all combos agree).
        for threads in [2usize, 3, 4, 8, 16, 32] {
            for do_pack in [false, true] {
                let parallel = super::matmul_2d_with_threads(
                    &a,
                    m,
                    k,
                    &b,
                    n,
                    super::MatmulPlan {
                        threads,
                        pack_b: do_pack,
                        block_k: false,
                    },
                );
                assert_eq!(serial.len(), parallel.len());
                for idx in 0..serial.len() {
                    assert_eq!(
                        serial[idx].to_bits(),
                        parallel[idx].to_bits(),
                        "threads={threads} do_pack={do_pack} mismatch at {idx}"
                    );
                }
            }
        }
    }

    #[test]
    fn matmul_thread_count_adapts_medium_gemm_without_all_core_fanout() {
        let available = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        assert_eq!(super::matmul_thread_count(1024, 256), 1);

        let medium = super::matmul_thread_count(256 * 256 * 256, 256);
        assert!(medium <= available.min(256));
        if available > 1 {
            assert!(
                medium > 1,
                "256^3 should use limited fanout on multi-core workers"
            );
        }

        let large = super::matmul_thread_count(512 * 512 * 512, 512);
        assert!(large >= medium);
        assert!(large <= available.min(512));
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
    fn matmul_2d_packed_bit_identical_to_ijk() {
        // Dims chosen so k·n = 500·290 = 145000 ≥ PACK_B_MIN_KN (131072): this
        // exercises the B-panel-packed microkernel path. m=70 and n=290 are NOT
        // multiples of MR/NR, so the MR-row-tile, NR-column-panel, column
        // remainder, and row remainder paths are all hit.
        // Packing only reorders where B is read from, never the ascending-l sum,
        // so the result must equal the textbook i-j-k accumulation bit-for-bit.
        let (m, k, n) = (70usize, 500usize, 290usize);
        assert!(k * n >= PACK_B_MIN_KN, "test must trip the pack threshold");
        let a: Vec<f64> = (0..m * k)
            .map(|i| (i as f64 * 0.011_31).sin() * 3.0 - 1.0)
            .collect();
        let b: Vec<f64> = (0..k * n)
            .map(|i| (i as f64 * 0.003_77).cos() * 2.0 + 0.5)
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

    /// Same-binary, same-worker A/B for the B-pack lever. Run with:
    ///   cargo test -p fj-lax --release matmul_2d_pack_ab_timing -- --ignored --nocapture
    /// Reports median-of-5 wall time for strided vs packed at GEMM sizes that
    /// spill L2/L3, and the speedup. Ignored by default (timing, not correctness;
    /// correctness is matmul_2d_packed_bit_identical_to_ijk).
    #[test]
    #[ignore]
    fn matmul_2d_pack_ab_timing() {
        use std::time::Instant;
        fn median_ms(mut v: Vec<f64>) -> f64 {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        }
        for &(m, k, n) in &[
            (1024usize, 1024usize, 1024usize),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        ] {
            let a: Vec<f64> = (0..m * k).map(|i| ((i % 97) as f64) * 0.01 - 0.3).collect();
            let b: Vec<f64> = (0..k * n).map(|i| ((i % 89) as f64) * 0.02 + 0.1).collect();
            let reps = 5;
            let mut strided = Vec::new();
            let mut packed = Vec::new();
            // Warm + interleave to share any worker drift across both arms.
            let _ = matmul_2d_with_pack(&a, m, k, &b, n, false);
            let _ = matmul_2d_with_pack(&a, m, k, &b, n, true);
            for _ in 0..reps {
                let t = Instant::now();
                let r0 = matmul_2d_with_pack(&a, m, k, &b, n, false);
                strided.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                let r1 = matmul_2d_with_pack(&a, m, k, &b, n, true);
                packed.push(t.elapsed().as_secs_f64() * 1e3);
                // Guard: both arms must agree bit-for-bit (no perf via wrong math).
                assert_eq!(r0, r1, "packed != strided at {m}x{k}x{n}");
            }
            let s = median_ms(strided);
            let p = median_ms(packed);
            let gflops = 2.0 * (m as f64) * (k as f64) * (n as f64) / 1e9;
            println!(
                "matmul {m}x{k}x{n}: strided {s:.2}ms ({:.0} GF/s) | packed {p:.2}ms ({:.0} GF/s) | speedup {:.2}x",
                gflops / (s / 1e3),
                gflops / (p / 1e3),
                s / p
            );
        }
    }

    #[test]
    fn matmul_2d_blocked_bit_identical_to_ijk() {
        // k=700 > KC (256) spans THREE pc-blocks (256+256+188), so the C
        // read-accumulate-store carry is exercised across multiple blocks. Force
        // four row-slice threads and B packing so this proof covers the blocked
        // kernel directly even when production gating routes smaller matrices to
        // the flat packed path. m,n are non-multiples of MR/NR, so the border
        // sweep is hit too. Must equal textbook i-j-k bit-for-bit.
        let (m, k, n) = (301usize, 700usize, 262usize);
        assert!(k > super::KC && k * n >= super::PACK_B_MIN_KN);
        let a: Vec<f64> = (0..m * k)
            .map(|i| (i as f64 * 0.007_93).sin() * 2.5 - 0.4)
            .collect();
        let b: Vec<f64> = (0..k * n)
            .map(|i| (i as f64 * 0.005_11).cos() * 1.7 + 0.3)
            .collect();

        let got = super::matmul_2d_with_threads(
            &a,
            m,
            k,
            &b,
            n,
            super::MatmulPlan {
                threads: 4,
                pack_b: true,
                block_k: true,
            },
        );

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
    fn pack_b_panels_layout_matches_source() {
        // Panel-major pack: bpack[jp*k*NR + l*NR + jj] == b[l*n + jp*NR + jj].
        let (k, n) = (5usize, 24usize); // 24 % NR == 0, three full panels
        let b: Vec<f64> = (0..k * n).map(|i| i as f64).collect();
        let bpack = pack_b_panels(&b, k, n, 3);
        for jp in 0..n / NR {
            for l in 0..k {
                for jj in 0..NR {
                    assert_eq!(
                        bpack[jp * k * NR + l * NR + jj],
                        b[l * n + jp * NR + jj],
                        "jp={jp} l={l} jj={jj}"
                    );
                }
            }
        }
    }

    #[test]
    fn pack_b_panels_parallel_matches_serial_bits() {
        let (k, n) = (17usize, 72usize); // nine full NR panels
        let b: Vec<f64> = (0..k * n)
            .map(|i| (i as f64 * 0.013_7).sin() * 3.0 - 0.25)
            .collect();
        let serial = pack_b_panels(&b, k, n, 1);

        for threads in [2usize, 3, 4, 7, 16] {
            let parallel = pack_b_panels(&b, k, n, threads);
            assert_eq!(parallel.len(), serial.len(), "threads={threads}");
            for (idx, (got, want)) in parallel.iter().zip(serial.iter()).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "threads={threads} packed-B bit mismatch at {idx}"
                );
            }
        }
    }

    #[test]
    fn pack_b_pc_panels_layout_matches_source() {
        let (k, n) = (KC + 5usize, 24usize); // two pc-blocks, three full panels
        let npanels = n / NR;
        let b: Vec<f64> = (0..k * n).map(|i| i as f64).collect();
        let bpack = pack_b_pc_panels(&b, k, n, 3);

        let mut pc = 0;
        while pc < k {
            let kc = KC.min(k - pc);
            let pc_base = pc * npanels * NR;
            for jp in 0..npanels {
                for l in 0..kc {
                    for jj in 0..NR {
                        assert_eq!(
                            bpack[pc_base + jp * kc * NR + l * NR + jj],
                            b[(pc + l) * n + jp * NR + jj],
                            "pc={pc} jp={jp} l={l} jj={jj}"
                        );
                    }
                }
            }
            pc += KC;
        }
    }

    #[test]
    fn pack_b_pc_panels_parallel_matches_serial_bits() {
        let (k, n) = (KC * 2 + 13usize, 72usize); // three pc-blocks, nine panels
        let b: Vec<f64> = (0..k * n)
            .map(|i| (i as f64 * 0.011_3).cos() * 2.25 + 0.5)
            .collect();
        let serial = pack_b_pc_panels(&b, k, n, 1);

        for threads in [2usize, 3, 4, 8, 16] {
            let parallel = pack_b_pc_panels(&b, k, n, threads);
            assert_eq!(parallel.len(), serial.len(), "threads={threads}");
            for (idx, (got, want)) in parallel.iter().zip(serial.iter()).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "threads={threads} pc-major packed-B bit mismatch at {idx}"
                );
            }
        }
    }

    #[test]
    fn matmul_2d_packed_nr8_golden_output_digest() -> Result<(), Box<dyn std::error::Error>> {
        let (m, k, n) = (12usize, 512usize, 320usize);
        assert!(k * n >= PACK_B_MIN_KN, "test must trip packed-B path");
        assert_eq!(n % NR, 0, "test must cover full NR-wide panels");
        let a: Vec<f64> = (0..m * k)
            .map(|i| (i as f64 * 0.006_31).sin() * 1.7 - 0.2)
            .collect();
        let b: Vec<f64> = (0..k * n)
            .map(|i| (i as f64 * 0.004_73).cos() * 2.1 + 0.4)
            .collect();

        let got = matmul_2d(&a, m, k, &b, n);
        let output_bits: Vec<u64> = got.iter().map(|value| value.to_bits()).collect();
        let digest = fj_test_utils::fixture_id_from_json(&output_bits)?;
        assert_eq!(
            digest, "e3762befad86e2a81da53a8413643b658a0be2d6136d69e195770b2beba48b3a",
            "packed NR-wide matmul output digest changed"
        );
        Ok(())
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
