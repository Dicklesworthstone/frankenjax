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
    if rows <= 1 || ops < MIN_PARALLEL_OPS {
        return 1;
    }
    let available = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    let by_work = (ops / OPS_PER_THREAD).max(1);
    available.min(rows).min(by_work).max(1)
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
    result.fill(0.0);
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    // Pack B's column panels once (shared, read-only across all output rows) when
    // B spills L2. Bit-identical to the strided read; just panel-contiguous.
    let bpack: Option<Vec<f64>> = if plan.pack_b && n >= NR {
        Some(pack_b_panels(b, k, n, plan.threads))
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
/// already packed into panel-major order (`bpack`, see [`pack_b_panels`]).
///
/// Loop order: pc (k in KC chunks) → jp (NR-col panel) → it (MR-row tile). The
/// packed B panel for `jp`'s pc-block lives at `bpack[jp*k*NR + pc*NR ..]` and is
/// reused across every row-tile while L1-resident; the [MR×KC] A tile is reused
/// across panels. C is read-modified-written once per pc-block: each pc-block
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
                let mut tile = row_super;
                while tile < row_super_end {
                    let i = tile * MR;
                    let abase = tile * KC * MR;
                    let mut jp = col_super;
                    while jp < col_super_end {
                        let j = jp * NR;
                        // This panel's pc-block: kc rows of NR, contiguous from the pack.
                        let panel = &bpack[jp * k * NR + pc * NR..];
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

/// Mixed-precision batched matmul: f32 inputs, **f64 accumulation**, f32 output.
///
/// Reads the `f32` operands directly (no f32->f64 promote buffer — for a
/// `[4096,512]@[512,512]` matmul the promote alloc+copy is ~19 MB) and widens
/// each element to `f64` inside the inner loop, accumulating in `f64`. Streaming
/// B as `f32` also halves its bytes through cache (the row-block kernel re-reads
/// all of B once per output row), which is the binding cost once B spills L1/L2.
///
/// BIT-FOR-BIT identical to "promote both operands to f64, run [`batched_matmul_2d`]
/// (f64 ascending-`l` accumulation), then round each output `as f32`": f32->f64 is
/// lossless, the per-element products `(a as f64)*(b as f64)` and their ascending-`l`
/// f64 sum are the same, and the final `acc as f32` is the same round. Proven by
/// batched_matmul_2d_f32_in_matches_promote_bits.
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

/// f32-input row-block kernel: accumulates each output row in an `f64` scratch
/// (ascending-`l`, widening f32->f64 per element) then rounds to `f32` into the
/// output. See [`batched_matmul_2d_f32_in`] for the bit-identity argument. The
/// `acc` scratch is reused across the block's rows to avoid per-row allocation.
fn batched_matmul_row_block_f32_in(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    g_start: usize,
    block: &mut [f32],
) {
    let mut acc = vec![0.0f64; n];
    for (ri, c_row) in block.chunks_mut(n).enumerate() {
        let g = g_start + ri;
        let bt = g / m;
        let a_off = g * k;
        let b_off = bt * k * n;
        acc.iter_mut().for_each(|x| *x = 0.0);
        for l in 0..k {
            let a_il = a[a_off + l] as f64;
            let src = &b[b_off + l * n..b_off + l * n + n];
            for j in 0..n {
                acc[j] += a_il * src[j] as f64;
            }
        }
        for (cj, &av) in c_row.iter_mut().zip(acc.iter()) {
            *cj = av as f32;
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

    /// The mixed-precision f32-input GEMM must be BIT-FOR-BIT identical to:
    /// promote both operands f32->f64, run the f64 `batched_matmul_2d`, then round
    /// each output `as f32`. Sized large enough to exercise the threaded path.
    #[test]
    fn batched_matmul_2d_f32_in_matches_promote_bits() {
        let (bt, m, k, n) = (2usize, 40usize, 33usize, 17usize);
        let af: Vec<f32> = (0..bt * m * k).map(|i| (i as f32 * 0.013).sin() * 2.0 - 0.5).collect();
        let bf: Vec<f32> = (0..bt * k * n).map(|i| (i as f32 * 0.019).cos() * 1.6 + 0.3).collect();
        let got = batched_matmul_2d_f32_in(&af, bt, m, k, &bf, n);
        // reference: promote -> f64 GEMM -> round as f32.
        let a64: Vec<f64> = af.iter().map(|&v| v as f64).collect();
        let b64: Vec<f64> = bf.iter().map(|&v| v as f64).collect();
        let want64 = batched_matmul_2d(&a64, bt, m, k, &b64, n);
        assert_eq!(got.len(), want64.len());
        for idx in 0..got.len() {
            assert_eq!(got[idx].to_bits(), (want64[idx] as f32).to_bits(), "mismatch at {idx}");
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
                promote * 1e3, gflop / promote, native * 1e3, gflop / native, promote / native
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
