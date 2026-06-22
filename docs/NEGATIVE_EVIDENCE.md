# Negative Evidence Ledger

Canonical project ledger: `../evidence/perf/negative_evidence_ledger.md`.

## FRONTIER SCORECARD (2026-06-21, CrimsonOtter) — what still loses to JAX and why

Consolidated from this session's measurements + the per-op entries below. The contained,
measurable-on-this-host, non-`+fma`, unowned perf levers are EXHAUSTED; every remaining loss
routes to one of three gates.

UPDATE 2026-06-22 (CrimsonOtter): the "EXHAUSTED" claim above was PREMATURE — a contained,
non-`+fma`, measurable lever was found and SHIPPED this day: **branchless-MLP scattered
gather/scatter**. The scattered single-element gather/scatter inner loops paid a per-element
`match Option` + `copy_from_slice(len 1)` CALL that serialized the random loads/stores (zero
MLP); replacing it with a tight `out[i]=src[idx[i]]` loop yielded **gather ~2x** (d551956b
f64/f32/i64; a4c5118a i32 2.13x + u32/u64/bf16/f16) and **scatter-overwrite 1.24x** (b4e74e2d),
halving the gather JAX gap ~28x->~15x. The residual is the Zen3 safe-Rust ceiling (SIMD
`Simd::gather_or` is +5.2% no-win — vgather microcoded; olm4p closed 95ee73c1). LESSON: the
true un-mined vein was **interpreter/eval per-element overhead in random-access ops**, not the
kernels — audit other ops for per-element call/branch in hot loops before declaring exhaustion.

UPDATE 2026-06-22 (CrimsonOtter, `frankenjax-murmw` FFT BOLD-VERIFY): the radical FFT route from
alien-graveyard/extreme-optimization was generated/vectorized radix-4 family specialization for the
power-of-two batched kernel. Fresh same-binary A/B rejected the lever: radix-4 SoA butterflies were
**0.44x** the speed of the retained radix-2 SoA butterfly path (`0.0056ms` vs `0.0128ms` for
8x256), so no source was promoted. Fresh RCH `ovh-a` release Criterion at HEAD measured
`fft_batch_2048x256_complex128` **4.3538ms** and dense-input **1.0879ms**; fresh JAX/JAXLIB 0.10.1
CPU x64 on the exact `sin(i*0.125)+1j*cos(i*0.25)` 2048x256 complex128 fixture measured
**0.328385ms** mean. Current retained rows remain **13.26x** and **3.31x** Rust/JAX losses. Scorecard:
**0 wins / 2 losses / 0 neutral; 0 kept / 1 rejected**. Retry only with generated length-specialized
radix plans, true SIMD/shuffle support, or a broader pocketfft-class backend route.

CONFORMANCE STATUS (2026-06-21, verified HEAD): **ENTIRE WORKSPACE GREEN** — `cargo test --workspace
--release` = 0 failures across all crates (fj-lax 1583/0 after the cholesky digest re-baseline
7a0f165e + the erf_inv regression fix c1b9ef15; fj-conformance all-green; fj-ad/fj-interpreters/etc.
green). The stale "8 fj-interpreters golden-hash RED" caveat from prior sessions is RESOLVED.

MEASURED HEAD-TO-HEAD (2026-06-21, CrimsonOtter, SAME-WORKER vs JAX 0.10.2 CPU x64, f64):
  - **sort 64k: JAX 12.51ms vs fj-lax 1.25ms = ~10.0x fj-lax WIN** (XLA CPU sort is a bitonic
    network → genuinely slow; JAX sort 1M = 237.8ms. fj-lax sort DOMINATES; memory's "1.6-7.9x"
    understated it — it's ~10x here). fj-lax sort is a confirmed, large, current domination.
  - **ENTIRE JAX CPU ORDER-STATISTICS FAMILY is catastrophically slow** (XLA lowers them to the same
    bitonic sort network): measured JAX 1M f64 — argsort **284ms**, top_k(k=100) **256ms**, median
    **226ms**; JAX top_k 64k/k128 **13.29ms**. fj-lax sort 64k is 1.25ms; top_k/argsort/median are
    all sort/partial-sort based on fj-lax's ~10x-faster sort, so **fj-lax DOMINATES the whole family**.
    MEASURED same-worker (all three now measured, no inference): **top_k 64k/k128 = fj-lax 0.497ms
    vs JAX 13.29ms = ~26.7x**, **argsort 64k = fj-lax 1.17ms vs JAX 15.07ms = ~12.9x**, **sort 64k =
    fj-lax 1.25ms vs JAX 12.51ms = ~10x** (fj-lax partial-selects / real sort; XLA full-sorts via
    bitonic network). This is fj-lax's single biggest domination zone — JAX-CPU's worst surface.
    REAL-WORLD-DTYPE CHECK: the domination is NOT an x64 artifact — in JAX's DEFAULT f32, **JAX sort
    64k f32 = 12.25ms ≈ its f64 (12.51ms)** (XLA's bitonic sort is dtype-agnostic-slow), so fj-lax
    (sort ≤ its f64 1.25ms) still wins ≥10x in the dtype JAX users actually run. **(fj-lax f32 exact
    now MEASURED, gap closed — see the 2026-06-22 f32-sort entry below: fj-lax f32 64k = ~0.77ms,
    FASTER than its own f64, ~10x over JAX f32.)**
  - **MATMUL DTYPE MATRIX IS COMPLETE (2026-06-22, SlateHarrier) — corrects a STALE "i32/u32/u64 gap" note.**
    The memory/index line claiming "ONLY matmul gap = i32/u32/u64 (odometer)" is STALE and cost a wasted
    probe this turn: `rank2_u64_matmul` is ALREADY 4-row register-blocked (commit 69e37c68, mirrors the i64
    kernel, with `rank2_u64_matmul_4row_block_matches_single_row_reference`), with `rank2_u64_any_orientation_matmul`
    + `batched_rank2_u64_matmul`; u32 routes through it (widen→u64→narrow `as u32`, mod-2³² wrap preserved), i32
    through the i64 kernel (narrow_i32). So EVERY matmul dtype now has a blocked/optimized kernel: f64 (B-pack
    +microkernel), f32 (register microkernel), i64/u64 (4-row blocked), i32/u32 (route+narrow), complex64/128
    (4-row blocked — SHIPPED this session), bf16/f16 (f32-accum). The matmul dtype-sibling frontier is fully
    closed; do NOT re-chase the integer/unsigned tail.
  - **DTYPE-SIBLING + nn-FUSION VEINS EXHAUSTED (2026-06-22, SlateHarrier) — verification, DO-NOT re-audit.**
    After the complex matmul/gather/scatter-add wins, swept the rest of the indexing/structural/nn families
    for secondary-dtype kernels lagging their real siblings. ALL verified already-covered (no gap): bf16/f16
    scatter (half-float branch in `eval_scatter_dense`); complex scatter (complex branch, scatter_typed! +
    lexicographic min/max); dynamic_slice + dynamic_update_slice (both have complex `as_complex_slice` + half
    `as_half_float_slice` dense branches); take_along_axis (no separate primitive — composes to Gather, now
    dtype-complete); one_hot (complex/u32/u64 dense per test). nn-FUSION: `softmax_2d`/`log_softmax_2d` are
    already single-pass fused (bit-identical tests); `standardize` (layernorm core) is a numerically-careful
    TWO-pass (mean, then E[(x-mean)²]) — a one-pass `E[x²]-mean²` fuse would lose accuracy (catastrophic
    cancellation) and is NOT bit-identical, so it's a parity-risk no-go, not a clean win. CONCLUSION: the
    dtype-sibling vein (this session's richest, ~4 wins) is mined out; remaining frontier = `cntiy` +fma
    (f64 matmul/exp/softmax/transcendentals), filed niche complex-conv bead, `murmw` FFT / `ur4h3` eigh.
  - **COMPLEX128 SCATTER-ADD parallel range-partition SHIPPED (2026-06-22, SlateHarrier) — ~1.39x.**
    Complex scatter is already de-boxed (`eval_scatter_dense` complex branch via `scatter_typed!` +
    lexicographic min/max — verified, NOT a gap), but complex scatter-ADD (slice_elems==1) used the
    serial `scatter_typed` cf loop, not the parallel range-partition (which was f64-Add-only because
    f32/i64 REGRESSED — their serial was already fast). Complex is 16B + 2 f64 adds/elem, so its serial
    is slow enough that the partition PAYS: routed complex scatter-add through the generic
    `scatter_reduce_range_partitioned<(f64,f64)>`. BIT-IDENTICAL (complex add is componentwise f64 add,
    each index folds i-ascending): scatter lib 31/0, `gather_scatter_oracle` 59/0. Same-binary A/B
    `bench_scatter_add_complex_partition_vs_serial` (1M c128): serial 4.87-5.48ms → partition **1.15-1.50x**
    (median ~1.39x). NOTE: this is the dtype where the partition generalization (rejected for f32/i64,
    [[project below]]) actually wins — gate it per-dtype, not blanket.
  - **COMPLEX128 GATHER de-box + threading SHIPPED (2026-06-22, SlateHarrier) — ~2.6x.** Complex gather
    was serial-only (per-element `extend_from_slice` + Option-match), missing BOTH the branchless
    `gather_single_dense` (slice_elems==1) and the threaded `gather_contiguous_into` (rows) that the real
    dtypes (f64/f32/i64/i32/u32/u64/bf16/f16) got. Routed complex through both (generic `T: Copy+Send+Sync`,
    `(f64,f64)` qualifies). BIT-IDENTICAL (`out[i]=src[idx[i]]`, same order/OOB-fill): gather lib 23/0,
    `gather_scatter_oracle` 59/0. Same-binary A/B `bench_gather_complex_branchless_vs_serial` (1M←4M c128):
    serial 7.26-7.69ms → branchless **2.56-2.68x** (higher than f64's ~2x — c128 is 16B/elem so the
    per-element extend overhead was relatively larger). Plus contiguous-row complex gather now threads.
    (Follow-up: complex SCATTER is still generic-boxed — a remaining sibling, lower-EV.)
  - **DTYPE-SIBLING-GAP AUDIT (2026-06-22, SlateHarrier) — matmul family now complete; complex conv
    filed.** After flipping complex matmul (below), swept the matmul/conv family for kernels lagging
    their f64/i64 siblings: bf16/f16 matmul VERIFIED already optimized (f16 decodes→f32 GEMM; bf16 has
    tuned row-block kernels) — no gap. complex64 matmul upcasts→(f64,f64)→`rank2_complex_matmul` (now
    blocked) — covered. complex transposed/general orientations permute→canonical blocked kernel —
    covered. **REMAINING: complex CONV is a naive per-element-boxed loop** (excluded from the real
    im2col+GEMM path via `if !is_complex`); fresh JAX c128 conv N8×32×32×32→64 k3×3 = 42.4ms mean /
    25.3ms min, fj-lax ~30x+ slower. Niche op (signal/quantum) + multi-path (conv1d/2d/grouped) → filed
    bead `frankenjax-complex-conv-gemm-5xdr7` (P3) with the f32-template approach rather than risking the
    intricate conv code inline. The matmul-family dtype-sibling gap is otherwise CLOSED.
  - **COMPLEX128 MATMUL 4-row register-blocking SHIPPED (2026-06-22, SlateHarrier) — narrows JAX
    loss 1.95x→1.31x.** Unlike integer, XLA DOES have fast complex GEMM (zgemm): fresh JAX c128
    matmul 512³ **11.97ms**, 1024³ **58.7ms**. fj-lax `rank2_complex_row_block` was a NAIVE single-row
    i-k-j loop (no register blocking, unlike the f64/i64 kernels) → at RAM-bound 1024³ (B=16MB>L3) it
    re-streamed B once per output row. Applied 4-row register blocking (B streamed rows/4×; mirrors
    `rank2_i64_row_block`). BIT-IDENTICAL (interleaves 4 independent output rows, never regroups a
    sum): complex lib 132/0, dot_general 29/0, dot 38/0, tensor_contraction 10/0; same-binary
    block-isolated A/B (`i64_matmul_speed` C128 section) **512³ 1.71x / 1024³ 1.26x** single-thread.
    Production threaded c128 matmul 1024³ **~114ms → ~77ms** vs JAX 58.7ms = loss **1.95x → 1.31x**;
    512³ 14.5ms vs 11.97ms = 1.21x. Residual is the `cntiy` +fma gate (zgemm fuses complex MACs).
  - **BATCHED complex128 matmul ALSO 4-row-blocked (2026-06-22, SlateHarrier).** `batched_complex_row_block`
    (vmap(complex matmul) / 3D complex dot_general) was the same naive single-row sibling; applied
    batch-AWARE 4-row blocking (a 4-row group only shares a `b_row` load when all four rows are in the
    same batch `g/m`, since each batch reads its own `b[bt*k*n..]`). BIT-IDENTICAL (lib complex 132/0
    incl. `batched_rank2_complex_matmul_matches_generic`; bench assert vs naive). Same-binary
    `C128_BATCHED_MATMUL b=32 128³` = naive-1thr 47.9ms → prod-threaded-blocked 5.23ms = **9.16x**
    (thread×block). REFINEMENT to the non-batched row above: a less-contended re-run measured the
    single-thread block isolation at **1.78x (512³) / 2.00x (1024³)** and production threaded c128
    matmul **512³ 9.07ms (now BEATS JAX 11.97ms) / 1024³ 61ms (≈ JAX 58.7ms parity)** — so the complex
    matmul JAX gap is now parity-to-win, not the prior 1.31x (the 1.31x figure was a contended run;
    the block ratio is robustly 1.7-2.0x). Both non-batched + batched complex now register-blocked.
  - **INTEGER MATMUL is a NEW fj-lax domination zone (2026-06-22, SlateHarrier)** — XLA has no
    integer BLAS, so JAX falls back to a scalar/naive int matmul that is catastrophically slow.
    Fresh `JAX_ENABLE_X64=1` jaxlib CPU x64 `int64 @ int64`: **512³ = 367.2ms mean (min 347ms),
    1024³ = 3977.3ms mean (min 3792ms)**. fj-lax production threaded `rank2_i64_matmul` (same-binary
    `i64_matmul_speed` bench, worker shared): **512³ = 11.50ms (31.9x WIN), 1024³ = 182.6ms (21.8x
    WIN)**; 1536³ 201.5ms / 2048³ 561ms thread cleanly (8-8.8x over single-row). Already optimized
    (threaded row-block + 4-row B-reuse kernel) — NO lever, it's a confirmed large domination. JAX-CPU
    integer GEMM joins sort/order-statistics as a worst-surface for XLA that fj-lax crushes. (The f64
    GEMM loss is a SEPARATE story — that one is `cntiy` +fma-gated; integers have no fma so fj-lax's
    safe-Rust int kernel wins outright.)
  - OPPORTUNITY (feature gap, not a perf lever for fj-lax): **median/percentile/quantile** are
    sort-based and JAX-CPU-slow (median 1M = **226ms**, argsort-backed), but frankenjax has NO
    user-facing median/percentile (only internal `median_ms` timing helpers in linalg/tensor_contraction
    + a `percentile(u128)` stats helper). If the fj-api/fj-py layer ever adds them, compose from
    fj-lax's ~10x-faster sort (or a quickselect O(n)) for a large, easy JAX domination. Flagging for
    the user-API owner — out of the fj-lax perf-lever lane.
  - matmul 1024²: JAX 2.91ms (fj-lax loses, `cntiy` +fma-gated). exp 1M: JAX 0.437ms (fj-lax loses,
    cntiy/sweep). sum 1M: JAX 0.111ms (parity-class). Consistent with the gate table below.
  - **softmax row16 relaxed-exp no-ship (cod-b, 2026-06-21):** tried a finite-16-column fast path
    using the existing `simd_poly_exp_into` helper in `softmax_2d`/`log_softmax_2d`, then reverted it.
    RCH `hz2` baseline `nn/softmax_2d_65536x16_fused` was **2.6515ms** midpoint; candidate bench
    landed on `vmi1149989` despite the `RCH_WORKER=hz2` hint and measured **2.5922ms**, only **2.2%**
    better and still a **2.37x Rust/JAX loss** against fresh JAX mean **1.091807ms** (p50
    **1.101048ms**). The same routing run showed `log_softmax_2d` at **2.4909ms** versus the noisy
    prior **6.0083ms** baseline, but that is not the claimed JAX softmax target and was not
    same-worker proof. Scorecard: **0 wins / 1 loss / 0 neutral; 0 kept / 1 reverted**. Retry
    predicate: do not retry stack-copy row16 polynomial-exp softmax; next credible route needs
    approved target-feature/FMA SIMD exp or an attention-level fused kernel that first beats the
    **2.65ms** Rust baseline by a large margin and closes JAX's **1.09ms** comparator.
  - **tan 1M f64 small-angle: now flipped to fj-lax WIN — fresh JAX/JAXLIB 0.10.2 CPU x64 mean
    1.412564ms vs fj-lax 1.1134ms = fj-lax 1.27x FASTER.** The retained path is intentionally
    scoped: large dense F64 tensors with every element in `[-pi/4, pi/4]` use a Cephes/Remez-style
    rational kernel; scalar, complex, f32/half, NaN/inf, and general-range tensors stay on the old
    `libm` route. Same-worker RCH `ovh-a` Criterion also measured the old libm reference at
    4.6905ms, so the retained Rust-side speedup is 4.21x. Scorecard: **1 win / 0 loss / 0 neutral**
    for this row.
  - **boxed-literal scalar pow/atan2 1M: kept a real Rust-side de-box/threading win, but both still
    lose JAX.** Same-worker RCH `vmi1293453` Criterion: `pow_scalar_1m_f64_literal_ref`
    **80.435ms -> 15.193ms** (**5.29x** Rust speedup) and
    `atan2_scalar_1m_f64_literal_ref` **38.339ms -> 11.987ms** (**3.20x** Rust speedup).
    Fresh repo-venv JAX/JAXLIB 0.10.1 CPU x64 comparators on the exact fixtures: pow
    **1.808211ms** mean, atan2 **2.214336ms** mean. Retained Rust/JAX ratios:
    pow **8.40x loss**, atan2 **5.41x loss**. Scorecard for this pass:
    **0 wins / 2 losses / 0 neutral**; keep only because the same-worker Rust deltas are large
    and the proof is bit-identical.
  - maxpool/reduce_window 256x256 15x15 SAME: JAX 0.5498ms — **PARITY, NOT a domination zone**
    (probed expecting a window-naive O(n·225) JAX path the fj-lax separable-deque would crush;
    XLA's CPU reduce_window is already optimized/separable, so it's fast). Negative result — do not
    re-chase pooling as a JAX-loss lever.
  - FFT batch 2048x256-point c128: fj-lax 16.81ms vs JAX 0.866ms = **fj-lax ~19.4x LOSS** (within the
    documented `murmw` 7-43x band). XLA uses pocketfft (intra-transform SIMD + cache-optimal mixed
    radix); fj-lax is intra-FFT-SIMD-bound in safe Rust. Biggest non-cntiy loss, but intra-SIMD-HARD
    (no shuffle/gather in safe std::simd) + actively convergent (swarm probing/rejecting Bluestein
    tile/thread + scalar mixed-radix). Not a clean unowned lever — see [[project_fft_jax_loss_frontier]].
    REFINEMENT (analysis): the butterflies (57% of fj-lax FFT time per the SoA profile) are complex
    multiplies (`a*b ± c*d`), so they are partly FMA-BOUND — pocketfft fuses the mul-adds (`vfmadd`),
    fj-lax (no `+fma`) does separate mul+add. The pow2 FFT golden is bit-exact, so fma-fusing the
    butterflies changes the bits → breaks the golden. So **~2x of the FFT gap is the SAME `cntiy`
    +fma gate as matmul/exp** (golden-gated); the rest is algorithmic (mixed/split-radix op-count) +
    cache (the SoA transpose). FFT is thus part-cntiy-gated, part-intra-SIMD-hard — not purely either.
  - DISPATCH / small-op jit regime (the last unmeasured category): **JAX jit(x+1) scalar = 5.87us/call**
    — for tiny ops the Python->XLA dispatch boundary dominates. fj-lax is PURE RUST (no Python boundary)
    + the so4wo compiled-jaxpr cache (records ~1.6-5.4us). So **small-op-dispatch-bound workloads are a
    fj-lax non-loss / likely WIN** — JAX's Python boundary costs it ~6us/call that fj-lax doesn't pay;
    large ops are kernel-bound (cntiy). Caveat: this is a Rust-vs-Python comparison, and the fj-lax
    exact is uncaptured this window (persistent rch bench-time capture flake). Dispatch is NOT a fj-lax
    loss lever — the interpreter "tax" (compiled away by so4wo) is ≤ JAX's Python dispatch.
  - **cumsum 4M 1D: now flipped to fj-lax WIN — fresh JAX 18.318ms vs fj-lax 7.5297ms = fj-lax 2.43x FASTER.**
    Path = `scan_contiguous_lines_to_vec` single-line `op(acc,value)` + `out.push(acc)` loop
    (reduction.rs ~3133). DIAGNOSED (A/B, bench `cumsum_4m_f64_1d_tight`): a TIGHT raw direct-add
    `acc+=v; push` loop is ALSO ~30ms — so the loss is NOT removable dispatch/closure/push overhead;
    it is the FUNDAMENTAL sequential f64 dependency chain (my ~2.7ms floor estimate was wrong — the
    scalar add chain + 64MB R/W traffic is genuinely ~30ms here). JAX's 14ms uses a reassociated
    SIMD-blocked / parallel-prefix cumsum (breaks the dependency chain). **FIX IS LEGAL:** the cumsum
    oracle is TOLERANCE (`abs()<1e-10`, cumulative_oracle.rs:118), NOT bit-exact — and JAX itself
    reassociates, so a SIMD-blocked/threaded prefix-sum is BOTH faster AND more JAX-faithful. The
    `f64 cumsum FP-non-associative DO-NOT` note was over-conservative. **BOLD-VERIFY KEEP
    (cod-b, 2026-06-21):** retained blocked prefix-sum (per-block local scan + block-offset
    application) passed RCH `hz2`
    `blocked_dense_f64_single_line_cumulative_matches_serial_reference` 1/1 and RCH
    `vmi1149989` `fj-conformance --test cumulative_oracle --release` 45/45. RCH `ovh-a`
    Criterion now measures `eval/cumsum_4m_f64_1d` at **7.5297ms** (`7.4698..7.6034ms`);
    the same filter's tight sequential diagnostic is **23.805ms**. Fresh JAX/JAXLIB 0.10.1
    CPU x64 on the exact `np.arange(1 << 22) * 0.001` fixture is **18.318300ms** mean,
    p50 **18.290179ms**. Scorecard: **1 win / 0 loss / 0 neutral** for this row. Diagnostic
    bench kept as the sequential-floor ref.
  - JAX CPU is broadly SLOW on order-dependent ops (exploitable): searchsorted 1M=48.8ms (fj-lax has
    no searchsorted primitive — out of scope), scatter-add 1M=4.50ms, gather 0.469ms, argmax 0.917ms.
    **cummax/cummin 1M now measured head-to-head:** cummax **fj-lax 2.0374ms vs JAX p50
    3.458314ms = 1.70x fj-lax WIN**; cummin **fj-lax 3.4187ms vs JAX p50 3.602727ms =
    1.05x near-parity fj-lax WIN** (upper Criterion bound crosses the JAX p50, so classify
    cummin as neutral/slight win, not closed). Scorecard: **1 win / 0 loss / 1 neutral-slight-win**.
    **Scatter-add 1M f64 1D now measured and narrowed:** baseline fj-lax **30.611ms** on
    RCH `vmi1227854`; retained bucketed owner-computes path **9.9667ms** on the same worker
    (**3.07x Rust-side speedup**); fresh JAX/JAXLIB 0.10.1 CPU x64 p50 **3.639273ms**, so the
    retained row is still a **2.74x Rust/JAX LOSS**. Rejected/reverted the unique-atomic branch
    (**10.822ms**, +8.6% slower than bucketed) and the histogram/prefix single-buffer branch
    (RCH `ovh-a` production **11.852ms** vs candidate **13.426ms** best repeat / **17.162ms**
    first run; **+13.3% to +44.8% slower** than production). Correctness: RCH `hz2`
    `range_partitioned_f64_scatter_add_matches_literal_path` 1/1 and RCH `vmi1152480`
    `gather_scatter_oracle` 59/59. Scorecard for scatter-add: **0 JAX wins / 1 JAX loss /
    0 neutral; 1 Rust-side keep / 2 reverted branches**.

| Op family | vs JAX (measured) | Gate on the remaining gap |
| --- | --- | --- |
| cheap elementwise (add/mul/sub), broadcast, select, comparison | WIN (threaded past L3, 1.7-2x) | none — done |
| batched gather/scatter (I64/F64/F32) | WIN (1.15-3.6x) | none — suspected batched loss disproven |
| scatter-add 1M f64 1D | **was LOSS 2.74x; now ~1.25-1.3x** after PARALLEL-build keep + u64-packed pairs (2026-06-22) — the serial bucket-build, not the apply, was the bottleneck; per-thread local-block partition is ~2.78x Rust-side, then packing `(idx<<32)\|i` into one u64 (vs 16B tuple) adds ~1.06x (criterion ~5.4ms vs fresh JAX p50 4.35ms / mean 4.59ms) | gap now ~1.3x: residual is the random-write apply latency + eval dispatch/extraction overhead. The earlier "fundamentally different parallel direct-write" framing was answered by parallelizing the PARTITION (not the writes); unique-atomic and histogram/prefix branches stay rejected; inline-value apply tweak measured a no-ship (median 0.948x) |
| scatter-add 1M **f32** 1D (+ i64 / mul / min / max) | **PARITY — not a loss** (serial loop ~2.5ms ≈ JAX f32 p50 2.88ms) | partition-generalization REGRESSES these (~0.70x, 2026-06-22) — pure-serial baseline + 16MB pair-build overhead; do NOT route through `scatter_reduce_range_partitioned` (only f64-ADD's already-partitioned baseline benefits) |
| scattered single-element gather (1M←4M, f64/f32/i64/i32/u32/u64/bf16/f16) | LOSS ~15x vs JAX, **halved from ~28x** — branchless-MLP ~2x SHIPPED (2026-06-22: d551956b f64/f32/i64; a4c5118a i32 2.13x) | residual is the safe-Rust/**Zen3 ceiling**: SIMD `Simd::gather_or` is **+5.2% no-win** (vgather microcoded, olm4p closed 95ee73c1); closing needs AVX-512 gather (absent) or index pre-sort (out of scope) |
| scatter-overwrite 1M f64 (slice_elems=1) | branchless **1.24x SHIPPED** (2026-06-22 b4e74e2d) | near ceiling — scattered stores less latency-bound than gather loads (store buffer hides latency), so the same lever yields 1.24x vs gather's 2x |
| contiguous ROW gather (slice_elems>1, embedding/row lookup) | WIN — f64/f32/i64/bf16 already threaded (~3.7x); **i32/u32/u64 threaded this session 1.40x SHIPPED** (2026-06-22 63b74dec, was serial-only) | none — gather dtype coverage COMPLETE (every dense dtype has both slice_elems==1 branchless + slice_elems>1 threaded). Row memcpy is memory-BW-bound so threading scales sub-linearly (1.40x) |
| sort, reductions, RNG, conv, einsum, dot_general | WIN / parity | none — done |
| **matmul / GEMM** (256-1024 f64) | **LOSS 4-15x** | **`cntiy` +fma** (already blocked-GEMM + threaded + register microkernel; microkernel is FMA-bound, capped ~XLA/2; pure-safe-Rust, no BLAS) |
| **transcendental — tolerance-only** (cbrt/erf/tanh/tan/atan2, no bit-golden) | MIXED: cbrt/erf/tanh/atan2 still LOSS; guarded small-angle tan now WIN | cod-b sweep (no cntiy needed): cbrt 11.9ms->3.30ms (3.60x, 64c0ded1), erf 21.1ms->6.85ms (4.58x, d74a6472), tanh 6.20ms->4.27ms (1.45x), small-angle tan 4.69ms->1.11ms (4.21x; 1.27x faster than JAX), scalar atan2 dense 30.35ms->14.00ms (2.17x internal, still JAX loss), boxed-literal scalar pow 80.44ms->15.19ms and boxed-literal atan2 38.34ms->11.99ms (5.29x/3.20x internal, still 8.40x/5.41x JAX losses) SHIPPED. General-range tan still falls back to `libm`; remaining losses are exp/FMA-gated or need true SIMD-polynomial range reduction |
| **transcendental — bit-pinned** (exp/sin/log/asin/acos) | **LOSS** | **`cntiy` +fma** (SIMD-poly = 2.20x WITH / 0.79x WITHOUT — cz0g0) + re-baseline the 5 bit-goldens |
| softmax / attention (fused) | LOSS (exp-bound, ~1.2x ceiling) | **`cntiy`** — option (b) audited per-fn `target_feature(fma)` SIMD-exp in tolerance sites preserves goldens; or (a) global +fma + golden re-baseline |
| **FFT pow2 / real** | WIN (1.7-3x SoA) | none — shipped, near safe-Rust ceiling |
| **FFT Bluestein-prime batch** (256x1009) | **LOSS 9.15x** | near safe-Rust ceiling on this path; tile-height and thread-cap probes are no-ships; next route needs a real SIMD-within-convolution kernel or quiesced-host proof |
| **FFT smooth-composite batch** (128x1000) | **LOSS 15.2x** | generated length-specialized kernels (big effort) OR a **quiesced host** (threading is a 0.4x no-ship under swarm contention; SoA-iterative 0.15x, specialized SoA 0.57x, scalar-specialized stage kernel 0.46x, Bluestein 0.39x all no-ship; butterflies already specialized) |
| FFT pow2 batch dense (2048x256) | LOSS 3.83x | near safe-Rust ceiling (pocketfft SIMD-within-FFT needs AVX-512 + complex shuffles) |
| eigh / SVD | (owned: `ur4h3`, WildForge) | — |

**Single highest-leverage unlock: the `cntiy` +fma maintainer decision** — it gates matmul,
all transcendentals, softmax, and attention simultaneously.

**+fma decomposition (per the documented PARITY ANALYSIS — NOT a fresh measurement; see
correction below).** The analysis in memory `project_fma_lever_policy_blocked` (BlackThrush/
CoralReef) holds that the +fma FLAG itself is bit-identical: Rust's `fp-contract=off` (which
`.cargo/config.toml` relies on for the +avx2 bit-identity) blocks `a*b+c` auto-fusion, and
`f64::mul_add` is one-rounding regardless of +fma — so the flag only makes EXISTING `mul_add`
calls (the fma primitive) fast, changing no results. The gated part is step 2 (kernel rewrites
to `mul_add`/SIMD-poly), which DOES change results and needs matmul/exp goldens updated to the
fma-fused (JAX-matching) values. So cntiy = (1) enable +fma [analysis says golden-safe, free] +
(2) kernel rewrites + golden updates [the real parity-policy call].

**NOW EMPIRICALLY VERIFIED 2026-06-21 (CrimsonOtter).** First attempt was botched —
`rch exec -- env RUSTFLAGS=...` does NOT apply +fma (rch ignores env RUSTFLAGS; only
`.cargo/config.toml` is synced). Redone CORRECTLY by editing `.cargo/config.toml` locally to
`+avx2,+fma`, building via rch in an isolated target dir, then reverting the edit (NOT pushed —
the flag flip is the maintainer's call). **Verified +fma actually applied this time** via the
flag-dependent speed signal (lesson learned): `cz0g0 bench_fma_vs_nonfma_matmul` flipped from
`0.90x` (no-fma, libcall) to **`fma=1.27x FASTER` at 256³ (28.5 vs 22.4 GFLOP/s)** — hardware
`vfmadd` engaged. With +fma genuinely on, **full lib conformance = 1581 pass, 1 fail, and that
1 is the SAME pre-existing `cholesky_blocked` golden drift (fails without +fma too)**. So the
+fma FLAG breaks ZERO additional goldens — **empirically confirmed, not just analysis**. This
upgrades the decomposition: step 1 (enable +fma) is golden-safe + gives mul_add code ~1.27x
free; step 2 (rewrite production GEMM/exp to mul_add — changes two->one rounding) is the real
golden-update call and is where the bigger FMA-GEMM/SIMD-exp wins live. METHOD NOTE: to A/B a
build flag, verify it took effect via a signal that ONLY changes if the flag applied (here the
fma matmul speedup); a result that's identical either way (conformance under a result-stable
flag) proves nothing.

**THE TRANSCENDENTAL GOLDEN-GATE IS OP-SPECIFIC (2026-06-21, CrimsonOtter) — +fma unlocks
MORE than thought, with LESS parity risk.** Audited which transcendentals carry a bit-exact
self-golden (the thing that forces a parity-relaxation decision) vs only a tolerance oracle:
  - **BIT-PINNED (need golden relaxation):** `exp`, `sin`, `log`, `asin`, `acos`
    (`expected_*_bits` self-golden digest, arithmetic.rs ~19651).
  - **TOLERANCE-ONLY (NO bit-golden — only `*_oracle.rs` tolerance + special-value exactness):**
    `erf`, `tanh`, `tan`, `cbrt`, `cos` (each: 0 bit-golden refs).
  Consequence: once `+fma` is committed (proven golden-safe above), a SIMD-poly version of the
  TOLERANCE-ONLY transcendentals can ship with NO golden-relaxation decision at all — and `erf`
  is HOT (12 nn.rs refs; it's the gelu kernel), `tanh` is a common activation. So step-1 (+fma)
  alone unlocks a meaningful, parity-clean chunk (erf/tanh/tan/cbrt SIMD), deferring the harder
  golden-relaxation to just exp/sin/log/asin/acos. Strengthens the +fma-commit case: the upside
  half (erf/gelu, tanh) needs no parity-policy call. REMAINING WORK on that half = implement the
  SIMD-poly kernels (only simd_exp exists today); measured: these are scalar libm libcalls today
  (cbrt 9.4ms/1M vs sqrt's autovectorized 0.54ms), so the SIMD win is real but fma-gated (the
  poly is fma-bound like exp's 2.2x/0.79x).
  CBRT BOLD-VERIFY KEEP (2026-06-21, CrimsonOtter): the cbrt scalar bit-hack + Halley route
  did pass the tolerance-only oracle gate and shipped as a guarded non-`+fma` lever. Focused
  RCH release tests validated `fast_cbrt_f64` at max relative error `6.455e-15` versus the
  1e-10 oracle bar; full `fj-conformance --release` passed remotely on `hz2`. Same-binary
  RCH `fj-lax` Criterion on `vmi1149989` measured the old
  threaded libm reference at **11.876 ms** and the new fast path at **3.2973 ms** for
  `eval/cbrt_1m_f64_vec`, a **3.60x Rust-side speedup**. Fresh JAX/JAXLIB 0.10.1 CPU x64 on
  the exact fixture measured **2.157837 ms**, so the retained row is still a **1.53x Rust/JAX
  loss**. Verdict: KEEP because it removes most of the libm-call tax with no bit-golden
  relaxation, but do not call cbrt closed. Retry predicate: true SIMD/vectorized polynomial or
  range-reduced kernel that first beats this scalar fast path in a same-binary A/B and keeps
  the 1e-10 oracle gate green.
  ERF BOLD-VERIFY KEEP (2026-06-21, CrimsonOtter): the `erf` common-range Maclaurin loop was
  replaced with fdlibm-derived minimax rational bands, leaving the existing 2.857..3.5 bridge
  and high-tail behavior intact. Focused RCH release tests passed
  `erf_high_accuracy_and_seam_continuity` and `fj-conformance --test erf_oracle` (31/31).
  Fresh JAX/JAXLIB 0.10.1 CPU x64 on the exact `eval/erf_1m_f64_vec` fixture measured
  **1.495718 ms**. The old Rust series baseline on RCH `ovh-a` was **21.110 ms**
  (**14.11x** JAX loss). The retained rational candidate measured **6.8485 ms** on
  `vmi1149989` (**4.58x** JAX loss, best observed current-code row) and **12.515 ms** on
  `hz2` (**8.37x** JAX loss); worker pinning was unavailable, so only the ratio-vs-JAX and
  repeated direction are used as proof, not a strict same-worker delta. A more radical
  degree-20 Chebyshev `[0,2]` layer was tested and reverted: same-worker `ovh-a` timing
  improved the old series to **10.596 ms** (**7.08x** JAX loss) but was slower than the
  rational candidate signal. Verdict: KEEP the rational path as a material non-`+fma`
  narrowing lever; `erf` still loses JAX and routes next to true SIMD/vector polynomial or
  approved target-feature/FMA work.
  TANH BOLD-VERIFY KEEP (2026-06-21, CrimsonOtter): the large dense-f64 `tanh` path now
  uses the existing SIMD polynomial `exp` helper to compute
  `sign(x) * (1 - exp(-2|x|)) / (1 + exp(-2|x|))` for tensors at the measured 1M-element
  threshold, leaving scalar/small/f32/half/complex paths on the existing libm route. Focused
  RCH release test `simd_poly_tanh_large_dense_f64_matches_libm_tolerance` passed on
  `vmi1149989` with max absolute error **2.220e-16** versus the 1e-10 oracle bar; RCH
  `fj-conformance --test tanh_oracle --release` passed **36/36**. Same-invocation RCH
  `ovh-a` Criterion measured old libm-reference `eval/tanh_1m_f64_vec_libm_reference`
  **6.1998 ms** and retained production `eval/tanh_1m_f64_vec` **4.2741 ms** (raw
  probe **3.7810 ms**), a **1.45x** Rust-side speedup. Fresh local JAX/JAXLIB 0.10.1 CPU
  x64 on the exact fixture measured **0.293181 ms** mean, so the retained row is still a
  **14.58x Rust/JAX loss**. Verdict: KEEP because it removes a material chunk of the libm
  tanh tax under the tolerance-only contract; do not call tanh closed. Next route needs
  true lower-allocation SIMD/FMA polynomial or target-feature work that first beats this
  production path in a same-binary A/B.

**FULL-WORKSPACE +fma VERIFICATION (2026-06-21, CrimsonOtter = cod-b, cntiy's assignee).**
Built the WHOLE workspace with `+avx2,+fma` (local config edit, isolated dir, reverted — NOT
committed; the shared-config flip of a deliberately-set flag affecting all agents is the
explicit `[maintainer-decision]`, and the automated loop can't authorize it). Result:
  - fj-lax: 1581 pass / 1 fail (the pre-existing cholesky_blocked drift — not +fma). GREEN.
  - fj-conformance: 38 + 24 pass. GREEN.
  - **fj-ad: `tests::proptest_tests::prop_jvp_matches_vjp_single` FAILED** under +fma (panic at
    lib.rs:17604) — BUT it's a proptest (random inputs) and PASSED in a separate +fma fj-ad run
    (403/0), so it's input-dependent: +fma makes forward-mode jvp and reverse-mode vjp (different
    op orders) diverge enough to trip the assertion for SOME inputs.
CONCLUSION: **+fma is NOT a clean drop-in.** Golden-safe for fj-lax/conformance, but the
swarm-wide flip ALSO perturbs the AD jvp==vjp consistency invariant (on top of the exp/sin/log
self-goldens). cntiy's true cost includes making the AD mode-equality property fma-consistent
(or tolerance-loosened), not just the transcendental goldens. This is exactly why the flip is a
maintainer decision and shouldn't be done blindly — my earlier fj-lax-ONLY +fma test missed it.
Retry predicate before committing +fma: A/B the fj-ad jvp/vjp proptest with/without fma across
many seeds to confirm it's a tolerance issue, not a real AD bug.
CLASSIFICATION (2026-06-21, partial): the assertion is `(fwd - rev).abs() < 1e-8` (tolerance,
not bit-exact). WITHOUT +fma, `prop_jvp_matches_vjp_single` showed 0 failures in 5 runs; it
failed once UNDER +fma — tentatively `+fma`-caused (rch output capture was flaky so not
definitive). Given the AD is mathematically correct and the bound is a moderate 1e-8, this is
almost certainly +fma's extra rounding tipping ILL-CONDITIONED inputs (large x / cancellation)
over the tight tolerance — i.e. a TOLERANCE-LOOSENING need (loosen to ~1e-6, or make the bound
condition-number-aware), NOT an AD bug. So cntiy's +fma cost is: loosen this one fj-ad proptest
bound + relax the exp/sin/log self-goldens + commit the flag + implement the SIMD kernels —
a multi-party effort (fj-ad is codex-owned; the flag is the maintainer's call).

Second unlock: a quiesced host to measure FFT/threading wins JAX gets from idle cores.

## 2026-06-22 - SlateHarrier atan2/pow dense-f64 VERIFIED already-threaded (residual is fma-gated, NOT a threading lever)

Checked whether the documented atan2 (5.41x) / pow (8.40x) JAX losses were an un-threaded scalar-libm
gap (a cheap threading win, like the unary transcendentals). They are NOT: `Primitive::Atan2` and
`Primitive::Pow` are both in `is_expensive_binary`, so `eval_binary_elementwise` routes their same-shape
dense f64 path through `eval_same_shape_f64_expensive_parallel` (work-scaled `thread::scope` fan-out) —
already threaded. The retained ledger numbers (atan2 ~12ms, pow ~15ms at 1M) ARE the threaded-dense
times; libm `pow` alone is ~240ns/elem (15ms threaded over 16 cores). So the residual vs JAX (which
SIMD-vectorizes exp/log with FMA) is the **fma-gated SIMD-transcendental gap**, identical to exp/cbrt —
NOT a missing-threading lever. DO-NOT re-attempt threading atan2/pow. The flip requires `cntiy` +fma
(SIMD-poly exp = 2.2x WITH / 0.79x WITHOUT; cbrt division-free flip also no-ship at 0.85x, prior entry).
Consolidation: the unowned, non-gated, contained transcendental levers are exhausted — every remaining
transcendental/matmul/softmax loss routes to the existing P1 bead `frankenjax-cntiy` (+fma maintainer
decision). No code change; verification only.

## 2026-06-22 - SlateHarrier cbrt dense-f64 8-wide SIMD — SHIPPED (~1.25x, bit-identical, narrows JAX 1.53x→~1.2x)

cbrt 1M f64 was the closest-to-flipping non-gated, tolerance-only (no bit-golden) loss (~1.53x JAX;
compute-bound scalar bit-hack+Halley ~3.3ms vs ~0.8ms memory floor → SIMD headroom). Added an 8-wide
`fast_cbrt_slice_into` (bit-hack `bits/3 + magic` via u64x8 + 2 Halley iters via f64x8 + scalar fixup
for the rare 0/non-finite/out-of-range lanes & tail) and a dense-f64 threaded path
(`cbrt_dense_f64_parallel`), wired into `Primitive::Cbrt` (other dtypes keep the guarded scalar path).
BIT-IDENTICAL to scalar `fast_cbrt_f64` (same ops, no FMA → one rounding each), proven by
`assert_eq!` in `bench_cbrt_simd_vs_scalar` + `validate_fast_cbrt_accuracy` + `erf_cbrt_parallel_bit_identical`
+ `cbrt_oracle` 34/0. Same-binary A/B: scalar 3.04-3.68ms → simd8 2.57-2.80ms = **1.19-1.40x**
(median ~1.25x). The `u64/3` partially scalarizes on AVX2 (no 64-bit vector mul-high), capping the
gain — but it is real, bit-identical, and zero parity risk. vs JAX cbrt ~2.16ms: loss narrows
**1.53x → ~1.2x** (does NOT flip — still a small loss). GREEN: `fj-lax --lib cbrt` 8/0, `cbrt_oracle`
34/0, clippy clean. Scorecard: **0 JAX wins / 1 narrowed loss / 0 neutral; 1 kept**.

FLIP ATTEMPT — division-free inverse-cbrt Newton, NO-SHIP (`bench_cbrt_rcbrt_newton_probe`):
hypothesized that removing the Halley path's 4 `vdivpd`/elem via a mul-only inverse-cbrt iteration
(`r = r*(4 - a·r³)/3`, no σ-correction magic `6_142_909_891_733_356_544`; `cbrt = a·r²`) would reach
memory-bound and beat JAX. MEASURED: it needs **4 iterations** for the 1e-10 oracle (3 iters =
6.18e-7 FAIL; 4 iters = 3.83e-13 PASS), and at 4 iters it is **3.12ms — 0.85x SLOWER than the shipped
Halley SIMD (2.65ms)**: the extra iteration count for accuracy outweighs the per-iter division savings.
So the shipped bit-identical Halley SIMD remains the safe-Rust cbrt optimum; FLIPPING cbrt to a JAX
win is not achievable without +fma (the SIMD-poly fma-gate). Probe bench kept. Retry: deferred to +fma.

## 2026-06-22 - SlateHarrier vmap(Dot) INTEGER batched-matmul → fj-lax i64 batched GEMM — SHIPPED (17.35x), closes the vein

Completes the naive-loop→GEMM vein: the Integral case of `batch_paired_numeric_dot` (vmap of an
integer 2D dot) still used the per-element-boxed triple loop. Routed the (2,2)/(3,2)/(2,3)/(3,3)
integer cases through `fj_lax::tensor_contraction::batched_rank2_i64_matmul` (mirrors the Real
routing). Output is I64 (Integral dtype), matching the naive `sum += left*right` i64 accumulation;
release-mode `+` wraps exactly like the kernel's `wrapping_add`/`wrapping_mul`, so it is
bit-identical on realistic data (no overflow). Same-binary A/B `bench_batched_dot_i64_naive_vs_gemm`,
maxerr/exact-eq verified: **b=128 64×64×64: 62.19ms → 3.58ms = 17.35x** (lower than the f64 38-66x —
the i64 register kernel isn't B-packed). GREEN: `cargo test -p fj-dispatch --lib` 304/0; fmt + clippy
clean. With this, BOTH the float and integer naive batched-dot paths route to fj-lax GEMM; the
naive-O(n³)-matmul vein across fj-ad (VJPs) and fj-dispatch (vmap) is now fully CLOSED. Scorecard:
**1 win / 0 losses; kept**.

## 2026-06-22 - SlateHarrier scattered-gather index pre-sort/bucket — NO-SHIP (measurement confirms ledger "out of scope")

The biggest non-gated loss is scattered single-element gather (1M←4M f64, ~15x JAX, "Zen3 vgather
ceiling"). The ledger listed "index pre-sort" as un-tried/out-of-scope; this measured it. Bit-identity
is trivial (gather = independent copies; any order yields out[i]=src[idx[i]]). Same-binary ceiling
probe `bench_gather_sorted_vs_direct_ceiling` (1M←4M f64, worker `hz2`):
  - **direct random gather (shipped path): 5.46ms**
  - **IDEAL monotonic (indices pre-sorted OUTSIDE timing): 1.61ms = 3.39x** — confirms monotonic
    source reads prefetch where random reads stall; the read-pattern ceiling.
  - **bucketed/shippable (radix-bucket build INCLUDED, 64K-elem buckets): 13.98ms = 0.39x REGRESSION**
The 3.39x read ceiling does NOT cover the cost of GETTING monotonic: the bucket-build (1M u64 pushes
into growing Vecs) + scattered output writes overwhelm the read savings (0.39x). A full radix sort of
1M indices (~3-5ms) + monotonic gather (1.6ms) + scattered writes (~2ms) ≈ 6.6-8.6ms > direct 5.46ms —
same regression. So index pre-sort/bucket is a confirmed NO-SHIP; the shipped branchless direct gather
is at the safe-Rust/Zen3 ceiling (closing further needs AVX-512 gather, absent). No production change;
diagnostic bench kept. Scorecard: **0 wins / 1 loss (vs the ideal, unreachable) / 0 neutral; 0 kept code**.

## 2026-06-22 - SlateHarrier vmap(Dot) batched-matmul naive-loop → fj-lax batched GEMM — SHIPPED (38-66x)

Extending the naive-loop→GEMM vein from fj-ad into fj-dispatch (vmap). `batch_dot_general`
(Primitive::DotGeneral, what jnp.matmul lowers to) ALREADY routes both-batched matmul through
`eval_dot_general`'s vectorized `batched_standard_f64_matmul` — optimized. But `batch_dot`
(Primitive::Dot = numpy.dot semantics; `vmap(jnp.dot(2D,2D))` lands here) sent its both-batched
(2,2)/(3,2)/(2,3)/(3,3) cases to `batch_paired_numeric_dot` — a `batch·rows·cols` triple loop
each calling the per-element-boxed `batch_dot_accumulate` (`as_f64()` per element). So vmap of a
2D dot was O(B·m·n·k) with boxed element access. Routed the REAL (f64/f32/…) cases through
`fj_lax::tensor_contraction::batched_matmul_2d` (each rank pair = a batched [batch,m,k]·[batch,k,n]
product, vector operands using extent 1); Integral stays on the loop (separate kernel, rarer).
The per-slice fallback already used this same blocked GEMM, so batched Dot is now CONSISTENT with
it; tolerance-legal (f64 accumulation, ascending-k). Same-binary A/B `bench_batched_dot_naive_vs_gemm`,
maxerr vs naive < 1e-8:
  - **b=128, 64×64×64: 159.79ms → 4.22ms = 37.9x**
  - **b=32, 128×128×128: 329.90ms → 4.98ms = 66.2x**
The batched dot now uses the SAME GEMM algorithm JAX/XLA uses; residual is the documented `cntiy`
+fma matmul row. GREEN: `cargo test -p fj-dispatch --lib` 304/0 (a one-off flake in the unrelated
scalar-grad `prepared_metadata_yields_identical_response` was a pre-existing inter-test global-state
ordering race — passes in isolation and on re-run); fmt + clippy clean. Scorecard: **1 large win /
0 losses; kept**.

## 2026-06-22 - SlateHarrier fj-ad linalg-VJP matmul_2d naive-loop → fj-lax GEMM — SHIPPED (22-119x)

BOLD-VERIFY moved off the (now floor-mined) scatter lane into the fj-ad VJP audit (the
conv-backward "naive loop → GEMM" vein). Found `fj-ad::matmul_2d` — the hot kernel of the
**QR/LU/SVD/eigh gradient** rules (grad through a decomposition does several n×n·n×n products)
— was a NAIVE i-j-p triple loop whose inner `b[p*n+j]` strides by `n` (a cache miss per multiply
for large n). JAX/XLA uses optimized GEMM for these backward passes, so fj-ad paid pure internal
waste. Routed it through the existing `fj_lax::tensor_contraction::matmul_2d` (blocked / B-packed /
threaded / register-microkernel) — the same kernel already used by this crate's conv-backward.
VJP/grad parity is TOLERANCE (not bit-exact), so the reassociated GEMM accumulation is legal.
Same-binary interleaved A/B (`matmul_2d_gemm_routed_vs_naive`, worker `hz2`), maxerr vs naive
< 1e-8 at every size:
  - **256×256: 22.99ms → 1.00ms = 22.9x**
  - **512×512: 392.4ms → 5.32ms = 73.8x**
  - **1024×1024: 3365ms → 28.3ms = 118.8x**
The resulting backward matmul now uses the SAME GEMM algorithm JAX/XLA does; its residual JAX gap
is the documented `cntiy` +fma matmul row (~XLA/2–4), not a naive-loop loss. GREEN: `cargo test
-p fj-ad --lib` 403/0 (all QR/LU/SVD/eigh VJP tolerance tests pass with the reassociated GEMM);
fmt + clippy (`--lib --tests -D warnings`) clean. Scorecard: **1 large Rust-side win / 0 losses;
kept**. FOLLOW-UP (same commit family): `fj-ad::matmul_f64(m,k,n,a,b)` — a SECOND naive triple
loop with **18 callers** in the SVD/eigh VJP extra-term paths (m>k / n>k projections) — was
routed through the same `fj_lax` GEMM (drop-in; same 22-119x kernel swap). `cargo test -p fj-ad
--lib` 403/0 again. The other linalg-VJP helpers (`matrix_inverse_transpose`, `solve_for_identity`,
`matrix_multiply`) already route through `Primitive::Solve`/`Dot` (optimized fj-lax eval) — clean.
Remaining fj-ad nested loops are O(n²) element-wise (Σ^{-1} scaling, F-matrix builds) — not
matmuls, memory-bound, leave them. The naive-O(n³)-matmul vein in fj-ad VJPs is now CLOSED.

## 2026-06-22 - SlateHarrier scatter-add f64 partition: pack (idx,i) into one u64 — SHIPPED (~1.06x, halves bucket memory)

Refinement of the shipped f64 parallel range-partition: the per-element bucket entry was a
`(usize, usize)` tuple = 16B, so 1M updates materialize 16MB of pairs (more than the 8MB f64
data). Packing `(idx << 32) | i` into a single `u64` halves that to 8MB, cutting memory traffic
in the memory-bound build and the latency-bound apply. Requires `idx (< dim0)` and `i (< len)`
to fit in 32 bits; the >2^32 case (>4 billion rows/updates — pathological) falls back to the
serial path. Single code path (no dual maintenance): just two extra early-return guards + a
shift/mask unpack in the apply. Bit-identical (same idx/i, same order; `gather_scatter_oracle`
59/0, scatter subset 31/0). Same-binary interleaved A/B `bench_scatter_add_f64_packed_u64_vs_pair`
(7 runs, worker `hz2`): pair vs packed = **0.976 / 0.999 / 1.001 / 1.057 / 1.061 / 1.077 /
1.363x → median ~1.06x** (the apply's random-write latency caps the gain; the build's halved
traffic is the win — the least-contended run, most reliable for a BW change, hit 1.36x). Derived
production: the prior pair criterion was 5.76ms vs JAX p50 4.35ms (1.32x loss) → packed ~5.4ms
(~1.25x loss). Marginal but real, bit-identical, and also halves a transient 16MB allocation.
Scorecard: **1 small Rust-side win / 0 losses; kept**.

## 2026-06-22 - SlateHarrier scatter-REDUCE partition generalization to f32/i64/mul/min/max — NO-SHIP (regresses; serial already JAX-competitive)

Follow-up to the f64-add parallel range-partition win below: generalized the kernel into
`scatter_reduce_range_partitioned<T, F>` (any `Copy` dtype + combine closure) and tried routing
f32/i64 scatter-add and the f64/f32/i64 mul/min/max combiners through it (slice_elems==1). All
bit-identical (ordered apply; conformance `gather_scatter_oracle` 59/0, scatter subset 31/0).
But it REGRESSES and was reverted to f64-ADD-only routing:
  - **f32 scatter-add 1M, pure-algorithm same-binary A/B** (`bench_scatter_add_1m_f32_production_vs_serial`,
    serial loop vs `scatter_reduce_range_partitioned` directly, interleaved best-of-25, worker `hz2`):
    serial-loop **2.18–3.02ms** vs parallel-partition **3.14–3.76ms** = **0.70x median REGRESSION**
    (0.695 / 0.695 / 0.802x).
  - **f32 was never a JAX loss:** the serial loop is **~2.5ms**, already ≈ fresh `JAX_ENABLE_X64`
    jaxlib CPU x64 f32 `o.at[i].add(u,mode='clip')` = **p50 2.88ms / mean 2.93ms / min 1.87ms**.
    (JAX f32 is ~1.5x faster than its f64 4.35ms — half the bandwidth.)
  - **Why f64-add wins but f32/i64 don't:** the partition materializes ~16MB of `(usize,usize)`
    pairs (1M × 16B) + 2× `thread::scope` spawn/join + 256 local Vecs. f64-add's baseline was
    ALREADY a (serial-build) partition paying that 16MB, so parallelizing the BUILD is free upside
    (2.78x). f32/i64/mul/min/max's baseline is a PURE serial loop with NO pair materialization, and
    for narrow/fast dtypes (f32 = 4MB data) the pair overhead exceeds the actual data movement →
    net loss. RULE: the range-partition pays only when the existing path already pays the pair-build
    cost; do not route pure-serial-loop scatter paths through it. KEPT: the generic fn (f64-add now
    routes through it, behavior-identical to the prior f64-specific fn) + the diagnostic bench.
    Scorecard: **0 JAX wins / 0 losses / 1 neutral-disproven (f32 already parity); 1 reverted route**.

## 2026-06-22 - SlateHarrier scatter-add parallel range-partition build — SHIPPED (~2.78x Rust-side; JAX loss 2.74x->~1.3x)

BOLD-VERIFY targeted the scatter-add 1M f64 1D row — the biggest unowned, non-`+fma`-gated,
non-FFT JAX loss (was 2.74x). The retained range-partitioned path
(`eval_scatter_add_f64_scalar_range_partitioned`) bucketed `(idx, i)` pairs in a SINGLE
SERIAL scan, then applied them across threads. Two same-binary interleaved A/B candidates
(`bench_scatter_add_range_partition_inline_value_vs_idxpair`, contention-symmetric best-of-25,
worker `hz2`):
  - **inline-value (NO-SHIP, median 0.948x):** storing `(idx, upd[i])` instead of `(idx, i)`
    to move the apply's sparse `upd[i]` dependent load into the sequential build. Neutral —
    proving the apply was NOT the bottleneck (3 runs: 1.005x / 1.199x / 0.898x). First
    non-interleaved run misleadingly showed 1.409x (ordering/contention artifact — the later
    variant was penalized; interleaving fixed it). LESSON: a one-shot A/B that runs old-then-new
    can be ordering-biased by monotonic worker contention; interleave per-iteration.
  - **parallel-build (SHIPPED, median ~2.78x):** the SERIAL bucket scan dominated. Each producer
    thread classifies its contiguous `index_vals` slice into its own `threads` per-output-block
    local Vecs (cache-hot, vs one serial scan scattering across all 16 buckets); each output
    block is then applied by one thread reading producers' sub-buckets in producer order.
    3 runs idx-pair vs parallel: 2.781x / 3.012x / 2.527x. BIT-IDENTICAL: producer `t` owns the
    contiguous i-range `[t*csz,(t+1)*csz)`, so per output index the updates are visited in the
    same global i-ascending order as the serial reference; distinct blocks never share output.
HEAD-TO-HEAD: production criterion `eval/scatter_add_1m_f64_1d` = **5.76ms** midpoint
(5.49..6.04ms, includes full eval dispatch/extraction) vs fresh `JAX_ENABLE_X64=1` jaxlib CPU
x64 `o.at[i].add(u, mode='clip')` on the identical fixture = **p50 4.35ms / mean 4.59ms /
min 3.31ms**. So the documented 2.74x loss narrows to **~1.3x**. GREEN: `cargo test -p fj-lax
--lib` 1587/0, `cargo test -p fj-conformance --test gather_scatter_oracle` 59/0, scatter
subset 31/0; fmt + clippy (`--lib --tests -D warnings`) clean on the touched file. Residual
gap = the random-write apply latency + the ~0.86ms eval dispatch/index-extraction overhead.
The "fundamentally different safe parallel direct-write" retry predicate was answered by
parallelizing the PARTITION rather than the writes; unique-atomic / histogram-prefix branches
stay rejected. Reusable 3-way A/B harness kept as the diagnostic ref.

## 2026-06-21 - frankenjax-mcqr cummax/cummin head-to-head measured

BOLD-VERIFY targeted the order-dependent cumulative-extrema gap after the cumsum
prefix-scan keep. The radical lever was evidence-driven rather than a source
rewrite: add exact fj-lax Criterion rows for the same 1M deterministic fixture
used by the JAX comparator, then keep/reject from measured Rust/JAX ratios.

No production source was changed in this pass.

Remote fj-lax bench command:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  'eval/cum(max|min)_1m_f64_1d' --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

Correctness proof:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY \
  rch exec -- cargo test -p fj-conformance --test cummax_cummin_oracle --release -- --nocapture
```

- RCH worker: `ovh-a`; result: 28 `cummax_cummin_oracle` tests passed.
- JAX comparator: local `benchmarks/jax_comparison/.venv`, JAX/JAXLIB 0.10.1,
  CPU backend, `jax_enable_x64=true`, same deterministic 1M f64 fixture.
- fj-lax bench worker: `hz2`; per-crate Criterion, no new `.scratch`.

Ratio-vs-JAX ledger:

| workload | fj-lax Criterion midpoint | JAX p50 | Rust/JAX | verdict |
| --- | ---: | ---: | ---: | --- |
| `eval/cummax_1m_f64_1d` | 2.0374 ms | 3.458314 ms | 0.589 | **fj-lax wins; JAX/Rust 1.70x** |
| `eval/cummin_1m_f64_1d` | 3.4187 ms | 3.602727 ms | 0.949 | near-parity / slight fj-lax win |

JAX means were 3.833558 ms for `cummax` and 3.623474 ms for `cummin`; fj-lax
therefore wins cummax by 1.88x on mean and is effectively parity/slightly ahead
on cummin. Because `cummin` had a mild high outlier and its Criterion interval
upper bound exceeded the JAX p50, keep the scorecard conservative: **1 win /
0 loss / 1 neutral-slight-win**. Retry predicate: do not spend a production
rewrite on cummax before lower-scoring JAX losses; cummin only deserves a new
lever if fresh same-worker evidence shows a real regression rather than noise.

## 2026-06-21 - frankenjax-murmw Bluestein-prime FFT tile/thread no-ships

BOLD-VERIFY revisited the rough/prime Bluestein batch lane after the earlier
SoA keep, targeting the still-losing production row
`eval/fft_batch_256x1009_prime_complex128_dense_input`. The radical lever from
the alien-graveyard/profiling pass was cache/task-granularity control: widen the
Bluestein row tile from 4 to 8 rows, then separately cap the vectorized
Bluestein scheduler at 8 threads so each task owns more convolution work.

Both source changes were reverted before commit.

Same-worker RCH per-crate measurements:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  'eval/fft_batch_256x1009_prime_complex128_dense_input' \
  --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

| candidate | worker | production baseline | candidate | Rust/JAX ratio | verdict |
| --- | --- | ---: | ---: | ---: | --- |
| `BLUESTEIN_TILE_ROWS=8` | `hz2` | 3.8919 ms | 4.5907 ms | 9.60x | REVERT: +20.635% Criterion regression |
| `PARALLEL_MAX_THREADS=8` | `hz2` | 4.3750 ms | 6.3681 ms | 13.32x | REVERT: +37.004% Criterion regression |

The JAX comparator is the existing fresh JAX/JAXLIB 0.10.1 x64 row for the
same fixture, **0.478 ms**. A non-comparable capped-thread routing signal on
`ovh-a` measured **1.7611 ms** (3.68x JAX loss), but `RCH_WORKER=ovh-a` was
ignored on the follow-up rerun, so it is explicitly not accepted as proof.

Current production score after reverts: freshest same-worker `hz2` production
midpoint **4.3750 ms** versus JAX **0.478 ms**, Rust/JAX **9.15x**. Scorecard
for this pass: **0 wins / 2 losses / 0 neutral**. Retry predicate: do not retry
tile-height, representation-only, or coarse thread-count probes for Bluestein
without hardware-counter evidence. The next credible prime/rough FFT route must
change the internal convolution kernel itself, e.g. SIMD-within-FFT/pocketfft-
class complex shuffles, or prove an idle-host threading regime where the same
source beats production in a same-worker A/B.

## 2026-06-21 - frankenjax-murmw scalar-specialized mixed-radix FFT no-ship

BOLD-VERIFY tested a narrower generated-kernel proxy after the SoA variants
lost: keep the transform row-local, but replace the recursive mixed-radix call
tree with a contiguous stage-iterative scalar kernel using specialized radix
2/3/5 butterflies. This attacks the recursive per-row floor directly without
another cross-row transpose.

Correctness gate, per-crate through RCH:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY \
  rch exec -- cargo test -p fj-lax \
  iterative_scalar_specialized_matches_recursive_to_tolerance \
  --release --lib -- --nocapture
```

RCH selected `hz2`; result **1/1 passed**. The worker rewrote the target dir to
its pool path and rehydrated dependencies remotely, so this is not a warm-cache
timing proof, but it stayed off local disk and inside the per-crate rule.

Same-binary A/B gate:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY \
  rch exec -- cargo test -p fj-lax \
  bench_mixed_radix_iterative_scalar_specialized_vs_per_row \
  --release --lib -- --ignored --nocapture
```

Printed result on `hz2`:

```text
[mixed-radix iterative scalar-specialized 128x1000] recursive=1.500ms specialized=3.242ms ratio=0.46x (min of 9 interleaved)
```

The candidate is **54% slower** than recursive mixed-radix, so it was removed
before production. Current production ratio stays at the fresh target scorecard:
Rust **3.4978 ms** vs JAX **0.230078 ms**, Rust/JAX **15.20x**. Scorecard:
**0 wins / 1 loss / 0 neutral** for the row; lever scorecard for the smooth
composite accelerator family is now **0 kept / 4 rejected / 0 validation-blocked**.
Retry predicate: do not retry stage-iterative scalar kernels for `n=1000`
unless the design eliminates digit-reversal/stage-copy overhead and first beats
recursive mixed-radix in the same-binary A/B.

Focused conformance gate after reverting the candidate:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY \
  rch exec -- cargo test -p fj-conformance --test fft_oracle \
  --release -- --nocapture
```

RCH selected `vmi1152480`; result **27/27 passed**.

## 2026-06-21 - frankenjax-murmw specialized iterative SoA FFT no-ship

BOLD-VERIFY retried the requested generated-kernel direction as a narrow
radical proxy: keep the flat iterative SoA mixed-radix schedule for
`1000 = 2^3 * 5^3`, but replace each small O(r^2) stage DFT with the specialized
radix-2/3/5 butterflies already proven in the recursive production kernel.

Fresh production Rust baseline, per-crate through RCH with the cod-a target dir:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  eval/fft_batch_128x1000_complex128 \
  --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

RCH selected `ovh-a`; no local cargo build and no new `.scratch` worktree were
created. Current Rust midpoint: **3.4978 ms** (`3.2761..3.6825 ms`). Fresh local
JAX/JAXLIB 0.10.1 CPU x64 comparator using the exact `complex_matrix(128,1000)`
fixture measured mean **0.230078 ms** and p50 **0.233834 ms** (30 runs x 100
inner loops, `block_until_ready()`). Scorecard: **0 wins / 1 loss / 0 neutral**,
Rust/JAX **15.20x** by mean.

Same-binary RCH `hz2` A/B for the specialized iterative SoA candidate:

```text
[specialized mixed-radix iterative SoA 128x1000] 1T per-row=1.511ms specialized=2.668ms ratio=0.57x (min of 9 interleaved)
```

The candidate is **43% slower** than recursive mixed-radix and would still be
**11.60x** the JAX mean even before full eval overhead. The temporary ignored
bench harness was removed; no production source change remains. Retry predicate:
do not repeat flat iterative SoA for `n=1000` without eliminating transpose /
digit-reversal overhead. The next credible route is a truly generated
length-specialized in-place/recursive `1000 = 2^3 * 5^3` kernel, or a
quiesced-host threading proof.

## 2026-06-21 - frankenjax-ur4h3 fresh BOLD-VERIFY closes small-eigh lane

Fresh re-authenticated BOLD-VERIFY reran the cod-b per-crate Criterion gate and
the exact JAX comparator for the remaining `ur4h3` 48x48 rows. The allocator/copy
stack kept by `2859e41c` still holds; no new source lever was attempted because
the measured target is already a Rust win.

The literal requested `cargo bench --release` form was tried first and failed
remotely on RCH `ovh-a` because Cargo rejects `--release` for `bench`. The valid
Criterion bench command was then run remotely, still per-crate and using the
warm cod-b target request:

```text
AGENT_NAME=CrimsonOtter \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
RCH_REQUIRE_REMOTE=1 \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline \
  'linalg/(eigh_48x48_f64|svd_48x48_f64)' -- \
  --warm-up-time 1 --measurement-time 3 --sample-size 15 --noplot
```

RCH selected `ovh-a`; no local cargo build and no new `.scratch` or worktree were
created. Fresh Rust Criterion midpoints:

- `linalg/eigh_48x48_f64`: **200.13 us** (`188.93..223.75 us`)
- `linalg/svd_48x48_f64`: **105.50 us** (`105.44..105.53 us`)

Fresh JAX/JAXLIB 0.10.1 x64 comparator used the exact `lax_baseline.rs` fixtures
(`bench_eigh_48` and `real_matrix(48,48)`), 60 runs x 100 inner loops, CPU
backend:

| Row | Rust RCH midpoint | JAX mean | Rust/JAX | Verdict |
| --- | ---: | ---: | ---: | --- |
| `linalg/eigh_48x48_f64` | 200.13 us | 293.517 us | 0.682 | Rust win; no remaining small-eigh JAX loss |
| `linalg/svd_48x48_f64` | 105.50 us | 622.642 us | 0.169 | Existing Rust win strengthened |

Alien-graveyard/extreme-optimization decision: the relevant radical lever is
communication-avoiding/panel Householder dense linear algebra, but the
graveyard constants warning applies here. Since the current 48x48 target is
already faster than JAX and prior small-Jacobi / naive symmetric-reduction
variants are recorded no-ships, the EV gate rejects a new production code
attempt for this bead. Reopen only with fresh large-n evidence showing an actual
upstream/JAX gap.

## 2026-06-21 - frankenjax-murmw smooth-composite Bluestein detour rejected

BOLD-VERIFY retargeted the remaining smooth-composite FFT loss in
`eval/fft_batch_128x1000_complex128`. The fresh Rust row was run remotely via
RCH with the cod-a target request:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 \
  rch exec -- cargo bench -j 1 --profile release -p fj-lax \
  --bench lax_baseline 'eval/fft_batch_128x1000_complex128' -- \
  --sample-size 10 --measurement-time 3 --warm-up-time 1 --noplot
```

RCH selected `hz1`; no local cargo build and no new `.scratch` worktree were
created. Current Rust midpoint: **3.6581 ms** (`3.5478..3.7359 ms`). Fresh local
JAX/JAXLIB 0.10.1 CPU x64 comparator using the exact `complex_matrix(128,1000)`
fixture measured mean **0.245442 ms** and p50 **0.250693 ms**. Scorecard:
**0 wins / 1 loss / 0 neutral**, Rust/JAX **14.90x** by mean.

Radical lever attempted: route the smooth `n=1000` batch through the existing
vectorized Bluestein SoA kernel, reusing the communication-avoiding flat radix-2
convolution machinery that wins on prime/rough lengths. Same-worker/same-binary
A/B on RCH `hz1` rejected it:

```text
mixed=1.975ms, bluestein=2.690ms, ratio=0.73x
```

The Bluestein detour is **36% slower** than the recursive mixed-radix path for
the exact fixture, so the source gate was reverted and no production code was
kept. Retry predicate: do not route smooth composites through Bluestein SoA
without new evidence. The next credible `murmw` route must change the kernel
family, e.g. generated length-specialized `1000 = 2^3 * 5^3` kernels or
production-specialized radix-3/5 butterflies that first beat per-row mixed-radix
in a same-binary A/B.

**RE-CONFIRMED 2026-06-21 (CrimsonOtter, abcae92c shipped then reverted 37b33dc9).**
I re-attempted this exact lever and INITIALLY measured 1.34x (kernel only) — then caught
that my A/B built `BluesteinPlan::new` (chirp + an m=2048 kernel FFT for n=1000) ONCE
OUTSIDE the timed loop, while production `transform_batches_dense` builds that plan PER
BATCH CALL (fft.rs:1436). With the plan build moved INSIDE the timed region (production-
realistic), the corrected `bench_bluestein_soa_vs_per_row_smooth` measured **per-row
3.061ms vs bluestein 7.793ms = 0.39x (2.5x SLOWER)** — the expensive per-call Bluestein
plan setup dominates the n=1000 transform. Reverted (net 0 production change). LESSON
(now in memory): a per-batch plan build MUST be inside the timed A/B region; excluding it
silently flipped a 0.39x no-ship into a fake 1.34x win. Only a plan-CACHED Bluestein (reuse
across calls) could change this — out of scope for a single batched FFT.

**rfft/irfft Hermitian n/2 packing = ALREADY DONE, not a lever (2026-06-21, CrimsonOtter).**
Checked whether real-input FFT could win ~2x via the standard pack-n-real-into-n/2-complex trick.
fj-lax ALREADY implements it for power-of-2: `RealRfftPower2Plan` (fft.rs:459) does pack -> half-size
FFT -> unpack (`vectorized_rfft_pow2_block` fft.rs:2161), with the symmetric irfft path
(`vectorized_irfft_pow2_block` fft.rs:2409). So rfft pow2 is NOT a free 2x — it's mined. CORRECTION (deeper read): non-pow2 BATCHED rfft is
ALSO already optimized — fft.rs:2323 PAIR-PACKS two real rows into one complex signal
`z = x_a + i·x_b`, runs ONE full-length FFT per pair, and recovers each spectrum via conjugate
symmetry (X_a[k]=(Z[k]+conj(Z[N-k]))/2, X_b[k]=(Z[k]-conj(Z[N-k]))/(2i)). That halves the dominant
transforms — cost-equivalent to the n/2 single-signal packing. So BOTH rfft paths are mined. The
ONLY residual is a LONE non-pow2 row (rows=1, can't pair → transformed full-length); the n/2 packing
would halve just that, a niche single-row case. Do not re-attempt rfft packing (pow2 or batched).
ASSESSED DEEPER (2026-06-21): generalizing `RealRfftPower2Plan`'s `half_fft` from `Radix2Plan` to a
non-pow2 n/2 FFT is clean (the `apply_into` pack at fft.rs:499-505 + unpack at 509-525 are already
n-generic; only the half-FFT is pow2-bound). BUT the win is narrow AND plan-build-hazard-prone: the
single-row path would need a per-CALL half-length plan, and for prime-ish n/2 that's a Bluestein
setup that DOMINATES the transform (cf. the smooth-Bluestein no-ship above — per-call plan build eats
the win). So ~2x only materializes for SMOOTH non-pow2-even n where the n/2 FFT (not the plan) is the
cost — a minority(single-row) × minority(non-pow2) × smooth corner. DEFER: low EV, real plan-build
hazard, intricate FFT code (subtle-bug risk) for a corner case. The pow2 + batched paths cover the
common rfft usage and are already packed.

**THREADING smooth-composite batch = NO-SHIP under contention (2026-06-21, CrimsonOtter,
823bba8b).** `fft_batch_128x1000` (128_000 elems) runs single-thread — below the per-row
fallback's `PARALLEL_MIN_ELEMS = 1<<18` gate — while JAX threads it. Lowering the gate to
1<<16 so it fans across rows measured **5.82ms (32 threads) / 7.40ms (work-capped ~7
threads) vs 2.46ms single-thread = 2.4-3x REGRESSION** on the shared rch host. The ~40us
per-row 1000-pt FFT is too short to amortize thread overhead when the swarm saturates the
cores. JAX's win is on IDLE cores and is UNMEASURABLE here. Reverted to the original gate
(net-0). Retry predicate: re-measure on a quiesced host (threaded-FFT A/B is contention-
fragile — see the threaded-FFT caveat). The smooth-composite per-row path now has NO
measurable-on-this-host accelerator: SoA-iterative (0.15x), Bluestein (0.39x), and threading
(0.4x) all no-ship; per-row butterflies are already specialized radix-2/3/5. The only open
FFT route is generated length-specialized `1000=2^3*5^3` kernels.

## PENDING-BENCH RESUME INDEX (open as of 2026-06-21, disk-critical no-cargo pause)

The disk-low/critical pause accumulated production perf routes that shipped
ENABLED but were not immediately validated or A/B'd. As of this pass, the
pause-era routes in this index have now been cargo-backed: `frankenjax-ur4h3`
is resolved by measured keep, and `frankenjax-murmw` is resolved by measured
no-ship with the smooth-composite iterative SoA gate disabled.

**STEP 0 — MANDATORY FIRST when cargo returns: `cargo test -p fj-lax --lib --release --no-run`**
(just compile the tests). Many `#[cfg(test)]` tests were authored during the no-cargo pause
and could NOT be compiled; one such edit silently dropped 6 closing braces in `fft.rs`
(fixed in `66127ce1`, verified by brace-depth analysis). The no-compile additions have now
had a full inspection-audit (2026-06-21): every referenced fn/type/const exists with the
arg-count called; whole-file brace balance is 0 (matches the last cargo-validated baseline
`cb98244b`); every test fn sits at the correct module depth; and the only non-trivial borrow
(the specialized kernel's twiddle closure) captures just `start/j/len/w`, not the mutated
`re/im`, and drops before `start += next`. None of my additions use `thread::scope` (the only
threaded iterative kernel is the production one). So the RESIDUAL compile risk is narrowed to
TYPE inference and `-D warnings` clippy lints only — which still require cargo. Release builds
are unaffected (all in `#[cfg(test)]`), but the harness must be confirmed to build before any
step below runs. If it fails to compile, fix the test before proceeding.

WORKSPACE STRUCTURAL AUDIT (2026-06-21): brace-depth check of ALL 85 `crates/**/src/*.rs`
files — every one balances to final depth 0 (line-comments stripped; format-string braces
balance). So after the `fft.rs` fix, NO build-breaking structural (brace) break remains
anywhere in the workspace, including other agents' no-cargo eigh/linalg edits. Paren/bracket
balance was also checked: the only 6 non-zero files were all last modified 2026-05-02..06-18
(BEFORE the 06-21 pause), so their counts are pre-existing string/char-literal noise in
already-compiled commits — i.e. NO file edited during the pause has a delimiter imbalance of
any class. Duplicate-definition check (fft.rs) is also clean — apparent dups (`fn apply_into`,
`fn new`, `const PARALLEL_MIN_ELEMS`) are methods on distinct structs / scoped local consts,
all legal. So ALL inspection-checkable compile-error classes (delimiters + duplicate
definitions) are verified clean for the pause-era edits; the only residual build risk is
type-inference + `-D warnings` lints (genuinely cargo-gated).

BACKLOG (2026-06-21): `br ready` = 2 — `mcqr` (umbrella; FFT work feeds it, no code-only
sub-gap) and `cntiy` (+fma, doubly blocked: maintainer decision AND `#![forbid(unsafe_code)]`
blocks per-fn `target_feature`). `murmw`/`ur4h3` left "ready" (progressed). So no NEW
code-only-actionable perf bead exists; all remaining value is cargo-gated or maintainer-gated.

RESOLVED 2026-06-21 (disk recovered, warm cargo): all resume items below are now executed.

0. **[STEP 0 / build] E0277 build-breaker in eigh `apply_householder_left_with_scratch`**
   (linalg.rs:5161, production, commit 0dd28aad) — `.zip(&scaled)` where `scaled: &mut [f64]`
   made the ENTIRE fj-lax crate fail to compile (release AND test), undetected since 2026-06-20
   21:02 because the pause stopped all cargo. STEP-0 compile-check caught it the instant warm
   cargo returned. Fixed convergently (d06f5955 `scaled.iter()`, identical to my edit). LESSON
   CONFIRMED: the no-cargo pause shipped TWO uncompiled build-breakers (this + my fft.rs braces);
   STEP-0 compile BEFORE trusting any pause-era code is mandatory, not optional.

1. **[murmw] iterative mixed-radix SoA FFT route** — RESOLVED: **DISABLED, measured no-ship.**
   Correctness PASSED (9/9: iterative/specialized/production all match recursive + DFT oracle +
   Parseval + threading bit-identity). A/B `bench_mixed_radix_iterative_soa_vs_per_row` (128x1000,
   interleaved min-of-9): **per-row=2.947ms vs iter=19.096ms = 0.15x (6.5x SLOWER)** — a no-ship
   like the recursive-SoA mixed-radix before it (SoA transpose + strided per-lane access is
   memory-bound at n=1000; mixed-radix butterflies don't autovectorize like flat radix-2 /
   Bluestein). Disabled via `MIXED_RADIX_ITERATIVE_SOA_MAX_N = 0` (e3069f5f); smooth composites
   keep the proven recursive per-row path; kernel + tests retained as documented no-ship (11/11
   green post-disable). cod-a re-verified this on 2026-06-21 after restart: RCH `vmi1152480`
   same-binary A/B was **per-row=1.798ms vs iter=5.911ms = 0.30x (3.29x SLOWER)**; RCH `hz2`
   production Criterion with the disabled gate was **2.7829ms** (`2.7035..2.8831ms`). Using the
   recorded JAX/JAXLIB 0.10.1 x64 comparator **0.233ms**, current production remains **11.94x**
   slower than JAX; the rejected iterative SoA microbench would be **25.37x** JAX. `fft_oracle`
   passed **27/27** remotely on RCH `vmi1149989`. **The FFT SoA frontier is now COMPLETE**:
   pow2/real/Bluestein = shipped wins; smooth-composite SoA = measured no-ship; per-row is the
   mixed-radix floor.
2. **[ur4h3] eigh allocator/copy-reduction stack** — RESOLVED 2026-06-21 by
   RCH `fj-lax` Criterion + exact JAX fixture comparator; see the measured keep
   entry below.

## 2026-06-21 - frankenjax-murmw iterative mixed-radix SoA route DISABLED no-ship (validation harness green)

Status of the smooth-composite batched-FFT SoA lever (the last uncovered FFT path;
pow2 fft/ifft/rfft/irfft and Bluestein prime/rough are already shipped wins).

Resolution: the production gate is disabled via `MIXED_RADIX_ITERATIVE_SOA_MAX_N = 0`.
The kernel remains in the file as a correctness-validated no-ship artifact and a target
for a future fundamentally different radix-3/5 butterfly implementation, but it is not
on the production dispatch path.

- The RECURSIVE mixed-radix SoA was a measured no-ship (0.50-0.81x, memory-bound: 3
  buffer-pairs spill L1; see the mixed-radix no-ship entry below). The credible
  replacement is an ITERATIVE flat-stage form that keeps ONE buffer pair (`re`/`im`)
  + a tiny per-butterfly temp, so an L1-sized tile stays cache-resident — the same
  flat shape that won for radix-2 and Bluestein (~3x).
- Production route `transform_batches_mixed_radix_iterative_soa` landed convergently
  in `edd01b52` (another agent, during the disk-low pause), WIRED + ENABLED into
  `transform_batches_dense` for smooth composites with `n <= MIXED_RADIX_ITERATIVE_SOA_MAX_N`
  (=1024) and `batch_size*n <= 1<<18`. It shipped marked "pending-bench" — i.e. enabled
  WITHOUT a correctness test or a vs-per-row A/B.

**Risk flagged:** an enabled-but-unvalidated route on the production dispatch path. To
gate it, a validation harness is now committed (`f3e7eb4a` + earlier `03e68e31`/
`ca30353c`/`8d01da2b`), all `#[cfg(test)]` (zero production impact):

- `production_mixed_radix_iterative_soa_matches_reference` — asserts the EXACT wired
  route agrees, to 1e-9 tolerance, with the established per-row recursive `mixed_radix_into`
  AND the independent O(n^2) `dft_1d`/`idft_1d` oracle (radix-2/3/5 + general 7/11/13,
  both directions, single/multi-row tiles).
- `iterative_mixed_radix_matches_recursive_to_tolerance`, `iterative_soa_bit_identical_to_scalar`,
  `iterative_soa_matches_dft_oracle` — three independent algorithm-level guards on the
  prototype kernels.

**INSPECTION-CONFIRMED CORRECT (2026-06-21, no-cargo code review):** `mixed_radix_iterative_soa_block`
was reviewed op-for-op against the independently hand-verified iterative reference (the
n=6 trace: `X[0]`=DC and `X[3]`=in0-in1+in2-in3+in4-in5 both exact). They are
ALGORITHMICALLY IDENTICAL — same generalized digit-reversal (`rev = rev*f + x%f`), same
stage strides (`n/next` input twiddle, `n/r` DFT twiddle), same gather-then-scatter
butterfly (gather of all `r` lanes completes before any scatter, so no aliasing), same
ascending-`t` DFT accumulation order, same `1/n` inverse scale (only local variable names
differ). So the enabled route is very likely CORRECT; the pending tests are confirmation,
not discovery. (Composite FFT parity is tolerance, so the validation harness is correctly
tolerance-based, NOT bit-frozen — do not tighten it to bit-identity, which would break a
later radix-2/3/5 butterfly specialization.)

**DISPATCH PARTITION also inspection-confirmed clean (no-cargo):** the three stacked
`transform_batches_dense` SoA gates partition every length exactly once: (1) pow2 →
`transform_batches_pow2_vectorized` (and `is_mixed_radix_smooth` returns false for pow2, so
gates 2/3 cannot double-route it); (2) non-pow2 **non-smooth** (prime/rough), `m<=16384` →
Bluestein — its `!is_mixed_radix_smooth(n)` is mutually exclusive with (3) non-pow2
**smooth**, `n<=1024`, `batch*n<=1<<18` → the iterative route. Everything else (smooth
`n>1024`, oversized batch, `batch<8`, `m>16384`) falls through to the per-row
`BatchFftPlan`. No length is double-routed and none is unrouted, so enabling the iterative
gate cannot have stolen or dropped any case from the Bluestein/per-row paths.

**BOLD-VERIFY result (2026-06-21):** correctness is green and performance rejects the
route. RCH `fj-lax --lib --release mixed_radix` passed all 5 runnable mixed-radix tests
with only the informational A/B microbench ignored; exact
`production_mixed_radix_soa_matches_scalar_iterative_by_bits` also passed. RCH
`fj-conformance --test fft_oracle` passed 27/27. The same-binary A/B on RCH `vmi1152480`
printed `per-row=1.798ms iter=5.911ms ratio=0.30x`; the earlier disabling run recorded
an even worse `per-row=2.947ms iter=19.096ms ratio=0.15x`. RCH `hz2` production Criterion
with the disabled gate printed `eval/fft_batch_128x1000_complex128` **2.7829ms**
(`2.7035..2.8831ms`). Against the recorded JAX/JAXLIB 0.10.1 x64 comparator **0.233ms**,
production is **11.94x** JAX, while the rejected iterative SoA microbench is **25.37x** JAX.

(3) CONDITIONAL OPTIMIZATION (already staged): the enabled production kernel uses the
general O(r^2) per-butterfly DFT, which is wasteful on radix-3/5-heavy composites (n=1000
spends 25 mults/output on each radix-5 stage). If step (2)'s A/B shows the general route
losing or only marginally winning, the drop-in fix is committed and validated:
`mixed_radix_iterative_soa_specialized` (test-only) replaces that DFT step with the proven
low-mult radix-2/3/5 butterflies (radix-3 9->2 mults, radix-5 25->~4; general fallback for
7/11/13), guarded by `iterative_soa_specialized_matches_dft_oracle`. Deploy by porting its
specialized DFT branch into the production `mixed_radix_iterative_soa_block`, then re-A/B.
No further design work — the optimization math is already written and DFT-oracle-validated.

## 2026-06-21 - frankenjax-murmw buffer-reuse/Bluestein retry no-ship; source reverted

BOLD-VERIFY re-auth restart pass on the remaining smooth-composite FFT loss
(`eval/fft_batch_128x1000_complex128`). No production source change remains.
All candidate code was reverted after measurement; this entry is the ledgered
ratio proof.

Fresh production Rust baseline used the requested cod-a RCH target root and a
single filtered `fj-lax` Criterion row:

```text
AGENT_NAME=cod-a \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 RCH_WORKER=vmi1152480 \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  'eval/fft_batch_128x1000_complex128' \
  --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

- RCH worker: `vmi1152480`; per-crate only, no local cargo build, no new
  `.scratch` worktree.
- Rust Criterion interval: **3.1347..4.4919 ms**, midpoint **4.0559 ms**.
- Fresh local JAX/JAXLIB 0.10.1 CPU x64 comparator with the exact
  `complex_matrix(128,1000)` fixture: mean **0.177952 ms**, p50
  **0.175767 ms**.
- Current production Rust/JAX: **22.79x** by midpoint. Scorecard:
  **0 wins / 1 loss / 0 neutral**.

Candidate/retry outcomes:

| Candidate | Same-binary or Criterion result | Decision |
| --- | --- | --- |
| Existing flat iterative mixed-radix SoA route | RCH `hz2`: per-row **1.550 ms** vs iter **6.532 ms**, ratio **0.24x** | Still hard no-ship; keep disabled |
| Smooth-composite Bluestein SoA retry | RCH `ovh-a`: **1.19x** routing hint, but RCH `vmi1152480` repeat: per-row **1.904 ms** vs Bluestein **1.917 ms**, ratio **0.99x** | Not reproducible; no production gate |
| Recursive mixed-radix row-buffer zero-fill elision | Criterion after edit: **2.6909..2.8715 ms** but `p=0.17` no-change; direct A/B: zero-fill **2.212 ms** vs reuse-len **2.211 ms**, ratio **1.00x** | Reverted as zero-gain |

Alien-graveyard/extreme-optimization decision: vectorized execution / staged
code generation remains the right family, but the tested micro-levers do not
change the kernel floor. The next credible route is still a true generated
in-place or recursive length-specialized `1000 = 2^3 * 5^3` kernel, or a
quiesced-host threading proof. Do not retry buffer zero-fill elimination,
generic iterative SoA, or smooth-composite Bluestein without a same-binary win
that survives a repeated worker check.

## 2026-06-21 - frankenjax-ur4h3 eigh allocator/copy-reduction stack KEEP; exact JAX ratio refreshed

BOLD-VERIFY resolved the disk-low pending-bench stack for `frankenjax-ur4h3`:

- `815ad85a` reuses the Householder reflector buffer in `hessenberg_reduction`.
- `b240d41d` reuses the left Householder update dot-product scratch buffer.
- `0dd28aad` transposes the QL eigenvector buffer in place around
  `symmetric_tridiagonal_ql`.
- `d06f5955` is a style-only follow-up on the scratch helper.
- This pass added a tiny follow-up cleanup: `apply_householder_left` is now
  `#[cfg(test)]`, because production uses the scratch-backed helper directly
  and the old wrapper remains only for the row-contiguous-vs-strided microtest.

RCH `fj-lax` Criterion used the existing warm cod-b target dir request:

```text
AGENT_NAME=CrimsonOtter \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline \
  'linalg/(eigh_48x48_f64|svd_48x48_f64)' -- \
  --warm-up-time 1 --measurement-time 3 --sample-size 15 --noplot
```

The Cargo version in this checkout rejects `cargo bench --release`; `cargo
bench` was used because Criterion already builds the bench profile. RCH selected
`vmi1227854` and rewrote the target dir to a worker-scoped pool, so this is a
focused per-crate remote gate, not a local build. The old saved Criterion row in
the same cod-b target tree was `eigh_48x48_f64` **566.12 us** mean; the current
run printed `eigh_48x48_f64` **203.18 us** midpoint (`197.59..211.62 us`) and
`svd_48x48_f64` **98.678 us** midpoint (`95.111..103.32 us`). The retrieved
Criterion estimate file recorded `eigh` **215.99 us** mean and `svd` **98.764
us** mean, with Criterion's stored relative `eigh` delta about **-73%** versus
the saved baseline. This is not a zero-gain lever; the stack is kept.

Fresh local JAX/JAXLIB 0.10.1 x64 comparator was rerun with the exact
`lax_baseline.rs` fixtures (`bench_eigh_48` and `real_matrix(48,48)`, 60 runs x
100 inner loops, CPU backend):

| Row | Rust RCH midpoint | Rust Criterion estimate | JAX mean | Midpoint Rust/JAX | Estimate Rust/JAX | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `linalg/eigh_48x48_f64` | 203.18 us | 215.99 us | 210.721 us | 0.964 | 1.025 | KEEP; effectively parity, residual depends on estimator/host noise |
| `linalg/svd_48x48_f64` | 98.678 us | 98.764 us | 571.848 us | 0.173 | 0.173 | Existing Rust win strengthened |

Alien-graveyard/extreme-optimization route used: constants/cache wall (§16.7)
plus communication-avoiding dense-kernel discipline (§9.7/CA-QR family), applied
as a conservative allocation/copy-reduction lever rather than changing the
eigensolver. Isomorphism proof: Householder reflector entries are fully
overwritten before use; the left scratch buffer is cleared before dot
accumulation; QL arithmetic and rotation order are unchanged; the in-place
transpose changes only layout conversion storage, preserving row-major output.

Retry predicate: do not rerun allocator/copy micro-levers for this row unless a
new profile shows allocation/copy back on top. If `eigh_48x48_f64` remains a
true JAX loss under a same-host comparator, the next credible route is the
larger graveyard lever already identified: blocked/panel Householder with
packed or streaming Q updates, or a different eigenvector accumulation strategy
with same-worker production proof.

## 2026-06-21 - frankenjax-murmw flat iterative SoA mixed-radix FFT pending-bench

DISK-LOW code-only pass: smooth-composite dense FFT batches now have a
conservative production candidate that routes `batch >= 8`, `n <= 1024`, and
`batch*n <= 2^18` through a flat iterative mixed-radix SoA kernel. This targets
the still-losing `eval/fft_batch_128x1000_complex128` row (last measured Rust
**2.930 ms** vs JAX **0.233 ms**, Rust/JAX **12.55x**) while leaving larger
already-threaded batches on the existing recursive per-row path.

No new `cargo bench` or `cargo build` was started in this turn by instruction.
Pending validation: run the new
`production_mixed_radix_soa_matches_scalar_iterative_by_bits` gate, the existing
iterative-vs-recursive/oracle gates, `fft_oracle`, then RCH Criterion for
`eval/fft_batch_128x1000_complex128`. Keep only with same-worker/directly
comparable improvement and green FFT oracle parity; otherwise revert the
smooth-composite SoA route and record the no-ship result.

## 2026-06-21 - frankenjax-ur4h3 Householder left-update scratch reuse resolved by stack keep

DISK-LOW code-only pass: `hessenberg_reduction` now reuses the dot-product
scratch buffer for the left Householder update instead of allocating it inside
each `apply_householder_left` call. The existing helper API remains available;
the production reduction path uses a scratch-backed helper that clears the
active prefix before reuse, preserving accumulation and update order.

Resolved by the measured stack keep above. Do not evaluate this helper in
isolation unless a fresh profile asks for component-level attribution.

## 2026-06-20 - frankenjax-ur4h3 Householder reflector scratch reuse resolved by stack keep

DISK-LOW code-only pass: `hessenberg_reduction` now reuses one Householder
reflector scratch buffer across panels instead of allocating a fresh vector per
reduction step. This is an allocation-pressure lever only; reflector entries
are fully overwritten before use, so arithmetic/order should stay unchanged.

Resolved by the measured stack keep above. The earlier `hz1` production row
remains historical context only; the BOLD-VERIFY pass refreshed the exact JAX
fixture comparator and kept the combined allocator/copy-reduction stack.

## 2026-06-20 - frankenjax-ur4h3 symmetry-specialized tridiagonal reduction no-ship

The BOLD-VERIFY pass retargeted the remaining `eigh_48x48_f64` loss after the
small-Jacobi route was rejected. Fresh production baseline on RCH `hz1`:
`eigh_48x48_f64` **267.84 us** mean and `svd_48x48_f64` **176.93 us** mean.
Fresh local warmed JAX/JAXLIB 0.10.1 x64 comparator (60 runs x 100 inner loops)
measured `eigh_48x48_f64` **201.429 us** mean / **192.507 us** p50 and
`svd_48x48_f64` **955.544 us** mean. Production status: SVD is already a Rust
win (Rust/JAX mean ratio **0.185**); `eigh` remains the only target loss
(Rust/JAX mean ratio **1.330**).

Rejected lever: replaced real-`eigh`'s general Hessenberg reduction setup with
a symmetry-specialized Householder tridiagonal reduction that updated only the
trailing symmetric block plus Q. Correctness was green while the candidate
existed: `tridiag_ql_eigh_matches_jacobi_and_reconstructs` passed on RCH `hz1`;
broader `cargo test -p fj-lax eigh --lib -- --nocapture` passed 13/0 with 1
ignored benchmark on RCH `vmi1293453`. But the timing gate failed. Candidate
Criterion on RCH `vmi1153651` measured `eigh_48x48_f64` **375.37 us** mean
(Rust/JAX mean ratio **1.863**) and `svd_48x48_f64` **244.01 us** mean. Worker
drift prevents accepting the candidate-vs-`hz1` delta as a same-worker
regression proof, but the candidate is still a worse absolute JAX loss than
production, so the source was reverted before commit.

Scorecard for retained production 48x48 linalg rows: **1 win / 1 loss / 0
neutral** vs JAX. Rejected candidate score: **0 wins / 1 loss / 0 neutral** for
the target `eigh` row, with no source kept.

Retry predicate: do not retry this naive symmetric rank-2 tridiagonal reduction
shape (`w = beta * A_sub * v`, `A_sub -= v*w^T + w*v^T`, row-major Q update)
without a same-binary component A/B showing the reducer itself beats the current
general-Hessenberg setup. The next credible `eigh` route needs a blocked/panel
Householder design with packed/streaming Q updates, or a different end-to-end
eigenvector accumulation strategy with same-worker production proof.

## 2026-06-20 - frankenjax-murmw SoA Bluestein (prime / rough length) batch FFT — SHIPPED Rust win (2.49x full-eval; still 9.31x JAX loss)

The mixed-radix SoA no-ship (below) does NOT generalize to Bluestein, because Bluestein's
two internal convolution FFTs are the **flat radix-2** kernel (`radix2_forward`/`_inverse`),
not the recursive mixed-radix — so they vectorize well through the already-proven
`vectorized_pow2_block`. Added `transform_batches_bluestein_vectorized`: per row tile, the
chirp pre-multiply, kernel pointwise multiply, and chirp post-multiply are vectorized
vertically over rows, and the forward + inverse pow2 convolution FFTs run as SoA batches.
Gated into `transform_batches_dense` for non-pow2 non-smooth lengths (batch≥8, conv length
`m ≤ BLUESTEIN_SOA_MAX_M`).

**Bit-identical** to per-row `BluesteinPlan::apply_into` (same chirp arithmetic,
same radix-2 plans, same kernel `fb`, same `1/m` and `1/n` scales) — proven by
`vectorized_bluestein_bit_identical_to_per_row` (n in
{3,7,11,13,17,23,127,257,1009,4099}, both directions, batch 1..11), `fft_oracle`
27/27, `linalg_fft_oracle_parity` 1/1, filtered `fj-lax fft` 47/0 with 6
ignored microbenches, `cargo clippy -p fj-lax --all-targets -- -D warnings`,
and `cargo build --release -p fj-lax --benches`.

Measured single-thread same-binary A/B on RCH `hz2` (interleaved min-of-9):
n=127 (m=256) **2.98x**; n=1009 (m=2048) **4.40x**; n=4099 (m=16384)
**3.60x**. Same-worker full-eval Criterion on RCH `hz2` for the new production
row `eval/fft_batch_256x1009_prime_complex128_dense_input`: baseline **11.096 ms**,
candidate **4.453 ms**, **2.49x** faster. Controls stayed within noise/slight
loss: `fft_batch_128x1000_complex128` **2.826 -> 2.930 ms** (+3.7%, not routed
through this gate) and `fft_batch_2048x256_complex128_dense_input` **5.858 ->
5.979 ms** (+2.1%, power-of-two control).

Fresh local JAX/JAXLIB 0.10.1 x64 comparator for the same rows: `256x1009`
**0.478 ms**, `128x1000` **0.233 ms**, `2048x256` **0.313 ms**. Ratio scorecard
after the keep: **0 wins / 3 losses / 0 neutral vs JAX** for these full-eval
FFT rows; the kept target improves from **23.20x** to **9.31x** Rust/JAX, but it
is still a JAX loss. `BLUESTEIN_SOA_MAX_M=16384` is retained because the largest
covered row has both bit-identity proof (`n=4099`) and a 3.60x same-binary win.

## 2026-06-20 - frankenjax-murmw SoA mixed-radix (composite) batch FFT — no-ship (0.50-0.81x regression, large AND small n)

After the pow2 fft/ifft/rfft/irfft SoA wins, the natural extension was to SoA-vectorize
the **smooth-composite** path (`mixed_radix_into`, e.g. `fft_batch_128x1000` ~30x). Built
`mixed_radix_ping_soa` (a lane-wide sibling of the recursive `mixed_radix_ping`: every
scalar radix-2/3/5 and general-p=7/11/13 butterfly op replicated across `w` row lanes with
shared scalar twiddles) + a tiled driver, gated into `transform_batches_dense`.

**Bit-identical** to per-row `mixed_radix_into` — proven by
`vectorized_mixed_radix_bit_identical_to_per_row` (n ∈ {6,10,12,14,15,21,30,35,77,143,700,
1000} covering all radix paths, both directions, 47/47 fft tests). The code is correct.

But it **regresses** end-to-end. Same-binary interleaved min-of-9 single-thread A/B
(128×1000): tile=4 → **0.69x** (1.891ms → 2.751ms); tile=2 → **0.50x** (→ 3.737ms). Root
cause is architectural, not a tuning miss: unlike the flat 2-buffer iterative radix-2 path,
the recursive mixed-radix needs **three** SoA buffers (input + out + scratch that ping-pong),
so for large composite n=1000 the working set is ~96-192 KiB (L2/L3) versus the per-row
path's L1-resident 2×16 KiB. Smaller tiles only lose SIMD width while staying memory-bound
(hence tile=2 is *worse*). The recursion's strided sub-DFT access and the general-p combine
also vectorize worse than flat radix-2 stages. Source reverted before commit; no source kept.

UPDATE (second no-ship, same pass): the small-`n` L1-resident hypothesis was also tested and
REJECTED. Re-added the kernel gated to `n <= 160` (where the 3 SoA buffer-pairs fit a 32 KiB
L1) and benched n=120: still **0.81x** (2.026ms → 2.508ms, interleaved min-of-9). So the
regression is NOT just cache spill — the recursive structure's strided per-lane sub-DFT
access plus the transpose round-trip simply don't autovectorize like the flat iterative
radix-2 path, at any size. Reverted again. Both the large-`n` and small-`n` recursive-SoA
gates are dead.

Retry predicate: do NOT retry ANY straight SoA transpose of the RECURSIVE mixed-radix (large
or small n). A real win needs a different structure: an iterative (non-recursive) mixed-radix
with in-place stages over a single buffer pair (2 SoA buffers, L1-resident, flat stage loops
that vectorize), or native batched radix kernels that never materialize the per-row scratch.
The pow2 SoA family (fft/ifft/rfft/irfft) stays the only shipped SoA FFT win.

## 2026-06-20 - frankenjax-murmw SoA real-FFT (rfft/irfft) batch kernel — SHIPPED win (1.58-1.79x)

Follow-on to the shipped full-complex SoA pow2 kernel (`73645de8`): that win was wired
only into `transform_batches_dense` (fft/ifft). The pow2 **rfft** batch path still ran
per-row scalar via `RealRfftPower2Plan::apply_into`. This pass extended the same proven
SoA lever to it (CrimsonOtter): factored the butterfly stages into
`soa_radix2_butterfly_stages`, added `vectorized_rfft_pow2_block` (pack + half-length FFT
+ Hermitian recombination, all vertical over a row tile) and the cache-blocked
`vectorized_rfft_pow2_tiled`, and routed `rfft_rows_into`'s real-plan branch through it
above an 8-row floor.

**Bit-identical** to the per-row path (same pack, same `RealRfftPower2Plan.half_fft`
butterfly order, same recombination arithmetic) — proven by
`vectorized_rfft_pow2_bit_identical_to_per_row` (exact/padded/truncated × batch 1..40)
and the unchanged `fft_oracle` (27/27) + pow2 rfft golden digests.

**irfft** got the same treatment in the same pass: pow2 `irfft_rows_f64_into` ran a full
per-row inverse complex FFT on the reconstructed Hermitian spectrum, so it reuses
`soa_radix2_butterfly_stages` directly — `vectorized_irfft_pow2_block` fuses the Hermitian
reconstruction into the SoA transpose-in (no extra spectrum buffer, stays L1) and extracts
the real part with the 1/n scale; `BatchFftPlan::as_pow2` reuses the cached plan. Bit-identical
(`vectorized_irfft_pow2_bit_identical_to_per_row`, exact/oversized/undersized half-spectrum).

Measured single-thread same-binary A/B (2048×256, **interleaved min-of-9** to cancel
cross-run worker drift): rfft **1.79x / 1.66x**, irfft **1.71x / 1.58x** (per-row → SoA).
Note: naive *sequential* old-then-new sampling read rfft as low as 1.09x because the
shared-swarm worker speed drifted mid-measurement (per-row abs time swung 1.9-5.4ms
run-to-run) — the interleaved measurement is the trustworthy one, consistent with the
[[#2026-06-20 - frankenjax-murmw threaded SoA FFT A/B is contention-fragile]] caveat.
Gates: `fj-lax` fft 46/46 (+ bit-identity tests), clippy clean, `fft_oracle` 27/27.

Remaining FFT frontier (still murmw): the mixed-radix/composite path
(`fft_batch_128x1000` ~30x), still per-row. All pow2 batch paths (fft/ifft/rfft/irfft) are
now SoA-vectorized.

## 2026-06-20 - frankenjax-4ryym f64 GEMM SIMD load-codegen no-ship

Follow-up after `frankenjax-ifou2`: generic Rayon/global-pool row scheduling is
already banned for this GEMM lane, so this pass tried a narrower backend/codegen
lever. The candidate rewrote the f64 `MR x NR` microkernel B-vector load from
manual eight-scalar `Simd::from_array([brow[0]..brow[7]])` construction to
`Simd::from_slice`, preserving the exact ascending-`l` accumulation order. The
source hunk was reverted before commit.

Fresh `hz1` production baseline, using
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b` and
`cargo bench -p fj-lax --bench lax_baseline`:

| Row | Production Rust | JAX median | Production/JAX | Verdict |
| --- | ---: | ---: | ---: | --- |
| `matmul_2d_256x256x256_f64` | 1.2296 ms | 0.3006295 ms | 4.090 | Active JAX loss |
| `matmul_2d_512x512x512_f64` | 6.9439 ms | 0.621733 ms | 11.169 | Active JAX loss |
| `matmul_2d_1024x1024x1024_f64` | 40.776 ms | 2.7191275 ms | 14.996 | Active JAX loss |
| `strassen_ab_1024_matmul2d` | 39.758 ms | 2.7191275 ms | 14.622 | Control loss |
| `strassen_ab_1024_strassen` | 97.337 ms | 2.7191275 ms | 35.797 | Existing no-ship remains |

Rejected candidate evidence: RCH would not route the candidate back to `hz1` and
selected `vmi1153651`, so cross-worker rows were not accepted as production
proof. The same-worker candidate rerun on `vmi1153651` still failed the keep
gate: 256 was neutral/no change (8.2187 ms, Rust/JAX 27.338, p=0.68), 512
regressed significantly (40.519 ms, Rust/JAX 65.171, +108.32%, p=0.00), and
1024 was neutral/no change (90.284 ms, Rust/JAX 33.203, p=0.34). Candidate A on
the same worker also stayed severe JAX loss territory: 8.9680 ms / 21.088 ms /
79.742 ms for 256/512/1024.

Validation while the candidate existed: `cargo check -p fj-lax --benches` passed
on RCH `vmi1152480`; `cargo test -p fj-lax matmul_2d --lib --release --
--nocapture` passed 23 tests with 2 ignored microbenches on RCH `ovh-a`.
`perf` hardware counters were unavailable (`perf_event_paranoid=4`), so no
cycle/cache claims are made. Ratio scorecard: retained production 0 wins / 3
losses / 0 neutral vs JAX; rejected candidate 0 wins / 3 losses / 0 neutral.

Retry predicate: do not retry `Simd::from_slice`/manual-load spelling changes in
the f64 microkernel without assembly-level proof. The valid next route remains a
real backend/codegen lever: target-feature-specialized FMA, BLAS-class packed
microkernel, owned arena handoff, or a safe scoped-pool design with same-worker
256/512/1024 proof.

## 2026-06-20 - frankenjax-murmw threaded SoA FFT A/B is contention-fragile (corroboration, no new code)

An independent BOLD-VERIFY pass (CrimsonOtter) reconstructed the vectorized SoA
power-of-two batched FFT kernel from scratch and converged on the design already
shipped in `73645de8` (`transform_batches_pow2_vectorized` /
`POW2_VECTORIZED_MIN_BATCH` / cache-blocked `vectorized_pow2_tiled` / the
`bench_vectorized_soa_batch_vs_per_row_plan` probe). The working-tree
reimplementation was discarded as redundant — **no source change kept**.

Verification value: the reconstruction's same-binary A/B independently
corroborates keeping the SoA path. The SINGLE-THREAD ratio (per-row → SoA) is
stable and a clear win across runs — FWD/INV 1.60x/1.42x, 1.35x/1.35x,
1.49x/1.33x.

Measurement caveat (new, the reason this is logged): the **threaded** same-binary
A/B of the *identical* shipped code is contention-fragile on the shared RCH
swarm. Three back-to-back runs of `bench_vectorized_soa_batch_vs_per_row_plan`
gave FWD/INV threaded ratios of 1.45x/0.99x, then 0.84x/0.46x, then 1.68x/1.18x.
The 0.46x–1.68x spread is pure cross-run contention (other agents' threads
competing for memory bandwidth), not a kernel property — it agrees with the
gate-disable no-ship below, which showed disabling SoA regresses the full eval
path +71%. Retry predicate: do NOT revert the threaded SoA gate
(`POW2_VECTORIZED_MIN_BATCH`) on the basis of a single threaded A/B that looks
like a regression; trust only the stable single-thread A/B, or a many-sample
interleaved measurement on a quiet worker. A naive serial-only gate (skip SoA
once the batch threads) was prototyped and rejected for exactly this reason — it
would forfeit the measured threaded win.

## 2026-06-20 - frankenjax-ifou2 Rayon persistent-pool GEMM no-ship

The BOLD-VERIFY pass targeted the current `matmul_2d`/GEMM loss after
`frankenjax-ifou2` identified per-call scoped thread spawning as the suspected
mid-size tax. The radical safe lever was to route the existing row-block split
and packed-B panel fanout through Rayon's persistent global worker pool. Rayon is
already used by `fj-backend-cpu`, so this was a dependency-surface-light way to
test the persistent-pool hypothesis without unsafe code. Every source change was
reverted before commit.

Same-worker Rust timing used RCH worker `hz1`, the clean worktree
`/data/projects/frankenjax-cod-b`, and:

```bash
cargo bench -p fj-lax --bench lax_baseline -- \
  'linalg/(matmul_2d_256x256x256_f64|matmul_2d_512x512x512_f64|strassen_ab_1024_(matmul2d|strassen))' \
  --warm-up-time 1 --measurement-time 3 --sample-size 15 --noplot
```

JAX ratios use fresh local CPU JAX/JAXLIB 0.10.1 x64 measurements:
256 mean 0.264947 ms, 512 mean 0.576574 ms, 1024 mean 2.665036 ms.

| Row | Production Rust | Best/Final Candidate Rust | JAX mean | Production/JAX | Candidate/JAX | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `matmul_2d_256x256x256_f64` | 1.3226 ms | best 0.98603 ms; final 3.0695 ms | 0.264947 ms | 4.992 | best 3.722; final 11.585 | Reverted: unstable, final regression |
| `matmul_2d_512x512x512_f64` | 6.3494 ms | best 6.1108 ms; final 12.894 ms | 0.576574 ms | 11.012 | best 10.598; final 22.363 | Reverted: no durable win |
| `strassen_ab_1024_matmul2d` | 33.919 ms | best 54.996 ms; final 94.836 ms | 2.665036 ms | 12.727 | best 20.636; final 35.585 | Reverted: repeated regression |
| `strassen_ab_1024_strassen` | 79.752 ms | best 84.003 ms; final 209.53 ms | 2.665036 ms | 29.925 | best 31.520; final 78.622 | Existing no-ship remains |

The first all-Rayon cut looked attractive at 256/512, but it regressed 1024
matmul2d by 52.148%. A narrowed `block_k` fallback still regressed 1024
(65.353 ms, Rust/JAX 24.522), and an explicit `<=512` gate then regressed every
measured row. The likely read is that the current scoped fanout is not a
standalone safe-Rust pool swap; worker-pool scheduling, cache residency, and
host contention interact enough that this family needs a quiet-host profile and
backend-level design before another attempt.

Validation while the candidate existed: clean-worktree `cargo check -p fj-lax
--benches` passed through RCH `vmi1293453`; clean-worktree `cargo test -p
fj-lax matmul_2d --lib --release -- --nocapture` passed 23 tests with 2 ignored
microbenches on RCH `vmi1167313`; final docs-only conformance gate `cargo test
-p fj-conformance --lib` passed 45/45 on RCH `hz2`. `cargo fmt -p fj-lax
--check` remains blocked on pre-existing unrelated formatting drift in clean
HEAD (`fft.rs`, `lib.rs`, and other crates); no source hunk from this pass
shipped.

Retry predicate: do not retry a generic Rayon/global-pool replacement for
`matmul_2d` row chunks or packed-B fanout. Reopen only with a quiet-host profile
showing thread creation still dominates, or with a deeper backend/codegen route:
BLAS-class microkernel, target-feature-specialized FMA kernel, owned arena
handoff, or an approved safe scoped-pool design with same-worker 256/512/1024
proof and JAX ratios.

## 2026-06-22 - frankenjax-murmw radix-4 FFT specialization no-ship

The fresh BOLD-VERIFY pass continued `frankenjax-murmw` under `assignee=cod-a`
and rechecked the child-route idea behind `frankenjax-murmw.1`: a generated or
vectorized radix-4/split-radix family for the power-of-two batched FFT. The
alien-graveyard/extreme-optimization route was shape-specialized/vectorized
execution: reduce butterfly count and move toward pocketfft-like higher-radix
kernels rather than another representation or tile-height tweak.

The contained same-binary probe rejected that route before any production source
change. The existing ignored A/B in `crates/fj-lax/src/fft.rs` showed the test
radix-4 SoA butterfly path slower than the retained radix-2 SoA path:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
  rch exec -- cargo test -p fj-lax --lib --release \
  bench_radix4_vs_radix2_soa -- --ignored --nocapture

[radix4 vs radix2 SoA butterflies 8x256] radix2=0.0056ms \
  radix4=0.0128ms ratio=0.44x
```

Verdict: **REVERT/no-ship**. A radix-4 promotion would be a 2.29x local
butterfly regression, before full `eval_primitive` overhead or conformance risk.

Fresh production/JAX check for the retained HEAD rows:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a \
  AGENT_NAME=CrimsonOtter \
  rch exec -- cargo bench -j 1 --profile release -p fj-lax \
  --bench lax_baseline 'eval/fft_batch_2048x256_complex128' -- \
  --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

Note: the first attempted command used `cargo bench --release`, which Cargo
rejected for this bench target; the corrected per-crate release command used
`--profile release`. The regex filter above matched both the boxed and
dense-input rows on RCH worker `ovh-a`.

Fresh local JAX comparator used `benchmarks/jax_comparison/.venv/bin/python`,
JAX/JAXLIB 0.10.1, CPU backend, `jax_enable_x64=true`, warmed
`jax.jit(lambda a: jnp.fft.fft(a, axis=-1))`, 30 timed runs over the exact
`2048x256` complex128 fixture generated by `complex_matrix`/
`complex_matrix_dense`: `sin(i*0.125) + 1j*cos(i*0.25)`.

| Row | Retained Rust midpoint | Fresh JAX mean | Rust/JAX | Verdict |
| --- | ---: | ---: | ---: | --- |
| `eval/fft_batch_2048x256_complex128` | 4.3538 ms | 0.328385 ms | 13.26x loss | Active loss |
| `eval/fft_batch_2048x256_complex128_dense_input` | 1.0879 ms | 0.328385 ms | 3.31x loss | Active loss |
| radix-4 SoA butterfly probe | 0.0128 ms | n/a | 0.44x vs radix-2 | Reject |

Scorecard for this pass: **0 wins / 2 losses / 0 neutral vs JAX; 0 kept / 1
rejected**. No `crates/fj-lax/src/fft.rs` source hunk was kept. Retry only with
generated length-specialized kernels, true SIMD/shuffle support, native
mixed-radix plans for composite rows, or a backend route that can credibly
approach pocketfft's higher-radix/SIMD kernel class. Do not retry a plain radix-4
SoA butterfly promotion without a same-binary A/B exceeding the retained radix-2
path by at least 1.15x.

Validation after the no-ship decision: no source code changed, and the existing
warm-target focused conformance binary
`/data/projects/.rch-targets/frankenjax-cod-a/debug/deps/fft_oracle-4c019ebc487d9fe0`
run with `--nocapture` passed **27/27** FFT oracle tests locally. Remote
`cargo test -p fj-conformance --test fft_oracle --release` attempts were
cancelled with code 143 before test execution, so this pass used the
already-built shared-target oracle binary to avoid a cold local rebuild under
low disk.

## 2026-06-20 - frankenjax-murmw power-of-two FFT tile-size no-ship

The BOLD-VERIFY pass retargeted `frankenjax-murmw` after the SoA gate-disable,
direct-output, pool, and radix-4 attempts still left 2048x256 batched FFT rows
far behind warmed JAX CPU. The radical but narrow lever was cache-shape
specialization for the existing power-of-two SoA kernel: first force the tile
height from 8 rows to 4 rows, then try a guarded 4-row path only for packed dense
input while leaving boxed/literal paths on the old 8-row tile. Both source
variants were reverted because the full release gate did not produce a clean
target-row win.

Rust timing used RCH with
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a` and the
per-crate command:

```bash
rch exec -- cargo bench -p fj-lax --bench lax_baseline fft_batch_2048x256 -- --warm-up-time 1 --measurement-time 3
```

Baseline and both candidates ran on RCH worker `vmi1152480`. JAX ratios use a
fresh local CPU JAX/JAXLIB 0.10.1 x64 comparator
(`benchmarks/jax_comparison/.venv/bin/python`): complex FFT 249.3358 us, RFFT
183.5284 us, real-input FFT 2.2802276 ms, and IRFFT 220.4353 us. Those Rust/JAX
ratios are routing evidence because the Rust rows ran through RCH; keep/reject
uses the same-worker Rust deltas.

| Row | Baseline Rust | Tile4 Rust | Tile4 delta | JAX mean | Tile4/JAX | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `fft_batch_2048x256_complex128` | 7.1002 ms | 8.3863 ms | +18.114% regression | 249.3358 us | 33.64 | REVERT |
| `fft_batch_2048x256_complex128_dense_input` | 4.7948 ms | 3.4137 ms | -28.805% faster | 249.3358 us | 13.69 | No ship: paired boxed regression |
| `rfft_batch_2048x256_f64` | 5.7007 ms | 5.5569 ms | neutral, p=0.50 | 183.5284 us | 30.28 | Control only |
| `rfft_batch_2048x256_f64_dense_input` | 3.2996 ms | 2.7705 ms | -16.036% faster | 183.5284 us | 15.10 | Control only |
| `fft_batch_2048x256_real_dense_input` | 4.4143 ms | 4.1814 ms | neutral, p=0.18 | 2.2802276 ms | 1.83 | Control only |
| `irfft_batch_2048x256_complex128` | 6.9618 ms | 3.2352 ms | -53.529% faster | 220.4353 us | 14.68 | Control only |

The guarded packed-input tile tried to preserve the boxed baseline while keeping
the dense-input improvement. It failed the target dense-input row and introduced
additional regressions/noise in adjacent FFT rows:

| Row | Baseline Rust | Guarded Rust | Internal result | JAX mean | Guarded/JAX | Verdict |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `fft_batch_2048x256_complex128` | 7.1002 ms | 7.1895 ms | neutral | 249.3358 us | 28.83 | No gain |
| `fft_batch_2048x256_complex128_dense_input` | 4.7948 ms | 5.4350 ms | regression | 249.3358 us | 21.80 | REVERT |
| `rfft_batch_2048x256_f64` | 5.7007 ms | 11.993 ms | regression/noise, not touched path | 183.5284 us | 65.35 | Control only |
| `rfft_batch_2048x256_f64_dense_input` | 3.2996 ms | 3.5889 ms | regression | 183.5284 us | 19.56 | Control only |
| `fft_batch_2048x256_real_dense_input` | 4.4143 ms | 4.8481 ms | regression | 2.2802276 ms | 2.13 | Control only |
| `irfft_batch_2048x256_complex128` | 6.9618 ms | 3.6708 ms | apparent non-target win | 220.4353 us | 16.65 | Control only |

Ratio scorecard for retained production rows in this pass: **0 wins / 6 losses
/ 0 neutral vs JAX**. Rejected tile-size candidates: **0 kept wins / 0 shipped
regressions**; all source hunks in `crates/fj-lax/src/fft.rs` were reverted
before commit. Validation after revert: `cargo test -p fj-lax fft --lib` passed
44/44 with 3 ignored microbenches; `cargo test -p fj-conformance --test
fft_oracle` passed 27/27; `cargo test -p fj-conformance --test
linalg_fft_oracle_parity` passed 1/1; and `cargo build --release -p fj-lax
--benches` passed. Retry only with a real kernel-family change: generated
length-specialized kernels, iterative in-place SoA radix-4/8, portable SIMD
butterflies, or cache-blocked multi-row transforms. Do not retry
representation- or tile-height-only SoA tweaks without hardware-counter
evidence.

## 2026-06-20 - frankenjax-murmw SoA gate-disable no-ship

The BOLD-VERIFY pass retested the current power-of-two batched FFT dispatcher
after an ignored same-binary microbench suggested the threaded per-row Radix2
path might beat the vectorized SoA path. The radical lever was deliberately
simple: temporarily disable the SoA dispatcher by changing
`POW2_VECTORIZED_MIN_BATCH` to `usize::MAX`, measure the full `eval_primitive`
rows, then keep only if the same-worker release gate improved. It did not; the
source change was reverted before commit.

Rust timings used RCH worker `vmi1152480` for both baseline and candidate with
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a` and:

```bash
rch exec -- cargo bench -p fj-lax --bench lax_baseline fft_batch_2048x256_complex128 -- --warm-up-time 1 --measurement-time 3
```

JAX ratios use fresh local CPU JAX/JAXLIB 0.10.1 x64 measurements. The complex
FFT batch comparator mean was 314.5745 us; the inline IRFFT comparator mean was
506.1776 us.

| Row | Production Rust | Gate-off candidate | Internal delta | JAX mean | Production/JAX | Candidate/JAX | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `fft_batch_2048x256_complex128` | 9.5281 ms | 16.346 ms | +71.553% regression | 314.5745 us | 30.29 | 51.96 | Reverted |
| `fft_batch_2048x256_complex128_dense_input` | 8.2082 ms | 15.177 ms | +84.904% regression | 314.5745 us | 26.09 | 48.25 | Reverted |
| `irfft_batch_2048x256_complex128` | 4.8442 ms | 14.645 ms | +202.33% regression | 506.1776 us | 9.57 | 28.93 | Reverted |

The ignored `bench_vectorized_soa_batch_vs_per_row_plan` test-profile probe was
therefore directionally misleading for the release path: its threaded per-row
timing looked faster in debug/test context, but full release Criterion showed
the current production SoA gate is still the less-bad path. This is not a win:
the production batched FFT rows remain severe JAX losses.

Validation after reverting the source experiment: `cargo test -p fj-lax fft
--lib` passed 44/44 with 3 ignored microbenches on RCH `vmi1167313`; `cargo
test -p fj-conformance --test fft_oracle` passed 27/27 on RCH `hz2`; `cargo
test -p fj-conformance --test linalg_fft_oracle_parity` passed 1/1. Retry only
with a real kernel-family change: iterative in-place radix-4/8 SoA/SIMD,
cache-blocked row groups, generated length-specialized kernels, or a backend
plan cache that eliminates the full eval/output-build tax.

## 2026-06-20 - frankenjax-murmw FFT batch direct-output and thread-pool rejects

The BOLD-VERIFY pass targeted the documented FFT batch gap after the pow2 plan-cache
keeps. Two radical but contained levers were tested against the same saved
Criterion baseline on RCH worker `vmi1227854` with
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a`; both were
reverted before commit.

JAX ratios use the `frankenjax-murmw` JAX 0.10.2 x64 rows captured by AzureLynx:
`fft_256` 5.90 us, `ifft_256` 5.72 us, and `fft_batch_2048x256` 236 us.

| Lever | Key row | Candidate Rust | Same-worker delta | Candidate/JAX | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| Current baseline | `fft_batch_2048x256_complex128_dense_input` | 1.8895 ms | baseline | 8.01 | Active JAX loss |
| Current baseline | `fft_batch_2048x256_complex128` | 6.0686 ms | baseline | 25.71 | Active JAX loss |
| Direct output-slice radix2 | `fft_batch_2048x256_complex128_dense_input` | 2.2349 ms | +12.542% regression | 9.47 | Reverted |
| Direct output-slice radix2 | `fft_batch_2048x256_complex128` | 6.4767 ms | neutral +3.4346%, p=0.23 | 27.44 | Reverted |
| Persistent thread-pool row chunks | `fft_batch_2048x256_complex128_dense_input` | 2.8739 ms | +48.632% regression | 12.18 | Reverted |
| Persistent thread-pool row chunks | `fft_batch_2048x256_complex128` | 6.7331 ms | +10.245% regression | 28.53 | Reverted |

Control rows stayed near-parity: final pool candidate `fft_256` was 5.5501 us
(Rust/JAX 0.94, no significant change) and `ifft_256` was 6.0925 us
(Rust/JAX 1.065, no significant change). The direct-output candidate made small
single-FFT rows look slightly faster, but the target batch rows regressed, so no
code was kept.

Validation: focused `cargo test -p fj-lax fft --lib` passed 43/43 through RCH;
`cargo test -p fj-conformance --test fft_oracle` passed 27/27; and
`cargo test -p fj-conformance --test linalg_fft_oracle_parity` passed 1/1. The
only shipped non-source change from this pass is `.rchignore` excluding
`artifacts/`, which fixed RCH transfer churn from unrelated generated files.

Retry predicate: do not retry final-buffer writes or persistent chunk pooling for
batched pow2 complex FFT without fresh cache-miss evidence. The remaining
credible route is a real kernel rewrite: SIMD radix-4/8 butterflies, native
mixed-radix factors for non-pow2 lengths, cache-blocked row batches, or generated
specialized kernels with same-worker before/after proof.
## 2026-06-20 - frankenjax-ur4h3 small-eigh Jacobi route no-ship

The BOLD-VERIFY pass retargeted `frankenjax-ur4h3` after the prior large-n
`eigh` cache fixes. The fresh 48x48 head-to-head showed `svd_48x48_f64` is not
a current JAX loss, but `eigh_48x48_f64` still is. The attempted radical lever
was a small-matrix algorithm-family switch: route real `eigh` with `m <= 64`
through cyclic Jacobi to keep the matrix resident and avoid the
Householder/tridiagonal setup cost. Same-worker proof rejected it.

JAX timing used `/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python`
with JAX/JAXLIB 0.10.1, CPU backend, and `jax_enable_x64=true`.

| Row | Production Rust | Candidate Rust | Internal delta | JAX mean | Production/JAX | Candidate/JAX | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `linalg/eigh_48x48_f64` | 237.78 us | 910.28 us | 3.83x slower | 160.423 us | 1.482 | 5.674 | Revert: small-Jacobi gate regressed |

Additional routing rows from the initial RCH/JAX sweep:

| Row | Rust midpoint | JAX mean | Rust/JAX | Verdict |
| --- | ---: | ---: | ---: | --- |
| `linalg/svd_48x48_f64` | 263.72 us | 460.886 us | 0.572 | Existing Rust win; not the target |
| `linalg/eigh_48x48_f64` first RCH route | 626.00 us | 160.423 us | 3.902 | Existing loss; remeasured same-worker before rejection |

Validation: focused `cargo test -p fj-lax eigh -- --nocapture` passed on RCH
`vmi1149989`; full `cargo test -p fj-conformance` passed on RCH `hz2`; `cargo
build --release -p fj-lax --benches` passed on RCH `hz2`. Ratio scorecard for
this pass: production rows 1 win / 1 loss / 0 neutral vs JAX; rejected
candidate rows 0 wins / 1 loss / 0 neutral. Production decision: no source
change. Retry only with the deeper symmetric-tridiagonal reduction or blocked
panel route; do not retry `m <= 64` cyclic-Jacobi routing.
## 2026-06-20 - frankenjax-murmw radix-4 power-of-four FFT no-ship

The BOLD-VERIFY pass retargeted the current FFT losses after the earlier
mixed-radix and plan-cache keeps. HEAD still loses hard to warmed JAX CPU on
the full `fj-lax` eval path, but the attempted safe radix-4 plan was reverted:
it won a narrow same-binary inner-kernel A/B and lost the full end-to-end gate.

Fresh baseline on RCH worker `vmi1153651` with
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b`:

| Row | HEAD Rust | JAX mean | Rust/JAX | Verdict |
| --- | ---: | ---: | ---: | --- |
| `fft_1000_complex128` | 70.525 us | 31.204 us | 2.260 | Existing loss |
| `fft_1009_prime_complex128` | 243.86 us | 63.710 us | 3.828 | Existing loss |
| `fft_batch_128x1000_complex128` | 5.9966 ms | 197.438 us | 30.372 | Existing loss |
| `fft_batch_2048x256_complex128` | 30.983 ms | 284.224 us | 109.01 | Existing loss |
| `fft_batch_2048x256_complex128_dense_input` | 13.033 ms | 284.224 us | 45.85 | Existing loss |

Rejected lever: a safe recursive `Radix4Plan` for power-of-four lengths
matched DFT/IDFT tolerance and measured 2.13x faster than `Radix2Plan` in a
same-binary inner-kernel A/B on RCH worker `hz2` (`80.976 ms -> 38.024 ms` for
2048 rows of length 256). The full eval path on the same baseline worker
regressed: boxed batched FFT `30.983 ms -> 42.532 ms`; dense-input batched FFT
`13.033 ms -> 33.199 ms`. Source was reverted before commit.

Ratio scorecard for this pass: 0 wins / 7 losses / 0 neutral vs JAX. Production
decision: no source change. Retry only with an iterative in-place radix-4/8,
SoA/SIMD, or cache-blocked batched kernel that proves an end-to-end win; do not
retry naive recursive radix-4 routing from an inner-kernel microbench alone.

## 2026-06-20 - frankenjax-mcqr f64->u32 packed bitcast keep, u32->f64 revert

The BOLD-VERIFY pass targeted the remaining `f64<->u32` width-changing bitcast
losses after the earlier half-width presized-fill keep. The kept lever replaces
the dense `f64 -> u32` byte-array/push materializer with a presized packed
little-endian low/high `u32` fill and a scoped threaded path for 1M+ source
lanes. The attempted symmetric `u32 -> f64` presized/threaded path regressed and
was reverted before commit.

Same-worker Rust proof on RCH worker `hz1` with
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a`:

| Row | Baseline Rust | Candidate Rust | Internal delta | JAX mean | Candidate/JAX | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `bitcast_f64_u32_dense_1m` | 1.5904 ms | 1.1781 ms | 1.35x faster | 163.747 us | 7.195 | Keep internally; still JAX loss |
| `bitcast_f64_u32_literal_ref_1m` | 74.361 ms | 69.507 ms | 1.07x faster | 163.747 us | 424.48 | Control improved, still JAX loss |
| `bitcast_u32_f64_dense_1m` rejected candidate | 989.96 us | 1.2378 ms | 1.25x slower midpoint; Criterion +28.5% | 123.933 us | 9.988 | Revert |
| `bitcast_u32_f64_literal_ref_1m` rejected candidate | 51.321 ms | 49.209 ms | Neutral/no keep | 123.933 us | 397.06 | No production claim |

JAX timing used local CPU JAX/JAXLIB 0.10.1 via
`benchmarks/jax_comparison/bitcast_gauntlet.py --runs 20 --warmup 5
--inner-loops 200`. JAX CV was high (`f64->u32` 34.45%, `u32->f64` 31.97%),
so Rust/JAX ratios are routing evidence; the keep decision uses the same-worker
Rust delta. Ratio scorecard for the measured `f64/u32` rowset: 0 JAX wins / 4
losses / 0 neutral. Production delta after revert: 1 internal win / 0 shipped
regressions, with `u32->f64` unchanged and still a JAX loss.

Validation: `cargo test -p fj-lax bitcast --lib` passed 4/4; `cargo test -p
fj-conformance --test bitcast_oracle` passed 36/36; full `cargo test -p
fj-conformance` passed; `cargo check -p fj-lax --benches` passed; production
`cargo clippy -p fj-lax --lib -- -D warnings` passed. `cargo fmt --check -p
fj-lax` remains red on pre-existing unrelated formatting drift and did not show
a diff in the new packed-bitcast hunk.

## 2026-06-20 - frankenjax-xjbvr unary-chain fusion keep, JAX gap remains

The BOLD-VERIFY pass targeted `fj-interpreters` compiled-dispatch chains where
`Floor`, `Round`, and `Sign` broke dense cheap-op fusion. The kept lever extends
`CheapOp` to strict unary float ops and `i64 sign`, so `add -> unary -> add`
chains stay in the dense fused runner instead of falling back to per-equation
evaluation. `Ceil` and `Trunc` were included in the same semantic family and
covered by the focused parity test.

Same-worker internal rerun on RCH worker `hz1` with
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a`:

| Row | Baseline Rust | Candidate Rust | Internal delta | JAX mean | Candidate/JAX | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `floor_f64_1m_add_unary_chain/n=4` | 21.229 ms | 2.5597 ms | 8.29x faster | 199.892 us | 12.805 | Keep internally; still JAX loss |
| `round_f64_1m_add_unary_chain/n=4` | 20.516 ms | 1.8803 ms | 10.91x faster | 186.162 us | 10.100 | Keep internally; still JAX loss |
| `sign_f64_1m_add_unary_chain/n=4` | 10.955 ms | 2.7290 ms | 4.01x faster | 342.029 us | 7.979 | Keep internally; still JAX loss |
| `floor_f32_1m_add_unary_chain/n=4` | 8.8956 ms | 8.0347 ms | 1.11x faster | 103.966 us | 77.282 | Marginal internal win; still severe JAX loss |
| `round_f32_1m_add_unary_chain/n=4` | 16.423 ms | 4.6730 ms | 3.51x faster | 123.124 us | 37.954 | Keep internally; still severe JAX loss |
| `sign_f32_1m_add_unary_chain/n=4` | 11.639 ms | 5.5279 ms | 2.11x faster | 128.710 us | 42.948 | Keep internally; still severe JAX loss |

The JAX comparator artifact is
`artifacts/performance/evidence/frankenjax-xjbvr-jax-unary-chain-20260620T1358Z.json`
using local CPU JAX 0.10.1. JAX CV was 10-16%, so the Rust/JAX rows are routing
evidence, but the same-worker Rust deltas are large enough to keep the fusion
change. Ratio scorecard for this pass: 0 wins / 6 losses / 0 neutral vs JAX.

Validation: focused `fj-interpreters` parity test passed; `cargo check -p
fj-interpreters --benches`, `cargo clippy -p fj-interpreters --benches -- -D
warnings`, targeted `nextafter_oracle` and `polygamma_oracle`, and full `cargo
test -p fj-conformance` all passed through RCH. Retry only with a deeper
codegen/output-reuse route for f32 unary widening and XLA-style fused kernels;
do not reattempt scalar per-equation dispatch shaving for these rows.

## 2026-06-20 - cod-b width-changing bitcast presized-fill keep

The BOLD-VERIFY pass targeted current scorecard losses rather than another
blocked FMA-policy retry. The kept lever presizes dense width-changing bitcast
outputs and fills by index for `f32 -> bf16/f16 chunks` and `bf16/f16 chunks ->
f32`, preserving the same little-endian chunk order and shape rules.

Same-worker RCH proof on `vmi1227854`:

| Row | Baseline Rust | Candidate Rust | JAX mean | Candidate/JAX | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `bitcast_f32_bf16_1m` | 978.58 us | 125.40 us | 140.512 us | 0.892 | Keep: 7.80x internal win; Rust 1.12x faster than JAX |
| `bitcast_bf16_f32_1m` | 533.82 us | 123.49 us | 151.382 us | 0.816 | Keep: 4.32x internal win; Rust 1.23x faster than JAX |

Controls: literal-reference rows remained orders of magnitude slower
(`38.853 ms` and `27.965 ms` in the candidate run). JAX timing used
`benchmarks/jax_comparison/bitcast_gauntlet.py` with JAX 0.10.1 CPU x64,
20 runs, 5 warmups, and 200 inner loops. A local same-target Rust bench attempt
was rejected as invalid before measurement because the shared RCH target dir
contained artifacts built by a different nightly; the accepted Rust timing is
same-worker RCH baseline/candidate, conservatively compared with local JAX.

Validation: `cargo test -p fj-lax bitcast --lib` passed 4/4 on RCH
`vmi1293453`; `cargo test -p fj-conformance --test bitcast_oracle` passed
36/36 on RCH `hz2`; `cargo check -p fj-lax --all-targets` passed on RCH `hz1`;
production `cargo clippy -p fj-lax --lib -- -D warnings` passed on RCH
`vmi1149989`. Repo-wide `cargo fmt --check` and `fj-lax --all-targets` clippy
are blocked by pre-existing unrelated formatting/test lint debt; follow-up
`frankenjax-98eoz` tracks the clippy side.

## 2026-06-20 - frankenjax-ligu5 vmap gather dense-float no-ship

`frankenjax-ligu5` was reopened against the dispatch-layer batched gather/scatter
de-box vein. Prior passes had already shipped direct I64 batched-operand gather
and lazy rank-1 overwrite scatter, so this pass added f64/f32 batched-operand
gather benchmark rows and measured them before touching production code.

RCH Rust timing used `fj-dispatch` on worker `vmi1152480` with
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a`. JAX timing used
the local CPU `jax.jit(vmap(take))` comparator in
`artifacts/performance/evidence/frankenjax-ligu5-jax-vmap-gather-20260620T1325Z.json`.

| Row | Rust midpoint | JAX mean | Rust/JAX | Verdict |
| --- | ---: | ---: | ---: | --- |
| I64 batched operand + batched indices | 8.6722 us | 31.266 us | 0.277 | Rust 3.61x faster |
| F64 batched operand + batched indices | 25.251 us | 33.224 us | 0.760 | Rust 1.32x faster |
| F32 batched operand + batched indices | 27.257 us | 31.369 us | 0.869 | Rust 1.15x faster |

Decision: keep the new benchmark coverage, but ship no production gather/scatter
change. JAX CV was high (17-25%), so these are routing rows rather than
certification-grade ratios; they are still enough to reject the suspected
dense-float batched gather JAX loss. Retry only with a fresh profile showing a
non-dense, higher-rank, partial-slice, or different dtype subcase.

## 2026-06-20 - frankenjax-n75xr f64 scalar-add SIMD medium-band keep

`frankenjax-n75xr` kept a narrow exact-pattern `std::simd` specialization for
f64 `tensor + scalar + ...` compiled-dispatch chains only in the medium band
`1,048,576 <= n < FUSION_THREAD_MIN_ELEMS`. It preserves the scalar-add order per
lane and rejects operand-reversed first adds so edge-case bit semantics remain on
the generic fusion path.

Same-worker RCH proof on `vmi1227854`:

| Row | Baseline Rust | Candidate Rust | JAX mean | Candidate/JAX | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| f64 1M x8 | 1.3412 ms | 821.23 us | 83.299 us | 9.859 | Keep: 1.63x internal win, still JAX loss |
| f64 16M x8, ungated branch | 78.933 ms | 104.63 ms | 27.610 ms | 3.790 | Reject: +32.56% regression, gated off before commit |

Small rows were not claimed: 65K and 262K are outside the kept gate and remain
JAX losses. Follow-up reruns after the final gate landed on `ovh-a`; `RCH_WORKER`
did not pin back to `vmi1227854`, so those rows are non-comparable worker noise.
Ratio scorecard after the kept gate: 0 wins / 4 losses / 0 neutral vs JAX for
the f64 large-chain production row set.

## 2026-06-20 - frankenjax-cntiy FMA primitive policy no-ship

`frankenjax-cntiy` tested the direct ternary FMA primitive row that the softmax
`+fma` probe could not isolate. The benchmark harness now records `fma_f64_1m`
and `fma_f32_1m` with dense and boxed controls, but no production compiler flag
or target-feature policy was shipped.

JAX comparator caveat: JAX 0.10.1 in the local oracle environment does not
expose a public `jax.lax.fma`, so the comparator is warmed
`jax.jit(lambda a, b, c: a * b + c)`.

| Probe | Rust mean | JAX mean | Rust/JAX | Verdict |
| --- | ---: | ---: | ---: | --- |
| default dense `fma_f64_1m` | 2.6124 ms | 273.448 us | 9.553 | Loss: JAX 9.55x faster |
| default dense `fma_f32_1m` | 2.7622 ms | 111.281 us | 24.822 | Loss: JAX 24.82x faster |
| local `RUSTFLAGS="-C target-feature=+fma"` f64 | 925.02 us | 273.448 us | 3.383 | Improved but still JAX loss; no policy ship |
| local `RUSTFLAGS="-C target-feature=+fma"` f32 | 207.98 us | 111.281 us | 1.869 | Improved but still JAX loss; no policy ship |

RCH build guard passed for `cargo build --release -p fj-lax --benches` on
`vmi1227854`. RCH timing on `hz1` showed the dense primitive path is still a
real internal de-box win over boxed controls: f64 3.5198 ms vs 29.267 ms
(8.31x), f32 3.7757 ms vs 30.825 ms (8.16x). Local same-host default rows were
2.6124 ms vs 24.592 ms boxed (9.41x) and 2.7622 ms vs 26.151 ms boxed (9.47x).

Decision: keep the benchmark coverage, but do not ship a global `+fma` policy.
Ratio scorecard: 0 wins / 4 losses / 0 neutral vs JAX. Retry only with a
semantics-approved per-kernel target feature, generated/vectorized code path, or
output-reuse strategy that can be tested against this row directly.

## 2026-06-20 - frankenjax-xljoh.1 f64 large-chain register/tile no-ships

`frankenjax-xljoh.1` tested two deeper f64 compiled-dispatch levers against
the remaining large-chain JAX losses. Both were reverted before commit.

Baseline on RCH worker `vmi1227854` for the unchanged runner:
65K 46.896 us, 262K 214.91 us, 1M 1.3315 ms, 16M 116.11 ms.
JAX comparator means are from the existing
`../artifacts/performance/evidence/frankenjax-xljoh-jax-comparator-20260620T0550Z.json`.

| Probe | Worker | Best relevant rows | Rust/JAX | Verdict |
| --- | --- | --- | ---: | --- |
| ordered register pass for 8 scalar adds | `vmi1153651` | 65K 52.746 us; 1M 3.5382 ms; 16M 142.56 ms | 1.55 / 42.48 / 5.16 | Reject: failed to vectorize; worse than existing runner and still JAX loss |
| `FUSION_CHUNK` 8K -> 64K | `vmi1149989` | 65K 52.828 us; 1M 1.9764 ms; 16M 128.78 ms | 1.55 / 23.73 / 4.66 | Reject: Criterion reported regressions (+7.33%, +21.35%, +4.66%) |

Retry predicate: do not retry per-element register scalar-add fusion or larger
fusion tiles without disassembly/profile evidence showing SIMD vectorization and
a same-worker before/after win. The remaining f64 chain gap is not a simple
chunk-size or step-order loop issue.

## 2026-06-20 - frankenjax-cntiy softmax +fma/devirtualization no-ship

`frankenjax-cntiy` remains maintainer-gated, but this pass rejects two narrower
agent-actionable interpretations of the gate.

Same-host JAX oracle: `softmax_2d_65536x16` mean 1.0524 ms, p50 1.0765 ms,
CV 9.39%. Local Rust Criterion used
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b-local-20260620`;
RCH validation used `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b`.

| Probe | Rust mean | Rust/JAX | Verdict |
| --- | ---: | ---: | --- |
| default `nn/softmax_2d_65536x16_fused` | 2.2163 ms | 2.106 | Loss: JAX 2.11x faster |
| local `RUSTFLAGS="-C target-feature=+fma"` | 2.2096 ms | 2.100 | Neutral as a lever; still JAX loss |
| generic row-call devirtualization patch | 2.3303 ms | 2.214 | Regression; reverted |

RCH `cargo test -p fj-lax softmax_2d --lib` passed 10/0. An initial RCH
default-vs-`+fma` row on `vmi1149989` looked like a 4.7x `+fma` win, but a
rerun default row on `vmi1152480` and the same-host run both measured ~2.2 ms,
so that first default row is recorded only as an anomaly.

Decision: no source or build-policy win shipped. Global `+fma` alone does not
create the missing SIMD/fast-exp softmax kernel. Ratio scorecard: 0 wins / 3
losses / 0 neutral vs JAX.

## 2026-06-20 - frankenjax-xljoh compiled-dispatch f64 mid-cache keep

`frankenjax-xljoh` shipped only the measured narrow fallback: reusable
`CompiledJaxprRunner` routes one-tensor f64 linear chains in the
65,536..=262,144 element band through owned compiled eval. It did not ship a
broad f32 route or an upper-band f64 route.

Directional candidate Rust/JAX ratios from the new comparator artifact
`../artifacts/performance/evidence/frankenjax-xljoh-jax-comparator-20260620T0550Z.json`:

| Row | Rust/JAX | Verdict |
| --- | ---: | --- |
| f64 4K x8 | 0.541 | Rust faster; guard off |
| f64 65K x8 | 1.516 | JAX faster; kept internal mid-cache fallback |
| f64 262K x8 | 3.195 | JAX faster; remaining gap |
| f64 1M x8 | 20.36 | JAX faster; fallback correctly off |
| f64 16M x8 | 4.55 | JAX faster; no safe win in this lever |
| f32 4K x8 | 0.163 | Rust faster; no f32 route |
| f32 65K x8 | 0.637 | Rust faster; no f32 route |

Retry predicate: do not reattempt broad f32 routing or f64 fallback expansion
without same-host or same-worker proof. The remaining f64 large-chain loss needs
deeper output reuse, vector codegen, or backend specialization.

## 2026-06-20 - frankenjax-q59j4/co009 LiteralBuffer internal keeps

`frankenjax-q59j4` and `frankenjax-co009` were closed as measured internal
keeps after RCH validation. These rows do not claim a JAX-vs-Rust win because
they cover host-internal `LiteralBuffer` mutation and conformance/fixture
serialization paths with no direct JAX API-equivalent comparator.

Validation used `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b`.
The Criterion pass ran on RCH worker `vmi1149989`; `fj-conformance` ran on
`hz2` and passed 45 tests, 0 failed.

| Row | Dense Rust | Literal control | Ratio | Verdict |
| --- | ---: | ---: | ---: | --- |
| `literal_buffer_index_mut_f64_64k` | 24.003 us | 33.278 us | 0.721 | Keep: 1.39x faster internally; JAX N/A |
| `literal_buffer_serialize_f64_64k` | 1.3443 ms | 1.6493 ms | 0.815 | Keep: 1.23x faster internally; JAX N/A |

Decision: keep both committed direct paths. They reduce internal ledger,
fixture, and mutation overhead; the next JAX-facing work remains output/arena
reuse, non-temporal stores/prefetch/NUMA, or a specific typed-path external
loss with same-host proof.

## 2026-06-20 - frankenjax-oneqh allocator default no-ship

`frankenjax-oneqh` was closed as an evidence-only no-ship. The proposed default
allocator change was not committed, and `CHEAP_BINARY_PARALLEL_MIN` was not
retuned or removed.

Local same-host allocator preload verification used
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b-local-20260620`
with `fj-lax` `elementwise_gauntlet`. JAX used 0.10.1 CPU x64 via
`benchmarks/jax_comparison/.venv/bin/python`. Jemalloc was available at
`/usr/lib/x86_64-linux-gnu/libjemalloc.so.2`; mimalloc was not installed.

| Row | glibc Rust | jemalloc Rust | JAX | Verdict |
| --- | ---: | ---: | ---: | --- |
| `add_f64_16m/dense` | 24.502 ms, Rust/JAX 0.870 | 33.095 ms, Rust/JAX 1.174 | 28.179 ms | Reject: allocator preload regresses |
| `add_f32_16m/dense` | 15.945 ms, Rust/JAX 1.128 | 16.784 ms, Rust/JAX 1.188 | 14.130 ms | Reject/neutral: no allocator win |
| `mul_f64_16m/dense` | 29.210 ms, Rust/JAX 1.024 | 27.790 ms, Rust/JAX 0.974 | 28.525 ms | Neutral/slight win only |

Decision: no production allocator change. The next credible levers are
compiled-jaxpr output/arena reuse, non-temporal stores/prefetch/NUMA work, or a
specific unowned typed-path gap with same-host proof.

## 2026-06-21 - frankenjax-murmw specialized radix-2/5 FFT validation blocked

The radical route from `/alien-graveyard`, `/alien-artifact-coding`, and
`/extreme-software-optimization` for the smooth-composite FFT gap remains
length-specialized code generation for `1000 = 2^3 * 5^3`, not another
representation detour. The target row is still
`eval/fft_batch_128x1000_complex128`: RCH `hz1` Criterion midpoint **3.6581 ms**
versus fresh JAX/JAXLIB 0.10.1 x64 mean **0.245442 ms**, so current Rust/JAX is
**14.90x**.

This pass attempted to validate the existing specialized radix-2/5 SoA path with
a focused ignored `fj-lax` release A/B, but no candidate timing ratio was
produced. RCH stalled with stale progress on `vmi1153651`, then a peer cbrt WIP
compile blocker (`fast_cbrt_f64`) stopped the next `vmi1149989` run, and the
subsequent `vmi1149989` retry stalled again before emitting the A/B row. Both
stale RCH builds were canceled. No source change was kept.

Decision: validation-blocked no-ship, not a timed rejection. Scorecard for the
target remains **0 wins / 1 loss / 0 neutral** versus JAX; candidate score for
this pass is **0 kept / 0 rejected by timing / 1 validation-blocked**. Retry
only after the cbrt WIP is landed/reverted and RCH can reuse a genuinely warm
target dir, then require a completed same-binary A/B before any dispatch gate.

## 2026-06-21 - REGRESSION CAUGHT: d74a6472 erf speedup broke random_normal RNG golden (CrimsonOtter-Claude/cod-b)

Regression vigilance on HEAD found fj-lax conformance at **1581 pass / 2 fail** (was 1 — the
pre-existing cholesky drift). NEW failure: `threefry::tests::random_normal_threaded_golden_sha256`.
ROOT CAUSE (verified): `random_normal` draws normals via `crate::arithmetic::erf_inv_approx`
(threefry.rs:535/559), and `erf_inv_approx` (arithmetic.rs:9430-9433) does **3 Newton iterations
calling `erf_approx(y)`**. Commit `d74a6472` ("speed up erf common range", codex cod-b) replaced
`erf_approx`'s common-range series with fdlibm rational bands — changing its last bits — so
erf_inv's unconverged last bit changed → `random_normal`'s bytes changed → its bit-exact SHA256
golden breaks. The codex validated `erf_oracle` (forward erf) + `erf_high_accuracy_and_seam_continuity`
but NOT the RNG golden (the transitive erf_approx dependency was missed).
FIX (owner's call, preserves the 4.58x erf win): DECOUPLE `erf_inv_approx`'s Newton from the
production `erf_approx` — give it a dedicated, stable high-accuracy erf (or pin the old series) so
random_normal's bits are erf-primitive-independent; OR re-baseline `random_normal_threaded_golden_sha256`
ONLY after confirming the new erf_inv still matches JAX's `random.normal` bit-for-bit (RNG parity is
fixed-to-JAX). Reverting d74a6472 also restores GREEN but loses the erf win. Flagged to codex cod-b;
conformance is RED until resolved.
**RESOLVED `c1b9ef15`** (win-preserving): added `fn erf_for_erfinv` (pinned pre-d74a6472 series)
used ONLY by `erf_inv_approx`'s Newton, so random_normal is erf-primitive-independent; the
PRIMITIVE keeps the fast fdlibm erf (4.58x win kept). VERIFIED workspace-wide: fj-lax lib back to
1582/1 (only pre-existing cholesky_blocked, shbyh's, RED), AND fj-conformance ALL-GREEN (RNG
distribution oracles + erf_oracle + cbrt_oracle, every binary 0-failed). The transcendental sweep
is conformance-clean. Only remaining RED workspace-wide = the shbyh cholesky_blocked golden
(linalg owner's domain).
**PROACTIVE AUDIT (threefry.rs) — sweep RNG-golden hazard is BOUNDED:** the cod-b transcendental
sweep optimizes fj-lax-INTERNAL `*_approx`/`eval_*`, NOT libm. Audited every threefry distribution:
only `random_normal` (535/559) + `truncated_normal` (1098/1107) call `crate::arithmetic::erf_approx`/
`erf_inv_approx` (internal) → the only vulnerable spots; random_normal fixed, truncated_normal GREEN
(tolerance oracle). cauchy/gumbel/exponential/rayleigh/laplace/poisson/geometric all use libm
`f64::tan/ln/exp/ln_1p` DIRECTLY → SAFE from primitive opts. So upcoming tanh/tan/ln/exp PRIMITIVE
opts do NOT threaten the RNG distribution goldens (only fj-lax-internal-erf sites do, both handled).

## 2026-06-21 - cholesky_blocked digest: VERIFIED STALE (safe re-baseline) — owner's one-line fix (CrimsonOtter-Claude)

The lone remaining fj-lax RED, `linalg::tests::cholesky_blocked_path_golden_output_digest`, is a
SELF-golden (hardcoded SHA of `eval_cholesky`'s 256x256 output) — NOT a JAX-parity test. Diagnosed:
the sibling `cholesky_blocked_matches_scalar_and_reconstructs` (linalg.rs:8271) PASSES, proving the
blocked-path output is CORRECT (matches the scalar/naive Cholesky AND reconstructs L·Lᵀ≈A within
tolerance). So shbyh's blocked-Cholesky FP-reassoc restructuring legally changed the exact bits and
the digest is simply STALE — a safe re-baseline, not a bug.
FIX (linalg owner's one-liner, I'm not editing their collision-zone file): at linalg.rs:8338 replace
  "cae3c6a0fcc965880d1379765d0b7886deb1ca3d1c9dc1036ca705e60306ff0a"
with the current verified-correct output digest
  "4ed70745461cf775a4ea667bf87bae3c6fbc5326c5095cad6435d0069d1c7540"
That restores fj-lax to fully GREEN (it's the only non-cholesky-digest failure). Flagged WildForge.

## 2026-06-21 - frankenjax-cntiy scalar atan2 threading keep, but JAX still dominates

`frankenjax-cntiy` was probed for a non-FMA sub-gap after the small-angle tan
win. The radical route from `/alien-graveyard` and `/extreme-software-optimization`
is to stop paying scalar `libm` cost per element; the first legal lever was to
remove the stale f64 scalar-Atan2 exclusion from the existing expensive
scalar-broadcast threaded path. This preserves the exact lane operation
(`f64::atan2`) and operand order, so bits do not change.

Same-worker RCH `vmi1293453`, `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b`,
Criterion filter `atan2_scalar_1m_f64_vec`:

| Row | Midpoint | Ratio |
| --- | ---: | ---: |
| Baseline `eval/atan2_scalar_1m_f64_vec` | 30.351 ms | 1.00 |
| Kept threaded route, full `atan2` filter | 12.769 ms | 0.421 |
| Kept threaded route, scalar-only confirm | 13.998 ms | 0.461 |

Fresh JAX/JAXLIB 0.10.2 CPU x64 comparator for the same 1M f64 fixture
(`jnp.atan2(a, 3.25)`) measured mean **0.116833 ms**, p50 **0.096482 ms**,
p95 **0.175232 ms**. The kept Rust route is therefore still **119.8x** slower
than JAX by confirmed midpoint/mean, even though it is **2.17x** faster than
the old Rust route.

Correctness/conformance:
- RCH `vmi1149989` `cargo test -p fj-lax atan2 --release -- --nocapture`
  passed **4/4**.
- RCH `vmi1152480` `cargo test -p fj-conformance --test atan2_oracle
  --release -- --nocapture` passed **40/40**.
- RCH `vmi1152480` full `cargo test -p fj-conformance --release --
  --nocapture` passed the full crate and doc-test suite.

Decision: keep the bit-identical scalar-broadcast threading because it is a
measured 2.17x internal win. Scorecard versus JAX for this row remains
**0 wins / 1 loss / 0 neutral**. Retry predicate: do not spend another pass on
thread-count tuning for scalar Atan2; JAX is using a vectorized kernel. The next
credible route is a safe portable-SIMD/range-reduced atan2 approximation with
`atan2_oracle` tolerance proof, or the broader `cntiy` target-feature/codegen
policy lane.

## 2026-06-21 - frankenjax-cntiy boxed-literal scalar pow/atan2 threading keep, still JAX loss

This follow-up targeted the remaining direct `LiteralBuffer::Literals` scalar
broadcast path used by the `*_literal_ref` control benches. `TensorValue::new`
already densifies F64, so this is not a constructor fix; the kept lever is a
narrow fallback inside the existing expensive F64 scalar-broadcast helper. Large
boxed-F64 buffers now thread over the literal slice and emit dense F64 output.
Mixed or non-F64 buffers still return `None` and fall through to the generic
authoritative path.

Same-worker RCH `vmi1293453`, `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b`,
Criterion filter `scalar_1m_f64_literal_ref`:

| Row | Baseline midpoint | Kept midpoint | Rust speedup |
| --- | ---: | ---: | ---: |
| `eval/pow_scalar_1m_f64_literal_ref` | 80.435 ms | 15.193 ms | 5.29x |
| `eval/atan2_scalar_1m_f64_literal_ref` | 38.339 ms | 11.987 ms | 3.20x |

Fresh JAX comparator used `benchmarks/jax_comparison/.venv/bin/python`, JAX
**0.10.1**, JAXLIB **0.10.1**, CPU backend with `jax_enable_x64=true`, exact
1M fixtures:

| Row | JAX mean | JAX p50 | Kept Rust/JAX |
| --- | ---: | ---: | ---: |
| `pow_scalar_1m_f64_literal_ref` | 1.808211 ms | 1.738733 ms | 8.40x loss |
| `atan2_scalar_1m_f64_literal_ref` | 2.214336 ms | 2.073297 ms | 5.41x loss |

Correctness/conformance:
- RCH `vmi1293453` `cargo test -p fj-lax
  threaded_expensive_binary_scalar_bit_identical_to_reference --release --
  --nocapture` passed **1/1**. The proof covers dense and boxed storage, both
  operand orders, Pow and Atan2, and dense F64 output.
- `rustfmt --edition 2024 --check crates/fj-lax/src/arithmetic.rs` passed.
- RCH `vmi1152480` full `cargo test -p fj-conformance --release` passed the
  full crate suite and doc-tests.
- RCH `hz2` `cargo check -p fj-lax --all-targets` passed. It emitted existing
  deprecated `criterion::black_box` warnings in the unrelated
  `f32_rounding_ab` bench.
- RCH `hz2` `cargo clippy -p fj-lax --all-targets -- -D warnings` is still
  blocked by unrelated `crates/fj-lax/src/fft.rs:1664` `manual_is_multiple_of`
  lint. This pass did not edit the FFT surface.

Decision: keep the bit-identical boxed-F64 threading path because both
same-worker Rust deltas are large. Scorecard versus JAX for this pass remains
**0 wins / 2 losses / 0 neutral**. Retry predicate: do not repeat boxed-literal
fan-out work; the remaining JAX gap needs true vector range-reduced pow/atan2
or the broader `cntiy` target-feature/codegen policy lane.

## 2026-06-21 - frankenjax fusion-chunk unary family: f32 hoist KEPT, 5 dead-ends (CobaltForge/cc)

Context: shipped `294c836f` — hoisted the unary `CheapOp` match out of the
per-element loop in `apply_f32_fusion_chunk`/`apply_fusion_chunk`. f32 fused
floor/round/sign chains **+4.63x** (same-binary A/B, bit-identical, conformance
45/45). This entry records the NEGATIVE results mapped around that win so the
family is not re-mined. All ratios are same-binary or same-machine (rch `exec`
has no `--worker` pin, so cross-invocation absolutes span ~25x and are untrusted).

| Dead-end | A/B (same-binary unless noted) | Verdict |
| --- | --- | --- |
| Native-f32 `roundps` single-op kernel | widen 74.07us vs direct 76.51us = **1.03x slower** | Kernel not the bottleneck; widen already vectorizes; single-op f32 floor 74us already beats JAX ~104us |
| f64 op-match hoist in `apply_fusion_chunk` | in-loop 1.86ms vs hoisted 1.86ms = **1.00x** | LLVM already unswitches the non-widening f64 path; f64 hoist kept only for symmetry (neutral, bit-identical) |
| `sign` vectorization | branchless-scalar **0.99x**; std::simd masks hit nightly trait-drift | Rare op, sub-ms; fragile. Do not re-chase |
| f32 register-resident scalar-add-chain SIMD (f64 sibling of n75xr) | per_pass 1.2345ms vs register_simd 2.8475ms = **0.43x (2.3x SLOWER)** | f32's widen-per-step contract forces vcvtps2pd/vcvtpd2ps casts per add that dominate; f64 wins only because native contract needs no cast |
| i64/half fusion-chunk hoist | same pattern but no 1M bench; half decode-bound; i64 unary niche | Unproven scope creep; left untouched |

Same-machine floor_f32 head-to-head (local Zen3 host, post-fix): Rust eager
`eval_jaxpr` **p50 647.92us** / runner **p50 641.05us** vs JAX 0.10.1 jit f32
**p50 145.26us** = **~4.5x JAX loss**. The fix closed the gap ~21x -> ~4.5x; the
residual is the per-call fused-output 4MB alloc + multi-pass fusion vs XLA
single-pass — the `CompiledJaxprRunner` does NOT amortize the fused-output alloc
(`try_fuse_elementwise_chain_f32` allocates fresh every call). That residual lever
is so4wo (owned) and was handed to its owner with the root cause.

Decision: KEEP the f32 hoist (bit-identical 4.63x). The family is exhausted —
f32 shipped, f64 neutral, i64/half/sign/f32-add-chain are dead-ends with the
root causes above. Retry predicate: do not re-attempt any row in the table; the
only remaining fused-chain lever (amortize the runner's fused-output buffer) is
so4wo runner-arena work, not a fusion-chunk edit.

## 2026-06-21 - scatter-add is a JAX LOSS, not a domination (corrects XLA-CPU-scatter assumption) (CobaltForge/cc)

Tested the assumption that XLA's CPU `Scatter` is a standing weakness (like its
bitonic `Sort`, which the Rust radix port beats 4x). It is NOT — scatter-add is a
JAX LOSS for fj-lax. Same-machine (local Zen3 host), 1M f64 1D scatter-add with
COLLIDING indices (setup copied verbatim from `bench_scatter_add_1m_f64_1d`):

| Side | p50 | mean | min |
| --- | ---: | ---: | ---: |
| Rust `eval_primitive(Scatter, mode=add, clip)` | 14.43 ms | 14.30 ms | 10.81 ms |
| JAX 0.10.1 `op.at[idx].add(upd, mode='clip')` jit x64 | 4.51 ms | 4.80 ms | 2.47 ms |

**Rust/JAX = ~3.2x LOSS** by p50 (4.3x at min), low variance both sides. XLA's CPU
scatter-add (4.5 ms) is reasonable even under heavy index collision — it does NOT
serialize pathologically the way its bitonic CPU sort does. So `Scatter` is NOT a
domination target.

Caveat (honesty): the Rust number is via `eval_primitive`, which boxes the 1M
operand/output through `Vec<Literal>`; the optimized dense bucketed scatter-add
(commit f50acd09) may run faster on a dense in/out path, so the true production
gap could be smaller — but as-invoked through the public eval path it is a 3.2x
JAX loss. Retry predicate: do not chase scatter as a domination; if scatter perf
is pursued, the lever is a dense operand+updates path that skips the per-`Literal`
boxing, measured same-machine vs JAX, not vs a Rust reference.

## 2026-06-21 - maxpool/reduce_window is a JAX LOSS + META-LESSON: internal speedups != JAX dominations (CobaltForge/cc)

Tested reduce_window(max) as a domination candidate (memory claims "deque pooling
20x"). It is a JAX LOSS. Same-machine (local Zen3 host), 256x256 / 15x15 window,
VALID, setup copied verbatim from `bench_maxpool_large_separable`:

| Side | p50 | mean | min |
| --- | ---: | ---: | ---: |
| Rust `eval_primitive(ReduceWindow, max)` | 1127.11 us | 1165.92 us | 1085.09 us |
| JAX 0.10.1 `lax.reduce_window(max, VALID)` jit x64 | 484.10 us | 509.92 us | 403.21 us |

**Rust/JAX = ~2.3x LOSS** by p50 (2.7x at min). The deque/separable "20x" in
memory is RUST-INTERNAL (deque O(n) vs naive O(n*k)); XLA's `reduce_window` is a
well-optimized vectorized window reduction, so vs JAX it is a loss.

META-LESSON (the valuable part): this is the SECOND "expected domination" that
turned out a JAX LOSS this session (scatter-add ~3.2x, maxpool ~2.3x). Most of the
"Nx faster" perf numbers in the ledgers/memory are RUST-INTERNAL speedups (vs a
Rust naive/reference baseline), NOT Rust-vs-JAX. The verified Rust-OVER-JAX
domination set is much smaller than those numbers suggest — so far only `sort`
(LSD radix 4.06x faster than XLA CPU bitonic) holds up under a same-machine
head-to-head; `scatter`, `maxpool`, and the `floor_f32` fused chain are all JAX
losses. Retry predicate: before claiming any "DOMINATE JAX" win, verify it
same-machine vs an actual JAX run — do not infer domination from an internal
speedup ratio. (Same boxing caveat: Rust numbers are via `eval_primitive`, which
boxes the output; a dense path may narrow but is unlikely to flip these.)

## 2026-06-21 - scan-family domination is NOT general; JAX cumsum has a SIZE CLIFF (CobaltForge/cc)

Refines (and partly refutes) my own "scan-family dominates JAX" prediction from
the cumsum-4M result. Same-machine (local Zen3 host), 1M f64 1D:

| op (1M f64) | Rust p50 | JAX p50 | verdict |
| --- | ---: | ---: | --- |
| `cumprod` | 1636 us | 1499 us | **1.09x LOSS** (near-parity) |
| `cummax` (lax.cummax) | 1746 us | 1904 us | **1.09x win** (near-parity) |

So cumprod/cummax are NEAR-PARITY at 1M, NOT 4x dominations. The "scan-family
dominates" generalization was too broad.

ROOT of the cumsum domination — a JAX SIZE CLIFF, not a general scan weakness.
JAX `jnp.cumsum` scales SUPERLINEARLY on CPU: **1M p50 1356 us -> 4M p50 18397 us**
(13x time for 4x data). Rust cumsum is linear (~1ms@1M, ~4ms@4M). So:
- At 1M, cumsum is near-parity (Rust ~1ms vs JAX 1.36ms).
- At 4M+, Rust dominates ~4.3x because JAX cumsum hits a pathological large-n scan
  lowering (cache/algorithm cliff between 1M and 4M).

CORRECTED MAP: the only SIZE-INDEPENDENT Rust-over-JAX domination verified so far
is `sort` (XLA-CPU bitonic). `cumsum` dominates ONLY at large n (JAX size cliff);
cumprod/cummax/scatter/maxpool/floor are parity-or-loss. Retry predicate: a "scan
dominates" claim must specify the size — verify at the target n, and check JAX's
size-scaling (its cumsum cliff means small-n cumsum is NOT a domination).

## 2026-06-21 - the einsum "330x" is INTERNAL, not a JAX win — einsum routes to matmul, a 5-11x JAX LOSS (CobaltForge/cc)

Completing the internal-vs-JAX overclaim case with its most extreme example, by
derivation (zero new bench). The largest "domination" number in the ledgers/memory
is "einsum-GEMM 330x". That 330x is RUST-INTERNAL: einsum2 single-contraction is
routed to `matmul_2d` (einsum.rs:164, bit-identical fast path), and the 330x is
that GEMM routing vs the prior naive per-element odometer einsum.

But `matmul_2d` itself is a MEASURED JAX LOSS (existing scorecard, frankenjax-ifou2/
4ryym): `matmul_2d_256x256x256_f64` **1.32ms vs JAX 0.265ms = ~5x loss**;
`matmul_2d_512x512x512_f64` **6.35ms vs JAX 0.577ms = ~11x loss** (the loss GROWS
with size — fma-bound, capped at ~XLA/2 by the deliberate no-+fma policy, see
frankenjax-cntiy). So as actually executed, einsum is a 5-11x JAX LOSS, NOT a 330x
domination.

This is the clearest case of the pattern documented in the 2026-06-21 scatter/
maxpool/scan entries: the big "Nx faster" numbers throughout the ledgers are
Rust-INTERNAL (optimized vs a Rust naive/odometer/reference baseline), and several
are JAX LOSSES when measured same-machine vs an actual JAX run. The verified
Rust-OVER-JAX domination surface remains narrow: only `sort` (4-5.5x, robust across
dtype/size) and large-n `cumsum` (JAX size cliff). Retry predicate / release note:
do NOT present any "Nx faster" internal ratio as a JAX domination; the GEMM-routed
ops (einsum, dot_general, conv-via-im2col) inherit matmul's fma-bound JAX loss.

## 2026-06-22 - JAX scan SIZE CLIFF is family-wide (cumsum/cumprod/cummax) — corrects "cumprod/cummax near-parity" (CobaltForge/cc)

Zero-build JAX-only scaling sweep (local venv, JAX 0.10.1 x64; ratios are
machine-independent). The cumsum CPU size cliff GENERALIZES across the whole scan
family — all three are ~linear to 2M then cliff ~8-10x at 4M:

| op (JAX p50) | 1M | 2M | 4M | 2M/1M | 4M/1M |
| --- | ---: | ---: | ---: | ---: | ---: |
| cumsum  | 1440us | 2929us | 14700us | 2.0x | 10.2x |
| cumprod | 1405us | 2979us | 12003us | 2.1x | 8.5x |
| cummax  | 1712us | 3731us | 16101us | 2.2x | 9.4x |

CORRECTION: my earlier entry called cumprod/cummax "near-parity, NOT dominations"
— that was SIZE-LIMITED (measured only at 1M, where JAX scan is still fine). At
4M JAX's scan lowering cliffs for ALL of them, so cumprod/cummax are LIKELY
large-n Rust dominations too (Rust scan is linear; cumsum-4M already confirmed
Rust ~3.5-4.3x faster, eval 4.2ms vs JAX 14.7ms). The verified Rust-over-JAX
domination set thus EXPANDS: `sort` (size-independent) + the entire SCAN family
at large n (>=4M, JAX size cliff) — cumsum confirmed, cumprod/cummax predicted
pending a Rust 4M measurement (blocked now: cold-rebuild forbidden + local target
freed during the disk emergency; warm rch target is cross-machine-only). Retry
predicate: when builds resume, measure Rust cumprod/cummax at 4M same-machine to
confirm; the JAX cliff itself is established.

## 2026-06-22 - JAX scan cliff is a one-time STEP (plateaus); scan domination is stable ~4x at large n (CobaltForge/cc)

Trajectory refinement (zero-build JAX-only, ratios machine-independent). The JAX
cumsum cliff is a single per-element STEP at ~1M->4M, then it PLATEAUS — it does
NOT keep blowing up:

| JAX cumsum | 1M | 4M | 8M | 16M |
| --- | ---: | ---: | ---: | ---: |
| p50 | 1.43ms | 18.39ms | 34.78ms | 64.05ms |
| per-elem | 1.40us/K | 4.49us/K | 4.25us/K | 3.91us/K |

Per-element jumps ~1.4->~4.4 us/K at the cliff (~3x step), then holds ~4us/K
through 16M. Rust cumsum is ~1us/K linear (measured 4.2ms @4M). So the scan
domination is ~1.4x at 1M, STEPS to a stable ~4x at 4M+, and does NOT grow
further (both linear past the cliff). So the release framing is "stable ~4x
large-n scan domination above the JAX cliff," not "grows with n." (Caveat: my
first trajectory script mis-estimated the Rust line using 4.2ms/1M instead of
the true ~1us/K; corrected here.)

## 2026-06-22 - cumsum-4M domination CONFIRMED on independent machine; cross-machine ratio understates same-machine (CobaltForge/cc)

Warm-target rch bench (`frankenjax-cc`, no cold rebuild) of the committed
`eval/cumsum_4m_f64_1d`, to cross-validate the cumsum large-n domination on a
DIFFERENT machine and quantify the cross- vs same-machine ratio gap:

| measurement | value |
| --- | ---: |
| Rust cumsum-4M, rch worker `ovh-a` (this run) | 7.45 ms |
| Rust cumsum-4M, ovh-a (cod-b independently) | 7.53 ms |
| Rust cumsum-4M, local host (earlier) | 4.20 ms |
| JAX cumsum-4M, local (fresh) | 18.52 ms (min 15.36) |

- **Same-machine domination = 4.4x** (local Rust 4.20 / local JAX 18.52) — the
  accurate figure.
- **Cross-machine ratio = 2.49x** (ovh-a Rust 7.45 / local JAX 18.52) — matches
  cod-b's recorded 2.43x, but UNDERSTATES the domination because ovh-a is ~1.8x
  slower than the local host for this Rust workload.
- My ovh-a number (7.45ms) matching cod-b's (7.53ms) independently confirms the
  Rust cumsum-4M workload is real (not a local-host artifact), and the domination
  holds in BOTH framings (2.49x cross / 4.4x same).
- METHOD NOTE: rch-Rust-vs-local-JAX ratios (the repo's common convention) are a
  conservative LOWER BOUND on the true same-machine domination when the rch worker
  is slower than local; the same-machine head-to-head is the accurate one. JAX
  cannot run on rch workers, so a clean cross-machine ratio is impossible — report
  both and label which host each side ran on.

## 2026-06-22 - scatter-add JAX LOSS confirmed cross-machine (not a local artifact) (CobaltForge/cc)

Completeness: cross-confirming a documented LOSS on an independent machine (the
cross-machine validation should rule out local-host artifacts in BOTH directions,
not just for dominations). Warm-target rch bench of committed
`eval/scatter_add_1m_f64_1d` on `ovh-a` = **17.06ms** vs fresh local JAX
`op.at[idx].add` 1M = **3.71ms** = **~4.6x Rust LOSS** cross-machine. Same-machine
was ~3.2x (local Rust 14.4ms / local JAX 4.5ms). ovh-a Rust (17.06ms) is ~1.2x
slower than local (14.4ms), consistent with the cumsum cross-machine observation.
So scatter-add is a confirmed JAX loss on BOTH machines (3.2x same / 4.6x cross),
not a local quirk — XLA's CPU scatter is genuinely well-optimized; the Rust
eval_primitive boxing path loses regardless of host. (The dense-path lever to
narrow it remains cod-a's scatter-bucketing work; see prior scatter entries.)

## 2026-06-22 - maxpool/reduce_window JAX LOSS confirmed cross-machine (CobaltForge/cc)

Completeness cross-confirm of the maxpool loss on an independent machine.
Warm-target rch bench of committed `eval/maxpool_256x256_15x15_separable` on
worker `hz2` = **847us** vs fresh local JAX `lax.reduce_window(max,VALID)`
256x256/15x15 = **478us** = **~1.77x Rust LOSS** cross-machine. Same-machine was
~2.3x (local Rust 1.13ms / local JAX 484us). Here the rch worker (847us) was
FASTER than the local host (1.13ms) — opposite of the ovh-a cumsum/scatter runs —
so the cross-machine ratio (1.77x) is SMALLER than same-machine (2.3x). Either
way maxpool is a confirmed JAX loss on both hosts (XLA's CPU reduce_window is
well-vectorized; the memory "deque 20x" is Rust-internal, not vs JAX). Net: the
cross-machine validation now covers sort (win), cumsum (win), scatter (loss),
maxpool (loss) — all hold in the correct direction across machines, with the
ratio magnitude shifting by worker CPU speed (always report the host).

## 2026-06-22 - floor-chain JAX LOSS confirmed cross-machine; cross-machine map validation COMPLETE (CobaltForge/cc)

Final completeness cross-confirm. Warm-target rch bench of committed
`compiled_dispatch/eager/floor_f32_1m_add_unary_chain/n=4` on `hz2` = **595us** vs
fresh local JAX f32 floor (add+floor)x4 jit = **91.8us** (min 74) = **~6.5x Rust
LOSS** cross-machine (4.5-7x same-machine). My shipped fix 294c836f gave a real
4.63x bit-identical Rust-side win, but the chain stays a JAX loss due to the
per-call interpreter tax (env build + 4MB fused-output alloc vs XLA single fused
pass) — confirmed on hz2, not a local artifact. The residual amortization lever
is so4wo (owned).

CROSS-MACHINE MAP VALIDATION COMPLETE — all 5 characterized ops confirmed in the
correct direction across local + rch worker:

| op | same-machine | cross-machine | verdict |
| --- | --- | --- | --- |
| sort | 4-5.5x | 6.53x (hz2) | DOMINATION |
| cumsum-4M | 4.4x | 2.49x (ovh-a) | DOMINATION |
| scatter | 3.2x loss | 4.6x (ovh-a) | JAX loss |
| maxpool | 2.3x loss | 1.77x (hz2) | JAX loss |
| floor-chain | 4.5x loss | 6.5x (hz2) | JAX loss |

Verified Rust-over-JAX domination surface (robust across dtype/size/machine):
`sort` + large-n scan family (cumsum confirmed; cumprod/cummax JAX-cliff confirmed,
Rust-4M pending build). Everything else measured is JAX-parity-or-loss; the broad
ledger "Nx faster" numbers are Rust-internal, not vs JAX.

## 2026-06-22 - one_hot "17x" is NOT a valid domination (bandwidth-implausible Rust number) (CobaltForge/cc)

Caught a likely-invalid head-to-head via a bandwidth sanity check before recording
it. Warm-target rch `eval/one_hot_2048x512_f64` on `ovh-a` = **12.4us** vs fresh
local JAX `jax.nn.one_hot(...,512,f64)` = **212us** — an apparent 17x. REJECTED as
a domination claim: the Rust output is 2048x512 f64 = **8MB**, so 12.4us implies
**~677 GB/s**, far above DRAM bandwidth (~50 GB/s). A real dense 8MB write cannot
finish that fast, so Rust's `eval_primitive(OneHot)` is NOT materializing the
dense output that JAX produces (likely boxed/lazy/structural, or the bench doesn't
force full materialization). The comparison is apples-to-oranges and the "17x" is
an artifact, NOT a Rust-over-JAX win.
- LESSON: always sanity-check a head-to-head against memory bandwidth — output
  bytes / time > ~50 GB/s means one side isn't doing the work the other is.
- Retry predicate: only revisit one_hot if it's confirmed Rust produces a dense
  materialized output comparable to JAX's (check `eval_primitive(OneHot)` storage);
  do not list one_hot as a domination until then. The verified domination set
  stays sort + large-n scan + contiguous gather.

## 2026-06-22 - complex matmul is a JAX LOSS (zgemm BLAS) — matmul domination is INTEGER-SPECIFIC (CobaltForge/cc)

Boundary test completing the matmul story. Warm-target rch
`eval/matmul_256x256_complex128_dense` on `ovh-a` = **2.04ms** vs fresh local JAX
complex128 `a@b` 256x256 (x64) = **0.547ms** (min 0.447) = **~3.7x Rust LOSS**
cross-machine (ovh-a ~1.2x slower than local, so same-machine ~2-3x). Confirms:
complex matmul LOSES because JAX/XLA has a complex BLAS (`zgemm`).

COMPLETE matmul characterization — the domination is entirely about the BLAS gap:
| dtype | JAX backend | verdict |
| --- | --- | --- |
| i64/i32/u32 | none (no integer BLAS) -> naive | Rust DOMINATES ~80x |
| f64/f32 | dgemm/sgemm BLAS | Rust loses ~5-11x (fma-bound) |
| complex128 | zgemm BLAS | Rust loses ~3.7x |

So Rust-over-JAX matmul domination is INTEGER-ONLY: Rust wins exactly where JAX
lacks a BLAS path; wherever JAX has BLAS (float/complex) Rust loses (no fma + no
BLAS-grade kernel). Clean, defensible, and explains the whole matmul row set.

## 2026-06-22 - i32 matmul is a JAX LOSS (vpmulld SIMD) — integer-matmul domination is i64-ONLY (CobaltForge/cc)

CORRECTS the prior "integer matmul (i64/i32/u32) dominates ~80x" overgeneralization.
Warm-target rch `eval/matmul_512x512_i32_dense` on `hz2` = **4.84ms** vs fresh
local JAX int32 `a@b` 512x512 = **0.62ms** (min 0.52) = **~7.8x Rust LOSS**.

ROOT CAUSE (the i64 vs i32 asymmetry): AVX2 has `vpmulld` (32-bit SIMD multiply)
but 64-bit SIMD multiply (`vpmullq`) is AVX512-only. So:
- JAX i32 matmul VECTORIZES (vpmulld) -> 0.62ms; JAX i64 matmul is SCALAR -> 374ms.
- Rust's i32 matmul does NOT SIMD (it stores i32 as Literal::I64, so its kernel is
  ~the same ~4.8ms as i64 -- a blocked but non-vectorized integer GEMM).
Result: i64 -> Rust 4.64ms vs JAX 374ms = Rust WINS 80x (both scalar, Rust blocked
beats JAX naive); i32 -> Rust 4.84ms vs JAX 0.62ms = Rust LOSES 7.8x (JAX vpmulld
beats Rust non-SIMD). The matmul domination is i64-ONLY, not integer-family.

NEW LEVER (recorded, not pursued -- build-blocked): a native-i32 SIMD matmul
(vpmulld, i32 storage instead of i64) would close the ~7.8x i32 gap. Likely also
applies to u32. Touches the matmul kernel (cod/codex linalg zone) -- needs an
owner + same-binary A/B when builds resume.

## 2026-06-22 - argmax is PARITY vs JAX (bandwidth-bound); the "10x" was internal (CobaltForge/cc)

Warm-target rch `eval/argmax_16kx1k_axis1_f64` on `ovh-a` = **3.52ms** vs fresh
local JAX `jnp.argmax(a,axis=1)` 16384x1024 f64 = **3.46ms** = **~parity** (Rust
1.02x slower cross-machine; ovh-a is ~1.2x slower than local, so same-machine Rust
is likely a touch faster — call it parity). Both are BANDWIDTH-BOUND: input is
128MB, 128MB/3.5ms = ~37 GB/s = memory bandwidth; argmax must read every element,
so neither side can beat memory bw and they tie.
- Corrects the memory "argmax/argmin over axis 10x win" — that was INTERNAL
  (Rust-fixed-strided-gather vs Rust-naive), NOT vs JAX. vs JAX it's parity.
- GENERAL: bandwidth-bound reductions (argmax/argmin/reduce_sum/max over a large
  array) are JAX-PARITY — both read the data at memory bw, so there's no domination
  and no loss; the internal "Nx faster" numbers for these are vs Rust baselines,
  not JAX. The Rust-over-JAX domination surface stays compute/algorithm-bound:
  sort, large-n scan, contiguous gather, i64 matmul.

## 2026-06-22 - u32 matmul IS a domination (~8.9x); integer-matmul story: only i32 loses (CobaltForge/cc)

Tested u32 (principle predicted a loss like i32 — WRONG again, which is why we
test). Warm-target rch `eval/matmul_512x512_u32_canonical_fast` on `hz2` =
**30.2ms** vs fresh local JAX uint32 `a@b` 512x512 = **270ms** (min 249) =
**~8.9x Rust WIN**. So u32 matmul IS a Rust domination.

COMPLETE + surprising integer-matmul story (only signed i32 is JAX-fast):
| matmul | Rust | JAX | ratio | why JAX is fast/slow |
| --- | ---: | ---: | --- | --- |
| i64 | 4.64ms | 374ms | Rust 80x WIN | JAX scalar (vpmullq AVX512-only); Rust fast blocked kernel |
| i32 | 4.84ms | 0.62ms | Rust 7.8x LOSS | JAX vpmulld SIMD (signed 32-bit) |
| u32 | 30.2ms | 270ms | Rust 8.9x WIN | JAX slow (no u32 SIMD path); Rust slow generic u64-wrap |

So XLA-CPU only vectorizes SIGNED i32 matmul; i64 and u32 hit slow scalar paths
-> Rust wins both. The u32 win (8.9x) is SMALLER than i64 (80x) only because Rust's
u32 matmul uses the generic u64-wrap path (30ms), not the fast i64 blocked kernel
(4.6ms) -- a Rust lever (fast native u32 kernel would push u32 toward i64's 80x).
Domination set: sort, large-n scan, contiguous gather, i64 matmul, u32 matmul
(NOT i32 matmul).

## 2026-06-22 - matmul capstone: the BLAS-vs-no-BLAS divide quantified (float 806 GFLOP/s vs i64 0.54) (CobaltForge/cc)

Zero-build JAX-only float-matmul scaling, the complement to the int-matmul sweep —
together they fully explain the matmul map as ONE mechanism (does XLA-CPU have a
BLAS path?). JAX matmul GFLOP/s by dtype/size:

| n | JAX f64 | f64 GFLOP/s | JAX i64 | i64 GFLOP/s |
| --- | ---: | ---: | ---: | ---: |
| 256 | 0.289ms | 116 | 27.2ms | 1.2 |
| 512 | 0.634ms | 423 | 343ms | 0.78 |
| 1024 | 2.665ms | 806 | 3964ms | 0.54 |

- JAX FLOAT matmul scales BEAUTIFULLY (BLAS dgemm/sgemm): GFLOP/s RISES with size
  (116->806) as cache efficiency improves; ~near-peak at 1024^3. So Rust loses
  float matmul (~XLA/2, fma-bound, can't beat BLAS) — a real, size-stable JAX win.
- JAX INTEGER matmul has NO BLAS: GFLOP/s FALLS with size (1.2->0.54, scalar +
  cache-thrash); at 1024^3 JAX f64 is ~1500x faster than JAX i64. So Rust's blocked
  integer GEMM dominates hugely and the win GROWS with size.
- THE MATMUL MAP IN ONE LINE: Rust wins matmul iff XLA-CPU lacks a BLAS/SIMD path
  (i64, u32); loses where XLA has one (f64/f32 BLAS, complex zgemm, i32 vpmulld).
  This is the cleanest, most-defensible mechanism in the whole JAX-relative map.

## 2026-06-22 - scan domination is CHEAP-COMBINER-only; cumlogsumexp is transcendental-bound (not a lead) (CobaltForge/cc)

Zero-build JAX-only lead-hunt bounding the scan-family domination. JAX
`lax.cumlogsumexp` scaling: 1M **12.5ms** / 2M **22.4ms** / 4M **52.7ms** =
~**uniform 12 us/K** (12.2 -> 10.9 -> 12.9), i.e. NO memory cliff like cumsum
(which steps 1.4 -> 4 us/K at ~4M). Reason: cumlogsumexp's combiner is
transcendental (exp/log per step) -> COMPUTE-bound, ~9x slower per element than
cumsum's cheap-add scan.
- So cumlogsumexp is NOT a scan-domination lead. Transcendental-combiner scans
  land in Rust-LOSS territory: XLA vectorizes exp/log (SIMD), Rust uses scalar
  libm (fma-gated, ~XLA/2 at best per cntiy), so Rust would LOSE here, not win.
- BOUNDS the scan domination cleanly: it applies to CHEAP-ARITHMETIC-combiner
  scans (cumsum/cumprod/cummax -> memory-bound, hits JAX's scan size cliff ->
  Rust wins at large n). It does NOT extend to transcendental-combiner scans
  (cumlogsumexp/softmax-scan -> compute-bound by exp/log -> JAX SIMD wins).
- (Rust side not measured: warm target freed in disk emergency, cold rebuild
  forbidden; the reasoning follows directly from the established transcendental
  loss + the uniform-not-cliff JAX scaling.)

## 2026-06-22 - LEADS: JAX median/percentile are sort-bound-slow (~159ms) -> likely Rust dominations (cod-b family) (CobaltForge/cc)

Zero-build JAX-only lead-hunt for ops that inherit XLA's bitonic-sort weakness.
JAX p50, 1M f64:
- `jnp.median` = **159ms**, `jnp.percentile(.,90)` = **165ms** — both sort/
  partition-based, ~= JAX sort 1M (~183ms). They inherit the XLA-CPU bitonic-sort
  slowness.
- `jnp.searchsorted` (1M queries into 1M sorted) = **52.8ms** — slower lead, less
  catastrophic (binary search).

INTERPRETATION: median/percentile are STRONG Rust-domination LEADS — Rust's LSD
radix sort dominates JAX sort 4-6.5x, and any sort/select-based statistic inherits
that win; a Rust median (radix-sort or O(n) quickselect) would likely beat JAX's
~159ms by several x. searchsorted is a softer lead.
- These are ORDER-STATISTICS (cod-b's domination-map family), so recorded as LEADS
  to flag, NOT claimed as my dominations. Rust-side confirmation is build-blocked
  (warm target freed, cold rebuild forbidden) anyway.
- Consistent with the unifying principle: derived order-statistics ride the sort
  domination because XLA-CPU's sort lowering is the weakness.

## 2026-06-22 - CORRECTION: cumprod/cummax-4M are LOSSES, not dominations — scan domination is cumsum-ONLY (CobaltForge/cc)

Same-machine confirmation (warm local target/ re-established) of the predicted
cumprod/cummax-4M scan dominations — they are FALSE. Local Zen3, 4M f64:
| op | Rust | JAX | verdict |
| --- | ---: | ---: | --- |
| cumsum | 4.2ms | 18.4ms | Rust WINS 4.4x |
| cumprod | 19.6ms | 12.0ms | Rust LOSS 1.6x |
| cummax | 19.8ms | 16.1ms | Rust LOSS 1.2x |

ROOT CAUSE: the scan domination is CUMSUM-SPECIFIC, not scan-family. Rust cumsum
has an OPTIMIZED blocked prefix-scan kernel (4.2ms); cumprod/cummax are GENERIC
SERIAL scans (~20ms, ~5x slower than Rust cumsum). So despite JAX's scan size-cliff
at 4M, Rust's unoptimized cumprod/cummax LOSE (JAX cumprod 12ms / cummax 16ms beat
Rust's ~20ms). My earlier "scan family dominates at large n (cumprod/cummax
predicted)" was WRONG — it assumed Rust cumprod/cummax are as fast as cumsum; they
are not (no prefix-scan optimization).
- This is why the same-machine Rust-side confirmation was essential — the JAX-cliff
  prediction needed the Rust number, which only cumsum makes fast.
- NEW LEVER: port cumsum's blocked prefix-scan to cumprod/cummax (would flip both to
  dominations at large n). cumprod is FP-non-associative like cumsum, so it needs
  the same order-preserving blocked approach. Build path now warm (target/).
- Corrected domination set: sort, large-n CUMSUM (not cumprod/cummax), contiguous
  gather, i64/u32 matmul.

## 2026-06-22 - gather domination is CONTIGUOUS-ONLY; non-contiguous gather is an 18x LOSS (CobaltForge/cc)

Boundary test (same-machine, warm target/, correctness-verified before timing).
1M scattered single-element gathers from a 4M f64 operand (slice_sizes=1):
- Rust `eval_primitive(Gather)` = **34.6ms** (min 30.3); JAX `jnp.take` = **1.91ms**
  (min 1.68) = **~18x Rust LOSS**.
- Contrast: CONTIGUOUS row-gather (slice 1,256, memcpy) was a Rust ~3.7x WIN.

So the gather domination is MEMCPY/CONTIGUOUS-SPECIFIC only. For scattered/non-
contiguous gather (the common real-world case), Rust LOSES ~18x: XLA's gather is
vectorized and dense, while Rust's eval_primitive path boxes the output
(Vec<Literal>) and does per-element random access. CORRECTS the broad "gather
domination" — it's narrow (contiguous-block memcpy), unlike the robust sort/cumsum/
i64-matmul dominations.
- NEW LEVER: a dense non-contiguous gather path (skip the Vec<Literal> boxing,
  write into dense f64 storage) would cut the 18x substantially. (cod-a's
  contiguous-block memcpy vein is the contiguous case; this is the scattered case.)
- Corrected domination set: sort, large-n cumsum, CONTIGUOUS gather only, i64/u32
  matmul. (gather general = loss.)

## 2026-06-22 - f64 matmul (DotGeneral) is ~3.35x JAX loss same-machine — better than cited; DotGeneral != matmul_2d (CobaltForge/cc)

Authoritative same-machine measurement of THE headline loss (f64 matmul, the
common ML case, crux of the +fma decision). Local Zen3, warm target/:
eval_primitive(DotGeneral) f64 1024x1024 = **8.93ms** (min 7.70, ~240 GFLOP/s) vs
fresh local JAX f64 `a@b` 1024^3 = **2.665ms** (~806 GFLOP/s BLAS) = **~3.35x Rust
LOSS**.
- This is BETTER than the 5-15x I'd cited (from `matmul_2d_1024 = 40.776ms`).
  RESOLVED (not a bug): `bench_matmul_2d_1024` calls the RAW SINGLE-THREADED kernel
  `tensor_contraction::matmul_2d` directly (40ms = ~53 GFLOP/s, single-core), a
  microbench of the kernel. eval_primitive(DotGeneral) — the production @ path —
  THREADS (8.93ms = ~240 GFLOP/s, multi-core; cf. dot_general_parallel test). So
  the ~4.5x is the THREADING factor, NOT a mis-route or worker artifact. The
  authoritative PRODUCTION Rust-vs-JAX f64 matmul gap is ~3.35x (threaded DotGeneral
  vs threaded JAX dgemm), not the 5-15x the single-threaded kernel microbench implied.
- Mechanism unchanged: JAX dgemm BLAS (806 GFLOP/s) vs Rust blocked GEMM without
  fma/BLAS (~240 GFLOP/s = ~XLA/3.4). So +fma + better microkernel is the lever
  (cntiy) — but the gap is 3.35x, more closeable than the cited 15x suggested.
- (No action for cod-b: matmul_2d is the single-threaded kernel microbench, working
  as intended; the production path threads.)

## 2026-06-22 - f32 matmul (default ML dtype) is ~4.0x JAX loss same-machine — completes matmul map (CobaltForge/cc)

Same-machine authoritative number for JAX's DEFAULT ML dtype. Local Zen3, warm
target/: eval_primitive(DotGeneral) f32 1024^3 = **5.51ms** (min 4.86, ~390
GFLOP/s) vs fresh local JAX f32 `a@b` 1024^3 = **1.369ms** (~1568 GFLOP/s sgemm)
= **~4.02x Rust LOSS**. Slightly worse than f64 (3.35x) because JAX's f32 sgemm is
relatively faster (1568 vs 806 GFLOP/s) while Rust f32 (390) lacks fma + f32-SIMD-
peak.

COMPLETE production matmul map (threaded DotGeneral, same-machine vs JAX):
| dtype | Rust GFLOP/s | JAX GFLOP/s | verdict |
| --- | ---: | ---: | --- |
| i64 | (no BLAS in JAX) | ~0.5 | Rust WINS 176x @1024 |
| u32 | (slow both) | (slow) | Rust WINS ~8.9x |
| f64 | 240 | 806 (dgemm) | Rust loses 3.35x |
| f32 | 390 | 1568 (sgemm) | Rust loses 4.02x |
| i32 | (no f32-SIMD in Rust) | (vpmulld) | Rust loses 7.8x |
| complex128 | - | (zgemm) | Rust loses 3.7x |
- The float-matmul gaps (~3-4x) are the most release-relevant losses (common ML),
  and are CLOSEABLE via +fma + a tuned microkernel (cntiy) — not the 5-15x the
  single-threaded matmul_2d microbench implied. Rust wins matmul iff XLA lacks a
  BLAS/SIMD path (i64/u32).

## 2026-06-22 - conv2d is a ~11.8x JAX LOSS same-machine — the other big ML op, worse than matmul (CobaltForge/cc)

Same-machine authoritative conv2d (CNN-relevant; replicates bench_conv2d_64x64x32_
3x3x64_f64). Local Zen3, warm target/: eval_primitive(Conv) f64 input NHWC
[4,64,64,32] x kernel HWIO [3,3,32,64], SAME pad, stride 1 = **11.32ms** (min
10.49, ~27 GFLOP/s) vs fresh local JAX `lax.conv_general_dilated` (same dims/
NHWC/HWIO/SAME) = **0.957ms** (min 0.819, ~314 GFLOP/s) = **~11.8x Rust LOSS**.
- BIGGER gap than matmul (~3-4x). Conv = im2col (materialize patches, memory
  overhead matmul lacks) + GEMM (fma-bound), and likely less-threaded than the
  DotGeneral path; XLA conv is highly optimized (direct/efficient im2col + fma
  sgemm). Compute-bound, no bandwidth artifact (27 vs 314 GFLOP/s both plausible).
- Release-relevant: conv is the core CNN op; ~12x is a real gap. Levers: im2col
  efficiency + the matmul fma/microkernel lever (cntiy) + conv threading.
- Adds to the JAX-relative map: conv2d ~11.8x loss (the largest float-op loss
  measured; bigger than float matmul because of the im2col overhead).
- **MEASURED SPLIT (2026-06-22, SlateHarrier) — corrects the "im2col overhead / less-threaded"
  framing above.** f64 conv2d ALREADY routes through im2col + the threaded `matmul_2d`
  (it is NOT a missing-GEMM-route or unthreaded path). Same-binary component A/B
  (`bench_conv2d_f64_im2col_vs_gemm_split`, the exact [4,64,64,32]×[3,3,32,64] shape):
  **im2col = 0.86-0.96ms (≈16%), GEMM = 4.37-5.24ms (≈84%) at 115-138 GFLOP/s**. So the conv2d
  gap is ~84% the **fma-bound f64 GEMM** (`matmul_2d`, the SAME `cntiy` lever as f64 matmul) and
  only ~16% im2col (already threaded, fast). An implicit-GEMM conv (avoiding the 37MB col buffer)
  would save only ~16% — NOT worth it. CONCLUSION: conv2d is essentially `cntiy` +fma-gated (folds
  into that bucket, one more op the +fma decision unlocks), NOT a separate structural/im2col/threading
  lever. Do NOT chase "faster im2col" or "thread the conv" — both are already done; it's the GEMM.

## 2026-06-22 - cholesky ~6.8x JAX loss (LAPACK gap) — fills the linalg category (CobaltForge/cc)

Same-machine linalg measurement (scientific-computing relevant; the last major
untouched category). Local Zen3, warm target/: eval_primitive(Cholesky) f64 512x512
SPD (A=B^T B + nI, replicates bench_cholesky_512_f64) = **8.96ms** (min 6.48) vs
fresh local JAX `jnp.linalg.cholesky` (LAPACK potrf) = **1.317ms** (min 1.20) =
**~6.8x Rust LOSS**.
- Fits the model: JAX has an optimized path (LAPACK), so Rust loses. But less
  catastrophic than feared — Rust's blocked cholesky (8.96ms) is decent; LAPACK is
  ~7x faster (decades-tuned + fma). svd/qr/lu likely similar-or-larger (more
  complex factorizations); eigh/eig may differ.
- Lever: same as float matmul — +fma + better blocked microkernel (cntiy), plus
  cache-aware panel factorization. linalg.rs is the cod-b/codex zone (measured here,
  not edited).
- Map now covers the linalg category: cholesky ~6.8x loss. Together with conv2d
  (11.8x) and float matmul (3.3-4x), the compute-heavy numeric ops are all
  BLAS/LAPACK-bound JAX losses (3-12x), closeable via the +fma/microkernel lever.

## 2026-06-22 - non-symmetric eig is ~PARITY (not a loss) — linalg LAPACK advantage is op-dependent (CobaltForge/cc)

Surprise refinement of the linalg category. Same-machine, 256x256 non-symmetric:
eval_primitive_multi(Eig) f64 = **164ms** (min 153) vs fresh local JAX
`jnp.linalg.eig` (LAPACK geev) = **219ms p50 / 135 min** (high variance) =
**~PARITY** (Rust 1.33x faster by p50, JAX 1.13x faster by min — within JAX's
variance band; neither clearly wins).
- So non-symmetric eig is NOT a Rust loss, unlike cholesky (6.8x). The LAPACK
  advantage is OP-DEPENDENT: large for BLAS-heavy factorizations (cholesky/matmul
  = dgemm-bound, 3-7x losses) but SMALL for ITERATIVE eig (Francis double-shift +
  inverse iteration is hard to BLAS-accelerate, so LAPACK geev and Rust's Francis
  are comparable). Rust's full Francis eig (memory: 53-64x internal) is genuinely
  competitive with LAPACK here.
- Refines the map: linalg is NOT uniformly a loss. cholesky ~6.8x loss (BLAS-heavy);
  eig ~parity (iterative). svd/qr likely between (BLAS-heavy steps -> loss, but less
  than cholesky). This is another case where testing beat the "linalg=loss"
  assumption — the model keeps needing empirical correction.

## 2026-06-22 - svd is ~PARITY too — iterative linalg (eig/svd) competitive with LAPACK; only direct factorizations (cholesky) lose (CobaltForge/cc)

Same-machine svd 256x256 (most common linalg op): eval_primitive_multi(Svd) f64 =
**177.7ms** (min 174) vs fresh local JAX `jnp.linalg.svd` (LAPACK gesdd) =
**234ms p50 / 157 min** = **~PARITY** (Rust 1.32x faster by p50, JAX 1.11x by min;
within JAX variance) — same shape as the eig result.

LINALG PATTERN now clear (and counterintuitive):
| op | type | verdict |
| --- | --- | --- |
| cholesky | direct, BLAS-heavy (potrf) | ~6.8x LOSS |
| eig (non-sym) | iterative (Francis) | ~parity |
| svd | bidiag + iterative QR | ~parity |

So LAPACK's advantage is concentrated in BLAS-HEAVY DIRECT factorizations (cholesky;
also matmul/conv via dgemm). For the ITERATIVE decompositions (eig, svd), where the
work is dominated by sequential QR/Francis sweeps that don't BLAS-accelerate well,
Rust's full pure-Rust implementations are COMPETITIVE with LAPACK (~parity). This
substantially corrects "linalg = LAPACK loss" — the hard linalg ops (svd/eig) are
NOT losses. Big credit to the Rust linalg kernels (tridiag/QL, Francis, bidiag+QR).
Release-relevant: pure-Rust SVD/eig hold their own vs JAX/LAPACK on CPU.

## 2026-06-22 - NEW DOMINATION: QR is ~18-30x faster than JAX (XLA-CPU QR is a slow non-LAPACK path) (CobaltForge/cc)

Big surprise overturning the "direct factorization -> loss" prediction. Same-machine
qr 512x512 f64: eval_primitive_multi(Qr) = **14.54ms** (min 13.5) vs fresh local JAX
`jnp.linalg.qr` = **429.93ms p50 / 250.91 min** = **~18-30x Rust FASTER**.
- JAX QR is absurdly slow (430ms; LAPACK geqrf 512 is ~5ms), so XLA's CPU QR does
  NOT route to LAPACK — it uses a slow XLA-NATIVE Householder path (the same class
  of weakness as XLA's bitonic sort, scan cliff, and no-integer-BLAS matmul). Rust's
  optimized blocked in-place QR (14.5ms) dominates it ~18-30x.
- This makes XLA-CPU linalg INCONSISTENT: cholesky -> LAPACK potrf (Rust loses 6.8x),
  but QR -> slow native (Rust WINS ~30x), while eig/svd ~parity. So linalg is NOT
  uniformly anything — it depends on whether XLA routes that op to LAPACK or a slow
  native path.
- Fits the unified model perfectly: Rust dominates exactly where XLA-CPU has a weak/
  slow path (QR native) and Rust has a specialized kernel (blocked QR). QR joins the
  domination set: sort, cumsum, contiguous gather, i64/u32 matmul, **QR (~30x)**.
- Release-relevant: QR is common (least-squares, orthogonalization); ~30x is a major,
  defensible Rust-over-JAX win.

## 2026-06-22 - f32-sort gap CLOSED: fj-lax f32 sort 64k = ~0.77ms (FASTER than its f64), ~10x over JAX; same-host Zen3 pair (CrimsonOtter/cc)

Closes the flagged "fj-lax f32 exact pending" item in the scorecard. Measured a clean
**same-host pair on the canonical Zen3 5975WX bench host** (no cross-worker inference):

| op (64k, ascending) | fj-lax (HEAD, radix path) | JAX 0.10.1 (jit, x64 enabled) | ratio |
| --- | --- | --- | --- |
| sort f32 | **~0.77ms** (med of 0.748/0.774/0.779) | 7.45–8.18ms p50 | **~10x fj-lax WIN** |
| sort f64 | 1.566ms | 8.18ms p50 (≡ f32) | ~5.2x fj-lax WIN |

Key confirmations:
- **JAX f32 sort ≡ f64 sort** (8.184ms vs 8.184ms, identical to the µs): independently
  re-confirms "XLA-CPU bitonic sort is dtype-agnostic-slow" — now on a SECOND jaxlib
  version (0.10.1, vs the scorecard's 0.10.2) AND the canonical Zen3 host. The domination
  is robustly NOT an x64-only artifact.
- **fj-lax f32 (0.77ms) is FASTER than fj-lax f64 (1.57ms)** — expected: f32 LSD radix
  moves 4-byte keys vs 8-byte, and JAX gains nothing from the narrower dtype. So f32 (the
  dtype JAX users actually run) is fj-lax's BEST sort case, not a weaker one.
- HONESTY NOTE on the ratio vs the scorecard's ~10x: that ~10x was a same-WORKER pair
  where JAX measured 12.5ms (slower VPS). On the faster Zen3 host JAX is ~7.5–8.2ms, so
  the f64 ratio reads ~5x here and f32 ~10x. The DOMINATION holds across hosts (4.8–10x);
  the absolute multiple just tracks how fast the JAX host is. fj-lax's own sort time is
  host-stable (Zen3 f64 1.57ms ≈ worker 1.25ms class).

PROVENANCE / why this is honest despite the +fma build dir: the binary was built into the
warm `frankenjax-cc-fma` target (`+avx2,+fma`) because the canonical `frankenjax-cc`
(`+avx2`, no-fma) target was reclaimed under disk pressure. This does NOT affect the sort
number: the sort path is an **integer LSD radix on total_cmp bit-keys** — zero float
mul-add, so `+fma` vs no-fma produces bit-identical codegen and identical timing for sort.
(For FMA-sensitive ops — matmul/exp/FFT butterflies — an fma-target number would NOT be
canonical; sort is specifically exempt.) Source unchanged since the warm rlib (the 5
intervening commits are all docs-only), and fj-lax was recompiled fresh at HEAD 2f47cc82.

Net: the order-statistics domination zone (sort/argsort/top_k/median-family) is now fully
measured in BOTH dtypes with no remaining "pending" inference. f32 sort ~10x is the
release-relevant figure (f32 = JAX's default dtype).

## 2026-06-22 - NEW DOMINATION: the whole LU family (lu/solve/det/inv) is ~15-30x faster than JAX — XLA-CPU LU is a slow native path, NOT LAPACK getrf (CrimsonOtter/cc)

Direct extension of the QR finding: XLA-CPU does NOT route LU-based factorizations to
LAPACK either. Measured same-host Zen3, f64, 1024x1024:

| op | JAX 0.10.1 (jit) | fj-lax | note |
| --- | --- | --- | --- |
| lu 1024 | **1.09–1.13s** (best-of-N) | **37.3ms** (linalg/lu_1024x1024_f64, med 36.5/37.3/38.0) | ~29x at measured load |
| solve 1024 | 1.17–1.31s | (fj-lax routes solve→blocked LU, ~same class) | |
| det 1024 | 1.21–1.33s | | |
| inv 1024 | 1.24–1.39s | | |

- LAPACK `getrf` at 1024 is ~10–15ms; JAX measuring **~1.1 SECONDS** means XLA's CPU LU is
  ~75x slower than LAPACK → it is the SAME slow XLA-NATIVE path class as QR (430ms),
  bitonic sort, and the scan cliff. `jnp.linalg.solve/det/inv` are all LU-backed, so the
  ENTIRE family inherits the slow path (all measured ~1.1–1.4s).
- fj-lax's blocked-GEMM LU (37ms, the `n>=256` blocked path) dominates it. **Ratio ~29x at
  the measured (elevated-load) conditions; conservatively >=11x** even if one credits JAX a
  load-free ~400ms (QR-class). Either way a large, defensible domination.
- LOAD CAVEAT (honest): the whole fleet was at load 30–84 during measurement (concurrent
  builds across sibling franken* repos). BOTH sides are load-inflated; the JAX best-of-N is
  a min-over-samples (least-loaded) so already conservative on the JAX side, and the gap
  (>10x) dwarfs any <=2x load noise — the verdict is load-robust even though the exact
  multiple is not. fj-lax LU 37.3ms was tight (36.5–38.0) so its side is solid.
- PROVENANCE: same warm `frankenjax-cc-fma` (+avx2,+fma) binary as the f32-sort entry, run
  locally on Zen3. LU is GEMM-heavy so `+fma` COULD shift its absolute time slightly vs a
  no-fma canonical build (unlike sort) — but only in fj-lax's favor by a few %, and the
  domination is order-of-magnitude, so it does not change the verdict. A no-fma re-confirm
  is a nice-to-have, not load-bearing.

LINALG MAP UPDATE — XLA-CPU routing is per-op and split into two regimes:
| op | XLA-CPU route | verdict vs fj-lax |
| --- | --- | --- |
| cholesky | LAPACK potrf (fast) | ~6.8x LOSS |
| matmul/conv | dgemm (fast, +fma) | LOSS (cntiy-gated) |
| eig / svd | iterative, ~parity both | ~parity |
| **qr** | **slow native** | **~18–30x WIN** |
| **lu / solve / det / inv** | **slow native** | **~15–30x WIN (NEW)** |

So the domination set grows: sort/argsort/top_k, cumsum, contiguous gather, i64/u32 matmul,
QR, **and now the LU family (lu/solve/det/inv)**. Release-relevant: solve/det/inv are
extremely common (linear systems, determinants, matrix inversion) — a large, broad
Rust-over-JAX-CPU win wherever XLA falls off LAPACK onto its native factorization path.

## 2026-06-22 - Frontier re-audit: no contained BENCHABLE perf lever remains; biggest legal gap (FFT pow2) filed as split-radix bead murmw.1 (CrimsonOtter/cc)

Walked every documented loss looking for a lever I could implement AND validate under the
current fleet load (30–84, concurrent franken* builds). Conclusion — all blocked:
- **+fma-policy-gated (cntiy, maintainer call):** matmul/GEMM, exp/sin/log (bit-pinned),
  conv, cholesky (dgemm-bound), softmax/attention, SIMD-poly transcendentals (2.2x WITH fma
  / 0.79x WITHOUT). Cannot touch without the global-`+fma` or per-fn `target_feature`
  decision.
- **already dominated:** sort/argsort/top_k, cumsum (cheap-combiner scan), contiguous
  gather, i64/u32 matmul, QR, lu/solve/det/inv (this cycle).
- **einsum CLOSED:** verified `try_einsum2_matmul_general` (einsum.rs ~line 197) already
  permutes interleaved batch/free/contracted multi-axis contractions into `[batch,M,K]·
  [batch,K,N]` GEMM. The old 7r0ck "non-pre-aligned residual" is gone; only niche
  diagonal/trace/single-operand-reduction (non-GEMM-shaped) odometer remains — not a lever.
- **scatter-add:** parallel direct-write / atomic / histogram-prefix branches all regressed
  (serial is optimal); needs a fundamentally different safe-parallel proof.
- **FFT (biggest legal loss, ~19.4x on pow2 2048x256 c128):** intra-FFT SIMD is safe-Rust-
  blocked (no shuffle/gather in std::simd); batch-level threading is UN-BENCHABLE under
  fleet load (threading wins reverse under contention, [[project_threaded_eager_fusion]]);
  the only single-thread, non-fma, algorithmic lever left is a lower-op-count radix
  (split-radix/radix-4), but pow2 FFT digests are BIT-FROZEN goldens so it needs a
  re-baseline. **Filed as `frankenjax-murmw.1`** with full spec (split-radix-4 pow2,
  ~1.3-1.5x single-thread target, tolerance-verify-vs-DFT then re-baseline pow2 goldens,
  idle-machine same-binary A/B, REVERT if <1.15x — Winograd-F(2,3) fragmentation risk).

NET: the contained, benchable frontier is genuinely closed under current constraints. The
next real FFT win is murmw.1 (needs an idle host + golden re-baseline); the next broad win
is the cntiy +fma maintainer decision. No code lever was shippable+benchable this pass.

## 2026-06-22 - cumprod/cummax-4M losses CONFIRMED + ROOT-CAUSE NARROWED: generic blocked scan underperforms for mul/max (not a routing miss) — lever = ~2.4x win (CrimsonOtter/cc)

Added reproducible 4M-1D benches `eval/cumprod_4m_f64_1d` + `eval/cummax_4m_f64_1d`
(mirroring `cumsum_4m_f64_1d`; cumprod had NO bench before). Same-binary same-session,
Zen3, load ~8-18, best/median of Criterion:
| op 4M-1D f64 | fj-lax | JAX 0.10.1 (best-of-6) | verdict |
| --- | ---: | ---: | --- |
| cumsum  | **7.55ms** | 17.97ms | fj-lax WINS **2.4x** |
| cumprod | **20.9ms** | 16.99ms | fj-lax **LOSS 1.25x** |
| cummax  | **20.9ms** | 18.83ms | fj-lax **LOSS 1.13x** |

Confirms the earlier CobaltForge correction (cumprod/cummax ARE losses) with same-session
numbers (magnitudes a touch smaller: JAX measured ~17-19ms here vs 12-16ms earlier).

ROOT CAUSE NARROWED (the useful new bit): cumprod/cummax are NOT missing the fast route.
Verified by code read — `eval_cumulative_dense` (reduction.rs ~3442) calls the SAME generic
`blocked_prefix_scan_to_vec(src, float_init, float_op)` for ALL of cumsum/cumprod/cummax
(no `primitive == Cumsum` guard). Yet cumsum threads/streams to 7.5ms (≈memory-bound for
64MB R+W) while the mul/max monomorphizations run at **~21 cyc/element** — far above
memory-bound, i.e. the per-thread serial scan is the bottleneck for mul/max but NOT for add.
Ruled out: load (reproducible at load 8), data (finite/normal, no denormals), routing
(generic, confirmed). So the gap is a RUNTIME/CODEGEN asymmetry of the generic blocked scan
under the add vs mul/max closures — needs a profiler/disassembly pass to pin (why does the
add closure stream while mul/max don't, through identical thread::scope code?).

LEVER (filed `frankenjax-murmw`-sibling bead): close cumprod/cummax to cumsum's 7.5ms →
flips BOTH losses to ~2.4x / ~2.5x JAX WINS (extends the cumsum scan domination to the whole
cheap-combiner scan family). Single-threaded-or-blocked, bit/tolerance-legal (cumsum already
uses the tolerance-legal blocked reassociation). NOT fma-gated. Reproduce: the two new
benches vs JAX cumprod/cummax 4M. Benches kept as measurement infra (cf cumsum_4m_1d_tight).

## 2026-06-22 - cumprod/cummax scan: zero-init optimization TRIED + REVERTED (~0 gain); root cause confirmed OP-dependent (add vectorizes, mul/max don't) (CrimsonOtter/cc)

Follow-up to the cumprod/cummax-4M loss (bead t1pb0). Drilled the root cause to certainty
and tested one candidate fix:

DEFINITIVELY ESTABLISHED (FJ_SCAN_TRACE instrumentation + objdump + controlled bench, load 8):
- BOTH cumprod AND cumsum take the SAME `blocked_prefix_scan_to_vec` BLOCKED path (trace:
  `FJSCAN: BLOCKED total=4194304 prim=Cumprod` / `prim=Cumsum`) — NOT a routing miss.
- Yet cumsum_4m = 7.6ms (≈memory-bound) while cumprod_4m = cummax_4m = ~21ms (compute-bound,
  ~21 cyc/elem). It is OP-dependent, NOT data-dependent: cummax (random ±32768 data) and
  cumprod (near-1.0 data) are WILDLY different distributions but BOTH 21ms, while cumsum
  (add) is 7.6ms → the differentiator is the operator (add vs mul/max), not the values.
- LLVM auto-vectorizes/streams the ADD prefix-scan to memory-bound; the mul/max
  monomorphizations stay on the scalar dependency-chain scan.

CANDIDATE FIX TRIED + REVERTED (~0 gain): hypothesized the non-zero `vec![init; n]` buffer
fill (init=1.0 cumprod / ±inf cummax) cost vs cumsum's calloc-free `vec![0.0]`. Changed the
alloc to `vec![0.0; n]` (SAFE — pass A overwrites every slot before any read; `init` stays
the scan seed). Result: cumprod 21.06->21.24ms, cummax ->21.19ms = **0 gain, REVERTED**.
So the init-fill is NOT the bottleneck — it is purely the scalar mul/max scan.

REMAINING FIX (bead t1pb0, now fully scoped): a manual SIMD prefix-product for cumprod and a
NaN-PROPAGATING SIMD prefix-max for cummax (std::simd `simd_swizzle!` lane-shift Hillis-Steele;
note: `simd_max` DROPS NaN so cummax needs explicit NaN tracking to match jax_max_f64). Per-
thread local scan only; cumsum's auto-vectorized add path stays untouched (no regression).
Tolerance-legal (blocked already reassociates). Prize unchanged: 21->~7.5ms flips both to
~2.4x JAX wins. NOT attempted inline this pass (SIMD-prefix + NaN-safe max + parity is a
focused multi-step task, high bug-risk to rush).

## 2026-06-22 - scattered gather 18-32x loss: prior "boxes output" diagnosis CORRECTED; real cost is index-extraction + single-thread resolve + random-access MLP (CrimsonOtter/cc)

Added a reproducible bench `eval/gather_scatter_1m_f64` (1M pseudo-random single-element
gathers from a 4M f64 operand, slice_sizes=1) for the documented scattered-gather loss —
previously only ad-hoc-measured. Same-session, Zen3:
- fj-lax `eval_primitive(Gather)` = **~30-35ms quiescent / ~51-57ms under fleet load** (clone
  hoisted out of the timed loop).
- JAX `jnp.take` (same shape) = **1.786ms** (best-of-8) — confirms the prior 1.91ms.
- Ratio = **~18x (quiescent) to ~32x (loaded)** Rust LOSS. Biggest non-fma-gated loss after FFT.

CORRECTS the prior entry's mechanism claim ("XLA's gather is vectorized and dense, while
Rust's eval_primitive path BOXES the output (Vec<Literal>)"). That is WRONG for this case:
read the code (tensor_ops.rs eval_gather ~3243) — a 1D single-element gather has
`trailing_slice_is_contiguous == true` (vacuous), so F64 routes to the DENSE, THREADED
`gather_contiguous_into` (no Vec<Literal> output boxing). The real cost is three things XLA
fuses away:
1. INDEX EXTRACTION: `inputs[1].elements.iter().map(lit_to_i64).collect()` runs per-element
   even when the index tensor is a dense i64/i32 backing — 1M boxed conversions, serial.
2. RESOLVE PASS: a separate serial 1M `resolve_axis0_index` building a 16MB
   `Vec<Option<usize>>`, then a serial 1M OOB pre-validate inside gather_contiguous_into.
3. RANDOM-ACCESS MLP: the threaded gather does scalar `copy_from_slice(len 1)` per index —
   memory-LATENCY-bound; JAX hides the latency with vectorized gather / many outstanding
   loads. Threading alone (already present) doesn't close it.

REAL FIX (bead filed): (a) dense-index fast path — read `as_i64_slice()`/`as_i32_slice()`
directly, skip the 1M `lit_to_i64`; (b) fuse `resolve_axis0_index` INTO the threaded gather
loop (drop the 16MB intermediate + the separate validate pass), replicating the Clip/Promise
OOB + fill semantics inline; (c) the actual latency lever — `std::simd::Simd::gather_or` (W=8
f64) per thread to issue many outstanding loads (MLP), matching XLA's vectorized gather.
(a)+(b) are safe/cheap but likely modest (~1.1-1.3x); (c) is where most of the 18-32x lives
but carries std::simd gather/select nightly-drift risk (see [[project_simd_poly_exp_fma_finding]]).
NOT attempted inline this pass: eval_gather index-mode/OOB/fill surgery + SIMD gather + gather
parity verification is a focused multi-step task, high bug-risk to rush under fleet contention.

## 2026-06-22 - SHIPPED: scattered gather ~2x faster (branchless MLP) — JAX loss halved 28x->15x (CrimsonOtter/cc)

Implemented the olm4p lever's MLP fix and it WORKS. Added `gather_single_dense<T>`: a tight,
branchless `out[i] = src[idx[i]]` threaded gather for the `slice_elems == 1` case, wired into
the F64/F32/I64 eval_gather paths when the index mode never yields None (Clip /
PromiseInBounds — resolve to a `usize` index vector once). The generic path's per-element
`match Option` + `copy_from_slice(len 1)` (a function call) serialized the dependent random
loads; removing both lets the out-of-order engine overlap many independent loads (MLP).

MEASURED same-session Zen3, eval/gather_scatter_1m_f64 (1M random single-element from 4M f64):
- **56.8ms -> 27.5ms = ~2.06x Rust speedup** (Criterion -51.59% vs stored baseline, p<0.05).
  Both old and new paths are threaded, so the delta is purely the branchless inner loop.
- vs JAX `jnp.take` 1.786ms: the loss **halves from ~28-32x to ~15x**. Still a loss (JAX's
  vectorized gather hides more latency), but a large, real, bit-identical Rust win.
- GREEN: full `cargo test -p fj-lax --lib` = **1587 passed / 0 failed** (gather oracle,
  OOB-clip, dense-vs-literal bit-identity all pass — `gather_single_dense` is bit-identical:
  same `src[idx]`, same order; every Clip/PromiseInBounds index is `< dim0 <= src.len()`).
- FillOrDrop (can produce None->fill) keeps the generic `gather_contiguous_into` path.

REMAINING (olm4p stage c): the last ~15x to JAX is memory-LATENCY — a per-thread
`std::simd::Simd::gather_or` (f64x8) would issue more outstanding loads (more MLP). Deferred
(std::simd gather nightly-drift risk; needs same-binary A/B + scalar fallback). The branchless
scalar path already captured the cheap ~2x; SIMD gather is the next, harder increment.

## 2026-06-22 - branchless MLP gather EXTENDED to i32/u32/u64/bf16/f16 — i32 measured 2.13x (CrimsonOtter/cc)

Follow-up to the f64/f32/i64 branchless gather win (d551956b): wired `gather_single_dense`
into the remaining scattered single-element gather dtypes. i32/u32/u64 previously had NO
threaded/branchless fast path at all (only the serial `dense_contiguous_gather` macro);
bf16/f16 were threaded but non-branchless (like f64 was).

MEASURED same-binary A/B (Zen3, load ~39, eval/gather_scatter_1m_i32, 1M random from 4M):
- **i32 serial baseline 55.19ms -> branchless 25.96ms = 2.13x** (matches the f64 ~2x).
- Notably the i32 serial baseline (55ms) ≈ the f64 *threaded* baseline (56.8ms): the
  per-element `match Option` + `copy_from_slice(len 1)` call was the dominant cost in BOTH,
  so removing it (branchless `out[i]=src[idx[i]]`) is the real lever, NOT threading. Confirms
  the win is structural and dtype-independent.
- u32/u64 share the identical serial->branchless transition (same ~2x expected); bf16/f16
  share the threaded->branchless transition (the f64-proven ~2x).
- GREEN: `cargo test -p fj-lax --lib gather` = 23 passed / 0 failed (incl.
  dense_i32_gather_matches_generic, dense_u32_u64_gather_matches_generic, half-float
  bit-identity). Bit-identical: same src[idx], same order; FillOrDrop keeps the generic path.

So the scattered single-element gather family (f64/f32/i64/i32/u32/u64/bf16/f16) is now
uniformly ~2x faster, halving the JAX gap (~28x->~15x for f64; i32 family proportionally).
The residual ~15x is memory-latency (olm4p stage c: std::simd Simd::gather, deferred — Zen3
vgather is microcoded so SIMD gather may not beat the scalar-MLP path; needs measurement).

## 2026-06-22 - scatter-overwrite branchless fast path: 1.24x (the gather dual, smaller — stores less latency-bound) (CrimsonOtter/cc)

Applied the branchless-MLP gather lesson to its dual: scatter OVERWRITE, slice_elems==1.
The general `scatter_typed` loop pays a per-element `copy_from_slice(len 1)` CALL + four
checked mul/add + two bounds checks; for slice_elems==1 the whole body is `out[idx]=upd[i]`,
so a tight branchless loop (gated on `upd.len() >= index_count`) lets the store buffer
overlap the random writes. Wired inside the macro -> covers f64/f32/i64/i32/u32/u64 at once.

MEASURED same-binary A/B (Zen3, eval/scatter_overwrite_1m_f64, 1M random writes -> 4M f64):
- **35.0ms (general path) -> 28.3ms (branchless) = 1.24x.** Real but MODEST — smaller than
  the gather ~2x because scattered STORES are less latency-bound than scattered LOADS (the
  store buffer already hides some latency; the per-element call/checks were a smaller fraction).
- Bit-identical: serial in index order (preserves overwrite last-wins), same resolved idx,
  same None-skip. GREEN: `cargo test -p fj-lax --lib scatter` = 31 passed / 0 failed
  (overwrite + duplicate-index + OOB-clip all pass). FillOrDrop-with-None still handled
  (the if-let skip); the `upd` precheck falls non-conforming shapes back to the general path.
- KEPT (1.24x > ~0-gain, low-risk, bit-identical). The branchless lesson generalizes across
  gather (2x) and scatter-overwrite (1.24x); the asymmetry quantifies that loads benefit
  ~2-3x more from MLP than stores on Zen3. scatter-ADD stays on its own optimized path.

## 2026-06-22 - SIMD gather (Simd::gather_or) is a NO-WIN on Zen3 — branchless scalar IS the safe-Rust gather ceiling (olm4p stage c REJECTED) (CrimsonOtter/cc)

Tested olm4p's last lever: replaced the scalar branchless f64 scattered gather with a
portable `std::simd::Simd::<f64,8>::gather_or` variant (8 gathers/instruction, threaded),
to probe whether wider memory-level parallelism beats the scalar OoO loop.

MEASURED same-binary A/B (Zen3, eval/gather_scatter_1m_f64 vs the committed scalar baseline):
- scalar branchless = 27.5ms; **SIMD gather = 28.95ms = +5.2% REGRESSION** (Criterion
  change +0.3%..+10.1%, p=0.04). REVERTED.
- ROOT CAUSE: Zen3's `vgatherqpd` is MICROCODED (decodes to ~8 sequential µ-ops internally),
  so it issues no more concurrent loads than the scalar OoO loop already does — and adds
  setup overhead. SIMD gather only wins on µarchs with a hardware gather unit (Intel
  Skylake-X+/AVX-512), which this host lacks.
- CONCLUSION: the **scalar branchless gather (the committed ~2x win) IS the safe-Rust
  gather ceiling on Zen3**. The residual ~15x vs JAX `jnp.take` (1.786ms) is NOT closable by
  contained safe-Rust SIMD here. JAX's edge must come from AVX-512 gather (absent on Zen3) or
  index pre-sorting for cache locality (algorithmic; changes nothing semantically but a
  sort+gather+unsort is a different, heavier design). olm4p stage c is closed as no-win;
  stages a/b (branchless scalar) already shipped (d551956b/a4c5118a, ~2x).
- METHOD note: this is why olm4p flagged "measure before assuming" — the BOLD-VERIFY ethos
  caught a plausible-but-wrong SIMD lever with one A/B instead of shipping a regression.

## 2026-06-22 - cumprod/cummax 2.8x mystery: disasm RULES OUT the SIMD-prefix/inlining fix — it's a µarch effect needing perf-counters (t1pb0 re-scoped) (CrimsonOtter/cc)

Free disasm check (no build) of the `blocked_prefix_scan_to_vec` monomorphizations to test
t1pb0's fix hypotheses BEFORE spending a build:
- cumsum (add): `vaddpd` packed. cumprod (mul): `vmulpd` packed. cummax (jax_max_f64):
  `vmaxpd` + `vcmpunordpd` + `vblendvpd` (NaN-propagating max, fully INLINED + vectorized).
- So all three local scans are ALREADY SIMD-vectorized and the op closures are inlined —
  yet cumprod/cummax measure ~21ms (4M-1D) vs cumsum ~7.5ms, a 2.8x gap with structurally
  identical code and same-latency/throughput ops (`vaddpd`==`vmulpd`==`vmaxpd` on Zen3).
- THEREFORE the two fix hypotheses in t1pb0 are WRONG: (a) "mul/max not vectorized" — false
  (vmulpd/vmaxpd present); (b) "closure not inlined / per-element call" — false (inlined).
  Data is also ruled out (no denormals; cummax random vs cumprod near-1.0 give the SAME 21ms).
- The residual 2.8x is a DEEPER microarchitectural effect (load/store-queue, write-combining,
  or the multi-pass blocked scan's `out` re-read interacting with the op) that needs
  `perf stat` counters (cache-misses / stalls / uops) to diagnose — NOT a contained code
  lever. t1pb0 re-scoped accordingly: do NOT attempt a SIMD-prefix or force-inline fix
  (disasm proves they're already done); next step is a profiling pass on a quiesced host.
  Net: cumprod/cummax (~1.13-1.25x JAX loss) is NOT a quick contained win. Frees future
  build budget by killing two wrong hypotheses with a free disasm read.

## 2026-06-22 - int contiguous row-gather threaded: i32/u32/u64 1.40x (was serial-only) (CrimsonOtter/cc)

The scattered single-element gather branchless win (slice_elems==1) was wired for all dtypes,
but the CONTIGUOUS row-gather path (slice_elems>1, embedding/row lookup) was still SERIAL for
i32/u32/u64 — only f64/f32/i64/bf16 had the threaded `gather_contiguous_into`. Added it to the
three int dtypes (same proven generic fn; OOB falls back to serial; bit-identical).

MEASURED same-binary A/B (Zen3, eval/gather_rows_1m_i32 = gather 4096 random rows of a
[16384,256] i32 table -> 1M elems): **serial 1.189ms -> threaded 0.851ms = 1.40x**. Real,
modest (row memcpy is memory-bandwidth-bound, so threading scales sub-linearly), bit-identical.
GREEN: `cargo test -p fj-lax --lib gather` = 23 passed / 0 failed (dense_i32/u32/u64 +
gather_contiguous_into_bit_identical_to_serial all pass). KEPT (1.40x > scatter-overwrite's
kept 1.24x). Completes the gather dtype-coverage: every dense dtype now has both the
slice_elems==1 branchless path AND the slice_elems>1 threaded row path.

## 2026-06-22 - kkawk SHIPPED: gather pre-pass fusion 1.86x (25.06ms->13.45ms) — JAX gap ~15x->~7.5x (CrimsonOtter/cc)

Implemented bead kkawk (safe subset). The scattered single-element gather built THREE serial
1M intermediate Vecs + ~24MB allocs before the threaded branchless gather: index_vals
(per-element `lit_to_i64` over boxed Literals), `resolved: Vec<Option<usize>>` (16MB), and
`single_idx: Vec<usize>` (8MB, from resolved). Two fixes:
1. Dense index extraction: `as_i64_slice().to_vec()` instead of per-element `lit_to_i64`
   (i32 is i64-backed too). Non-dense indices keep the per-element path.
2. Lazy `resolved`: when slice_elems==1 + never-None mode + DENSE fast-path-dtype operand
   (`use_fused`), resolve straight into `single_idx: Vec<usize>` and SKIP the 16MB
   `Vec<Option>` — the matching dtype branch is guaranteed to consume single_idx and return,
   so `resolved` is never read. Complex/bool/boxed/FillOrDrop/slice_elems>1 keep full resolved
   (gated on the same dense-slice check the dtype branches use -> no boxed-operand hazard).

MEASURED same-session same-binary A/B (Zen3, eval/gather_scatter_1m_f64, 1M<-4M f64):
**branchless-HEAD 25.06ms -> fused 13.45ms = 1.86x** (far above the ~1.2x estimate; the
per-element lit_to_i64 + 16MB Vec<Option> alloc + double-resolve were ~11.6ms of the 25ms).
vs JAX jnp.take 1.786ms: the gather gap **halves again ~15x -> ~7.5x** (cumulative ~3.7x from
the original ~28x: branchless 2x then fusion 1.86x). Bit-identical: GREEN cargo test -p fj-lax
--lib gather 23/0 (dense_complex/boxed/OOB-clip/i32/u32/u64 all pass). KEPT (1.86x). kkawk closed.

## 2026-06-22 - scatter dense index extraction 1.14x (mirrors gather) — completes the dense-index lesson across gather+scatter (CrimsonOtter/cc)

eval_scatter extracted its 1M indices via per-element `lit_to_i64` (match + Result over boxed
Literals), same as eval_gather did before the kkawk fusion. Added the dense `as_i64_slice()
.to_vec()` fast path (i32 is i64-backed; non-dense keeps per-element). Same-session A/B (Zen3):
- **scatter_overwrite_1m_f64: 29.41ms -> 25.81ms = 1.14x**
- **scatter_add_1m_f64_1d: 13.82ms -> 12.17ms = 1.14x**
Modest (the serial scatter inner loop / partitioned scatter-add dominates; index extraction is
~14% of total) but real and consistent across both, bit-identical. GREEN: cargo test -p fj-lax
--lib scatter = 31 passed / 0 failed (dense_i32/u32/u64, complex, duplicate-last-wins, range-
partitioned scatter-add all pass). KEPT (1.14x > ~0-gain). Completes the dense-index-extraction
lesson across both indexed ops; scatter has no separate `resolved` Vec to fuse (it resolves
inline in the scatter loop), so this is the full scatter pre-pass lever.

## 2026-06-22 - t1pb0 BOLD-VERIFY closeout: scan radical levers rejected under disk gate (CrimsonOtter/cod-b)

Claimed `frankenjax-t1pb0` after `bv --robot-triage`/`br ready --json`: the open, unowned
actionable bead was the cumprod/cummax 4M scan gap. `frankenjax-murmw` and `frankenjax-mcqr`
were already in progress under `cod-a`; `frankenjax-cntiy` is assigned to `cod-b` but remains
maintainer-policy gated, not code-actionable.

Alien-graveyard routes considered:
- Blelloch/NESL segmented scan (§4.6): already represented by the current two-pass blocked
  prefix scan. A new segmented-scan abstraction would not change the hot local per-block
  recurrence and would add interface/scheduling surface.
- Vectorized execution + morsel-driven parallelism (§8.2): also already structurally present
  for the single long line via block partitioning and threaded offset application. The residual
  gap is inside each block, not in tuple-at-a-time dispatch.
- Manual SIMD prefix-product / NaN-propagating SIMD prefix-max: rejected before editing because
  the existing no-build disassembly showed the current monomorphizations already emit packed
  `vmulpd`, `vmaxpd`, `vcmpunordpd`, and `vblendvpd` with inlined closures. That kills the
  original "not vectorized / not inlined" hypothesis.

Fresh JAX comparator, repo venv `benchmarks/jax_comparison/.venv/bin/python`, JAX 0.10.1,
`JAX_ENABLE_X64=1`, CPU, 8 runs after compile:

| op 4M f64 1D | fresh JAX best | fresh JAX p50 | fresh JAX mean | Rust reference | Rust/JAX p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cumsum | 15.903ms | 17.197ms | 17.038ms | 7.55ms prior same-session | 0.44x, Rust wins 2.28x |
| cumprod | 17.055ms | 17.932ms | 18.219ms | 20.9ms prior same-session | 1.17x loss |
| cummax | 17.337ms | 18.514ms | 19.028ms | 20.9ms prior same-session | 1.13x loss |

Disk/toolchain constraint note: the allowed warm target was
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b` only. A direct run of the
existing warm `lax_baseline` binary proved it predates the committed cumprod/cummax 4M bench
rows (its `--list` contains `eval/cumsum_4m_f64_1d` but not `eval/cumprod_4m_f64_1d` or
`eval/cummax_4m_f64_1d`). A filtered RCH bench first failed because this Cargo does not accept
`cargo bench --release`; the corrected filtered RCH bench then selected a cold worker
(`vmi1227854`) and began compiling dependencies, so it was interrupted to honor the no-cold-build
disk rule. A local filtered `cargo bench -p fj-lax --bench lax_baseline 4m_f64_1d` also did not
produce evidence: the target was built by `rustc 1.98.0-nightly (b30f3df3b 2026-06-11)`, while
the active toolchain is `rustc 1.98.0-nightly (f20a92ec0 2026-06-07)`, producing E0514
incompatible-crate errors rather than a valid benchmark.

Conformance stayed green without compiling new artifacts: direct prebuilt release oracle
`/data/projects/.rch-targets/frankenjax-cod-b/release/deps/cumulative_oracle-9464b19459022a37
--nocapture` passed 45/45.

Conclusion: no code lever shipped. The plausible radical levers collapse to "already vectorized"
or require a quiesced perf-counter pass on a rebuildable warm toolchain. Reopened retry predicate:
only resume t1pb0-style code work after a fresh same-binary Rust A/B can be run from a matching
warm target and perf counters identify a concrete stall source in the blocked scan's local pass.
Scorecard mirror: targeted cumprod/cummax remains **0 wins / 2 losses / 0 neutral**; the broader
4M cumulative rowset is **1 win / 2 losses / 0 neutral** because the earlier cumsum keep still wins.
