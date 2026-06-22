# Negative Evidence Ledger

Canonical project ledger: `../evidence/perf/negative_evidence_ledger.md`.

## FRONTIER SCORECARD (2026-06-21, CrimsonOtter) — what still loses to JAX and why

Consolidated from this session's measurements + the per-op entries below. The contained,
measurable-on-this-host, non-`+fma`, unowned perf levers are EXHAUSTED; every remaining loss
routes to one of three gates.

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
    (sort ≤ its f64 1.25ms) still wins ≥10x in the dtype JAX users actually run. (fj-lax f32 exact
    pending — recurring rch two-stage-grep capture flake on the bench-time line.)
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
| scatter-add 1M f64 1D | **LOSS 2.74x vs JAX** after 3.07x Rust-side keep | next needs a fundamentally different safe parallel direct-write proof; unique atomic and histogram/prefix bucket-build branches regressed |
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
