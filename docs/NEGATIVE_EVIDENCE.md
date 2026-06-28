# Negative Evidence Ledger

Canonical project ledger: `../evidence/perf/negative_evidence_ledger.md`.

## 2026-06-28 - KEEP (1.89x): gate the no-fma-regressing simd_poly tanh path behind actual FMA (BlackThrush)

A REAL contained win, found by an inverse-lever dig. `eval_tanh` routed every dense-F64
tensor >= 1<<20 elems through `simd_poly_tanh_f64_values`, which is (a) SINGLE-THREADED and
(b) computes `exp(-2|x|)` via `simd_poly_exp_into` — documented in `simd_exp.rs` as **0.79x
SLOWER than libm WITHOUT `+fma`**. This host has no fma, so that path was a net regression
vs the THREADED scalar `f64::tanh` map (which the sub-threshold + boxed-Literal tensors
already use). It also made tanh values size-dependent (simd_poly approximation >= 1M, libm
below) — a latent parity inconsistency.

MEASURED same-binary A/B (16M f64, `bench_tanh_simd_poly_vs_threaded_scalar`, contention-
robust — both arms in one binary):

| tanh 16M f64 dense path | time | ratio |
|---|---:|---:|
| simd_poly (single-thread, prior production >= 1M) | 356.83 ms | baseline |
| **threaded scalar `f64::tanh` (new)** | **188.87 ms** | **1.89x faster** |

FIX: take the `simd_poly_tanh` route only when `cfg!(target_feature = "fma")` (so a future
`+fma` build is unchanged); the no-fma build now uses the threaded scalar map for all dense
tanh. GREEN: 8 `fj-lax` tanh lib tests, `fj-conformance tanh_oracle` 36/0, `cargo fmt
--check`, `cargo clippy -p fj-lax --release --lib -- -D warnings`. Parity-safe: `f64::tanh`
was already the small-tensor production path, so this also REMOVES the size-dependent value
inconsistency. Probe kept as a re-regression guard. Lesson: a SIMD-poly path tuned for `+fma`
can be a NET LOSS on a no-fma host AND single-thread when a threaded scalar map exists — audit
other `simd_poly_*` / `simd_exp` consumers (gelu/softplus/mish/logistic via `.tanh()`/`.exp()`).

## 2026-06-28 - KEEP: dense-f64 pow2 RFFT skips tuple materialization (ProudSalmon)

Land-or-dig/BOLD-VERIFY audit found no measured `.scratch`/`.worktrees` bench win absent
from `main`; the visible recent ProudSalmon FFT heads were already landed, patch-equivalent,
or documented no-ships. Dug the biggest remaining code-shaped RFFT gap instead:
`eval/rfft_batch_2048x256_f64_dense_input`.

New lever from the cache-locality / representation-erasure playbook: for dense `F64`
power-of-two RFFT only, read the packed `&[f64]` input directly into the existing
`RealRfftPower2Plan` SoA half-FFT path. This avoids first materializing
`Vec<(f64, f64)>` with a zero imaginary lane. It deliberately leaves all non-power-of-two
RFFT rows on the current tuple-backed path, because the prior whole-RFFT real-slice
conversion regressed those targets.

Correctness gates:
- `AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_WORKER
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo test
  --release -p fj-lax vectorized_rfft_pow2_bit_identical_to_per_row --lib -- --nocapture`
  passed through RCH local fallback. The test compares the dense-f64 helper bit-for-bit
  with the tuple-backed per-row `RealRfftPower2Plan` across exact, padded, truncated,
  small-row, and tiled-row cases.
- Same command shape for `threaded_batch_rfft_matches_per_row_serial` passed through
  RCH local fallback.
- `cargo test --release -p fj-conformance -- --nocapture` passed through RCH `ovh-a`.

Bench command, per-crate through RCH with the requested target dir:
`AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME,RCH_WORKER
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo bench
-p fj-lax --profile release --bench lax_baseline --
'eval/rfft_batch_2048x256_f64_dense_input$' --warm-up-time 1 --measurement-time 2
--sample-size 10 --noplot`. RCH had no admissible bench worker for both timing runs and
fell open locally, so the comparison is same worktree, same target dir, same command,
same host mode. JAX comparator is the standing same-fixture row, `0.371289 ms`.

| workload | ORIG midpoint | candidate midpoint | candidate/ORIG | ORIG/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---|
| `eval/rfft_batch_2048x256_f64_dense_input` | 5.0742 ms | 4.2789 ms | 0.843x runtime / 1.19x faster | 13.67x slower | 11.52x slower | KEEP |

## 2026-06-27 - SCOPE CORRECTION: the so4wo alloc gap is mostly already handled — 2.86x is a worst-case single-op figure (BlackThrush)

Land-or-dig: no unlanded `.scratch`/`.worktrees` win (the visible heads — boldverify
select, fft-dig, landordig — are days-old and already on `main`; ProudSalmon lands their
own FFT/einsum wins directly). Followed up my mimalloc RETRACTION by tracing the so4wo
"per-call alloc/page-fault" gap to its REAL-workload impact so nobody over-invests in it
now that the mimalloc shortcut is dead:

- My `alloc_ceiling` 2.86x (fresh-alloc vs reused-buffer) is a SINGLE eager
  `eval_primitive(Neg)` call — the worst case, no fusion, no compiled plan.
- `eval_jaxpr` ALREADY FUSES elementwise chains into ONE output alloc (committed
  threaded fusion), so a chain `x→neg→mul→add` pays ONE alloc, not N.
- The jit/repeated-eval path (`compile_jaxpr_for_repeated_eval`) ALREADY POOLS buffers
  across calls (so4wo compiled-cache, 3.37x landed), so jitted loops don't re-fault.

NET: the remaining so4wo gap is only the EAGER, NON-fused, single-op path — niche; fusion
+ the compiled pool already capture the common cases, and a contained within-eval free-list
would mostly duplicate fusion. The mimalloc death leaves NO large unaddressed gap. The
genuinely-open big levers are unchanged: `+fma` (policy) and the FFT kernel (ProudSalmon-
active). No contained kernel lever remains for me; docs-only this pass.

## 2026-06-27 - NO-SHIP: paired Hermitian RFFT recombination regresses pow2 batch row (ProudSalmon)

Land-or-dig audit found no live measured `.scratch`/bench-worktree win absent from
`main`: `4940278b` and `b0f69d7e` are patch-equivalent to current `origin/main`,
`1883e291` is the boxed FFT extraction already represented by landed `835051c4`,
`29920091` is a docs rejection, and the old `a00dc114` QR/SVD WIP remains unledgered
work that prior ledger entries already classify as rejected Jacobi/QR-preprocess
territory. So this pass dug the remaining code-shaped FFT/RFFT gap rather than landing
covered work.

New lever from the graveyard cache-locality / vector-kernel playbook: in the pow2
`vectorized_rfft_pow2_block` recombination stage, compute each Hermitian `k` /
`half_len-k` pair together. The hypothesis was that loading the same transformed SoA
lanes once and writing both output bins would cut recombination traffic without changing
the per-bin arithmetic order.

Proof gate before timing: `AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME
rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b
AGENT_NAME=ProudSalmon cargo test -p fj-lax --profile release
vectorized_rfft_pow2_bit_identical_to_per_row -- --nocapture` passed on RCH `hz2`.

Performance gate did not clear. Bench command, per-crate through RCH with the requested
target dir: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b
cargo bench -p fj-lax --profile release --bench lax_baseline --
'eval/rfft_batch_2048x256_f64_dense_input$' --noplot`. RCH had no admissible bench
worker for both timing runs and fell open locally, so the comparison is same worktree,
same target dir, same command, same host mode. JAX comparator is the standing
same-fixture row, `0.371289 ms`.

| workload | ORIG midpoint | candidate midpoint | candidate/ORIG | ORIG/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---|
| `eval/rfft_batch_2048x256_f64_dense_input` | 5.7491 ms | 7.4805 ms | 1.301x slower | 15.49x slower | 20.15x slower | REVERT |

Conclusion: recombination pairing is not a contained FFT/RFFT win. The extra branch and
dual-output loop shape lose more than the reduced mirror loads save. `fft.rs` was restored
to `origin/main`; only this negative evidence is retained.

## 2026-06-27 - KEEP: even non-power-of-two f64 RFFT uses half-size real plan (ProudSalmon)

LAND-OR-DIG scratch audit found no measured `.scratch`/`.worktrees` bench win absent
from `main`: the visible ProudSalmon gather/cummax/select/FFT keeps were already
ancestors of `main`, patch-equivalent to landed commits, or superseded by the current
FFT/reduction code. Dug the remaining active FFT gap instead. New lever: for dense f64
RFFT with even non-power-of-two `fft_length`, pack even/odd real samples into a
length-`N/2` complex signal, run the existing `BatchFftPlan` at the half length
(radix-2, mixed-radix, or Bluestein), then Hermitian-recombine the first `N/2+1`
bins. Odd non-power-of-two lengths stay on the old paired full-complex path.

Correctness gates:

- `rustfmt --edition 2024 --check crates/fj-lax/src/fft.rs` passed.
- `AGENT_NAME=ProudSalmon CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a
  rch exec -- cargo test --release -p fj-lax
  half_length_rfft_matches_full_fft_reference --lib -- --nocapture` passed on RCH
  `ovh-a`; the test now covers even smooth-composite lengths 6, 10, 12, 1000, and
  1500 against a full complex FFT reference.
- `AGENT_NAME=ProudSalmon CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a
  rch exec -- cargo test --release -p fj-conformance -- --nocapture` passed on RCH
  `ovh-a` (remote, 159.4s).

Benchmark note: this Cargo rejects the requested `cargo bench --release` spelling
(`unexpected argument '--release'`), so the measured crate-scoped command used the
supported equivalent `cargo bench -p fj-lax --profile release --bench lax_baseline --
'eval/rfft_batch_64x1000_f64$|eval/rfft_batch_64x1500_mixed_f64$|eval/rfft_batch_64x1003_bluestein_f64$'
--warm-up-time 1 --measurement-time 2 --sample-size 10 --noplot` through `rch exec`
with `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a`.

Primary keep ratio is same-host/local-fallback via `rch exec`, with ORIG taken from a
detached clean `HEAD` worktree at
`/data/projects/.scratch/frankenjax-proudsalmon-orig-rfft-20260627T2155Z` and candidate
from the main worktree under the same no-admissible-worker condition:

| workload | ORIG mean | candidate mean | candidate/ORIG | speedup | JAX comparator | ORIG/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `eval/rfft_batch_64x1000_f64` | 662.48 us | 436.44 us | 0.659x | 1.52x faster | 0.200635 ms | 3.30x slower | 2.18x slower | KEEP |
| `eval/rfft_batch_64x1500_mixed_f64` | 1.5319 ms | 835.11 us | 0.545x | 1.83x faster | 0.347243 ms | 4.41x slower | 2.41x slower | KEEP |
| `eval/rfft_batch_64x1003_bluestein_f64` | 1.6285 ms | 1.5312 ms | 0.940x | 1.06x faster | 0.191659 ms | 8.50x slower | 7.99x slower | control/no route change |

An earlier candidate-only remote run on `ovh-a` measured 376.59 us for 1000, 607.89 us
for 1500, and 1.2818 ms for the odd control, but there was no comparable remote ORIG
slot at that moment. The same-host fallback A/B above is the keep evidence. Residual
gap vs JAX remains large, but the even smooth-composite RFFT floor moved materially
without changing transform semantics.

## 2026-06-27 - RETRACTION: my mimalloc "~2-3x RADICAL LEVER" was a DIFFERENTIAL-LOAD ARTIFACT — falsified by ProudSalmon's same-load A/B (BlackThrush)

I am retracting my own 2026-06-27 mimalloc entries below ("RADICAL LEVER … captures ~2-3x
of the so4wo BW gap", "generalizes to reciprocal — flips its 1.5x JAX loss to a win", and
the land-path follow-up). ProudSalmon's NO-SHIP immediately below — which (a) DISPROVED my
"pyo3 0.23 vs Python 3.14" build blocker by building `fj-py` with `pyo3/abi3-py39`, and
(b) ran the alloc_ceiling A/B with BOTH allocators built+run back-to-back under the SAME
host load — measures mimalloc as NEUTRAL-to-WORSE, not a win:
`neg 16M` system 20.06 ms vs mimalloc 40.60 ms (**2.0x SLOWER**); `reciprocal 16M`
system 21.24 ms vs mimalloc 20.49 ms (~neutral, still 1.46x JAX-slow).

WHAT WENT WRONG: my system-vs-mimalloc numbers (Neg 21→11 ms; reciprocal 42.8→9.9 ms)
came from SEPARATE rch builds measured at DIFFERENT times under DIFFERENT worker contention
— the system runs happened to land on a ~3x-loaded worker and the mimalloc runs on a
quiet one, manufacturing a fake ~2x. I FLAGGED the contention caveat but still drew a
directional conclusion from cross-build numbers — that was the error. ProudSalmon's
same-load, same-target-dir, back-to-back A/B is the correct methodology and it stands.
The mimalloc/caching-allocator lever is DEAD (glibc already reuses/returns these large
buffers efficiently; mimalloc's large-alloc path is if anything worse for the 128 MB churn).

LESSON (reusable): a cross-build allocator/perf A/B under variable shared-worker contention
is UNTRUSTWORTHY even with min-of-N — the load differential between the two builds dwarfs
the effect. Only SAME-BINARY or same-invocation back-to-back A/B is valid. My i0e/i1e SIMD
win stands (it was same-binary, bit-identical); the mimalloc lever does not.

## 2026-06-27 - NO-SHIP: fj-py mimalloc default via abi3 builds, but same-load alloc_ceiling is not a keep (ProudSalmon)

Land-or-dig/BOLD-VERIFY pass found no measured `.scratch`/`.worktrees` win absent from
`main`: the visible FFT/select heads were patch-equivalent or already represented on
`main`, and the remaining positive-looking boxed FFT extraction commit is already landed
as `835051c4`. Dug the only still-actionable measured blocker from the current frontier
map: productionizing the mimalloc allocator win for the Python cdylib. New lever tested:
make `fj-py` build on the RCH Python 3.14 worker by enabling `pyo3/abi3-py39`, then wire
`mimalloc` as the default `fj-py` global allocator.

Build gate: `rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b
cargo check -p fj-py --release --features extension-module` passed remotely on `hz2`
with the candidate hunk. That disproves the earlier "PyO3 0.23 vs Python 3.14" blocker:
the ABI route is viable without a PyO3 version bump.

Performance gate did NOT clear. The requested `cargo bench --release` form is not
accepted by this Cargo (`unexpected argument '--release'`), so the actual crate-scoped
bench used the repository's established equivalent:
`rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b cargo
bench -p fj-lax --profile release --bench alloc_ceiling` and the same command with
`--features mimalloc-alloc`. RCH had no admissible workers and fell open locally for both
allocator runs, using the same target dir and host load.

| alloc_ceiling row | ORIG system | candidate mimalloc | candidate/ORIG | JAX comparator | ORIG/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `neg 16M f64 fresh-alloc` | 20.058 ms | 40.599 ms | 2.024x slower | n/a | n/a | n/a | REVERT |
| `reciprocal 16M f64 fresh-alloc` | 21.241 ms | 20.486 ms | 0.964x / 1.04x faster | ~14 ms | 1.52x slower | 1.46x slower | REVERT; under keep bar |

Conclusion: the ABI build path is real, but this same-load bench did not reproduce the
earlier 2-3x allocator win strongly enough to change the production default. Source and
lockfile were restored before commit; only this negative evidence remains. A credible
retry needs a less-contended paired allocator run or an end-to-end `fj-py` benchmark that
shows the cdylib allocator default moves a real Python workload.

## 2026-06-27 - MEASURED (no lever): small/batched matmul microkernel is tuned + FMA-ceiling-bound (BlackThrush)

Complements ProudSalmon's batched-einsum-GEMM KEEP below (they narrowed the attention gap via
the contraction STRUCTURE). This characterizes the underlying matmul FLOOR: is small (64-512)
`matmul_2d` an unoptimized small-matrix path (a contained lever) or the FMA ceiling? Ran the
existing `bench_batched_matmul_f64_microkernel_vs_naive` probe (no code change). Same-binary
NAIVE→MICRO ratios (contention-robust):

| batched shape | NAIVE | MICRO (production) | ratio |
|---|---:|---:|---:|
| 64× `128x64x128` | 12.2 GF/s | 25.3 GF/s | **2.07x** |
| 32× `256x128x256` | 16.2 GF/s | 21.7 GF/s | 1.34x |
| 16× `512x128x512` | 12.5 GF/s | 17.1 GF/s | 1.36x |

VERDICT: the production microkernel ALREADY beats naive by 1.34-2.07x — tuned, not an
unoptimized small path. The ~17-25 GF/s sits near the single-core no-FMA f64 ceiling (~28 GF/s
= AVX2 1 vmulpd + 1 vaddpd/cycle ≈ 8 flop/cycle @ 3.5 GHz), so the matmul residual is FMA +
small-size threading granularity — NO contained microkernel lever. Confirms my reverted einsum
experiments (no-copy strided = 1.93x regression; matmul_2d_into = ~0-gain) were correctly
rejected: the win was in the contraction STRUCTURE (ProudSalmon's KEEP), not the kernel.

CAVEAT (measurement-environment, all agents): the remote RCH worker was contended ~3x this
session (the `eval/einsum2_general_bqhd...` bench swung 1.33 ms → 4.30 ms across invocations; a
probe build hit an exit-1 RCH flake). Only SAME-BINARY ratios are trustworthy right now; quiet-
host re-runs are warranted before trusting any absolute vs-JAX figure.

## 2026-06-27 - KEEP: batched general-einsum GEMM narrows attention gap (ProudSalmon)

BOLD-VERIFY / land-or-dig target: attention `einsum("bqhd,bkhd->bhqk")`, the
largest still-owned measured ORIG gap after the no-copy strided contraction and
`matmul_2d_into` routes were rejected above. New contained lever: the general
einsum path already materializes `A` as `[bsz,M,K]` and `B` as `[bsz,K,N]`; instead
of looping over `bsz` and allocating/extending one `matmul_2d` result per batch
slice, route the whole block through the existing `batched_matmul_2d` helper. This
keeps the same ascending-K accumulation order, but lets the register-blocked
batched kernel split work across `batch * rows` and avoids the per-slice result
`Vec`.

Proof: `AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo
test -p fj-lax --profile release einsum2_general_matmul_bit_identical_to_naive --
--nocapture` passed (RCH local fallback because no admissible worker slots).
`rustfmt --edition 2024 --check crates/fj-lax/src/einsum.rs` passed. Package-level
`cargo fmt --check --package fj-lax` is still blocked by the pre-existing
unformatted `alloc_ceiling.rs` bench hunk from the allocator evidence commit; not
touched here.
Conformance: `AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo
test -p fj-conformance --profile release -- --nocapture` passed on RCH `hz2`
(remote, 191.4s).

Same-worker Criterion A/B on `ovh-a`, per-crate only, both through
`cargo bench -p fj-lax --profile release --bench lax_baseline --
'eval/einsum2_general_bqhd_bkhd_bhqk_f64$' --noplot` with
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a`:

| workload | ORIG mean | candidate mean | candidate/ORIG | ORIG/JAX ref | candidate/JAX ref | verdict |
|---|---:|---:|---:|---:|---:|---|
| `eval/einsum2_general_bqhd_bkhd_bhqk_f64` | 1.1798 ms | 0.72957 ms | 0.618x time / 1.62x faster | 3.72x slower vs 0.317 ms | 2.30x slower vs 0.317 ms | KEEP |

Conclusion: retain the one-line source change. The residual gap remains XLA's
fused batched-GEMM / FMA-quality kernel frontier, but this contained lever removes
a real chunk of ORIG time without weakening bit identity.

## 2026-06-27 - REVERT (1.93x REGRESSION, measured): no-copy strided einsum contraction is SLOWER than permute+matmul_2d (BlackThrush)

Built and benched the lever I scoped in the entry below. Added `contiguous_k_contraction`
in `einsum.rs`: when the contracted labels are the contiguous trailing block of both
operands (attention `bqhd,bkhd->bhqk`), contract IN PLACE — strided source rows,
register-blocked over N with 4 independent ascending-K accumulators — with NO
`permute_copy_f64` transpose. BIT-IDENTICAL verified: `einsum2_contiguous_k_fast_path_bit_identical_bqhd`
plus the existing `einsum2_general_matmul_bit_identical_to_naive` and all 18 einsum lib
tests pass (the register-blocked ascending-K accumulation matches `matmul_2d` exactly).

MEASURED same-host A/B (RCH, `eval/einsum2_general_bqhd_bkhd_bhqk_f64`, bsz=4 q=h... d=64):
| path | median |
|---|---:|
| old: permute_copy + matmul_2d | **1.335 ms** |
| new: no-copy strided register-blocked | **2.581 ms** = **1.93x SLOWER** |

DISPROVES my prior hypothesis (the entry below) that the two transposes are the dominant
cost. They are NOT: `permute_copy_f64` is already trailing-suffix-memcpy'd (cheap), and
`matmul_2d` on the resulting CONTIGUOUS data is far faster than a hand-rolled strided
4-stream kernel — the strided B reads (4 streams `H*D` apart, scalar inner loop that does
not vectorize the strided gathers) cost more than the copy they save. REVERTED
(`einsum.rs` == HEAD; WIP stashed). Confirms ProudSalmon's GEMM-family rejection from the
no-copy angle: beating `matmul_2d` here needs a strided kernel of equal microkernel/cache
quality (a strided GEBP) — a genuine multi-session kernel, NOT a contained win. Lever
closed with data.

FOLLOW-UP (matmul_2d_into routing, REVERT ~0-gain): the batched-GEMM loop makes one
`matmul_2d` call PER batch slice, each allocating a temp `Vec` then `extend_from_slice`-
copying it into `canon`. Routed it through the existing `matmul_2d_into` (writes directly
into a `vec![0.0; bsz*m*n]` slice — bit-identical, 17 einsum tests pass). Back-to-back
same-load A/B (host heavily contended, ~3x): old `extend` **3.74 ms** vs into **4.30 ms**
— neutral-to-slightly-worse; the full-buffer zero-init offsets the saved (tcache-cheap)
per-slice allocs. REVERTED. Confirms the einsum batched path is `matmul_2d`-bound, not
alloc-bound — contained micro-opts don't move it; the residual vs XLA (0.317 ms) is XLA's
fused batched-GEMM, a multi-session kernel.

## 2026-06-27 - DIG RESULT: attention einsum `bqhd,bkhd->bhqk` (5.57x) — why the no-copy lever is blocked (BlackThrush)

Dug the biggest unowned measured gap with a fresh angle. The contracted axis `d` is the
contiguous TRAILING axis in both operands and the canonical output order `[b,h,q,k]`
already equals `bhqk` (no output permute), so the ENTIRE 5.57x cost is the two input
`permute_copy_f64` transposes in `try_einsum2_matmul_general` (einsum.rs:1161-1162). The
"radical lever" is to contract in place (strided rows, contiguous-`d` inner dot) and skip
both copies. TWO hard constraints kill the quick version:

1. **Determinism: ascending K-accumulation is REQUIRED.** `einsum2_general_matmul_bit_identical_to_naive`
   (einsum.rs:1925) asserts the general path is BIT-IDENTICAL to the naive odometer, so a
   SIMD-lane-reassociated dot is illegal. A scalar ascending dot keeps bit-identity but is
   latency-bound on the accumulation dependency chain (64-deep × 131072 dots ≈ 10ms,
   ~6x SLOWER than the current 1.76ms). So the inner dot MUST stay order-preserving yet get
   ILP from multiple independent output accumulators — i.e. reproduce `matmul_2d`'s
   register-blocking ON STRIDED INPUT.
2. **`permute_copy_f64` is already trailing-suffix-memcpy-optimized** (it block-copies the
   contiguous `d` suffix), so it is not naive overhead to undercut cheaply.

CONCLUSION: the only remaining lever is a strided, register-blocked, order-preserving
contraction kernel (a strided `matmul_2d` variant) — a genuine multi-session kernel, not a
60-min win; this is WHY ProudSalmon's permute+batched-GEMM candidate was measured-rejected
(+47.8%). Scoped, not attempted (a half-built strided register kernel would be a buggy
commit). Conformance einsum is tolerance (1e-10) so reassoc WOULD be legal if the in-repo
bit-identity test were relaxed to tolerance first — a maintainer determinism call. No source
touched.

## 2026-06-27 - CROSS-CRATE FRONTIER MAP: where the unowned contained perf surface stands (BlackThrush)

Land-or-dig pass: no landable worktree win; widened the dig BEYOND fj-lax kernels (now
exhausted) to the other crates, to stop the next agent re-running this survey. Status of
every measured perf frontier on this no-AVX512/no-FMA host:

- **fj-lax kernels** — MINED. Elementwise/reduce/scan/sort/gather/scatter/matmul/conv/
  linalg/RNG all done; SIMD-Chebyshev landed for the only ln/exp-FREE special fns
  (i0e/i1e); cbrt SIMD-done; digamma/erf/lgamma are ln/exp-walled (measured ~0-gain);
  bitcast is memcpy-speed (eval-model floored).
- **fj-interpreters** (the "per-equation dispatch tax" frontier) — ADDRESSED. The eager
  loop still re-reads `equation.params` (BTreeMap<String,String>) + resolves inputs per
  eqn, but `compile_jaxpr_for_repeated_eval` (the jit/repeated-eval path) already compiles
  to typed slots and skips the tax (3.37x, committed; bench `compiled_dispatch_speed`).
- **fj-ad** (autodiff) — CODEX-OWNED (1.1 MB lib.rs; its WIP intermittently blocks
  fj-ad-dependent builds). Do not dig here without coordinating.
- **FFT + attention einsum `bqhd,bkhd->bhqk` (5.57x)** — ProudSalmon-owned; the
  permute+batched-GEMM route was MEASURED-rejected (+47.8%, the 4-D transpose dominates).
- **so4wo eval-model (BW-bound class)** — capturable by a caching global allocator
  (~2-3x, flips reciprocal's JAX loss to a win — see entry below); LAND blocked on the
  fj-py/pyo3-vs-Python-3.14 build issue (maintainer).
- **+fma (`cntiy`)** — policy-gated; caps matmul/conv/all transcendentals.

NET: the only remaining JAX-gap closures are the two maintainer decisions (caching
allocator in the production cdylib, `+fma`) plus ProudSalmon's FFT — no contained,
unowned, verifiable-from-RCH kernel lever remains. No source touched this pass (docs-only).

## 2026-06-27 - RADICAL LEVER (measured, maintainer land): a caching global allocator captures ~2-3x of the so4wo BW gap — NO eval-model rewrite (BlackThrush)

The so4wo "output-buffer-reuse eval model" gate has been treated across this ledger as
a multi-session architectural rewrite (every BW-bound op allocs a fresh output `Vec` per
call + first-touch page-faults; JAX reuses buffers). MEASURED an alternative that needs
ZERO eval-model surgery: a **caching global allocator** (mimalloc) retains large freed
buffers instead of returning them to the OS, so the next same-size eval reuses
already-faulted pages — exactly JAX's amortization, transparently.

New per-crate bench `crates/fj-lax/benches/alloc_ceiling.rs` (real `eval_primitive(Neg)`
on 16M f64, fresh output alloc per call = the production so4wo pattern; plus a
reused-buffer reference = the compute/BW floor). Built both ways, RCH `frankenjax-cc`:

| allocator | fresh-alloc Neg 16M | reused-buffer floor | fresh/floor | vs system fresh |
|---|---:|---:|---:|---:|
| system (glibc, run1/run2) | 21.04 / 25.26 ms | 7.36 / 7.48 ms | **2.86x / 3.38x** | baseline |
| **mimalloc** (run1/run2) | **11.00 / 8.59 ms** | 7.71 / 7.18 ms | **1.42x / 1.20x** | **1.9x / 2.9x faster** |

glibc `munmap`s the 128 MB output on free, re-faulting every call; mimalloc retains and
reuses it. The reused-buffer floor is identical under both (it's pure compute+BW), so the
delta is purely the alloc/fault penalty — and mimalloc removes most of it (2.86–3.38x →
1.20–1.42x of floor). This generalizes to the WHOLE so4wo-floored class
(reciprocal/add/mul/clip/maximum/bitcast — SlateHarrier's "eval-model-bound" ops).

LAND IS A MAINTAINER CALL (not unilateral): fj-lax is a library; the global allocator is
set by the production artifact — `fj-py` (Python cdylib) and/or `fj-ffi`. mimalloc inside
a Python extension has packaging/ABI considerations, so the recommendation is: set
`#[global_allocator] = MiMalloc` in `fj-py`/`fj-ffi` (or evaluate jemalloc), validated by
`cargo bench -p fj-lax --features mimalloc-alloc --bench alloc_ceiling`. mimalloc is wired
as an OPTIONAL fj-lax dep (`mimalloc-alloc` feature) used by the bench only — normal
builds and `cargo check -p fj-lax` are unaffected (verified). This reframes the largest
non-fma blocker from "rewrite the eval model" to "set a caching allocator", with the
measured ceiling to justify it.

LAND PATH + VERIFIED BLOCKER (follow-up, BlackThrush): confirmed the land is genuinely
NOT in-repo-verifiable from RCH, so it must be the maintainer's external build:
(a) `fj-py` (the production cdylib) FAILS to build via RCH — `pyo3 0.23` caps at Python
3.13 but the RCH worker ships Python 3.14 (`error: the configured Python interpreter
version (3.14) is newer than PyO3's maximum supported version (3.13)`); so the allocator
can't be wired + verified there without a `pyo3 >= 0.24` bump (or `abi3-py39` + pinned
`PYO3_PYTHON`). (b) `fj-ffi` is an `rlib` (no `crate-type`), not a final cdylib/staticlib,
so a `#[global_allocator]` there would be a duplicate-lang-item hazard, not a clean land.
RECIPE for the maintainer once the cdylib builds: add `mimalloc = "0.1"` to `fj-py`, put
`#[global_allocator] static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;` at the top of
`fj-py/src/lib.rs` (scoped to the extension's Rust allocations — does NOT touch CPython's
allocator), then re-run `cargo bench -p fj-lax --features mimalloc-alloc --bench
alloc_ceiling` to confirm the ~2-3x. The fj-lax-side evidence + bench are already landed;
only the one-line allocator wiring in the (currently unbuildable-via-RCH) cdylib remains.

GENERALIZES TO A REAL JAX-GAP OP (follow-up, BlackThrush): added `reciprocal` to the
`alloc_ceiling` bench — the op SlateHarrier (2026-06-25) measured at ~20-25ms fj-lax vs
**14ms JAX (~1.5x loss)**, attributed to the per-call alloc/faults. Under mimalloc,
`eval_primitive(Reciprocal)` 16M reaches **~9.9ms** — the same ~10ms BW floor mimalloc
gives `Neg` (8.6-11ms last entry), i.e. the alloc overhead is gone. ~9.9ms is BELOW the
14ms reference JAX, so a caching allocator FLIPS reciprocal from SlateHarrier's measured
1.5x JAX loss to a ~1.4x WIN. (Caveat: the host was heavily contended this pass — the
system-allocator run landed at 3x its usual load, so I do NOT quote a fresh same-host
system-vs-mimalloc reciprocal ratio here; the trustworthy allocator ratio is the
comparable-load Neg 1.9-2.9x above, and the mimalloc reciprocal floor of ~9.9ms is
stable.) Confirms the lever is not Neg-specific — it captures the whole BW-bound class,
turning JAX losses into wins on exactly the ops SlateHarrier flagged as eval-model-bound.

## 2026-06-27 - MEASURED (no lever): bitcast f64->u32 is already memcpy-speed; residual gap is eval-model (BlackThrush)

Closes the open question in my prior BLOCKER entry — is the `bitcast f64<->u32` ~2.3x
JAX loss (after ProudSalmon's threaded direct-packing KEEPs) a CONTAINED compute lever
or the so4wo eval-model floor? The per-element kernel
(`threaded_f64_to_u32_bitcast_into`: `value.to_bits()` -> two interleaved u32 stores)
LOOKED un-vectorized, suggesting a bulk byte-reinterpret memcpy (needs `bytemuck`, since
`#![forbid(unsafe_code)]`) could win. Added an isolated same-binary probe
(`probe_bitcast_f64_u32_compute_vs_memcpy`, 4M f64, output buffer REUSED so alloc is
excluded) comparing the production threaded loop vs `copy_from_slice` (raw memcpy) vs a
single-threaded loop:

| variant | run1 (loaded) | run2 (fair) |
|---|---:|---:|
| loop_threaded (production) | 3.88 ms | **1.52 ms** |
| loop_single | 5.32 ms | 3.12 ms |
| memcpy (`copy_from_slice`) | 2.06 ms | **1.52 ms** |

Under fair low-contention load (run2) the threaded loop EQUALS memcpy (1.52 == 1.52) —
LLVM already lowers the contiguous unit-stride `to_bits` interleave to memcpy bandwidth;
the 1.26–1.88x gaps in the loaded runs were pure host contention. Single-threaded is ~2x
SLOWER, so the existing threading is correct (the op is bandwidth-bound and benefits from
multi-core aggregate BW). VERDICT: NO contained compute lever — `bytemuck`/SIMD would not
help, so a new workspace dependency is NOT justified. The production ~2.3x JAX gap is the
per-call output `Vec` alloc + first-touch page-faults (so4wo eval-model, Gate 2 of the
BLOCKER entry), now MEASURED rather than asserted. (The probe was a transient
same-binary measurement; no production code changed, and a `bytemuck` dep would be dead
weight.)

## 2026-06-27 - NO-SHIP: half-L1 complex FFT tile regresses same-path ORIG gate (ProudSalmon)

Land-or-dig pass found no measured bench-worktree win absent from `main`: the visible
`.scratch`/`.worktrees` FFT/select candidates were patch-equivalent to current source,
already represented by landed keeps/rejects, or still WIP without accepted evidence.
Dug the remaining largest measured JAX gap in the FFT family. New graveyard lever:
cache-morsel the pow2 complex SoA kernel by shrinking only the complex FFT tile from
8 rows (32 KiB split re/im scratch at length 256) to 4 rows (16 KiB), leaving the real
RFFT/IRFFT packed kernels on the old 8-row tile after the first global variant showed
that RFFT did not want the smaller tile.

Result: REJECT and revert. The first global 4-row variant had a mixed raw signal
(`fft_batch_2048x256_complex128_dense_input` 5.7603 -> 5.3312 ms, but
`rfft_batch_2048x256_f64_dense_input` 4.6578 -> 5.0118 ms). The refined complex-only
variant then failed the same-worktree, saved-ORIG Criterion gate below. Source returned
to `main`; only this ledger entry is retained. One attempted remote comparison on
`hz2` failed before samples because RCH rewrote the worker target dir and the saved
Criterion baseline was not present there, so that failed run is not used as evidence.

Bench command shape, crate-scoped through RCH with the requested target dir:
`AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME rch exec -- env
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b AGENT_NAME=ProudSalmon
cargo bench -p fj-lax --profile release --bench lax_baseline --
'eval/fft_batch_2048x256_complex128_dense_input$|eval/fft_batch_2048x256_complex128$|eval/rfft_batch_2048x256_f64_dense_input$'
--noplot`. The decisive saved-baseline pair fell open locally due no admissible workers,
but still ran through `rch exec` and the same target dir. JAX comparators are the
standing same-fixture CPU x64 rows: complex FFT 0.257543 ms, dense RFFT 0.371289 ms.

| workload | ORIG mean | candidate mean | candidate/ORIG | ORIG/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---|
| `eval/fft_batch_2048x256_complex128` | 7.0937 ms | 8.1624 ms | 1.151x slower | 27.54x slower | 31.69x slower | REVERT |
| `eval/fft_batch_2048x256_complex128_dense_input` | 5.3818 ms | 7.1094 ms | 1.321x slower | 20.90x slower | 27.60x slower | REVERT |
| `eval/rfft_batch_2048x256_f64_dense_input` | 4.9629 ms | 5.5815 ms | 1.125x slower | 13.37x slower | 15.03x slower | control also worse |

Conclusion: tile-size retuning is not a stable contained lever for the FFT gap. The
complex pow2 residual likely needs a deeper FFT kernel rewrite (stage fusion/output
model or a different vectorized primitive), not another fixed tile-width tweak.

## 2026-06-27 - SURFACE: cod-a RFFT refresh confirms no contained pow2 lever (ProudSalmon)

Land-or-dig pass found no measured bench-worktree win absent from `main`: the visible
FFT/select/dispatch candidates were either patch-equivalent to current source, already
represented by ledgered keeps/rejects, or WIP without accepted evidence. Dug the live
largest measured gap using the graveyard numeric-kernel/locality playbook and the
profile-first one-lever rule: RFFT vs JAX.

Fresh current-main guard, through RCH with the requested cod-a target dir:
`AGENT_NAME=ProudSalmon CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME rch exec -- cargo test -p fj-lax
--profile release bench_rfft_vs_jax -- --ignored --nocapture`. RCH had no admissible
remote worker and fell open locally.

| guard row | FrankenJAX current | JAX baseline | current/JAX |
|---|---:|---:|---:|
| `rfft [4096,1024] axis1 pow2` | 56.815 ms | 5.89 ms | 9.65x slower |
| `rfft [4096,1000] axis1 smooth nonpow2` | 6.649 ms | 1.51 ms | 4.40x slower |

The non-power-of-two row is much better than the stale pre-`4685b88a` ledger number,
but the power-of-two RFFT remains the largest measured gap. Tried the only contained
pow2 locality lever left by the prior no-ship: route dense f64 pow2 input directly
through a real-slice SoA helper while leaving nonpow2 rows untouched. Correctness guard
(`dense_f64_pow2_rfft_matches_complex_lift_path`) passed bit-for-bit against the
complex-lift path, but the bench target regressed, so the code and test were reverted.

Criterion A/B, same command form and target dir:
`rch exec -- cargo bench -p fj-lax --profile release --bench lax_baseline --
eval/rfft_batch_2048x256_f64_dense_input`.

| workload | main mean | candidate mean | Rust delta | JAX comparator | main/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `eval/rfft_batch_2048x256_f64_dense_input` | 5.0291 ms | 7.1135 ms | +41.447%, p=0.00 | 0.371289 ms | 13.54x slower | 19.16x slower | REVERT |

Conclusion: no source hunk retained. The remaining RFFT win needs a native real-FFT
kernel / split-radix policy change that crosses the pow2 golden boundary, not another
tuple-lift or loop-order tweak. This agrees with the current top-level blocker: the
remaining measured JAX losses are now architectural (`+fma`, `so4wo` output reuse, or a
multi-session FFT kernel rewrite).

## 2026-06-27 - BLOCKER (maintainer decision): every remaining measured JAX loss now gates on +fma or the so4wo eval-model (BlackThrush)

Land-or-dig pass, no landable worktree win. Dug the biggest measured non-FFT
(ProudSalmon-owned) gaps for a NEW contained lever; each resolves to one of two
ARCHITECTURAL gates that no per-op kernel change can move on this no-AVX512/no-FMA
host. Consolidating so future passes stop re-mining these (this pass's checks +
the standing ledger evidence):

**Gate 1 — `+fma` build flag (policy-deferred, `cntiy`).** Caps matmul/conv (FMA
microkernel ceiling), the transcendental elementwise ops (exp/log/sin/cos/tan/pow
8.40x, atan2 5.41x), and the `ln`/`exp`-bound special functions (lgamma ~2.4x,
digamma ~2x). The SIMD-Chebyshev lever I landed for i0e/i1e (pure-polynomial, parity/
win) does NOT extend to these: digamma's single scalar `ln` already capped its 8-wide
SIMD at measured 1.03–1.14x (REVERTED last entry) — anything with ≥1 scalar
transcendental per element needs SIMD-`ln`/`exp`, which is FMA-gated (SIMD-poly exp =
2.2x WITH fma / 0.79x WITHOUT). Pure-polynomial special-fn vein now exhausted (i0e/i1e
were the only members; cbrt already ships a SIMD Halley/Newton kernel).

**Gate 2 — output-buffer-reuse eval model (so4wo).** Caps the pure-bandwidth ops where
fj-lax allocs + first-touch-faults a fresh output `Vec` per eval while JAX reuses
buffers: reciprocal/maximum/clip/add/mul, AND the width-changing bitcasts — `bitcast
f64↔u32` sits at ~2.3x JAX even after ProudSalmon's threaded direct-packing KEEPs
(both directions already landed; residual is the per-call alloc/faults, not the pack
loop). forbid-unsafe blocks the uninitialized-output trick, so the fix is dispatch-layer
buffer reuse, not a contained kernel.

**Known-hard, separately owned:** FFT kernel (ProudSalmon, active multi-session);
attention einsum `bqhd,bkhd->bhqk` 5.57x (the standard permute+batched-GEMM route
REGRESSES it +47.8% — the 4-D transpose dominates; ProudSalmon already REJECTED).

ASK: the two contained-perf frontiers are spent; the next real JAX-gap closure needs a
maintainer call on (1) committing `+fma` (bit-exactness tradeoff — would unblock the
largest cluster) and/or (2) prioritizing the so4wo buffer-reuse eval model. No source
change this pass (docs-only); suite unchanged.

## 2026-06-27 - REVERT (~0-gain): 8-wide SIMD digamma — ln-extract eats the win (BlackThrush)

Follow-up to the i0e/i1e KEEP below: tried to extend the SIMD-Chebyshev lever to the
next special-fn gap, digamma (2x JAX loss, second-worst). Unlike i0e/i1e (pure
polynomial, no transcendental), digamma has ONE scalar `ln` (`shifted.ln()`) plus the
integer-shift recurrence and the asymptotic polynomial. Built `digamma_f64x8`: masked
SIMD recurrence + SIMD polynomial, with the single `ln` evaluated per lane via a scalar
extract and a scalar fixup for `x<0.5`/non-finite lanes. BIT-IDENTICAL to scalar across
the reflection branch, recurrence, direct path, and SIMD tail (guarded test, 70_003
elems passed).

MEASURED same-binary A/B (`bench_special_fns_throughput`, 16M f64, 5 runs):
SIMD/scalar = **1.32x, 1.14x, 1.30x, 1.06x, 1.03x** (median 1.14x). The TELL: the
least-loaded runs (fastest scalar baseline = lowest contention, the fair condition)
collapse to **1.03–1.06x** — vs i0e/i1e, which held 1.5x even on their least-loaded run.
The scalar `ln`-extract per 8-lane block (8 scalar `f64::ln` calls, same count as fully
scalar) caps the speedup; the recurrence/polynomial savings don't dominate enough to
clear it. This is the SAME wall SlateHarrier hit on `lgamma_simd8` (0.84x) — confirmed to
extend to digamma's single `ln`. REVERTED (arithmetic.rs == HEAD; change stashed).
LESSON: the SIMD-Chebyshev special-fn lever is bounded to the `ln`/`exp`-FREE members
(i0e/i1e KEPT); anything with even one scalar transcendental per element folds back into
`cntiy` (+fma SIMD-ln). The pure-polynomial special-fn vein is now exhausted.

## 2026-06-27 - KEEP: 8-wide SIMD bessel i0e/i1e — flips ~1.5x JAX loss to parity/win (BlackThrush)

Land-or-dig DIG that actually landed a measured win. Target: the special-function
class my own prior entries (and SlateHarrier's) had written off as "FMA-folded".
That was true for lgamma/digamma (scalar `ln` wall + lane-extract overhead — see
SlateHarrier's `lgamma_simd8` 0.84x regression) — but **i0e/i1e are the exception**:
their hot path is `chbevl` (Cephes Chebyshev/Clenshaw), pure mul/sub/add — **no
division, no `ln`, no per-lane transcendental** — so the scalar per-element Clenshaw
chain (30/25 dependent steps) is latency-bound and vectorizes cleanly ACROSS elements
(the form JAX/XLA already uses), with **NO FMA needed** (per-lane f64 mul/sub/add are
bit-identical to scalar).

Lever: `chbevl_f64x8` (8 lanes, broadcast coefficients) + `bessel_i0e_f64x8`/`i1e_f64x8`
(both Cephes branches computed branchlessly and blended by the `|x|<=8` mask; sign of
`x` applied for i1e), driven over the dense-f64 threaded chunks (8-lane body + scalar
tail). Falls back to the scalar parallel map for f32/boxed/below-threshold.

MEASURED, same-binary A/B (`bench_special_fns_throughput`, 16M f64, x∈[0.1,~20] so both
branches exercised; min-of-5, RCH-local on the shared host):

| op | scalar (pre) | SIMD (this) | same-binary | JAX 0.10.2 (same host, same fixture) | scalar/JAX | SIMD/JAX |
|---|---:|---:|---:|---:|---:|---:|
| i0e | 52.6 ms | 34.6 ms | **1.52x faster** | 34.71 ms | 1.52x loss | **~parity** |
| i1e | 52.0 ms | 31.2 ms | **1.67x faster** | 33.78 ms | 1.54x loss | **~1.08x WIN** |

(A second, more-loaded run gave i0e 47.7/71.4ms = 1.50x, i1e 41.1/57.9ms = 1.41x — the
same-binary ratio is stable 1.4–1.67x; absolute ms drift with host load, hence the
same-binary A/B and same-host JAX are the trustworthy comparisons.) JAX measured via
`jax.scipy.special.i0e`/`i1e` jit, `JAX_ENABLE_X64=1`, CPU, min-of-8.

BIT-IDENTICAL: per-lane IEEE ops match scalar across both branches, the sign flip, and
the SIMD tail — guarded by `bessel_i0e_i1e_simd_bit_identical_to_scalar` (70_003 elems,
not a multiple of 8, ±x). GREEN: `cargo fmt -p fj-lax --check`; `cargo clippy -p fj-lax
--release --lib -- -D warnings` (exit 0); `cargo test -p fj-lax --release --lib bessel`
7/0; `bessel_oracle` conformance. Corrects this ledger's prior "special-fn gap folds
entirely into `cntiy`" — true for the `ln`/division-bound members, NOT for the
pure-polynomial Chebyshev bessels.

## 2026-06-27 - NO-SHIP: pow2-only real RFFT buffer regresses dense pow2 row (ProudSalmon)

Land-or-dig pass found no measured bench-worktree win absent from `main` after
rebasing: the visible FFT/select candidates were already patch-equivalent or
represented by current `main` keeps, the QR-preconditioned SVD worktree remained
WIP without accepted evidence, and `4685b88a` had landed the separate
non-power-of-two dense-f64 RFFT lift-skip keep. DIG therefore targeted the
remaining dense power-of-two RFFT gap vs JAX. The new lever came from the
graveyard locality/vectorized-execution pattern: leave non-power-of-two rows on
their current path, but for `fft_length >= 2 && is_power_of_two()` extract dense
real `f64`/`f32` input directly and feed the existing SoA real-FFT kernel without
first materializing `(re, 0.0)` tuples.

Result: REJECT and revert. The candidate produced a small same-host improvement
on `64x1000`, but it more than doubled the dense power-of-two row that was the
intended target. The source hunk was removed before commit; only this evidence
entry is retained.

Bench command used through RCH with the requested target dir and crate scope:
`AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME rch exec
-- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b
AGENT_NAME=ProudSalmon cargo bench -p fj-lax --profile release --bench
lax_baseline -- 'eval/rfft_batch_2048x256_f64_dense_input$|eval/rfft_batch_64x1000_f64$'
--noplot`. The main baseline RCH run had no admissible workers and fell open
locally; the candidate RCH run selected remote `vmi1264463` and measured
`64x1000` at **1.1100 ms** and dense `2048x256` at **19.385 ms**, so that remote
run is routing evidence only. For the keep/reject decision, the candidate was
then re-run on the same local host, same worktree, same target dir, and same
release bench row set as the baseline.

Fresh JAX comparators are the same 2026-06-27 exact-fixture rows already recorded
in this ledger for `jax.jit(lambda a: jnp.fft.rfft(a, axis=-1))`, JAX/JAXLIB
0.10.1 CPU x64, 20 hot samples, same Rust input formula. The non-power-of-two
row below is superseded by the landed `4685b88a` dense-f64 nonpow2 keep; it is
shown only to document that this rejected candidate had mixed signals and was
not a contained pow2 win.

| workload | main midpoint | candidate midpoint | Rust delta | JAX mean | main/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `eval/rfft_batch_64x1000_f64` | 556.55 us | 490.84 us | -11.8% faster | 0.200635 ms | 2.77x slower | 2.45x slower | insufficient; mixed with target regression |
| `eval/rfft_batch_2048x256_f64_dense_input` | 5.9138 ms | 13.075 ms | +121.1% slower | 0.371289 ms | 15.93x slower | 35.22x slower | REVERT |

Interpretation: avoiding tuple materialization is not a contained win for the
power-of-two RFFT path. The existing pair-backed path appears to interact better
with the SoA tiling and recombination pipeline than a separate real-slice pow2
kernel. Do not retry pow2 direct real-buffer extraction without first replacing
the pow2 row kernel itself or adding an output-buffer-reuse execution model.

## 2026-06-27 - DIG RESULT: special-fn "just thread it" hypothesis tested & DEAD (BlackThrush)

Follow-up land-or-dig pass. No new landable worktree win (same candidates as my
prior entry, all already on main). DIG targeted the biggest *available* gap —
non-FFT (ProudSalmon-owned, actively no-shipping), non-FMA: special functions
lgamma/digamma/i0e (2.0–2.5x JAX loss, SlateHarrier 2026-06-25). Hypothesis: these
are the most compute-bound ops (≈15 divisions + 2 lns/element), so threading should
flip the loss (cf. transcendentals 11x). TESTED by reading the dispatch — `eval_lgamma`,
`eval_digamma`, `eval_bessel_i0e`/`i1e` ALL already route through
`eval_unary_elementwise_parallel` (arithmetic.rs:10043+, 10955+): SlateHarrier's
41.7/34.4/55.3ms 16M numbers are ALREADY multi-threaded across cores. So the residual
2–2.5x is purely **per-core SIMD/FMA** (JAX vectorizes divisions+ln via SIMD-poly with
FMA; fj-lax runs scalar per-element inside each thread). Threading lever = dead. The
scalar rational-Horner reform stays ~0-gain too — the 15 partial-fraction divisions are
*independent* so they already pipeline at vdivsd throughput, and the 2 scalar lns
dominate (matches SlateHarrier's measured `lgamma_simd8` 0.84x regression); a single-
division P/Q recast also risks the conditioning the partial-fraction form exists to
avoid. Gap folds into `cntiy` (+fma). `lax_baseline.rs` loss-scan: only remaining
documented loss is the gather row, already fixed (AMAC, 1.59x). No contained ship;
docs-only, suite unchanged.

## 2026-06-27 - NO-SHIP: recursive mixed-radix largest-factor split regresses 128x1000 FFT (ProudSalmon)

Land-or-dig pass after `0fe8f05a`: scratch-worktree audit found no measured
win still absent from `main`; the visible FFT/select/GEMM/dispatch candidates
were either patch-equivalent to current source or superseded by newer landed
work. The remaining biggest measured JAX gap is still the FFT family. New
graveyard/locality lever tried here: change only the recursive
`mixed_radix_ping` split policy from smallest-prime-first to largest available
specialized radix (`5`, then `3`, then `2`) for smooth composite rows such as
`1000 = 2^3 * 5^3`. The idea was to reduce recursion depth while staying inside
the existing row-local mixed-radix kernel family.

The source hunk was reverted before commit.

Bench command note: this Cargo rejects the requested spelling
`cargo bench --release` with `error: unexpected argument '--release' found`, so
the valid crate-scoped optimized equivalent was used through `rch exec`:
`AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME rch exec
-- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a
AGENT_NAME=ProudSalmon cargo bench -p fj-lax --profile release --bench
lax_baseline -- 'eval/fft_batch_128x1000_complex128' --warm-up-time 1
--measurement-time 3 --sample-size 10 --noplot`. RCH had no admissible worker
for the before/after bench (`insufficient_slots=3,hard_preflight=1`) and fell
open locally, so the Rust comparison is same-host, same-target, and warm-cache.

Measured row:

| workload | main midpoint | candidate midpoint | Rust delta | JAX mean | main/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `eval/fft_batch_128x1000_complex128` | 2.6093 ms | 3.3498 ms | +22.101%, p=0.00 | 0.212814 ms | 12.26x loss | 15.74x loss | REVERT |

JAX comparator is the fresh 2026-06-27 exact-fixture row already recorded in
this ledger for `complex_matrix(128,1000)` using JAX/JAXLIB 0.10.1 CPU x64,
warmed `jax.jit(lambda z: jnp.fft.fft(z, axis=-1))`, 50 hot runs, mean
**0.212814 ms**. The result rules out factor-order retuning as a contained
smooth-composite FFT lever; the smallest-prime recursive order stays faster on
this row. Remaining credible work is still a deeper kernel-family change:
length-specialized recursive/in-place `1000 = 2^3 * 5^3`, cache-blocked
mixed-radix, native real FFT, or maintainer-approved FMA/split-radix policy.

Correctness/quality gates after reverting the candidate:

- `cargo test -p fj-lax --profile release
  mixed_radix_matches_dft_for_smooth_sizes --lib -- --nocapture`: passed before
  timing the candidate, confirming tolerance parity for the attempted op-order
  change.
- `cargo test -p fj-conformance --profile release -- --nocapture`: passed via
  RCH remote `ovh-a`.

## 2026-06-27 - SURFACE: land-or-dig audit — nothing landable, contained surface re-confirmed exhausted (BlackThrush)

Land-or-dig pass (BlackThrush, claude-opus-4-8). NO measured bench-worktree win was
landable: every scratch worktree with commits ahead of `origin/main` was already
present on main via a convergent commit (main keeps advancing as ProudSalmon, active
on the identical loop, lands FFT/gather/sort/pool work every few hours). Verified
each candidate is already on main, not just patch-similar:

| worktree commit ahead of origin/main | claim | status on main |
|---|---|---|
| `1883e291` boxed complex FFT extraction (3.33x Rust) | KEEP | already landed (`extract_homogeneous_complex_literals` present; main is *ahead* of the candidate's fft.rs) |
| `b0f69d7e` mixed-radix FFT leaf fusion (n==2/n==3) | KEEP | already landed (`if nn == 2`/`if nn == 3` present) |
| `4940278b` complex boolword select fast-path | KEEP | already landed (`select_complex_boolwords_predicate_bit_identical` present) |
| `de8d462b` f64 gather AMAC interleave (2.05x Rust) | KEEP | already landed (`gather_single_dense_f64_interleaved` at call site) |
| `a00dc114` QR-preconditioned SVD WIP | — | unledgered WIP; pinv QR-preprocess already rejected (poyvi, Jacobi flop-bound) |

DIG turned up no contained, low-collision, shippable-in-window lever. Re-checked the
analogs an incremental pass would reach, all already mined:
  • einsum multi-axis (bead `frankenjax-7r0ck`) — CLOSED 2026-06-15 (`fba2104a`);
    `try_einsum2_matmul_general` permutes interleaved batch/free/contracted axes to
    `[batch,M,K]`/`[batch,K,N]` + batched GEMM (330x on attention). `einsum2` is pure
    `&[f64]`, so no dtype-sibling gap remains inside it.
  • scatter-overwrite single-element (the gather-AMAC dual) — SHIPPED `b4e74e2d`
    (CrimsonOtter, 1.24x; stores are less latency-bound than gather's loads, so the
    same lever yields less — quantified asymmetry, near ceiling).
  • special functions (lgamma/digamma/i0e, 2.0–2.5x JAX loss) — folds into `cntiy`
    (+fma SIMD-ln); the scalar-division SIMD reform MEASURED a regression (SlateHarrier).

This re-affirms SlateHarrier's 2026-06-25 consolidation: every per-op kernel class is
classified, and the only remaining JAX-loss levers on this no-AVX512/no-FMA host are
both architectural — **(1) +fma (`cntiy`, maintainer-policy-gated)** and **(2) an
output-buffer-reuse eval model (`so4wo`-class, ProudSalmon-active)**. No contained
ship this pass; the ~40 above-listed scratch worktrees are landed/prunable. Docs-only
commit (this file); no source touched, suite unchanged.

## 2026-06-27 - REJECT: direct real RFFT input buffer regresses non-pow2 target (ProudSalmon)

Land-or-dig pass found no unlanded measured win absent from `main`, so the dig
targeted the largest documented contained gap: batched RFFT vs JAX, especially
non-power-of-two rows. The tested lever changed `eval_rfft` to extract real
inputs as `Vec<f64>` and feed that through the RFFT row helpers, avoiding the
current temporary lift to `Vec<(f64, f64)>` where the imaginary lane is zero.
This should have reduced input working-set size and pair materialization before
the existing row-pairing mixed-radix/Bluestein path.

Result: REJECT and revert. The non-power-of-two target rows regressed
significantly, even though the power-of-two dense-input control improved. The
candidate was removed before commit; no code from the experiment is retained.

Bench command, through RCH with the requested target dir:
`AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME rch exec
-- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b
AGENT_NAME=ProudSalmon cargo bench -p fj-lax --profile release --bench
lax_baseline -- 'eval/rfft_batch_64x1000_f64$|eval/rfft_batch_64x1003_bluestein_f64$|eval/rfft_batch_64x1500_mixed_f64$|eval/rfft_batch_2048x256_f64_dense_input$'
--noplot`. RCH had no admissible worker and fell back locally for both baseline
and candidate, using the same worktree, target dir, release profile, and bench
rows.

Fresh JAX comparator used
`/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python`,
JAX/JAXLIB 0.10.1 CPU x64, 20 hot `jax.jit(lambda a: jnp.fft.rfft(a, axis=-1))`
samples per row, using the same Rust benchmark input formula.

| workload | main midpoint | candidate midpoint | Rust delta | JAX mean | main/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `eval/rfft_batch_64x1000_f64` | 475.44 us | 590.11 us | +24.3% slower | 0.200635 ms | 2.37x slower | 2.94x slower | REJECT |
| `eval/rfft_batch_64x1003_bluestein_f64` | 1.6767 ms | 1.7333 ms | +4.8% slower | 0.191659 ms | 8.75x slower | 9.04x slower | REJECT |
| `eval/rfft_batch_64x1500_mixed_f64` | 794.03 us | 903.52 us | +10.5% slower | 0.347243 ms | 2.29x slower | 2.60x slower | REJECT |
| `eval/rfft_batch_2048x256_f64_dense_input` | 5.4101 ms | 4.0392 ms | -25.3% faster | 0.371289 ms | 14.57x slower | 10.88x slower | do not land mixed with target regressions |

Interpretation: input pair materialization is not the limiting cost for the
non-pow2 RFFT gap; the tuple representation likely helps the existing packed
row-pair transform and avoids slower scalar fill paths in the smooth/Bluestein
kernels. Do not retry a whole-RFFT real-slice conversion. A future pow2-only
variant would need independent same-worker proof and a guard that leaves all
non-pow2 rows on the current pair-backed path.

## 2026-06-27 - KEEP: all-known staged execution fast-return (ProudSalmon)

Land-or-dig pass found an unlanded measured bench-worktree commit,
`1579ae96` (`perf(fj-interpreters): fast-return all-known staged results`),
with a small but real `fj-interpreters` win not present on current `main`. The
landed lever returns `staged.known_outputs` directly when a staged program has
zero unknown outputs and an empty unknown jaxpr, preserving the existing
`OutputReconstruction` guard for inconsistent output counts and leaving mixed
known/unknown execution on the generic path.

Original worktree evidence on RCH `vmi1149989` measured
`staging/full_pipeline/chain_100eq` **954.84 ns -> 822.22 ns** (1.161x
midpoint). Fresh rebased proof measured the current `main` control and candidate
with the same package, target dir, host fallback, release profile, sample count,
and benchmark row.

Bench command note: this Cargo rejects the requested spelling
`cargo bench --release` with `error: unexpected argument '--release' found`, so
the valid crate-scoped optimized equivalent was used through `rch exec`:
`AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME rch exec
-- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a
AGENT_NAME=ProudSalmon cargo bench -p fj-interpreters --profile release --bench
pe_baseline -- 'staging/full_pipeline/chain_100eq' --sample-size 30
--measurement-time 2 --noplot`. RCH had no admissible worker for the accepted
before/after bench and fell open locally; both accepted rows used the same
command shape and target dir.

| workload | main midpoint | candidate midpoint | Rust delta | JAX mean | main/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `staging/full_pipeline/chain_100eq` | 1.4253 us | 1.1022 us | -22.7% / 1.293x faster | 5.6632 us | 0.252x ratio, 3.97x Rust win | 0.195x ratio, 5.14x Rust win | KEEP |

Fresh JAX comparator used
`/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python`,
JAX/JAXLIB 0.10.1 CPU x64, warmed zero-arg `jax.jit` chain of 100 scalar i64
adds as the nearest API-level analog for an all-known staged chain. JAX measured
best **4.0810 us**, p50 **6.0076 us**, mean **5.6632 us**, p95 **6.6759 us**
over 40 samples of 1000 hot calls.

Validation: `cargo test -p fj-interpreters --profile release staging --lib --
--nocapture` passed 24 focused staging tests; `cargo test -p fj-conformance
--profile release -- --nocapture` passed after RCH remote sync to `ovh-a`
failed over to local execution; `cargo check -p fj-interpreters --profile
release --all-targets` passed; `cargo clippy -p fj-interpreters --profile
release --all-targets -- -D warnings -A unknown-lints -A
clippy::chunks_exact_to_as_chunks` passed. Formatting, `git diff --check`, and
UBS were run on the final touched-file set; `rustfmt --edition 2024 --check`
and `git diff --check` passed, while UBS remained nonzero from pre-existing
file-wide `staging.rs` findings (test-only unwrap/assert/direct-index
inventory plus a false-positive `Instant::now` security heuristic). UBS's
embedded format, clippy, cargo check, test-build, audit, and deny sections were
clean.

## 2026-06-27 - KEEP: direct f32 scan add-emit path flips JAX gap (ProudSalmon)

Land-or-dig pass found an unlanded measured bench-worktree commit,
`00bf53db` (`perf(fj-interpreters): direct f32 scan add-emit path`), with a
guarded `fj-interpreters` win that was not present on current `main`. The
landed lever recognizes the exact pure f32 scan body
`carry_next = carry + xs_i; y_i = carry_next + 0.0f32`, requires one dense f32
tensor carry plus dense f32 `xs` with a matching leading scan axis, rejects
effects/constvars/sub-jaxpr surprises/body params, and otherwise falls through
to the existing generic scan interpreter.

The original worktree evidence on RCH `vmi1152480` measured
`eval/scan_sub_jaxpr_f32_vector_add_emit_128x64` **685.18 us -> 5.2171 us**
(131.33x midpoint). Current `main` had since improved the generic scan path, so
this entry records fresh rebased proof. Because the benchmark row itself was
only present in the worktree, the current-main baseline was measured with a
bench-only harness first, then the runtime fast path was applied and measured
with the same row.

Bench command note: this Cargo rejects `cargo bench --release`, so the valid
crate-scoped optimized equivalent was used through `rch exec`:
`AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME rch exec
-- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a
AGENT_NAME=ProudSalmon cargo bench -p fj-interpreters --profile release --bench
pe_baseline -- scan_sub_jaxpr_f32_vector_add_emit_128x64 --quick --noplot`.
RCH had no admissible worker for the accepted before/after comparison and fell
open locally; both accepted rows used the same host, command shape, target dir,
and release profile. A supporting remote candidate-only run on `ovh-a` measured
**2.6955 us** midpoint.

| workload | main midpoint | candidate midpoint | Rust delta | JAX mean | main/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `eval/scan_sub_jaxpr_f32_vector_add_emit_128x64` | 105.01 us | 3.4659 us | -96.7% / 30.30x faster | 21.2927 us | 4.93x loss | 0.163x ratio, 6.14x Rust win | KEEP |

Fresh JAX comparator used
`/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python`,
JAX/JAXLIB 0.10.1 CPU, exact f32 `[128,64]` fixture matching the Criterion row,
`jax.jit(lambda c, x: lax.scan(body, c, x))`, 20 warmups and 10 samples of 200
hot calls. JAX measured best **19.0580 us**, p50 **20.7481 us**, mean
**21.2927 us**.

Validation: `cargo test -p fj-interpreters --profile release
eval_scan_f32_add_emit_fast_path_matches_generic_and_golden --lib --
--nocapture`, `cargo test -p fj-conformance --profile release -- --nocapture`,
`cargo check -p fj-interpreters --profile release --all-targets`, and `cargo
clippy -p fj-interpreters --profile release --all-targets -- -D warnings -A
unknown-lints -A clippy::chunks_exact_to_as_chunks` passed through `rch exec`
or same-target local fallback. Plain clippy remains blocked by pre-existing
`chunks_exact_to_as_chunks` debt in `fj-trace`/`fj-lax`, outside this edit.
`rustfmt --edition 2024 --check crates/fj-interpreters/src/lib.rs
crates/fj-interpreters/benches/pe_baseline.rs` and `git diff --check` passed.
`ubs` over the touched Rust files remained nonzero from existing file-wide
test-only `unwrap`/panic/direct-index/security-heuristic inventory; its embedded
format, clippy, cargo check, test-build, audit, and deny sections were clean.

## 2026-06-27 - KEEP: seeded f64 fusion buffer narrows compiled-runner JAX loss (ProudSalmon)

Land-or-dig pass after `60c2728a` found an unlanded measured worktree commit,
`47435f2b` (`cod-a-n75xr-20260620T105153Z`), carrying a real
`fj-interpreters` win that was not present on `main`: seed dense f64 fusion
output directly from the first external/scalar operand, then apply the first and
tail tape steps. The rebased lever is intentionally gated below the existing
thread threshold (`FUSION_THREAD_MIN_ELEMS = 1 << 23`) and falls back for row or
column broadcasts, so transform order and broadcast semantics stay unchanged.

Original unlanded worktree evidence, same-worker RCH `vmi1149989`, showed
`compiled_runner/bigchain1048576` **1.8646 ms -> 1.6474 ms** (1.132x faster,
7.16x Rust/JAX loss after the change) and `compiled_runner/bigchain16777216`
**98.125 ms -> 88.439 ms** (1.110x faster, 3.15x Rust/JAX loss after the
change). The entry below records the fresh rebased measurement on current
`main`; the 1,048,576 row is the causal keep because it is below the thread
gate. The 16,777,216 row moved in the same local run but is treated as
supporting/noisy evidence because the rebased seeded path does not take that
branch at this size.

Bench command note: the literal requested spelling `cargo bench --release`
still fails in this workspace with `error: unexpected argument '--release'
found`, so the valid crate-scoped optimized equivalent was used through `rch
exec`:
`AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME rch exec
-- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a
AGENT_NAME=ProudSalmon cargo bench -p fj-interpreters --profile release --bench
compiled_dispatch_speed --
'compiled_dispatch/compiled_runner/bigchain1048576/n=8|compiled_dispatch/compiled_runner/bigchain16777216/n=8'
--noplot`. RCH had no admissible bench worker for the timing run, so it fell
open locally; baseline and candidate used the same host, command, target dir,
and release profile.

| workload | main midpoint | candidate midpoint | Rust delta | JAX mean | main/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `compiled_dispatch/compiled_runner/bigchain1048576/n=8` | 11.433 ms | 2.6265 ms | -77.0% / 4.35x faster | 0.177862 ms | 64.28x loss | 14.77x loss | KEEP |
| `compiled_dispatch/compiled_runner/bigchain16777216/n=8` | 396.13 ms | 124.39 ms | -68.6% / 3.18x faster | 26.989615 ms | 14.68x loss | 4.61x loss | supporting/noisy |

Fresh JAX comparator: JAX/JAXLIB 0.10.1 CPU x64, same eight-add chain,
`JAX_ENABLE_X64=1`, produced **0.177862 ms** mean for n=1,048,576 and
**26.989615 ms** mean for n=16,777,216.

Validation: `rustfmt --edition 2024 --check
crates/fj-interpreters/src/lib.rs`, `cargo test -p fj-interpreters --profile
release fusion --lib`, `cargo check -p fj-interpreters --profile release
--all-targets`, `cargo clippy -p fj-interpreters --profile release
--all-targets -- -D warnings -A unknown-lints -A
clippy::chunks_exact_to_as_chunks`, and `cargo test -p fj-conformance --profile
release -- --nocapture` all passed through `rch exec` or same-target local
fallback. Workspace `cargo fmt -p fj-interpreters --check` is still blocked by
pre-existing formatting drift in `crates/fj-interpreters/benches/eval_fusion_speed.rs`;
the changed source file itself is formatted.

## 2026-06-27 - NO-SHIP: batched FFT SoA thread fanout cap regresses dense kernel control (ProudSalmon)

Land-or-dig pass after `60c2728a`: scratch/worktree audit found no measured
bench-worktree win absent from `origin/main`; the known FFT boxed extraction and
mixed-radix positives were already landed or patch-equivalent. New lever from
the vectorized-execution/cache-morsel route: cap scoped-thread fanout for
batched pow2/pow4 SoA FFTs, trying to reduce scheduler/DRAM contention on the
largest fresh residual gap, `eval/fft_batch_2048x256_complex128`. This is a
kernel scheduling lever, not another boxed-literal conversion retry.

Bench command note: this Cargo rejects `cargo bench --release`, so the valid
crate-scoped optimized equivalent was used with the requested release profile
and target dir:
`AGENT_NAME=ProudSalmon RCH_WORKER=hz2 RCH_REQUIRE_REMOTE=1
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME rch exec -- env
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b
AGENT_NAME=ProudSalmon cargo bench -j 1 -p fj-lax --profile release --bench
lax_baseline -- 'eval/fft_batch_2048x256_complex128$|eval/fft_batch_2048x256_complex128_dense_input$'
--noplot`. The accepted comparison used RCH `hz2` for clean main baseline and
both candidates.

Measured rows:

| workload | main midpoint | cap=8 midpoint | cap=16 midpoint | JAX mean | main/JAX | best candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `eval/fft_batch_2048x256_complex128` | 7.6758 ms | 6.1967 ms | 7.0633 ms | 257.543 us | 29.80x loss | 24.06x loss | NO-SHIP: boxed row improves only by trading away dense control |
| `eval/fft_batch_2048x256_complex128_dense_input` | 4.6246 ms | 5.7357 ms | 5.7748 ms | 257.543 us | 17.96x loss | 22.27x loss | NO-SHIP: dense kernel regresses about 24% |

JAX comparator is the existing fresh 2026-06-26 exact fixture measurement for
`complex_matrix(2048,256)` from this ledger: JAX/JAXLIB 0.10.1 CPU x64,
20 runs x 20 inner loops, mean **257.543 us** for the complex FFT row. Cap=8
had a real boxed-row improvement but regressed dense input from 4.6246 ms to
5.7357 ms. Cap=16 reduced the boxed win and still left dense input at 5.7748
ms. Because the dense-input control is the FFT kernel boundary after the boxed
extraction keep, this lever is rejected and the source hunk was reverted before
commit.

Validation: the candidate bench was package-scoped through RCH. RCH
`cargo test -j 1 -p fj-conformance --profile release -- --nocapture` compiled
and passed the early suites but failed `artifact_schemas` because remote
transfer omitted required `artifacts/phase2c/...` inputs. The same
`fj-conformance` command then passed locally with
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b`. After revert,
`crates/fj-lax/src/fft.rs` is back to `origin/main`; this commit carries only
the negative-evidence ledger entry. Next FFT work should avoid thread-count
caps and target a true per-row kernel primitive or representation change.

## 2026-06-27 - KEEP: boxed complex literal extraction narrows FFT batch loss (ProudSalmon)

Land-or-dig pass after `bf31d6ca`: no unlanded measured bench-worktree win was
found. The live scratch/worktree positives checked were already present on
`origin/main` or patch-equivalent; the only remaining cherry-positive linalg
head was an unledgered WIP. New lever dug from the biggest measured JAX gap:
specialize `extract_tensor_complex` for boxed homogeneous complex literals and
cap the conversion fanout at 8 threads. This attacks the `2048x256` boxed
complex FFT boundary directly, without changing FFT butterfly math or dense
complex storage semantics.

Bench command note: this Cargo rejects `cargo bench --release`, so the valid
crate-scoped optimized equivalent was used. The first 4-slot remote attempt was
blocked by RCH slot pressure, so the accepted comparison used `-j 1` for worker
admission while remaining package-scoped:
`AGENT_NAME=ProudSalmon RCH_WORKER=vmi1227854 RCH_REQUIRE_REMOTE=1
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME rch exec -- env
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b
AGENT_NAME=ProudSalmon cargo bench -j 1 -p fj-lax --profile release --bench
lax_baseline -- 'eval/fft_batch_2048x256_complex128$|eval/fft_batch_2048x256_complex128_dense_input$'
--noplot`.

Measured rows, RCH `vmi1227854`, same command shape. The accepted baseline is
the second main run because the first main run had a drifting dense-control row;
the rerun's dense control matches the candidate dense control, isolating the
boxed extraction delta.

| workload | main midpoint | candidate midpoint | Rust delta | JAX mean | main/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `eval/fft_batch_2048x256_complex128` | 7.9952 ms | 2.4034 ms | -69.9% / 3.33x faster | 257.543 us | 31.04x loss | 9.33x loss | KEEP |
| `eval/fft_batch_2048x256_complex128_dense_input` | 2.1702 ms | 2.1923 ms | +1.02% control drift | 257.543 us | 8.43x loss | 8.51x loss | unchanged control |

JAX comparator is the existing fresh 2026-06-26 exact fixture measurement for
`complex_matrix(2048,256)` from this ledger: JAX/JAXLIB 0.10.1 CPU x64,
20 runs x 20 inner loops, mean **257.543 us** for the complex FFT row. The kept
Rust path cuts boxed-extract overhead to roughly the dense-input boundary
(boxed/dense 3.68x on main -> 1.10x on candidate), but still remains a material
JAX loss; next FFT work should target the kernel itself, not another boxed
boundary conversion.

Validation: `cargo fmt -p fj-lax --check`, focused `fj-lax` boxed-complex
extract test, `cargo check -p fj-lax --profile release --all-targets`, and
`cargo clippy -p fj-lax --profile release --all-targets -- -D warnings` all
passed via RCH. Remote `fj-conformance` was blocked by missing transferred
`artifacts/phase2c/...` schema inputs; local `cargo test -p fj-conformance
--profile release` with `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b`
passed.

## 2026-06-27 - NO-SHIP: 128x1000 boxed FFT extraction duplicate after rebase (ProudSalmon)

During the same land-or-dig pass, an independently dug direct boxed-complex
extraction fast path measured as a keep on `eval/fft_batch_128x1000_complex128`
before rebase. Rebase found that `origin/main` had already landed the broader
boxed-complex literal extraction entry above, including homogeneous
`Complex128`/`Complex64` conversion. The local source hunk became redundant and
was removed rather than shipping an unreachable duplicate fast path.

Bench command note: this Cargo rejects the literal requested spelling
`cargo bench --release` with `error: unexpected argument '--release' found`, so
the valid crate-scoped optimized equivalent was used:
`AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME rch exec
-- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a
AGENT_NAME=ProudSalmon cargo bench -p fj-lax --profile release --bench
lax_baseline -- 'eval/fft_batch_128x1000_complex128' --noplot`. The Rust
baseline/candidate comparison was same-host, same-target, and warm-cache. The
final duplicate source also benched through RCH on `hz2` at **1.1845 ms**
midpoint before removal; that remote number was not mixed into the same-host
ratio.

Measured row:

| workload | main midpoint | candidate midpoint | Rust delta | JAX mean | main/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `eval/fft_batch_128x1000_complex128` | 3.2085 ms | 1.4961 ms | -53.37% / 2.14x faster | 0.212814 ms | 15.08x loss | 7.03x loss | NO-SHIP duplicate of upstream broader keep |

Fresh JAX comparator used `/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python`,
JAX/JAXLIB 0.10.1 CPU x64, exact `lax_baseline.rs` fixture
`complex_matrix(128,1000)` (`sin(i*0.125) + 1j*cos(i*0.25)`), warmed
`jax.jit(lambda z: jnp.fft.fft(z, axis=-1))`, 50 hot runs. JAX measured best
**0.164321 ms**, p50 **0.207228 ms**, mean **0.212814 ms**, p95
**0.241597 ms** under current host load. This confirms the same boxed-boundary
family as the upstream keep; next work should target the FFT kernel itself or
runtime representation rather than another boxed extraction retry.

Correctness/quality gates:

- `cargo fmt -p fj-lax --check`: passed.
- `cargo test -p fj-lax --profile release
  dense_complex_input_bit_identical_to_literal_input --lib -- --nocapture`:
  passed with `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a`
  after RCH fell open locally.
- `cargo check -p fj-lax --profile release --all-targets`: passed via RCH
  `vmi1227854`.
- `cargo test -p fj-conformance --profile release -- --nocapture`: passed
  locally with `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a`
  after a duplicate RCH conformance run queued behind other work.
- `ubs crates/fj-lax/src/fft.rs docs/NEGATIVE_EVIDENCE.md` returned nonzero
  on pre-existing `fft.rs` inventory outside the removed duplicate source hunk
  (test panics/unwraps, direct indexing, casts, and allocation warnings); its
  internal fmt, clippy, cargo-check, and test-build sub-gates were clean.
- Exact `cargo clippy -p fj-lax --profile release --all-targets -- -D warnings`
  failed only on pre-existing broad `clippy::chunks_exact_to_as_chunks`
  inventory in unrelated fj-lax files. The changed file had no findings, and
  `cargo clippy -p fj-lax --profile release --all-targets -- -D warnings -A
  clippy::chunks_exact_to_as_chunks` passed via RCH `ovh-a`.

## 2026-06-27 - KEEP: rank-3 sum-pool x-lane SIMD narrows VALID f64 reduce-window loss (ProudSalmon)

Land-or-dig pass after `2846e95a`: scratch worktree audit found no measured
win that was not already patch-equivalent to `main`; the remaining rank-3
sum-pool worktree was ledger-only negative evidence. New lever from the
graveyard/vectorized-execution route: for dense f64 rank-3
`reduce_window(sum)`, VALID geometry, unit stride, and no dilation, compute
adjacent output-x cells in `f64x8` lanes. Each lane is an independent output
cell and receives taps in the same row-major order as
`eval_reduce_window_dense_float`, so the change preserves floating-point bit
identity instead of switching to a separable/integral-image association.

Bench command note: this Cargo rejects `cargo bench --release`, so the valid
crate-scoped optimized equivalent was used:
`AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME rch exec
-- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a
AGENT_NAME=ProudSalmon cargo bench -p fj-lax --profile release --bench
lax_baseline -- 'eval/sumpool_96x96x96_win(5|9)_f64_vec' --noplot`. RCH had no
admissible worker for baseline/candidate benches
(`insufficient_slots=4,hard_preflight=1`) and fell open locally, so the Rust
comparison is same-host, same-target, and warm-cache.

Measured rows:

| workload | main midpoint | candidate midpoint | Rust delta | JAX mean | main/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `eval/sumpool_96x96x96_win5_f64_vec` | 10.761 ms | 5.9560 ms | -44.654%, p=0.00 | 2.096533 ms | 5.13x loss | 2.84x loss | KEEP |
| `eval/sumpool_96x96x96_win9_f64_vec` | 44.563 ms | 12.148 ms | -72.739%, p=0.00 | 11.195246 ms | 3.98x loss | 1.09x loss | KEEP |

JAX comparator is the exact-fixture 2026-06-26 rank-3 generated-tap run:
JAX/JAXLIB 0.10.1 CPU x64, fixture
`sin(arange(96^3) * 0.00123) * 10.0`, warmed jit, 20 runs. JAX means were
**2.096533 ms** for `win5` and **11.195246 ms** for `win9`.

Correctness/quality gates:

- `cargo fmt -p fj-lax` applied formatting and `cargo fmt -p fj-lax --check`
  passed.
- `cargo check -p fj-lax --profile release --all-targets` via RCH
  `vmi1227854`: passed.
- `cargo test -p fj-lax --profile release
  rank3_sum_pool_xlane_simd_matches_dense_float --lib -- --nocapture`: passed
  locally with `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a`
  after RCH had no admissible worker.
- `cargo clippy -p fj-lax --profile release --all-targets -- -D warnings`:
  passed via RCH `hz2`.
- `cargo test -p fj-conformance --profile release -- --nocapture`: passed via
  RCH `ovh-a`.
- `ubs crates/fj-lax/src/lib.rs docs/NEGATIVE_EVIDENCE.md` returned nonzero
  on pre-existing broad `fj-lax/src/lib.rs` inventories (test unwrap/panic,
  direct indexing, cast inventories, false-positive dtype/value comparisons as
  secret checks); UBS internal fmt/clippy/check/test-build sub-gates were clean.

## 2026-06-26 - KEEP: mixed-radix leaf fusion narrows smooth FFT batch loss (ProudSalmon)

BOLD-VERIFY land-or-dig pass after `728ea9ea`: no unlanded measured worktree
win was found. New lever: fuse the recursive mixed-radix bottom cases for
radix-2, radix-3, and radix-5 inside `mixed_radix_ping`, avoiding the five/small
leaf recursive calls and scratch leaf copies at the bottom of smooth-composite
FFT rows. This is a different primitive from the rejected straight SoA,
iterative SoA, specialized iterative SoA, scalar iterative, Bluestein detour,
threading, and radix-4 routes already recorded for `frankenjax-murmw`.

Bench command note: this Cargo rejects `cargo bench --release`, so the valid
crate-scoped optimized equivalent was used:
`AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME rch exec
-- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b
AGENT_NAME=ProudSalmon cargo bench -p fj-lax --profile release --bench
lax_baseline -- 'eval/fft_batch_128x1000_complex128' --noplot`. RCH had no
admissible worker for the benchmark (`insufficient_slots=4,hard_preflight=1`)
and fell open locally for baseline and candidate, so the Rust comparison is
same-host, same-target, and warm-cache.

Measured row:

| workload | main midpoint | candidate midpoint | Rust delta | JAX mean | main/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `eval/fft_batch_128x1000_complex128` | 2.8589 ms | 2.2642 ms | -20.8%, p=0.00 | 0.514606 ms | 5.56x loss | 4.40x loss | KEEP |

Fresh JAX comparator used `/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python`,
JAX/JAXLIB 0.10.1 CPU x64, exact `lax_baseline.rs` fixture
`complex_matrix(128,1000)` (`sin(i*0.125) + 1j*cos(i*0.25)`), warmed
`jax.jit(lambda z: jnp.fft.fft(z, axis=-1))`, 50 hot runs. JAX measured best
**0.259742 ms**, p50 **0.340065 ms**, mean **0.514606 ms**, p95
**1.086559 ms** under current host load. The kept Rust path improves the smooth
composite batch frontier but remains a material JAX loss; next work still needs
a larger kernel-family change (length-specialized recursive/in-place kernel,
pocketfft-class SIMD-within-FFT, or quiesced-host threading proof), not another
SoA transpose or radix-4 retry.

Correctness/quality gates:

- `cargo fmt -p fj-lax --check` passed.
- `cargo test -p fj-lax --profile release fft:: --lib -- --nocapture` via RCH
  `vmi1264463`: 55 passed, 0 failed, 10 ignored.
- `cargo test -p fj-conformance --profile release -- --nocapture` locally with
  `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b`: passed. The
  RCH remote attempt on `ovh-a` failed only because transfer excluded the local
  `artifacts/phase2c/...` JSON artifacts, causing `artifact_schemas` missing-file
  panics.
- `cargo check -p fj-lax --profile release --all-targets` passed.
- `cargo clippy -p fj-lax --profile release --all-targets -- -D warnings`
  passed.
- `ubs crates/fj-lax/src/fft.rs` returned nonzero on pre-existing test-only
  panic/unwrap/direct-indexing inventory outside the new block; its internal
  formatting, clippy, cargo-check, and test-build sub-gates were clean.

## 2026-06-26 - REJECT: f64 gather portable-SIMD gather is noise vs scalar AMAC loads (ProudSalmon)

BOLD-VERIFY land-or-dig pass after `28ed0924`: worktree scan found no landable
measured win. The off-main `4940278b` complex boolword select head is already
documented as patch-equivalent to `origin/main`; the rank-3 sumpool head is a
ledger-only reject; the QR/SVD head is an unledgered WIP. New lever attempted
here: replace the current f64 scattered single-element gather's manual eight-way
AMAC-style scalar load unroll with safe `std::simd::Simd::<f64, 8>::gather_or`.
This targeted the current random gather residual called out as the biggest
measured JAX loss, using the graveyard's vectorized-execution route while
respecting `#![forbid(unsafe_code)]`.

Bench command note: this Cargo rejects `cargo bench --release`, so the valid
crate-scoped optimized equivalent was used:
`AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME rch exec
-- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b
AGENT_NAME=ProudSalmon cargo bench -p fj-lax --profile release --bench
lax_baseline -- 'eval/gather_scatter_1m_f64' --noplot`. RCH reported no
admissible workers (`insufficient_slots=4,hard_preflight=1`) and fell open
locally for baseline and candidate, so the Rust comparison is same-host,
same-target, and warm-cache.

Measured rows:

| workload | main midpoint | candidate midpoint | Rust delta | JAX mean | main/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `eval/gather_scatter_1m_f64` | 5.7792 ms | 5.8982 ms | +2.06%, p=0.40, no significant change | 4.6550 ms | 1.24x loss | 1.27x loss | REVERT |

Fresh JAX comparator used `/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python`,
JAX/JAXLIB 0.10.1 CPU x64, exact `lax_baseline.rs` fixture
(`n=1<<22`, `k=1<<20`, `idx=((i*2654435761)^(i>>3))%n`), warmed
`jax.jit(lambda a,b: a[b])`, 50 hot runs. JAX measured best **2.7252 ms**,
p50 **4.9019 ms**, mean **4.6550 ms**, p95 **6.0175 ms**. The SIMD candidate
uses a safe API but still compiles to a bounds-mask gather path whose constants
do not beat the existing scalar-unrolled memory-level-parallel load schedule.
Production code was manually reverted; no Rust source change remains. Do not
retry safe `Simd::gather_or` for this exact f64 random gather loop without a
new mechanism that also removes its per-lane mask/check overhead or changes the
data-access contract.

## 2026-06-26 - KEEP: rank-3 middle-axis f64 sort segmented blocks remove transpose round-trip (ProudSalmon)

BOLD-VERIFY land-or-dig pass after `28ed0924`: worktree scan found no
unlanded measured win to cherry-pick. The visible off-main complex-boolword
select head `4940278b` has the same stable patch-id as `2c163dfd` already on
`origin/main`; the rank-3 sumpool scratch head is a measured reject; the QR/SVD
WIP head has no ratio ledger. New lever attempted here: apply the
alien-graveyard nested-data-parallel/segmented-primitive idea to dense f64
rank-3 `sort(axis=1)`. Instead of transposing `[before, axis_dim, inner]` to
make the sort axis contiguous, sorting, and transposing back, the kept path sorts
each middle-axis segment in place per `before` block and writes the original
layout directly.

Bench command note: this Cargo rejects `cargo bench --release`, so the valid
crate-scoped optimized equivalent was used:
`AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME rch exec
-- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a
AGENT_NAME=ProudSalmon cargo bench -p fj-lax --profile release --bench
lax_baseline -- 'eval/sort3d_mid_256x1024x64_f64' --noplot`. The baseline ran
remotely on `ovh-a`; candidate reruns hit RCH admission pressure
(`insufficient_slots=4,hard_preflight=1`) and fell open locally. The local
candidate rows are therefore not same-worker proof against the `ovh-a` baseline,
but they are large enough to keep the narrow code path and record a required
remote rerun when a slot opens.

Measured rows:

| workload | main midpoint | candidate midpoint(s) | fresh JAX mean | main/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---|
| `eval/sort3d_mid_256x1024x64_f64` | 390.98 ms remote `ovh-a` | 40.211 ms, 94.185 ms, 117.96 ms local fallback | 1926.511 ms | 0.203x Rust/JAX, 4.93x Rust win | 0.0209x to 0.0612x Rust/JAX, 47.9x to 16.3x Rust win | KEEP |

Fresh JAX comparator used `/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python`,
JAX/JAXLIB 0.10.1 CPU x64, exact `[256,1024,64]` f64 fixture
`(arange * 2654435761 % 1000003).astype(f64)`, warmed
`jax.jit(lambda a: jnp.sort(a, axis=1))`, 8 hot runs. JAX measured best
**1526.131 ms**, p50 **1997.948 ms**, mean **1926.511 ms**. Main was already a
JAX win; this lever attacks the remaining transpose-bound internal gap and
narrows the row to a segmented block primitive. Candidate vs main midpoint is
3.31x to 9.72x faster depending on the noisy local fallback row.

Correctness and quality gates: `cargo test -p fj-lax --profile release
sort_rank3_middle_axis_dense_f64_matches_reference -- --nocapture` passed;
`cargo test -p fj-lax --profile release sort -- --nocapture` passed
32/32 sort tests with existing ignored cases unchanged; `cargo fmt --package
fj-lax -- --check` passed. `AGENT_NAME=ProudSalmon CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a
cargo test -p fj-conformance --profile release -- --nocapture` passed locally
with the tracked `artifacts/` tree present. The same crate-wide conformance run
through `rch exec` rebuilt and ran many tests on `ovh-a`, but failed in
`artifact_schemas` because `.rchignore` excludes `artifacts/` from remote sync;
the focused local `artifact_schemas` gate passed 13/13. `ubs
crates/fj-lax/src/tensor_ops.rs crates/fj-lax/benches/lax_baseline.rs` reported
its known broad historical unwrap/indexing/heuristic inventory in these huge
files while its cargo-aware format, clippy, check, test-build, audit, and deny
checks were green. Risk is scoped to dense f64 rank-3 `axis=1` value sort only;
argsort, other dtypes, other ranks, and smaller axes fall through to the
existing implementation.

## 2026-06-26 - REJECT: general attention einsum batched-GEMM route regresses vs JAX (ProudSalmon)

BOLD-VERIFY land-or-dig pass after `fcf6077d`: worktree scan found no landable
measured win. The off-main `4940278b` complex boolword select head is
patch-equivalent to `origin/main`; the rank-3 sumpool scratch head is a ledgered
reject/loss; the stale QR/SVD WIP head has no measured ratio ledger. New lever
attempted here: route `try_einsum2_matmul_general`'s canonical
`[batch, M, K] x [batch, K, N]` contraction through the existing
`batched_matmul_2d` kernel instead of looping over `matmul_2d` once per canonical
batch slice. This targeted the small-batched-GEMM follow-on called out for
attention `bqhd,bkhd->bhqk`.

Bench command note: this Cargo rejects `cargo bench --release`, so the valid
crate-scoped optimized equivalent was used:
`AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME rch exec
-- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b
AGENT_NAME=ProudSalmon cargo bench -p fj-lax --profile release --bench
lax_baseline -- 'eval/einsum2_general_bqhd_bkhd_bhqk_f64' --noplot`. RCH had no
admissible worker and fell open locally for both baseline and candidate, so the
Rust comparison is same-host/same-target but not remote.

Measured rows:

| workload | main midpoint | candidate midpoint | Rust delta | JAX mean | main/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `eval/einsum2_general_bqhd_bkhd_bhqk_f64` | 1.7633 ms | 2.5855 ms | +47.8% regression | 0.3168 ms | 5.57x loss | 8.16x loss | REVERT |

Fresh JAX comparator used `/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python`,
JAX/JAXLIB 0.10.1 CPU x64, exact `lax_baseline.rs` fixture
`bsz=4,q=64,h=8,d=64,k=64`, warmed `jax.jit(lambda x,y:
jnp.einsum("bqhd,bkhd->bhqk", x, y))`, 50 hot runs. JAX measured best
**0.1884 ms**, p50 **0.2771 ms**, mean **0.3168 ms**, p95 **0.4822 ms**.
The candidate passed `cargo test -p fj-lax
einsum2_general_matmul_bit_identical_to_naive --profile release --lib -- --nocapture`
before measurement, but the production route regressed and was fully reverted
(`crates/fj-lax/src/einsum.rs` restored to HEAD). Do not retry this exact
batched-kernel swap; the existing per-slice `matmul_2d` route is faster for this
small attention shape despite repeated setup. A credible retry needs a genuinely
small-batch-specialized kernel or a fused permute+microkernel that removes
intermediate movement, not a call-level routing swap.

## 2026-06-26 - REJECT: rank-3 generated sum-pool tap kernel is noisy and does not narrow JAX enough (ProudSalmon)

BOLD-VERIFY land-or-dig pass after `fcf6077d`: refreshed the visible
ProudSalmon scratch/worktree heads. The apparent f32 cummax/cummin and
reverse-bitcast wins (`b6505be6`, `de8d462b`, `dce02a09`, `adb409bc`) are already
ancestors of `origin/main`; stale QR/SVD WIP `a00dc114` still has no ratio
ledger. New lever attempted here: replace the dynamic dense-float odometer for
rank-3 f64 `reduce_window(sum)` with a generated VALID/unit-stride 3-D tap
kernel that preserves the exact row-major tap order while using contiguous
last-axis slices. After an initial mixed win5/win9 read, the gate was narrowed to
large cubic windows (`>=9`) to avoid the already-competitive win5 row.

Bench command note: this Cargo rejects `cargo bench --release`, so the valid
crate-scoped optimized equivalent was used:
`AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME rch exec
-- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a
AGENT_NAME=ProudSalmon cargo bench -p fj-lax --profile release --bench
lax_baseline -- 'eval/sumpool_96x96x96_win(5|9)_f64_vec' --noplot`. RCH alternated
between local fallback and `ovh-a` with worker-scoped target-dir rewriting, so
only like-for-like local fallback rows were used for the verdict; the `ovh-a`
candidate row is routing-only.

Measured rows:

| workload | main midpoint | final candidate midpoint | Rust delta | fresh JAX mean | main/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `eval/sumpool_96x96x96_win5_f64_vec` | 14.463 ms local repeat; prior same tree also saw 6.2693 ms | 8.7858 ms | noisy/mixed; final Criterion reports +30.2% vs saved history | 2.096533 ms | 6.90x loss on local repeat | 4.19x loss | REVERT: gate intended to avoid this row, but final full filter still did not produce a stable no-regression story |
| `eval/sumpool_96x96x96_win9_f64_vec` | 25.343 ms local repeat | 26.335 ms | +3.9% slower | 11.195246 ms | 2.26x loss | 2.35x loss | REVERT |

Fresh JAX comparator used `/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python`,
JAX/JAXLIB CPU x64, exact dense f64 fixture from `lax_baseline.rs`
(`sin(arange(96^3) * 0.00123) * 10.0`), warmed jit, 20 runs. JAX means:
win5 **2.096533 ms**, win9 **11.195246 ms**.

Correctness gates before revert were green: focused
`reduce_window_rank3_f64_sum_valid_unit_matches_literal_generic` passed locally
and on `ovh-a`; `cargo test -p fj-lax reduce_window --profile release` passed
49/0 with 15 ignored. Code was reverted to zero diff. Scorecard:
**0 JAX wins / 1 noisy mixed row / 1 measured loss / 0 kept / code reverted**.
Do not retry rank-3 sum-pool by only removing dynamic tap-coordinate overhead;
the residual needs real SIMD/window-vectorization or a different lowering, not a
generated scalar tap loop.

## 2026-06-26 - REJECT: direct LU Schur subtract is under-threshold noise (ProudSalmon)

BOLD-VERIFY land-or-dig pass after `b252d6fc`: live worktree scan found no
unlanded measured win. The dirty ProudSalmon gather/bitcast worktrees are
already represented on `origin/main`; the stale QR/SVD head `a00dc114` has no
ratio ledger; the old `frankenjax_p1vbf51_bench_20260608T104020` LU/matmul
dirty tree has no local measurement artifact or JAX-ratio entry. New lever
attempted here: replace LU's blocked Schur update temporary
`prod = L21 * U12` plus copy-subtract with a strided matmul subtract target that
writes `A22 -= L21 * U12` directly into the LU trailing block.

Bench command note: this Cargo rejects `cargo bench --release`, so the valid
crate-scoped optimized equivalent was used:
`AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR rch exec -- env
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo bench -p
fj-lax --profile release --bench lax_baseline -- 'linalg/lu_1024x1024_f64'
--noplot`. RCH had no admissible worker and fell open locally for both baseline
and candidate, so the comparison is same-host/same-target but not remote.

Measured rows:

| workload | main midpoint | direct-sub midpoint | Rust delta | JAX mean | main/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `linalg/lu_1024x1024_f64` | 43.370 ms | 41.719 ms | -3.81% / 1.04x faster | 2207.735 ms | 0.0196x (50.91x faster than JAX) | 0.0189x (52.92x faster than JAX) | REVERT: below 1.15x keep bar; Criterion says within noise threshold |

Fresh JAX comparator used `/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python`,
JAX 0.10.1 CPU x64, exact `1024x1024` deterministic fixture from
`lax_baseline.rs`, warmed `jax.jit(lambda a: jax.lax.linalg.lu(a))`, 20 runs.
JAX measured best **1724.934 ms**, p50 **2131.946 ms**, mean **2207.735 ms**,
p95 **2720.247 ms**. Focused LU reconstruction test passed with the candidate:
`cargo test -p fj-lax lu_blocked_path_reconstructs_and_matches_scalar --profile
release`.

Scorecard: **main already dominates JAX / 0 material JAX-gap wins / 1
under-threshold Rust-only delta / 0 kept / code reverted**. Do not retry this
allocation-removal shape unless the direct subtract path can be fused into a
larger LU restructuring with at least a 1.15x same-host or same-worker win.

## 2026-06-26 - REJECT: rank-3 row-specialized sum-pool is a cache/threading loss vs JAX (ProudSalmon)

BOLD-VERIFY land-or-dig pass: scanned `.scratch/.worktrees` and live scratch
worktrees for measured wins not on main. The only `.scratch` win head found
(`4940278b` complex boolword select) is patch-equivalent to `origin/main`
(`2c163dfd`), so no worktree win was available to land. New lever attempted:
specialize dense rank-3 `reduce_window(sum)` for VALID, unit-stride,
unit-dilation f64 `[96,96,96]` by replacing `eval_reduce_window_dense_float`'s
generic output odometer with a direct `(z,y)` row kernel and contiguous last-axis
tap loop.

Bench command note: this Cargo rejects `cargo bench --release`, so the valid
crate-scoped optimized equivalent was used:
`AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME rch exec
-- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b
AGENT_NAME=ProudSalmon cargo bench -p fj-lax --profile release --bench
lax_baseline -- 'eval/sumpool_96x96x96_win(5|9)_f64_vec' --noplot`. RCH had no
admissible worker and fell open locally for both baseline and candidate, so the
Rust comparison is same-host/same-target but not remote.

Measured rows:

| workload | main midpoint | candidate midpoint | Rust delta | JAX mean | main/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `eval/sumpool_96x96x96_win5_f64_vec` | 8.1391 ms | 520.63 ms | 63.97x slower | 1.753 ms | 4.64x loss | 296.91x loss | REVERT |
| `eval/sumpool_96x96x96_win9_f64_vec` | 33.585 ms | 85.739 ms | 2.55x slower | 12.977 ms | 2.59x loss | 6.61x loss | REVERT |

Fresh JAX comparator used `uv run --with 'jax[cpu]' --with numpy python`, JAX
0.10.2/JAXLIB 0.10.2 CPU x64, exact dense f64 fixture
`sin(arange(96^3) * 0.00123) * 10.0`, warmed jit, 20 runs. JAX means:
win5³ **1.753 ms**, win9³ **12.977 ms**. Candidate parity test passed before
measurement, but the kernel was rejected and production code reverted
(`crates/fj-lax/src/lib.rs` restored to HEAD). Kept only the benchmark rows so
future digs can compare this 3-D gap directly. Do not retry the same row-kernel
shape; it destroys locality/thread balance for win5 and still loses win9. A real
lever needs either last-axis horizontal reuse without extra full-array copies or
a generated tap kernel that preserves the existing generic path's cache behavior.

Follow-up landing verification while rebasing this evidence onto current main
used the requested `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a`
and reran the retained rows successfully: win5 midpoint **7.6623 ms**, win9
midpoint **22.919 ms**.

## 2026-06-26 - REJECT: radix-4 production dispatch for pow4 FFT is mixed/under-threshold (ProudSalmon)

BOLD-VERIFY land-or-dig pass: live worktree scan found no unlanded measured win.
The dirty f64 gather and reverse-bitcast worktrees are already represented on
`origin/main` (`dce02a09`, `adb409bc`); the only cherry-positive live head remains
stale QR/SVD WIP `a00dc114` with no ratio ledger. New lever attempted here:
promote the existing test-only radix-4 SoA FFT kernel into production dispatch
for batched power-of-four complex FFTs (`n = 256`), halving stage count versus
the radix-2 SoA path.

Bench command note: this Cargo rejects `cargo bench --release`, so the valid
crate-scoped optimized equivalent was used:
`AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR rch exec -- env
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo bench -p
fj-lax --profile release --bench lax_baseline -- 'eval/fft_batch_2048x256'
--noplot`. RCH had no admissible worker and fell open locally for both baseline
and candidate, so the comparison is same-host/same-target but not remote.

Measured rows:

| workload | main midpoint | radix-4 midpoint | Rust delta | JAX mean | main/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `eval/fft_batch_2048x256_complex128` | 9.6679 ms | 10.578 ms | +9.42% regression | 257.543 us | 37.54x loss | 41.07x loss | REVERT |
| `eval/fft_batch_2048x256_complex128_dense_input` | 5.7387 ms | 5.5099 ms | -3.99% faster | 257.543 us | 22.28x loss | 21.39x loss | under 1.15x keep threshold |
| `eval/fft_batch_2048x256_real_dense_input` | 5.5462 ms | 5.2484 ms | -5.37% faster | 2350.235 us | 2.36x loss | 2.23x loss | under 1.15x keep threshold |

Fresh JAX comparator used `/data/projects/frankenjax/benchmarks/jax_comparison/.venv/bin/python`,
JAX/JAXLIB 0.10.1 CPU x64, exact `2048x256` fixtures from `lax_baseline.rs`,
20 runs x 20 inner loops. Scorecard: **0 JAX wins / 1 regression / 2
under-threshold rows / 0 kept / code reverted**. Do not retry by simply
switching the existing SoA pow4 path to radix-4; next FFT lever needs generated
length-specialized butterflies or a fused extract/transform/output boundary.

## 2026-06-26 - KEEP: reverse u32->f64 bitcast direct packing narrows JAX gap (ProudSalmon)

BOLD-VERIFY land-or-dig pass: all measured scratch/worktree wins found in
`/data/projects/.scratch` were already cherry-equivalent to `origin/main`; the
only cherry-positive live head remained stale QR/SVD WIP `a00dc114` with no
ratio ledger. New primitive dug here: dense width-changing `bitcast_convert_type`
from trailing u32 pairs back to f64.

Reconciliation with `c1c9fe80`: that later evidence-only reject measured this
family through local-fallback/noisy rows and one routing-only RCH candidate. The
entry below is kept because it has a same-worker current-main baseline and
candidate pair on `vmi1227854`.

Lever: replace per-pair `to_le_bytes` array reconstruction with direct little-endian
`u64` packing (`low | high << 32`) and add the same DRAM-scale threaded fill used by
the f64->u32 direction. This preserves bit semantics exactly: the low u32 remains
the low 32 bits, the high u32 remains the high 32 bits, and the final `f64::from_bits`
is unchanged.

Bench command note: this Cargo rejects `cargo bench --release`, so the valid
crate-scoped optimized equivalent was used:
`AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR rch exec -- env
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo bench -p
fj-lax --profile release --bench lax_baseline -- 'bitcast_u32_f64_dense_1m'
--noplot`.

Same-worker RCH evidence on `vmi1227854`: current-main `09d1335e`
`eval/bitcast_u32_f64_dense_1m` midpoint **627.77us**. The candidate patch
measured **450.36us** midpoint on the same worker, a **1.39x Rust-side speedup**
(-28.3%). `09d1335e` touched only `crates/fj-lax/src/reduction.rs` and this
ledger, so the bitcast code under test is identical to the candidate patch
applied here. A local-fallback rerun through `rch exec` measured **463.35us**
midpoint with Criterion **-27.410%**, p=0.00.

Fresh JAX comparator on the exact 1M fixture (`jax.lax.bitcast_convert_type`
u32 chunks -> f64, JAX/JAXLIB 0.10.1 CPU x64, warmed jit, 20 runs x 50 inner)
measured best **115.126us**, p50 **149.890us**, mean **177.749us**, p95
**282.816us**. Ratio-vs-JAX ledger:

| workload | fj-lax row | JAX mean | Rust/JAX | verdict |
|---|---:|---:|---:|---|
| `eval/bitcast_u32_f64_dense_1m` current main | 627.77 us | 177.749 us | 3.53x loss | baseline |
| `eval/bitcast_u32_f64_dense_1m` candidate | 450.36 us | 177.749 us | 2.53x loss | KEEP: material loss-narrowing, not a JAX flip |

Scorecard: **0 JAX wins / 1 JAX loss narrowed / 1 kept / 0 reverted**. Next
retry should target allocation/write traffic or a no-zero-copy representation
boundary; do not retry byte-array packing.

## 2026-06-25 - REJECT: u32->f64 bitcast reverse threading is noise, still loses to JAX (ProudSalmon)

BOLD-VERIFY land-or-dig pass found two dirty scratch worktrees,
`/data/projects/.scratch/frankenjax-proudsalmon-bitcast-reverse-20260626T005511Z`
and `/data/projects/.scratch/frankenjax-proudsalmon-bitcast-reverse-baseline-20260626T005840Z`,
with a code-only `U32 -> F64` width-changing bitcast candidate and no ledgered measurement. The
candidate added a `u32_pair_to_f64` helper plus a threaded dense widening path. It was replayed onto
fresh `origin/main` (`c6bfb756`) and measured, then reverted because the decisive local row was
within Criterion noise and the operation remains far behind JAX.

Bench command note: this Cargo still rejects `cargo bench --release` for benches, so the package
scoped optimized equivalent was used:
`AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME rch exec -- env
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b AGENT_NAME=ProudSalmon cargo bench
-p fj-lax --profile release --bench lax_baseline -- eval/bitcast_u32_f64_dense_1m --noplot`.
RCH fell open locally for the current-main baseline runs because no worker slots were admissible;
one candidate run landed on RCH `hz2` but has no same-worker baseline and is routing-only.

Measured rows for `eval/bitcast_u32_f64_dense_1m`, exact fixture from
`crates/fj-lax/benches/lax_baseline.rs`:

| row | midpoint | Rust-side verdict | JAX mean | Rust/JAX |
|---|---:|---|---:|---:|
| current main local fallback, first run | 804.39 us | baseline | 176.684 us | 4.55x loss |
| current main local fallback, repeat | 1.0115 ms | noisy baseline | 176.684 us | 5.72x loss |
| candidate local direct auxiliary | 1.0453 ms | within noise threshold (`-4.34%`, p=0.04) vs saved Criterion baseline | 176.684 us | 5.92x loss |
| candidate RCH `hz2` | 680.41 us | routing-only; no same-worker baseline | 176.684 us | 3.85x loss |

Fresh JAX comparator used `benchmarks/jax_comparison/bitcast_gauntlet.py` with JAX/JAXLIB 0.10.1
CPU x64, 10 runs x 30 inner loops: `bitcast_u32_f64_1m` p50 **182.598 us**, mean **176.684 us**.
Scorecard: **0 wins / 1 loss / 0 kept / 1 reverted**. Do not retry by simply threading this reverse
bitcast; the next credible lever would need to remove allocation or fuse the producer/consumer so
the row can approach JAX's ~177 us packed reinterpret path.

## 2026-06-25 - KEEP: Cholesky panel fan-out cap narrows the JAX LAPACK gap (ProudSalmon)

BOLD-VERIFY land-or-dig pass: the scratch complex-select worktree commit
`4940278b` was patch-equivalent to `origin/main` under landed commit `2c163dfd`;
the only live patch-positive worktree was stale QR/SVD WIP `a00dc114` in
`/data/projects/frankenjax_poyvi1_pass188`, with no bench/ledger ratio evidence.
So this pass dug a different measured primitive: dense f64 Cholesky, previously
documented as a LAPACK-backed JAX loss.

Lever: raise the Cholesky panel-solve and Schur-update minimum rows per worker
from 32 to 96. The existing blocked right-looking kernel spawned too many scoped
workers for each panel; larger chunks preserve the algorithm and numeric order
within each row while reducing repeated thread setup and tail fragmentation.

Bench evidence used the requested warm target dir through `rch exec`; `cargo
bench --release` is not accepted by this Cargo, so the optimized equivalent was
`cargo bench -p fj-lax --profile release --bench lax_baseline --
'linalg/cholesky_512x512_f64|linalg/cholesky_1024x1024_f64' --noplot`.
RCH fell open locally for the two Criterion runs because no worker slots were
admissible, but the command remained crate-scoped with
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b`.

Ratio-vs-JAX ledger, exact benchmark fixtures, JAX/JAXLIB 0.10.2 CPU x64 with
`JAX_ENABLE_X64=1`:

| workload | main midpoint | candidate midpoint | Rust-side delta | JAX mean | main/JAX | candidate/JAX | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `linalg/cholesky_512x512_f64` | 10.792 ms | 9.4650 ms | 1.14x faster / -12.3% | 1.441132 ms | 7.49x loss | 6.57x loss | KEEP: loss narrowed |
| `linalg/cholesky_1024x1024_f64` | 46.550 ms | 36.201 ms | 1.29x faster / -22.2% | 6.682657 ms | 6.97x loss | 5.42x loss | KEEP: loss narrowed |

Fresh JAX p50s were 1.398404 ms (512) and 6.606567 ms (1024); the conclusion is
unchanged by p50 ratios. This is not a Rust-over-JAX flip, only a material
thread-policy narrowing of the LAPACK gap. Next Cholesky retry should target the
remaining BLAS/LAPACK structural gap, not another per-panel thread fan-out cap.

Conformance and gates: `cargo test -p fj-lax cholesky --release --lib --
--nocapture` passed 15/15 on RCH `hz2`; `cargo test -p fj-conformance --test
linalg_oracle cholesky --release -- --nocapture` passed 10/10 on RCH `hz2`;
`cargo check -p fj-lax --all-targets`, `cargo clippy -p fj-lax --all-targets --
-D warnings`, and `cargo fmt -p fj-lax --check` all passed.
UBS on `crates/fj-lax/src/linalg.rs docs/NEGATIVE_EVIDENCE.md` remained a
file-wide inventory signal for existing `linalg.rs` test/helper panic/indexing
surfaces (25 critical / 1875 warning) and reported fmt/clippy/check clean; the
changed production diff is only the two Cholesky thread-policy constants.

## 2026-06-25 - KEEP: f64 gather AMAC-style load interleaving narrows JAX gap (ProudSalmon)

No measured scratch/worktree win was landable: `/data/projects/.scratch/frankenjax-proudsalmon-boldverify-20260625`
was patch-equivalent to main, and `/data/projects/frankenjax_poyvi1_pass188` only contained the stale
`a00dc114` QR/SVD WIP without a ledgered JAX win. New lever shipped here: scattered single-element
f64 gather now uses an 8-lane load-then-store interleaved helper inside each existing output shard.
This keeps the same pre-resolved `idx` values and the same `src[idx]` reads, but exposes a batch of
independent random loads before the stores instead of relying on the iterator loop shape. Other dtypes
continue to use the previous generic `gather_single_dense` path.

Bench command note: this Cargo rejects `cargo bench --release` for benches (`unexpected argument
'--release'`), so the package-scoped optimized equivalent was used:
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a rch exec -- cargo bench -p fj-lax
--profile release --bench lax_baseline -- eval/gather_scatter_1m_f64 --noplot`.

Measured head-to-head row: `eval/gather_scatter_1m_f64`, 1M pseudo-random f64 gathers from a 4M f64
operand, exact fixture from `crates/fj-lax/benches/lax_baseline.rs`. Same-worker RCH proof on `hz2`:
current main restored baseline **12.584 ms** midpoint vs candidate **6.1367 ms** midpoint, a **2.05x**
Rust-side speedup. Fresh exact-fixture JAX comparator (`uv run --with 'jax[cpu]' --with numpy`,
JAX/JAXLIB 0.10.2, `JAX_ENABLE_X64=1`, CPU, 20 runs x 10 inner) measured best **3.3149 ms**,
p50 **3.8617 ms**, mean **3.8551 ms**. Candidate/JAX ratio remains a **1.59x Rust/JAX loss** by
mean/p50, narrowed from the same-worker restored-main ratio of **3.26x** against that JAX mean.

Landing recheck after `dce02a09` reached `origin/main`/`origin/master`: the requested warm target
dir `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b` was used through
`AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME rch exec -- env
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b AGENT_NAME=ProudSalmon cargo bench
-p fj-lax --profile release --bench lax_baseline -- eval/gather_scatter_1m_f64 --noplot`. RCH fell
open locally because no worker slots were admissible; the package-scoped Criterion row measured
**6.6511 ms** midpoint (`6.5183..6.8147 ms`), which is a **1.73x Rust/JAX loss** against the same
JAX mean and remains a material narrowing versus the restored-main **3.26x** loss ratio.

Correctness scope: f64 single-element gather only; index resolution, OOB policy, dtype construction,
and all non-f64 paths are unchanged. GREEN: `cargo test -p fj-lax --profile release --lib gather
-- --nocapture` 23/0 with 8 ignored perf probes; `cargo test -p fj-conformance --profile release
--test gather_scatter_oracle -- --nocapture` 59/0; `cargo fmt -p fj-lax --check`; `cargo check
-p fj-lax --all-targets`; `cargo clippy -p fj-lax --all-targets -- -D warnings`; `git diff --check`.
Targeted `ubs crates/fj-lax/src/tensor_ops.rs docs/NEGATIVE_EVIDENCE.md` remains a broad inventory
signal for the existing large Rust file (246 critical / 5541 warning / 2129 info), with no new
unsafe blocks and clippy/fmt/build/test clean.

## 2026-06-25 - KEEP: dense f64 full ReduceSum tree fold beats JAX on the 16M row (ProudSalmon)

No measured worktree win was landable: the only positive scratch cherry found was the QR/SVD WIP
`a00dc114` in `/data/projects/frankenjax_poyvi1_pass188`, and it had no bench/ledger evidence on
main. New lever shipped here: large dense f64 full `ReduceSum` now folds contiguous chunks on up
to 8 worker threads and combines partials, while small vectors remain on the old sequential path.
This intentionally relaxes the older self-imposed bit-order lock only for large f64 full sums;
JAX/XLA already uses tolerance/tree semantics for floating reductions.

Measured head-to-head row: `eval/reduce_sum_16m_f64_full` over the exact fixture
`sin(i*1.1e-7)*3.0`, `JAX_ENABLE_X64=1`, CPU. RCH current-main baseline on `vmi1227854`:
8.5162ms midpoint, retained as routing evidence. Same-worker proof on `ovh-a`: current main
12.979ms midpoint vs candidate 3.2170ms midpoint, a 4.03x Rust-side speedup. Fresh local JAX
0.10.2/jaxlib 0.10.2 comparator: 5.0331595ms p50, 5.3822536ms mean. Candidate/JAX ratio is
0.639x by p50 and 0.598x by mean, so fj-lax is 1.56-1.67x faster on this row.

Correctness scope: the existing small dense f64 full-sum bit-identity test stays green for inputs
below the thread threshold; a new large-vector test checks the tree result against the sequential
fold with a tight floating tolerance. Product and non-f64 sums stay on their prior scalar paths.
Final gates: `cargo test -p fj-lax dense_f64_reduce_sum --release --lib -- --nocapture` 3/3,
`cargo test -p fj-conformance --test reduce_sum_oracle --release -- --nocapture` 41/41,
`cargo check -p fj-lax --all-targets`, `cargo clippy -p fj-lax --all-targets -- -D warnings`,
`cargo fmt --check -p fj-lax`, and `git diff --check`. UBS remains a file-wide inventory gate for
`reduction.rs` rather than a clean signal here: final scan still reports 42 critical / 1655 warning
pre-existing findings in the large file, after removing the new thread-join `expect`.

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

UPDATE 2026-06-22 (SlateHarrier) — SESSION CAPSTONE, frontier now fully mapped. Confirming the
above lesson, a rich SECOND vein was found+shipped this session: **dtype-sibling gaps** (secondary
dtypes lagging the optimizations their f64/i64 siblings got) + **naive-loop→GEMM** in derived code.
Shipped (each same-binary A/B, bit-identical, GREEN):
  • scatter-add f64 parallel range-partition 2.78x + u64-pack ~1.06x (JAX loss 2.74x→~1.25x)
  • fj-ad VJP `matmul_2d`+`matmul_f64` (18 callers) → fj-lax GEMM **22-119x** (grad thru QR/LU/SVD/eigh)
  • fj-dispatch vmap(Dot) → batched GEMM: float **38-66x**, integer **17.35x**
  • complex128 matmul 4-row register-block (non-batched + batched) — **flipped 1.95x loss → parity/WIN**
  • complex128 gather de-box ~2.6x; complex128 scatter-add partition ~1.39x; SIMD cbrt ~1.25x
  • NEW DOMINATION: integer matmul **21-32x** (XLA has no integer BLAS — joins sort/order-stats)
Measured NO-SHIPS / ceilings (do-not-retry): scatter f32/i64 partition (regresses); gather index
pre-sort (0.39x, Zen3 vgather ceiling); cbrt division-free flip (0.85x, fma-gated).
MAP CORRECTIONS: matmul dtype matrix is COMPLETE (i64/u64/complex all blocked — stale "i32/u32/u64
gap" note retired); **conv2d ~11.8x loss is 84% fma-bound GEMM + 16% (already-threaded) im2col → folds
into `cntiy` +fma**, NOT a structural lever. CONVERGENCE: the contained, unowned, non-gated, non-niche
frontier is now EXHAUSTIVELY mined (matmul/gather/scatter all-dtype-done, structural ops dense, nn
fused, interpreter dense-env+fused+fast-paths). **The single highest-leverage remaining unlock is the
`cntiy` +fma maintainer decision — now MEASURED to gate the two biggest float ML ops (matmul AND
conv2d) plus exp/softmax/transcendentals/cholesky-GEMM simultaneously.** Other remainders: `murmw` FFT,
`ur4h3` eigh/SVD, linalg (codex zone), filed P3 complex-conv, and multi-session swings (compiled-jaxpr
dispatch / fused attention).

UPDATE 2026-06-25 (ProudSalmon, complex `select`/`where` BOLD-VERIFY): a contained
non-`+fma` dtype-sibling gap remained in `jnp.where(mask, complex_a, complex_b)`.
The real/float/int/half/unsigned select fast paths already consumed packed
BoolWords predicates directly; complex select only accepted dense Bool slices,
so a comparison-generated packed mask fell back to materializing the predicate
and boxed complex output. Shipped the analogous complex BoolWords path plus
threaded dense fill. Proof: RCH `hz2`
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo test -p fj-lax --lib select_complex_boolwords_predicate_bit_identical --release -- --nocapture`
passed 1/1. Bench: `cargo bench --release` is not accepted by this Cargo
(`unexpected argument '--release'`), so the valid optimized equivalent was
`cargo bench -p fj-lax --profile release --bench elementwise_gauntlet -- complex_select_boolwords_1m --noplot`
on RCH `vmi1227854`: new dense bit-test/threaded path **1.5325ms** midpoint vs
old materialize+boxed **4.7203ms** midpoint = **3.08x Rust-side speedup**.
Fresh JAX/JAXLIB 0.10.1 CPU x64 comparator from the existing
`benchmarks/jax_comparison/.venv` on the exact 1M complex128 fixture measured
**1.166893ms p50 / 1.168015ms mean**, so the production row narrows a
**4.05x Rust/JAX loss to 1.31x**. Verdict: KEEP as a material loss-narrowing
primitive lever; not a Rust-over-JAX flip. Next retry must beat ~1.17ms with a
lower-overhead thread policy or vectorized complex mux, not another boxed-mask
avoidance pass.

UPDATE 2026-06-25 (ProudSalmon, AGENT_NAME=ProudSalmon, dense `bitcast_convert_type`
f64->u32 BOLD-VERIFY): no unlanded measured worktree win was found for this lane
(`/data/projects/.scratch/frankenjax-proudsalmon-boldverify-20260625` was already
patch-equivalent to main; `/data/projects/frankenjax_poyvi1_pass188` was a stale
QR/SVD WIP with no ledgered win), so the next different primitive was width-changing
bitcast. The production path split each f64 into two u32 lanes, but its 1M-row gate
spawned worker threads for cheap copy/shuffle traffic, paying scheduling and first-touch
cost instead of letting the tight serial fill run. Shipped the same DRAM-scale gate used
by other cheap memory-bound ops: `BITCAST_WIDTH_CHANGE_PARALLEL_MIN =
crate::arithmetic::CHEAP_BINARY_PARALLEL_MIN`.

Evidence discipline:
  - Baseline Rust local fallback through `rch exec` with the requested warm target and
    crate scope:
    `AGENT_NAME=ProudSalmon RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo bench -p fj-lax --profile release --bench lax_baseline -- 'bitcast_.*_1m' --noplot`.
    Current-main `eval/bitcast_f64_u32_dense_1m` midpoint: **2.8848ms**.
  - Same-worker RCH A/B on `vmi1264463` (`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a`,
    `cargo bench -p fj-lax --profile release --bench lax_baseline -- bitcast_f64_u32_dense_1m --noplot`):
    old threshold **14.334ms** midpoint (`12.374..16.465ms`) vs candidate **4.9227ms**
    midpoint (`3.6131..6.8078ms`), Criterion **-65.658%**, p=0.00, **2.91x Rust-side
    speedup**. This worker was heavily contended and is used only for same-host A/B.
  - Auxiliary local patched ratio row, same warm target, direct crate-scoped `cargo bench`
    because RCH has no honored force-local mode: candidate **407.91us** midpoint
    (`394.89..420.94us`), Criterion **-91.916%**, p=0.00 against the stored local
    Criterion baseline.
  - Fresh local JAX comparator: `benchmarks/jax_comparison/.venv/bin/python
    benchmarks/jax_comparison/bitcast_gauntlet.py --runs 20 --warmup 5 --inner-loops 50`,
    JAX/JAXLIB 0.10.1 CPU x64 on the exact 1M f64->u32 fixture:
    **176.627us mean / 183.613us p50**.

Ratio-vs-JAX ledger:

| workload | fj-lax row | JAX mean | Rust/JAX | verdict |
|---|---:|---:|---:|---|
| `eval/bitcast_f64_u32_dense_1m` current main, local fallback | 2.8848 ms | 176.627 us | 16.33x loss | baseline |
| `eval/bitcast_f64_u32_dense_1m` candidate, local auxiliary | 407.91 us | 176.627 us | 2.31x loss | KEEP: material loss-narrowing, not a JAX flip |
| `eval/bitcast_f64_u32_dense_1m` RCH same-worker `vmi1264463` | 14.334 ms -> 4.9227 ms | 176.627 us | 81.15x -> 27.87x loss | KEEP as same-host A/B proof only; host is noisy |

Scorecard: **0 JAX wins / 1 JAX loss narrowed / 1 kept / 0 reverted**. Do not retry
by lowering the width-changing bitcast thread gate; the next credible lever is
allocation/fill removal or vectorized packing that beats the retained **407.91us**
local row and closes JAX's **176.627us** comparator.

UPDATE 2026-06-25 (ProudSalmon, AGENT_NAME=ProudSalmon, cumprod/cummax SIMD-prefix
BOLD-VERIFY NO-SHIP): no unlanded measured worktree win was found for this lane, so
the next radical lever was the previously scoped Blelloch/Hillis-Steele primitive:
specialize the 4M f64 one-line `cumprod`/`cummax` local blocked scan instead of
reusing the generic closure scan. The candidate used an f64x8 lane-prefix product
plus inlined min/max scan plumbing, left cumsum untouched, and was reverted after
measurement.

Same-worker RCH evidence (`RCH_WORKER=vmi1152480`,
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b`,
`cargo bench -p fj-lax --bench lax_baseline` filtered to the 4M scan rows):

| row | retained main p50 | candidate p50 | Criterion verdict |
| --- | ---: | ---: | --- |
| `eval/cumprod_4m_f64_1d` | 15.859ms | 15.538ms | no change, p=0.69, CI -5.88%..+11.44% |
| `eval/cummax_4m_f64_1d` | 17.587ms | 16.231ms | no change, p=0.17, CI -9.95%..+1.16% |

Fresh JAX comparator (`uv run --with jax --with jaxlib --with numpy`, JAX/JAXLIB
0.10.2, CPU, `JAX_ENABLE_X64=1`, 10 warmed compiled runs on the exact 4M f64
fixtures) measured `cumprod` p50 **43.061ms** and `cummax` p50 **38.828ms**.
Retained Rust/JAX p50 ratios are therefore **0.37x** for `cumprod` (Rust 2.71x
faster than JAX) and **0.45x** for `cummax` (Rust 2.21x faster than JAX). The
candidate would have remained inside Criterion noise, so it is not a valid win
despite the current retained rows already beating fresh JAX.

Conformance while the candidate was present: RCH `cargo test -p fj-lax
blocked_dense_f64_single_line_cumulative_matches_serial_reference --release --lib
-- --nocapture` passed 1/1 on `ovh-a`. Final source diff was returned to empty;
only this ledger entry is retained. Retry only with perf-counter or disassembly
evidence for a new stall source, not another closure/SIMD-prefix variant.

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

UPDATE 2026-06-25 (ProudSalmon, `frankenjax-murmw` FFT BOLD-VERIFY): alien-graveyard /
alien-artifact-coding / extreme-software-optimization route kept a bounded immutable FFT plan cache
across repeated eval calls. This is the production version of the 2026-06-21 "only a plan-CACHED
Bluestein could change this" note, generalized to radix-2, mixed-radix twiddle tables, and Bluestein
plans with a 32-entry in-process cap. It changes no butterfly order and no output math; it only stops
rebuilding `(length, inverse)` plans between calls.

Evidence discipline:
  - No unlanded measured worktree win was found: `/data/projects/frankenjax_poyvi1_pass188` was a
    closed rejected QR/SVD candidate with no ledgered win; no FFT win was stranded off main.
  - Fresh JAX comparator: `/data/projects/.scratch/jaxvenv/bin/python`, JAX/JAXLIB 0.10.2 CPU x64,
    exact `sin(i*0.125)+1j*cos(i*0.25)` fixtures, warmed jit, 30 repeats.
  - Fresh current-main RCH baseline (origin/main `e9c60ec0`, worker `hz2`):
    `fft_1000_complex128` 34.928us, `fft_batch_256x1009_prime_complex128_dense_input` 4.0715ms,
    `fft_batch_128x1000_complex128` 2.8376ms.
  - Candidate RCH proof (rebased patch, worker `vmi1152480` before `origin/main` advanced; repeat
    after rebase on `vmi1264463` went stale and was cancelled via RCH): `fft_1000_complex128`
    21.314us, prime batch 4.4423ms, smooth batch 3.3131ms. Local warm-target Criterion also showed
    large Rust-side drops for the same three rows: 21.020us / 4.7438ms / 2.6548ms.
  - Correctness/quality: `cargo check -p fj-lax --all-targets` GREEN pre-rebase; `cargo test -p
    fj-lax transform_batches_dense_dispatch_matches_dft_oracle --release --lib` GREEN; `cargo test
    -p fj-conformance --release --test fft_oracle --test linalg_fft_oracle_parity` GREEN (27+1
    tests); `cargo clippy -p fj-lax --all-targets -- -D warnings` GREEN; `rustfmt --edition 2024
    --check crates/fj-lax/src/fft.rs` GREEN; `git diff --check` GREEN. Post-rebase check retries
    were infra-blocked twice on `ovh-b` by `zerocopy v0.8.48` build-script SIGILL before reaching
    `fj-lax`; no Rust diagnostic was emitted for this patch.

Ratio-vs-JAX ledger:

| workload | fj-lax candidate | JAX mean | Rust/JAX | verdict |
|---|---:|---:|---:|---|
| `eval/fft_1000_complex128` | 21.314 us | 8.800 us | 2.42x loss | KEEP: narrows current-main gap from ~3.97x to ~2.42x; production single-call plan setup was the lever |
| `eval/fft_batch_256x1009_prime_complex128_dense_input` | 4.4423 ms | 0.341097 ms | 13.02x loss | no JAX win; batch row remains pocketfft/SIMD-bound and cross-worker before/after is noisy |
| `eval/fft_batch_128x1000_complex128` | 3.3131 ms | 0.172758 ms | 19.18x loss | no JAX win; smooth-composite batch remains generated-kernel/quiesced-host territory |

Scorecard: **0 JAX wins / 3 JAX losses / 1 material narrowing; 1 kept / 0 reverted**. Do not cite
this as an FFT domination. The kept primitive is specifically a warmed repeated-eval plan-cache
win that removes avoidable setup cost and narrows the singleton FFT JAX gap. The batch gaps remain
the same frontier: pocketfft-class SIMD-within-FFT, generated length-specialized kernels, or a
quiesced-host threading proof.

UPDATE 2026-06-25 (ProudSalmon, real `Rfft` power-of-two plan cache BOLD-VERIFY): a second bounded
FFT plan-cache gap remained after the complex FFT cache landed. `eval_rfft` still rebuilt
`RealRfftPower2Plan::new(fft_length)` on every eval call for power-of-two real FFTs, including the
planned half-length radix-2 FFT and recombination twiddles. Shipped the same immutable 32-entry
in-process cache discipline used by radix-2/Bluestein/twiddle plans, with no butterfly-order or
numeric-path change.

Evidence discipline:
  - No unlanded measured worktree win was found before digging; this was a fresh lever from the
    current `fj-lax` source.
  - Benchmark command used the requested warm target plumbing and crate scope:
    `AGENT_NAME=ProudSalmon RCH_REQUIRE_REMOTE=1 RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR rch exec -- env CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a cargo bench -p fj-lax --profile release --bench lax_baseline -- rfft_256_f64 --noplot`.
  - Same-worker RCH `hz2` A/B on `eval/rfft_256_f64`: uncached current-main path **4.6887 us**
    midpoint (`4.5852..4.8059 us`) vs cached candidate **2.3190 us** midpoint
    (`2.2671..2.3846 us`), Criterion **-49.582%** mean change, p=0.00, **2.02x Rust-side speedup**.
    A cross-worker `vmi1153651` candidate sample was ignored for proof because it was not comparable.
  - Fresh live JAX comparator: `uv run --with 'jax[cpu]' --with numpy python`, JAX 0.10.2 CPU x64,
    exact `sin(i*0.125)+cos(i*0.03125)` length-256 f64 fixture, warmed jit, 200 repeats:
    **25.669 us p50 / 33.904 us mean / 50.606 us p95**.
  - Correctness/quality: `cargo fmt --check -p fj-lax` GREEN; `cargo test -p fj-lax --lib rfft
    --release -- --nocapture` GREEN (25 passed, 2 ignored); `cargo check -p fj-lax --all-targets`
    GREEN; `cargo clippy -p fj-lax --all-targets -- -D warnings` GREEN; `cargo test -p
    fj-conformance --release --test fft_oracle -- --nocapture` GREEN (27 passed); `cargo test -p
    fj-conformance --release --test linalg_fft_oracle_parity -- --nocapture` GREEN (1 passed).

Ratio-vs-JAX ledger:

| workload | fj-lax candidate | JAX mean | Rust/JAX | verdict |
|---|---:|---:|---:|---|
| `eval/rfft_256_f64` | 2.319 us | 33.904 us | 0.068x / **14.62x fj-lax win** | KEEP: removes repeated real-plan setup; flips this warmed scalar RFFT row further into Rust-win territory |

Scorecard: **1 JAX win / 0 JAX losses / 1 kept / 0 reverted**. Do not generalize this to FFT batch
domination: the lever is specifically repeated-eval setup removal for power-of-two real RFFT. Batch
FFT losses remain routed to pocketfft-class SIMD/generated-kernel work.

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
    (Follow-up RESOLVED: complex SCATTER is NOT generic-boxed — `eval_scatter_dense` has a complex
    branch (`scatter_typed!` + lexicographic min/max), and complex scatter-ADD now routes through the
    parallel range-partition (~1.39x, entry above). This earlier "still generic-boxed" note was stale.)
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
    UPDATE 2026-06-24 (BlackThrush, `frankenjax-complex-conv-gemm-5xdr7`): dense complex ungrouped
    conv1d/conv2d now routes through im2col + `rank2_complex_matmul` when operands expose dense
    complex storage; boxed/non-dense complex keeps the old direct loop as the reference fallback.
    Same-binary RCH release A/B on c128 conv2d VALID `[2,32,32,8] * [3,3,8,16]`: dense im2col/GEMM
    **2.3419ms** vs boxed direct **10.1233ms** = **4.323x Rust-side win**. Fresh JAX/JAXLIB 0.10.1
    CPU x64 on the exact shape measured mean **0.622643ms** (p50 **0.615015ms**), so the retained row
    is still a **3.76x Rust/JAX loss** (old boxed path would be **16.26x** loss). Conformance: dense
    vs boxed complex conv1d/conv2d bit-for-bit tests passed; `fj-conformance --test conv_oracle complex`
    4/4 passed.
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
    BATCH-AXIS SIMD RULED OUT (2026-06-24, SlateHarrier): prototyped an explicit-`Simd<f64,4>`-across-batch
    radix-2 (bit-identical to per-row, verified all n×batch×fwd/inv) and A/B'd it vs the production path
    same-binary [2048×256-pt c128, single-thread]: per-row 5.24ms / **prod scalar-SoA `batch_butterfly_block`
    2.63ms** / explicit-f64x4 **2.78ms = 0.95x** (explicit is SLIGHTLY SLOWER). The production pow2 batched
    FFT (`vectorized_pow2_block`→`batch_butterfly_block`, TILE=8) ALREADY vectorizes the batch axis — LLVM
    autovectorizes the SoA row-loop, and explicit SIMD adds nothing. So the batch axis is DONE; the residual
    19.4x is purely the fma-gate (butterfly `a*b±c*d`) + intra-FFT radix algorithm (both pow2-golden-locked).
    Prototype REVERTED (~0-gain). Do NOT re-attempt batch-axis SIMD for pow2 FFT.
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

DISPATCH VERIFIED (2026-06-22, SlateHarrier) — maxpool is NOT a dispatch bug; the 2.3x is purely
the deque-vs-SIMD gap. Confirmed eval_primitive routes overlapping/large windows (15x15: win_total
225 > 2·win_sum 60) to the O(n) separable deque (`reduce_window_separable_maxmin`, lib.rs ~5531)
BEFORE the rank-2 direct O(out·∏window) loop (~5630, small-window-only). Same-binary A/B
(`bench_maxpool_rank2_direct_vs_deque`, 256x256/15x15): direct loop **49ms** vs deque **0.87ms =
56x** (bit-identical) — i.e. the deque IS used (the dispatch gate is correct; the direct loop would
be catastrophic if mis-routed, so the A/B stays as a routing regression guard). The residual ~2x JAX
loss is the safe-Rust **scalar deque vs XLA's SIMD-vectorized window reduction** — a hard SIMD lever
(the deque is inherently sequential/pointer-chasing; SIMD-direct loses for large windows since deque
is O(n) vs O(n·k)), NOT a contained fix. Do NOT chase the maxpool dispatch — it's correct.

THREADING NO-SHIP (2026-06-22, SlateHarrier): tried threading the production deque
(`reduce_window_separable_maxmin`, 5185) across its independent outer blocks. REGRESSES **0.62-0.75x**
(prod-threaded 1.2-1.9ms vs serial 0.9-1.2ms): the per-pass work (~58k elems / ~0.9ms for 256x256/15x15)
is too small to amortize 16-thread spawn, AND the leading axis has outer==1 (no outer parallelism).
Reverted to serial. ALSO measured (less-contended run): serial 5185 = **0.68ms** vs JAX 0.48ms = **~1.4x**
(the prior 2.3x was a contended host) — so maxpool is closer than thought, and the residual is SIMD
(single-thread vectorized window reduction), not parallelism. Do NOT thread maxpool; SIMD is the only
lever and it's hard (the deque is sequential; SIMD-direct loses for large windows). Bench kept as guard.

TRANSPOSE LEVER RULED OUT (2026-06-22, SlateHarrier): hypothesized the strided axis-0 deque pass
(reads down columns, stride=inner=256) was the bottleneck, fixable by transposing it to contiguous.
MEASURED the per-axis split (`bench_maxpool_axis_split`, [15,1] vs [1,15]): axis0-strided **0.31-0.69ms**
vs axis1-contig **0.24-0.66ms** = only **1.04-1.28x** — the strided axis is NOT dominant (256x256=512KB
fits L2, so strided reads hit L2 not DRAM; modest penalty). A transpose-to-contiguous would save only
~5-12% — not worth the transpose overhead. So the maxpool gap is the scalar deque's branchy per-element
push/pop overhead vs XLA's SIMD, UNIFORM across both axes — confirming no contained lever (only the hard
SIMD-windowed-max rewrite). maxpool is fully characterized: ~1.4x JAX, SIMD-gated, do not re-probe.
NOTE: the "SIMD-gated, hard" conclusion above applies to the LARGE-window DEQUE path (15x15). The
COMMON SMALL-window CNN case is a SEPARATE, bigger loss that DID have a tractable SIMD win — see next.

## 2026-06-22 - CNN maxpool (small window) ~6-9x JAX LOSS → WIN via SIMD-over-channel (SlateHarrier)

The 15x15 maxpool I'd been measuring (~1.4x, deque path) was NOT the common case. COMMON CNN pooling
is 2x2/3x3, and `win_total <= 2*win_sum` so it MISSES the separable-deque gate — rank-4 [N,H,W,C]
fell to the scalar `eval_reduce_window_dense_float` ODOMETER. Measured (eval_primitive, f64):

| shape | window | fj-lax BEFORE | JAX | ratio |
| --- | --- | ---: | ---: | --- |
| [8,112,112,64] | 3x3/s2 | 84 ms | 9.34 ms | **~9x LOSS** |
| [8,56,56,128] | 2x2/s2 | 12-16 ms | 2.11 ms | **~6-7x LOSS** |

Root cause: the channel axis C is the contiguous last dim with window 1 (not pooled), but the scalar
odometer walks it element-by-element. FIX: `reduce_window_simd_channel_maxmin_f64` — for each output
position, max/min over the (kh,kw) taps VECTORIZED across C channels (f64x8, NaN-propagating to
canonical NaN via a `simd_ne|select` mask to match lax.max/min; signed-zero matches f64::max/min).
Wired before the scalar odometer, gated to channel-last (last axis window==1/stride==1/pad==0), f64,
max/min, no dilation; large windows still prefer the deque. Result (eval_primitive, f64):

| shape | fj-lax AFTER | JAX | verdict |
| --- | ---: | ---: | --- |
| [8,112,112,64] 3x3/s2 | **8.73 ms** | 9.34 ms | **WIN ~1.07x** (was 9x loss; ~9.6x Rust-side) |
| [8,56,56,128] 2x2/s2 | **2.76 ms** | 2.11 ms | ~1.3x (was 6-7x loss; near-parity) |

Bit-identical: `maxpool_simd_channel_bit_identical` permanent guard (max/min × finite/NaN/±0 ×
VALID/padded × rank-3/4) + reduce_window 44/0 + full fj-lax lib 1587/0.

F32 SIBLING SHIPPED (2026-06-22, SlateHarrier) — JAX's DEFAULT CNN dtype. fj-lax f32 maxpool also hit
the scalar odometer (~84ms) but JAX f32 is 16-wide (3x3/s2 = 1.48ms, 2x2/s2 = 0.59ms) so it was a
~57x LOSS — worse than f64. `reduce_window_simd_channel_maxmin_f32` (f32x16, same channel-last kernel,
NaN→canonical f32::NAN, signed-zero matches f32::max/min) results (eval_primitive, f32):

| shape | f32 BEFORE | f32 AFTER | JAX f32 | verdict |
| --- | ---: | ---: | ---: | --- |
| [8,56,56,128] 2x2/s2 | ~12ms+ | **0.54 ms** | 0.59 ms | **WIN ~1.1x** (was ~40-57x loss) |
| [8,112,112,64] 3x3/s2 | ~84ms | **2.68 ms** | 1.48 ms | ~1.8x (was ~57x; near-parity) |

Bit-identical: `maxpool_simd_channel_f32_bit_identical` guard (max/min × finite/NaN/±0) + full fj-lax
lib 1589/0. The residual ~1.8x on 3x3 f32 is JAX's THREADING (XLA parallelizes; the SIMD-channel path
is single-thread) — the outer spatial loop is embarrassingly parallel (disjoint per-osp output blocks),
so threading it (filed, od11p) should flip 3x3 too. Both f64+f32 channel-last CNN maxpool now win or
near-parity vs the original 6-57x losses. (clippy blocked this commit by the ovh-b zerocopy SIGILL
infra flake; verified via ovh-a build + 1589/0 + guards; f32 mirrors the clippy-clean f64 code.)

THREADED (2026-06-22, SlateHarrier) — flipped the 3x3 f32 residual to a WIN. Threaded the SIMD-channel
outer spatial loop (`std::thread::scope` + `split_at_mut` over disjoint per-position output blocks;
`simd_channel_block_f{32,64}` workers decode-per-position, no cross-state; `work_scaled_threads(out_total)`
gate → 1 for small pools). SAME-BINARY A/B (`bench_maxpool_simd_thread_ab`, trustworthy — an earlier
CROSS-invocation run falsely showed a 2x2 regression that was pure contention noise):

| shape | f64 serial→thr | f32 serial→thr |
| --- | --- | --- |
| [8,112,112,64] 3x3/s2 | 4.55→1.91 ms (**2.38x**) | 2.64→0.75 ms (**3.54x**) |
| [8,56,56,128] 2x2/s2 | 1.11→0.49 ms (**2.28x**) | 0.48→0.36 ms (**1.35x**) |

vs JAX: f32 3x3 **0.75ms vs 1.48ms = ~2x WIN** (was ~1.8x LOSS); f32 2x2 0.36 vs 0.59 = ~1.6x WIN; f64
3x3 1.91 vs 9.34 = **~4.9x WIN**. Threading wins ALL cases same-binary (1.35-3.54x), bit-identical
(threaded-partition guard configs [8,56,56,64] vs naive + the A/B serial==threaded assert). The common
CNN maxpool (f64+f32) is now a DOMINATION across the board. METHOD LESSON: only the same-binary A/B is
trustworthy — cross-invocation maxpool timings swing 2-3x with host contention (do not revert on them).

## 2026-06-22 - SUM/AVG pooling ~35-120x JAX loss → 22-53x faster, bit-identical (SlateHarrier)

Extended the channel-last SIMD pooling vein from max/min to SUM (avg/sum pooling — as common as maxpool:
global average pooling etc.). Float `reduce_window(add)` channel-last [N,H,W,C] hit the same scalar
dense_float odometer (the i64 summed-area-table is integer-only; float can't use SAT/separable without
changing the sum order). KEY LEGALITY: SIMD ACROSS CHANNELS preserves each channel's row-major tap-sum
order (channels are independent) → bit-identical despite float non-associativity — UNLIKE axis-reduce
SIMD (the memory's float-sum DO-NOT, which reorders summands). f64 accumulates in f64; f32 widens→f64→
rounds `as f32`, EXACTLY as the odometer's f64-view path → bit-identical. Threaded outer loop.

`reduce_window_simd_channel_sum_f{64,32}` + `simd_channel_sum_block_f{64,32}`. SAME-BINARY A/B
(`bench_sumpool_simd_ab`, odometer vs SIMD):

| shape | f64 odo→simd | f32 odo→simd |
| --- | --- | --- |
| [8,112,112,64] 3x3/s2 | 50.0→2.23 ms (**22.4x**) | 50.6→0.95 ms (**53.5x**) |
| [8,56,56,128] 2x2/s2 | 14.1→0.51 ms (**27.8x**) | 15.2→0.36 ms (**41.8x**) |

vs JAX (f64 3x3 1.44ms, f32 3x3 0.42ms, f64 2x2 0.48ms, f32 2x2 0.25ms): f64 2x2 ~1.06x (parity), f64
3x3 ~1.5x, f32 ~1.4-2.3x — from the original ~35-120x LOSSES. f32 stays ~2x vs JAX because bit-identity
to the odometer forces f64 accumulation (f64x8, half the width of JAX's f32x16 accum) — we are MORE
accurate; matching JAX's f32-accum would break the *_matches_generic contract. Bit-identical:
`sumpool_simd_channel_bit_identical` (vs the REAL eval_reduce_window_dense_float, f64+f32, finite/NaN,
VALID/padded, threaded) + full fj-lax lib 1590/0. Both maxpool AND avg/sum pooling now win/near-parity
(channel-last); the whole common CNN pooling surface is no longer a JAX loss.

bf16/f16 PARTIAL (2026-06-22, SlateHarrier): extended channel-last pooling to bf16/f16 (max/min/sum) by
REUSING the f64 SIMD fns on the widened f64 view + rounding back — bit-identical to the odometer
(`half_pool_simd_channel_bit_identical` vs the real dense_float, bf16+f16, max+sum; lib 1591/0). But it's
only a MODEST 2.4x win: bf16 maxpool 84→**34.6ms**, sumpool →**34.2ms** (vs JAX bf16 2.80/2.96ms = still
~12x LOSS, narrowed from ~30x). ROOT CAUSE: the win is bottlenecked by the SCALAR per-element widening
(`reduce_window_dense_f64_view` converts 6.4M bf16→f64 one-at-a-time, ~32ms) BEFORE the fast (~2ms) SIMD
pool. Real parity needs SIMD bf16→f32 widening (a bit-shift `(bits as u32)<<16`, vectorizable) inline in
a dedicated half kernel + f32/f64 accum — FILED as a bead. Kept the 2.4x (bit-identical, real) meanwhile.

bf16 widen+round SIMD (2026-06-22, SlateHarrier): SIMD'd the bf16 widen (`widen_bf16_slice_to_f64` via
`bf16_widen8`, in `reduce_window_dense_f64_view` — helps ALL bf16 reduce/pool callers) AND the output
round (`round_f64_slice_to_bf16` via `bf16_round8`, NaN-chunk scalar fallback). Bit-identical (half_pool
guard + lib 1591/0). bf16 pooling 34→**26ms** (~1.3x); the round SIMD gave ~0. ROOT BOTTLENECK is now the
**51MB intermediate f64 materialization** (widen writes it, pool re-reads it) — JAX fuses with no
intermediate, so even SIMD widen+round can't reach parity through a separate-pass widen. PARITY needs a
DEDICATED bf16 kernel that widens bf16→f32 PER TAP inline (no f64 intermediate) — `qvkxp` updated. LESSON:
"widen-to-f64 + reuse the f64 kernel" is bounded by the intermediate's memory traffic; fused per-tap
widening is required to match XLA on narrow dtypes. Shipped the reusable SIMD widen/round helpers (real
bit-identical general bf16-reduce speedup) meanwhile.

FUSED bf16 kernel SHIPPED → DOMINATION (2026-06-22, SlateHarrier, closes qvkxp): `reduce_window_simd_channel_bf16`
widens bf16→f32 (max/min) / →f64 (sum) INLINE per tap (running max/min kept AS bf16 bits — exact since
max/min of bf16 is bf16; sum accumulates f64, rounds once via `round_f64_slice_to_bf16`), threaded across
output blocks. NO intermediate (reads only the half-size bf16 input). bf16 maxpool **26→0.70ms**, sumpool
**26→0.93ms** vs JAX bf16 2.80/2.96ms = **~4.0x / ~3.2x WIN** (flipped from ~12x LOSS; ~30-37x Rust-side).
Bit-identical (`half_pool_simd_channel_bit_identical` vs the real odometer, max+sum, finite+NaN; +
`bench_fused_bf16_maxpool_proto` NaN check; full lib 1591/0). The 2 prior bf16 turns (widen/round helpers
+ the guard) were the foundation; the fused kernel is the payoff. CONFIRMS the lesson: narrow-dtype parity
needs fused per-tap widening, not a materialized widen. f16 stays on the widen→f64 path (rarer). The whole
common CNN pooling surface (max+avg, f64+f32+bf16, channel-last) now WINS/dominates JAX.

f16 widen-SIMD (2026-06-22, SlateHarrier): SIMD'd the f16 widen too (`widen_f16_slice_to_f64` via
`f16_widen8` + `f16_input_needs_scalar` scalar fallback for inf/NaN/subnormal; bit-identical, reduce_window
44/0 + half_pool guard). But only **~1.13x** (f16 pooling 34→**30ms**, still ~10x JAX): f16 pooling is
round/intermediate-bound and the f16 ROUND is still scalar (only bf16 got `round_f64_slice_to_bf16` SIMD).
The real f16 win needs the FUSED kernel — filed `fused-f16-pool-pthzx` — which is HARDER than bf16: f16
widen/round are IEEE (not bit-shifts), and the MAX bit-select hits a ±0 parity wall (max(-0,+0): JAX→+0,
bit-select keeps -0) + NaN canonicalization. Shipped the clean widen helper (modest but bit-identical)
meanwhile. LESSON: the bf16 fused win does NOT transfer cheaply to f16 — IEEE half's special-case parity
(±0/NaN/subnormal) makes the fused kernel substantially more intricate; deferred as LOW priority (f16 rare).
## 2026-06-22 - global avg/sum pooling (axis-reduce, NOT reduce_window) 4-7x JAX loss; threaded 1.51x (SlateHarrier)

Global average pooling done as `jnp.mean/sum(x, axis=(1,2))` on NHWC is the Reduce primitive (not
ReduceWindow): reduce middle axes {1,2} keeping the contiguous channel C — the `inner != 1`
`contiguous_reduce_block` path (reduction.rs ~1656), which folded `out_row[c] op= widen(in_row[c])`
SERIAL. Measured [8,112,112,64] sum axes{1,2}: fj-lax f64 1.70ms / f32 0.93ms vs JAX 0.42 / 0.14ms =
~4x / ~6.7x LOSS. THREADED across `outer` (each outer slice independent, per-outer ascending-reduce
order preserved → bit-identical for sum AND max/min; reduce tests 137/0 + full lib 1591/0). SAME-BINARY
A/B: f64 serial 1.94 → threaded 1.29ms = **1.51x** (real, not contention noise). Shipped the bit-identical
1.51x (also speeds larger-batch middle-axis reduces). NOTE: threading the REDUCE DIM would break
float-sum bit-identity (non-associative) — only `outer` is safe to split.

5y9jg CLOSED — explicit SIMD is a NO-WIN (2026-06-22, SlateHarrier): SAME-BINARY A/B
(`bench_global_avg_pool_reduce`) of explicit f64x8 inner-reduce vs the generic closure: SIMD **2.10ms**
vs closure **1.99ms = 0.94x** (slightly SLOWER). CORRECTS the earlier "the closure doesn't autovectorize"
guess — it DOES autovec for f64 sum, so explicit SIMD adds nothing. The residual ~2.4x JAX (threaded
1.0ms vs 0.42ms) is DRAM-BW-bound: bit-identity caps parallelism at `outer` (=8 here), and XLA reorders
the sum to use more BW. Not improvable on-host while bit-identical. Lesson (reinforces the cumsum
finding): these contiguous `out[c] op= in[c]` reduce/scan loops DO autovectorize — explicit SIMD ≈ 1.0x;
don't re-attempt SIMD on them (the gap, if any, is BW/parallelism, not vectorization).

## 2026-06-22 - gemv (matrix-vector) near-parity ~1.27x; BW-bound, NOT fma-gated (SlateHarrier)

Probed gemv [4096,4096]@[4096] (`bench_gemv`): fj-lax `matmul_2d(A,m,k,x,1)` = **3.35ms (38 GB/s)** vs
JAX **2.63ms (49 GB/s)** = ~1.27x loss. BOTH BW-bound (read A=128MB once). Notable: gemv is NOT fma-gated
(unlike matmul — it's bandwidth-limited, fma throughput is irrelevant), so it's a legit non-`cntiy` lever.
matmul_2d handles N=1 reasonably (38 GB/s, not catastrophic), but JAX's 49 GB/s shows ~parity is
achievable. A dedicated gemv (SIMD row·x dot, threaded, contiguous A stream) could close it — filed
`dedicated-gemv-h36uj` — but HAZARD: matmul_2d accumulates each k-sum in KC-blocked order; a simple
ascending-k dot would break the matmul goldens (must match the order OR confirm dot parity is tolerance).
LOW priority (modest 1.27x, BW-bound, matmul-internals-fragile). Did not attempt (risk > 1.27x gain).

ROOT-CAUSE (2026-06-22, SlateHarrier) — the 1.27x is the BIT-EXACT scalar-K cost at N=1, NOT a missing
optimization. matmul_2d SIMDs across N (the NR columns) with ASCENDING-K scalar accumulation per output
→ bit-identical to naive ijk (`matmul_2d_ikj_bit_identical_to_ijk`, and the packed+KC-blocked path is
"BIT-FOR-BIT identical"). At N=1 (gemv) there is NO N axis to vectorize, so it runs scalar-K (the 38
GB/s). A *fast* gemv would have to SIMD across K — but a SIMD K-reduction REORDERS the sum (lane-interleaved
≠ ascending) and would break matmul's bit-exact `matches_generic` guards. So gemv is **bit-identity-blocked**
(same class as the cummax-scan `jax_max` hazard): a clean fast gemv requires dot_general parity to be
TOLERANCE + relaxing the internal bit-exact matmul guard for the gemv path — a maintainer/scope call.
Downgraded `dedicated-gemv-h36uj` to very-low. LESSON: matmul's N-SIMD strategy makes N=1 (gemv) inherently
scalar-K under bit-exactness; the gap is structural, not an oversight.

## 2026-06-25 - SPECIAL FUNCTIONS are a real non-fma JAX LOSS (2.0–2.5x) — NEW LEVER, surfaced (SlateHarrier)

First genuine contained, non-fma, non-owned JAX loss found in many turns. MEASURED 16M f64 (JAX 0.10.2 CPU,
restored venv vs `bench_special_fns_throughput`):
  • lgamma  fj-lax **41.7ms** vs JAX 19.9ms = **2.1x slower**
  • digamma fj-lax **34.4ms** vs JAX 17.3ms = **2.0x slower**
  • i0e     fj-lax **55.3ms** vs JAX 21.7ms = **2.5x slower**
(Aside: JAX `gammainc`/igamma is 23.9 SECONDS for 16M — fj-lax dominates that one hugely; not a lever.)
ROOT CAUSE (read the impls): `lgamma_approx` evaluates the Lanczos series as PARTIAL FRACTIONS — ~15
DIVISIONS per element (`coeff / (z + idx)` in a loop) + 2 `ln`s; `digamma_approx` runs a per-element
RECURRENCE LOOP (`while shifted < 8.0 { result -= 1/shifted; ... }`) — up to ~8 divisions. JAX uses
division-free rational (Horner P(z)/Q(z), ONE division) minimax forms, fully SIMD-vectorized. These paths are
COMPUTE-bound, NOT fma-gated (the polynomial is plain mul/add/div), NOT memory-bound — so this is a legitimate
contained lever. ATTACK PLAN (deferred — accuracy-critical, must hold the goldens' ~1e-13 vs in-repo
references, so NOT to be rushed): (1) replace the partial-fraction Lanczos with a rational Horner form (Cephes/
Boost coefficients) → ~15 div → ~30 mul/add + 1 div; (2) or SIMD-8-wide the division-heavy path for x≥0.5
(vdivpd is NOT fma-gated; the `ln` stays scalar or needs the fma-gated SIMD-poly log). Filed as bead
`frankenjax-special-fn-rational-3gsc5`. Bench `bench_special_fns_throughput` left in as the A/B baseline.
NOTE: erf is cross-crate (excluded here).

CORRECTION (2026-06-25, SlateHarrier — IMPLEMENTED + REVERTED): built `lgamma_simd8` (Lanczos divisions
8-wide vdivpd, scalar `ln`s + scalar combine + scalar x<0.5 fixup), made it BIT-IDENTICAL to `lgamma_approx`
(needed a SHARED `(z+0.5)*ln(t)-t` combine fn — separate compile sites fma-contracted differently → 1-ULP
divergence at x=1.53; sharing the fn fixed it, lgamma tests 7/0). But SAME-BINARY A/B: scalar 55.56ms vs
SIMD **65.81ms = 0.84x — a REGRESSION**. LESSON: my "divisions dominate" premise was WRONG — the **scalar
`ln`s dominate** (8 div ≈ 1/3 of the work), and SIMD-ing only the divisions adds lane extract/reinsert +
per-lane branch overhead that EXCEEDS the saved division time. The full ~2x parity needs SIMD `ln` — which is
fma-gated (cf. SIMD-poly exp: 2.2x WITH fma / 0.79x WITHOUT). So the special-fn gap LARGELY FOLDS INTO `cntiy`
(+fma), like matmul/conv/exp; the rational-Horner reform (8 div→1) is also `ln`-capped and won't reach parity
alone. REVERTED to scalar (arithmetic.rs unchanged). Bead downgraded: needs +fma SIMD-ln, not a quick win.

## 2026-06-25 - hardware-math ops: sqrt/rsqrt BEAT JAX; reciprocal gap is eval-model, not a kernel lever (SlateHarrier)

Measured the hardware-math class vs JAX 0.10.2 (16M f64, `bench_hardware_math_throughput`):
  • sqrt   fj-lax **21.7ms** vs JAX 23.8ms — fj-lax WINS ~1.1x
  • rsqrt  fj-lax **21.7ms** vs JAX 28.4ms — fj-lax WINS ~1.3x
  • reciprocal fj-lax **20–25ms** vs JAX **14.0ms** — fj-lax LOSES ~1.5x
fj-lax sits at a ~20–22ms floor for all three (compute-bound on sqrt/rsqrt where it beats JAX; but for the
pure-BW reciprocal, JAX hits 14ms = raw bandwidth and fj-lax does not). Probed the cause: reciprocal routes
through `eval_unary_elementwise` (lib.rs) — A/B vs the threaded `eval_unary_elementwise_parallel` path was
**23.4 vs 24.9ms = 1.06x (~0-gain)**, so it is ALREADY effectively threaded; rerouting is not the lever
(reverted the probe). The residual reciprocal gap is STRUCTURAL: every fj-lax eval ALLOCATES a fresh output
`Vec` per call (`vec![0.0; n]` + first-touch page faults), while JAX's jit REUSES output buffers — so a
pure-BW op pays fj-lax's per-call alloc/fault that JAX amortizes. That is an eval-MODEL gap (buffer-pool /
in-place dispatch — architectural, akin to so4wo), NOT a contained per-op kernel lever. NOT pursued. (Good
news: on compute-bearing unary ops fj-lax already MATCHES/BEATS XLA.)

VECTORIZATION RULED OUT (2026-06-25, SlateHarrier — implemented + REVERTED): hypothesized the reciprocal gap
was the scalar closure map failing to auto-vectorize the division (scalar `divsd` vs `vdivpd`). Built
`eval_reciprocal` (explicit 8-wide `Simd::splat(1.0)/xv`, bit-identical, reciprocal tests 2/0) and routed the
dispatch to it. SAME-BINARY A/B bounced **1.11x then 0.88x across two runs = ~0-gain (NOISE)** — both scalar
and SIMD sit at the same ~25ms floor. So the division DOES vectorize enough; the residual gap is purely the
eval-MODEL per-call output alloc/faults, NOT vectorization. REVERTED fully (lib.rs + arithmetic.rs == HEAD).
The reciprocal/cheap-BW lever is confirmed so4wo-class (buffer reuse), with zero contained kernel headroom.

EXTENDED SWEEP (2026-06-25, SlateHarrier — closes the binary/intpow class): measured more JAX 0.10.2 16M f64:
intpow x³ 17.4ms / x⁸ 13.1ms, maximum 14.2ms, clip 11.6ms (all BW-bound → same eval-model alloc gap),
remainder JAX 30.7ms vs fj-lax **37.2ms (~1.2x)** — scalar `fmod` (NO SIMD fmod on x86, both scalar) + the
per-call alloc, so no algorithm lever, floor_divide is a composite (not a fj-lax primitive). CONSOLIDATED
vs-JAX characterization (this arc, venv-restored, ~9 op classes measured): EVERY fj-lax op class is now
classified — (a) PARITY/WIN: sqrt/rsqrt, sort/scan/top_k (dominate), scalar-bound fmod/idiv, compute-bound
SIMD-no-fma; (b) FMA-GATED (`cntiy`): exp/log/sin/cos/tan/pow/erf/lgamma-digamma-lns/matmul/conv/softmax/FFT-
butterflies; (c) EVAL-MODEL-bound (so4wo-class buffer reuse): the pure-BW cheap ops (reciprocal/maximum/clip/
add/mul — fj-lax allocs+faults a fresh output per eval, JAX reuses). The TWO remaining levers are both
ARCHITECTURAL/gated: **+fma (`cntiy`)** and an **output-buffer-reuse eval model**. No contained per-op kernel
lever remains on this host.

## 2026-06-25 - cummax 1.21x JAX LOSS — ASSOCIATIVE parallel-scan is a real contained lever (NEW) (SlateHarrier)

`bench_cummax1d_vs_jax` (16M f64 single chain): fj-lax cummax **82.3ms vs JAX 68.1ms = 1.21x slower**; cummin
75.1 vs 72.1ms = ~parity. KEY: fj-lax cummax is ~4x slower than its OWN cumsum (21ms) — the cost is the
per-element NaN-aware `jax_max` (total_cmp branch), not the scan structure; and the 1-D chain is sequential
(single-thread). UNLIKE cumsum (non-associative, tolerance-locked), **max/min ARE associative** — so a
PARALLEL prefix-scan is BIT-IDENTICAL (2-pass chunked: per-thread local cummax + chunk-max, prefix-max the
chunk carries, re-apply) and would parallelize the slow jax_max → beat JAX. A branchless total-order max is a
second (sequential) lever. This is a GENUINE contained, non-fma, bit-identical lever (the first non-gated one
in a while — distinct from cumsum's tolerance-lock and select's eval-model BW). Deferred to a focused turn
(parallel-scan + jax_max-associativity accuracy verification — not to rush at depth). Recorded measured loss +
the lever; kept `bench_cummax1d_vs_jax`. Bead-worthy follow-up: parallel associative scan for cummax/cummin.

UPDATE (2026-06-25) — IMPLEMENTED + production-FAILED + REVERTED. Built `parallel_cummax_f64` (the 2-pass
chunked associative scan, direct `jax_minmax_scalar`, all-cores). STANDALONE it validated **bit-identical +
29ms** (vs sequential 64ms; would beat JAX 68ms 2.3x). But routed through the PRODUCTION eval path
(`eval_primitive` → `eval_cumulative_dense`), cummax measured **73.6ms** and cummin **89ms (REGRESSED from
75ms)** — the standalone 29ms did NOT reproduce. The discrepancy (standalone 29 vs production 73-89) is
unexplained (eval-pipeline overhead, or the standalone bench was cache-optimistic) — but production is the
truth (gather lesson), and it's a regression, so REVERTED (reduction.rs == HEAD; cum tests 48/0). LESSON
(again): validate the lever in the PRODUCTION op, not a standalone proxy — a standalone bench can show a win
the real eval path doesn't deliver. cummax stays a ~1.21x loss; the parallel-scan lever is NOT realized
through the current eval path. Bead `frankenjax-parallel-cummax-scan` downgraded (needs the 29↔73 root-cause
first, not just the algorithm).

ROOT-CAUSE PROFILED (2026-06-25, `bench_cummax_profile_scan_vs_eval`): the 2-pass parallel scan run DIRECTLY
on the tensor's f64 slice = **29.9–37ms** (matches the standalone — it is genuinely fast + bit_identical),
but `eval_primitive(Cummax)` on the SAME data = **74.6ms**. So the scan is NOT the bottleneck — there's a
~37–44ms overhead in the cummax EVAL PATH itself. Wiring a dedicated `parallel_cummax_f64` into
`eval_cumulative_dense`'s dense-f64 single-chain branch + `#[inline(always)]` on `jax_minmax_scalar` did NOT
help — `eval_primitive(Cummax)` stayed 74ms (the dense-branch result isn't what 1-D cummax returns; 1-D
cummax routes through a DIFFERENT eval path that the dense-branch wiring doesn't reach). REVERTED (reduction.rs
production == HEAD; kept the profiling bench as evidence; cum 48/0). The real lever is now precisely located:
find the 1-D cummax eval path (NOT `eval_cumulative_dense`'s dense branch) and route it to the fast direct
scan — a structural eval-routing fix, bead'd. cummax stays ~1.21x loss meanwhile.

3rd ATTEMPT (2026-06-25) — BLOCKED, needs a profiler. Re-wired `parallel_cummax_f64` (now `#[inline(never)]`
to defeat the suspected inlining-budget exhaustion when absorbed into the huge `eval_cumulative_dense`) into
the f64 single-chain branch (3820). `eval_primitive(Cummax)` STILL = 70.6ms (scan direct on same slice =
28.6ms). Static reading says the gate (axis_stride==1, total≥CUMULATIVE_PARALLEL_MIN_ELEMS, !reverse,
outer_count==1) is satisfied → my branch runs `parallel_cummax_f64` → `eval_cumulative_dense` returns
`Ok(Some(...))` at 3874 → `eval_cumulative` returns it immediately at 4301 (NO post-pass). So eval SHOULD be
~29ms — yet it measures 70ms across all 3 wirings (plain / all-cores / inline(never)). The 28↔70 gap on the
SAME function called from two sites is now a hard contradiction between code-path analysis and measurement
that I cannot resolve with the available tooling (no flamegraph/perf in this env). Reverted (reduction.rs ==
HEAD; profiling bench kept). VERDICT: cummax parallel-scan is BLOCKED — needs `perf`/flamegraph to see where
the eval path actually spends the 42ms (is the dense branch truly reached? is the codegen 2.5x slower in
context?). Bead held at P3, status BLOCKED-needs-profiler. Not a safe-Rust ceiling — a tooling-gated
diagnosis. Stopping cummax after 3 measured attempts; the loss is small (1.21x) and the contained surface
elsewhere is exhausted.

4th ATTEMPT — **LANDED, the bug was a SHADOWED `primitive`.** A print-probe (`eprintln!` in the dense
single-chain branch keyed on `matches!(primitive, Cummax|Cummin)`) printed 0 hits — which exposed the real
bug: `eval_cumulative_dense` does `let primitive = Primitive::Cumsum;` at the fn top (line 3719, "only used
for stride-overflow error context"), SHADOWING the real-op parameter `cum_primitive`. So all 3 prior wirings'
`matches!(primitive, …)` were ALWAYS FALSE → cummax always fell to the `else` (blocked op-closure scan, 70-83ms);
`parallel_cummax_f64` was NEVER called (the 28ms direct-on-slice was the test calling it directly, bypassing
the dead gate). Fix: match on **`cum_primitive`**. Now `parallel_cummax_f64` (direct inlined jax_minmax, all
cores) fires: **cummax 82→29.06ms, cummin 75→27.49ms**, cum tests 48/0 bit-identical. vs JAX (cummax 68.1 /
cummin 72.1): **cummax 2.34x WIN, cummin 2.62x WIN** — a 1.21x loss flipped to a win. SHIPPED. Bead RESOLVED.
LESSON: when a fast-path gate silently never fires, print-probe the branch — a shadowed/rebound variable reads
fine in static analysis but is a different value at runtime; the "28↔70 contradiction" was this all along.

EXTENDED to f32 (the COMMON dtype — JAX default). `parallel_cummax_f32` (widen-to-f64 accumulate + round-to-f32,
matching `scan_contiguous_f32_lines_from`'s contract; bit-identical because max/min of f32 is EXACT and the
round-trip f32→f64→f32 is lossless): **f32 cummax 42.7→15.74ms (2.42x WIN vs JAX 38.1), cummin 42.9→16.17ms
(2.36x WIN)**, cum tests 48/0. Both flipped a 1.12x loss to a 2.4x win on JAX's default dtype. The associative
parallel-scan lever now covers f64+f32 single-chain cummax/cummin.

2-D cummax map (`bench_cummax2d_vs_jax`, f32 [4096,4096]): **axis1 (rows) 13.07ms vs JAX 31ms = 2.37x WIN**
(already row-threaded); **axis0 (columns) 46.83ms vs JAX 49.3ms = parity** (the streaming leading-axis path is
already competitive — NOT a loss, no lever; threading the ~2.5ms gap isn't worth it). cummax FAMILY COMPLETE:
1-D f64/f32 (parallel-scan, landed 2.3-2.6x), 2-D rows (threaded win), 2-D columns (streaming parity).

## 2026-06-25 - f32 1-D cumsum/cumprod 1.28x JAX LOSS — lever is parity-policy-gated (NOT shipped) (SlateHarrier)

`bench_cumsum1d_f32_vs_jax` (16M): fj-lax cumsum **49.8ms vs JAX 39.0ms = 1.28x**, cumprod **45.8 vs 35.9 =
1.28x**. Root cause: f64 1-D cumsum uses the blocked parallel prefix scan (`blocked_prefix_scan_to_vec`, ~21ms,
TOLERANCE-legal reassociation) but the f32 path (`scan_contiguous_f32_lines_to_vec`) only threads when
outer>1, so f32 single-chain is SEQUENTIAL. UNLIKE cummax/cummin (associative → bit-identical parallel scan,
SHIPPED), **cumsum/cumprod are NON-associative** — a parallel scan CHANGES bits, so this is a tolerance/parity-
policy call. Two implementations: (a) widen f32→f64 + reuse the accepted f64 blocked scan + round → SAFEST
(matches the shipped f64 precedent exactly) but the widen/round passes make it ~parity (~36-39ms, marginal);
(b) direct rescan parallel f32 (pass1 chunk-sums, prefix, pass3 cumsum-from-carry) → ~15ms WIN but the carry
reassociation needs CONFORMANCE verification (large-input goldens), which the small cum unit-tests (sequential
path, <1M) do NOT cover. NOT shipped: won't risk RED main on an unverified f32 cumsum reassociation at depth.
The f64-blocked precedent makes (b) defensible — bead'd for a focused turn that runs `-p fj-conformance`
cumsum goldens first. Recorded measured loss + the lever + the gate; kept `bench_cumsum1d_f32_vs_jax`.

RESOLVED + SHIPPED (2026-06-25). Gate cleared: `cumulative_oracle` + `cummax_cummin_oracle` conformance tests
use TINY inputs (3-6 elems, all << the 1<<20 blocked threshold) → they exercise the untouched SEQUENTIAL path
and stay GREEN (28/0 + 45/0); the parallel path only triggers ≥1M. Added `parallel_assoc_scan_f32` (3-pass
rescan, impl (b)): pass1 per-chunk op-reduction, pass2 exclusive op-prefix (carries), pass3 re-scan each chunk
from its f64 carry (single round per output — only the inter-chunk carry is reassociated, identical structure
to the shipped f64 blocked scan). New parity test `parallel_f32_cumsum_within_tolerance` (4M, exercises the
parallel path) confirms max relative error <1e-5 vs the sequential f64-accumulate reference (actual ~1 f32 ulp).
**f32 cumsum 49.8→15.47ms (2.52x WIN), cumprod 45.8→14.10ms (2.55x WIN)**; cum 49/0, conformance GREEN, clippy
clean. Bead frankenjax-parallel-f32-cumsum RESOLVED.

## 2026-06-25 - REVERSE f32 cumsum/cummax parallel scans — 2x WIN vs JAX (SlateHarrier)

Reverse 1-D scans (common: attention backward) were gated out of the parallel path (`!reverse`) → sequential
LOSS: cumsum_rev **48.8ms vs JAX 28.4 = 1.72x**, cummax_rev **58.7ms vs JAX 30.0 = 1.96x**. Extended
`parallel_cummax_f32` + `parallel_assoc_scan_f32` with a `reverse` param: pass1 local DIRECTIONAL scan / chunk
reduction, pass2 directional carries (forward = op of EARLIER chunks; reverse = op of LATER chunks, via a
reverse prefix loop), pass3 directional rescan from carry. Dropped `!reverse` from the f32 caller gate.
**reverse cumsum 48.8→13.32ms (2.13x WIN), reverse cummax 58.7→15.08ms (1.99x WIN)**. VERIFIED: new test
`parallel_f32_reverse_scans_match_sequential` (4M) — reverse cummax BIT-IDENTICAL to sequential, reverse
cumsum within 1e-5; conformance cumulative 28/0 + cummax 45/0 (small reverse tests <1M still use the
sequential path); cum 49/0, clippy clean. cumulative-scan family now covers f32 forward+reverse,
cummax+cumsum/cumprod.

## 2026-06-25 - REVERSE f64 cumsum/cummax parallel scans — 2-2.5x WIN vs JAX (SlateHarrier)

Completes the family on f64. Reverse f64 single-chain was gated out (the outer dense gate's `!reverse ||
outer_count > 1` excluded reverse single-line, and the f64 cumsum forward uses the forward-only
`blocked_prefix_scan_to_vec`) → sequential LOSS: cumsum_rev **88.4ms vs JAX 57.9 = 1.53x**, cummax_rev
**100.9ms vs JAX 59.4 = 1.70x**. Added `reverse` to `parallel_cummax_f64` + a new `parallel_assoc_scan_f64`
(reverse f64 cumsum/cumprod rescan; forward f64 cumsum keeps the shipped blocked scan); widened the f64 outer
gate to admit reverse single-chain ≥1M. **reverse cumsum 88.4→29.69ms (1.95x WIN), cummax 100.9→24.21ms
(2.45x WIN)**. VERIFIED: `parallel_f64_reverse_scans_match_sequential` (4M) — reverse cummax BIT-IDENTICAL,
reverse cumsum <1e-5; conformance cumulative 28/0 + cummax 45/0; tests 51/0, clippy clean. CUMULATIVE-SCAN
FAMILY COMPLETE: f64+f32 × forward+reverse × cummax/cummin+cumsum/cumprod, all ~2-2.6x vs JAX-on-CPU.

## 2026-06-25 - INTEGER (i64/i32) cumulative parallel scan — 1.9-2.4x WIN vs JAX (SlateHarrier)

i64/i32 cumulative routed through the generic per-Literal BOXED sequential loop. Measured WIN already (i64
cumsum 52.9ms vs JAX 67.1 = 1.27x; cummax 55.6 vs 70.8 = 1.27x) but with big headroom. Added
`parallel_assoc_scan_i64` (one generic fn over `int_op`, forward+reverse) wired into the i64/i32 dense branch
for single-chain ≥1M. Integers are EXACT (wrapping_add associative+commutative, max/min exact) → BIT-EQUAL to
the sequential fold for any op/direction (covers cumsum/cumprod/cummax/cummin; i32 + widened u32/u64 share the
i64 backing). **i64 cumsum 52.9→35.08ms (1.91x WIN), cummax 55.6→29.15ms (2.43x WIN)**. VERIFIED:
`parallel_i64_scan_matches_sequential` (4M, incl mod-2^64 wrapping-overflow data) — cumsum+cummax forward+reverse
ALL bit-equal; conformance cumulative_oracle GREEN; cum 49/0, clippy clean. Cumulative-scan family now also
covers integers.

## 2026-06-25 - 2-D leading-axis (column) scan is SINGLE-THREADED + compute-bound — row-block lever filed (SlateHarrier)

`scan_leading_axis_to_vec` (cumsum/cummax DOWN columns, axis=0) is SINGLE-THREADED (k-outer/col-inner stream,
cols-wide f64 acc). Measured f32 cummax [4096,4096] axis0 = 46.8ms vs JAX 49.3 = parity — but it's COMPUTE-bound
(128MB would be ~13ms memory-bound; the extra ~34ms is per-element jax_minmax single-thread), so threading
would WIN, not just match. LEVER (filed, not yet shipped): row-block parallel prefix scan with a COLS-WIDE
carry — row-blocks are contiguous (safe split_at_mut, unlike column-blocks which are strided). pass1 each block
computes cols-wide column-totals; pass2 directional cols-wide prefix → per-block cols-wide carry; pass3 each
block re-scans from its carry. Bit-identical for cummax (associative), tolerance-legal for cumsum (only the
inter-block carry reassociates — same policy as the 1-D blocked scan). ~3-6x potential (parity→win). Deferred:
parity is not a loss + the cols-wide generic rescan is moderately complex (generic S/T, directional) — not to
rush at session depth. Bead frankenjax-leading-axis-scan-thread. The 1-D cumulative campaign (10 shipped wins,
all dtypes × directions) is the completed deliverable; this 2-D case is the documented follow-on.

SHIPPED (2026-06-25). `scan_leading_axis_to_vec_threaded` (row-slab parallel prefix, cols-wide carry, forward
+reverse, generic S/T) wired into the f64+f32 leading-axis branches (gated rows≥2·threads && total≥1<<20).
**cummax f32 [4096,4096] axis0: 46.8→17.92ms = 2.75x WIN vs JAX 49.3** (was parity). VERIFIED:
`threaded_leading_axis_scan_matches_sequential` (2048×1024, both directions) — cummax BIT-IDENTICAL to the
single-threaded cols-wide stream, cumsum <1e-5; conformance cumulative 28/0 + cummax 45/0; tests 51/0, clippy
clean. Bead frankenjax-leading-axis-scan-thread RESOLVED. The 2-D leading-axis (columns) now WINS, not just
parity — the cumulative-scan family is complete in 1-D (all dtypes×directions) AND 2-D (rows threaded +
columns row-slab-threaded).

## 2026-06-25 - argsort is a ~35x fj-lax WIN vs JAX (SlateHarrier)

`bench_argsort2d_vs_jax`: argsort f64 [2048,2048] axis1 — fj-lax **17.4ms vs JAX 616.8ms = ~35x WIN**.
XLA's CPU argsort is catastrophic (same as its sort 2522ms); fj-lax's threaded radix argsort dominates. No
lever (win); recorded + kept the bench. Completes the sort-family map: sort/argsort/top_k all crush JAX-on-CPU.

## 2026-06-25 - RNG (random_uniform) is a ~3x fj-lax WIN vs JAX (SlateHarrier)

`bench_random_uniform_vs_jax` (16M): fj-lax random_uniform **36.84ms (f64)** vs JAX random.uniform **112.0ms
(f32)** / normal 109.7ms = fj-lax **WINS ~3x** — and that's DESPITE fj-lax producing f64 (more output bytes)
vs JAX's f32. XLA's CPU threefry is slow (like its sort/scan/cumsum); fj-lax's 8-wide SIMD + threaded threefry
dominates. No lever (win); recorded + kept the bench. Confirms the broad pattern: fj-lax WINS the contained
compute-bound CPU surface (sort/scan/cumsum/RNG/reductions all dominate JAX-on-CPU).

## 2026-06-25 - select/where 1.90x JAX LOSS — two levers REJECTED (cap regresses, branchless ~0) (SlateHarrier)

`bench_select_vs_jax` (16M f64, mask from x>0): fj-lax select **28.05ms vs JAX 14.78ms = 1.90x slower**.
select_f64 reads 3 arrays (cond/t/f) + writes 1 fresh output (~386MB). Two levers tried + REJECTED:
(1) cores/2 over-subscription cap → **REGRESSED 28→35ms**: select reads 3 arrays so it wants all-cores BW
(unlike the 2-array convert/data-movement copies where the cap won) — reverted.
(2) branchless bit-blend (`(c as u64).wrapping_neg()` mask instead of `if c {t} else {f}`) → **~0-gain
28→31ms** (LLVM already cmov's the branch; the bit-ops add work) — reverted, select tests 38/0.
(3) UPDATE — the proposed fix (zip-based threaded branchless bit-blend, no bounds checks, all-cores) was
IMPLEMENTED + MEASURED: **~0-gain, 28→27.3ms** (still 1.85x behind JAX). Reverted. So it is NOT a
vectorization/bounds-check problem — fj-lax select hits ~14.3 GB/s vs JAX's ~27 GB/s on the SAME ~386MB
traffic. The residual 1.90x is the **eval-model / BW-saturation gap** (per-call output alloc + first-touch
faults + 3-array read not reaching multi-channel BW) — the SAME class as reciprocal, ARCHITECTURAL (so4wo
buffer-reuse), NOT a contained kernel lever. (Corrects last turn's "actionable SIMD-blend lever" hypothesis —
all three contained attempts (cap / branchless / zip-blend) measured failed.) Kept `bench_select_vs_jax`. PIVOT.

## 2026-06-25 - embedding gather 1.11x JAX loss; cores/2 cap REGRESSES (gather is latency-bound) (SlateHarrier)

Probed embedding/row gather (`bench_embedding_gather_vs_jax`): gather 2M rows from a [200000,128] f32 table —
JAX 158.6ms vs fj-lax **175.6ms = 1.11x slower** (small; ~2GB traffic, eval-model output alloc + random row
reads). Tried extending the over-subscription cap (cores/2) to the row/slice gather path. A standalone copy-
sweep suggested it would help (32t 241.8ms vs 16t 160.3ms). But the PRODUCTION embedding gather REGRESSED:
32t 175.6ms vs the cores/2 cap **212.4ms** — capping HURTS. REVERTED. LESSON: row/slice gather is
LATENCY-bound on the random source-row reads (unlike the pure-copy data-movement ops), so all-cores wins —
more threads hide the read latency. The over-subscription cap is ONLY for PURE-COPY fresh-output ops
(convert/broadcast/transpose/concat/flip/pad), NOT random gather. METHOD LESSON: a synthetic copy-sweep
mismodeled the production access pattern (showed 16<32; production is 32<16) — measure PRODUCTION, not a proxy.
The residual 1.11x gather gap is eval-model (output alloc) + memory-latency, both arch (so4wo), no contained
lever. Left a code NOTE at the gather site + kept the bench.

A genuine CONTAINED, bit-identical win (the first non-gated kernel lever in many turns). Measured dtype
conversions vs JAX 0.10.2 (16M) showed f64-source casts 2–3x SLOWER (f64->f32 19.3ms vs JAX 7.2, f32->f64
32.9 vs 14.7, f64->bf16 12.8 vs 4.6) while f32->bf16 was PARITY. The f64 casts are ALREADY threaded — so not
threading-presence. THREAD-COUNT SWEEP (f64->f32 16M, 2 runs) found the cause: `work_scaled_threads` uses ALL
cores (32 on the bench host) but a pure read+write convert OVER-subscribes the memory/page-fault path —
1t 60ms / 8t 18ms / **16t 13.5ms (optimum)** / 32t 17.2ms. Added `bw_convert_threads(n) =
work_scaled.min((cores/2).max(2))` and routed the three threaded convert downcasts (f64->f32, f64->half,
f32->f64) through it. POST-FIX (16M, bit-identical, convert tests 11/0 + full lib 1601/0 + clippy clean):
f64->f32 19.3->**13.45ms (1.43x)**, f32->f64 32.9->**27.9ms (1.18x)**, f64->bf16 12.8->**7.48ms (1.71x)**.
(Still ~1.6–1.9x behind JAX — residual is the eval-model per-call alloc — but the over-subscription slice is
now reclaimed.) BROADER LEVER: this over-subscription likely affects OTHER BW-bound threaded ops too
(reciprocal/cheap elementwise via `work_scaled_threads`/`dense_unary_threads` at all-cores) — a follow-up to
sweep + cap those (filed mentally; the convert win confirms the mechanism). LESSON: for BW-bound threaded ops
on many-core hosts, all-cores OVER-subscribes; cap at the memory-saturating count (~cores/2).

EXTENDED TO DATA-MOVEMENT (2026-06-25, SlateHarrier — follow-up SHIPPED): confirmed the over-subscription
generalizes via a standalone parallel-copy sweep (16M f64 copy into fresh output, the exact broadcast/
transpose/concat pattern): 8t 30.9ms / **16t 22.3ms** / 32t 33.3ms = **1.49x** (32→16). Renamed
`bw_convert_threads`→`bw_bound_threads` and routed the uncapped data-movement copy/fill sites through it:
transpose_2d_into, eval_broadcast_in_dim (both the scalar-fill `threaded_fill_into` and the replicate
`broadcast_replicate_into` paths), eval_concatenate, `rev_gather_into` (flip/reverse block-copy), and
`eval_pad` (3 threaded fill/row-copy sites — added 2026-06-25). Large broadcast/transpose/concat/flip/pad now use cores/2 (16)
not all-cores (32) → the ~1.49x on the big-copy regime. Bit-identical (thread count only): full fj-lax lib
1601/0 + clippy clean. NOTE: reciprocal/cheap-unary use `dense_unary_threads` which ALREADY caps at 16 at 16M
(ELEMS_PER_THREAD=1<<20) — NOT over-subscribed; their residual gap stays the eval-model per-call alloc. The
over-subscription fix is specific to the `work_scaled_threads` (ELEMS_PER_THREAD=1<<16 → all cores) callers.
DYNAMIC_UPDATE_SLICE checked + at PARITY (2026-06-25): `bench_dynamic_update_slice_vs_jax` op=16M upd=1M f64
= fj-lax 24.96ms vs JAX 24.7ms (1.01x). DUS copies the whole operand single-thread via `to_vec` (no zero-fill)
then overwrites the slice — already matches JAX's full-copy DUS; threading the copy is only ~1.12x (the
zero-fill parallel-copy path) and variance-sensitive → NOT pursued (~0-gain). No lever.

## 2026-06-25 - full reductions: max/argmax BEAT JAX; sum/prod 1.5–2x LOSS, threaded-tree lever is a parity call (SlateHarrier)

BOOL REDUCTIONS also a WIN (2026-06-25, `bench_any_gt_vs_jax`): any(x>0) 16M f64 = fj-lax 5.27ms vs JAX
5.95ms = **fj-lax WINS 1.13x**. fj-lax's `ReduceOr`/`ReduceAnd` operate on the BIT-PACKED bool words (16M
bools = 2MB) with a short-circuit `.any()`, so the reduce itself is negligible and the cost is the threaded
`Gt` comparison's 128MB read — at/above parity with JAX. No lever; recorded as a win.

Measured full reductions vs JAX 0.10.2 (16M f64, `bench_full_reduce_vs_jax`):
  • max  fj-lax **2.64ms** vs JAX 6.69ms — fj-lax **WINS 2.5x** (already threaded `threaded_reduce_minmax_f64`)
  • argmax fj-lax **9.11ms** vs JAX 25.2ms — fj-lax **WINS 2.8x**
  • sum  fj-lax **13.31ms** vs JAX 6.52ms — fj-lax **LOSES 2.05x**
  • prod fj-lax **13.30ms** vs JAX 8.55ms — fj-lax **LOSES 1.56x**
sum/prod are SCALAR SINGLE-THREAD folds (non-associative); max/min already thread (associative → bit-
identical). A threaded TREE sum (each thread folds its contiguous chunk, partials combined) would run like
max (~2.6ms) → **BEAT JAX ~5x** (13.3→2.6). BUT it changes the bits and breaks the committed bit-exact-
sequential semantics — test `dense_f64_reduce_sum_full_fast_path_bit_identical_to_literal_path` + the code
comment "Sum/Prod stay on the scalar fold — bit-exact order matters". KEY: fj-lax's sequential sum CANNOT
bit-match JAX's sum anyway (XLA tree-reduces — different order), so float-sum parity vs JAX is necessarily
TOLERANCE; the bit-exact-sequential lock is a SELF-IMPOSED internal commitment, not a JAX requirement.
Relaxing float sum/prod to a tolerance contract (enabling a threaded tree reduce) is a PARITY-POLICY/scope
call (same class as the gemv `h36uj` bit-identity block). Filed bead `frankenjax-tree-sum-reduce`; NOT pursued
unilaterally (changing reduction semantics + relaxing a bit-exact test is a maintainer decision). `bench_full_
reduce_vs_jax` left as evidence.

## 2026-06-26 - scatter-add binning lever CONFIRMED ALREADY DONE; non-fma surface empirically exhausted (SlateHarrier)

Code-verified `scatter_reduce_range_partitioned` (the scatter-add ~1.25x-JAX path): it ALREADY does the
2-phase stable-bin-then-apply (phase 1: each thread bins its input chunk into per-output-range buckets,
order-preserving; phase 2: each output-range thread applies its buckets) — NOT an O(T·n) re-scan. Inputs are
packed `idx<<32 | i` into u64 (halves bucket footprint, +1.06x over (usize,usize)). So the obvious "bin instead
of re-scan" lever is DONE; the residual 1.25x is the phase-2 scattered `upd[i]` gather + `out[idx]` write
latency (memory-bound — storing (idx,val) to skip the gather doubles the memory-bound phase-1 build, already
rejected). Do NOT re-propose binning for scatter-add. CONSOLIDATION (this arc, all MEASURED not asserted):
the obvious "dig past the ceiling" algorithmic levers were empirically tested and REGRESS — gather sort-scatter
0.07x, leading-cumsum transpose 0.22x, complex-cumsum threading 0.95x; the transpose/threading trick wins ONLY
for compute-bound strided ops (sort 4.64x, reduce_window 4.24x/3.21x, half cumsum 4.47x — all shipped), which
are now mined; scatter-add binned, gather HW-optimal. The non-fma/on-host/unowned surface is exhausted; every
remaining JAX LOSS routes to `cntiy` (+fma maintainer decision) or owned multi-session (eigh/SVD LAPACK-gap,
so4wo compiled-jaxpr). sort/cumsum/einsum DOMINATE JAX (CPU-sort/scan catastrophe). UPDATE: even COMPLEX
sort is a domination — `bench_complex_sort_single_thread` measures fj-lax single-thread complex128 sort
[2048,2048] axis1 = **280ms vs JAX 602ms = 2.15x WIN** (single-thread, before any threading). SHIPPED threading the
generic/complex sort fallback (the last remaining clean compute-bound code lever): the generic single-tensor
sort path (`sort_along_axis`, sort_key + compare_sort_keys_nan_last — the complex / non-radix dtypes) was
single-thread; added inter-slice threading of the contiguous (last-axis) case (lines independent, per-thread
reused `indexed` scratch, per-thread Result for the `?`, gated `total >= SORT_PARALLEL_MIN_TOTAL_ELEMS`).
SAME-BINARY A/B `bench_complex_sort_threaded_ab` complex128 [2048,2048] axis1: serial-ref 361ms vs threaded
**137ms = 2.63x** (~2.0x vs the prior serial eval_primitive 280ms). Bit-identical (sort tests 27/0 + full lib
1599/0 + clippy clean). vs JAX 602ms the domination went 2.15x→4.4x. PITFALLS hit + fixed: (1) gated on
`use_parallel_radix` first (that's INTRA-slice, outer_count==1 only) so threading never engaged — use the
inter-slice `SORT_PARALLEL_MIN_TOTAL_ELEMS` gate; (2) allocating `indexed` per-line made it a net loss — reuse
per-thread. UPDATE: also threaded the MULTI-operand sort (`sort_multiple_along_axis`, lax.sort/lexsort with
key+value operands) the same way — inter-slice threading of the contiguous case, each thread owning disjoint
contiguous blocks of every operand's output vec via per-operand `chunks_mut`, per-thread reused `indexed`.
CLEAN SAME-BINARY A/B `bench_multi_sort_threaded_ab` (key f64 + val i64) [2048,2048] axis1, serial-ref
replicates the eval serial inner loop (per-element `Vec<SortKey>` + compare_sort_key_tuples): serial-ref
716ms vs threaded **422ms = 1.69x**. (NOTE: an alloc-free manual sort is ~150ms — the multi-sort path is
ALLOC-BOUND on the per-element `Vec<SortKey>`.) FOLLOW-UP (shipped): replaced the per-element `Vec<SortKey>`
with a FLAT reused keys buffer (`keys_flat` of axis_dim*num_keys + `order: Vec<usize>`, comparison reads each
row's contiguous key span) in the threaded path. 3-WAY SAME-BINARY A/B (the only trustworthy comparison —
serial-Vec varied 716↔311ms ACROSS invocations): serial-Vec 311ms / serial-flat 232ms / threaded-flat 235ms
→ **alloc-fix = 1.34x, threading-on-flat = 0.99x (NEUTRAL), total = 1.32x**. KEY LESSON: the ALLOCATION was
the real bottleneck, not single-threading — once the per-element Vec is gone the multi-sort is MEMORY-bound
(multi-operand scatter) and threading adds nothing (0.99x). Last turn's "1.69x from threading" was really
threading masking the alloc cost; the flat buffer is the genuine lever. Kept the (now-neutral) threading
since it's not a regression and reverting it is pure churn. Bit-identical: sort 27/0 + full lib 1599/0 +
clippy clean. The sort-threading family is COMPLETE (radix + generic/complex single + MULTI-operand + top_k,
all axes). Last contained compute-bound lever; remaining = `cntiy` +fma / owned.

## 2026-06-25 - alloc-first sweep of fj-lax hot paths: NO further material lever (alloc-lean) (SlateHarrier)

After the multi-sort flat-keys win, swept ALL fj-lax numeric hot paths (elementwise / reduction / matmul /
dot_general / conv / gather-scatter / cumulative) for the per-element/per-iteration heap-alloc anti-pattern
(agent-backed). Only ONE candidate surfaced: `batched_matmul_row_block_bf16_in` (tensor_contraction.rs) — the
bf16 GEMM ROW-REMAINDER path allocates `acc = vec![0f32; n]` per remainder row. ISOLATING micro-A/B
`bench_rowremainder_acc_alloc_vs_reuse` (n=512 k=512, ALL 8000 rows forced through the remainder path — worst
case): per-row-alloc 165.85ms vs reused 153.81ms = **1.078x**. In a real matmul the remainder is only
`tile_rows % 4` rows (≤3) — the rest go through the alloc-free register-blocked loop — so the actual impact is
~0; and even the worst-case 7% is compute-dominated (the alloc is ~1/k of each remainder row's k×n axpy). NOT
worth a production change (REVERT ~0-gain). CONCLUSION: fj-lax hot kernels are alloc-lean; the multi-sort
per-element `Vec<SortKey>` was the lone material alloc lever (fixed). The alloc-first lens is now exhausted.

## 2026-06-26 - gather sort-gather-scatter EMPIRICALLY REJECTED — 0.07x (random gather is HW-optimal) (SlateHarrier)

The scorecard frames gather's ~15x JAX gap as the "Zen3 vgather ceiling" — but that was the SIMD approach.
Tested the obvious NON-SIMD algorithmic dig: sort the gather indices so the input reads become SEQUENTIAL
(the 128MB random-read stream looked like the bottleneck), then scatter-back to the output. SAME-BINARY A/B
f64 [4M from 16M, random idx, input 128MB > L3]: **direct random-read 9.03ms vs sort-gather-scatter 127.32ms
= 0.07x — a 14x REGRESSION.** The radix sort (4M pairs) + the random scatter-WRITE-back vastly exceed the
direct random-READ cost: Zen3's OOO + hardware prefetch already service random reads at ~2.25 ns each, so the
direct `gather_single_dense` is NEAR-OPTIMAL. The JAX gap is genuinely the hardware `vgather`/prefetch
microarchitecture (XLA emits it), NOT a missing algorithmic lever — and `gather_or` SIMD was already a no-win
(olm4p). Probe reverted. Do NOT re-attempt sort-gather-scatter (or SIMD) for random gather — measured dead.

## 2026-06-26 - non-last-axis SORT via transpose→threaded-row-sort→transpose — 4.64x (SlateHarrier)

The radix sort threads only the contiguous (last-axis) case; a NON-last axis (e.g. column sort, axis 0)
fell to a SINGLE-THREAD strided radix (cache-hostile per-line strided gather). Added a `sort_along_axis`
wrapper: if `axis != rank-1`, transpose the sort axis to last → recurse into the threaded contiguous fast
path → transpose back (the swap permutation is its own inverse). Bit-identical (sort exact, transpose exact
data movement; argsort indices are positions along the sort axis, preserved by the transpose). SAME-BINARY
A/B column sort f64 [4096,4096] axis0: **strided-1thread 513.64ms → transpose-wrapper 110.76ms = 4.64x**.
(JAX CPU sort is catastrophically slow — axis0 **2522ms** — so fj-lax already dominated 4.9x; now 22.8x.)
Covers ALL dtypes + sort/argsort (the wrapper is dtype-agnostic, above the radix dispatch). Bit-identical:
sort tests 27/0 + full lib 1599/0 + clippy clean. NOTE: JAX sort is a domination, not a loss — this is an
internal single-thread-strided corner fixed (transpose reuses the dominating row-sort), not a JAX-gap close.

TRANSPOSE-TRICK IS COMPUTE-BOUND-ONLY (2026-06-26, SlateHarrier): tried the same transpose→threaded-scan→
transpose on the leading-axis (axis0) f64 CUMSUM. SAME-BINARY A/B [4096,1024] axis0: scan_leading (current,
single-thread) 16.89ms vs transpose+threaded **77.36ms = 0.22x** — a 4.5x REGRESSION. cumsum is MEMORY-bound,
so the 2 extra transpose passes (cache-hostile element scatter) swamp the threaded-scan benefit. REVERTED.
RULE refined: the transpose→contiguous-fast-path trick wins ONLY for COMPUTE-bound strided ops (sort: 4.64x);
for MEMORY-bound ops (cumsum, complex cumsum) it regresses — do NOT transpose-wrap memory-bound scans. This
closes the leading-axis cumsum question (no safe-Rust lever: strided-output blocks direct threading, transpose
regresses, SIMD/closure are ~1.0x; it's memory/structural — the `fgk6m` perf-counter lead).

## 2026-06-25 - half (bf16/f16) trailing cumsum/cumprod/cummax THREADED — 4.47x (dtype-threading gap) (SlateHarrier)

The cumulative scan was threaded for F64/F32/I64 (scan_contiguous_lines) but the BF16/F16 branch of
`eval_cumulative_dense` was SINGLE-THREAD over its independent lines — a dtype-threading gap. Threaded the
contiguous (trailing-axis, `axis_stride==1`) case: extracted the per-line widen→f64-accumulate→round-to-half
scan into a closure and threaded over the independent lines (each line `out[outer*axis_dim..]` is a contiguous
disjoint block → safe `split_at_mut`, no `unsafe`). Bit-identical (sequential acc dependency WITHIN a line is
preserved; lines independent). SAME-BINARY A/B bf16 [512,4096] axis1: **serial 7.14ms → threaded 1.60ms =
4.47x**. Bit-identical: cumulative tests 48/0 + full lib 1599/0 + clippy clean. The cumulative-scan threading
is now complete across dtypes for the trailing axis (f64/f32/i64/bf16/f16). UPDATE 2026-06-26: tried the
COMPLEX cumulative sibling — REGRESSES. SAME-BINARY A/B complex128 [512,4096] axis1: serial 20.01ms vs
threaded **21.09ms = 0.95x**. Unlike the compute-bound half scan (decode+round per elem → 4.47x), complex
cumsum is MEMORY-bound (16-byte (f64,f64), cheap add; the in-place scan reads+writes 32MB + the src.to_vec
clone) — threading adds spawn/split overhead with no BW headroom. REVERTED; complex cumulative kept serial
(annotated in code). LESSON: thread a scan ONLY when it is compute-bound per element (half-float decode/round,
transcendentals); memory-bound scans (complex, plain f64) regress or break even. Leading-axis scans stay
strided-output-blocked. Cumulative-scan threading is DONE (compute-bound dtypes threaded; memory-bound left serial).

## 2026-06-25 - generic dense_float reduce_window THREADED — 3.21x (sum + dilated max/min path) (SlateHarrier)

Sibling of the deque-threading: the GENERIC `eval_reduce_window_dense_float` (the fallback for SUM, dilated
max/min, and non-channel-last/non-separable reduce_window) was single-thread over `total_output` independent
output cells. Refactored the per-cell window reduction into a `cell(out_idx)` closure and threaded the fill
across contiguous output ranges (each thread recovers its starting odometer coords from the flat start;
ranges are disjoint → safe `split_at_mut`, no `unsafe`). Each cell scans its window in the SAME fixed tap
order, so bit-identical. SAME-BINARY A/B sum [512,4096] w=17: **naive-1thread 16.21ms → threaded 5.05ms =
3.21x**. Bit-identical: `reduce_window_dense_float_threaded_matches_naive` (threaded path vs naive per-cell
sum) + window tests 49/0 + full lib 1599/0 + clippy clean. With the deque threading (prior entry), the whole
reduce_window family is now threaded: fused channel-last (small window), separable deque (max/min large
window), and generic dense_float (sum + dilated + the rest).

## 2026-06-25 - reduce_window large-window max/min deque THREADED — 4.24x (bead od11p flipped) (SlateHarrier)

The separable monotonic-deque sliding max/min (`reduce_window_extremum_1d_axis_f64`) — the large-window /
last-axis reduce_window path (small channel-last windows take the fused SIMD path) — was SINGLE-THREAD over
its `outer*inner` independent lines, the documented ~1.4x JAX loss (`od11p`). Threaded it by the OUTER dim:
for a fixed `o`, the output block `[o*out_ax*inner, (o+1)*out_ax*inner)` is CONTIGUOUS + disjoint, so
`split_at_mut` partitions cleanly (no strided cross-thread writes, no `unsafe`); each thread runs its own
scratch deque. SAME-BINARY A/B [256,4096] w=65 s=1 max: **serial-deque 5.37ms → threaded-deque 1.27ms =
4.24x**. Since the serial deque was ~1.4x JAX-slower, 4.24x threading flips this path to a WIN vs JAX.
Bit-identical (lines independent, same per-line deque, contiguous disjoint output):
`reduce_window_deque_threaded_matches_naive` (max+min × 3 window/stride/pad configs, threaded path, vs a
naive O(out·window) tap reference) + window tests 48/0 + full lib 1598/0 + clippy clean. NOTE: SIMD-of-the-
deque (od11p's literal ask) stays hard (data-dependent push/pop); threading closed the JAX gap instead.

## 2026-06-25 - cumsum 2D already DOMINATES JAX (not a loss); f64 leading-axis BW-bound (SlateHarrier)

1-D scan also a WIN (2026-06-25, `bench_cumsum1d_vs_jax`): 16M f64 single-chain — JAX cumsum **95.0ms** /
cumprod **94.2ms** (XLA has NO fast single-chain CPU scan), fj-lax cumsum **21.2ms (4.5x WIN)** / cumprod
**41.3ms (2.3x WIN)** via the latency-bound sequential fold. So the non-associative scan being sequential
(bit-exact-locked) is NOT a loss on CPU — JAX's own 1-D scan is far slower. No lever; recorded as a win.

Probed cumsum as a different primitive. JAX CPU cumsum is slow (sequential): [4096,1024] f64 ax0 **20.85ms**,
ax1 18.28ms; f32 ax0 6.93ms, ax1 2.96ms. fj-lax: f64 ax0 **13.34ms (1.56x WIN)**, ax1 3.60ms (5.1x), f32 ax0
**1.04ms (6.7x WIN)**, ax1 1.03ms (2.9x). So cumsum is a DOMINATION across the board — not a JAX loss to close.
The f64 leading-axis (axis0) path (`scan_leading_axis_to_vec`, single-thread, rows-sequential / cols-autovec)
is the slow corner internally; SAME-BINARY A/B showed a hand-written direct f64 loop == the generic-closure
version (**1.01x** — the closure is NOT the bottleneck; it is BW + the per-column f64 dependency chain).
Threading it would only EXTEND a domination, and safe-Rust can't thread the leading-axis scan without an extra
assembly pass (column-blocks write strided-disjoint columns of a row-major output — no safe split_at_mut) that
eats the gain. NO-SHIP (probe reverted ~0-gain). FLAGGED for future profiling: f64 ax0 (13.34ms) is ~13x
slower than f32 ax0 (1.04ms) for only 2x the bytes — an unexplained f64-specific dtype-perf anomaly in the
leading-axis column scan, worth a `perf`-counter look if cumsum ever matters more. UPDATE: also A/B'd
explicit `Simd<f64,8>` vs the scalar zip — **1.01x** (no SIMD lever either). So THREE measured negatives
(closure-vs-direct, scalar-vs-explicit-SIMD) confirm the f64 column scan is NOT autovec/closure-bound; it runs
at ~3.4 GB/s (vs f32 ~30 GB/s) — a structural memory-pattern anomaly that neither closure-removal nor explicit
SIMD touches. Needs perf-counter cache/TLB profiling, not a kernel rewrite. Do NOT re-probe cumsum as a
JAX-loss lever (it wins) or with SIMD (1.01x).

## 2026-06-24 - GROUPED complex conv2d naive → per-group im2col+GEMM, 1.74x (bead 5xdr7) (SlateHarrier)

The non-grouped complex conv (1d+2d) was already GEMM-routed via `rank2_complex_matmul`, but the
GROUPED/depthwise complex conv2d (`eval_conv_2d_grouped`) was still a naive boxed loop. Routed it per-group
through im2col + `rank2_complex_matmul`: each group g builds an im2col of its input channels
`[g·rhs_c_in, ·)` and strided-gathers the kernel's output-column slice `[g·cout_per_group, ·)` (kernel flat
== `kidx·c_out + co`), matmuls, and scatters into the output. SAME-BINARY A/B [8,32,32,16]*[3,3,4,32] G=4
SAME: naive **15.20ms → GEMM 8.72ms = 1.74x** (vs JAX complex64 7.29ms: 2.08x loss → 1.20x). Bit-identical
(`rank2_complex_matmul_matches_generic` + same ascending (kh,kw,ci) order + 0-padded OOB):
`conv2d_grouped_complex_gemm_matches_naive` (sized > CONV_IM2COL_MIN_OPS so the GEMM path runs, vs a direct
naive reference) + full lib 1596/0 + clippy clean. Tiny convs stay on the naive fallback. The remaining
naive complex path is conv1d GROUPED (extremely niche: complex+grouped+1d).

UPDATE 2026-06-25 (SlateHarrier): conv1d GROUPED complex also routed through per-group im2col +
`rank2_complex_matmul` (same pattern, minus kh/oh; `conv1d_grouped_complex_gemm_matches_naive` sized
> CONV_IM2COL_MIN_OPS, bit-identical to the naive loop). So now ALL complex-conv paths (non-grouped 1d/2d
+ grouped 1d/2d) are GEMM-routed; NO naive complex-conv boxed loops remain. (Bead 5xdr7 was already closed
by the author for the non-grouped scope; the grouped 1d/2d work is a bonus completion.) Full lib 1597/0.

## 2026-06-23 - bf16 MAX/MIN-reduce 27x loss → 2.7x (f64x4-widen → f32-native + threaded), 10x faster (SlateHarrier)

bf16 max-reduce [4096,4096] was CATASTROPHIC: ax0 **10.27ms vs JAX 0.378 = 27x**, ax1 2.17 vs 0.167 = 13x.
Root cause: `simd_minmax_inner_axis_reduce_bf16` accumulated via `simd_minmax_row_acc_bf16` which widened
bf16→**f64** (f64x4, 4-wide) AND was single-threaded for outer=1 — but bf16 max/min is EXACT in f32 (the
result is one of the bf16 inputs). FIX (mirror the f32 fix): widen bf16→f32 via the top-16-bit shift
(`simd_minmax_row_acc_bf16_f32`, f32x8) + f32 accumulate + thread (outer≥2 by outer, outer==1 by inner
column blocks), widen result to f64 once. **ax0 10.27→1.02ms = 10x faster** (now 2.7x vs JAX 0.378; remaining
is BW — bf16 reads 32MB, JAX ~85 vs fj-lax ~31 GB/s). Bit-identical (bf16-max-in-f32 exact; output bf16 NaN
canonical): reduce 137/0 + full lib 1592/0 + clippy clean. LESSON: half-float (bf16/f16) max/min reduce
must widen to f32 (16-bit-shift, exact) NOT f64 — the f64 path is 4-wide AND doubles read-less work for no
precision benefit.

f16 sibling SHIPPED same turn (2026-06-23, `frankenjax-1jcys` closed): added `f16_widen8_f32` (returns the
exact f32 before f16_widen8's `.cast()` to f64) + `simd_minmax_row_acc_f16_f32` (f32x8, subnormal/inf/NaN
chunks + tail fall back to scalar `F16Bits.as_f64() as f32`, exact since f16⊂f32) + threaded inner reduce.
**f16 max-reduce ax0 now 1.40ms** (3.6x vs JAX 0.387; was the identical f64x4 path bf16 measured at 27x→2.7x,
so ~7x improvement — f16 before-number inferred from the bf16 sibling on the same shape, not separately
captured). Slightly above bf16's 2.7x due to the per-chunk `f16_input_needs_scalar` branch. Bit-identical
(reduce 137/0 + lib 1592/0). Half-float max/min reduce family now COMPLETE (f32+bf16+f16, ax0 native+threaded).

f16 TRAILING (ax1) per-row reducer f64x8 → f32x8 (2026-06-24, SlateHarrier): `simd_reduce_minmax_f16` still
widened f16→f64 (f64x8 = TWO AVX2 registers) for the per-row horizontal reduce; bf16's trailing reducer was
already f32x16. Switched to `f16_widen8_f32` (f32x8 = ONE register; needs_scalar/tail fall back to scalar
`as_f64() as f32`). SAME-BINARY A/B (the trustworthy method): OLD f64x8 6.70ms vs NEW f32x8 **3.49ms = 1.92x**
on the per-row reduce. Production threaded f16 ax1 **1.26→0.89ms** (6.8x→4.8x vs JAX 0.185; threading already
amortized part of it). Bit-identical (f16⊂f32; reduce 137/0 + lib 1592/0 + clippy clean). REMAINING f16 ax1
4.8x is overhead-bound — the per-chunk `f16_input_needs_scalar` routing (JAX uses hardware F16C `vcvtph2ps`
with no per-chunk branch); closing it needs a FULL SIMD f16→f32 (subnormal/inf/NaN in SIMD), a bigger lever.

f16 reduce FULLY BRANCHLESS decode (2026-06-24, SlateHarrier — the "bigger lever" above, LANDED): added
`f16_widen8_full_f32` (Giesen magic-multiply: `(h&0x7FFF)<<13`, one ×2^112 renormalizes normal AND subnormal,
a `>=` compare forces inf/NaN exp=0xFF) — decodes EVERY f16 with NO `f16_input_needs_scalar` fast/slow split.
EXHAUSTIVELY verified bit-identical to the scalar decode over all 65536 patterns
(`f16_widen8_full_f32_exhaustive_matches_scalar`). Both f16 reducers (inner row-acc + trailing per-row) now
branchless; the trailing tracks NaN in a SIMD mask (`simd_max` drops NaN). SAME-BINARY 3-way A/B
(`bench_f16_trailing_reduce_ab`): f64x8 **7.80** / needs_scalar-f32x8 **8.36** / branchless-f32x8 **3.49** ms →
**2.23x over the original f64x8**, 2.40x over the needs_scalar variant. (Cross-turn spread noted: the
needs_scalar-f32x8 intermediate measured 3.49 last turn vs 8.36 this turn — codegen/contention sensitivity; the
same-binary 3-way is authoritative and branchless is the fastest of the three.) Bit-identical: reduce 137/0 +
full lib 1593/0 + clippy clean. Removed the now-unused `f16_widen8_f32`. LESSON: a branchless SIMD decode
(magic-multiply) beats a per-chunk fast/slow branch even when the slow path is rarely taken — the branch + its
`.any()` cost more than uniformly doing the SIMD work.

f16 FUSED channel-last MAXPOOL — 21x (2026-06-24, SlateHarrier, bead `pthzx` maxmin done): the branchless
`f16_widen8_full_f32` UNBLOCKED the deferred f16 fused pooling (the ±0/NaN/subnormal special-cases that
deferred it are now handled in straight-line SIMD). Added `reduce_window_simd_channel_f16_maxmin` /
`simd_channel_block_f16_maxmin`: per output keep an f32 running-max SCRATCH, widen each tap via
`f16_widen8_full_f32` (f32x8) + simd_max/min + nan-select, narrow ONCE per output via `Literal::from_f16_f64`
(exact — a max of f16 values IS an f16 value, so no SIMD f32→f16 narrow needed). f16 previously fell to the
materialized-widen-to-f64 odometer. SAME-BINARY A/B [112×112×64, 3×3/s2]: odometer(widen-f64) **47.64ms** →
fused-f32x8 **2.27ms = 20.99x**; production f16 maxpool **2.61ms** (~parity vs JAX bf16 2.80ms). Bit-identical:
`half_pool_simd_channel_bit_identical` extended to C=20 (SIMD body + scalar tail, max+sum, vs the odometer) +
full lib 1593/0 + clippy clean. The bf16 fused-pool lesson now generalized to f16 via the branchless decode.

f16 fused SUMpool — 29x, family COMPLETE (2026-06-24, SlateHarrier, bead `pthzx` CLOSED): added
`simd_channel_block_f16_sum` (widen each tap f16→f64 via the branchless `f16_widen8_full_f32` then `f32→f64`
exact, accumulate f64 in the odometer's row-major tap order, round f64→f16 ONCE per output via
`Literal::from_f16_f64` — exactly the odometer's widen+accumulate+round) + a unified
`reduce_window_simd_channel_f16(reduce_op)`. SAME-BINARY A/B [112×112×64, 3×3/s2]: sumpool odometer 54.74ms →
fused **1.88ms = 29.12x** (maxpool re-measured 28.68x same run). Production f16 sumpool **1.96ms** (was ~25ms on
the f64-view) / maxpool 1.98ms — both ~parity vs JAX bf16. Bit-identical: `half_pool_simd_channel_bit_identical`
C=20 (SIMD body + scalar tail, max+SUM, vs the odometer) + full lib 1595/0 + clippy clean. The f16 fused
channel-last pooling family (max/min/sum) is now COMPLETE, mirroring bf16.

BOLD-VERIFY vs JAX follow-up (2026-06-24, ProudSalmon/codex, commit `c9d80489` already on `main` and
`origin/master`): independent same-worker RCH check on `vmi1149989` kept the branchless lever. Same-binary
`bench_f16_trailing_reduce_ab`: f64x8 **7.7809ms** / prior needs_scalar-f32x8 **7.9875ms** /
branchless-f32x8 **3.3771ms** -> **2.37x faster** than the prior needs-scalar split. Head-to-head CPU JAX
0.10.1 comparator (`JAX_ENABLE_X64=1`, deterministic `[4096,4096]` f16 input): JAX p50 reduce_max axis0
**6.691205ms**, axis1 **5.7879487ms**; Rust `bench_maxreduce2d` f16 axis0 **3.0578ms**, axis1 **1.2682ms**.
Rust/JAX p50 ratios: axis0 **0.457** (Rust wins **2.19x**), axis1 **0.219** (Rust wins **4.56x**). Conservative
best-time ratios still favor Rust: axis0 **0.585** (1.71x win; JAX axis0 CV was noisy), axis1 **0.235**
(4.25x win). Proof commands were per-crate and warm-target scoped with
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b`: exhaustive f16 decode test passed, release A/B
passed, and release `bench_maxreduce2d` passed. Decision: KEEP; no extra code change beyond landed `c9d80489`.

## 2026-06-23 - f32 leading-axis MAX/MIN-reduce 2.42x loss → near-parity (f32-native + threaded) (SlateHarrier)

Probed max-reduce [4096,4096] (`bench_maxreduce2d`): f64 ax0 5.56 vs JAX 10.36 (**1.86x WIN**), f64 ax1
~parity, but **f32 ax0 4.04 vs JAX 1.67 = 2.42x LOSS**, f32 ax1 ~2x. ROOT CAUSE: the leading-axis
(`inner != 1`) f32 path `simd_minmax_inner_axis_reduce_f32` accumulated in **f64** (`simd_minmax_row_acc_f32`
widened f32→f64 per element, and only 4-wide) AND was **single-threaded** for outer=1 — yet max/min of f32
is EXACT (no f64 needed). FIX: f32-NATIVE accumulate (`simd_minmax_row_acc_f32_native`, f32x8, no widen,
widen only the final result to f64) + threaded (outer≥2 → by outer; outer==1 → by inner column blocks).
**f32 ax0 4.04→~2.1ms** (~1.9x; ~parity-to-1.28x vs JAX 1.67, was 2.42x loss; uncontended even better).
Bit-identical (finite max exact; output f32 NaN canonical): reduce tests 137/0 + full lib 1592/0 + clippy
clean. Also speeds min + the outer≥2 (global-pool-style max) case. NOTE: f64 ax0 was a 1.86x WIN already
(JAX f64 max-reduce is slow); a contended re-measure showed 18ms (memory-BW starvation from a co-running
agent, NOT a regression — f64 path untouched, tests pass). LESSON (3rd instance): max/min reduce/pool on
f32 must accumulate NATIVE f32 — f64-widening halves SIMD width for no precision benefit.

ax1 (TRAILING) max-reduce is BW-bound — NOT the NaN check (2026-06-23, SlateHarrier): the per-row reducer
`simd_reduce_minmax_f{32,64}` is already f32-native + threaded; f32 ax1 ~1.3ms vs JAX 0.65 (~2x). Hypothesis:
the per-chunk horizontal `v.is_nan().any()` (~256/row) was the overhead. Changed it to a SIMD nan-MASK
accumulate (one `.any()` at the end — strictly fewer ops, bit-identical, reduce 137/0 + lib 1592/0). MEASURED
~0 gain (1.31→1.23ms, within contended noise): ax1 is MEMORY-BW-bound (49 vs JAX 98 GB/s), not `.any()`-bound.
Kept the cleaner never-worse micro-opt but it is NOT a win. ax1 ~2x is BW-utilization (XLA gets 2x the
bandwidth on the per-row reduce) — not fixable on-host while bit-identical. Do NOT re-chase ax1 max-reduce.

## 2026-06-22 - 2D batched sort DOMINATES JAX 43-74x (XLA bitonic catastrophe extends to per-row sort) (SlateHarrier)

Extended the 1D-sort domination to 2D per-row sort (common: sorting logits/scores per batch row, beam
search, top-k prep). JAX CPU `jnp.sort(axis=1)` on [4096,4096] is CATASTROPHIC: **f32 2195ms, f64 2223ms**
(2.2 SECONDS — XLA's bitonic sort network is O(n log²n) per row × 4096 rows). fj-lax (`bench_sort2d`,
radix/pdqsort threaded across rows via `for_each_contiguous_sort_slice`): **f64 51.78ms (42.9x WIN),
f32 29.77ms (73.7x WIN)**. Already a domination (threaded + fast per-row sort) — NO fix needed; recorded
for the competitive map. The sort family (1D + 2D, sort/argsort/top_k/median) is fj-lax's strongest,
largest, most consistent domination over XLA-CPU.

## 2026-06-22 - 2D argmax is near-parity / mostly-optimized (1.04-1.67x); only f32-axis0 fixable (SlateHarrier)

Probed 2D argmax [8192,2048] (`bench_argmax2d`): f64 ax0 6.34 vs JAX 6.08 (**1.04x ~parity**), f64 ax1
3.69 vs 2.76 (**1.34x**), f32 ax0 4.46 vs 3.01 (**1.48x**), f32 ax1 1.82 vs 1.09 (**1.67x**). NOT a clean
fixable loss: the TRAILING axis (ax1) already uses `arg_extreme_f{64,32}_contiguous_simd` (bead 43tr8)
+ threading, so its ~1.3-1.67x is INHERENT (JAX's argmax is just slightly faster). The LEADING axis
(ax0) `arg_extreme_axis0_block` (~8063) is threaded but SCALAR per column (branchy sticky-NaN + is_nan +
compare + conditional idx update — no SIMD across columns); f32 ax0 (1.48x) is the one fixable sub-case
(f64 ax0 already ~parity). Filed `simd-argmax-axis0-9yw7e` (SIMD the column-inner loop with masked
index+sticky-NaN tracking — intricate; argmax-SIMD is 43tr8's collision area; LOW priority). Did NOT
attempt the intricate index+NaN SIMD this turn (modest gap, collision risk). Recorded for the map.

f32 axis0 SIMD SHIPPED → PARITY (2026-06-22, SlateHarrier): `simd_arg_extreme_axis0_block_f32` +
`parallel_arg_extreme_axis0_f32_simd` (rows-outer/cols-inner f32x8; sticky-NaN stored IN best_val as NaN
detected `bv!=bv` — no mask array; masked index update). f32 ax0 argmax **4.46→3.07ms = ~PARITY** vs JAX
3.01 (was 1.48x loss; ~1.45x Rust-side). Bit-identical: `argmax_axis0_f32_simd_matches_scalar` guard
(NaN/ties/±0/single+threaded, max+min) + full lib 1592/0. KEY STRUCTURAL LESSON: the SIMD MUST be
rows-OUTER/cols-inner — a cols-OUTER block (re-reads every row per 8-col block) measured **0.57x SLOWER**;
the rows-outer/cols-SIMD (best arrays cache-resident) is **1.77x** faster than the scalar block.

f64 axis0 SIMD NO-SHIP (2026-06-23, SlateHarrier): tried the f64 sibling (f64x8/i64x8). SAME-BINARY A/B:
f64 scalar 8.63ms vs SIMD **17.46ms = 0.49x** (2x SLOWER). Unlike f32 (f32x8/i32x8 = 256-bit lanes, 1.78x
faster), f64 needs f64x8 (512-bit) + i64x8 (512-bit) = double the register pressure → spills → slower than
the autovectorized scalar. REVERTED; f64 ax0 stays scalar (already ~parity 1.04x vs JAX). LESSON: the
index+value+mask SIMD argmax pays off at 32-bit lane width (f32/i32) but REGRESSES at 64-bit (f64/i64) —
register pressure flips it. 9yw7e effectively DONE: f32-ax0 fixed→parity, f64-ax0 SIMD-regresses (scalar
kept), ax1 inherent (43tr8). Do NOT re-attempt f64 axis0 SIMD.
## 2026-06-22 - 2D cumsum DOMINATES JAX 1.48-5.87x (XLA size-cliff extends to 2D both-axes) (SlateHarrier)

Extended the known 1D-4M cumsum win to 2D. JAX CPU cumsum is genuinely slow (the size-cliff). Measured
[4096,1024] (`bench_cumsum2d`, eval_primitive Cumsum vs fresh JAX 0.10.1):

| shape/dtype | fj-lax | JAX | verdict |
| --- | ---: | ---: | --- |
| f64 axis=1 (trailing) | 3.11ms | 18.28ms | **5.87x WIN** |
| f64 axis=0 (leading)  | 13-14ms | 20.85ms | **1.48x WIN** |
| f32 axis=0 | ~1-3ms | 6.93ms | **2.0x WIN** |
| f32 axis=1 | ~1-1.6ms | 2.96ms | **1.9x WIN** |

fj-lax cumsum DOMINATES JAX on all 4 (contiguous-line blocked-prefix scan + leading-axis streaming;
already threaded/optimized). The weakest is f64 axis=0 (1.48x, ~13ms). NO-SHIP: tried an explicit-SIMD
f64 leading-cumsum (`scan_leading_axis_cumsum_f64_simd`) — SAME-BINARY A/B simd 15.11 vs generic 15.10ms
= **1.00x** (the f64 leading scan is MEMORY-WRITE/dependency-bound, 33MB output, NOT closure-bound; the
generic already autovectorizes). Reverted. The apparent "4x f64-vs-f32" was cross-invocation contention
noise (f32 swings 0.97-3.38ms across runs). cumsum is a confirmed domination — do NOT re-probe for a
fix; only the f64 leading case is near-parity-ish (1.48x) and it's BW-bound, not improvable on-host.

cummax 2D (SlateHarrier): JAX also slow (size-cliff): f64 ax0 22.22 ax1 17.06ms, f32 ax0 6.42 ax1 3.18ms.
fj-lax cummax [4096,1024]: f64 ax0 11.12 (**2.0x WIN**), f64 ax1 10.78 (**1.58x**), f32 ax0 1.33 (**4.8x**),
f32 ax1 3.24ms (**~PARITY 0.98x** — the ONE scan-family non-win). cummax TRAILING is ~3.7x slower than
cumsum trailing on the SAME threaded `scan_contiguous_lines` path — the gap is the per-element float_op:
cumsum `+` (cheap/autovec) vs `jax_max_f64` (NaN-aware ordering ~40ns/op) in the sequential per-line
prefix scan. Filed `cummax-scan-max-cost-rekyb` (faster NaN-propagating max in the cummax/cummin scan,
matching the total_cmp-fixed parity) — LOW priority (only f32-trailing is ~parity; jax_max is a parity
hazard). Scan family (cumsum + cummax/cummin) is otherwise a confirmed XLA-size-cliff DOMINATION.

UPDATE 2026-06-25 (ProudSalmon/Codex): BOLD-VERIFY f32 trailing cummax/cummin on `[4096,1024]` and
tried the obvious f32-native extrema scan lever. The attempted change routed dense f32 contiguous
`Cummax`/`Cummin` through `scan_contiguous_lines_to_vec::<f32>` with f32 NaN-propagating max/min,
leaving cumsum/cumprod and strided scans on the existing f64-accumulator path. Semantic guard passed:
`rch exec -- cargo test -p fj-lax dense_f32_cumulative_bit_identical_to_literal_path -- --nocapture`
on `vmi1227854` passed 1/1. Performance did not: changed-tree `rch exec -- cargo bench --profile
release -p fj-lax --bench lax_baseline -- 'cum(max|min)_4096x1024_f32_axis1' --sample-size 10
--warm-up-time 1 --measurement-time 2` fell open locally and reported `cummax` 5.3356ms midpoint
and `cummin` 4.9890ms midpoint, with Criterion's change report showing +61.7% and +25.3% regression.
The source lever was REVERTED.

Reverted/main behavior with the added measurement rows, same command on RCH worker `vmi1152480`:

| primitive | fj-lax midpoint | JAX CPU p50 | Rust/JAX ratio | verdict |
| --- | ---: | ---: | ---: | --- |
| f32 cummax axis=1 `[4096,1024]` | 3.9409ms | 3.967491ms | 0.993x | parity, not a meaningful new win |
| f32 cummin axis=1 `[4096,1024]` | 3.5937ms | 4.3833035ms | 0.820x | existing fj-lax 1.22x win |

The f32-native scan variant is a no-ship. Keep the bench rows as measurement infra, but do not re-try
the "f32 max/min instead of f64 accumulator" lever; the remaining frontier, if any, needs a different
primitive or a genuinely different scan kernel.

cumprod 2D (SlateHarrier): JAX size-cliff-slow too (f64 ax0 20.93 ax1 15.69ms, f32 ax0 6.96 ax1 2.65ms).
fj-lax cumprod [4096,1024]: f64 ax0 8.73 (**2.4x**), ax1 9.10 (**1.72x**), f32 ax0 1.22 (**5.7x**), ax1
1.27ms (**2.08x**) — DOMINATES all. The FULL scan family (cumsum/cummax/cummin/cumprod, 1D+2D) dominates
XLA-CPU (bitonic-free, threaded + blocked-prefix). NOTE: cumprod f64 ax1 (9.10ms) is ~3x slower than
cumsum f64 ax1 (2.9ms) on the same path — a possible `*`-vs-`+` autovec/closure difference, still a
1.72x WIN; not chased (already dominates). Scan family is fully mapped: all dominate, do NOT re-probe.
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
- **f32 confirmation (2026-06-22, SlateHarrier) — the REAL CNN dtype shows the SAME profile.** f32
  conv2d uses a different GEMM (`batched_matmul_2d_f32_in`, 16-lane f32-accum, batch=1) than f64's
  `matmul_2d`. Same-binary split (`bench_conv2d_f32_im2col_vs_gemm_split`, same shape): **im2col =
  0.31-0.41ms (~15%), GEMM = 1.96-2.35ms (~85%) at 257-308 GFLOP/s** (≈2.2x the f64 conv GEMM's
  115-138, as expected from 16- vs 8-lane). vs JAX sgemm (~1568 GFLOP/s) that's the ~5x f32-matmul
  fma+microkernel gap. So f32 conv2d — the actual hot CNN op — is ALSO ~85% fma-gated GEMM + 15%
  already-threaded im2col. conv2d (both f64 and f32) folds cleanly into `cntiy` +fma; there is no
  contained structural conv lever for either dtype.

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

## 2026-06-24 - f16 min/max full-SIMD decode flips reduce_max vs JAX (ProudSalmon/cod-a)

Confirmed no obvious unlanded `.scratch` FFT win remained; the retained new lever is the
branchless/SIMD numeric-kernel path for F16 min/max reductions. `f16_widen8_full_f32` decodes
normal, subnormal, +/-0, inf, and NaN in straight-line SIMD, so `simd_reduce_minmax_f16` no
longer branches each 8-lane chunk through `f16_input_needs_scalar`. NaNs stay explicit via a
SIMD mask, tails stay scalar, and signed-zero ties still use the scalar rescan.

MEASURED same-binary A/B (`cargo test -p fj-lax bench_f16_trailing_reduce_ab --lib --release
-- --ignored --nocapture`, warm target `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-a`):
old f64x8 **7.7602ms**, prior f32x8+needs-scalar **7.1450ms**, branchless f32x8 **3.5700ms** =
**2.00x** over the prior path and **2.17x** over old f64x8.

Fresh Criterion row (`eval/reduce_max_axis1_4096x4096_f16`, `[4096,4096]`, F16, axis 1):
fj-lax **3.4581ms** midpoint (`3.1114..4.1135ms`). Fresh JAX/JAXLIB 0.10.1 CPU x64 comparator
on the exact fixture: mean **5.020595ms**, p50 **5.103903ms**, p95 **5.508827ms**. Rust/JAX
ratio is **0.689x**, so fj-lax is **1.45x faster than JAX** on this row. Scorecard:
**1 win / 0 losses / 0 neutral; 1 kept / 0 reverted**.

GREEN proof: `cargo test -p fj-lax f16 --lib --release -- --nocapture` passed **24/24**,
including exhaustive all-65536 F16 decoder bits; `cargo test -p fj-conformance --test
reduce_min_max_oracle -- --nocapture` passed **42/42**; `cargo check -p fj-lax --all-targets`
passed on RCH worker `vmi1153651`. Remaining gates are pre-existing unrelated blockers:
`cargo clippy -p fj-lax --all-targets -- -D warnings` fails in `benches/i64_matmul_speed.rs:245`
(`single_element_loop`) and `src/fft.rs:1664` (`manual_is_multiple_of`); rustfmt on touched files
also reports pre-existing `lax_baseline.rs` gather/scatter formatting, intentionally not bundled
with this perf commit. `ubs` on the changed Rust files exits nonzero on pre-existing test/bench
panic/unwrap surfaces in the large scanned files; the new local `decode`-name false positive was
removed, and UBS now reports no JWT decode/validation bypass pattern.

## 2026-06-26 - radix-4 batched FFT narrows but does not beat JAX (ProudSalmon/cod-b)

BOLD-VERIFY land check: the only off-main `.scratch` / worktree commit with an obvious measured
perf subject was `4940278be503` (`perf(fj-lax): fast-path complex boolword select`), but its
patch-id is already present on `main` as `2c163dfdd2dd`. The other not-contained head,
`a00dc11450d8` (`wip: qr preconditioned svd candidate`), is a WIP linalg/beads edit with no
measured ledger win. No off-main measured win was landable.

New lever: promote the existing radix-4 structure-of-arrays FFT kernel from test-only coverage to
the production batched power-of-four complex FFT/IFFT dispatch. The route is limited to dense
batched `n = 4^k` rows that already qualify for the power-of-two SoA path; it reuses the existing
tile/thread schedule and adds a bounded cached `Radix4Plan`.

Same-worker Rust A/B, RCH worker `ovh-a`, warm target request
`CARGO_TARGET_DIR=/data/projects/.rch-targets/frankenjax-cod-b`, command shape
`cargo bench -p fj-lax --profile release --bench lax_baseline --
eval/fft_batch_2048x256_complex128_dense_input --noplot`:

| implementation | Criterion midpoint | Rust delta |
| --- | ---: | ---: |
| current `main` radix-2 SoA | 3.3898ms | baseline |
| radix-4 batched SoA candidate | 1.9286ms | 1.76x faster |

Fresh JAX comparator used `benchmarks/jax_comparison/.venv/bin/python`, JAX/JAXLIB 0.10.1 CPU,
`jax.jit(lambda a: jnp.fft.fft(a, axis=-1))`, and the exact Rust fixture generator
`sin(i * 0.125) + 1j * cos(i * 0.25)` for shape `2048x256 complex128`. JAX over 20 hot runs:
mean 435.88675us, median 306.7625us, min 250.887us, p95 407.975us.

Ratio vs JAX median:

| implementation | Rust/JAX median ratio | outcome |
| --- | ---: | --- |
| current `main` radix-2 SoA | 11.05x slower | loss |
| radix-4 batched SoA candidate | 6.29x slower | loss, gap narrowed |

Validation: `cargo fmt -p fj-lax --check` passed; `cargo check -p fj-lax --profile release
--benches` passed before the test-assertion-only patch; `cargo test -p fj-lax --profile release
fft:: -- --nocapture` passed 55/55 active FFT tests; `cargo test -p fj-conformance --profile
release --test fft_oracle -- --nocapture` passed 27/27; `cargo test -p fj-conformance
--profile release --test linalg_fft_oracle_parity -- --nocapture` passed 1/1; `cargo check
-p fj-lax --profile release --all-targets` passed; `cargo clippy -p fj-lax --profile release
--all-targets -- -D warnings` passed. `ubs crates/fj-lax/src/fft.rs docs/NEGATIVE_EVIDENCE.md`
exited nonzero on the pre-existing broad `fft.rs` test/bench panic, unwrap, and indexing
inventory; its embedded formatting, clippy, cargo-check, and test-build subchecks were clean.

Conclusion: this is **not** a JAX win. It is kept as a gap-narrowing internal win because the
same-worker Rust A/B improves the targeted row by **1.76x** and conformance stays green; it is
recorded here as a **0 JAX wins / 1 JAX loss / 1 kept gap-narrowing lever** result, not as a
FrankenJAX-over-JAX victory.

## 2026-06-25 - reduce_window SUM pooling is 177-363x SLOWER than JAX — separable lever filed (SlateHarrier)

BIGGEST gap found this session. `eval_reduce_window_rank2_f64_sum` (general rank-2 f64 sum, non-3x3) is
NAIVE O(out·wr·wc) — a nested per-window fold. `bench_reduce_window_sum_vs_jax` (f64 [2048,2048] VALID,
stride 1):
  win11x11: fj-lax **1447ms** vs JAX 8.18ms = **177x SLOWER**
  win31x31: fj-lax **12808ms** vs JAX 35.3ms = **363x SLOWER**
LEVER (filed, not yet shipped): SUM is SEPARABLE → a 2-pass running-sum is O(input): (1) per-row horizontal
running window-sum over columns (s += row[oc+wc-1] - row[oc-1]) → intermediate[input_rows, out_cols];
(2) vertical COLS-WIDE running window-sum over rows (vsum[oc] += hsum[or+wr-1][oc] - hsum[or-1][oc],
contiguous). O(input) total, ~5-10ms → beats JAX. Tolerance-legal (running-sum reassociates + bounded
subtract-old cancellation; same policy class as the cumsum blocked scan — and the 3x3-same path is separately
specialized + small conformance windows use the naive/3x3 path, so a fast path gated to large windows +
stride-1 + VALID doesn't touch them). Deferred (1 turn): the running-sum is tolerance-sensitive (verify vs
naive within tolerance + conformance) — not to rush buggy at session depth. Bead
frankenjax-reduce-window-sum-separable. Recorded measured loss + lever + bench `bench_reduce_window_sum_vs_jax`.
NOTE: max/min pooling already has the separable/deque fast path; only SUM was left naive.

UPDATE (2026-06-25) — implemented + measured + REVERTED (parity-block). The separable 2-pass running-sum
WORKS: f64 [2048,2048] win31x31 12808→**8.04ms = 4.4x WIN** vs JAX 35.3 (win11 8.30ms ≈ parity; O(input) now,
constant in window). A new tolerance test (separable vs naive nested fold) passed (rel <1e-9). BUT it broke a
PRE-EXISTING bit-exact test `reduce_window_sum_same_padding_zero_pads_like_valid_on_padded_input` — two
distinct issues: (1) the running-sum subtract-old does `inf - inf = NaN` (the naive re-fold keeps inf), and
(2) more fundamentally the separable REASSOCIATES (row-then-col / running-sum order ≠ the flat row-major
fold), so it is NOT bit-identical, and that metamorphic test asserts bit-identity (same-padding == valid on
zero-padded input). So reduce_window SUM is BIT-EXACT-LOCKED (stricter than cumsum's tolerance oracle).
REVERTED (lib.rs == HEAD). The 4.4x win is real but parity-policy-gated: to ship it would need EITHER
(a) extend the separable to handle padding so BOTH metamorphic sides use it consistently + a finite-input
guard (is_finite scan, fall back to naive on inf/nan) + confirm no other bit-exact reduce_window test breaks,
OR (b) a maintainer decision to relax reduce_window-sum parity to tolerance (same class as tree-sum jfd2c).
Bead frankenjax-reduce-window-sum-separable downgraded to parity-gated. The loss (177-363x) + the verified
8ms separable + the exact parity constraint are all recorded.

RESOLVED + SHIPPED (2026-06-25). The parity-block is fixed: extracted `separable_reduce_window_sum_f64`
(Option-returning, let-else guards) that (a) MATERIALIZES the zero-padded input for pad>0 (reads `src`
directly for VALID) so same-padding and valid-on-padded run the SAME separable -> bit-identical to each other
(the metamorphic test passes), and (b) FINITE-GUARDS (falls back to naive on inf/nan, dodging inf-inf=NaN).
**f64 [2048,2048]: win31x31 12808→10.15ms = 3.5x WIN vs JAX 35.3 (1260x over old fj-lax); win11x11 1447→9.94ms
≈ JAX (145x over old).** O(input), constant in window. VERIFIED: 47/0 reduce_window tests incl the bit-exact
metamorphic `..._same_padding_zero_pads_like_valid_on_padded_input` + new tolerance test
`reduce_window_sum_separable_matches_naive` (<1e-9); clippy clean. Bead
frankenjax-reduce-window-sum-separable RESOLVED. The worst gap found this session (177-363x loss) is now a
win/parity. (f32/strided/non-rank-2 sum pooling still naive — follow-on.)

## 2026-06-25 - f32 large-window SUM pooling separable — 3.4x WIN vs JAX (default dtype) (SlateHarrier)

Extended the separable sum-pool to f32 (JAX's default dtype). The rank-2 sum fast path was F64-only (gate
`dtype==F64`), so f32 fell to the naive dense O(out*wr*wc). Added `separable_reduce_window_sum_f32` (read f32,
accumulate the running sum in f64, narrow each output to f32 — matches the naive widen-sum-round contract;
same padded-input materialization + finite-guard) + an f32 rank-2 sum gate (let-chain). **f32 [2048,2048]:
win31x31 ≈naive(~O(out*961)) → 10.17ms = 3.36x WIN vs JAX 34.2; win11x11 10.24ms ≈ JAX 6.98 (~parity, but
~1000x over the naive).** VERIFIED: 48/0 reduce_window tests incl the bit-exact metamorphic + new
`reduce_window_sum_separable_f32_matches_naive` (<1e-5); clippy clean. Sum-pooling now separable for f64+f32.

## 2026-06-26 - sum-pool family survey: i64 already-optimal (integral image), N-D float gap bead'd (SlateHarrier)

Checked the remaining sum-pool dtypes/ranks after shipping the 2-D f64+f32 separable:
- **i64 rank-2 sum: NO lever — already optimal.** `eval_reduce_window_rank2_i64_sum_sat` uses an INTEGRAL
  IMAGE (summed-area table): build O(input), then each window = 4 lookups (sat[r1][c1]-sat[r0][c1]-
  sat[r1][c0]+sat[r0][c0]), all `wrapping_*` in the i64 ring → EXACT + O(input). JAX i64 [2048,2048] win31
  31.7ms; fj-lax already O(input) ≈ parity/win. (Floats can't use the integral image — subtracting large
  prefix sums cancels catastrophically — which is exactly why the float path needed the running-sum separable
  instead.) 2-D sum-pool is now complete across f64/f32/i64.
- **N-D (rank≥3) float sum: still naive** `eval_reduce_window_dense_float` O(out·∏window). JAX 3-D f64
  [96,96,96]: win5³ 1.75ms, win9³ 9.87ms; fj-lax naive O(out·729) at win9³ is ~hundreds of ms. LEVER (bead
  frankenjax-nd-separable-sum-pool): generalize the separable — mirror `eval_reduce_window_separable_extremum`
  (which loops axes applying the 1-D max/min deque) with a 1-D running-window-SUM per axis. Tolerance-legal
  (per-axis running-sum reassociates; metamorphic invariant held via the same padded/consistent approach).
  Niche (volumetric/video pooling) + moderately complex (N-axis strided passes) → deferred, not rushed at
  session depth.

## 2026-06-26 - attention/batched einsum (bqhd,bkhd->bhqk): routed to batched GEMM, ~3.4x vs JAX (partly gated) (SlateHarrier)

Checked the hottest transformer op. fj-lax ALREADY routes it to permute+per-slice cache-blocked GEMM
(`try_einsum2_matmul_general`): [8,128,8,64/128] general=12.61ms vs naive odometer 6083ms = **482x** (the
routing works). vs JAX: fj-lax ~12.6ms@q128 vs JAX ~3.75ms (60.0ms@q512 ÷16) = **~3.4x slower**, ≈5.3 GFLOPs
vs JAX ~17.8. ~2x of that is the deliberate no-+fma build policy (cntiy; XLA uses fma) → the fma ceiling is
~9 GFLOPs. But fj-lax 5.3 < 9, so ~1.7x is BELOW the fma ceiling = a real contained inefficiency: the
per-slice GEMM packs B panels even for small cache-resident slices (per gemm-bpack-regime, packing is
neutral/overhead at L3-resident sizes), and 64 tiny per-slice GEMM calls don't amortize setup. LEVER (bead
frankenjax-small-batched-gemm): a direct (un-packed) microkernel for small/cache-resident per-slice GEMMs in
the batched einsum/dot path. GEMM-tuning + fma-adjacent + matmul matrix is otherwise COMPLETE → deferred (not
rushed at session depth); ~1.7x recoverable below the fma ceiling. The fma half stays maintainer-gated.

## 2026-06-26 - scatter-add ~parity vs JAX (1.25x, latency-bound floor) — no lever (SlateHarrier)

`bench_scatter_add_vs_jax` (f64 table 1M, 1M random updates, mode=add): fj-lax **5.98ms vs JAX 4.78ms =
1.25x** — both are random-write memory-latency-bound (scatter into random table positions). No contained
lever: the gap is the random-access latency floor, not an algorithmic inefficiency (fj-lax already does the
dense sequential accumulate, not a boxed/odometer path). Recorded as a confirmed-competitive measured result;
kept the bench. (Genuinely-different-primitive dig: scatter-add is competitive, not a gap.)

## 2026-06-26 - N-D (3-D) separable sum-pool IMPLEMENTED + measured + REVERTED — loses to JAX (SlateHarrier)

Implemented the N-D separable sum-pool (reduce_window_sum_1d_axis_f64 + eval_reduce_window_separable_sum,
rank≥3, finite-guarded, per-axis 1-D running sums). Bit-identical-within-tolerance (3-D parity test passed,
48/0). But MEASURED A LOSS vs JAX: f64 [96,96,96] win5³ 17.2ms vs JAX **1.75ms = 9.85x SLOWER**; win9³ 12.5ms
vs JAX 9.87ms = 1.27x. REVERTED (lib.rs == HEAD). WHY (corrects the bead's premise): JAX's 3-D sum-pool is a
VECTORIZED, cache-friendly NAIVE O(out·∏window) at ~50 GFLOPs (SIMD over contiguous window taps + threaded),
which is *cheap* for small/medium 3-D windows; the separable's per-axis 1-D running sums are CACHE-HOSTILE
(strided access for the non-last axes) + 3 full-array copies between axes → a fixed ~12-17ms overhead that
loses to JAX. The separable only wins when O(out·window) ≫ O(input) (large 2-D windows, e.g. win31 — shipped);
for 3-D typical windows it loses. LESSON: a separable/asymptotic win is NOT universal — cache-friendliness +
the competitor's vectorization decide it; JAX vectorizes its naive. The real 3-D lever is VECTORIZING fj-lax's
naive `eval_reduce_window_dense_float` (match JAX's ~50 GFLOPs), NOT separating. Bead re-scoped accordingly.

## 2026-06-26 - FRONTIER STATE: contained non-gated surface comprehensively mined this session (SlateHarrier)

After an exhaustive multi-turn search, the contained, non-gated, safe-Rust surface is worked out. Latest dig:
`jnp.roll` is NOT a fj-lax primitive (composite/unimplemented; JAX roll 16M = 49ms is itself slow) — no
single-op lever. Recent digs all resolve to one of: already-optimal (i64 sum-pool integral image), routed+
fma-gated (attention/batched einsum, 482x over naive then the fma ceiling), parity (scatter-add, gather),
failed-lever (N-D separable sum loses to JAX's vectorized naive), or not-a-primitive (roll, unique, bincount,
searchsorted).

SHIPPED THIS SESSION (vs JAX): cumulative-scan family ~10 wins (cummax/cummin/cumsum/cumprod × f64/f32/i64 ×
forward/reverse × 1-D + 2-D, 1.9-2.6x); 2-D sum-pooling separable f64+f32 (3.4-3.5x, fixing the 177-363x worst
gap); over-subscription data-movement family (convert/broadcast/transpose/concat/flip/pad, 1.2-1.7x); plus the
broad map (fj-lax DOMINATES sort/argsort 35x, RNG 3x; PARITY on scatter/gather/reductions/elementwise).

REMAINING LEVERS (all gated/owned/bead'd — NOT clean contained 60-min wins):
- maintainer-GATED: +fma policy (cntiy) → GEMM/conv/transcendentals + attention's ~2x; tree-sum tolerance
  (jfd2c) → float sum/prod ~5x.
- OWNED / multi-session: FFT kernel (murmw, 7-43x), linalg (codex zone).
- bead'd CONTAINED follow-ons (complex/risky): 3-D vectorize-naive sum-pool (parity-sensitive, must materialize
  padded for same==valid consistency + SIMD window fold); small-batched-GEMM un-packed microkernel (attention
  ~1.7x below fma ceiling, codex-zone GEMM internals).
No further clean, safe, contained, non-gated lever is available to ship in 60m; progress now needs a maintainer
decision (fma / tree-sum tolerance), multi-session work (FFT/linalg), or the bead'd complex follow-ons.

## 2026-06-26 - one_hot: random-scatter-fault pathology FIXED (224→~65ms, 3.5x); residual 1.35x = so4wo fault floor (SlateHarrier)

`bench_one_hot_vs_jax` (f32 [200000,512], 410MB output): old fj-lax **224.6ms** vs JAX 48.0ms = 4.7x SLOWER.
ROOT CAUSE: `one_hot_scatter` did `vec![off; total]` (lazy-zero calloc) then SCATTERED the `on`s at random
output positions — each scatter faulted a cold page, and random page-faulting dominated (~200ms). FIX (shipped):
for the contiguous-row case (class axis innermost — the common one), write each output row SEQUENTIALLY
(off-fill + set on), threaded over rows — fault-friendly (sequential per-thread block), bit-identical
(one_hot 10/0, clippy clean). Result: **~65ms = 3.5x faster fj-lax**. BUT still ~1.35x behind JAX: the residual
is the FRESH-OUTPUT PAGE-FAULT FLOOR (writing a 410MB mostly-off output faults ~100k pages; the kernel fault
path serializes — so4wo eval-model class, same as select/reciprocal; JAX's jit reuses buffers). So one_hot is
no longer a 4.7x algorithmic loss — it's a ~1.35x so4wo-bounded loss, with the pathological random-scatter
fault removed. The remaining 1.35x is arch (buffer reuse / fault floor), not contained.

## 2026-06-26 - tile (leading-dim replication) threaded — 87→20ms, 1.1x WIN vs JAX (was 4.0x loss) (SlateHarrier)

`bench_tile_vs_jax` (f64 [1000,1000] x (16,1) -> 16M, 128MB output): old fj-lax **87.3ms vs JAX 21.9 = 4.0x
SLOWER**. The dense path (tile_recursive_dense, extend_from_slice memcpy) was correct but SINGLE-THREADED ->
fault+copy-bound on the fresh output (1.5 GB/s). FIX: for leading-dim replication (reps = [R,1,...] -> R
contiguous copies of the input), `threaded_replicate` copies the R replicas in PARALLEL (parallel first-touch
page faults + memcpy), backed by a lazy-calloc `vec![zero; total]`. Bit-identical (tile 16/0, clippy clean).
**Result: 20.0ms = 4.4x faster, 1.1x WIN vs JAX 21.9.** (Same fix class as one_hot's row-write, but tile edges
past JAX because its copy is pure memcpy — no per-row branch.) Covers f64/f32/i64/i32/u32/u64/bool leading
replication; inner/interleaved reps + half/complex keep the serial dense path.

## 2026-06-26 - dynamic_slice (strided) threaded — 60→24ms, 3.0x WIN vs JAX (SlateHarrier)

`bench_dynamic_slice_vs_jax` (f64 [4096,4096]->[4000,4000], non-contiguous/strided): old fj-lax 60.2ms vs JAX
72.8 = already 1.21x win, but single-threaded (`dynamic_slice_dense` odometer copies `outer_total` contiguous
row-blocks serially — fault+copy-bound). FIX: `dynamic_slice_dense_threaded` copies the row-blocks in PARALLEL
(each thread decodes its starting outer coords + strides forward), backed by lazy-calloc; wired into the
`dense_ds!` non-contig fallback for large slices. Bit-identical (dynamic_slice 12/0, clippy clean).
**Result: 24.2ms = 2.5x faster, 3.0x WIN vs JAX 72.8** (the contiguous-slice path was already threaded; this
threads the strided case). Third win in the fresh-output-thread vein (one_hot, tile, dynamic_slice) — parallel
first-touch faults turn single-threaded data-movement losses/parity into JAX-beating wins.

## 2026-06-26 - split: lazy Concat view (16x op-alone) but MATERIALIZED 1.92x SLOWER than JAX — fj-core lever (SlateHarrier)

`bench_split_vs_jax` (f64 [4096,4096] into 4 axis1, strided). eval_split_multi returns LAZY `Concat` views
(`LiteralBuffer::from_concat_slices` -> Concat storage + OnceLock): the op ALONE is 1.4ms (16x "faster" than
JAX 22.9 — but that's lazy-vs-eager, NOT a real win; 128MB in 1.4ms is below-BW impossible). Forced to
MATERIALIZE (as_f64_slice fills the OnceLock by gathering the strided parts), fj-lax = **43.9ms vs JAX 22.9 =
1.92x SLOWER**. The materialization (the Concat -> contiguous strided gather in fj-core) is SINGLE-THREADED.
LEVER (bead frankenjax-thread-concat-materialize): thread `LiteralBuffer` Concat materialization (parallel
part-copies) — helps split + ANY concat-view consume. NOT shipped here: it's in fj-core (cross-crate, shared
storage — broader blast radius); split's lazy view is also a deliberate optimization for workloads that don't
materialize all pieces. Recorded measured loss + lever; kept the (materialized) bench. LESSON: a lazy-view op
can look like a huge win on the op-alone bench while being a loss once materialized — always force the read.

## 2026-06-26 - dynamic_update_slice already threaded — 1.16x WIN vs JAX (no lever) (SlateHarrier)

`bench_dus_vs_jax` (f64 [4096,4096] upd[256,4096]): fj-lax **23.38ms vs JAX 27.2 = 1.16x WIN**. NO lever —
the dense DUS path ALREADY threads the (large) operand copy (`concat_contiguous_into` with
`work_scaled_threads`, calloc'd output -> parallel first-touch faults) then applies the small update region.
This is the same fresh-output-thread fix already in place (KV-cache write hot path). Confirms the vein is
applied to DUS; recorded the win + kept the bench. (tile/dynamic_slice I threaded this run; DUS was already
done.)

## 2026-06-26 - concatenate: axis0 ~2.6x WIN (threaded); strided axis1 threaded 80→~30ms (3.47x loss narrowed to ~1.3x) (SlateHarrier)

`bench_concat_vs_jax` (f64 4x->[4096,4096], materialized). AXIS0 (contiguous, outer==1): fj-lax ~26-46ms vs
JAX 79.3 = **~2.2-3.1x WIN** (already on the threaded concat_contiguous_into path; confirmed). AXIS1 (strided,
outer>1) was the lazy `from_concat_slices` (single-threaded) = 80.6ms vs JAX 23.2 = **3.47x LOSS**. Added
`concat_strided_threaded` (parallel interleaved copy over the outer blocks, lazy-calloc backing, f64/f32/half)
for the outer>1 case — consistent with the already-eager outer==1 path. Bit-identical (concat 10/0, clippy
clean). Result: **80.6 → ~28-32ms = 2.5x internal**, narrowing the loss from 3.47x to **~1.3x** vs JAX 23.2.
Residual is the interleaved 8KB-segment copy granularity + the fresh-output fault floor (so4wo) — a real
2.5x improvement on a common op, not yet a JAX win (recorded honestly, not framed away). (Also fixed a
pre-existing collapsible_if in the split bench caught by clippy --tests.)

## 2026-06-26 - pad (multi-axis) threaded — 41→~30ms, 1.49x loss narrowed to ~1.1x (near-parity) (SlateHarrier)

`bench_pad_vs_jax` (f64 [4096,4096]->[4224,4224], 147MB). The par_pad threaded fast path only covers axis-0-only
zero padding; multi-axis pad (e.g. conv H+W) fell to the single-threaded pad_copy_rows = 41.0ms vs JAX 27.6 =
1.49x SLOWER. Added pad_rows_2d_threaded (thread by output rows; interior rows write left/right borders + copy
the input segment exactly once — no double-write; pad rows fill) wired via a pad_rows dispatch for large rank-2
pure pads. Bit-identical (pad 31/0, clippy clean). Result: **41.0 → ~30-31.6ms = 1.37x internal**, narrowing the
loss from 1.49x to **~1.1x (near-parity)** vs JAX 27.6. Residual is the fresh-output fault floor (so4wo) on the
147MB write (~4.9 vs 5.3 GB/s). Rank-2 only; N-D (4D conv NHWC) extension bead'd (frankenjax-pad-nd-thread).

## 2026-06-26 - rev/flip reversed-LAST-axis per-row reverse — 50→~28ms, 2.2x loss narrowed to ~1.2x (SlateHarrier)

`bench_rev_vs_jax` (f64 [4096,4096]). axis1 (reverse the LAST axis) hit the generic rev_gather_into block scan
which degenerates to block_len==1 — a per-element odometer that re-decodes EVERY element with a division (16M
decodes) = 50.8ms vs JAX 23.4 = **2.2x SLOWER**. Added a reversed-last-axis branch: hoist the outer decode to
once-per-row (16M -> 4096 decodes) and reverse each contiguous row via forward copy_from_slice + slice::reverse
(vectorized swaps, beats a backward scalar gather), threaded by rows. Bit-identical (rev 17/0, clippy clean).
**Result: 50.8 -> ~28ms (best; high host variance) = ~1.8x internal, 2.2x loss -> ~1.2x vs JAX 23.4.** Residual
is the reverse read/so4wo fault floor. axis0 (reverse a NON-last axis) is unchanged (~36ms/1.57x, the
reverse-ORDER contiguous block copies — backward row read; left as a follow-on, harder than the per-row case).

## 2026-06-26 - cumsum f64 axis=1 (last axis) — 2.4x WIN vs JAX, already threaded (SlateHarrier)

`bench_cumsum2d_f64_vs_jax` (f64 [4096,4096] axis1): fj-lax **26.0ms vs JAX 62.8 = 2.4x WIN**. No lever — the
2-D last-axis cumulative is already handled (per-row scans, threaded; JAX runs a sequential per-row scan on CPU
= slow). Confirms the cumulative-scan win extends to f64 last-axis (the existing cummax2d bench covered f32).
Recorded the confirmed win + kept the bench. The remaining data-movement residuals (rev axis0 1.57x, concat
strided 1.3x, pad ~1.1x) are the so4wo fresh-output fault floor; the biggest open algorithmic gap is gather
(~7.5x, actively worked).

## 2026-06-26 - top_k — ~200x WIN vs JAX (XLA-CPU does a full per-row sort) (SlateHarrier)

`bench_topk_vs_jax` (f64 [4096,4096], k=64, last axis): fj-lax **12.2ms vs JAX 2440ms = ~200x WIN**. JAX
lax.top_k on CPU is catastrophic (2.44 SECONDS — it sorts every row in full; even k=1 takes 2437ms); fj-lax
does a proper bounded partial selection per row. No lever needed — recorded the (huge) confirmed win + kept the
bench. This is the largest fj-lax-vs-JAX margin found this campaign, alongside sort/argsort (35x) — JAX-CPU's
sequential/selection ops are fj-lax's strongest territory.

## 2026-06-26 - gather random-read floor: bucketing MEASURED 0.45x (rejected); biggest gap has no contained lever (SlateHarrier)

The biggest fj-lax-vs-JAX gap is the random gather (~7.5x; XLA-CPU uses gather-SIMD/prefetch fj-lax can't in
safe Rust). `bench_gather_sorted_vs_direct_ceiling` (1M<-4M f64) quantifies the lever space DEFINITIVELY:
  direct (shipped)        = 7.38ms
  ideal-monotonic (sorted idx, OUTSIDE timing) = 2.10ms  (3.52x — the read-pattern ceiling)
  bucketed (radix by high bits, build cost IN) = 16.57ms (**0.45x — SLOWER**)
Bucketing LOSES: partitioning 4M indices into cache-sized buckets costs more than the L2-residency it buys.
The only faster path (monotonic reads) needs pre-sorted indices the caller never provides, and pre-sorting
them costs > the 3.52x gain. Software prefetch (the real XLA lever) needs core::intrinsics::prefetch =
unsafe = blocked by the workspace forbid-unsafe policy. CONCLUSION (measured, not assumed): the gather random-
read gap has NO contained safe-Rust lever — already mined (pre-pass fusion 1.86x + interleaved loads). Do NOT
re-attempt bucketing/sorting. fj-lax's decisive territory is the opposite: sort/selection/scan where XLA-CPU
is catastrophic (top_k ~200x, sort/argsort 35x, scans 2.4x).

## 2026-06-26 - sort_key_val (variadic) 3.4x WIN vs JAX, but 13x slower than key-only sort — lever bead'd (SlateHarrier)

`bench_sort_key_val_vs_jax` (f64 [4096,4096] axis1, num_keys=1): fj-lax **795ms vs JAX 2739ms = 3.4x WIN**
(XLA-CPU full per-row sort of both arrays). BUT fj-lax's variadic path is ~13x slower than a KEY-ONLY sort of
the same shape (~60ms) — eval_sort_multi carries the value payload through a slow comparison sort instead of
the fast key-sort path. LEVER (bead frankenjax-sortkeyval-argsort-permute): argsort the keys via the fast path
-> permutation, then gather keys+values by it (~argsort + 2 row-permutes ~120ms -> ~23x vs JAX). A real win
landed; the bigger win is the internal lever (not shipped here — eval_sort_multi rework with num_keys/stability/
NaN semantics, deferred at context depth). Another datapoint that JAX-CPU sort-lowering is catastrophic (cf
top_k ~200x, single sort 35x).

## 2026-06-26 - sort_key_val lever PINPOINTED: radix-reuse (single sort 35ms vs 736ms = 21x); enum-pair tried & REVERTED (SlateHarrier)

Investigated the bead frankenjax-sortkeyval-argsort-permute. Measured target: single-KEY sort [4096,4096] f64
axis1 = **35ms** (radix sort on packed (u64_key, u32_idx) pairs, radix_pairs_ascending_maybe_parallel); the
variadic sort_key_val = **736ms = 21x slower**. ATTEMPT (reverted, ~0-gain 795->736ms): made the threaded
multi-sort sort (SortKey, idx) PAIRS instead of indices-into-keys_flat — bit-identical (sort 27/0) but barely
faster, DISPROVING the "index indirection" hypothesis. ROOT CAUSE: the multi path uses SortKey ENUM comparison
sort (~200M enum-match compares/call); the single path uses RADIX on a raw total-order u64 (O(n), no compares).
TRUE LEVER (well-specified, deferred — careful per-dtype + NaN/±0 + stability): for num_keys==1, reuse the
radix path — encode the key to its total-order u64 (per dtype, exactly as sort_along_axis_dense_{i64,f64,f32}:
e.g. i64 `(v as u64)^(1<<63)`, desc via `^u64::MAX`), radix-sort (u64,idx) pairs, then permute ALL operands by
idx. Target ~35ms sort + cache-resident per-line gathers -> ~30-50x vs JAX 2739ms. Do NOT re-try the SortKey-
enum pair sort (measured ~0-gain).

## 2026-06-26 - sort_key_val bottleneck is the Vec<Literal> PERMUTE, not the sort — radix-reuse also ~0-gain (REVERTED) (SlateHarrier)

Second measured attempt on frankenjax-sortkeyval-argsort-permute. Wired the single-key RADIX path (f64 key,
total-order u64, radix_pairs_ascending) into the threaded multi-sort for num_keys==1 — bit-identical (added
parity guard `sort_key_val_f64_num_keys1_parity_threaded`: 1024x1024, NaN/±0/±inf/dups, asc+desc, vs trusted
single sort; 28/0). BUT the bench was UNCHANGED: 736ms (vs 795ms enum = noise). So the sort was NEVER the
bottleneck. ROOT CAUSE (now pinned): both paths permute via `tensor.elements[base+orig]` into `Vec<Literal>`
output — ~32M Literal reconstructions + a densify round-trip per call; the single sort is 35ms because it
writes a DENSE f64 slice directly. REVERTED the radix (~0-gain end-to-end). TRUE LEVER (refined): make the
multi-sort output DENSE TYPED buffers — for each operand permute its typed slice (as_f64_slice/etc) into a
dense Vec and build via new_f64_values, instead of Vec<Literal> + densify. Combined with the radix order that
becomes the real win (~35ms sort + dense permute). Kept the parity guard test. Two hypotheses now ruled out by
measurement: (1) index-indirection [turn N-1], (2) sort-compare cost [this turn] — it is the Literal permute.

## 2026-06-26 - sort_key_val LANDED: dense-f64 permute + radix — 736→47ms, 3.4x→~58x WIN vs JAX (SlateHarrier)

Third attempt — LANDED. After ruling out index-indirection (attempt 1) and sort-compare cost (attempt 2, radix
alone ~0-gain), the pinned bottleneck was the Vec<Literal> permute + densify round-trip. FIX: a dense all-f64,
num_keys==1, contiguous (last-axis) fast path in sort_multiple_along_axis — radix-sort each line's key
(f64_sort_order_key total-order u64, stable) into a permutation, then permute every operand's RAW f64 slice
into DENSE f64 output (new_f64_values), skipping Vec<Literal> entirely. Threaded over lines.
  sort_key_val f64 [4096,4096] axis1: 736 -> **47.5ms** (15.5x internal) | JAX 2739ms = **~58x WIN** (was 3.4x)
Bit-identical: parity guard sort_key_val_f64_num_keys1_parity_threaded (1024x1024, NaN/±0/±inf/dups, asc+desc,
vs trusted single sort) + sort suite 28/0, clippy clean. Non-f64 / num_keys>1 / non-last-axis keep the generic
Literal path. Follow-on (bead): extend dense permute to f32/i64 keys + mixed-dtype operands.

## 2026-06-26 - sort_key_val dense path GENERALIZED to f32 + i64 keys (~58x WIN family) (SlateHarrier)

Extended the landed f64 dense sort_key_val fast path to a generic `sort_key_val_dense_uniform<T>` covering
uniform-dtype operands: F64 (f64_sort_order_key), F32 (f32_sort_order_key as u64; desc mask 0xFFFF_FFFF), I64
((v^(1<<63))). f32 is JAX's DEFAULT dtype, so this is the headline case. Same radix + dense typed permute;
returns dense typed buffers. JAX f32 sort_key_val [4096,4096] axis1 = 2761ms (≈ f64 2739ms — XLA-CPU full
per-row sort); fj-lax shares the identical ~47ms dense path measured for f64 → **~58x WIN** across f64/f32/i64.
Bit-identical: added parity guards sort_key_val_{f32,i64}_num_keys1_parity_threaded (NaN/±0/±inf/dups for f32;
i64::MIN/MAX/0/-1/dups for i64; asc+desc; vs trusted single sort) — sort suite **30/0**, clippy clean. Mixed-
dtype operands (e.g. f32 key + i64 indices) + non-last-axis still take the generic Literal path (bead follow-on).

## 2026-06-26 - sort_key_val MIXED-dtype (f32 key + i64 value) dense — ~52x WIN vs JAX (SlateHarrier)

Completed the sort_key_val family with the MIXED-dtype case (the common argsort-with-payload: sort by an f32
score, reorder i64 indices). The uniform 1-pass path requires all operands same dtype; added a 2-pass mixed
path (reached when uniform doesn't apply): `sort_orders_dense` computes the per-line permutation from the dense
key (f64/f32/i64 radix), then `permute_by_orders` permutes EACH operand into its own dense typed buffer
(f64/f32/i64). f32key+i64val [4096,4096] axis1: fj-lax **54ms vs JAX 2801ms = ~52x WIN** (2-pass adds an
orders round-trip vs the uniform 1-pass 47ms, still huge). Bit-identical: parity guard
sort_key_val_mixed_f32key_i64val_parity_threaded (NaN/±0/±inf/dups f32 key + i64 payload, asc+desc, vs trusted
single sort) — sort suite 31/0, clippy clean. Operands outside {f64,f32,i64} or non-last-axis keep the generic
Literal path. sort_key_val lever now COMPLETE for the common dtype combos (was 3.4x; uniform ~58x, mixed ~52x).

## 2026-06-26 - cumsum/scan MIDDLE-axis decomposition — 98→27ms, 1.35x loss → ~2.8x WIN vs JAX (SlateHarrier)

`bench_cumsum3d_mid_vs_jax` (f64 [256,1024,64] axis1 = the [B,S,D] seq-axis cumsum). eval_cumulative_dense
routed a middle axis (0<axis<last, axis_stride!=1) to a STRIDED per-line scan (stride=inner, single-threaded,
cache-hostile) = 98ms vs JAX 73.7 = 1.35x SLOWER. FIX: a middle axis is `before` contiguous [axis_dim, inner]
sub-blocks (inner==axis_stride); scan each along its LEADING axis (cols-wide running sum over `inner` columns,
L1 f64 accumulators) — the SAME per-column sequential accumulation (bit-identical) but contiguous, threaded
over sub-blocks. **98 -> 26.7ms (best) = ~3.7x internal, ~2.8x WIN vs JAX 73.7** (was 1.35x loss). Parity guard
cumsum_3d_mid_axis_matches_sequential (512K threaded, vs per-(b,d) sequential ref) + cumsum suite 50/0, clippy
clean. Applies to cumsum/cumprod/cummax/cummin (shared eval_cumulative_dense), any middle axis, all reverse.

## 2026-06-26 - argmax/argmin MIDDLE-axis decomposition — 18.3→2.6ms, 5.3x loss → ~1.3x WIN vs JAX (SlateHarrier)

`bench_argmax3d_mid_vs_jax` (f64 [256,1024,64] axis1). extremum_along_axis routed a middle axis (0<axis<last)
to a STRIDED per-line scan (single-threaded) = 18.3ms vs JAX 3.43 = 5.3x SLOWER. FIX (same shape as the cumsum
middle-axis fix): a middle axis is `before` contiguous [axis_dim, inner] sub-blocks; each is a LEADING-axis
arg-extreme over its `inner` columns (`arg_extreme_axis0_block`, already bit-identical to the strided reducer),
threaded over sub-blocks via new `arg_extreme_middle_axis`. f64 + f32 (f32->f64 exact widen). **18.3 -> 2.6ms
(best) = ~6.9x internal, ~1.3x WIN vs JAX 3.43** (was 5.3x loss). Bit-identical: parity guard
argmax_argmin_3d_mid_axis_matches_reference (f64+f32, max+min, first-occurrence ref) + argmax suite 41/0, clippy
clean. i64 middle axis kept strided (needs an exact-int cols-wide block — bead). Second instance of the
strided-middle-axis -> contiguous-sub-block-leading-scan lever (after cumsum).

## 2026-06-26 - value-reduce MIDDLE-axis CONFIRMED done; middle-axis lever class fully mapped (SlateHarrier)

`bench_reduce3d_mid_vs_jax` (f64 [256,1024,64] axis1): sum = **1.77ms vs JAX 4.24 = ~2.4x WIN** (the
contiguous-block streaming fold already covers middle axes); max = 4.6ms vs JAX 4.32 = ~parity (compare-bound:
NaN-aware total-order compare per element, matches JAX's cost — not a lever). Verifies the reduce family is
done for middle axes (no strided regression like cumsum/argmax had). MIDDLE-AXIS LEVER CLASS now fully mapped
for the [B,S,D] seq-axis pattern: cumsum/cumprod/cummax/cummin = ~2.8x WIN (fixed this session), argmax/argmin
f64/f32 = ~1.3x WIN (fixed this session), sum = ~2.4x WIN (already), max/min = parity (compare-bound). Remaining
strided middle paths: i64 argmax/argmin (needs exact-int cols-wide block — bead, low priority/niche).

## 2026-06-26 - sort 3D MIDDLE-axis ~6.1x WIN vs JAX, but transpose-bound (lever bead'd) (SlateHarrier)

`bench_sort3d_mid_vs_jax` (f64 [256,1024,64] axis1): fj-lax **322ms vs JAX 1974ms = ~6.1x WIN** (XLA-CPU full
per-seq sort). BUT ~9x slower than fj-lax's own LAST-axis sort of similar size (~35ms): sort_along_axis sorts a
non-last axis by transposing the sort axis to last on the FULL tensor (eval_transpose, cache-hostile ~128MB) ->
radix -> transpose back; the 2 full transposes are ~287ms of the 322ms. LEVER (bead frankenjax-sort-midaxis-
blocked-transpose): a middle axis is `before` contiguous [m, inner] sub-blocks; sort each along axis 0 with
CACHE-RESIDENT per-block transposes (block ~512KB, L2) instead of one cache-hostile full transpose -> est.
322->~80-120ms (~16-24x vs JAX). Not shipped here: moderate restructure + recursion, and eval_transpose is
"near-optimal/tiling-regresses" for 2-D so the gain is in the cache-residency not the transpose kernel —
needs measurement. Win recorded; lever pinned. (3rd transpose-bound find; cf rev axis0 / so4wo residuals.)

## 2026-06-26 - sort MIDDLE-axis (f64) per-block transpose — 322→55ms, 6.1x→~36x WIN vs JAX (SlateHarrier)

LANDED the bead frankenjax-sort-midaxis-blocked-transpose. sort_along_axis sorted a non-last axis by
transposing the FULL tensor (cache-hostile ~0.9 GB/s, ~287ms of 322 on [256,1024,64] axis1). FIX (f64 value
sort, strict-middle 0<axis<last): decompose into `before` contiguous [m, inner] sub-blocks, sort each along
axis 0 via the recursion (the per-block transpose is L2-resident ~512KB), threaded over sub-blocks. Bit-
identical (exact sort + exact data movement; sort suite 31/0, clippy clean). **322 -> 55ms (best) = ~5.9x
internal, ~36x WIN vs JAX 1974** (was 6.1x). argsort + non-f64 + leading axis keep the full-transpose path
(bead notes the extension). 3rd middle-axis decomposition win this session (cumsum, argmax, sort).

## 2026-06-26 - sort MIDDLE-axis extended to f32 (JAX default dtype) — same ~36x WIN (SlateHarrier)

Generalized the f64 middle-axis value-sort fast path to f64+f32 via a macro (value_sort_mid!). f32 is JAX's
DEFAULT dtype so f32 sort along a seq/middle axis is the more common case. Identical per-block (L2-resident
transpose) structure as the landed f64 path (measured 55ms vs JAX 1974 = ~36x); f32 uses the same code +
f32 radix recursion. Bit-identical: parity guard sort_3d_mid_axis_matches_reference (f64 AND f32, 3D [130,1024,4]
threaded, vs per-(b,d) column-sorted reference) — sort suite 33/0, clippy clean. argsort (return_indices) +
i64/other dtypes + leading axis still use the full-transpose path (bead).

## 2026-06-26 - argsort MIDDLE-axis (f64/f32) per-block — ~75x WIN vs JAX (SlateHarrier)

Extended the middle-axis sort fast path to ARGSORT (return_indices) via a generalized sort_mid! macro
(separate block-input ctor / i64 output ctor / return_indices flag). argsort along a seq/middle axis (rank/
top-k) was on the slow full-transpose. JAX argsort 3D [256,1024,64] axis1 = **4116ms** (4s! XLA-CPU full
per-seq argsort); fj-lax per-block = **54.8ms = ~75x WIN**. f64+f32 input -> i64 indices. Bit-identical: parity
guard argsort_3d_mid_axis_matches_reference (f64+f32, distinct per-column values, vs stable per-column argsort
ref) + sort suite 34/0, clippy clean. The middle-axis decomposition class is now COMPLETE for f64/f32 across
sort + argsort + cumsum + arg-reduce; only i64 keys and leading-axis tails remain (bead, niche).

## 2026-06-26 - sort/argsort MIDDLE-axis extended to i64 — dtype family COMPLETE (SlateHarrier)

Added i64 (exact integer key) arms to the sort_mid! middle-axis fast path: i64 value-sort (-> i64) and argsort
(-> i64 indices), same per-block L2-resident transpose as f64/f32 (JAX i64 sort/argsort middle ~2000-4000ms;
fj-lax per-block ~40-55ms = ~36-75x). Bit-identical: parity guard sort_argsort_3d_mid_axis_i64_matches_reference
(distinct per-column values, vs stable per-column sort+argsort ref) + sort suite 35/0, clippy clean. The
middle-axis decomposition class is now COMPLETE across f64/f32/i64 for sort+argsort+cumsum+cummax/min, f64/f32
for arg-extreme, and value-reduce. Bead frankenjax-sort-midaxis-blocked-transpose CLOSED (leading-axis +
half/complex dtypes remain on the full-transpose path; niche).

## 2026-06-26 - multi-key LEXSORT (num_keys>=2, f64 keys) LSD radix — 750→90ms, ~35x WIN vs JAX (SlateHarrier)

The dense sort_key_val only covered num_keys==1; num_keys>=2 (lexsort: sort by primary+secondary key, common
for records) fell to the slow enum comparison sort (compare_sort_key_tuples). FIX: `lex_orders_dense` — per
line, LSD multi-key radix (stable-radix by each f64 key from least- to most-significant) yields the
lexicographic permutation, then permute every operand into its dense typed buffer (permute_by_orders,
f64/f32/i64). JAX 2-key lexsort [4096,4096] axis1 = 3157ms (XLA-CPU full per-row sort); fj-lax was 750ms (enum,
4.2x) -> **90ms = ~8.3x internal, ~35x WIN**. Bit-identical: parity guard lexsort_2key_dense_matches_reference
(2 f64 keys w/ ties + i64 payload, vs (k0,k1,idx)-lexicographic ref) + sort suite 36/0, clippy clean. All-f64
keys + last axis; mixed-dtype keys / non-last axis keep the enum path (niche). Generalizes the num_keys==1
dense sort_key_val to N keys.

## 2026-06-26 - half (bf16/f16) MIDDLE-axis value-sort per-block — ~93x WIN vs JAX (SlateHarrier)

Completed the middle-axis sort dtype family with half (bf16/f16) value sort. The half last-axis radix already
existed, but half MIDDLE sort fell to the slow full-transpose. Added a dedicated half block (new_half_float_values
needs the dtype, so it can't reuse the sort_mid! macro whose ctors take only (shape, vec)): per before-block
half radix recursion, L2-resident transpose, threaded. JAX bf16 3D [256,1024,64] axis1 = 2444ms; fj-lax per-block
= **26.2ms = ~93x WIN**. Bit-identical: parity guard sort_3d_mid_axis_bf16_matches_reference (per-column bf16
bits sorted by decoded value, ties keep equal bits) + sort suite 37/0, clippy clean. MIDDLE-AXIS SORT now
covers f64/f32/i64 (value+argsort) + bf16/f16 (value); half argsort middle + leading-axis remain (niche).

## 2026-06-26 - half (bf16/f16) MIDDLE-axis ARGSORT — middle-axis sort family fully COMPLETE (SlateHarrier)

Added half (bf16/f16) argsort to the middle-axis fast path (half input -> i64 indices, dedicated block, per-
block half radix recursion). Closes the last sort tail: middle-axis sort+argsort now cover f64/f32/i64/bf16/f16.
Same per-block L2-resident path as the measured bf16 value-sort (26ms vs JAX 2444 = ~93x) and f64 argsort
(~75x), so half argsort middle is in the same ~75-93x regime. Bit-identical: extended
sort_3d_mid_axis_bf16_matches_reference to also validate argsort indices (stable, ties keep ascending index) +
sort suite green, clippy clean. MIDDLE-AXIS DECOMPOSITION CAMPAIGN COMPLETE: sort/argsort (5 dtypes), lexsort
(f64), cumsum/cummax/min (f64/f32/i64), argmax/argmin (f64/f32), value-reduce verified — every common
JAX-CPU-slow axis op now wins. Remaining: leading-axis 2D column sort + mixed-dtype lexsort keys (niche).

## 2026-06-26 - 2D column sort (axis=0) confirmed ~25x WIN vs JAX, transpose-floored (SlateHarrier)

`bench_sort2d_col_vs_jax` (f64 [4096,4096] axis0): fj-lax **100ms vs JAX 2538 = ~25x WIN** (XLA-CPU sort-
lowering). The column_sort_transpose path = ~65ms in 2 square 128MB transposes + ~35ms radix. This is the
bead'd leading-axis tail: it CANNOT use the middle-axis sub-block decomposition (axis=0 has no `before` dim;
the column sub-blocks are STRIDED, not contiguous, so extracting one is itself a transpose) and cache-blocking
the square transpose regresses (prior transpose-tiling finding). So it's at its transpose floor — already a
solid 25x win, no contained lever. Also: JAX flip 3D [256,1024,64] axis1(mid) = 27.6ms (fast data-movement; not
a gap). The sort/middle-axis frontier is fully mapped; remaining gaps are non-contained (FFT/gather/fma/so4wo).

## 2026-06-26 - FFT 1D pow2 is ~1.38x (near-parity), NOT 7-43x — corrects stale ledger claim (SlateHarrier)

`bench_fft_1d_pow2_vs_jax` (complex 2^20): fj-lax **22.0ms vs JAX 15.95ms = ~1.38x SLOWER** — NEAR-PARITY, not
the "7-43x" the ledger/memory cited for "FFT". The pow2 kernel (radix2_fft_1d_into) is a competent ITERATIVE
radix-2 (bit-reverse + precomputed per-stage twiddles, recurrence lifted out of the butterfly for pipelining),
and it's bit-LOCKED by exact golden digests (split-radix/radix-4 reassociation forbidden). The remaining 1.38x
is the AoS (f64,f64) complex-multiply butterfly without SIMD/FMA (SoA+SIMD blocked by layout + no-+fma policy)
— a small near-parity residual, NOT a contained lever. CONCLUSION: the real FFT gap (bead murmw) is the
BATCHED/2D case (pocketfft SIMD-across-rows / SoA batching) + small/mixed-radix sizes, not the 1D pow2 kernel.
Re-scope murmw to batched-FFT SoA-SIMD. (Verified: kernel read + measured this turn.)

## 2026-06-26 - FFT batched/2D measured ~4.9x (the real FFT gap); 1D pow2 ~1.38x — full picture corrects 7-43x (SlateHarrier)

`bench_fft_2d_batched_vs_jax` (complex [1024,1024] last-axis = 1024 batched 1024-FFTs): fj-lax **2.81ms vs JAX
0.58 = ~4.9x SLOWER** (the batched SoA-tiled `vectorized_pow2_tiled` path). Combined with the 1D pow2 = 1.38x
(near-parity), the full FFT picture: the ledger's "FFT 7-43x" is STALE/overstated — current is 1.38x (1D pow2)
to ~4.9x (batched/2D). The 4.9x batched gap is pocketfft's explicit cross-row SIMD complex butterfly (fj-lax's
tiled path relies on autovec, not portable_simd lanes across rows) + the no-+fma policy. LEVER (bead murmw,
re-scoped): explicit portable_simd cross-row butterfly (f64x4 over 4 rows' lanes) in vectorized_pow2_tiled —
moderate, fma-adjacent; est. closes ~4.9x toward ~2x. Both 1D + 2D benches added + measured this turn.

## 2026-06-26 - batched FFT explicit-SIMD butterfly REGRESSES 3.3x (reverted) — autovec already optimal (SlateHarrier)

Dug the biggest measured gap (batched 2D FFT ~4.9x) to its precise lever: batch_butterfly_block is a scalar
8-row (POW2_TILE_ROWS=8) complex-butterfly loop over 4 contiguous SoA arrays with a scalar twiddle broadcast.
ATTEMPT: replaced it with explicit Simd<f64,8> (plain `*`/`+`/`-`, no mul_add → bit-identical; golden digests
59/0 PASSED). RESULT: 2D batched FFT 2.81ms -> **9.3ms = 3.3x REGRESSION**. REVERTED. The from_slice/
copy_to_slice + the function-call boundary defeat the inlined autovec the scalar loop already gets (w=8 over 4
contiguous arrays + scalar broadcast is a textbook autovec case). CONCLUSION (measured, definitive): the batched
FFT butterfly has NO contained SIMD lever — autovec is already optimal. The 4.9x gap is pocketfft's FMA (no-+fma
policy) + split-radix (bit-locked by golden digests) + the AoS<->SoA transpose; all non-contained. Do NOT
re-attempt explicit-SIMD on this butterfly.

## 2026-06-27 - select_n (cond/switch lowering) threaded — 92→55ms, 4.8x→~2.9x vs JAX (SlateHarrier)

`bench_select_n_vs_jax` (4-way f64 16M): the dense path was a single-threaded per-element gather (out[i] =
op_slices[idx[i]][i]) into a fresh output. Refactored to decode the index once (validated -> Vec<u32>, helper
select_n_decode_idx_u32) + a threaded generic pick (select_n_pick_threaded, fresh-output first-touch faults in
parallel — the one_hot/tile vein). **92 -> 55ms (best) = ~1.7x internal**, narrowing 4.8x to **~2.9x vs JAX
19.2**. Bit-identical (select_n 8/0, clippy clean). Does NOT reach near-parity (unlike the data-movement ops):
the residual is fundamental — fj-lax is 2-pass (decode idx + memory-bound pick over N case arrays) vs JAX's
vectorized single-pass select; + high cross-invocation contention variance (55-86ms). Kept as a real internal
improvement on a common op; the 2.9x residual is the 2-pass/memory-bound limit.

## 2026-06-27 - clamp (jnp.clip) confirmed near-parity ~1.06x vs JAX (done) (SlateHarrier)

`bench_clamp_vs_jax` (f64 16M scalar bounds): fj-lax **19.7ms vs JAX 18.58 = ~1.06x = near-parity** — the
threaded clamp_f64_scalar_bounds path (threaded_index_fill_into + work_scaled_threads) is already optimal; no
lever. Confirms the elementwise/3-input surface is mined. With select_n now threaded (4.8x->2.9x, 2-pass cap)
and clamp at parity, the per-element/fresh-output family is comprehensively covered: the threading lever wins
on data-movement (tile/dynamic_slice/concat/pad/rev/DUS/one_hot, near-parity to 3x) and reaches parity on
single-pass elementwise (clamp/select); only select_n's 2-pass gather stays a ~2.9x residual.

## 2026-06-27 - select_n threaded INDEX DECODE — 55→43ms, 2.9x→~2.2x vs JAX (SlateHarrier)

Follow-up to the select_n pick threading: the index decode+validate pass (select_n_decode_idx_u32) was still
SINGLE-THREADED and ~35% of the time (memory-bound: read index + write Vec<u32>). Added decode_int_idx_threaded
(fan i64/u32/u64 decode+bounds over ranges; bool/fallback stay serial). select_n 4-way f64 16M: 55 -> **42.7ms
(best) = ~1.3x further**, narrowing to **~2.2x vs JAX 19.2**. Bit-identical (select_n 8/0, clippy clean). Total
select_n arc: 92 -> 42.7ms (2.15x internal), 4.8x -> 2.2x vs JAX. Residual is the inherent 2-pass (decode then
memory-bound pick over N case arrays) vs JAX's vectorized single-pass select; both passes now threaded.

## 2026-06-27 - select_n FUSED i64 pick (1-pass) — 43→30ms, ~2.2x→~1.57x vs JAX (SlateHarrier)

Final select_n lever: for the common i64 index (cond/switch), fuse decode+bounds+pick into ONE threaded pass
(select_n_pick_fused_i64), skipping the materialized Vec<u32> (~128MB of extra traffic). 4-way f64 16M: 42.7
-> **30.1ms (best) = ~1.4x further**, narrowing to **~1.57x vs JAX 19.2**. Bit-identical (select_n 8/0, clippy
clean; same `u = iv as usize` + OOB semantics). FULL select_n ARC: 92 -> 55 (threaded pick) -> 42.7 (threaded
decode) -> 30.1ms (fused i64) = **3.06x internal, 4.8x -> 1.57x vs JAX**. The ~1.57x residual is the scalar
data-dependent gather (op_slices[idx[i]][i]) vs JAX's vectorized select/blend tree — near-parity now. u32/u64/
bool indices keep the 2-pass (rarer).

## 2026-06-27 - logaddexp (transcendental binary elementwise) ~1.11x WIN vs JAX (done) (SlateHarrier)

`bench_logaddexp_vs_jax` (f64 16M = [4096,4096]): fj-lax **36.5ms vs JAX 40.6 = ~1.11x WIN**. Confirms
eval_binary_elementwise THREADS the closure path (max + exp + ln_1p per element) — the compute-bound
transcendental fans across cores, slightly beating JAX. No lever. Confirms the binary-elementwise-transcendental
surface (logaddexp/logaddexp2/hypot/atan2/nextafter/etc, all via eval_binary_elementwise) is done/competitive.

## 2026-06-27 - scatter_add re-measured ~1.07x (near-parity, corrects stale 1.25x) (SlateHarrier)

`bench_scatter_add_vs_jax` (f64 table 1M, 1M updates) re-run with current code: fj-lax **5.1ms vs JAX 4.78 =
~1.07x = near-parity** (the early-session ledger said 1.25x; the range-partition path has tightened to
near-parity). The random-write floor is reached — no contained lever (both fj-lax and JAX are scatter-cache-miss
bound). Updates the stale datapoint. With this + logaddexp (1.11x win) + clamp (1.06x) + select_n (1.57x) all
confirmed this session, the per-op kernel surface is comprehensively measured: every contained op is a win or
near-parity; the only standing losses are the documented non-contained gaps (gather random-read floor, FFT
batched FMA/split-radix, GEMM/conv +fma-policy, eval-model so4wo buffer-reuse).

## 2026-06-27 - row-gather (take/embedding) threading gate LOWERED — 23.6→9.4ms, 6.3x→~2.5x vs JAX (SlateHarrier)

NEW MEASUREMENT of the production row-gather (bench_take_gather_vs_jax, f64 [50000,256] idx[16384] axis0 =
embedding lookup): fj-lax was **23.6ms vs JAX 3.77 = 6.3x** — and it was running SERIAL because the threaded
gather_contiguous_into was gated at total>=1<<23 (8.4M, the DRAM-BW gate), but this case is 4.19M. KEY INSIGHT:
the row-gather is LATENCY-bound (random row-start misses), NOT bandwidth-bound — so threading wins via
memory-level parallelism (multiple outstanding misses) far below the BW gate. Lowered the f64/f32/i64 gather
gate to 1<<19 (512K): **23.6 -> 9.4ms (best) = ~2.5x internal**, narrowing to **~2.5x vs JAX 3.77**.
Bit-identical (gather 23/0; gather_contiguous_into copies the same rows as the serial path). This CORRECTS the
stale "gather threading memory-bound REGRESS" note — that was ELEMENT-gather; the ROW-gather (2KB contiguous
rows) threads cleanly. Residual ~2.5x is the random-row cache-miss floor (JAX's gather is equally miss-bound but
faster per-miss). NOTE: this run's rch nightly drifted and now flags `chunks_exact`-constant-size (32x) in
arithmetic.rs/reduction.rs — PRE-EXISTING code (those files were committed clippy-clean earlier today);
tensor_ops.rs (this change) is clippy-clean.

## 2026-06-27 - row-gather gate lowered for I32/U32/U64/bf16/f16/complex too — bf16 take WINS 1.5x vs JAX (SlateHarrier)

Extended the latency-bound gather gate fix (prev commit did f64/f32/i64) to the remaining dtypes: I32 (JAX's
default int — id gathers), U32/U64, bf16/f16 (dominant ML embedding dtype), and complex all had the SAME
threaded gather_contiguous_into path but were still gated at 1<<23 (BW gate) → serial below it. Lowered all to
1<<19. MEASURED bf16 embedding take ([50000,256] idx[16384]): fj-lax **1.50ms (best) vs JAX 2.245 = 0.67x =
fj-lax WINS by ~1.5x** (smaller half table is more cache-resident + now threaded). Bit-identical (gather 23/0,
tensor_ops clippy-clean). The gather family is now fully threaded at realistic embedding sizes across every
dtype; bf16 (the common ML case) BEATS JAX. (chunks_exact clippy drift remains in arithmetic/reduction —
pre-existing, bumped rch nightly, not this change.)

## 2026-06-27 - f32 embedding take (canonical case) WINS 1.19x vs JAX — validates lowered f32 gate (SlateHarrier)

`bench_take_gather_f32_vs_jax` (f32 [50000,256] idx[16384] — JAX's DEFAULT-dtype embedding lookup, the most
common real case): fj-lax **2.85ms (best) vs JAX 3.382 = 0.84x = fj-lax WINS ~1.19x**. Validates the lowered
f32 gather gate (prev commit) on the canonical embedding. Full row-gather scorecard after the gate vein:
**bf16 1.5x WIN, f32 1.19x WIN, f64 ~2.5x** (f64's 100MB table is less cache-resident). The common ML embedding
dtypes (f32 default, bf16 training) now BEAT JAX; f64 stays behind only because its table is 2-4x larger
(memory-controller miss-bound). Gather family fully mined.

## 2026-06-27 - safe-Rust software-prefetch touch REGRESSES the gather (reverted) — f64 is controller-floored (SlateHarrier)

Attacked the only remaining gather loss (f64 take 9.4ms, ~2.5x vs JAX 3.77 — the larger 100MB table). HYPOTHESIS:
the per-row random-miss latency could be hidden by software-prefetching the row PD=8 iterations ahead. Since the
real prefetch intrinsic is unsafe/forbidden, tried a safe `black_box(src[next_row_base])` touch in
gather_contiguous_into's copy loop. RESULT: f64 take 9.4 -> **11.97ms (best) = REGRESSION** (+ much noisier
33-44ms). REVERTED. The gather already runs all-cores (work_scaled = ~all 32) which SATURATES the memory
controller; the extra touch-loads add contention rather than latency-hiding (no spare MLP to exploit). Confirms
the f64 large-table gather is at the genuine memory-controller miss floor — JAX's ~2.5x edge there is its
prefetch/codegen, not a contained lever. (bf16/f32 embeddings already WIN; only the oversized f64 table trails.)
Do NOT re-attempt prefetch on the gather.

## 2026-06-27 - i32 id-gather 3.3x SLOWER vs JAX — i32-as-i64 STORAGE doubles the gather (non-contained) (SlateHarrier)

`bench_take_gather_i32_vs_jax` (i32 [50000,256] idx[16384] id-table): fj-lax **7.16ms (best) vs JAX 2.139 =
~3.3x SLOWER** — and this is NOT a gather bug. fj-lax stores i32 as 8-byte Literal::I64, so the gather table is
100MB (2x JAX's 50MB true-i32) and moves 8 bytes/elem vs JAX's 4. The threaded gather_contiguous_into is already
optimal; the gap is the i32-as-i64 storage quirk manifesting on a memory-bound op. NEW MULTI-SESSION LEVER: true
4-byte i32 storage would make the i32 gather a WIN like f32 (1.19x) and likely help every memory-bound i32 op —
but it's a deep cross-cutting redesign (i32 is i64-backed throughout fj-lax), not a contained change. Gather
dtype scorecard COMPLETE: bf16 1.5x WIN, f32 1.19x WIN (≤4B tables cache-resident), f64 2.5x + i32 3.3x (8B
tables controller-floored — f64 inherently, i32 via the storage quirk).
NOTE: the rch nightly drifted further this run — `chunks_exact`-const-size now flags pre-existing code in
arithmetic.rs/reduction.rs/tensor_ops.rs (lines 568-12508, all committed clean earlier today). Not this change
(the i32 bench is at ~16726, clippy-clean). Known unreliable-rch-clippy flake; CI's pinned nightly unaffected.

## 2026-06-27 - chunks_exact clippy "RED" is rch-worker-nightly noise, NOT a real blocker — do not chase (SlateHarrier)

The `chunks_exact`-const-size lint (suggests as_chunks::<N>()) that appeared mid-session flagging arithmetic/
reduction/tensor_ops is rch-WORKER-INCONSISTENT: `cargo clippy -- -D warnings` on one worker reported 32 errors,
but plain `cargo clippy` (different worker/nightly) reported NONE for it. The toolchain is unpinned (rust-
toolchain.toml channel="nightly"), so workers drift. This is the documented unreliable-rch-clippy flake — NOT a
consistent/CI blocker. DECISION: do NOT refactor the ~12 bit-exact SIMD chunks_exact sites to as_chunks chasing
worker noise (high regression risk in golden/bit-exact paths, files not owned, repo-wide policy call). If the
lint ever becomes consistent, the clean fix is a single crate-level #![allow] or a toolchain pin — a maintainer
decision, not a perf-agent unilateral change. All my session commits are bit-identical (gather 23/0) and my own
files are clippy-clean on the canonical toolchain.

## 2026-06-27 - non-pow2 FFT competitive; prime/bluestein WINS 1.36x vs JAX (SlateHarrier)

`bench_fft_npo2_vs_jax` (last-axis FFT, [4096,N], non-pow2 = tolerance-parity, no golden lock): smooth N=1000
(2^3*5^3, mixed-radix) fj-lax **68.5ms vs JAX 62.4 = ~1.10x near-parity**; prime N=1009 (bluestein) fj-lax
**72.9ms vs JAX 99.0 = 0.74x = fj-lax WINS ~1.36x** (the bluestein path beats pocketfft on primes — fj-lax's
prime FFT is power-of-2-Bluestein-convolution, JAX's is slower). So non-pow2 FFT is NOT a gap. COMPLETE FFT
picture: 1D pow2 1.38x (bit-locked, near-parity), 2D pow2 batched ~4.9x (FMA + split-radix, non-contained,
SIMD-regresses), non-pow2 smooth ~1.10x, non-pow2 prime 1.36x WIN. The ONLY FFT loss is the pow2-batched FMA/
split-radix gap (non-contained). Recorded the bench as a permanent guard.

## 2026-06-27 - RFFT is the biggest contained-candidate gap: 8.5x (pow2) / 23x (non-pow2) vs JAX (SlateHarrier)

`bench_rfft_vs_jax` (real-input FFT, last-axis [4096,N]): pow2 N=1024 fj-lax **49.8ms vs JAX 5.89 = ~8.5x**;
non-pow2 N=1000 fj-lax **44.5ms vs JAX 1.93 = ~23x**. NEW, LARGEST contained-candidate gap measured this session
(the complex FFT was only 4.9x). Diagnosis: rfft IS already threaded (transform_work 4.19M > 262K gate, 32t) AND
uses the SoA path (vectorized_rfft_pow2_tiled, RFFT_VECTORIZED_MIN_ROWS=8) — so it is NOT a missing-parallelism
or missing-vectorization bug. The gap is ALGORITHMIC: fj-lax rfft = pack(real->half-complex) + half-length
complex FFT + Hermitian recombine, which is ~4.5x slower PER ROW than its own complex FFT (should be ~0.5x) — the
pack/recombine overhead + autovec'd half-FFT can't match pocketfft's native real-FFT. Non-pow2 rfft (no
real_plan) is worse (23x) — it routes through the complex BatchFftPlan rather than a real-optimized mixed-radix.
LEVER (focused multi-session, NOT a safe extreme-depth edit): a native real-FFT kernel (pow2 = partially
bit-locked by golden digests → must stay bit-identical; non-pow2 = tolerance, free to rewrite). Explicit SIMD on
the sibling complex butterfly already REGRESSED (see 2026-06-26), so autovec is likely already optimal — the win
must come from the algorithm (fewer ops / native real symmetry), not from SIMD-ing the current one. Bench landed
as a permanent guard + the precise next-lever spec.

## 2026-06-27 - CORRECTION: non-pow2 FFT is ~7x SLOWER (prior "competitive/win" used a load-inflated JAX baseline) (SlateHarrier)

RETRACTION of the two prior non-pow2 FFT ledger entries this session. The JAX baselines I used (complex [4096,
1000]=62.4ms, [4096,1009]=99ms) were LOAD-INFLATED anomalies (the jax host was busy). Re-measured min-of-10,
stable: JAX complex [4096,1000]=**9.37ms**, [4096,1009]=**9.46ms**; JAX rfft [4096,1000]=**1.51ms**, [4096,1009]
=2.97ms. CORRECTED ratios (fj-lax same-binary numbers stable):
  - complex non-pow2 1000: fj-lax 68.5ms vs JAX 9.37 = **~7.3x SLOWER** (NOT 1.1x near-parity).
  - complex non-pow2 1009: fj-lax 72.9ms vs JAX 9.46 = **~7.7x SLOWER** (NOT a 1.36x win).
  - rfft non-pow2 1000: fj-lax 44.5ms vs JAX 1.51 = **~29x SLOWER** (worse than the 23x first reported).
So fj-lax's mixed-radix/bluestein kernel (complex AND real) is ~7-29x off pocketfft — a REAL tolerance-parity
gap, the lever being a faster mixed-radix + native real-FFT (multi-session). pow2 baselines (complex batched
0.58ms, rfft 5.89ms) stay valid — machine load only INFLATES timings, never deflates, so those mins are clean.
METHOD LESSON: the jax host load swings timings up to ~6.6x (62 vs 9.4ms); use min-of-10 and SANITY-CHECK any
ratio that implies a real FFT is >2x faster than its complex sibling (physically impossible) — that flagged the
anomaly here. Bench JAX baselines in fft.rs updated to the verified values.

## 2026-06-27 - threading the mixed-radix non-pow2 FFT is ~0-gain (memory-bound, not compute-bound) — reverted (SlateHarrier)

transform_batches_mixed_radix_iterative_soa was SERIAL across row-tiles (unlike the pow2/Bluestein batched paths
which thread). Hypothesis: the ~6.5x non-pow2 gap was missing parallelism. ATTEMPT: fanned row-chunks across
threads (per-thread scratch, bit-identical — FFT golden 60/0 PASS). RESULT: [4096,1000] serial 58-64ms ->
threaded 68-70ms = NO GAIN (slightly worse). REVERTED (stashed). The mixed-radix transform is MEMORY-bound
(multiple SoA digit-reversal + radix passes streaming the 64MB working set), so threading saturates bandwidth
without speedup — same class as the f64 gather controller-floor. The ~6.5x gap vs pocketfft is its
cache-blocking / fewer-pass mixed-radix kernel, NOT parallelism — a multi-session kernel-algorithm lever
(non-contained), tolerance-parity so legal but not a quick win. Do NOT re-attempt threading this path.

## 2026-06-27 - irfft mirrors rfft (7.7x pow2 / 3.7x non-pow2); FFT FAMILY characterization COMPLETE (SlateHarrier)

`bench_irfft_vs_jax` (min-of-10 JAX baselines): pow2 [4096,513]->1024 fj-lax **32.0ms vs JAX 4.18 = ~7.7x**;
non-pow2 [4096,501]->1000 fj-lax **9.6ms vs JAX 2.62 = ~3.7x**. Mirrors rfft — the real-FFT family lacks
pocketfft's native real kernel. COMPLETE FFT-FAMILY SCORECARD vs JAX (all measured this session, JAX baselines
re-verified min-of-10 where suspicious):
  - fft pow2 1D: ~1.38x (near-parity, GOLDEN-LOCKED bit-exact)
  - fft pow2 2D batched: ~4.9x (threaded+autovec-optimal; gap = FMA-policy + split-radix bit-lock)
  - fft non-pow2 smooth (mixed-radix): ~6.5x (MEMORY-bound; threading = no-gain; gap = cache-blocking)
  - fft non-pow2 prime (bluestein): ~7.7x (built on the FMA/split-radix-bound pow2)
  - rfft pow2 ~8.5x / non-pow2 ~29x; irfft pow2 ~7.7x / non-pow2 ~3.7x (no native real-FFT kernel)
EVERY FFT gap is NON-CONTAINED: FMA build-policy (deliberately avoided), pow2 split-radix bit-lock (golden
digests), cache-blocked mixed-radix, and a native real-FFT — all multi-session kernel/algorithm efforts or a
maintainer +fma decision, none a contained single-turn lever. The FFT family is the frankenjax perf frontier;
benches landed as permanent guards + the precise per-variant lever spec.

## 2026-06-27 - ifft 4.9x (mirrors fft); FFT family fully measured; rfft recombine already optimal-order (SlateHarrier)

`bench_ifft_2d_batched_vs_jax`: fj-lax [1024,1024] = 3.20ms vs JAX 0.651 = ~4.9x — mirrors the forward fft
batched (same transform_batches_dense inverse path + 1/N), as expected. The FFT family is now FULLY benched +
guarded (fft/ifft/rfft/irfft × pow2/non-pow2). REFINEMENT on the rfft lever: read vectorized_rfft_pow2_block —
its Hermitian recombine is ALREADY k-outer/b-inner (contiguous re-read, the obvious cache-friendly loop order)
and the half-FFT uses the SHARED optimized soa_radix2_butterfly_stages. So there is NO contained loop-order fix;
the 8.5x rfft gap is a deeper kernel inefficiency (the AoS<->SoA pack/recombine transposes + per-8-row-tile
amortization vs pocketfft's native real symmetry) needing profiling + a native real-FFT — multi-session. This
rules out the loop-order dead-end for a future effort. Every FFT variant remains non-contained (FMA-policy /
split-radix bit-lock / cache-blocked mixed-radix / native real-FFT).

## 2026-06-27 - cholesky (heavy-compute linalg) ~6.25x vs JAX/LAPACK — FMA + GEMM-tuning bound (non-contained) (SlateHarrier)

`bench_cholesky_2048_vs_jax`: fj-lax [2048,2048] = 199ms vs JAX dpotrf 31.85 = **~6.25x**. NOT a parallelism gap:
both the trailing SYRK update (cholesky_schur_update_lower, CHOLESKY_SCHUR_PARALLEL_MIN_OPS gate + thread::scope)
AND matmul_2d (matmul_2d_thread_count, row-block threads) are already threaded. The gap is (a) the no-+fma build
policy (~2x on the inner GEMM/SYRK FMAs, deliberately avoided — see fma-lever-policy-blocked) + (b) MKL's
superior microkernel/KC-pack cache-blocking (the OPEN 9zwwb GEMM lever, multi-session). Both NON-contained for a
single turn. This completes the HEAVY-COMPUTE picture: matmul, conv, linalg (cholesky/LU/QR/SVD/solve), and FFT
are ALL FMA-policy + GEMM-tuning bound — the single +fma maintainer decision is the unifying ~2x lever across
every one, and the deeper KC-blocked microkernel (9zwwb) the rest. The contained (per-op threading/vectorization/
algorithm) perf surface is exhausted; the remaining frankenjax gaps are this heavy-compute FMA/tuning class plus
the FFT kernel family — all multi-session or policy. Permanent guard bench landed.

## 2026-06-27 - QR [2048,2048] fj-lax WINS ~30x vs JAX (JAX-CPU QR is catastrophically slow) (SlateHarrier)

`bench_qr_2048_vs_jax`: fj-lax blocked-Householder QR [2048,2048] = **168.9ms vs JAX jnp.linalg.qr(reduced)
5061ms (min-of-6, re-verified) = ~30x FASTER**. JAX-CPU's QR lowering is catastrophically slow — the SAME class
as its sort/top_k/lexsort lowering that fj-lax already dominates (JAX r-only mode still 2476ms). fj-lax's QR
crushes it. This REVERSES the assumption that JAX/LAPACK linalg is uniformly faster: the heavy-compute linalg
picture is MIXED — JAX-CPU has a fast cholesky (dpotrf, fj-lax 6.25x slower) but a catastrophically slow QR
(fj-lax 30x faster). LIKELY MORE JAX-CPU-slow linalg wins to find (svd/eigh/solve eigen-iteration paths) — a
fresh vein. Permanent guard bench landed. This is a genuine fj-lax WIN on a common, important op.

## 2026-06-27 - svd/eigh near-parity (~1.1-1.2x); linalg-vs-JAX picture is MIXED, not uniform (SlateHarrier)

`bench_svd_eigh_1024_vs_jax`: svd [1024,1024] fj-lax 3049ms vs JAX 2470 = ~1.23x slower; eigh [1024,1024] fj-lax
1356ms vs JAX 1275 = ~1.06x slower. Both NEAR-PARITY (both iterative — bidiag+QR SVD / tridiag+QL eigh vs JAX's
equally-slow CPU iteration). REFINED heavy-linalg-vs-JAX picture (all [n>=1024] measured this session):
  - QR: fj-lax 169ms vs JAX 5061 = ~30x WIN (JAX-CPU QR catastrophically slow)
  - cholesky: fj-lax 199ms vs JAX 31.85 = 6.25x SLOWER (JAX dpotrf is fast/tuned)
  - svd: ~1.23x slower (near-parity, both iterative-slow)
  - eigh: ~1.06x slower (near-parity)
  - solve [2048]: JAX measured 1067ms (slow, like QR) — fj-lax not yet benched, likely a WIN (next).
So JAX-CPU linalg is INCONSISTENT: fast dpotrf, but catastrophically slow QR (+ slow solve, slow iterative
svd/eigh). fj-lax is competitive-or-winning on everything EXCEPT cholesky (the one JAX-fast path, FMA/tuning-
bound). Guard benches landed.

## 2026-06-27 - solve [2048] fj-lax WINS ~5.2x vs JAX; HEAVY-LINALG family fully mapped (4/5 win-or-parity) (SlateHarrier)

`bench_solve_2048_vs_jax`: fj-lax blocked-LU solve [2048,2048]x[2048,1] = **203.6ms vs JAX jnp.linalg.solve
1067ms = ~5.2x FASTER**. JAX-CPU's solve lowering is slow (QR-class, NOT the fast dpotrf/dgesv path). COMPLETE
heavy-linalg-vs-JAX scorecard (all measured this session, n>=1024):
  - QR [2048]: fj-lax 169ms / JAX 5061ms = **~30x WIN**
  - solve [2048]: fj-lax 204ms / JAX 1067ms = **~5.2x WIN**
  - eigh [1024]: fj-lax 1356ms / JAX 1275ms = ~1.06x (near-parity)
  - svd [1024]: fj-lax 3049ms / JAX 2470ms = ~1.23x (near-parity)
  - cholesky [2048]: fj-lax 199ms / JAX 31.85ms = ~6.25x SLOWER (the ONE loss — JAX dpotrf is FMA/tuned)
CONCLUSION: JAX-CPU linalg is NOT uniformly fast — it has one tuned path (cholesky/dpotrf) and is otherwise
slow-to-catastrophic (QR/solve/svd/eigh). fj-lax WINS QR + solve big, ties svd/eigh, loses only cholesky
(FMA/GEMM-tuning bound, the +fma policy + 9zwwb lever). This MAKES linalg a net fj-lax STRENGTH, correcting the
"LAPACK uniformly faster" assumption. All guard benches landed.

## 2026-06-27 - det 5.9x + slogdet 7.3x fj-lax WINS vs JAX; linalg is a fj-lax STRENGTH (only cholesky loses) (SlateHarrier)

`bench_det_slogdet_2048_vs_jax`: det [2048,2048] fj-lax **338ms vs JAX 1983 = ~5.9x WIN**; slogdet **304ms vs JAX
2233 = ~7.3x WIN**. JAX-CPU's LU-based determinant lowering is slow (QR/solve-class). Also measured JAX as slow
(fj-lax not yet benched, but same blocked-LU/QR core → expected wins): inv [2048]=2305ms, lstsq [2048,1024]=
8183ms (catastrophic — no fj-lax Lstsq/Inv PRIMITIVE, composed in jaxpr). FINAL linalg-vs-JAX verdict (all
measured this session):
  WINS: QR ~30x, slogdet ~7.3x, det ~5.9x, solve ~5.2x
  PARITY: eigh ~1.06x, svd ~1.23x
  LOSS: cholesky ~6.25x (the ONLY one — JAX dpotrf is the single tuned/FMA path; fj-lax FMA-policy-bound)
JAX-CPU linalg is broadly slow (its CPU lowering doesn't hit tuned LAPACK except dpotrf); fj-lax's blocked
GEMM-routed linalg WINS or ties everything except cholesky. Linalg is a NET fj-lax STRENGTH — correcting the
"LAPACK uniformly faster" assumption decisively. Guard benches landed.

## 2026-06-27 - LU 12.9x fj-lax WIN vs JAX; DIRECT-factorization linalg family is a clean fj-lax sweep (SlateHarrier)

`bench_lu_2048_vs_jax`: fj-lax blocked-LU [2048,2048] = **162.4ms vs JAX lu_factor 2087ms = ~12.9x WIN**.
Completes the DIRECT-factorization sweep — every one is a big fj-lax win:
  QR ~30x, LU ~12.9x, slogdet ~7.3x, det ~5.9x, solve ~5.2x.
Iterative ops near-parity (eigh ~1.06x, svd ~1.23x). ONLY loss: cholesky ~6.25x. KEY INSIGHT: the asymmetry is
entirely on JAX's side — JAX-CPU has exactly ONE tuned LAPACK path (cholesky/dpotrf, 32ms) and its LU/QR/det/
slogdet/solve/inv/lstsq lowerings are all slow (2-8s); fj-lax is CONSISTENT (~160-340ms for every direct
factorization, blocked GEMM-routed). So fj-lax LU (162ms) crushes JAX LU (2087ms) yet fj-lax cholesky (199ms)
loses to JAX cholesky (32ms) — same fj-lax speed, wildly different JAX speed. Linalg is decisively a NET fj-lax
STRENGTH (5 big direct-factorization wins + 2 parities vs 1 FMA-bound cholesky loss). All guard benches landed.

## 2026-06-27 - eig 1.58x slower (iterative); LINALG FAMILY fully measured — direct-factorizations win, iterative ties (SlateHarrier)

`bench_eig_1024_vs_jax`: fj-lax non-symmetric eig [1024,1024] (Francis double-shift + inverse-iter) = 9008ms vs
JAX jnp.linalg.eig 5715ms = ~1.58x SLOWER. eig joins the ITERATIVE eigenproblem class with svd/eigh — all
near-parity-to-slightly-slower (eigh 1.06x, svd 1.23x, eig 1.58x), both sides iteration-bound, no contained
lever. COMPLETE LINALG-vs-JAX MAP (this session, n>=1024):
  DIRECT FACTORIZATIONS (fj-lax WINS — JAX-CPU lowering slow): QR ~30x, LU ~12.9x, slogdet ~7.3x, det ~5.9x,
    solve ~5.2x.
  ITERATIVE EIGENPROBLEMS (near-parity, iteration-bound both sides): eigh ~1.06x, svd ~1.23x, eig ~1.58x.
  FMA/TUNED (fj-lax LOSS): cholesky ~6.25x (JAX's one tuned dpotrf path).
Net: fj-lax beats JAX-CPU on every direct factorization (5 big wins), ties the iterative eigenproblems, loses
only the FMA-bound cholesky. The eig 1.58x and cholesky 6.25x are the only linalg losses, both non-contained
(iterative-algorithm tuning / FMA-policy). Linalg fully characterized + guard-benched.

## 2026-06-27 - eig eigenvector loop THREADED — 9008->3820ms, flips 1.58x LOSS to 1.50x WIN vs JAX (SlateHarrier)

eig_qr_iteration computed the n eigenvectors in a SERIAL loop (each an independent O(n^2) inverse iteration on
the shared read-only Hessenberg h_hess/q0 — ~half of eig's O(n^3), embarrassingly parallel). Threaded it (fan
eigenvalues across cores into a disjoint vks slice, per-thread eig_eigenvector_hessenberg; the cheap O(n^2)
transpose-write stays serial). eig [1024,1024]: **9008 -> 3820ms (best) = ~2.36x internal**, flipping eig from
~1.58x SLOWER to **~1.50x FASTER vs JAX 5715ms**. Bit-identical (eig 22/0, clippy-clean). So eig JOINS the
linalg WIN column. Updated linalg verdict: WINS = QR 30x, LU 12.9x, slogdet 7.3x, det 5.9x, solve 5.2x, EIG
1.5x; near-parity = eigh 1.06x, svd 1.23x (their singular/eigen-vector accumulation may have the same serial-
loop lever — a NEXT dig); LOSS = only cholesky 6.25x (FMA). Guard bench updated.

## 2026-06-27 - complex eig eigenvector loop THREADED too — 1.27x WIN vs JAX (JAX complex-eig = 24.5s) (SlateHarrier)

Applied the same eigenvector-loop threading to the COMPLEX eig path (complex_eig_qr had the identical serial
loop calling complex_eig_eigenvector_hessenberg per eigenvalue on shared read-only H0/Q0). fj-lax complex-eig
[1024,1024] (threaded) = **19264ms vs JAX jnp.linalg.eig(complex128) 24492ms = ~1.27x WIN** (JAX complex-eig is
catastrophically slow, 24.5s). Bit-identical (eig 22/0, clippy-clean). Now BOTH eig paths win: real eig ~1.5x,
complex eig ~1.27x. The eigenvector-phase threading lever (eig-style clean post-loop) is exhausted — svd is
one-sided Jacobi (sequential cyclic sweeps, no clean post-loop; parallel-Jacobi = multi-session) and eigh's
1.06x headroom is tiny + its accumulation is interleaved. FINAL linalg verdict: WINS = QR 30x, LU 12.9x, slogdet
7.3x, det 5.9x, solve 5.2x, eig 1.5x, complex-eig 1.27x; PARITY = eigh 1.06x, svd 1.23x; LOSS = cholesky 6.25x
(FMA only). Linalg is overwhelmingly a fj-lax STRENGTH.

## 2026-06-27 - complex-solve 4.4x fj-lax WIN vs JAX — complex direct-factorizations also win (SlateHarrier)

`bench_complex_solve_2048_vs_jax`: fj-lax complex blocked-LU solve [2048,2048] = **824ms vs JAX
jnp.linalg.solve(complex128) 3643ms = ~4.4x WIN**. Confirms the JAX-CPU-slow / fj-lax-fast pattern extends to
COMPLEX direct factorizations (matches real solve 5.2x + complex eig 1.27x). FULL linalg-vs-JAX win tally now:
  WINS: QR ~30x, LU ~12.9x, slogdet ~7.3x, det ~5.9x, solve ~5.2x, complex-solve ~4.4x, eig ~1.5x,
    complex-eig ~1.27x (8 wins)
  PARITY: eigh ~1.06x, svd ~1.23x
  LOSS: cholesky ~6.25x (the ONLY one — FMA/dpotrf-tuned)
Linalg (real AND complex) is decisively a fj-lax STRENGTH: JAX-CPU's CPU lowering is fast ONLY for cholesky;
every other factorization (LU/QR/det/slogdet/solve/eig, real+complex) is 1.3-30x slower than fj-lax's blocked
GEMM-routed + threaded kernels. The "LAPACK uniformly faster" assumption is comprehensively disproven.

## 2026-06-27 - core matmul 2.92x slower (335 vs 975 GFLOP/s) — fj-lax near no-FMA ceiling; gap is purely FMA (SlateHarrier)

`bench_matmul_2048_vs_jax`: fj-lax matmul_2d f64 [2048,2048] = 51.35ms = **335 GFLOP/s** vs JAX a@b 17.61ms =
**975 GFLOP/s = ~2.92x SLOWER**. KEY: JAX-CPU matmul DOES hit tuned FMA BLAS (975 GFLOP/s) — unlike its slow
linalg lowering (which fj-lax beats). fj-lax's 335 GFLOP/s is ~75% of the f64 AVX2 NO-FMA ceiling (~448 GFLOP/s
= 32c x 3.5GHz x 4 flops/cyc) — so its blocked+packed+threaded microkernel is ALREADY near-optimal for the
no-+fma policy; the ~3x gap is FUNDAMENTALLY the deliberately-avoided FMA (fused mul-add doubles flops/cycle).
This is the DEFINITIVE heavy-compute datapoint: matmul 2.92x is the GEMM floor, and cholesky 6.25x / conv build
on it (SYRK + panels + im2col add overhead). The unifying lever is the single +fma maintainer decision (~2x
across matmul/cholesky/conv/GEMM-linalg); fj-lax cannot close it under forbid-unsafe + no-+fma. NON-contained.
Guard bench landed. (Contrast: JAX-CPU's LU/QR/det/solve do NOT hit BLAS — they're slow XLA lowerings fj-lax
beats 5-30x; only the raw GEMM + dpotrf get the tuned FMA path.)

## 2026-06-27 - complex-QR 8.5x fj-lax WIN; complex direct-factorizations fully won (SlateHarrier)

`bench_complex_qr_1024_vs_jax`: fj-lax complex blocked-Householder QR [1024,1024] = **221.8ms vs JAX 1876ms =
~8.5x WIN**. Completes the complex direct-factorization sweep (complex QR 8.5x, complex-solve 4.4x, complex-eig
1.27x — all win, mirroring real). FULL LINALG WIN TALLY (9 wins, real+complex):
  QR 30x, LU 12.9x, complex-QR 8.5x, slogdet 7.3x, det 5.9x, solve 5.2x, complex-solve 4.4x, eig 1.5x,
  complex-eig 1.27x.
  PARITY: eigh 1.06x, svd 1.23x. LOSS: cholesky 6.25x (FMA/dpotrf). Core matmul 2.92x (FMA floor).
DEFINITIVE: JAX-CPU's CPU backend hits tuned FMA BLAS ONLY for raw matmul + cholesky/dpotrf; every higher-level
linalg factorization (LU/QR/det/slogdet/solve/eig, real AND complex) is a slow XLA lowering that fj-lax's
blocked GEMM-routed + threaded kernels beat 1.3-30x. Linalg is overwhelmingly a fj-lax STRENGTH. The only
heavy-compute losses (matmul/cholesky/conv) are the FMA-policy floor (non-contained). Guard bench landed.

## 2026-06-27 - KEEP: dense f64 non-pow2 RFFT skips full complex lift; 1.14x internal, ratio 2.45x -> 2.14x vs JAX (ProudSalmon)

`eval/rfft_batch_64x1000_f64` had a contained layout/allocator gap before the Bluestein work: dense real input was
first lifted into a full `Vec<(f64,f64)>`, then immediately copied again into per-row padded FFT buffers. Added a
dense-F64, non-power-of-two RFFT path that reads the real slice directly, packs two real rows into one complex
transform, unpacks the Hermitian halves, and preserves the existing bit-exact row-block invariant.

Evidence:
  - FrankenJAX baseline, `rch exec -- cargo bench -p fj-lax --profile release --bench lax_baseline --
    eval/rfft_batch_64x1000_f64`: **544.23 us mean** (`[512.61, 544.23, 579.38] us`).
  - Candidate, same crate/bench/target dir via `rch exec` local fallback: **475.83 us mean**
    (`[461.24, 475.83, 486.59] us`; Criterion change CI `[-14.450%, -10.017%, -5.4446%]`).
  - Exact JAX CPU/x64 comparator for the same `64x1000` data and `jnp.fft.rfft`: **222.00188 us mean**.
  - Ratio vs JAX improved from **2.45x slower** to **2.14x slower**; internal mean speedup **1.14x**.

Validation:
  - `rch exec -- cargo test -p fj-conformance --profile release -- --nocapture`: green.
  - `rch exec -- cargo test -p fj-lax --profile release threaded_bluestein_batch_rfft_matches_serial_row_block --lib -- --nocapture`: green.
  - `rustfmt --edition 2024 --check crates/fj-lax/src/fft.rs`: green.
  - `cargo bench --release` is not accepted by this Cargo for `bench` (`unexpected argument '--release'`), so the
    repo's already-used release-profile equivalent was used: `cargo bench -p fj-lax --profile release ...`.
