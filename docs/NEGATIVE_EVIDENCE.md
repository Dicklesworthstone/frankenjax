# Negative Evidence Ledger

Canonical project ledger: `../evidence/perf/negative_evidence_ledger.md`.

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
