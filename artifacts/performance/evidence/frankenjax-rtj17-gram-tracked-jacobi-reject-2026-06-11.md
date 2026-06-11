# frankenjax-rtj17 Gram-Tracked Jacobi SVD Rejection

## Target

- Bead: `frankenjax-rtj17`
- Hotspot: `fj-lax` real thin SVD, Criterion `linalg/svd_48x48_f64`
- Baseline target: current one-sided Jacobi SVD on the deterministic 48x48 F64 benchmark matrix.
- Candidate primitive: square real-SVD route that maintained the working Gram matrix in a cache-resident `n x n` buffer during one-sided Jacobi sweeps, while still applying rotations to `W = A * V` and `V`.

This was a deeper route than the rejected DBDSQR family: it kept the high-relative-accuracy one-sided Jacobi contract, but tried to replace repeated per-pair column dot scans with cache-resident Gram reads plus O(n) row/column Gram updates after accepted rotations.

## Baselines

Initial routing baseline on `vmi1153651`:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1153651 rch exec -- cargo bench -j 1 -p fj-lax --bench lax_baseline -- linalg/svd_48x48_f64
linalg/svd_48x48_f64 time: [2.2334 ms 2.3095 ms 2.4122 ms]
```

The candidate benchmark landed on `vmi1152480`, so a clean matched baseline was taken from the detached scratch worktree at `a73c3cd0`:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -j 1 -p fj-lax --bench lax_baseline -- linalg/svd_48x48_f64
linalg/svd_48x48_f64 time: [1.3401 ms 1.3704 ms 1.4018 ms]
```

## Candidate

Candidate benchmark on `vmi1152480`:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -j 1 -p fj-lax --bench lax_baseline -- linalg/svd_48x48_f64
linalg/svd_48x48_f64 time: [2.3779 ms 2.5175 ms 2.6666 ms]
```

Same-worker delta on `vmi1152480`:

- Median ratio: `1.3704 / 2.5175 = 0.54x`
- Conservative ratio: `1.3401 / 2.6666 = 0.50x`
- The candidate is about `1.84x` slower by median time.

## Behavior Proof

The temporary candidate proof used the same 48x48 trig matrix as the benchmark. It checked:

- reconstruction against the input matrix,
- descending singular values,
- singular-value parity against the existing Jacobi implementation,
- V orthonormality,
- U orthonormality for nonzero singular columns and the existing zero-column policy for zero singular values,
- deterministic golden-output digest over `sigma`, `U`, and `V` bits.

Focused RCH proof:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -j 1 -p fj-lax --lib svd_gram_tracked_square_profile_shape_golden_digest -- --nocapture
test linalg::tests::svd_gram_tracked_square_profile_shape_golden_digest ... ok
```

Temporary golden digest:

```text
99429ef86a52db54d79dbedf4041beed8ef08188fab9e74cb115d0972c5e4759
```

No source hunk or temporary test was kept. `crates/fj-lax/src/linalg.rs` is restored to the production source, so this rejection commit changes no runtime behavior.

## Rejection Gate

Score:

```text
Impact 0 x Confidence 5 / Effort 3 = 0.0
```

Decision: rejected. The full working-Gram route moved convergence checks into cache, but the per-accepted-rotation Gram row/column update dominated this 48x48 profile shape and regressed the same-worker benchmark.

## Next Route

Do not repeat full working-Gram tracking for `linalg/svd_48x48_f64`.

The next SVD primitive should first profile the sweep structure directly:

- count sweeps, accepted rotations, skipped rotations, and zero/near-zero singular columns for the 48x48 target;
- then attack a different route such as deterministic active-pair compaction or disjoint-pair wavefront sweeps only if the counters show enough skipped/independent work to remove whole pair visits, not just make each visit heavier.

Target ratio remains at least `2.0x` on a same-worker `linalg/svd_48x48_f64` comparison with reconstruction/order/sign/RNG contracts pinned by a golden digest.
