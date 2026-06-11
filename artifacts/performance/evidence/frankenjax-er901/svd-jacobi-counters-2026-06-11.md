# frankenjax-er901: SVD Jacobi Counter Baseline

## Scope

- Bead: `frankenjax-er901`
- Target: `fj-lax` real one-sided Jacobi SVD, `linalg/svd_48x48_f64`
- Pass: `pass225` measurement/proof setup
- Production behavior change: none. This pass adds test-only profiling counters and a public-output golden digest.

## Baseline

Command:

```bash
rch exec -- cargo bench -p fj-lax --bench lax_baseline -- linalg/svd_48x48_f64 --sample-size 20 --warm-up-time 1 --measurement-time 3
```

Worker/result:

- Worker: `vmi1227854`
- Criterion time: `[1.1656 ms 1.1996 ms 1.2447 ms]`
- RCH target: `.rch-target-vmi1227854-job-29883510449766842-1781214107606628679-0`

## Golden Proof

Command:

```bash
rch exec -- cargo test -p fj-lax --lib 48x48 -- --nocapture
```

Result:

- Worker: `vmi1227854`
- Tests: `2 passed`
- Golden digest: `6f1b0069586dda5b23d377bbb171a18ac0e24b6e0309dabc4ad0e0d2d1864d90`

The digest covers the public `eval_svd` 48x48 F64 outputs `[U, S, Vt]` as F64 output bits.

## Counter Findings

`one_sided_jacobi_svd_real_48x48_profile_counters`:

- Sweeps: `13`
- Visited pairs: `14664`
- Accepted rotations: `10264`
- Skipped rotations: `4400`
- Per-sweep pair visits: `[1128, 1128, 1128, 1128, 1128, 1128, 1128, 1128, 1128, 1128, 1128, 1128, 1128]`
- Per-sweep active pairs: `[1128, 1128, 1128, 1128, 1059, 957, 955, 961, 956, 711, 149, 4, 0]`
- Exactly zero singular columns: `0`
- Near-zero singular columns at the SVD guard: `44`

## Interpretation

The current cyclic Jacobi path still visits all `48 * 47 / 2 = 1128` pairs every sweep. The final three sweeps visit `3384` pairs but accept only `153` rotations, and the profile fixture has `44` near-zero singular columns. This is a profile-backed signal for a structural pass that removes whole pair visits, either by deterministic active-pair compaction or a disjoint-pair wavefront schedule with inactive-column pruning.

This pass does not claim a speedup. It establishes the profile and proof harness for the next one-lever optimization.

## Isomorphism

- Ordering preserved: yes; production code is unchanged.
- Tie-breaking unchanged: yes; production sort and initial ordering are unchanged.
- Floating-point behavior: unchanged; only test-only replay counters and a golden digest were added.
- RNG: N/A.
- Golden output: `6f1b0069586dda5b23d377bbb171a18ac0e24b6e0309dabc4ad0e0d2d1864d90`.

## Alien Primitive Route

- Graveyard mapping: communication-avoiding numerical linear algebra contracts (`alien_cs_graveyard.md` section 9.6) and profile-first opportunity gating.
- Candidate next lever: deterministic active-pair compaction after the first few full sweeps, preserving the exact current pair order among retained pairs; fallback to full cyclic sweep if the active set bookkeeping would alter convergence or output digest.
- Target ratio: at least `1.3x` on the 48x48 row for a measurement-only next step; keep only if Score `>= 2.0`.

## Validation

- `rch exec -- cargo test -p fj-lax --lib 48x48 -- --nocapture`: passed.
- `rch exec -- cargo check -p fj-lax --lib`: passed.
- `rch exec -- cargo clippy -p fj-lax --lib -- -D warnings`: passed.
- `rustfmt --edition 2024 --check crates/fj-lax/src/linalg.rs`: passed.
- `git diff --check`: passed.
- `cargo fmt -p fj-lax --check`: blocked by pre-existing formatting drift outside this pass in `crates/fj-lax/benches/lax_baseline.rs` and `crates/fj-lax/src/tensor_contraction.rs`.
- `ubs crates/fj-lax/src/linalg.rs`: nonzero from pre-existing file-wide panic/unwrap/indexing inventory; UBS built-in fmt, clippy, cargo check, and test-build sections were clean.
