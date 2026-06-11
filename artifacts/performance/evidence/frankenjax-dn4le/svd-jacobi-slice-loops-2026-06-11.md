# frankenjax-dn4le: SVD Jacobi slice-index hot loops

Date: 2026-06-11
Agent: BeigeMouse
Target crate: fj-lax
Target bench: `linalg/svd_48x48_f64`
Worker: `vmi1227854`

## Target

`frankenjax-dn4le` was opened after two cached-state Jacobi candidates preserved
the 48x48 SVD golden digest but failed the keep gate. The deeper route remains a
rank-revealing or blocked SVD primitive, but the fresh profile path still showed
the current cyclic one-sided Jacobi inner loops paying repeated column-base
index arithmetic on every row visit.

This pass keeps the public algorithm exactly the same and changes one lever:
derive non-overlapping column slices once per `(p, q)` pair, then index those
slices inside the existing alpha/beta/gamma and rotation loops.

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- \
  cargo bench -p fj-lax --bench lax_baseline -- \
  linalg/svd_48x48_f64 --sample-size 20 --warm-up-time 1 --measurement-time 3
```

Clean baseline after `frankenjax-v2qyi` closeout:

```text
linalg/svd_48x48_f64 [1.1364 ms 1.1640 ms 1.1992 ms]
```

## Candidate

Same command and worker after the slice-index lever:

```text
linalg/svd_48x48_f64 [1.0656 ms 1.0799 ms 1.0954 ms]
```

Median ratio: `1.078x`.
Conservative lower/upper ratio: `1.037x`.

Score: `5.4 = Impact 1.08 * Confidence 5 / Effort 1`. The change is small,
profile-directed, same-worker, and has exact public-output proof.

## Isomorphism proof

- Pair order is unchanged: cyclic `(p, q)` traversal remains ascending `p`, then
  ascending `q`.
- Row order is unchanged inside every column dot product and rotation.
- Floating-point expression order is unchanged for `alpha`, `beta`, `gamma`,
  `c`, `s`, W rotations, and V accumulation.
- The same convergence predicate, sweep limit, skip threshold, and singular
  vector fallback execute in the same positions.
- The lever replaces `w[p * m + i]`, `w[q * m + i]`, `v[p * n + i]`, and
  `v[q * n + i]` with slices computed from the same bases.
- There is no RNG, no tie-breaking surface, no data reordering, and no public
  contract change.

Golden digest from the focused profile-shape proof remained:

```text
6f1b0069586dda5b23d377bbb171a18ac0e24b6e0309dabc4ad0e0d2d1864d90
```

The counter profile also remained the same: `13` sweeps, `14664` pair visits,
`10264` accepted rotations, `4400` skipped rotations, and active-pair counts
`[1128,1128,1128,1128,1059,957,955,961,956,711,149,4,0]`.

## Validation

Passed:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- \
  cargo test -p fj-lax --lib 48x48 -- --nocapture
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- \
  cargo check -p fj-lax --lib
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- \
  cargo clippy -p fj-lax --lib -- -D warnings
rustfmt --edition 2024 --check crates/fj-lax/src/linalg.rs
git diff --check
```

`ubs crates/fj-lax/src/linalg.rs` exited nonzero from pre-existing file-wide
panic/unwrap/direct-indexing inventory. Its built-in fmt, clippy, check,
test-build, cargo-audit, and cargo-deny sections were clean.

## Next primitive

This pass does not claim the rank-revealing or blocked SVD route is complete.
The follow-up should attack a fundamentally different SVD primitive with a
proof harness, not another cached-state Jacobi micro-lever:

- deterministic rank-revealing reduction or deflation before Jacobi;
- blocked/cache-resident Jacobi with explicit public-output fallback;
- communication-avoiding or tiled SVD reduction if the golden-output contract can
  be pinned before production routing.
