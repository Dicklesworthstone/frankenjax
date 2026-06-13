# frankenjax-14w36: cache all-used DCE linear-chain plans

Date: 2026-06-13
Agent: BeigeMouse
Crate: `fj-interpreters`
Target: `dce/all_used/1000eq`
Bead: `frankenjax-14w36`

## Profile target

Post-`frankenjax-wtrks` RCH reprofile ranked the DCE all-used 1000-equation row as the next
profile-backed `fj-interpreters` target:

- RCH `vmi1152480`, pre-change full reprofile:
  `dce/all_used/1000eq [2.1322 us 2.1870 us 2.2483 us]`
- RCH `vmi1153651`, focused pre-change baseline:
  `dce/all_used/1000eq [4.4350 us 4.6879 us 4.9982 us]`

The accepted comparison uses the same-worker `vmi1152480` row because the after-run selected that
worker.

## Lever

Add one bounded thread-local positive cache for the already-validated all-used linear-chain DCE
shape. On a cache miss, the original recognizer still proves the exact shape before inserting the
Jaxpr canonical fingerprint. On a hit, DCE returns the same `(jaxpr.clone(), vec![true])` that the
validated linear-chain path returns.

Alien primitive: incremental/self-adjusting computation, applied as a differential-dataflow-style
reused retained-set plan for repeated structural Jaxprs.

## Benchmark gate

- Before, RCH `vmi1152480`:
  `dce/all_used/1000eq [2.1322 us 2.1870 us 2.2483 us]`
- After, RCH `vmi1152480`:
  `dce/all_used/1000eq [637.55 ns 668.64 ns 697.96 ns]`
- Median ratio: `3.27x`
- Score: `3.27 impact * 0.95 confidence / 0.50 effort = 6.21`

Decision: keep. Score is above the `2.0` threshold.

## Isomorphism proof

- Cache entries are inserted only after `try_dce_all_used_linear_chain` validates the exact current
  accepted surface: all outputs used, one input, no constvars, one output, nonempty equations, one
  output per equation, and linear consumption of the current chain variable.
- Cache keys are `Jaxpr::canonical_fingerprint()`, which includes invars, constvars, outvars,
  effects, equations, params, and nested sub-Jaxpr fingerprints.
- A cache hit returns the same value as the validated path: the original cloned Jaxpr and
  `used_inputs == [true]`.
- Equation order and output order are unchanged because the original Jaxpr is cloned.
- No primitive execution is added or removed, so floating-point evaluation, tie-breaking, and RNG
  behavior are unchanged.
- The cache is bounded and thread-local. Eviction only changes whether the miss path revalidates;
  it cannot change DCE output.

Golden proof:

- Command:
  `rch exec -- cargo test -j 1 -p fj-interpreters --lib test_dce_all_used_large_chain_golden_hash -- --nocapture`
- Result: passed on RCH `vmi1152480`
- Golden SHA-256:
  `3729e2d5cc19c0abec46fb5b188cc7576b9853ee7d0cd523f3656b1ac57e8ad8`

## Validation

- `rustfmt --edition 2024 --check crates/fj-interpreters/src/partial_eval.rs`: passed
- `git diff --check`: passed
- `rch exec -- cargo check -j 1 -p fj-interpreters --lib`: passed on RCH `vmi1227854`
- `rch exec -- cargo clippy -j 1 -p fj-interpreters --lib -- -D warnings`: blocked by
  pre-existing `crates/fj-trace/src/lib.rs:1808` unused variable before the edited crate's lint
  surface.
- `ubs crates/fj-interpreters/src/partial_eval.rs`: nonzero due pre-existing file-wide inventory;
  its built-in formatting, clippy, cargo check, test-build, audit, and deny sections were clean.

## Reprofile route

Reprofile after landing. If DCE remains a top row, route to a reusable retained-set plan keyed by a
structural compact fingerprint rather than adding another exact-shape DCE predicate.
