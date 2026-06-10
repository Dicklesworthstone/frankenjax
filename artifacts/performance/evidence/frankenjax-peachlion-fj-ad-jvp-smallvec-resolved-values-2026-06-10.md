# fj-ad JVP SmallVec resolved-values pass

## Target

- Directive: `frankenjax-mcqr` no-gaps root.
- Ready perf beads remained occupied or policy-blocked, so this pass continues on the profile-backed fj-ad lane under the active fj-ad reservation.
- Profile-backed lane: fj-ad Criterion `ad_baseline`, JVP rows after the vector-reduce split-loop pass.
- Lever: replace the per-equation `Vec<Value>` scratch buffers in `jvp_inner` with the crate's existing `TapeValues = SmallVec<[Value; 2]>`. This keeps primitive input order and slice semantics unchanged while avoiding heap allocation for the common unary/binary JVP equations.

## Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 RCH_PRIORITY=normal RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY,RCH_PRIORITY rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-peachlion-fjad-pass2-baseline cargo bench -j 1 -p fj-ad --bench ad_baseline -- --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot
```

Worker: `vmi1227854`

| Benchmark | Baseline mean |
| --- | ---: |
| `ad/jvp_square` | `398.50 ns` |
| `ad/jvp_poly_x3+x2+x` | `1.0652 us` |
| `ad/jvp_sin_cos_mul` | `929.25 ns` |

## Candidate

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 RCH_QUEUE_WHEN_BUSY=1 RCH_PRIORITY=normal RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_REQUIRE_REMOTE,RCH_WORKER,RCH_WORKERS,RCH_QUEUE_WHEN_BUSY,RCH_PRIORITY rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-peachlion-fjad-pass3-candidate cargo bench -j 1 -p fj-ad --bench ad_baseline -- 'ad/jvp_(square|poly_x3\+x2\+x|sin_cos_mul)' --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot
```

Worker: `vmi1227854`

| Benchmark | Candidate mean | Speedup |
| --- | ---: | ---: |
| `ad/jvp_square` | `384.69 ns` | `1.04x` |
| `ad/jvp_poly_x3+x2+x` | `1.0036 us` | `1.06x` |
| `ad/jvp_sin_cos_mul` | `813.92 ns` | `1.14x` |

Score: Impact `2` x Confidence `3` / Effort `1` = `6.0`.

## Isomorphism Proof

- Ordering: atom traversal order, primitive evaluation order, output insertion order, and final output collection order are unchanged.
- Tie-breaking: not applicable; no comparator, sort, or selection changed.
- Floating point: no arithmetic or JVP rule changed. The same cloned `Value`s are presented to `eval_primitive_multi`, `jvp_rule`, and `jvp_rule_multi` in the same order; only the temporary container changed from heap `Vec` to inline `SmallVec`.
- RNG: not applicable; this path has no RNG state.
- Resource envelope: allocation count drops for unary/binary equations; overflow behavior remains bounded because `SmallVec` spills to heap for arities above two.
- Golden digest: `jvp_dense_store_matches_sparse_store_and_golden_sha256` pins `0de357fc34872ec8485ed674662a533718400a68b9e690ae8016f84b6fba4b31` for the polynomial JVP and compares dense low-VarId and sparse high-VarId paths.

## Validation

- `rustfmt --edition 2024 --check crates/fj-ad/src/lib.rs`: pass
- `git diff --check -- crates/fj-ad/src/lib.rs`: pass
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 ... rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-peachlion-fjad-pass3-validate cargo check -j 1 -p fj-ad --all-targets`: pass on `vmi1227854`
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 ... rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-peachlion-fjad-pass3-validate cargo clippy -j 1 -p fj-ad --all-targets --no-deps -- -D warnings`: pass on `vmi1227854`
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 ... rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-peachlion-fjad-pass3-validate cargo test -j 1 -p fj-ad jvp_dense_store_matches_sparse_store_and_golden_sha256 --lib -- --nocapture`: pass on `vmi1227854`
- Known unrelated warnings during remote validation: `fj-lax` dead helper `cell_f64_reference`; `fj-trace` unused local `num_spatial`.
