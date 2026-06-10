# fj-ad JVP dense store pass

## Target

- Directive: `frankenjax-mcqr` no-gaps root
- Ready perf beads were occupied or policy-blocked: `frankenjax-lu4yw` was reserved by BoldFalcon, `frankenjax-mcqr.30`/`frankenjax-p1vbf` were assigned to IcyGlacier, and `frankenjax-cz0g0` requires a policy decision.
- Profile-backed lane: fj-ad Criterion `ad_baseline`, forward-mode JVP rows.
- Lever: replace `jvp_inner`'s `BTreeMap<VarId, Value>` primal/tangent environments with the existing `AdValueStore` dense/sparse VarId-indexed store.

## Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_QUEUE_WHEN_BUSY=1 RCH_PRIORITY=normal RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_REQUIRE_REMOTE,RCH_QUEUE_WHEN_BUSY,RCH_PRIORITY rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-peachlion-fjad-baseline cargo bench -j 1 -p fj-ad --bench ad_baseline -- --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot
```

Worker: `vmi1227854`

| Benchmark | Baseline mean |
| --- | ---: |
| `ad/jvp_square` | `617.43 ns` |
| `ad/jvp_poly_x3+x2+x` | `1.0781 us` |
| `ad/jvp_sin_cos_mul` | `926.39 ns` |

## Candidate

Command:

```text
RCH_FORCE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 RCH_QUEUE_WHEN_BUSY=1 RCH_PRIORITY=normal RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_FORCE_REMOTE,RCH_WORKER,RCH_WORKERS,RCH_QUEUE_WHEN_BUSY,RCH_PRIORITY rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-peachlion-fjad-candidate cargo bench -j 1 -p fj-ad --bench ad_baseline -- 'ad/jvp_(square|poly_x3\+x2\+x|sin_cos_mul)' --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot
```

Worker: `vmi1227854`

| Benchmark | Candidate mean | Speedup |
| --- | ---: | ---: |
| `ad/jvp_square` | `371.99 ns` | `1.66x` |
| `ad/jvp_poly_x3+x2+x` | `967.92 ns` | `1.11x` |
| `ad/jvp_sin_cos_mul` | `743.63 ns` | `1.25x` |

Score: Impact `3` x Confidence `4` / Effort `2` = `6.0`.

## Isomorphism Proof

- Ordering: equation traversal, input resolution, primitive execution, and output collection order are unchanged.
- Tie-breaking: not applicable; no comparator, sort, or reduction ordering changes.
- Floating point: no primitive arithmetic or JVP rule changed. Values are cloned from the same `Value` instances and fed to the same `eval_primitive_multi`/`jvp_rule` calls in the same order.
- RNG: not applicable; JVP path and these benchmarks have no RNG state.
- Fallback: `AdValueStore::for_jaxpr` keeps the previous sparse `BTreeMap` behavior for huge or overflowed VarId spaces.
- Golden digest: `0de357fc34872ec8485ed674662a533718400a68b9e690ae8016f84b6fba4b31` from `jvp_dense_store_matches_sparse_store_and_golden_sha256`, which compares dense-store JVP against the sparse high-VarId fallback and hashes the scalar primal/tangent bit strings.

## Validation

- `rustfmt --edition 2024 --check crates/fj-ad/src/lib.rs`
- `git diff --check -- crates/fj-ad/src/lib.rs`
- `RCH_FORCE_REMOTE=1 ... cargo check -p fj-ad --all-targets` on `vmi1227854`: pass
- `RCH_FORCE_REMOTE=1 ... cargo clippy -p fj-ad --all-targets --no-deps -- -D warnings` on `vmi1227854`: pass
- `RCH_FORCE_REMOTE=1 ... cargo test -j 1 -p fj-ad jvp_dense_store_matches_sparse_store_and_golden_sha256 --lib -- --nocapture` on `vmi1227854`: pass
- `rch exec ... cargo test -p fj-ad jvp --lib -- --nocapture`: pass with local fallback after test-slot admission refusal (`99 passed`)

Known unrelated warnings during remote validation: `fj-trace` unused `num_spatial`; `fj-lax` unused `cell_f64_reference`.
