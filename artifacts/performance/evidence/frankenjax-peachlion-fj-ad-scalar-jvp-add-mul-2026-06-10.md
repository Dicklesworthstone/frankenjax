# fj-ad scalar Add/Mul JVP fast path

## Target

- Directive: `frankenjax-mcqr` no-gaps root.
- Ready perf beads in `fj-core`/`fj-lax` remained policy-blocked or owned by another agent, so this pass continues the reserved, profile-backed fj-ad lane.
- Profile-backed hotspot after pass 3: `ad/jvp_poly_x3+x2+x` was the largest current fj-ad Criterion row, with `ad/jvp_square` sharing the same scalar Add/Mul JVP structure.
- Lever: add a narrow scalar-F64 Add/Mul-only Jaxpr JVP fast path. It executes the same primal operations and product-rule tangent operations directly in equation order, and falls back before slot setup for nonmatching Jaxprs.

## Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 RCH_QUEUE_WHEN_BUSY=1 RCH_PRIORITY=normal RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_REQUIRE_REMOTE,RCH_WORKER,RCH_WORKERS,RCH_QUEUE_WHEN_BUSY,RCH_PRIORITY rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-peachlion-fjad-pass4-profile cargo bench -j 1 -p fj-ad --bench ad_baseline -- --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot
```

Worker: `vmi1227854`

| Benchmark | Baseline mean |
| --- | ---: |
| `ad/jvp_square` | `436.23 ns` |
| `ad/jvp_poly_x3+x2+x` | `1.0487 us` |
| `ad/jvp_sin_cos_mul` | `836.84 ns` |

## Candidate

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 RCH_QUEUE_WHEN_BUSY=1 RCH_PRIORITY=normal RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_REQUIRE_REMOTE,RCH_WORKER,RCH_WORKERS,RCH_QUEUE_WHEN_BUSY,RCH_PRIORITY rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-rch-peachlion-fjad-pass4-candidate-final cargo bench -j 1 -p fj-ad --bench ad_baseline -- 'ad/jvp_(square|poly_x3\+x2\+x|sin_cos_mul)' --sample-size 20 --measurement-time 2 --warm-up-time 1 --noplot
```

Worker: `vmi1227854`

| Benchmark | Candidate mean | Speedup |
| --- | ---: | ---: |
| `ad/jvp_square` | `113.42 ns` | `3.85x` |
| `ad/jvp_poly_x3+x2+x` | `138.00 ns` | `7.60x` |
| `ad/jvp_sin_cos_mul` | `793.23 ns` | `1.05x` |

Score: Impact `5` x Confidence `3` / Effort `2` = `7.5`.

## Isomorphism Proof

- Ordering: Jaxpr equations are traversed in the same order; output collection order follows `jaxpr.outvars`.
- Tie-breaking: not applicable; no comparator, sort, selection, or map iteration order changed.
- Floating point: Add and Mul primals use the same scalar F64 operations as the scalar primitive fast path. Mul tangents preserve the generic JVP operation order exactly: `da * b`, then `a * db`, then their sum. Add tangents preserve `da + db`.
- Custom rules: Jaxpr custom JVP lookup still happens before the fast path. Primitive custom Add/Mul JVPs disable the fast path. Params, effects, sub-Jaxprs, non-F64 values, and high VarIds fall back to the generic interpreter.
- RNG: not applicable; this path has no RNG state.
- Golden digest: `jvp_dense_store_matches_sparse_store_and_golden_sha256` still pins `0de357fc34872ec8485ed674662a533718400a68b9e690ae8016f84b6fba4b31`, comparing low-VarId fast-path output against high-VarId fallback output.

## Validation

- `rustfmt --edition 2024 --check crates/fj-ad/src/lib.rs`: pass
- `git diff --check -- crates/fj-ad/src/lib.rs artifacts/performance/evidence/frankenjax-peachlion-fj-ad-scalar-jvp-add-mul-2026-06-10.md`: pass
- `RCH ... cargo check -j 1 -p fj-ad --all-targets`: pass on `vmi1227854`
- `RCH ... cargo clippy -j 1 -p fj-ad --all-targets --no-deps -- -D warnings`: pass on `vmi1227854`
- `RCH ... cargo test -j 1 -p fj-ad jvp_dense_store_matches_sparse_store_and_golden_sha256 --lib -- --nocapture`: pass on `vmi1227854`
- Known unrelated warnings during remote validation: `fj-lax` dead helper `cell_f64_reference`; `fj-trace` unused local `num_spatial`.
- `ubs crates/fj-ad/src/lib.rs artifacts/performance/evidence/frankenjax-peachlion-fj-ad-scalar-jvp-add-mul-2026-06-10.md`: exit 1 on pre-existing `fj-ad` whole-file inventory. UBS sub-gates reported no unsafe blocks, clean formatting, clean clippy, clean cargo check, and clean test build. Added diff contains no panic/unwrap/expect/unreachable, unchecked numeric casts, or direct indexing.
