# FrankenJAX Pass 4: Borrowed API Dispatch

Agent: BoldFalcon
Benchmark base: e859a6e7
Landing base: 980c74af
Surface: fj-api -> fj-dispatch -> fj-core transform proof path
Target: value_and_grad_runtime/shared/deep_100_nodes

## Profile target

The post-pass-3 Criterion profile was recorded with:

```text
perf record -F 997 -g --call-graph dwarf \
  -o perf-fj-api-deep100-pass4.data \
  -- /data/tmp/cargo-target/release/deps/api_overhead-1630fb8ebf560d30 \
  --bench value_and_grad_runtime/shared/deep_100_nodes --profile-time 10 --noplot
```

Top self-time buckets included:

```text
12.01%  _int_malloc
10.42%  fj_core::verify_transform_composition
 8.25%  fj_ad::forward_with_tape
 6.73%  fj_ad::backward
 5.68%  core::fmt::write
 5.55%  sha2::sha256::x86::digest_blocks
 2.39%  fj_ad::AdValueStore::for_jaxpr
```

The actionable profile-backed target was allocation and canonical fingerprint work on the hot API wrapper path. A first draft tried to preserve the Jaxpr fingerprint cache across ordinary `Clone`, but `fj-core` clone-then-mutate tests correctly rejected that as stale-cache unsound. The accepted lever keeps ordinary `Jaxpr::clone` cold and instead avoids the root Jaxpr clone in API dispatch.

## Lever

One lever: add borrowed dispatch/proof entrypoints and route fj-api wrappers through them.

- `fj-core` now exposes `composition_signature_for` and `verify_transform_composition_parts`, matching the owned `TraceTransformLedger` verifier without constructing or cloning a ledger.
- `fj-dispatch` now exposes `DispatchRequestRef` and `dispatch_ref`; owned `dispatch` remains as an adapter.
- `fj-api` wrappers now pass their immutable root `Jaxpr` by reference and synthesize the same transform evidence strings used by the prior ledger path.
- `Jaxpr::clone` still creates a fresh empty `OnceLock`, so clone-then-mutate fingerprint semantics remain intact.

## Benchmark

Same target and same host class, using crate-scoped RCH runs for `fj-api`.

```text
baseline e859a6e7:
  value_and_grad_runtime/shared/deep_100_nodes
  [71.027 us 71.319 us 71.701 us]

candidate borrowed dispatch:
  value_and_grad_runtime/shared/deep_100_nodes
  [52.654 us 53.162 us 53.740 us]
```

Median speedup: 71.319 / 53.162 = 1.342x.

Score: Impact 1.342 * Confidence 0.95 / Effort 0.60 = 2.12.

## Isomorphism proof

- Ordering: transform stack order and transform evidence order are unchanged.
- Tie-breaking: no dispatch selection or backend fallback policy changed.
- Floating point: execution receives the same immutable root Jaxpr and argument values; no arithmetic kernel changed.
- RNG: no RNG state or randomized algorithm is involved.
- Cache key semantics: cache key construction still consumes the same backend, root Jaxpr, transform stack, compile options, hook, and unknown-feature list.
- Transform proof semantics: `transform_composition_parts_match_owned_ledger_proof` proves the borrowed verifier output equals the owned ledger verifier output.
- Clone soundness: `canonical_fingerprint_changes_when_effects_change` and `jaxpr_equation_storage_clone_isolation_and_plain_serde` pass with ordinary `Jaxpr::clone` preserving an empty cache.

Golden SHA:

```text
value_and_grad_borrowed_dispatch_golden_sha256 =
2d2d457ca9efee1db747a6578e6f6fbf9e5d84802cad03f84b69d3b52d669f96
```

## Validation

Passed:

```text
cargo fmt -p fj-core -p fj-api -- --check
rustfmt --edition 2024 --config skip_children=true --check crates/fj-dispatch/src/lib.rs
git diff --check -- crates/fj-core/src/lib.rs crates/fj-api/src/transforms.rs crates/fj-dispatch/src/lib.rs
rch exec -- cargo check -p fj-core -p fj-api -p fj-dispatch --all-targets
rch exec -- cargo test -p fj-core --lib
rch exec -- cargo test -p fj-dispatch --lib
rch exec -- cargo test -p fj-api --all-targets
rch exec -- cargo test -p fj-api value_and_grad_borrowed_dispatch_golden_sha256 -- --nocapture
rch exec -- cargo clippy -p fj-core -p fj-api -p fj-dispatch --all-targets -- -D warnings \
  -A unused-variables -A clippy::too_many_arguments -A clippy::manual_is_multiple_of \
  -A clippy::needless_range_loop -A clippy::useless_vec -A clippy::manual_repeat_n
rch exec -- cargo clippy -p fj-api --all-targets -- -D warnings \
  -A unused-variables -A clippy::too_many_arguments -A clippy::manual_is_multiple_of \
  -A clippy::needless_range_loop -A clippy::useless_vec -A clippy::manual_repeat_n \
  -A clippy::doc_lazy_continuation -A clippy::excessive_precision
```

Notes:

- Full `cargo fmt -p fj-dispatch` is blocked by preexisting formatting drift in `crates/fj-dispatch/src/batching.rs`; only the touched dispatch file was checked with rustfmt.
- The final landing commit was rebased onto `980c74af`, after unrelated upstream `fj-lax` SIMD-exp/FMA evidence commits. That upstream dependency adds `doc_lazy_continuation` and `excessive_precision` clippy warnings in `crates/fj-lax/src/simd_exp.rs`, so the final `fj-api` clippy replay used explicit dependency allowances for those two new lints.
- UBS on the three touched Rust files plus this artifact exited 1 because it inventories broad preexisting panic/assert/indexing patterns in the scanned files. Its section 12 and 13 gates were clean: formatting clean, no clippy warnings/errors, cargo check clean, and tests build clean.
