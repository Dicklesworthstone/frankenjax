# frankenjax-jyr2u: cold compiled JIT metadata skip

Date: 2026-06-15
Agent: SilverMaple
Bead: frankenjax-jyr2u
Commit: pending

## Profile-backed target

Post-88cb89ea RCH reprofile on `ovh-a` showed cold JIT wrapper overhead as the next
non-overlapping `fj-api` hotspot:

```text
api_overhead/jit/scalar_add                         [2.2349us 2.2708us 2.2797us]
api_overhead/jit/scalar_add_repeated_call           [155.43ns 156.04ns 156.19ns]
api_vs_dispatch/api_jit_add                         [2.2401us 2.2420us 2.2425us]
api_mode_config/strict_jit                          [2.1406us 2.1560us 2.1599us]
api_mode_config/hardened_jit                        [2.2454us 2.2663us 2.3499us]
jit_compiled_eval_cache/api_compiled/scalar_add      [157.98ns 158.69ns 161.54ns]
```

## Lever

`JitWrapped::call` now probes the existing default-CPU compiled evaluator before
constructing transform evidence, compile options, prepared dispatch metadata, or
the generic dispatch call.

The compiled evaluator is not new behavior. It is the same cached
`compile_jaxpr_for_repeated_eval` plan that the previous wrapper already used
after prepared metadata construction succeeded.

## Benchmark

Command:

```text
RCH_WORKER=ovh-a rch exec -- cargo bench -j 1 -p fj-api --bench api_overhead -- 'api_overhead/jit/scalar_add|api_overhead/jit/scalar_add_repeated_call|api_vs_dispatch/api_jit_add|jit_compiled_eval_cache/api_compiled/scalar_add|api_mode_config/strict_jit|api_mode_config/hardened_jit' --quick --noplot
```

Candidate on `ovh-a`:

```text
api_overhead/jit/scalar_add                         [763.06ns 764.78ns 765.21ns]
api_overhead/jit/scalar_add_repeated_call           [96.870ns 99.509ns 100.17ns]
api_vs_dispatch/api_jit_add                         [777.72ns 781.17ns 794.93ns]
api_mode_config/strict_jit                          [743.58ns 746.45ns 747.17ns]
api_mode_config/hardened_jit                        [759.17ns 760.79ns 761.20ns]
jit_compiled_eval_cache/api_compiled/scalar_add      [105.55ns 106.06ns 108.13ns]
```

Ratios:

```text
api_overhead/jit/scalar_add                2.2708us -> 764.78ns (2.97x)
api_overhead/jit/scalar_add_repeated_call  156.04ns -> 99.509ns (1.57x)
api_vs_dispatch/api_jit_add                2.2420us -> 781.17ns (2.87x)
api_mode_config/strict_jit                 2.1560us -> 746.45ns (2.89x)
api_mode_config/hardened_jit               2.2663us -> 760.79ns (2.98x)
jit_compiled_eval_cache/api_compiled        158.69ns -> 106.06ns (1.50x)
```

Score: 2.8 = impact 2.97 x confidence 0.95 / effort 1.0.

## Isomorphism proof

- Output ordering is unchanged: the compiled evaluator returns the same ordered
  `Vec<Value>` produced by the existing compiled plan.
- Floating-point behavior is unchanged: the same interpreter compiled plan runs
  the same primitive sequence; no reassociation, fused operation, or dtype
  conversion was added.
- Tie-breaking is unchanged: the scalar-add JIT path has no comparison or
  selection tie surface.
- RNG behavior is unchanged: the benchmarked and tested Jaxprs are deterministic
  and effect-free.
- Shape, dtype, and error behavior are unchanged for unsupported graphs and
  non-default backends because those still return `None` and fall through to the
  previous generic dispatch path.
- The only skipped work on the default compiled path is args-independent wrapper
  metadata construction that was not independently observable before dispatch.

Golden-output proof:

```text
cargo test -j 1 -p fj-api jit_repeated_call_compiled_cache_matches_dispatch_golden_sha256 -- --nocapture
golden sha256: 358ba10d12a581c6dd0ec4adb2ab3f69b71e12df1316ff855ab5387f328dbf38
```

## Validation

```text
cargo fmt --check --package fj-api
RCH_WORKER=ovh-a rch exec -- cargo check -j 1 -p fj-api --all-targets
RCH_WORKER=ovh-a rch exec -- cargo clippy -j 1 -p fj-api --all-targets -- -D warnings
RCH_WORKER=ovh-a rch exec -- cargo test -j 1 -p fj-api jit_repeated_call_compiled_cache_matches_dispatch_golden_sha256 -- --nocapture
```

Notes:

- The clippy run selected `vmi1227854` despite the worker request; this does not
  affect the benchmark comparison, which was same-worker on `ovh-a`.
- Repo-wide `cargo fmt --check` is currently blocked by peer-owned formatting
  drift in `crates/fj-lax/src/tensor_ops.rs`; `fj-api` scoped formatting passed.

## Decision

Kept. The primary cold JIT API row improves by 2.97x on the same worker with
unchanged golden output and unchanged fallback semantics.
