# frankenjax-so4wo compiled JIT eval cache evidence

Date: 2026-06-15
Agent: SilverMaple
Lever: cache a compiled dense scalar Jaxpr evaluator on `JitWrapped` for repeated default-CPU plain-JIT calls.

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p fj-api --bench api_overhead -- jit/scalar_add --warm-up-time 1 --measurement-time 3 --sample-size 20
```

Worker: `vmi1152480`

Rows:

- `api_overhead/jit/scalar_add`: 3.2127 us, 3.3044 us, 3.4234 us
- `api_overhead/jit/scalar_add_repeated_call`: 1.3323 us, 1.3809 us, 1.4499 us

## Same-binary A/B rebench

Command:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p fj-api --bench api_overhead -- jit_compiled_eval_cache --warm-up-time 1 --measurement-time 3 --sample-size 20
```

Worker: `vmi1149989`

Rows:

- `jit_compiled_eval_cache/dispatch_prepared/scalar_add`: 1.0255 us, 1.1117 us, 1.1915 us
- `jit_compiled_eval_cache/api_compiled/scalar_add`: 241.92 ns, 268.87 ns, 292.79 ns

Speedup:

- Midpoint: 4.14x
- Conservative interval: 3.50x
- Score: 10.0 (Impact 4, Confidence 5, Effort 2)

Supplemental public API row:

- `api_overhead/jit/scalar_add_repeated_call` on `ovh-a`: 152.24 ns, 152.69 ns, 153.24 ns
- `api_overhead/jit/scalar_add` on `ovh-a`: 2.2857 us, 2.3297 us, 2.3785 us

## Isomorphism proof

- Ordering: compiled eval reuses the existing dense scalar plan and runs the same step order as `run_dense_plan`.
- Tie-breaking: no comparison or sort tie policy changes.
- Floating point: no reassociation, FMA, vector lane reduction, or mixed precision changes; primitive closures are unchanged.
- RNG: no RNG or effectful program is eligible.
- Error/fallback surface: the compiled cache only accepts no-const, effect-free, sub-jaxpr-free, uniquely-bound dense scalar Jaxprs with bound inputs and a scalar plan. Everything else falls back to the existing dispatch/backend path.
- Golden output sha256: `358ba10d12a581c6dd0ec4adb2ab3f69b71e12df1316ff855ab5387f328dbf38`.

## Validation

- `cargo fmt --check`
- `git diff --check -- crates/fj-interpreters/src/lib.rs crates/fj-api/src/transforms.rs crates/fj-api/benches/api_overhead.rs`
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p fj-api jit_repeated_call_compiled_cache_matches_dispatch_golden_sha256 --lib -- --nocapture`
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p fj-api -p fj-interpreters --all-targets`
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p fj-api -p fj-interpreters --all-targets -- -D warnings`
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p fj-api -p fj-interpreters --lib`
