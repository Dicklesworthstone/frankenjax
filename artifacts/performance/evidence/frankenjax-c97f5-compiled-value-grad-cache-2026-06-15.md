# frankenjax-c97f5 compiled scalar-F64 value_and_grad cache

Date: 2026-06-15
Agent: SilverMaple
Bead: frankenjax-c97f5

## Target

`GradWrapped` / `ValueAndGradWrapped` still paid the value-dependent tape AD pass on every repeated call after the wrapper metadata caches landed. This pass ships the first cached backward-plan primitive for the pure scalar-F64, single-output subset and wires it into the public API wrappers with fallback for every unsupported case.

## Lever

- Added `fj_ad::compile_value_and_grad_jaxpr_for_repeated_eval`.
- The compiled plan validates the static Jaxpr once, then evaluates equations in original order and propagates cotangents in reverse equation order.
- Supported primitives: add, sub, mul, div, neg, sin, cos, exp, log.
- `GradWrapped` and `ValueAndGradWrapped` lazily cache the compiled plan, only on default CPU backend, no custom VJP, and after dispatch metadata preparation succeeds.
- Unsupported graphs, constvars, effects, sub-jaxprs, non-scalar/non-finite runtime values, custom VJP, and non-default backends all fall back to the existing tape/dispatch path.
- Backend/mode builder mutations invalidate both dispatch metadata and compiled AD cache for the AD wrappers.

## Isomorphism Proof

- Equation order: forward evaluation follows the original Jaxpr equation order exactly.
- Reverse order: cotangents are propagated by reversing the original equation sequence, matching the tape AD dependency order for the supported pure scalar subset.
- Floating point: the plan uses the same `f64` primitive operations (`+`, `-`, `*`, `/`, `sin`, `cos`, `exp`, `ln`) and does not reassociate terms. Non-finite inputs or intermediates fall back to the old path.
- Tie-breaking: no comparisons or reductions are introduced.
- RNG: no RNG surface exists in this subset.
- Error behavior: unsupported primitives, effects, constvars, sub-jaxprs, duplicate bindings, arity mismatches, custom VJP, and non-default backends preserve the existing fallback path.
- Golden payload: `["3fdd18f6ead1b447","bfdaa22657537203"]`.
- Golden logical SHA-256: `984585309be003365780a1f999422efc949c360ec9933d354a6bd50b5b41653a`.

## Baseline And Result

Final corrected RCH Criterion run on `ovh-a`:

| Row | Midpoint |
| --- | ---: |
| `ad_compiled_reverse_plan/tape/value_and_grad_trig` | 1.1521 us |
| `ad_compiled_reverse_plan/compiled/value_and_grad_trig` | 105.50 ns |
| `ad_compiled_reverse_plan/api_warmed/value_and_grad_trig` | 188.70 ns |

Public API speedup: `1.1521 us / 188.70 ns = 6.11x`.

Direct compiled-plan speedup: `1.1521 us / 105.50 ns = 10.92x`.

Score: Impact `6.11` x Confidence `0.95` / Effort `2.0` = `2.90`; keep.

## Validation

- `cargo fmt -p fj-ad -p fj-api --check`
- `git diff --check -- crates/fj-ad/src/lib.rs crates/fj-api/src/transforms.rs crates/fj-api/benches/api_overhead.rs`
- RCH `cargo test -j 1 -p fj-ad compiled_scalar_f64_reverse_plan_matches_generic_bits -- --nocapture`
- RCH `cargo test -j 1 -p fj-api value_and_grad_compiled_ad_cache_matches_tape_golden_sha256 -- --nocapture`
- RCH `cargo check -j 1 -p fj-ad -p fj-api --all-targets`
- RCH `cargo clippy -j 1 -p fj-ad -p fj-api --all-targets -- -D warnings`
- RCH `cargo bench -j 1 -p fj-api --bench api_overhead -- ad_compiled_reverse_plan --quick --noplot`
- UBS on touched files completed nonzero from existing file-wide panic/unwrap/indexing/perf heuristic inventory; its built-in formatting, clippy, cargo check, test-build, audit, and cargo-deny sections were clean.

## Next Primitive

Continue `frankenjax-c97f5` decomposition by broadening the compiled backward plan beyond pure scalar-F64: shape-signature keyed tensor backward Jaxprs, then route those through the existing dense compiled evaluator. Do not repeat wrapper metadata work unless a fresh profile identifies another unprepared wrapper.
