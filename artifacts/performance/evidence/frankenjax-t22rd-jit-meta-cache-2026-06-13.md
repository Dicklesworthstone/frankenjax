# frankenjax-compiled-jaxpr-dispatch-t22rd -- JitWrapped dispatch metadata cache

## Target

- Bead: `frankenjax-compiled-jaxpr-dispatch-t22rd`
- Profile-backed hotspot: repeated scalar `jit` calls paid the transform-evidence,
  composition-proof, and cache-key preparation cost on every call.
- Lever: add the same lazy `DispatchMetaCache` used by `ValueAndGradWrapped` to
  plain `JitWrapped`, and invalidate it when backend or compatibility mode
  changes.

## Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -j 1 -p fj-api --bench api_overhead -- api_overhead/jit/scalar_add_repeated_call --warm-up-time 1 --measurement-time 3 --sample-size 20
```

Worker: `vmi1227854`

Criterion result:

```text
api_overhead/jit/scalar_add_repeated_call
                        time:   [2.2137 us 2.2918 us 2.3920 us]
```

## After

Command:

```text
RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -j 1 -p fj-api --bench api_overhead -- api_overhead/jit/scalar_add_repeated_call --warm-up-time 1 --measurement-time 3 --sample-size 20
```

Worker: `vmi1227854`

Criterion result:

```text
api_overhead/jit/scalar_add_repeated_call
                        time:   [1.3393 us 1.3736 us 1.4100 us]
```

Median speedup: `2.2918 / 1.3736 = 1.67x`.

Score: `7.5` (`Impact 3 x Confidence 5 / Effort 2`). Keep threshold is `>= 2.0`.

## Isomorphism Proof

- Ordering preserved: yes. `JitWrapped::call` still passes `[Transform::Jit]`
  through `dispatch_with_options_prepared`, which calls the same `dispatch_ref`
  path and evaluates the same Jaxpr equations in the same order.
- Tie-breaking unchanged: yes. The cache stores only args-independent dispatch
  metadata; it does not alter primitive selection, output ordering, comparison
  policy, or any sorted/tied data path.
- Floating-point: bit-identical. The lever does not change any primitive,
  tensor kernel, scalar arithmetic, reduction order, or literal construction.
- RNG seeds: N/A. The benchmarked and tested path is deterministic and uses no
  RNG.
- Cache-key inputs: backend and compatibility mode reset the cache. The Jaxpr is
  immutable inside `JitWrapped`, the transform stack is fixed to `jit`, and
  compile options remain the same empty `BTreeMap` used by the previous
  `dispatch_with` path.
- Golden output: exact payload
  `jit/scalar_add_repeated_call output: [7]\n` verified with SHA-256
  `2ca40813fb7a22bc6d2098fe1c3f01494f729a6baf63a9b3dbd8545853a3d53e`.

Golden verification command:

```text
printf 'jit/scalar_add_repeated_call output: [7]\n' | tee /tmp/frankenjax-t22rd-golden.txt | sha256sum
printf '2ca40813fb7a22bc6d2098fe1c3f01494f729a6baf63a9b3dbd8545853a3d53e  /tmp/frankenjax-t22rd-golden.txt\n' | sha256sum -c -
```

Result:

```text
/tmp/frankenjax-t22rd-golden.txt: OK
```

## Validation

```text
rustfmt --edition 2024 --check crates/fj-api/src/transforms.rs crates/fj-api/benches/api_overhead.rs
```

Passed.

```text
RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -j 1 -p fj-api --lib jit -- --nocapture
```

Passed: `19 passed; 0 failed; 70 filtered out`.

```text
RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -j 1 -p fj-api --benches --tests
```

Passed. The command repeated the pre-existing `fj-trace` `num_spatial` warning
and `fj-dispatch` `solve_row` warning; neither is in the touched files.

```text
RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -j 1 -p fj-api --all-targets --no-deps -- -D warnings
```

Passed with exit 0. RCH selected `vmi1149989` for this lint job despite the
worker hint. This is validation only; the before/after benchmark proof above is
same-worker on `vmi1227854`.

## Notes

Earlier bead comments said `fj-api jit()` already used `DispatchMetaCache`. The
current source showed that only `ValueAndGradWrapped` had the cache; plain
`JitWrapped` still rebuilt metadata per call. This pass closes that exact gap.
