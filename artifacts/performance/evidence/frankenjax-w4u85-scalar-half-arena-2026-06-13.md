# frankenjax-w4u85 scalar BF16/F16 arena arithmetic

## Target

- Bead: `frankenjax-w4u85`
- Crate: `fj-interpreters`
- Profile-backed hotspot: scalar interpreter overhead in repeated exact-output arithmetic bodies after `frankenjax-compiled-jaxpr-dispatch-t22rd` removed the plain `jit()` metadata-cache floor.
- Lever: add one dense scalar arena executor for BF16/F16 arithmetic Jaxprs containing only `Add`, `Sub`, `Mul`, `Div`, `Max`, `Min`, `Neg`, and `Abs`.

## Same-worker Criterion gate

Worker: `vmi1227854`

Command:

```text
RCH_WORKER=vmi1227854 rch exec -- cargo bench -j 1 -p fj-interpreters --bench pe_baseline -- half_arith --warm-up-time 1 --measurement-time 3 --sample-size 30
```

| Benchmark | Baseline mean | After mean | Ratio |
| --- | ---: | ---: | ---: |
| `eval/scalar_bf16_half_arith_body` | 1.5738 us | 448.01 ns | 3.51x |
| `eval/scalar_f16_half_arith_body` | 1.4973 us | 400.93 ns | 3.73x |

Score: Impact 3.6 x Confidence 0.90 / Effort 1 = 3.24.

## Isomorphism proof

- Ordering: every equation is executed in original Jaxpr order, with one scalar slot write per output variable and no reassociation.
- Floating point: operands are widened with the existing `half_fusion_widen`, operations use the same `half_fused_binary`/`CheapOp` helpers, and every intermediate is rounded back through `half_fusion_round` to the same half dtype.
- Tie/NaN policy: `Max` and `Min` reuse the existing JAX NaN-propagating half helpers, preserving the current generic half behavior.
- Type guards: BF16 plans accept only BF16 scalar values/literals; F16 plans accept only F16 scalar values/literals. Mismatches bail to the generic interpreter.
- RNG: no RNG, side effects, sub-Jaxprs, or params are admitted by the fast path.
- Golden SHA-256: `2a61385a28dd56a659b204ce161fb1d9cbcd30bd6a6f8e42e57f328894746d1c` from planned-vs-forced-generic half output-bit rows.

## Validation

- `RCH_WORKER=vmi1227854 rch exec -- cargo test -j 1 -p fj-interpreters scalar_half_arith_plan_matches_generic_bits -- --nocapture`: passed.
- `RCH_WORKER=vmi1227854 rch exec -- cargo check -j 1 -p fj-interpreters --benches --tests`: passed; RCH repeated a pre-existing dependency warning in `fj-trace`.
- `RCH_WORKER=vmi1227854 rch exec -- cargo clippy -j 1 -p fj-interpreters --all-targets --no-deps -- -D warnings`: passed; RCH repeated the same dependency warning.
- Final `RCH_WORKER=vmi1227854 rch exec -- cargo bench -j 1 -p fj-interpreters --bench pe_baseline -- half_arith --warm-up-time 1 --measurement-time 3 --sample-size 30`: passed with BF16 `[431.63 ns 448.01 ns 462.51 ns]` and F16 `[377.59 ns 400.93 ns 425.46 ns]`.
- `rustfmt --edition 2024 --check crates/fj-interpreters/src/lib.rs crates/fj-interpreters/benches/pe_baseline.rs`: passed.
- `git diff --check`: passed.
- `ubs crates/fj-interpreters/src/lib.rs crates/fj-interpreters/benches/pe_baseline.rs`: nonzero from pre-existing file-wide `fj-interpreters` panic/unwrap/indexing/test-inventory findings; its built-in formatting, clippy, check, test-build, audit, and deny sections were clean.
- Workspace `cargo fmt --check`: blocked by unrelated pre-existing formatting drift in other crates; touched-file rustfmt passed.
