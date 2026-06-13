# frankenjax-7j35t: dense bf16/f16 cumulative fast path

Date: 2026-06-13
Owner: BlackThrush
Worker: ovh-a / vmi1152480 (rch)
Crate: fj-lax (reduction.rs)

## Lever

`eval_cumulative_dense` (cumsum/cumprod/cummax/cummin) had dense f64/f32/i64
paths but returned `None` for **BF16/F16**, dropping half-float cumulatives onto
the generic per-`Literal` scan (`elements.to_vec()` materializes a 24-byte
`Literal` per element, then per-element `as_f64` + `reduce_real_literal`, then
re-densify). Mixed-precision cumulatives (causal-attention masks, sequence
models) are common in ML.

Add a half-float branch mirroring the dense f32 path: read the raw `u16` backing
(`as_half_float_slice`), widen each element to f64 **exactly** as the generic
path does (`Literal::BF16Bits(u).as_f64()` / `F16Bits`), accumulate in a full-f64
running `acc`, store each step rounded back via the SAME `reduce_real_literal`
(bits extracted), and emit dense half storage (`new_half_float_values`). Serial
per line (mirrors f32; the scan is a sequential dependency).

## Result (same-invocation A/B, dense HalfFloat storage vs boxed Literal storage)

```text
rch exec -- cargo test -j 1 -p fj-lax --lib bench_half_cumsum_dense_vs_generic --release -- --ignored --nocapture

BENCH BF16 cumsum axis1 [2048,2048]: generic(per-Literal)=75.1262ms dense=22.9380ms speedup=3.28x
BENCH F16  cumsum axis1 [2048,2048]: generic(per-Literal)=85.2729ms dense=36.4053ms speedup=2.34x
```

Keep: **BF16 3.28x, F16 2.34x** (both ≥ 2.0). Score: 3.28 × 0.95 / 1 = 3.1 (BF16).

## Isomorphism Proof

- Widen `u16 → f64` uses `Literal::{BF16Bits,F16Bits}(u).as_f64()` — the EXACT
  conversion the generic path applies to each materialized literal.
- The f64 accumulator is identical: same `float_init`, same `float_op`, same
  ascending per-line order (`line_base` + `axis_stride`), never rounded mid-scan.
- Each step stores `reduce_real_literal(dtype, acc)` bits — the EXACT rounding the
  generic path uses.
- Output dtype/shape preserved via `new_half_float_values`.
- Sequential dependency ⇒ no reassociation/vectorization ⇒ exact incl. NaN.

Behavior proof: `bench_half_cumsum_dense_vs_generic` asserts the dense HalfFloat
output's half bits equal the boxed-`Literal` generic scan's, element-for-element,
for both BF16 and F16 (`assert_eq!` before timing). All existing cum* lib tests
(41 passed) remain green.
