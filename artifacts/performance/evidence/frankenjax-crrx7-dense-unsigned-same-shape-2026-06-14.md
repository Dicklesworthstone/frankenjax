# frankenjax-crrx7 dense unsigned same-shape arithmetic

Bead: `frankenjax-crrx7`
Agent: `SilverMaple`
Date: 2026-06-14 local / 2026-06-15 UTC

## Target

Profile-backed gap from `frankenjax-crrx7`: dense `U32`/`U64` tensor storage existed (`as_u32_slice` / `as_u64_slice`), but same-shape arithmetic still fell through `binary_literal_op`, materializing/matching `Literal` values. This pass applies the dense typed-storage primitive from the no-gaps lane: operate directly on contiguous unsigned slices and emit dense unsigned output.

One production lever only: same-shape dense `U32`/`U64` arithmetic for `Add`, `Sub`, `Mul`, `Div`, `Rem`, `Max`, and `Min`. Scalar broadcast is intentionally unchanged for the next pass.

## Baseline

Command:

```text
RCH_WORKER=vmi1149989 rch exec -- cargo bench -p fj-lax --bench lax_baseline -- arith_64k
```

Worker: `vmi1149989`

Baseline medians:

```text
eval/u32_arith_64k_add_dense          [1.5662 ms 1.5821 ms 1.5992 ms]
eval/u32_arith_64k_add_boxed          [1.6311 ms 1.6552 ms 1.6787 ms]
eval/u32_arith_64k_scalar_mul_dense   [894.67 us 919.68 us 945.46 us]
eval/u32_arith_64k_scalar_mul_boxed   [948.64 us 1.0127 ms 1.0819 ms]
eval/u64_arith_64k_add_dense          [1.4398 ms 1.4521 ms 1.4650 ms]
eval/u64_arith_64k_add_boxed          [1.4420 ms 1.4588 ms 1.4768 ms]
eval/u64_arith_64k_scalar_mul_dense   [902.80 us 950.05 us 997.62 us]
eval/u64_arith_64k_scalar_mul_boxed   [1.1850 ms 1.2357 ms 1.2855 ms]
```

## Candidate

Command:

```text
RCH_WORKER=vmi1149989 rch exec -- cargo bench -p fj-lax --bench lax_baseline -- arith_64k
```

Worker: `vmi1149989`

Candidate medians:

```text
eval/u32_arith_64k_add_dense          [103.30 us 105.29 us 107.11 us]
eval/u32_arith_64k_add_boxed          [1.5706 ms 1.5914 ms 1.6119 ms]
eval/u32_arith_64k_scalar_mul_dense   [1.0315 ms 1.0935 ms 1.1563 ms]
eval/u32_arith_64k_scalar_mul_boxed   [892.09 us 930.01 us 970.17 us]
eval/u64_arith_64k_add_dense          [103.38 us 106.58 us 110.23 us]
eval/u64_arith_64k_add_boxed          [1.4966 ms 1.5208 ms 1.5451 ms]
eval/u64_arith_64k_scalar_mul_dense   [785.14 us 801.23 us 819.69 us]
eval/u64_arith_64k_scalar_mul_boxed   [812.34 us 839.42 us 867.53 us]
```

Measured keep rows:

```text
u32 same-shape add dense: 1.5821 ms -> 105.29 us = 15.03x
u64 same-shape add dense: 1.4521 ms -> 106.58 us = 13.62x
```

Scalar rows are reported only as non-target controls; this lever does not route scalar broadcast.

## Behavior Proof

Focused proof command:

```text
rch exec -- cargo test -p fj-lax dense_unsigned_same_shape_arithmetic_bit_identical_to_literal_path --lib -- --nocapture
```

Final proof result:

```text
test arithmetic::tests::dense_unsigned_same_shape_arithmetic_bit_identical_to_literal_path ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1541 filtered out
```

Golden output SHA256:

```text
161e98590749a45847f6aa614b97a7e984240dc01bff16548b1bc12d16ba19fd
```

The test computes the SHA via `fj_test_utils::fixture_id_from_json` over dense outputs for all routed operations and asserts the exact digest.

Isomorphism:

- Ordering: elementwise traversal remains row-major slice order; one output per input index.
- Tie-breaking: `Max`/`Min` use `u32::max/min` and `u64::max/min`, identical to the unsigned arm of `binary_literal_op`.
- Integer overflow: `Add`/`Sub`/`Mul` use the same wrapping arithmetic as `binary_literal_op`.
- Divide/remainder by zero: `checked_div/rem(...).unwrap_or(0)` is preserved.
- Floating point: none involved in the routed dense unsigned path.
- RNG: none involved.
- Error behavior: helper returns `None` unless both operands are dense same-dtype unsigned tensors; malformed or mixed cases still fall through to the generic path.

## Score

Impact 9.0 x Confidence 0.95 / Effort 2.0 = 4.28.

Keep: yes, Score >= 2.0.
