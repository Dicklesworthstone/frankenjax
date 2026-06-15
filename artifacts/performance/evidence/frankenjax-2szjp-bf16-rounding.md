# frankenjax-2szjp: BF16 Matmul Output Rounding

Date: 2026-06-15
Agent: SilverMaple
Worker: vmi1149989

## Profile Target

Full `fj-lax` linalg Criterion profile identified the native BF16 row-block as a current hotspot:

- `linalg/bf16_matmul_1024_blocked`: [50.047 ms, 51.873 ms, 53.742 ms]
- Nearby f32 GEMM core rows were materially faster, so the BF16 output/decode path had remaining overhead.

Before this accepted lever, one f32 4096 gate candidate was rejected:

- `linalg/f32_gemm_4096_packed`: [1.9929 s, 2.0837 s, 2.1778 s]
- `linalg/f32_gemm_4096_kcblocked`: [1.7992 s, 2.1181 s, 2.4660 s]
- Rejected: median was slower and confidence interval overlapped.

## Lever

Vectorize only the final `F32xN` accumulator to BF16 output conversion in the native BF16 matmul row-block and its same-binary row reference.

No change to:

- BF16 decode: still exact `u16 -> u32 << 16 -> f32`
- Accumulation order: still ascending `l` per output lane
- Arithmetic precision: still f32 multiply/add
- RNG/tie-breaking: none involved
- Rounding contract: same round-to-nearest-even f32-to-BF16 bits as the scalar `round_f32_to_bf16`

## Proof

RCH proof command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 rch exec -- cargo test -p fj-lax bf16 --lib -- --nocapture
```

Result: 15 passed, 0 failed, 10 ignored.

Golden SHA-256 fixture:

- Test: `tensor_contraction::tests::batched_matmul_2d_bf16_native_golden_digest`
- Digest: `ff880f79bfc352fc4c4943db02e7f7e34b4c6bcb0989c960bc5f88cd6cee7bb9`

Additional bit-identity coverage:

- `round_f32xn_to_bf16_matches_scalar_edges`
- `bf16_in_matches_f32_accum`
- `bf16_register_blocked_remainders_match_reference`
- `batched_matmul_2d_bf16_batch1_packed_route_matches_register_kernel`

## Benchmark

RCH after command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 rch exec -- cargo bench -p fj-lax --bench lax_baseline -- linalg/bf16_matmul_1024_blocked --noplot
```

After:

- `linalg/bf16_matmul_1024_blocked`: [41.220 ms, 42.276 ms, 43.377 ms]

Delta:

- Median: 51.873 ms -> 42.276 ms
- Speedup: 1.23x
- Score: Impact 3 x Confidence 4 / Effort 2 = 6.0

## Gates

- `cargo check -p fj-lax --lib` via RCH: passed.
- `cargo clippy -p fj-lax --lib -- -D warnings` via RCH: blocked by unrelated existing `arithmetic.rs` and `linalg.rs` warnings, not by this change.
- `rustfmt --check crates/fj-lax/src/tensor_contraction.rs`: blocked by pre-existing formatting drift elsewhere in the file; local import ordering from this lever was fixed manually.
- `ubs crates/fj-lax/src/tensor_contraction.rs`: build/fmt/clippy/audit/deny sections clean; command exits 1 due existing inventories and false-positive JWT "decode" matches in this numeric file.
