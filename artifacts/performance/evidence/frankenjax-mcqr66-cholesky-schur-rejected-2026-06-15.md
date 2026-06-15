# frankenjax-mcqr.66 Cholesky Schur Detours Rejected

Date: 2026-06-15
Agent: SilverMaple
Bead: frankenjax-mcqr.66
Surface: `fj-lax` `linalg/cholesky_1024x1024_f64`
Decision: rejected, no source hunk kept

## Target

Fresh post-NN linalg profiling showed Cholesky and QR as dominant `fj-lax`
surfaces. QR row-thread fanout had already regressed in `frankenjax-mcqr.64`, so
this pass evaluated a distinct Cholesky Schur-update lever:

`A22 -= L21 * L21^T` for large trailing blocks.

The focused pre-edit baseline was:

```text
worker: vmi1227854
command: rch exec -- cargo bench -p fj-lax --bench lax_baseline -- 'linalg/cholesky_1024x1024_f64'
linalg/cholesky_1024x1024_f64: [29.112 ms 29.898 ms 30.729 ms]
```

## Candidate A: full GEMM detour

The first candidate packed `L21^T`, computed the full `rem x rem` product with
the existing safe-Rust GEMM, and subtracted only the lower triangle.

Proof:

```text
worker: vmi1293453
command: rch exec -- cargo test -p fj-lax cholesky -- --nocapture
result: 15 passed, including lower-triangle proof and blocked Cholesky golden
```

Benchmark:

```text
worker: vmi1149989
command: rch exec -- cargo bench -p fj-lax --bench lax_baseline -- 'linalg/cholesky_1024x1024_f64'
linalg/cholesky_1024x1024_f64: [22.922 ms 23.428 ms 23.945 ms]
```

This was not directly comparable to the focused baseline worker. Against the
earlier noisy broad same-worker `vmi1149989` pre-change slice
`[39.148 ms 46.287 ms 53.710 ms]`, the midpoint ratio was 1.98x and the
conservative interval ratio was 1.63x, below the campaign keep bar after
confidence discounting. The source hunk was not kept.

## Candidate B: packed lower-triangle SYRK microkernel

The second candidate avoided the full product allocation and strict-upper work by
packing `L21^T` once and updating only consumed lower-triangle rows with a
4-row by 8-column safe-Rust microkernel.

Proof:

```text
worker: vmi1156319
command: rch exec -- cargo test -p fj-lax cholesky -- --nocapture
result: 15 passed, including packed-threshold lower-triangle proof and blocked Cholesky golden
```

Benchmark:

```text
worker: vmi1167313
command: rch exec -- cargo bench -p fj-lax --bench lax_baseline -- 'linalg/cholesky_1024x1024_f64'
linalg/cholesky_1024x1024_f64: [78.971 ms 84.226 ms 90.341 ms]
```

This clearly regressed relative to the existing Cholesky range, so it was
rejected. The source hunk was restored.

## Isomorphism Notes

Both candidates preserved dtype, shape, deterministic row/triangle write
surfaces, RNG absence, error behavior, and strict-upper zeroing. Existing
large-block Cholesky is a tolerance-equivalent linalg path; the small blocked
golden digest remained below the candidate threshold and passed during proof.

## Next Primitive

Do not retry Cholesky full-product or packed triangular Schur detours as a
microkernel family. Reprofile and, if linalg remains dominant, attack a deeper
algorithmic structure: communication-avoiding QR, compact-WY buffer reuse,
recursive/blocked Cholesky with a different panel shape, or a fundamentally
different safe-Rust linalg primitive with a fresh same-worker baseline.
