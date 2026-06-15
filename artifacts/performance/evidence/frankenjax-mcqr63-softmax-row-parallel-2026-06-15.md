# frankenjax-mcqr.63: fj-lax softmax_2d row-parallel exact path

## Target

- Bead: `frankenjax-mcqr.63`
- Crate: `fj-lax`
- Function: `nn::softmax_2d`
- Profile-backed row: `nn/softmax_2d_65536x16_fused`
- Lever: split independent 2D softmax rows into scoped-thread row chunks above `SOFTMAX_2D_PARALLEL_MIN`.

## Baseline

Command:

```bash
rch exec -- cargo bench -p fj-lax --bench lax_baseline -- 'nn/softmax_2d_65536x16'
```

Worker: `ovh-a`

Criterion rows:

- `nn/softmax_2d_65536x16_rowmap_ref`: `[5.8629 ms 5.8915 ms 5.9262 ms]`
- `nn/softmax_2d_65536x16_fused`: `[4.8206 ms 4.8398 ms 4.8653 ms]`

## Candidate

Command:

```bash
rch exec -- cargo bench -p fj-lax --bench lax_baseline -- 'nn/softmax_2d_65536x16_fused'
```

Worker: `ovh-a`

Criterion row:

- `nn/softmax_2d_65536x16_fused`: `[1.7551 ms 1.7774 ms 1.7997 ms]`

Comparable midpoint ratio: `4.8398 / 1.7774 = 2.72x`.
Conservative interval ratio: `4.8206 / 1.7997 = 2.68x`.

Extra current-diff routing check:

- Worker `vmi1149989`: `[1.8010 ms 1.8498 ms 1.8997 ms]`

## Isomorphism Proof

- Each row still uses the exact prior fused row algorithm: max scan, shifted `exp`, sum in left-to-right column order, then left-to-right division.
- Row-major output order is preserved; only independent row ownership changes.
- `log_softmax_2d` was not parallelized in this commit because its candidate run was noisy/cross-worker and did not prove a win.
- Floating-point operation order within each output element, tie behavior, signed zero handling, NaN/inf propagation, dtype/shape/error behavior, and RNG absence are unchanged.
- Scoped threads only receive disjoint `result` row chunks; no shared mutable row state and no unsafe code.

Golden output SHA-256:

- `softmax_2d_parallel_bit_identical_to_serial_fused`
- SHA: `804d2a4abc52612601a766311fbe9a8e230766c65110b667b35676a7803bc6dd`

## Validation

```bash
cargo fmt -p fj-lax
cargo fmt --check -p fj-lax
rch exec -- cargo test -p fj-lax softmax_2d -- --nocapture
rch exec -- cargo check -p fj-lax --all-targets
rch exec -- cargo clippy -p fj-lax --all-targets -- -D warnings
ubs crates/fj-lax/src/nn.rs
```

Results:

- Format: passed.
- Focused tests: passed on `ovh-a`; 6 passed, 0 failed.
- `cargo check -p fj-lax --all-targets`: passed on `ovh-a`.
- `cargo clippy -p fj-lax --all-targets -- -D warnings`: passed on `ovh-a`.
- UBS: exit 0; no critical findings, no unsafe, no unwrap/expect. Remaining warnings are assertion/indexing inventory; the new row slicing is guarded by the public `x.len() == rows * cols` assertion plus row-aligned chunk partitioning.

## Score

- Impact: `2.68` conservative interval ratio.
- Confidence: `0.95` same-worker Criterion + bit proof + golden SHA.
- Effort: `1.0`.
- Score: `2.55`, keep.
