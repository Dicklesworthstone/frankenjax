# frankenjax-8ndmr: batched Cholesky work-scaled fan-out

Date: 2026-06-11
Agent: BeigeMouse
Bead: frankenjax-8ndmr
Commit candidate: fj-dispatch batched Cholesky parallel fan-out

## Target

`batch_cholesky` in `crates/fj-dispatch/src/batching.rs` still processed each
batch matrix serially. The already-landed batched eig/SVD fan-out wins showed
that independent, compute-heavy per-slice linalg operations can profit from
`std::thread::scope` when each worker receives enough work to amortize spawn
cost.

## Change

One lever:

- Extract batched Cholesky input into one contiguous `Vec<f64>`.
- Factor each independent batch slice through the same scalar
  `cholesky_decompose_into` loop used by the old serial path.
- Use a work-scaled thread count (`batch*n^3 / 2^21`, capped by batch size and
  available parallelism), returning to serial for small batches.
- Write each slice to a disjoint output range, then wrap through
  `TensorValue::new` exactly as before.

## Benchmark

Command:

```bash
rch exec -- cargo test -j 1 -p fj-dispatch --lib bench_batch_cholesky_parallel_vs_serial -- --ignored --nocapture
```

Remote worker: `vmi1149989`

Best-of-7 internal baseline/candidate benchmark from the ignored test:

| Shape | Threads | Serial | Parallel | Speedup |
|---|---:|---:|---:|---:|
| batch=64 n=24 | 1 | 1.872 ms | 1.255 ms | 1.49x |
| batch=128 n=48 | 6 | 16.614 ms | 5.146 ms | 3.23x |
| batch=256 n=48 | 10 | 34.431 ms | 8.308 ms | 4.14x |
| batch=512 n=64 | 10 | 163.686 ms | 34.915 ms | 4.69x |

Score: Impact 4 * Confidence 4 / Effort 2 = 8.0.

## Isomorphism Proof

- Ordering preserved: yes. Output shape and batch-major row-major layout are
  unchanged; each matrix writes to its original output slice.
- Tie-breaking unchanged: N/A.
- Floating-point behavior: bit-identical to the old per-slice scalar Cholesky
  loop. The only parallelism is across independent matrices; the inner
  `i,j,k` fold order inside each matrix is unchanged.
- Error behavior: non-square and wrong-rank branches are unchanged. Non-numeric
  element errors are still raised before output construction.
- RNG: N/A.
- Golden output: `ac98b8f45349b186e327a13d55a61ca2a044321ff6ea15d5af8d1bcd02caf0d6`.

## Validation

```bash
cargo fmt -p fj-dispatch --check
git diff --check
ubs crates/fj-dispatch/src/batching.rs
rch exec -- cargo test -j 1 -p fj-dispatch --lib batch_cholesky_parallel_path_is_bit_identical_to_serial -- --nocapture
rch exec -- cargo check -j 1 -p fj-dispatch --all-targets
rch exec -- cargo clippy -j 1 --no-deps -p fj-dispatch --all-targets -- -D warnings
rch exec -- cargo test -j 1 -p fj-dispatch --lib
```

Results:

- `cargo fmt -p fj-dispatch --check`: pass.
- `git diff --check`: pass.
- `ubs crates/fj-dispatch/src/batching.rs`: exit 0; no critical findings, with
  pre-existing broad warning inventory for this large test-heavy file.
- Focused golden proof: pass on `vmi1149989`.
- `cargo check -p fj-dispatch --all-targets`: pass on `vmi1149989`; dependency
  compile still reports the pre-existing `fj-trace` unused-variable warning.
- Strict no-deps clippy for `fj-dispatch`: pass on `vmi1152480`; same dependency
  warning appears during compilation but not in the touched package lint.
- `cargo test -p fj-dispatch --lib`: pass on `vmi1152480`, `286 passed; 0
  failed; 4 ignored`.

## Routing

This closes the profitable `std::thread::scope` fan-out family for
`frankenjax-8ndmr`: eig, F64 SVD, non-F64/F32 SVD, and Cholesky now have proof
or rejection evidence. The prior cheap single-pass QR/LU/solve/det route should
not be repeated without a different primitive such as a persistent worker pool
or fresh profile evidence showing spawn overhead is no longer dominant.
