# frankenjax-p1vbf.53 GEMM2048 Superpanel Rejection

Date: 2026-06-09
Agent: BoldFalcon
Target: `linalg/matmul_2d_2048x2048x2048_f64`

## Profile-Backed Baseline

Clean worktree: `origin/main` at `835644d3`
RCH worker: `ovh-a`
Command:

```bash
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR CARGO_TARGET_DIR=target/rch-boldfalcon-linalg-baseline \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  'linalg/(eig_48x48_f64|cholesky_1024x1024_f64|qr_1024x1024_f64|lu_1024x1024_f64|pinv_256x128_f64|matmul_2d_1024x1024x1024_f64|matmul_2d_2048x2048x2048_f64)' \
  --sample-size 10 --warm-up-time 1 --measurement-time 2
```

Observed rows:

- `linalg/matmul_2d_2048x2048x2048_f64`: 199.62 ms mean, [194.22, 203.97] ms
- `linalg/matmul_2d_1024x1024x1024_f64`: 30.870 ms mean, [30.497, 31.492] ms
- `linalg/cholesky_1024x1024_f64`: 53.606 ms mean, [51.150, 56.427] ms
- `linalg/qr_1024x1024_f64`: 44.072 ms mean, [43.259, 44.812] ms
- `linalg/lu_1024x1024_f64`: 52.055 ms mean, [43.766, 64.303] ms

## Candidate Lever

One lever tested in `crates/fj-lax/src/tensor_contraction.rs`:
inside each row/column superpanel, changed traversal from tile-outer/panel-inner to
panel-outer/tile-inner so a packed B panel was visited across all row tiles before
advancing to the next column panel.

Arithmetic isomorphism obligation: every output cell still accumulates products in
ascending `k`; no FMA, no Strassen, no mixed precision, no floating-point
reassociation, no RNG, and no dtype/shape/error-surface changes.

## Behavior Proof

Command:

```bash
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR CARGO_TARGET_DIR=target/rch-boldfalcon-gemm-test \
  rch exec -- cargo test -p fj-lax tensor_contraction::tests:: -- --nocapture
```

Result: passed. The run covered 32 tensor-contraction tests, including the
bit-identical blocked-matmul, B-pack layout, parallel/serial B-pack equality,
and packed-NR8 golden digest tests. Four ignored stress tests remained ignored.

Golden proof: existing focused tensor-contraction golden digest test passed
unchanged. The candidate altered independent tile scheduling only; per-cell
product order and output bits stayed covered by the bit-identical tests.

## Same-Worker Rebench

RCH worker: `ovh-a`
Command:

```bash
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR CARGO_TARGET_DIR=target/rch-boldfalcon-gemm-candidate \
  rch exec -- cargo bench -p fj-lax --bench lax_baseline -- \
  'linalg/matmul_2d_(1024x1024x1024|2048x2048x2048)_f64' \
  --sample-size 10 --warm-up-time 1 --measurement-time 2
```

Observed rows:

- `linalg/matmul_2d_1024x1024x1024_f64`: 30.320 ms mean, [29.927, 30.807] ms
- `linalg/matmul_2d_2048x2048x2048_f64`: 203.72 ms mean, [200.22, 207.56] ms

## Decision

Rejected. Same-worker GEMM2048 moved from 199.62 ms to 203.72 ms:

- median speedup: 0.980x
- conservative confidence: regression with overlapping intervals
- Score: 0.0, below the required 2.0 keep threshold

The source change was restored. Do not repeat this exact panel-outer/tile-inner
superpanel schedule.

## Next Route

The rejection indicates loop-order micro-levers are the wrong family here. The
next profile-backed route should attack a deeper communication-avoiding primitive:
thread/topology-aware row-block assignment, cache-oblivious recursive paneling, or
an exact-order blocked sink that reduces materialization without changing any
per-cell ascending-k arithmetic.
