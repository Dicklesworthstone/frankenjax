# frankenjax-wpjbg QR 4096 closeout

- Date: 2026-06-11
- Bead: `frankenjax-wpjbg`
- Target: `fj-lax` real QR large-n WY-blocked path
- Outcome: evidence-only closeout. The production gate already routes `eval_qr_real_matrix` through the WY-blocked implementation when `min(m,n) >= QR_BLOCK_MIN` (`2048`).

## Baseline / A-B

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_REQUIRE_REMOTE \
rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-wpjbg-qr4096-profile-remote \
  cargo bench -j 1 -p fj-lax --bench lax_baseline -- \
  linalg/qr_4096 --sample-size 10 --measurement-time 3 --warm-up-time 1 --noplot
```

Worker: `vmi1227854`

Results:

- `linalg/qr_4096_scalar`: `907.94 ms` mean, interval `[898.39 ms, 919.93 ms]`
- `linalg/qr_4096_blocked`: `723.06 ms` mean, interval `[698.69 ms, 751.04 ms]`

Ratio: `907.94 / 723.06 = 1.256x`.

This validates the bead's remaining large-n hypothesis: the WY-blocked path loses at `n <= 2048` in older evidence, but wins once the trailing matrix repeatedly spills cache at `n=4096`.

## Proof

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,RCH_REQUIRE_REMOTE \
rch exec -- env CARGO_TARGET_DIR=/data/tmp/frankenjax-wpjbg-proof \
  cargo test -j 1 -p fj-lax --lib qr_blocked_reconstructs_and_orthonormal -- --nocapture
```

Worker: `vmi1227854`

Result: `linalg::tests::qr_blocked_reconstructs_and_orthonormal ... ok`.

## Isomorphism

The large-n WY-blocked path is intentionally tolerance-equivalent rather than bit-identical to the scalar reflector loop because it reassociates trailing updates. This is isolated behind the large-size gate; small conformance/golden cases remain on the scalar bit-identical path. The proof checks reconstruction and orthonormality for the blocked route.

No source change was required for this closeout.
