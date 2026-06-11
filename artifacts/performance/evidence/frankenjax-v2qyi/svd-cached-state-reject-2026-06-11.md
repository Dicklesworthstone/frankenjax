# frankenjax-v2qyi: SVD Cached-State Reject

## Scope

- Bead: `frankenjax-v2qyi`
- Target: `fj-lax` real one-sided Jacobi SVD, `linalg/svd_48x48_f64`
- Pass: `pass226`
- Production source after rejection: unchanged from `HEAD`

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -p fj-lax --bench lax_baseline -- linalg/svd_48x48_f64 --sample-size 20 --warm-up-time 1 --measurement-time 3
```

Result:

- Worker: `vmi1227854`
- Time: `[1.1364 ms 1.1640 ms 1.1992 ms]`

## Candidate A: Versioned Inactive-Pair Certificate

Idea: record the column versions for pairs that fail the Jacobi threshold. If neither column changed by the next visit, skip the dot product because the pair would make the same no-rotation decision.

Proof:

- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo test -p fj-lax --lib 48x48 -- --nocapture`
- Golden digest stayed `6f1b0069586dda5b23d377bbb171a18ac0e24b6e0309dabc4ad0e0d2d1864d90`

Rebench:

- Time: `[1.1829 ms 1.2278 ms 1.2828 ms]`
- Decision: rejected. Median and conservative interval both regressed.

## Candidate B: Cached Column Norms

Idea: maintain squared norms for working columns and refresh them during the row-ordered rotation loop, avoiding alpha/beta rescans while keeping the gamma dot and pair order unchanged.

Proof:

- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo test -p fj-lax --lib 48x48 -- --nocapture`
- Golden digest stayed `6f1b0069586dda5b23d377bbb171a18ac0e24b6e0309dabc4ad0e0d2d1864d90`

Rebench:

- Time: `[1.3545 ms 1.4023 ms 1.4511 ms]`
- Decision: rejected. The extra norm maintenance outweighed removed rescans.

## Conclusion

Cached-state Jacobi variants are the wrong lever for this profile target. The next SVD pass should be algorithmically different, with an explicit golden-output fallback: deterministic rank-revealing reduction/deflation before Jacobi, or a blocked Jacobi primitive whose output contract is pinned before production routing.
