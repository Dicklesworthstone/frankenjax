# Optimization Isomorphism Proof Template

- scoreboard row: `<hotspot_id>`
- bead: `<br-id>`
- one lever: `<single behavior-preserving optimization lever>`
- baseline artifact: `<path>`
- profile artifact: `<path>`
- candidate artifact: `<path>`
- replay command: `<command>`

## Behavioral Surface

List the exact public operation, transform stack, mode, dtype, shape, edge cases,
and error paths affected by the lever.

## Equivalence Claim

State the pre-change and post-change outputs that must remain identical or within
the existing tolerance policy. Include cache-key, ledger, strict/hardened, and
durability side effects when relevant.

## Proof Evidence

- unit tests:
- conformance tests:
- e2e forensic logs:
- benchmark delta:
- RSS or allocation delta:
- rejected alternatives:

## Rebaseline Decision

Record whether the lever was kept, reverted before commit, or turned into a
follow-up. A kept lever must include a same-worker before/after comparison and a
non-regression explanation for every changed benchmark family.
