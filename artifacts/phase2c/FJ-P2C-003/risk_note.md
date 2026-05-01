# FJ-P2C-003 Risk Note

Subsystem: Partial evaluation and staging
Generated: 2026-05-01T00:00:00Z

## Threat Surface

This packet is exposed to compatibility, correctness, and operability drift between the legacy JAX anchors, the Rust implementation, and the committed oracle/e2e evidence.

## Top Open Risk

- Risk ID: `risk.p2c003.residual-graph-drift`
- Category: `correctness`
- Description: Residual graph behavior needs broader upstream oracle capture for high-rank and nested transform programs.
- Likelihood: medium
- Impact: high
- Mitigation status: in progress
- Tracking bead: `frankenjax-fcxy.2`

## Mitigations

- Packet anchor maps and contract tables remain committed under `artifacts/phase2c/FJ-P2C-003/`.
- `fixture_manifest.json` binds packet evidence to checksummed fixture or e2e artifacts.
- `parity_gate.yaml` records strict and hardened gate status for the scoped evidence.
- `parity_report.json` is protected by `parity_report.raptorq.json` and `parity_report.decode_proof.json`.

## Residual Risk

Residual risk is accepted only for the declared V1 scope. Any expansion beyond the scoped evidence must add oracle fixtures, update the parity gate, and refresh the packet durability artifacts.
