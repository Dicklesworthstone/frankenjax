# FJ-P2C-005 Risk Note

Subsystem: Compilation cache/keying
Generated: 2026-05-01T00:00:00Z

## Threat Surface

This packet is exposed to compatibility, correctness, and operability drift between the legacy JAX anchors, the Rust implementation, and the committed oracle/e2e evidence.

## Top Open Risk

- Risk ID: `risk.p2c005.cache-signal-parity`
- Category: `security`
- Description: Strict cache keys are deterministic, but full legacy signal alignment remains an expansion target.
- Likelihood: medium
- Impact: high
- Mitigation status: in progress
- Tracking bead: `frankenjax-fcxy.5`

## Mitigations

- Packet anchor maps and contract tables remain committed under `artifacts/phase2c/FJ-P2C-005/`.
- `fixture_manifest.json` binds packet evidence to checksummed fixture or e2e artifacts.
- `parity_gate.yaml` records strict and hardened gate status for the scoped evidence.
- `parity_report.json` is protected by `parity_report.raptorq.json` and `parity_report.decode_proof.json`.

## Residual Risk

Residual risk is accepted only for the declared V1 scope. Any expansion beyond the scoped evidence must add oracle fixtures, update the parity gate, and refresh the packet durability artifacts.
