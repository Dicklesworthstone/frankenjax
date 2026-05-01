# FrankenJAX Schema Version Governance

This document governs schema evolution for artifacts defined under `artifacts/schemas/`.

## Scope

The following schema families are versioned and required for packet evidence and gate automation:

- `frankenjax.legacy-anchor-map.v1`
- `frankenjax.contract-table.v1`
- `frankenjax.fixture-manifest.v1`
- `frankenjax.parity-gate.v1`
- `frankenjax.risk-note.v1`
- `frankenjax.compatibility-matrix.v1`
- `frankenjax.test-log.v1`
- `frankenjax.e2e-forensic-log.v1`

## Stability Contract

- A schema version is immutable once published.
- Producers must emit `schema_version` matching the schema constant.
- Consumers must fail closed in strict mode when `schema_version` is unknown.
- Hardened mode may only accept unknown versions if an explicit allowlist entry exists and an audit record is emitted.

## Versioning Model

- Version token format: `frankenjax.<artifact-name>.vN` where `N` is a positive integer.
- This project uses major-only schema versioning (`v1`, `v2`, ...).
- Non-breaking clarifications update docs and examples, not `schema_version`.

## Mandatory Major-Bump Triggers

Bump to `v(N+1)` when any of the following occurs:

1. A required field is added or removed.
2. A field type changes (for example `string` to `object`).
3. Enum values are removed or semantics are narrowed.
4. Validation constraints become stricter in a way that can reject previously valid artifacts.
5. The meaning of a field changes incompatibly.

## No-Bump Changes

The following do not require a major bump:

1. Documentation clarifications.
2. Additional non-required example files.
3. Relaxing constraints in a backwards-compatible way.

## Release Checklist for Schema Changes

1. Create new schema file (`*.vN.schema.json`) and retain prior versions.
2. Add one valid and one invalid example for the new version.
3. Extend validation tests to include the new files.
4. Update this governance document and changelog notes.
5. Attach migration notes in packet risk/evidence artifacts.

## CI Gate Expectations

- Every schema must have at least one valid and one invalid example.
- Validation tests must assert:
  - valid examples pass,
  - invalid examples fail,
  - missing required fields mark artifact `NOT READY`.
