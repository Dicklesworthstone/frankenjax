# FrankenJAX Test Conventions (v1)

## Naming Rules

- Unit test: `test_<subsystem>_<behavior>_<condition>`
- Property test: `prop_<subsystem>_<invariant>`
- E2E test: `e2e_<scenario_id>`

Examples:

- `test_cache_key_unknown_features_fail_closed`
- `prop_dispatch_transform_order_deterministic`
- `e2e_transform_fixture_bundle_smoke`

## Structured Log Contract

All new tests should emit or construct a `frankenjax.test-log.v1` record through `fj-test-utils::TestLogV1`.

Required identity fields:

- `test_id`
- `fixture_id` (SHA-256 over canonical fixture JSON)
- `mode`
- `result`
- `seed` (when property test seed exists)

## E2E Forensic Log Contract

New E2E scenarios must emit `frankenjax.e2e-forensic-log.v1` records through the
shared `fj_conformance::e2e_log` contract or an explicitly tracked adapter bead.

Required replay and diagnosis fields include:

- `bead_id`, `scenario_id`, `test_id`
- `command`, `working_dir`, `replay_command`
- `environment`, `feature_flags`, `fixture_ids`, `oracle_ids`
- `transform_stack`, `mode`, `inputs`, `expected`, `actual`
- `tolerance`, `error`, `timings`, `allocations`
- hash-bound `artifacts`
- `status`, `failure_summary`, `redactions`, `metadata`

Validation commands:

```bash
./scripts/validate_e2e_logs.sh
./scripts/bootstrap_e2e_forensic_log.sh
```

The validator accepts unknown future fields but rejects missing required fields,
malformed status values, empty or redacted replay commands, stale artifact hashes,
unredacted secret-like values, failing logs without summaries, and empty log sets.

## Oracle Recapture And Drift Gate

The oracle fixture table is checked by:

```bash
./scripts/run_oracle_recapture_gate.sh
```

The command emits a matrix, drift report, markdown preview, and shared-schema E2E
log. It must show all required fixture families, exact case counts, legacy
anchors, JAX/x64 metadata, fixture hashes, and strict recapture commands or
explicit unsupported rows. `--enforce` turns stale versions, changed hashes,
missing baselines, unsupported recapture paths, and missing families into a
nonzero gate.

## Property Test Configuration

Use `fj_test_utils::property_test_case_count()` for test-case volume:

- local default: `256`
- CI default: `1024`
- override: `FJ_PROPTEST_CASES=<N>`

Seed capture precedence:

1. `FJ_PROPTEST_SEED`
2. `PROPTEST_RNG_SEED`

## Coverage Gate Policy

Coverage should be measured with `cargo-llvm-cov` and compared against per-crate floors.

Suggested command:

```bash
cargo llvm-cov --workspace --lcov --output-path artifacts/testing/coverage.lcov
```

Initial floor targets:

- core crates (`fj-core`, `fj-dispatch`, `fj-lax`, `fj-cache`): line >= 90%, branch >= 80%
- supporting crates (`fj-ad`, `fj-interpreters`, `fj-runtime`, `fj-ledger`, `fj-egraph`, `fj-conformance`): line >= 85%, branch >= 75%
