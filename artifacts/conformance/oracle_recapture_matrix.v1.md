# Oracle Recapture Matrix

- matrix status: `fail`
- drift gate: `fail`
- cases: `848/848`

| Family | Cases | Oracle | X64 | Status | Recapture |
|---|---:|---|---|---|---|
| `transforms` | 613/613 | `0.9.2.dev20260316+12a2449` | `true` | `pass` | `python3 crates/fj-conformance/scripts/capture_legacy_fixtures.py --legacy-root legacy_jax_code/jax --output crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json --strict` |
| `rng` | 25/25 | `0.9.2.dev20260316+12a2449` | `true` | `pass` | `python3 crates/fj-conformance/scripts/capture_legacy_fixtures.py --legacy-root legacy_jax_code/jax --output crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json --rng-output crates/fj-conformance/fixtures/rng/rng_determinism.v1.json --strict` |
| `linalg_fft` | 33/33 | `0.9.2` | `unknown` | `fail` | `python3 crates/fj-conformance/scripts/capture_linalg_fft_oracle.py --legacy-root legacy_jax_code/jax --output crates/fj-conformance/fixtures/linalg_fft_oracle.v1.json` |
| `composition` | 15/15 | `0.9.1` | `true` | `fail` | `missing` |
| `dtype_promotion` | 162/162 | `0.9.1` | `true` | `fail` | `missing` |

## Gate Issues

- `x64_mode_mismatch` `linalg_fft`: expected x64 Some(true), found None
- `stale_oracle_version` `composition`: expected oracle version prefix 0.9.2, found 0.9.1
- `missing_recapture_command` `composition`: family has no strict automated recapture command
- `stale_oracle_version` `dtype_promotion`: expected oracle version prefix 0.9.2, found 0.9.1
- `missing_recapture_command` `dtype_promotion`: family has no strict automated recapture command
