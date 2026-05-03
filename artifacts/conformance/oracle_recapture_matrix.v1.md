# Oracle Recapture Matrix

- matrix status: `pass`
- drift gate: `pass`
- cases: `861/861`

| Family | Cases | Oracle | X64 | Status | Recapture |
|---|---:|---|---|---|---|
| `transforms` | 613/613 | `0.9.2.dev20260316+12a2449` | `true` | `pass` | `python3 crates/fj-conformance/scripts/capture_legacy_fixtures.py --legacy-root legacy_jax_code/jax --output crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json --strict` |
| `rng` | 25/25 | `0.9.2.dev20260316+12a2449` | `true` | `pass` | `python3 crates/fj-conformance/scripts/capture_legacy_fixtures.py --legacy-root legacy_jax_code/jax --output crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json --rng-output crates/fj-conformance/fixtures/rng/rng_determinism.v1.json --strict` |
| `linalg_fft` | 46/46 | `0.9.2` | `true` | `pass` | `python3 crates/fj-conformance/scripts/capture_linalg_fft_oracle.py --legacy-root legacy_jax_code/jax --output crates/fj-conformance/fixtures/linalg_fft_oracle.v1.json` |
| `composition` | 15/15 | `0.9.2` | `true` | `pass` | `python3 crates/fj-conformance/scripts/capture_composition_oracle.py --legacy-root legacy_jax_code/jax --output crates/fj-conformance/fixtures/composition_oracle.v1.json` |
| `dtype_promotion` | 162/162 | `0.9.2` | `true` | `pass` | `python3 crates/fj-conformance/scripts/capture_dtype_promotion_oracle.py --legacy-root legacy_jax_code/jax --output crates/fj-conformance/fixtures/dtype_promotion_oracle.v1.json` |
