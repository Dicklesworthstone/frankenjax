# Conformance Fixtures

This folder stores normalized oracle-vs-target fixtures for `fj-conformance`.

## Files

- `smoke_case.json`: bootstrap fixture ensuring harness wiring works.
- `transforms/legacy_transform_cases.v1.json`: transform fixture suite for `jit`/`grad`/`vmap` plus composition cases.
- `rng/rng_determinism.v1.json`: random determinism fixture suite (`key`, `split`, `fold_in`, `uniform`, `normal`) over a fixed seed set.
- `linalg_fft_oracle.v1.json`: linalg and FFT oracle cases. Current fixture metadata records JAX 0.9.2 but lacks an explicit x64 flag and is intentionally flagged by the recapture gate until refreshed.
- `composition_oracle.v1.json`: transform composition oracle cases. Current fixture metadata records JAX 0.9.1 and is intentionally flagged by the recapture gate until refreshed.
- `dtype_promotion_oracle.v1.json`: dtype promotion oracle matrix. Current fixture metadata records JAX 0.9.1 and is intentionally flagged by the recapture gate until refreshed.

## Regeneration

Use the legacy capture script:

```bash
python3 crates/fj-conformance/scripts/capture_legacy_fixtures.py \
  --legacy-root /data/projects/frankenjax/legacy_jax_code/jax \
  --output /data/projects/frankenjax/crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json \
  --rng-output /data/projects/frankenjax/crates/fj-conformance/fixtures/rng/rng_determinism.v1.json
```

If JAX/jaxlib are unavailable in the environment, the script will fail with an explicit setup error.

Run the recapture matrix and drift gate with:

```bash
./scripts/run_oracle_recapture_gate.sh
```

The generated matrix records every required fixture family, case count, legacy
anchor, recapture command, oracle version, x64 mode, fixture hash, and drift-gate
issue. Use `--enforce` when CI should fail on stale or unsupported rows.
