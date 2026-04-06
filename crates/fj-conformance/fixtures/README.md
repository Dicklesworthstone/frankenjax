# Conformance Fixtures

This folder stores normalized oracle-vs-target fixtures for `fj-conformance`.

## Files

- `smoke_case.json`: bootstrap fixture ensuring harness wiring works.
- `transforms/legacy_transform_cases.v1.json`: transform fixture suite for `jit`/`grad`/`vmap` plus composition cases.
- `rng/rng_determinism.v1.json`: random determinism fixture suite (`key`, `split`, `fold_in`, `uniform`, `normal`) over a fixed seed set.

## Regeneration

Use the legacy capture script:

```bash
python3 crates/fj-conformance/scripts/capture_legacy_fixtures.py \
  --legacy-root /data/projects/frankenjax/legacy_jax_code/jax \
  --output /data/projects/frankenjax/crates/fj-conformance/fixtures/transforms/legacy_transform_cases.v1.json \
  --rng-output /data/projects/frankenjax/crates/fj-conformance/fixtures/rng/rng_determinism.v1.json
```

If JAX/jaxlib are unavailable in the environment, the script will fail with an explicit setup error.
