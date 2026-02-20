# Invariant Checklist: API Front-Door (FJ-P2C-002)

## Checked Invariants

- [x] **jit(f)(x) == f(x) for all pure functions**
  - Oracle: `oracle_jit_output_matches_direct_eval` (exact match)
  - Metamorphic: `metamorphic_jit_transparent` (3 program specs)
  - Property: `prop_jit_is_identity` (256+ random inputs)
  - E2E: `e2e_p2c002_user_api_jit_basic` (API → eval_jaxpr comparison)

- [x] **grad returns correct derivative for test suite**
  - Oracle: `oracle_grad_output_matches_analytical` (7 test points, tolerance 1e-3)
  - Metamorphic: `metamorphic_grad_linear_constant` (5 test points, ratio check)
  - Property: `prop_grad_square_is_2x` (256+ random inputs)
  - E2E: `e2e_p2c002_user_api_grad_polynomial` (grad + value_and_grad cross-check)

- [x] **vmap(f)(batch)[i] == f(batch[i]) for all i**
  - Oracle: `oracle_vmap_output_matches_elementwise` (5-element batch)
  - Metamorphic: `metamorphic_vmap_distributes` (10-element batch)
  - Property: `prop_vmap_add_one_increments` (1-20 element random batches)
  - E2E: `e2e_p2c002_user_api_vmap_batch` (5-element batch with direct eval comparison)

- [x] **Transform composition order matches user specification**
  - Oracle: `oracle_composition_jit_grad_consistent` (4 test points)
  - Unit: `jit_grad_composition`, `jit_vmap_composition`, `vmap_grad_composition`, `compose_helper`
  - Integration: `stacking_jit_grad`, `stacking_jit_vmap`, `stacking_vmap_grad`, `stacking_compose_jit_vmap_grad`
  - Property: `prop_jit_grad_consistent_with_grad` (256+ inputs)
  - E2E: `e2e_p2c002_user_api_stacking` (builder vs compose vs standalone)
  - Adversarial: `adversarial_grad_vmap_composition` (illegal order rejected)

- [x] **Error messages are actionable and user-friendly**
  - Integration: `error_display_is_user_friendly` (no internal types leaked)
  - Unit: `grad_non_scalar_input_fails`, `grad_empty_args_fails`, `vmap_scalar_input_fails`, `grad_vmap_fails_non_scalar`
  - E2E: `e2e_p2c002_user_api_error_messages` (4 error paths, no panics, no DispatchError)

## Verification Summary

All 5 invariants verified. 63 tests covering oracle, metamorphic, adversarial, property, unit, integration, and e2e validation.
