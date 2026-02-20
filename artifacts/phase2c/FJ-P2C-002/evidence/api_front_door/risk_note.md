# Risk Note: API Front-Door (FJ-P2C-002)

## Threat Model

### 1. Argument Injection
**Risk**: Malformed user input (wrong types, wrong arity) causes panics or undefined behavior.
**Mitigation**: All API entry points (`jit`, `grad`, `vmap`, `value_and_grad`) validate inputs at the dispatch layer. Errors are caught and wrapped in `ApiError` variants. `#![forbid(unsafe_code)]` on all crates.
**Evidence**: `adversarial_grad_no_args`, `adversarial_scalar_to_vmap`, `adversarial_vmap_mismatched_dims`, `error_grad_empty_args`, `error_vmap_scalar_input`.
**Residual Risk**: LOW

### 2. Composition Bomb
**Risk**: Deeply nested or illegal transform compositions cause resource exhaustion or bypass safety checks.
**Mitigation**: `verify_transform_composition()` in fj-core enforces legal orderings. `compose()` passes through to dispatch which validates. Double jit is safely idempotent.
**Evidence**: `adversarial_double_jit` (idempotent), `adversarial_grad_vmap_composition` (rejected), `oracle_composition_jit_grad_consistent`, `stacking_compose_jit_vmap_grad`.
**Residual Risk**: LOW

### 3. Error Message Information Leakage
**Risk**: Internal error types (DispatchError, TransformExecutionError) leak implementation details to users.
**Mitigation**: `ApiError` wraps internal errors with user-friendly messages. `From<DispatchError>` maps internal variants to public-facing ones.
**Evidence**: `error_display_is_user_friendly` verifies messages don't contain "DispatchError". `e2e_p2c002_user_api_error_messages` validates 4 error paths.
**Residual Risk**: NEGLIGIBLE

### 4. API Fingerprinting
**Risk**: Timing or error-message differences between FrankenJAX and JAX allow distinguishing implementations.
**Mitigation**: Error categories align with JAX equivalents (GradRequiresScalar for non-scalar grad input, EvalError for arity mismatches). Timing differences are inherent to different implementations.
**Evidence**: `oracle_error_behavior_vector_to_grad`, `oracle_error_behavior_scalar_to_vmap` verify equivalent error categories.
**Residual Risk**: LOW (timing differences acceptable for clean-room implementation)

### 5. Value/Gradient Consistency
**Risk**: `value_and_grad` returns value and gradient from separate dispatch calls; inconsistency possible if state changes between calls.
**Mitigation**: Both calls use the same immutable Jaxpr and same args. No mutable state in dispatch pipeline.
**Evidence**: `cross_validate_value_and_grad` (6 test points), `prop_value_and_grad_consistent` (256+ property test cases), `e2e_p2c002_user_api_grad_polynomial`.
**Residual Risk**: NEGLIGIBLE

## Invariant Checklist

| Invariant | Status | Evidence |
|-----------|--------|----------|
| jit(f)(x) == f(x) for all pure functions | VERIFIED | oracle_jit_output_matches_direct_eval, metamorphic_jit_transparent, prop_jit_is_identity |
| grad returns correct derivative for test suite | VERIFIED | oracle_grad_output_matches_analytical, prop_grad_square_is_2x, api_grad_square_strict |
| vmap(f)(batch)[i] == f(batch[i]) for all i | VERIFIED | oracle_vmap_output_matches_elementwise, metamorphic_vmap_distributes, prop_vmap_add_one_increments |
| Transform composition order matches user specification | VERIFIED | oracle_composition_jit_grad_consistent, stacking_compose_jit_vmap_grad, adversarial_grad_vmap_composition |
| Error messages are actionable, not internal | VERIFIED | error_display_is_user_friendly, e2e_p2c002_user_api_error_messages |

## Performance Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| API wrapper overhead | 150-220ns | <500ns | PASS |
| jit(add)(x,y) round-trip | 1.66µs | <3µs | PASS |
| grad(x^2)(x) round-trip | 1.96µs | <3µs | PASS |
| 3-deep composition | 2.98µs | N/A | baseline |
| Mode switch overhead | ~50ns | N/A | negligible |

Evidence: `artifacts/phase2c/FJ-P2C-002/evidence/perf/api_front_door_profile.json`

## Overall Assessment

API Front-Door subsystem is **LOW RISK** for Phase 2C deployment. All 63 validation tests pass. Error handling wraps internal types correctly. Transform composition is correctly enforced. Performance exceeds all targets.
