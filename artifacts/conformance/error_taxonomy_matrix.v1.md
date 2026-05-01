# Error Taxonomy Matrix Gate

- schema: `frankenjax.error-taxonomy-matrix.v1`
- bead: `frankenjax-cstq.8`
- status: `pass`
- cases: `13`
- typed error rows: `12`
- success rows: `1`
- strict/hardened divergences: `2`

| Case | Boundary | Mode | Input | Expected | Actual | Panic | Divergence |
|---|---|---|---|---|---|---|---|
| `ir_validation_unknown_outvar` | `fj-core::Jaxpr::validate_well_formed` | `strict_and_hardened` | `jaxpr outvar has no defining input, const, or equation binding` | `ir_validation.unknown_outvar` | `ir_validation.unknown_outvar` | `no_panic` | `none` |
| `transform_proof_duplicate_evidence` | `fj-core::verify_transform_composition` | `strict_and_hardened` | `TraceTransformLedger repeats an evidence id across two transforms` | `transform_proof.duplicate_evidence` | `transform_proof.duplicate_evidence` | `no_panic` | `none` |
| `transform_proof_missing_evidence` | `fj-core::verify_transform_composition` | `strict_and_hardened` | `TraceTransformLedger has fewer evidence entries than transforms` | `transform_proof.evidence_count_mismatch` | `transform_proof.evidence_count_mismatch` | `no_panic` | `none` |
| `primitive_arity_add` | `fj-lax::eval_primitive` | `strict_and_hardened` | `add called with one scalar operand` | `primitive.arity_mismatch` | `primitive.arity_mismatch` | `no_panic` | `none` |
| `primitive_shape_add_broadcast` | `fj-lax::eval_primitive` | `strict_and_hardened` | `add called with non-broadcastable [2] and [3] tensors` | `primitive.shape_mismatch` | `primitive.shape_mismatch` | `no_panic` | `none` |
| `primitive_type_sin_bool` | `fj-lax::eval_primitive` | `strict_and_hardened` | `sin called with bool scalar` | `primitive.type_mismatch` | `primitive.type_mismatch` | `no_panic` | `none` |
| `interpreter_missing_variable` | `fj-interpreters::eval_jaxpr` | `strict_and_hardened` | `equation references v7 before any binding` | `interpreter.missing_variable` | `interpreter.missing_variable` | `no_panic` | `none` |
| `cache_strict_unknown_feature` | `fj-cache::build_cache_key` | `strict` | `cache key input contains unknown incompatible feature marker` | `cache.unknown_incompatible_features` | `cache.unknown_incompatible_features` | `no_panic` | `allowed` |
| `cache_hardened_unknown_feature` | `fj-cache::build_cache_key` | `hardened` | `cache key input contains unknown incompatible feature marker` | `none` | `none` | `no_panic` | `allowed` |
| `vmap_axis_mismatch` | `fj-dispatch::dispatch` | `strict_and_hardened` | `vmap over two vectors with leading dimensions 2 and 3` | `transform_execution.vmap_mismatched_leading_dimension` | `transform_execution.vmap_mismatched_leading_dimension` | `no_panic` | `none` |
| `durability_missing_artifact` | `fj-conformance::durability::encode_artifact_to_sidecar` | `strict_and_hardened` | `durability sidecar generation points at a missing artifact path` | `durability.io` | `durability.io` | `no_panic` | `none` |
| `unsupported_transform_tail_without_fallback` | `fj-dispatch::dispatch` | `strict_and_hardened` | `grad(vmap(square)) with finite-difference fallback explicitly denied` | `transform_execution.finite_diff_grad_fallback_disabled` | `transform_execution.finite_diff_grad_fallback_disabled` | `no_panic` | `none` |
| `unsupported_control_flow_grad_vmap_vector` | `fj-dispatch::dispatch` | `strict_and_hardened` | `grad(vmap(cond)) row receives a vector first argument in V1 scope` | `transform_execution.non_scalar_gradient_input` | `transform_execution.non_scalar_gradient_input` | `no_panic` | `none` |

## Strict/Hardened Divergence Allowlist

- `cache_strict_unknown_feature`
- `cache_hardened_unknown_feature`

## Replay Hints

- `ir_validation_unknown_outvar`: Validate a Jaxpr whose only outvar is v99. (`./scripts/run_error_taxonomy_gate.sh --enforce --case ir_validation_unknown_outvar`)
- `transform_proof_duplicate_evidence`: Verify a JIT/JIT ledger with the same evidence string in both slots. (`./scripts/run_error_taxonomy_gate.sh --enforce --case transform_proof_duplicate_evidence`)
- `transform_proof_missing_evidence`: Verify a ledger with one JIT transform and zero evidence ids. (`./scripts/run_error_taxonomy_gate.sh --enforce --case transform_proof_missing_evidence`)
- `primitive_arity_add`: Evaluate add with a single I64 scalar input. (`./scripts/run_error_taxonomy_gate.sh --enforce --case primitive_arity_add`)
- `primitive_shape_add_broadcast`: Evaluate add on F64 vectors with lengths 2 and 3. (`./scripts/run_error_taxonomy_gate.sh --enforce --case primitive_shape_add_broadcast`)
- `primitive_type_sin_bool`: Evaluate sin(true). (`./scripts/run_error_taxonomy_gate.sh --enforce --case primitive_type_sin_bool`)
- `interpreter_missing_variable`: Evaluate a one-equation Jaxpr whose add input is an unbound var. (`./scripts/run_error_taxonomy_gate.sh --enforce --case interpreter_missing_variable`)
- `cache_strict_unknown_feature`: Build a strict cache key with unknown_incompatible_features=[future_xla_flag]. (`./scripts/run_error_taxonomy_gate.sh --enforce --case cache_strict_unknown_feature`)
- `cache_hardened_unknown_feature`: Build a hardened cache key with unknown_incompatible_features=[future_xla_flag]. (`./scripts/run_error_taxonomy_gate.sh --enforce --case cache_hardened_unknown_feature`)
- `vmap_axis_mismatch`: Dispatch vmap(add) with vector lengths 2 and 3. (`./scripts/run_error_taxonomy_gate.sh --enforce --case vmap_axis_mismatch`)
- `durability_missing_artifact`: Attempt sidecar generation for a missing artifact file. (`./scripts/run_error_taxonomy_gate.sh --enforce --case durability_missing_artifact`)
- `unsupported_transform_tail_without_fallback`: Dispatch grad(vmap(square)) with allow_finite_diff_grad_fallback=deny. (`./scripts/run_error_taxonomy_gate.sh --enforce --case unsupported_transform_tail_without_fallback`)
- `unsupported_control_flow_grad_vmap_vector`: Dispatch grad(vmap(cond_select)) with a vector first argument. (`./scripts/run_error_taxonomy_gate.sh --enforce --case unsupported_control_flow_grad_vmap_vector`)

## Issues

No error taxonomy matrix issues found.
