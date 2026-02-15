# Risk Note: Partial Evaluation and Staging (FJ-P2C-003)

## Threat Model

### 1. Incorrect Constant Folding
**Risk**: Partial eval with all-known inputs folds constants incorrectly, producing wrong values that propagate silently through downstream computations.
**Mitigation**: Constant folding reuses `eval_jaxpr` for known sub-programs, ensuring folded values match full evaluation. All-known PE produces identical results to direct `eval_jaxpr`.
**Evidence**: `oracle_pe_constant_folding_correctness`, `metamorphic_pe_all_known_equals_full_eval`, `e2e_p2c003_jit_constant_folding`, `test_pe_constant_fold_*` (6 tests), `prop_pe_staging_semantic_equivalence`.
**Residual Risk**: LOW

### 2. Residual Jaxpr Semantic Drift
**Risk**: The residual (unknown) Jaxpr fails to reproduce the same outputs as the original program when combined with known constants, causing staging pipeline to produce incorrect results.
**Mitigation**: Staging roundtrip verification: `stage_jaxpr` + `execute_staged` must match `eval_jaxpr` for all input combinations. 200-input roundtrip test covers both neg_mul and Add2 programs across the full i64 range.
**Evidence**: `e2e_p2c003_staging_roundtrip` (200 inputs), `oracle_staging_pipeline_equivalence`, `prop_pe_staging_semantic_equivalence`, `test_pe_staging_equivalence_*` (2 tests).
**Residual Risk**: LOW

### 3. Variable Renumbering Corruption
**Risk**: During PE split, VarId remapping between known/unknown sub-Jaxprs creates dangling references or variable collisions that cause panics or undefined behavior.
**Mitigation**: Residual Jaxpr construction carefully maps unknown variables via `residual_vars` and uses `VarId(n + i)` allocation to prevent collisions. All residual Jaxpr variable references are verified through staging execution.
**Evidence**: `oracle_pe_residual_shape_correctness`, `adversarial_pe_identity_jaxpr`, `test_pe_mixed_residual_count_matches`, `test_pe_full_residual_*` (5 tests).
**Residual Risk**: LOW

### 4. DCE Removes Live Equations
**Risk**: Dead code elimination incorrectly removes equations that are transitively needed by outputs, causing missing variables in the resulting Jaxpr.
**Mitigation**: DCE performs backward reachability from output variables, marking all transitively-needed equations. Chain dependency tests verify that intermediate equations are preserved.
**Evidence**: `oracle_dce_correctness`, `test_dce_keeps_chain_dependencies`, `test_dce_selective_multi_output`, `prop_dce_all_used_preserves_chain`, `prop_dce_never_increases_equations`, `adversarial_dce_large_chain`.
**Residual Risk**: LOW

### 5. Large Graph Resource Exhaustion
**Risk**: Partial evaluation of very large Jaxpr programs causes excessive memory allocation or processing time, potentially enabling denial-of-service.
**Mitigation**: Bitset optimization uses O(max_var_id) memory (not O(equations)). 1000-equation stress tests complete within <500ms budget. No unbounded allocations in the PE hot path.
**Evidence**: `adversarial_pe_large_chain` (10K equations), `e2e_p2c003_large_graph_staging` (1000 equations within budget), `adversarial_dce_large_chain` (10K equations).
**Residual Risk**: MEDIUM (no hard size limits, by design mirrors JAX)

## Performance Summary

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| PE all-known 10eq | 569ns | 478ns | 16.0% |
| PE all-unknown 10eq | 1125ns | 493ns | 56.2% |
| DCE all-used 10eq | 832ns | 601ns | 27.8% |
| DCE all-used 1000eq | 65958ns | 46450ns | 29.6% |
| Staging chain 100eq | 10751ns | 10062ns | 6.4% |

Optimization lever: VarId-indexed bitset with fast-path shortcuts for all-known and all-unknown cases.
Evidence: `artifacts/performance/evidence/partial_eval/bitset_fast_path.json`

## Overall Assessment

Partial Evaluation subsystem is **LOW RISK** for Phase 2C deployment. All 86 validation tests pass. Constant folding correctness is verified through oracle comparison and metamorphic properties. Staging roundtrip equivalence confirmed across 200 input values. Performance exceeds all baseline targets by >4x margin. No regressions detected.
