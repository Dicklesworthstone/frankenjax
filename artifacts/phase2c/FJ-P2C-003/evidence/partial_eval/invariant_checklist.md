# Invariant Checklist: Partial Evaluation (FJ-P2C-003)

## Checked Invariants

- [x] **Partial eval with all knowns == full eval**
  - 100x determinism replay: zero drift (`metamorphic_pe_determinism_100x`)
  - Oracle comparison: all-known PE produces same output as `eval_jaxpr` (`metamorphic_pe_all_known_equals_full_eval`)
  - Constant folding correctness verified for 6 program shapes (`test_pe_constant_fold_*`)
  - E2E verification: 2+3=5 via constant folding (`e2e_p2c003_jit_constant_folding`)

- [x] **Partial eval with no knowns preserves original Jaxpr**
  - Oracle test: all-unknown PE yields full unknown jaxpr (`oracle_pe_all_unknown_yields_full_unknown_jaxpr`)
  - Metamorphic test: identity property verified (`metamorphic_pe_no_knowns_is_identity`)
  - Property test: all-unknown yields empty known jaxpr (`prop_pe_all_unknown_yields_empty_known`)
  - Full residual tests: 5 program shapes verified (`test_pe_full_residual_*`)

- [x] **Residual Jaxpr + known values reproduce full output**
  - 200-input roundtrip: neg_mul and Add2 programs (`e2e_p2c003_staging_roundtrip`)
  - Oracle pipeline equivalence: stage+execute matches eval_jaxpr (`oracle_staging_pipeline_equivalence`)
  - Property test: staging semantic equivalence (`prop_pe_staging_semantic_equivalence`)
  - Unit roundtrip: `test_pe_staging_module_roundtrip`, `test_staging_execute_roundtrip`

- [x] **Constant folding produces correct values**
  - Single equation folding: neg(5) = -5 (`test_pe_constant_fold_single_eqn`)
  - Chain folding: add(add(x,1),1) = x+2 (`test_pe_constant_fold_add_chain`)
  - Deep chain: 100-equation add chain (`test_pe_constant_fold_deep_chain`)
  - Oracle: constant folding matches eval_jaxpr output (`oracle_pe_constant_folding_correctness`)
  - E2E: neg(5)*3 = -15 via mixed PE + staging (`e2e_p2c003_jit_mixed_known_unknown`)

- [x] **Dead code elimination does not remove live equations**
  - All-used preservation: chain dependencies kept (`test_dce_keeps_chain_dependencies`)
  - All-unused removal: equations with no output dependencies removed (`test_dce_all_unused_removes_everything`)
  - Selective multi-output: only truly unused equations removed (`test_dce_selective_multi_output`)
  - Property: DCE never increases equation count (`prop_dce_never_increases_equations`)
  - Property: all-used DCE preserves chain (`prop_dce_all_used_preserves_chain`)
  - Oracle: DCE correctness verified (`oracle_dce_correctness`)
  - Adversarial: 10K-equation chain DCE (`adversarial_dce_large_chain`)

## Verification Summary

All 5 invariants verified. 86 tests covering unit, property, oracle, metamorphic, adversarial, and E2E validation categories.
