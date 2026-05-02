# Numerical Stability Matrix

- Schema: `frankenjax.numerical-stability.v1`
- Bead: `frankenjax-cstq.20`
- Status: `pass`
- Rows: `11`
- Platform fingerprints: `1`

| Family | DType | Edge | Guard | Max abs | Non-finite | Dashboard |
|---|---|---|---|---:|---|---|
| `special_math_tails` | `f64` | `tail_and_cancellation` | `named_polynomial_or_series_approximation` | `8.000e-8` | `finite` | `stability/special_math_tails` |
| `linalg_near_singular` | `f64` | `near_singular_or_clustered_spectrum` | `diagonal_and_eigenvalue_gap_guards` | `1.600e-2` | `finite` | `stability/linalg_near_singular` |
| `ad_gradient_check` | `f64` | `finite_difference_step_governance` | `sqrt_epsilon_step_clamped_to_safe_range` | `4.000e-5` | `finite` | `stability/ad_gradient_check` |
| `fft_scaling` | `complex128` | `hermitian_scaling_and_inverse_normalization` | `exact_inverse_length_scaling` | `5.000e-13` | `finite` | `stability/fft_scaling` |
| `dtype_promotion` | `mixed` | `mixed_dtype_exact_contract` | `dtype_lattice_row_must_match_oracle` | `0.000e0` | `not_applicable` | `stability/dtype_promotion` |
| `complex_branch_values` | `complex128` | `signed_zero_and_componentwise_branch` | `componentwise_real_imag_comparison` | `6.000e-13` | `finite` | `stability/complex_branch_values` |
| `rng_determinism` | `u32` | `counter_based_seed_stream` | `threefry_constants_and_counter_order` | `0.000e0` | `deterministic_seed_stream` | `stability/rng_determinism` |
| `literal_bit_roundtrip` | `f64` | `nan_payload_and_signed_zero_bits` | `store_float_literals_as_raw_bits` | `0.000e0` | `nan_payload_preserved:0008000000001234` | `stability/literal_bit_roundtrip` |
| `non_finite_division` | `f64` | `nan_inf_division_by_zero` | `ieee_754_non_finite_classification` | `0.000e0` | `nan_and_signed_infinity_classified` | `stability/non_finite_division` |
| `finite_diff_policy` | `f64` | `fallback_step_and_threshold_governance` | `fallback_policy_can_be_denied_or_replayed` | `5.000e-5` | `finite_diff_guarded` | `stability/finite_diff_policy` |
| `platform_metadata` | `metadata` | `cross_platform_replay_metadata` | `os_arch_endian_rust_cargo_target_fingerprint` | `0.000e0` | `not_applicable` | `stability/platform_metadata` |

No numerical-stability issues found.
