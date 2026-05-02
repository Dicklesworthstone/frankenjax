# TTL Semantic Proof Matrix Gate

- schema: `frankenjax.ttl-semantic-proof-matrix.v1`
- bead: `frankenjax-cstq.3`
- status: `pass`
- accepted rows: `6`
- rejected rows: `6`

| Case | Status | Decision | Stack | Oracle | Rejection |
|---|---:|---|---|---|---|
| `valid_single_jit_square` | `pass` | `accept` | `jit` | `analytic:jax.jit(lambda x: x*x)(4.0)` | `n/a` |
| `valid_single_grad_square` | `pass` | `accept` | `grad` | `analytic:jax.grad(lambda x: x*x)(4.0)` | `n/a` |
| `valid_single_vmap_square` | `pass` | `accept` | `vmap` | `vmap_square` | `n/a` |
| `valid_jit_grad_fixture` | `pass` | `accept` | `jit>grad` | `jit_grad_poly_x5.0` | `n/a` |
| `valid_grad_jit_order_sensitive` | `pass` | `accept` | `grad>jit` | `analytic:jax.grad(jax.jit(lambda x: x*x+3*x))(5.0)` | `n/a` |
| `valid_vmap_grad_fixture` | `pass` | `accept` | `vmap>grad` | `vmap_grad_sin` | `n/a` |
| `fail_closed_grad_vmap_vector_output` | `pass` | `reject` | `grad>vmap` | `analytic:strict-fail-closed:jax.grad(jax.vmap(lambda x: x*x))` | `transform_execution.non_scalar_gradient_input` |
| `invalid_duplicate_evidence` | `pass` | `reject` | `grad>grad` | `n/a` | `transform_invariant.duplicate_evidence` |
| `invalid_missing_evidence` | `pass` | `reject` | `jit` | `n/a` | `transform_invariant.missing_evidence` |
| `invalid_stale_input_fingerprint` | `pass` | `reject` | `jit` | `analytic:stale-input-fingerprint-regression` | `semantic.stale_input_fingerprint` |
| `invalid_wrong_transform_binding` | `pass` | `reject` | `jit` | `n/a` | `transform_invariant.wrong_transform_binding` |
| `invalid_missing_fixture_link` | `pass` | `reject` | `jit>grad` | `n/a` | `semantic.missing_oracle_fixture_link` |

## Issues

No TTL semantic proof matrix issues found.
