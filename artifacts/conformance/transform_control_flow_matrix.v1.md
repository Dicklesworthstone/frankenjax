# Transform Control-Flow Matrix Gate

- schema: `frankenjax.transform-control-flow-matrix.v1`
- bead: `frankenjax-cstq.2`
- status: `pass`
- supported rows: `18`
- fail-closed rows: `3`

| Case | Support | Status | Stack | Control flow | Oracle | Comparison |
|---|---|---:|---|---|---|---|
| `jit_grad_cond_false` | `supported` | `pass` | `jit>grad` | `cond` | `analytic:jax.grad(lambda x: lax.cond(False, x, x*x))(5.0)` | `absolute_error <= 1e-6` |
| `vmap_grad_cond_mixed_predicates` | `supported` | `pass` | `vmap>grad` | `cond` | `analytic:jax.vmap(jax.grad(cond_square_identity))([3,5,7],[T,F,T])` | `all absolute_errors <= 1e-6` |
| `jit_vmap_grad_cond_false` | `supported` | `pass` | `jit>vmap>grad` | `cond` | `analytic:jax.jit(jax.vmap(jax.grad(cond_square_identity)))` | `all absolute_errors <= 1e-6` |
| `vmap_grad_scan_mul` | `supported` | `pass` | `vmap>grad` | `scan` | `analytic:jax.vmap(jax.grad(lambda init: lax.scan(mul, init, [2,5])))` | `all absolute_errors <= 1e-6` |
| `jit_vmap_grad_scan_mul` | `supported` | `pass` | `jit>vmap>grad` | `scan` | `analytic:jax.jit(jax.vmap(jax.grad(lambda init: lax.scan(mul, init, [3,4]))))` | `all absolute_errors <= 1e-6` |
| `vmap_grad_while_mul` | `supported` | `pass` | `vmap>grad` | `while` | `analytic:jax.vmap(jax.grad(lambda init: lax.while_loop(init * 2 < 8)))` | `all absolute_errors <= 1e-6` |
| `vmap_switch_batched_indices` | `supported` | `pass` | `vmap` | `switch` | `analytic:jax.vmap(lax.switch)([0,1,2],[5,6,7])` | `exact integer vector match` |
| `vmap_switch_scalar_index_batched_operand` | `supported` | `pass` | `vmap` | `switch` | `analytic:jax.vmap(lambda x: lax.switch(1, branches, x))([2,3,4])` | `exact integer vector match` |
| `grad_grad_square` | `supported` | `pass` | `grad>grad` | `n/a` | `fixtures/composition_oracle.v1.json#grad_grad_poly` | `absolute_error <= 1e-6` |
| `vmap_grad_grad_square` | `supported` | `pass` | `vmap>grad>grad` | `n/a` | `analytic:jax.vmap(jax.grad(jax.grad(lambda x: x*x)))` | `all absolute_errors <= 1e-6` |
| `jit_vmap_grad_grad_square` | `supported` | `pass` | `jit>vmap>grad>grad` | `n/a` | `analytic:jax.jit(jax.vmap(jax.grad(jax.grad(lambda x: x*x))))` | `all absolute_errors <= 1e-6` |
| `vmap_jit_grad_square` | `supported` | `pass` | `vmap>jit>grad` | `n/a` | `analytic:jax.vmap(jax.jit(jax.grad(lambda x: x*x)))` | `all absolute_errors <= 1e-6` |
| `value_and_grad_multi_output` | `supported` | `pass` | `grad` | `n/a` | `fixtures/composition_oracle.v1.json#value_and_grad_poly` | `exact value_and_grad output vector match` |
| `jacobian_quadratic` | `supported` | `pass` | `` | `n/a` | `fixtures/composition_oracle.v1.json#jacobian_quadratic` | `all absolute_errors <= 1e-6` |
| `hessian_quadratic` | `supported` | `pass` | `` | `n/a` | `fixtures/composition_oracle.v1.json#hessian_quadratic` | `all absolute_errors <= 1e-3` |
| `scan_multi_carry_state` | `supported` | `pass` | `` | `scan` | `analytic:lax.scan multi-carry count/sum` | `exact multi-carry and stacked-output match` |
| `vmap_multi_output_return` | `supported` | `pass` | `vmap` | `n/a` | `analytic:jax.vmap(lambda x: (x+1, x*2))` | `exact multi-output vector match` |
| `jit_mixed_dtype_add` | `supported` | `pass` | `jit` | `n/a` | `analytic:jax.jit(lambda x, y: x + y)(int64, float64)` | `absolute_error <= 1e-9 and output dtype f64` |
| `grad_vmap_vector_output_fail_closed` | `fail_closed` | `pass` | `grad>vmap` | `n/a` | `unsupported:v1_grad_requires_scalar_first_input_before_vmap_tail` | `expected typed error transform_execution.non_scalar_gradient_input, got transform_execution.non_scalar_gradient_input` |
| `vmap_empty_batch_fail_closed` | `fail_closed` | `pass` | `vmap` | `n/a` | `unsupported:v1_empty_vmap_output_has_no_materialized_batch_shape` | `expected typed error transform_execution.empty_vmap_output, got transform_execution.empty_vmap_output` |
| `vmap_out_axes_none_nonconstant_fail_closed` | `fail_closed` | `pass` | `vmap` | `n/a` | `unsupported:v1_out_axes_none_requires_identical_outputs` | `expected typed error transform_execution.vmap_unmapped_output_mismatch, got transform_execution.vmap_unmapped_output_mismatch` |

## Performance Sentinels

| Workload | Status | Iterations | p50 ns | p95 ns | p99 ns | Peak RSS bytes |
|---|---:|---:|---:|---:|---:|---:|
| `perf_vmap_scan_loop_stack` | `pass` | `16` | `108355` | `155184` | `155184` | `11706368` |
| `perf_vmap_while_loop_stack` | `pass` | `16` | `113064` | `134695` | `134695` | `11710464` |
| `perf_jit_vmap_grad_cond` | `pass` | `16` | `121079` | `179450` | `179450` | `11710464` |
| `perf_batched_switch` | `pass` | `16` | `88678` | `123564` | `123564` | `11714560` |

## Issues

No transform control-flow matrix issues found.
