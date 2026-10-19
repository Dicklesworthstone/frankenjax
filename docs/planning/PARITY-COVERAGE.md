# JAX Parity Coverage — FrankenJAX

Audit date: 2026-10-19

## Scope Declaration

FrankenJAX is a **reference implementation of JAX's transform semantics**, NOT a full JAX replacement. The V1 scope covers:

1. **Core transforms**: `jit`, `grad`, `vmap`, `value_and_grad`, `jacobian`, `jacfwd`, `jacrev`, `hessian`, `checkpoint`, `jvp`, `vjp`, `linearize`, `linear_transpose`, `eval_shape`, `make_jaxpr`, `custom_vjp`, `custom_jvp`
2. **Primitive operations**: LAX arithmetic, linalg, FFT, control flow, reductions, bitwise, shape manipulation, neural activations (`fj_api::nn`), PyTrees (`fj_api::tree_util`), and FFI/custom calls (`CustomCall`)
3. **Automatic differentiation**: Symbolic `grad_to_jaxpr` IR differentiation + analytical VJP & JVP rules for all 157 V1 local canonical primitives (+ `CustomCall`, zero finite-difference fallback)
4. **RNG**: ThreeFry2x32 with `key`/`split`/`fold_in` and 28 distributions (`fj_api::random`)

## Coverage Summary

| JAX Module | Upstream Count | FrankenJAX Count | Coverage | Notes |
|------------|----------------|------------------|----------|-------|
| **jax.* transforms** | 18 core | 17 | 94% | All 17 single-device transforms covered (`pmap`/`shard_map` excluded for V1 single-device scope; `disable_jit` gap) |
| **jax.lax primitives** | 157 V1 local (+ 5 pmap + 1 FFI) | 157 (+ 5 fail-closed + 1 FFI) | 100% | 162 canonical primitive variants (`Primitive::ALL` = 157 V1 local + 5 pmap collectives) + 1 FFI `CustomCall` = 163 total `Primitive` enum variants |
| **jax.lax control flow** | 5 | 5 | 100% | `cond`, `scan`, `while_loop`, `fori_loop`, `switch` (+ `associative_scan`) |
| **jax.lax linalg** | 10 | 10 | 100% | `cholesky`, `qr`, `svd`, `eigh`, `eig`, `triangular_solve`, `lu`, `solve`, `det`, `slogdet` |
| **jax.random** | 31 | 28 | 90% | Exposed in `fj-lax::threefry` and `fj_api::random` |
| **jax.numpy** | 300+ | 0 | 0% | OUT OF SCOPE for V1 (use `fj_api` / `fj_trace` / `fj_lax` directly) |
| **jax.nn** | 35 | 33 | 94% | Exposed in `fj-lax` and `fj_api::nn` |
| **jax.scipy** | 100+ | 0 | 0% | OUT OF SCOPE for V1 |
| **jax.tree_util** | 15 | 15 | 100% | Exposed in `fj-core` and `fj_api::tree_util` |
| **pmap/sharding** | 20+ | 5 IR variants (fail-closed) | 0% | OUT OF SCOPE for V1 execution (requires multi-device runtime) |

## Detailed Transform Coverage

### Core Transforms (`jax.*` / `fj_api`)

| Function | Status | Notes |
|----------|--------|-------|
| `jit` | COVERED | Partial evaluation + staging + default e-graph optimization + bounded LRU `CompiledProgram` cache via `fj-api` / `fj-dispatch` |
| `grad` | COVERED | Symbolic `grad_to_jaxpr` IR differentiation + analytical reverse-mode VJP |
| `value_and_grad` | COVERED | Shared primal + adjoint execution |
| `vmap` | COVERED | `BatchTrace` with per-primitive batching rules and `in_axes`/`out_axes` |
| `jacobian` | COVERED | Analytical JVP-based forward Jacobian |
| `jacfwd` | COVERED | Forward-mode JVP Jacobian |
| `jacrev` | COVERED | Reverse-mode VJP Jacobian |
| `hessian` | COVERED | Exact forward-over-reverse symbolic AD (`hessian_jaxpr`) |
| `checkpoint` / `remat` | COVERED | Rematerialization wrapper + `Primitive::Checkpoint` |
| `custom_vjp` | COVERED | Thread-safe registry integrated into `eval_jaxpr_vjp` and `grad_jaxpr` |
| `custom_jvp` | COVERED | Thread-safe registry integrated into `eval_jaxpr_jvp` |
| `make_jaxpr` | COVERED | Full 157-primitive (+ `CustomCall`) Rust closure tracing (`fj-trace`) + Python callable tracing (`fj-py`) |
| `jvp` | COVERED | Analytical forward-mode AD across all 157 V1 local primitives (+ `CustomCall`) |
| `vjp` | COVERED | Analytical reverse-mode AD across all 157 V1 local primitives (+ `CustomCall`) |
| `linearize` | COVERED | Primal + linear tangent function via `fj-api` |
| `linear_transpose` | COVERED | Transpose of linear function via `fj-api` |
| `eval_shape` | COVERED | Abstract shape/dtype inference without array materialization via `fj-api` |
| `pmap` | EXCLUDED | 5 collective IR variants (`Psum`, `Pmean`, `AllGather`, `AllToAll`, `AxisIndex`) fail closed without multi-device runtime |
| `shard_map` | EXCLUDED | Requires distributed sharding runtime |
| `disable_jit` | GAP | Not implemented |

### LAX Primitives Coverage

FrankenJAX defines **162 canonical primitive variants** in `Primitive::ALL` (**157 V1 local** primitives 100% implemented across evaluation, shape inference, tracing, batching, JVP, and VJP + **5 pmap collectives** that fail closed without multi-device context), plus **1 FFI `CustomCall` primitive** in `Primitive::FFI_PRIMITIVES` (**163 total `Primitive` enum variants**).

#### Arithmetic (23/23 = 100%)
Add, Sub, Mul, Div, Neg, Abs, Rem, Pow, Max, Min, Exp, Log, Sqrt, Rsqrt,
Floor, Ceil, Round, Expm1, Log1p, Sign, Square, Reciprocal, Logistic

#### Trigonometric (7/7 = 100%)
Sin, Cos, Tan, Asin, Acos, Atan, Atan2

#### Hyperbolic (6/6 = 100%)
Sinh, Cosh, Tanh, Asinh, Acosh, Atanh

#### Special Math (13/13 in V1 scope = 100%)
- COVERED: Erf, Erfc, Cbrt, Lgamma, Digamma, ErfInv, IsFinite, IntegerPow, Nextafter, BesselI0e, BesselI1e, RegularizedIncompleteBeta, Zeta
- OUT OF V1 SCOPE: unscaled `bessel_j0`, `bessel_j1`, `bessel_y0`, `bessel_y1`

#### Linear Algebra (10/10 = 100%)
- COVERED: Cholesky, Qr, Svd, Eigh, Eig, TriangularSolve, Lu, Solve, Det, Slogdet

#### FFT (4/4 = 100%)
Fft, Ifft, Rfft, Irfft

#### Reductions (8/8 = 100%)
ReduceSum, ReduceMax, ReduceMin, ReduceProd, ReduceAnd, ReduceOr, ReduceXor, ReduceWindow

#### Shape & Indexing (15/15 = 100%)
Reshape, Transpose, BroadcastInDim, Slice, DynamicSlice, DynamicUpdateSlice,
Gather, Scatter, Concatenate, Pad, Rev, Squeeze, Split, ExpandDims, Copy

#### Bitwise (10/10 = 100%)
BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot, ShiftLeft, ShiftRightArithmetic,
ShiftRightLogical, PopulationCount, CountLeadingZeros, CountTrailingZeros

#### Control Flow & Effect / Interop Primitives (100%)
- Control flow: Cond, Scan, While, Switch, AssociativeScan, Checkpoint
- Interop & effects: FfiCall, CustomCall, TieIn,Infeed, Outfeed, AfterAll, OrderedEffect

### `jax.random` Coverage (28/31 = 90%)

- COVERED (in `fj_lax::threefry` and `fj_api::random`): `key`, `split`, `fold_in`, `uniform`, `normal`, `bernoulli`, `categorical`, `exponential`, `gamma`, `beta`, `poisson`, `weibull`, `rayleigh`, `chisquare`, `student_t`, `laplace`, `cauchy`, `pareto`, `truncated_normal`, `dirichlet`, `geometric`, `gumbel`, `permutation`, `randint`, `logistic`, `maxwell`, `choice`, `shuffle`
- GAP: `generalized_normal`, `ball`, `orthogonal` / `multivariate_normal`

### `jax.nn` Coverage (33/35 = 94%)

- COVERED (in `fj_lax` and `fj_api::nn`): `relu`, `relu6`, `sigmoid`, `hard_sigmoid`, `hard_tanh`, `silu`/`swish`, `hard_silu`/`hard_swish`, `gelu`, `elu`, `celu`, `selu`, `leaky_relu`, `softplus`, `softsign`, `mish`, `log_sigmoid`, `logsumexp`, `logmeanexp`, `softmax`, `log_softmax`, `standardize`, `glu`, `squareplus`, `sparse_plus`, `sparse_sigmoid`, `log1mexp`, `identity`, `one_hot`
- GAP: `dot_product_attention`, `scaled_matmul`/`scaled_dot_general`

### `jax.tree_util` Coverage (15/15 = 100%)

Exposed in `fj_core` and `fj_api::tree_util`:
`tree_flatten`, `tree_unflatten`, `tree_leaves`, `tree_structure`, `tree_map`,
`tree_map2`, `tree_reduce`, `tree_all`, `tree_any`, `tree_leaf_count`,
`tree_zeros_like`, `tree_ones_like`, `tree_add`, `tree_sub`, `tree_mul`

## Explicit Out-of-Scope (V1)

These require infrastructure beyond V1 single-device transform semantics:

1. **jax.numpy** - NumPy-compatible array API (300+ functions)
2. **jax.scipy** - SciPy-compatible API (100+ functions)
3. **jax.pmap / jax.shard_map** - Multi-device parallelism (5 collective primitives fail closed)
4. **jax.sharding / NamedSharding** - Distributed array sharding
5. **jax.profiler** - Performance profiling
6. **jax.distributed** - Multi-process coordination
7. **jax.export** - Model export/serialization
8. **jax.dlpack** - DLPack interop
9. **XLA compilation** - Clean-room Rust interpreter/backend (`fj-interpreters` + `fj-backend-cpu`), not XLA

## Gap Summary for V1 Scope

| Gap | Priority | Bead |
|-----|----------|------|
| ~~linearize~~ | ~~P2~~ | ~~frankenjax-6duf~~ (DONE) |
| ~~linear_transpose~~ | ~~P2~~ | ~~frankenjax-awjz~~ (DONE) |
| ~~eval_shape~~ | ~~P3~~ | ~~frankenjax-jm5s~~ (DONE) |
| ~~associative_scan~~ | ~~P2~~ | ~~frankenjax-xszg~~ (DONE) |
| ~~solve (general linalg)~~ | ~~P2~~ | ~~frankenjax-qzwm~~ (DONE) |
| ~~det/slogdet~~ | ~~P3~~ | ~~frankenjax-b08e~~ (DONE) |
| ~~eig (non-symmetric)~~ | ~~P3~~ | ~~frankenjax-xx09~~ (DONE) |
| ~~symbolic grad_to_jaxpr + zero FD fallback~~ | ~~P0~~ | ~~frankenjax-rc26-n1emn.4–.7~~ (DONE) |
| ~~full 157-primitive (+ CustomCall) TracerRef + fj_api nn/random/tree_util~~ | ~~P1~~ | ~~frankenjax-rc26-n1emn.10–.11~~ (DONE) |
| unscaled bessel j0/j1/y0/y1 | P4 | (deferred beyond V1) |
| multivariate_normal / orthogonal RNG | P4 | (deferred beyond V1) |

## Overall V1 Parity

**In-scope V1 coverage: ~97%**

- **Transforms**: 94% (17/18 single-device transforms; only `disable_jit` omitted)
- **LAX primitives**: 100% of V1 local scope (157/157 V1 local canonical primitives + 5 fail-closed pmap collectives = 162 canonical primitive variants in `Primitive::ALL`, plus 1 FFI `CustomCall` = 163 total `Primitive` enum variants)
- **Control flow**: 100% (5/5 core control flow + `associative_scan`)
- **Linear algebra & FFT**: 100% (10/10 linalg + 4/4 FFT)
- **AD (VJP + JVP + symbolic `grad_to_jaxpr`)**: 100% (157/157 V1 local canonical primitives + `CustomCall` with analytical rules; zero finite-difference fallback)
- **`jax.random`**: 90% (28/31 distributions)
- **`jax.nn`**: 94% (33/35 functions)
- **`jax.tree_util`**: 100% (15/15 functions)

